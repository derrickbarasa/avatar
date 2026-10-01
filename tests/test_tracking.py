"""Face tracking maths and the tracker thread, with a synthetic face (no camera or MediaPipe needed)."""
import math
import os
import sys
import time
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from avatarkit import tracking as T  # noqa: E402

SIZE = (640, 480)
PX = 200.0                       # pixels per face unit (the face is 2 units wide and 2 units tall)


def face(yaw=0.0, pitch=0.0, roll=0.0, eyes=1.0, mouth=0.0, wide=0.0, smile=0.0, brows=0.0, gaze=(0.0, 0.0)):
    """A 478-point face in 0..1 image units. x right, y down, z away from the camera (as MediaPipe does).
    Angles in degrees: yaw + turns the nose toward the right of the image, pitch + looks down, roll + tilts clockwise."""
    pts = np.zeros((T.N_REFINED, 3))

    def put(i, x, y, z=0.0):
        pts[i] = (x, y, z)

    put(T.FOREHEAD, 0, -1.0)
    put(T.CHIN, 0, 1.0)
    put(T.CHEEK_A, -1.0, 0.0)
    put(T.CHEEK_B, 1.0, 0.0)
    put(1, 0.0, 0.25, -0.5)                                     # the nose tip, toward the camera
    for e, sx in ((T.EYE_A, -1), (T.EYE_B, 1)):                 # eye A on the image's left
        cx = sx * 0.30
        put(e["outer"], sx * 0.45, -0.2)
        put(e["inner"], sx * 0.15, -0.2)
        put(e["up"], cx, -0.2 - 0.06 * eyes)
        put(e["down"], cx, -0.2 + 0.06 * eyes)
        put(e["up2"], cx + 0.03, -0.2 - 0.06 * eyes)
        put(e["down2"], cx + 0.03, -0.2 + 0.06 * eyes)
        put(e["iris"], cx + gaze[1] * 0.15, -0.2 + gaze[0] * 0.105)
        put(e["brow"], cx, -0.26 - 0.06 * eyes - 0.15 - 0.05 * brows + 0.06 * (1 - eyes) * 0)
    put(T.UPPER_LIP, 0, 0.55)
    put(T.LOWER_LIP, 0, 0.59 + 0.2 * mouth)
    corner_y = 0.57 + 0.1 * mouth - 0.07 * smile
    put(T.MOUTH_A, -0.4 * (1 + 0.25 * wide), corner_y)
    put(T.MOUTH_B, 0.4 * (1 + 0.25 * wide), corner_y)

    ya, pa, ra = (math.radians(a) for a in (yaw, pitch, roll))
    ry = np.array([[math.cos(ya), 0, -math.sin(ya)], [0, 1, 0], [math.sin(ya), 0, math.cos(ya)]])
    rx = np.array([[1, 0, 0], [0, math.cos(pa), -math.sin(pa)], [0, math.sin(pa), math.cos(pa)]])
    rz = np.array([[math.cos(ra), -math.sin(ra), 0], [math.sin(ra), math.cos(ra), 0], [0, 0, 1]])
    pts = pts @ (rz @ rx @ ry).T
    out = pts.copy()
    out[:, 0] = (SIZE[0] / 2 + pts[:, 0] * PX) / SIZE[0]
    out[:, 1] = (SIZE[1] / 2 + pts[:, 1] * PX) / SIZE[1]
    out[:, 2] = pts[:, 2] * PX / SIZE[0]
    return out


def calibrated(**base):
    cal = T.Calibration(frames=5)
    for _ in range(5):
        cal.add(T.measure(face(**base), SIZE, mirror=False))
    return cal


def read(cal, **kw):
    return cal.state(T.measure(face(**kw), SIZE, mirror=False))


class MeasureTests(unittest.TestCase):
    def test_the_neutral_face_reads_as_zero(self):
        s = read(calibrated())
        for name in ("yaw", "pitch", "roll", "blink", "mouth_open", "mouth_wide", "smile", "brows"):
            self.assertAlmostEqual(getattr(s, name), 0.0, delta=0.02, msg=name)
        self.assertAlmostEqual(s.gaze[0], 0.0, delta=0.5)
        self.assertTrue(s.present)

    def test_head_angles_are_recovered_with_the_right_signs(self):
        cal = calibrated()
        for angle in (-30, -10, 15, 35):
            self.assertAlmostEqual(read(cal, yaw=angle).yaw, angle, delta=1.0)
            self.assertAlmostEqual(read(cal, pitch=angle).pitch, angle, delta=1.0)
            self.assertAlmostEqual(read(cal, roll=angle).roll, -angle, delta=1.0)     # + clockwise in, the avatar's z out

    def test_a_head_that_started_off_centre_is_measured_from_where_it_was(self):
        cal = calibrated(yaw=12, pitch=-8)
        self.assertAlmostEqual(read(cal, yaw=12, pitch=-8).yaw, 0.0, delta=0.5)
        self.assertAlmostEqual(read(cal, yaw=32, pitch=-8).yaw, 20.0, delta=1.0)

    def test_blinking(self):
        cal = calibrated()
        self.assertGreater(read(cal, eyes=0.25).blink, 0.95)
        self.assertAlmostEqual(read(cal, eyes=0.7).blink, 0.55, delta=0.15)
        self.assertEqual(read(cal, eyes=1.2).blink, 0.0)                    # eyes wide open is not a blink

    def test_mouth_open_wide_and_smile(self):
        cal = calibrated()
        self.assertAlmostEqual(read(cal, mouth=1.0).mouth_open, 1.0, delta=0.1)
        self.assertAlmostEqual(read(cal, mouth=0.4).mouth_open, 0.4, delta=0.1)
        self.assertGreater(read(cal, wide=1.0).mouth_wide, 0.8)
        self.assertLess(read(cal, wide=-1.0).mouth_wide, -0.8)
        self.assertGreater(read(cal, smile=1.0).smile, 0.8)
        self.assertLess(read(cal, smile=-1.0).smile, -0.8)

    def test_brows(self):
        cal = calibrated()
        self.assertGreater(read(cal, brows=1.0).brows, 0.8)
        self.assertLess(read(cal, brows=-1.0).brows, -0.8)

    def test_gaze(self):
        cal = calibrated()
        pitch, yaw = read(cal, gaze=(0.0, 0.8)).gaze
        self.assertAlmostEqual(yaw, 24.0, delta=3.0)                         # looking to the right of the image
        self.assertAlmostEqual(read(cal, gaze=(-0.0, -0.8)).gaze[1], -24.0, delta=3.0)
        self.assertGreater(read(cal, gaze=(0.6, 0.0)).gaze[0], 8.0)           # looking down

    def test_mirror_swaps_left_and_right(self):
        flipped = face(yaw=20, gaze=(0.0, 0.5))
        flipped[:, 0] = 1.0 - flipped[:, 0]
        a = T.measure(face(yaw=20, gaze=(0.0, 0.5)), SIZE, mirror=False)
        b = T.measure(flipped, SIZE, mirror=True)                            # the same face, filmed flipped, then mirrored
        for key in a:
            self.assertAlmostEqual(a[key], b[key], delta=1e-6, msg=key)
        raw = T.measure(face(yaw=20), SIZE, mirror=True)                     # mirroring a normal frame turns the other way
        self.assertAlmostEqual(raw["yaw"], -20, delta=0.5)

    def test_calibration_waits_for_enough_frames(self):
        cal = T.Calibration(frames=10)
        for _ in range(9):
            cal.add(T.measure(face(), SIZE, mirror=False))
        self.assertFalse(cal.done)
        cal.add(T.measure(face(), SIZE, mirror=False))
        self.assertTrue(cal.done)

    def test_values_stay_in_range_for_extreme_faces(self):
        cal = calibrated()
        s = read(cal, yaw=80, pitch=80, roll=80, eyes=0.0, mouth=3, wide=3, smile=3, brows=3, gaze=(3, 3))
        self.assertTrue(-70 <= s.yaw <= 70 and -45 <= s.pitch <= 45 and -45 <= s.roll <= 45)
        self.assertTrue(0 <= s.blink <= 1 and 0 <= s.mouth_open <= 1 and -1 <= s.mouth_wide <= 1)
        self.assertTrue(-1 <= s.smile <= 1.3 and -1 <= s.brows <= 1)

    def test_without_iris_points_the_gaze_is_straight_ahead(self):
        m = T.measure(face(gaze=(0.0, 0.8))[:468], SIZE, mirror=False)
        self.assertEqual((m["gaze_x"], m["gaze_y"]), (0.0, 0.0))


class SmootherAndPoseTests(unittest.TestCase):
    def test_smoothing_converges_and_blinks_are_quicker_than_head_turns(self):
        sm = T.Smoother()
        sm.update(T.FaceState(present=True), 0.0)
        target = T.FaceState(present=True, yaw=30.0, blink=1.0)
        out = [sm.update(target, 0.03 * i) for i in range(1, 60)]
        self.assertTrue(all(b.yaw >= a.yaw for a, b in zip(out, out[1:])))      # monotone, no overshoot
        self.assertAlmostEqual(out[-1].yaw, 30.0, delta=0.5)
        self.assertGreater(out[0].blink / 1.0, out[0].yaw / 30.0)                # the eye closes faster than the head turns

    def test_losing_the_face_resets_instead_of_gliding_back(self):
        sm = T.Smoother()
        sm.update(T.FaceState(present=True, yaw=20.0), 0.0)
        self.assertFalse(sm.update(T.FaceState(), 0.05).present)

    def test_pose_follows_the_head_and_brows(self):
        pose = {"head": (3.0, 3.0, 3.0), "torso": (0.0, 0.0, 0.0)}
        T.apply_to_pose(pose, T.FaceState(present=True, yaw=20.0, pitch=10.0, roll=-8.0, brows=1.0))
        rx, ry, rz = pose["head"]
        self.assertGreater(ry, 10.0)
        self.assertGreater(rx, 5.0)
        self.assertLess(rz, -4.0)
        self.assertLess(pose["brows"][0], -3.0)                                # raised brows
        self.assertGreater(pose["torso"][1], 0.0)                              # the shoulders follow a little


class FakeCapture:
    def __init__(self, index, api=0):
        self.index, self.opened = index, True

    def isOpened(self):
        return self.opened

    def set(self, *a):
        pass

    def read(self):
        time.sleep(0.005)
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.opened = False


class FakeMesh:
    closed = False
    yaw = 0.0

    def __init__(self, **kw):
        pass

    def process(self, rgb):
        pts = face(yaw=FakeMesh.yaw)
        landmarks = [types.SimpleNamespace(x=x, y=y, z=z) for x, y, z in pts]
        return types.SimpleNamespace(multi_face_landmarks=[types.SimpleNamespace(landmark=landmarks)])

    def close(self):
        FakeMesh.closed = True


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: sys.modules.get(k, 0) for k in ("cv2", "mediapipe")}
        sys.modules["cv2"] = types.SimpleNamespace(
            CAP_DSHOW=700, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, COLOR_BGR2RGB=4,
            VideoCapture=FakeCapture, cvtColor=lambda frame, code: frame)
        sys.modules["mediapipe"] = types.SimpleNamespace(
            solutions=types.SimpleNamespace(face_mesh=types.SimpleNamespace(FaceMesh=FakeMesh)))
        FakeMesh.closed, FakeMesh.yaw = False, 0.0
        self.addCleanup(self.restore)

    def restore(self):
        for k, v in self.saved.items():
            if v == 0:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def wait(self, test, seconds=5.0):
        end = time.time() + seconds
        while time.time() < end:
            if test():
                return True
            time.sleep(0.02)
        return False

    def test_tracker_calibrates_then_follows_the_face_and_stops_cleanly(self):
        tracker = T.FaceTracker(mirror=False)
        tracker.start()
        self.assertTrue(tracker.running)
        self.assertTrue(self.wait(lambda: tracker.calibrated))
        FakeMesh.yaw = 25.0
        self.assertTrue(self.wait(lambda: abs(tracker.state().yaw - 25.0) < 2.0))
        self.assertTrue(tracker.state().present)
        tracker.stop()
        self.assertFalse(tracker.running)
        self.assertTrue(FakeMesh.closed)
        self.assertFalse(tracker.state().present)

    def test_missing_packages_explain_what_to_install(self):
        sys.modules["mediapipe"] = None
        with self.assertRaisesRegex(T.TrackingError, "mediapipe"):
            T.FaceTracker().start()
        sys.modules["cv2"] = None
        with self.assertRaisesRegex(T.TrackingError, "OpenCV"):
            T.FaceTracker().start()

    def test_a_camera_that_will_not_open_is_reported(self):
        class Dead(FakeCapture):
            def isOpened(self):
                return False
        sys.modules["cv2"].VideoCapture = Dead
        with self.assertRaisesRegex(T.TrackingError, "camera"):
            T.FaceTracker(camera=3).start()


if __name__ == "__main__":
    unittest.main()
