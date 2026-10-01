"""Face tracking: the avatar follows your head, eyes, mouth and brows from a real webcam.

A face-landmark model (MediaPipe's face mesh, an optional package) finds about 470 points on your face
in every camera frame. `measure` turns those points into a handful of numbers (head turn, nod and tilt,
how closed each eye is, how open and wide the mouth is, brow height, where the eyes look). They are
measured against your own neutral face, learned over the first second, so a wide mouth or a heavy brow
does not offset the avatar. Everything except `FaceTracker` is plain numpy and is tested without a camera.

Directions are for a mirror view: turning your head toward your right turns the avatar toward the right
of the screen. (`mirror=False` flips that.)
"""
import math
import threading
import time
from dataclasses import dataclass

import numpy as np


class TrackingError(RuntimeError):
    """Face tracking could not start (missing package, no camera)."""


# MediaPipe face-mesh landmark numbers (468 face points, then 10 iris points with refine_landmarks)
FOREHEAD, CHIN = 10, 152
CHEEK_A, CHEEK_B = 234, 454                  # the sides of the face (A is the subject's right in a raw frame)
EYE_A_OUT, EYE_B_OUT = 33, 263               # outer eye corners
UPPER_LIP, LOWER_LIP = 13, 14
MOUTH_A, MOUTH_B = 61, 291                   # mouth corners
# per eye: outer corner, inner corner, upper lid, lower lid, two more lid points for the aspect ratio, iris, brow
EYE_A = dict(outer=33, inner=133, up=159, down=145, up2=158, down2=153, iris=468, brow=105)
EYE_B = dict(outer=263, inner=362, up=386, down=374, up2=385, down2=380, iris=473, brow=334)
N_FACE, N_REFINED = 468, 478


@dataclass
class FaceState:
    """What the face is doing, in the avatar's terms (all zero for a neutral face looking at the camera)."""
    present: bool = False
    yaw: float = 0.0            # head turn in degrees, + toward the right of the screen
    pitch: float = 0.0          # nod in degrees, + looking down
    roll: float = 0.0           # tilt in degrees as the avatar's own z rotation: + tilts the top to the left of the screen
    blink: float = 0.0          # 0 open .. 1 closed (both eyes averaged)
    mouth_open: float = 0.0     # 0 .. 1
    mouth_wide: float = 0.0     # -1 puckered .. +1 stretched wide
    smile: float = 0.0          # -1 frown .. +1 broad smile
    brows: float = 0.0          # -1 lowered .. +1 raised
    gaze: tuple = (0.0, 0.0)    # eyes: (pitch, yaw) in degrees, + down / + to the right of the screen


def _px(lm, size, mirror):
    """Landmarks as pixel coordinates (x to the right, y down, z toward the screen), mirrored if asked."""
    w, h = size
    p = np.asarray(lm, float)[:, :3].copy()
    if mirror:
        p[:, 0] = 1.0 - p[:, 0]
    p[:, 0] *= w
    p[:, 1] *= h
    p[:, 2] *= w                                  # MediaPipe's z is on the same scale as x
    return p


def measure(lm, size, mirror=True):
    """Raw measurements from landmarks (N, 3) in 0..1 image units, N >= 468. Needs `calibrate` to become a state."""
    p = _px(lm, size, mirror)
    face_w = abs(p[CHEEK_A, 0] - p[CHEEK_B, 0]) + 1e-6
    face_h = abs(p[CHIN, 1] - p[FOREHEAD, 1]) + 1e-6
    # head angles from the depth of the face's corners; "right" is the right-hand side of the image
    right, left = (CHEEK_A, CHEEK_B) if p[CHEEK_A, 0] > p[CHEEK_B, 0] else (CHEEK_B, CHEEK_A)
    yaw = math.degrees(math.atan2(p[right, 2] - p[left, 2], p[right, 0] - p[left, 0]))
    pitch = math.degrees(math.atan2(p[CHIN, 2] - p[FOREHEAD, 2], p[CHIN, 1] - p[FOREHEAD, 1]))
    r_eye, l_eye = (EYE_A_OUT, EYE_B_OUT) if p[EYE_A_OUT, 0] > p[EYE_B_OUT, 0] else (EYE_B_OUT, EYE_A_OUT)
    roll = -math.degrees(math.atan2(p[r_eye, 1] - p[l_eye, 1], p[r_eye, 0] - p[l_eye, 0]))

    ear, brow, gaze = [], [], []
    for e in (EYE_A, EYE_B):
        width = np.linalg.norm(p[e["outer"], :2] - p[e["inner"], :2]) + 1e-6
        opening = (np.linalg.norm(p[e["up"], :2] - p[e["down"], :2])
                   + np.linalg.norm(p[e["up2"], :2] - p[e["down2"], :2])) / (2 * width)
        ear.append(opening)
        brow.append(abs(p[e["up"], 1] - p[e["brow"], 1]) / face_h)
        if len(p) >= N_REFINED:                               # the iris points need refine_landmarks
            a, b = sorted((p[e["outer"]], p[e["inner"]]), key=lambda q: q[0])       # image-left, image-right corner
            gx = (p[e["iris"], 0] - (a[0] + b[0]) / 2) / (width / 2)
            lid = (p[e["up"], 1] + p[e["down"], 1]) / 2
            gy = (p[e["iris"], 1] - lid) / (width * 0.35)
            gaze.append((gy, gx))
    mouth_h = abs(p[LOWER_LIP, 1] - p[UPPER_LIP, 1]) / face_h
    mouth_w = abs(p[MOUTH_A, 0] - p[MOUTH_B, 0]) / face_w
    lips_mid = (p[UPPER_LIP, 1] + p[LOWER_LIP, 1]) / 2
    corners = (p[MOUTH_A, 1] + p[MOUTH_B, 1]) / 2
    return {"yaw": yaw, "pitch": pitch, "roll": roll,
            "ear": float(np.mean(ear)), "brow": float(np.mean(brow)),
            "mouth_h": mouth_h, "mouth_w": mouth_w, "corner_lift": (lips_mid - corners) / face_h,
            "gaze_y": float(np.mean([g[0] for g in gaze])) if gaze else 0.0,
            "gaze_x": float(np.mean([g[1] for g in gaze])) if gaze else 0.0}


class Calibration:
    """Learns your neutral face from the first `frames` measurements, then turns measurements into a FaceState."""

    def __init__(self, frames=20):
        self.frames, self.samples, self.base = frames, [], None

    @property
    def done(self):
        return self.base is not None

    def add(self, m):
        self.samples.append(m)
        if len(self.samples) >= self.frames:
            self.base = {k: float(np.median([s[k] for s in self.samples])) for k in m}

    def state(self, m):
        b = self.base
        clamp = lambda v, lo, hi: float(min(hi, max(lo, v)))
        closed = clamp((b["ear"] - m["ear"]) / max(b["ear"] * 0.55, 1e-6), 0.0, 1.0)
        return FaceState(
            present=True,
            yaw=clamp(m["yaw"] - b["yaw"], -70, 70), pitch=clamp(m["pitch"] - b["pitch"], -45, 45),
            roll=clamp(m["roll"] - b["roll"], -45, 45),
            blink=closed,
            mouth_open=clamp((m["mouth_h"] - b["mouth_h"]) / 0.10, 0.0, 1.0),
            mouth_wide=clamp((m["mouth_w"] - b["mouth_w"]) / 0.10, -1.0, 1.0),
            smile=clamp((m["corner_lift"] - b["corner_lift"]) / 0.035, -1.0, 1.3),
            brows=clamp((m["brow"] - b["brow"]) / 0.025, -1.0, 1.0),
            gaze=(clamp((m["gaze_y"] - b["gaze_y"]) * 25.0, -22, 22), clamp((m["gaze_x"] - b["gaze_x"]) * 30.0, -30, 30)))


class Smoother:
    """Exponential smoothing with a time constant per field, so the avatar is calm but still quick to blink."""
    TAU = {"yaw": 0.07, "pitch": 0.07, "roll": 0.08, "blink": 0.025, "mouth_open": 0.04, "mouth_wide": 0.06,
           "smile": 0.10, "brows": 0.10, "gaze": 0.05}

    def __init__(self):
        self.state, self.t = None, None

    def update(self, new, now):
        if self.state is None or not new.present or not self.state.present:
            self.state, self.t = new, now
            return new
        dt = max(1e-3, now - self.t)
        self.t = now
        out = {}
        for f, tau in self.TAU.items():
            a = 1.0 - math.exp(-dt / tau)
            old, cur = getattr(self.state, f), getattr(new, f)
            out[f] = tuple(o + (c - o) * a for o, c in zip(old, cur)) if f == "gaze" else old + (cur - old) * a
        self.state = FaceState(present=True, **out)
        return self.state


def apply_to_pose(pose, state, brow_nodes=True):
    """Move the avatar's head, neck and brows to follow `state` (changes `pose`, a node -> degrees dict)."""
    rx, ry, rz = state.pitch, state.yaw, state.roll
    pose["head"] = (rx * 0.75, ry * 0.75, rz * 0.8)                    # the head takes most of it ...
    pose["torso"] = tuple(a + b * 0.25 for a, b in zip(pose.get("torso", (0.0, 0.0, 0.0)), (rx * 0.3, ry * 0.4, rz * 0.2)))
    if brow_nodes:
        b = pose.get("brows", (0.0, 0.0, 0.0))
        pose["brows"] = (b[0] - 6.0 * state.brows, b[1], b[2])
    return pose


# ---------------------------------------------------------------------------
# Camera and landmark model
# ---------------------------------------------------------------------------
class FaceTracker:
    """Reads a webcam on a background thread and keeps the latest FaceState ready for the animation loop."""

    def __init__(self, camera=0, mirror=True):
        self.camera, self.mirror = camera, mirror
        self._lock = threading.Lock()
        self._state = FaceState()
        self._stop = threading.Event()
        self._thread = None
        self.calibration = Calibration()
        self.smoother = Smoother()
        self.fps = 0.0
        self.error = ""

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def calibrated(self):
        return self.calibration.done

    def start(self):
        if self.running:
            return
        try:
            import cv2
        except ImportError as exc:
            raise TrackingError("face tracking needs OpenCV (pip install opencv-python)") from exc
        try:
            import mediapipe as mp
            mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except ImportError as exc:
            raise TrackingError("face tracking needs MediaPipe (pip install mediapipe)") from exc
        except AttributeError as exc:
            raise TrackingError("this MediaPipe has no face-mesh solution (try pip install \"mediapipe<0.10.20\")") from exc
        api = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") and _is_windows() else 0
        cap = cv2.VideoCapture(self.camera, api)
        if not cap.isOpened():
            cap.release()
            mesh.close()
            raise TrackingError(f"could not open camera {self.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.calibration, self.smoother, self.error = Calibration(), Smoother(), ""
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(cv2, cap, mesh), daemon=True)
        self._thread.start()

    def _loop(self, cv2, cap, mesh):
        last = time.time()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                h, w = frame.shape[:2]
                found = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).multi_face_landmarks
                now = time.time()
                if found:
                    lm = np.array([[q.x, q.y, q.z] for q in found[0].landmark])
                    m = measure(lm, (w, h), self.mirror)
                    if not self.calibration.done:
                        self.calibration.add(m)
                        new = FaceState(present=True)
                    else:
                        new = self.calibration.state(m)
                else:
                    new = FaceState()
                with self._lock:
                    self._state = self.smoother.update(new, now)
                self.fps += (1.0 / max(now - last, 1e-3) - self.fps) * 0.1
                last = now
        except Exception as exc:                         # the camera was unplugged, the model failed...
            self.error = str(exc)
        finally:
            cap.release()
            mesh.close()

    def state(self):
        with self._lock:
            return self._state

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._state = FaceState()


def _is_windows():
    import sys
    return sys.platform.startswith("win")


def list_cameras(limit=4):
    """[index] of cameras that open (or [] without OpenCV). Opening each one takes a moment."""
    try:
        import cv2
    except ImportError:
        return []
    found = []
    for i in range(limit):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if _is_windows() and hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found
