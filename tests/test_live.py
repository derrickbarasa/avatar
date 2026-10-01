"""Microphone lip-sync analysis and the virtual webcam wrapper (no audio or camera hardware needed)."""
import os
import sys
import time
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from avatarkit import live  # noqa: E402

RATE = 16000


def vowel(f1, f2, seconds=0.02, amp=0.2, f0=120.0, start=0.0):
    """A buzzing voice (harmonics of f0) shaped by two formant peaks, like a spoken vowel."""
    t = start + np.arange(int(RATE * seconds)) / RATE
    out = np.zeros_like(t)
    for k in range(1, int(4000 / f0)):
        f = k * f0
        gain = np.exp(-((f - f1) / 150.0) ** 2) + 0.7 * np.exp(-((f - f2) / 250.0) ** 2) + 0.02
        out += gain * np.sin(2 * np.pi * f * t)
    return amp * out / max(1e-9, np.abs(out).max())


def speak(voice, f1, f2, seconds=0.6, amp=0.2):
    for i in range(int(seconds / 0.02)):
        voice.feed(vowel(f1, f2, amp=amp, start=i * 0.02), RATE)
    return voice.shape()


def make_voice():
    """A voice that has already listened to a second of room noise and a loud sentence."""
    voice = live.LiveVoice()
    rng = np.random.default_rng(0)
    for _ in range(50):
        voice.feed(rng.normal(0, 0.002, 320), RATE)
    return voice


class MouthFromSoundTests(unittest.TestCase):
    def test_silence_keeps_the_mouth_closed(self):
        voice = make_voice()
        rng = np.random.default_rng(1)
        for _ in range(50):
            voice.feed(rng.normal(0, 0.002, 320), RATE)
        self.assertLess(voice.shape().open, 0.05)

    def test_nothing_heard_lately_closes_the_mouth(self):
        voice = make_voice()
        speak(voice, 800, 1300)
        self.assertGreater(voice.shape().open, 0.3)
        voice._stamp -= 5
        self.assertEqual(voice.shape().open, 0.0)

    def test_open_vowel_opens_the_mouth_wide(self):
        mouth = speak(make_voice(), 800, 1300)                  # "ah"
        self.assertGreater(mouth.open, 0.6)

    def test_rounded_vowel_puckers(self):
        mouth = speak(make_voice(), 320, 800)                   # "oo"
        self.assertLess(mouth.wide, -0.3)

    def test_front_vowel_spreads_the_lips(self):
        mouth = speak(make_voice(), 300, 2400)                  # "ee"
        self.assertGreater(mouth.wide, 0.3)
        self.assertLess(mouth.open, speak(make_voice(), 800, 1300).open)

    def test_hissing_gives_a_thin_spread_mouth(self):
        voice = make_voice()
        rng = np.random.default_rng(2)
        for _ in range(40):
            hiss = np.diff(rng.normal(0, 0.2, 321))             # differenced noise: all treble
            voice.feed(hiss, RATE)
        mouth = voice.shape()
        self.assertGreater(mouth.wide, 0.3)
        self.assertLess(mouth.open, 0.3)

    def test_louder_speech_opens_further(self):
        quiet = speak(make_voice(), 650, 1200, amp=0.02)
        loud = speak(make_voice(), 650, 1200, amp=0.4)
        self.assertGreaterEqual(loud.open, quiet.open - 0.05)   # the level adapts to the input...
        self.assertGreater(loud.energy, 0.2)                    # ...but loud speech is still lively

    def test_mouth_closes_again_after_speech(self):
        voice = make_voice()
        speak(voice, 800, 1300)
        rng = np.random.default_rng(3)
        for _ in range(40):
            voice.feed(rng.normal(0, 0.002, 320), RATE)
        self.assertLess(voice.shape().open, 0.1)

    def test_values_stay_in_range(self):
        voice = make_voice()
        rng = np.random.default_rng(4)
        for _ in range(100):
            voice.feed(rng.normal(0, rng.uniform(0.001, 1.0), 320), RATE)
            m = voice.shape()
            self.assertTrue(0.0 <= m.open <= 1.0 and -1.0 <= m.wide <= 1.0 and 0.0 <= m.energy <= 1.0)

    def test_starting_without_the_package_explains_how_to_fix_it(self):
        saved = sys.modules.get("sounddevice", 0)
        sys.modules["sounddevice"] = None                       # makes the import fail
        try:
            with self.assertRaisesRegex(live.LiveError, "sounddevice"):
                live.LiveVoice().start()
            self.assertEqual(live.list_microphones(), [])
        finally:
            if saved == 0:
                del sys.modules["sounddevice"]
            else:
                sys.modules["sounddevice"] = saved


class FakeCamera:
    instances = []

    def __init__(self, width, height, fps, fmt=None, device=None):
        self.args, self.sent, self.closed, self.device = (width, height, fps, fmt, device), [], False, "Fake cam"
        FakeCamera.instances.append(self)

    def send(self, frame):
        self.sent.append(frame.copy())

    def close(self):
        self.closed = True


class VirtualCamTests(unittest.TestCase):
    def setUp(self):
        FakeCamera.instances.clear()
        self.saved = sys.modules.get("pyvirtualcam", 0)
        sys.modules["pyvirtualcam"] = types.SimpleNamespace(Camera=FakeCamera,
                                                           PixelFormat=types.SimpleNamespace(RGB="rgb"))
        self.addCleanup(self.restore)

    def restore(self):
        if self.saved == 0:
            sys.modules.pop("pyvirtualcam", None)
        else:
            sys.modules["pyvirtualcam"] = self.saved

    def test_sends_frames_at_the_requested_size_and_stops_cleanly(self):
        cam = live.VirtualCam((64, 36), fps=30)
        cam.start()
        self.assertTrue(cam.running)
        self.assertEqual(FakeCamera.instances[0].args[:3], (64, 36, 30))
        cam.send(np.zeros((36, 64, 3), np.uint8))
        self.assertEqual(len(FakeCamera.instances[0].sent), 1)
        cam.stop()
        self.assertTrue(FakeCamera.instances[0].closed)
        self.assertFalse(cam.running)
        cam.send(np.zeros((36, 64, 3), np.uint8))               # after stop: ignored
        self.assertEqual(len(FakeCamera.instances[0].sent), 1)

    def test_paces_frames_to_the_frame_rate(self):
        cam = live.VirtualCam((64, 36), fps=10)
        cam.start()
        now = time.time()
        self.assertTrue(cam.due(now))
        cam.send(np.zeros((36, 64, 3), np.uint8), now)
        self.assertFalse(cam.due(now + 0.02))
        self.assertTrue(cam.due(now + 0.11))

    def test_missing_driver_reports_a_helpful_error(self):
        def broken(*a, **k):
            raise RuntimeError("No camera backend found")
        sys.modules["pyvirtualcam"].Camera = broken
        with self.assertRaisesRegex(live.LiveError, "OBS"):
            live.VirtualCam((64, 36)).start()

    def test_missing_package_reports_a_helpful_error(self):
        sys.modules["pyvirtualcam"] = None
        with self.assertRaisesRegex(live.LiveError, "pyvirtualcam"):
            live.VirtualCam((64, 36)).start()


class FitFrameTests(unittest.TestCase):
    def test_wide_and_tall_pictures_are_letterboxed_not_stretched(self):
        import pygame
        pygame.init()
        red = pygame.Surface((100, 100))
        red.fill((255, 0, 0))
        frame = live.fit_frame(red, (160, 90), (0, 0, 255))
        self.assertEqual(frame.shape, (90, 160, 3))
        self.assertGreater(int(frame[45, 80][0]), 240)           # the picture is in the middle...
        self.assertLess(int(frame[45, 80][2]), 15)
        self.assertEqual(tuple(frame[45, 5]), (0, 0, 255))      # ...with the fill colour at the sides
        self.assertEqual(tuple(frame[45, 154]), (0, 0, 255))
        self.assertGreater(int(frame[2, 80][0]), 240)           # full height used


if __name__ == "__main__":
    unittest.main()
