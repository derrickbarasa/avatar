"""Live input and output: lip-sync from a microphone and a virtual webcam.

Both use optional packages (`sounddevice`, `pyvirtualcam`) that are imported only when switched on, so
the rest of the app never needs them. The sound analysis itself (`LiveVoice.feed`) is plain numpy and
runs without any audio hardware.
"""
import threading
import time

import numpy as np

from .speech import MouthShape, smoothstep


class LiveError(RuntimeError):
    """A live feature could not start (missing package, no device, camera driver not installed)."""


# ---------------------------------------------------------------------------
# Microphone -> mouth shape
# ---------------------------------------------------------------------------
class LiveVoice:
    """Turns a stream of microphone samples into the avatar's mouth, a few dozen times a second.

    There is no text to follow, so the shape comes from the sound itself: loudness sets how far the
    mouth opens, the first formant (how open the vowel is) separates "ah" from "oo" / "ee", the second
    formant decides between pucker and spread, and a hissy spectrum gives the thin spread of "s" / "f".
    The level adapts to the microphone and to the room, so it works with quiet and loud inputs alike.
    """
    ATTACK, RELEASE = 0.025, 0.075          # seconds: how fast the mouth opens and closes
    STALE = 0.5                             # no audio for this long -> mouth closed

    def __init__(self):
        self._lock = threading.Lock()
        self._shape = MouthShape()
        self._stamp = 0.0
        self.floor, self.peak = 1e-4, 0.03   # noise floor and loud reference (RMS); both adapt
        self._env = 0.0
        self._open = self._wide = 0.0
        self._prev_env = 0.0
        self.stream = None
        self.device_name = ""

    # ---- analysis ---------------------------------------------------------------------------
    def feed(self, samples, rate):
        """Analyse one chunk of mono samples in [-1, 1] and update the mouth."""
        x = np.asarray(samples, np.float64).ravel()
        if len(x) < 64:
            return
        dt = len(x) / rate
        x = x - x.mean()
        rms = float(np.sqrt(np.mean(x * x)))
        # adapt: the floor creeps up slowly and drops fast, the loud reference does the opposite
        self.floor += (rms - self.floor) * (0.02 * dt if rms > self.floor else min(1.0, 8 * dt))
        self.peak = max(0.02, rms if rms > self.peak else self.peak * (0.2 ** (dt / 6.0)))
        span = max(self.peak - self.floor * 1.5, 0.01)
        level = float(np.clip((rms - self.floor * 1.5) / span, 0.0, 1.2))
        gate = float(smoothstep(0.08, 0.30, level))

        open_t, wide_t = 0.0, 0.0
        if gate > 0.0:
            spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=max(2048, len(x)))) ** 2
            freq = np.fft.rfftfreq(max(2048, len(x)), 1.0 / rate)

            def centroid(lo, hi):
                m = (freq >= lo) & (freq < hi)
                e = spec[m].sum()
                return float((freq[m] * spec[m]).sum() / e) if e > 0 else (lo + hi) / 2

            total = spec[(freq >= 100) & (freq < 8000)].sum() + 1e-12
            hiss = float(spec[(freq >= 4000) & (freq < 8000)].sum() / total)
            f1, f2 = centroid(200, 1000), centroid(900, 3200)
            vowel_open = float(np.clip((f1 - 380.0) / 320.0, 0.0, 1.0))          # "ee" .. "ah"
            wide_v = float(np.clip((f2 - 1500.0) / 700.0, -1.0, 1.0))             # pucker .. spread
            open_t = (0.30 + 0.70 * vowel_open) * min(1.0, 0.45 + level)
            wide_t = (0.80 * wide_v) * (1.0 - 0.6 * vowel_open)                   # "ah" stays neutral
            fric = float(smoothstep(0.35, 0.65, hiss))                            # s, f, sh: thin and spread
            open_t, wide_t = open_t * (1 - fric) + 0.14 * fric, wide_t * (1 - fric) + 0.65 * fric
            open_t, wide_t = open_t * gate, wide_t * gate

        up = lambda cur, tgt: cur + (tgt - cur) * (1 - np.exp(-dt / (self.ATTACK if tgt > cur else self.RELEASE)))
        self._open = float(up(self._open, open_t))
        self._wide = float(up(self._wide, wide_t))
        self._env = float(up(self._env, level))
        beat = max(0.0, level - self._prev_env)
        self._prev_env = level
        with self._lock:
            self._shape = MouthShape(open=min(1.0, self._open), wide=self._wide, press=0.0,
                                     energy=min(1.0, self._env), beat=min(1.0, beat * 3.0))
            self._stamp = time.time()

    def shape(self):
        """The current mouth (closed when nothing has been heard lately)."""
        with self._lock:
            if time.time() - self._stamp > self.STALE:
                return MouthShape()
            return self._shape

    # ---- hardware -----------------------------------------------------------------------------
    @property
    def running(self):
        return self.stream is not None

    def start(self, device=None):
        """Open the microphone (`device` is a name fragment or index; the default input otherwise)."""
        if self.stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise LiveError("the microphone needs sounddevice (pip install sounddevice)") from exc
        try:
            info = sd.query_devices(device, "input")
            rate = int(info["default_samplerate"])
            self.stream = sd.InputStream(device=device, channels=1, samplerate=rate, blocksize=int(rate * 0.02),
                                         dtype="float32", callback=lambda data, frames, t, status:
                                         self.feed(data[:, 0], rate))
            self.stream.start()
            self.device_name = str(info["name"])
        except Exception as exc:                            # PortAudioError, ValueError, no input device...
            self.stream = None
            raise LiveError(f"no microphone ({exc})") from exc

    def stop(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        with self._lock:
            self._shape = MouthShape()


def list_microphones():
    """[(index, name)] of input devices, or [] if sounddevice is not installed."""
    try:
        import sounddevice as sd
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Virtual webcam
# ---------------------------------------------------------------------------
class VirtualCam:
    """Sends rendered frames to a virtual camera (OBS Virtual Camera on Windows and macOS, v4l2loopback
    on Linux) so Zoom, Teams, Meet and OBS can use the avatar as a webcam."""

    def __init__(self, size=(1280, 720), fps=30, device=None):
        self.size, self.fps, self.device = size, fps, device
        self.cam = None
        self.name = ""
        self._next = 0.0
        self.measured_fps = 0.0        # frames per second actually delivered (smoothed)
        self._last = None

    @property
    def running(self):
        return self.cam is not None

    def start(self):
        if self.cam is not None:
            return
        try:
            import pyvirtualcam
        except ImportError as exc:
            raise LiveError("the virtual webcam needs pyvirtualcam (pip install pyvirtualcam)") from exc
        try:
            self.cam = pyvirtualcam.Camera(self.size[0], self.size[1], self.fps,
                                           fmt=pyvirtualcam.PixelFormat.RGB, device=self.device)
        except Exception as exc:
            hint = " Install OBS Studio and start its Virtual Camera once." if "obs" in str(exc).lower() or \
                "camera" in str(exc).lower() else ""
            raise LiveError(f"no virtual camera ({exc}).{hint}") from exc
        self.name = getattr(self.cam, "device", "") or "virtual camera"
        self._next, self._last, self.measured_fps = 0.0, None, 0.0

    def due(self, now=None):
        """True when it is time for the next frame (the app calls this every frame and only renders
        a capture when it is)."""
        return self.cam is not None and (time.time() if now is None else now) >= self._next

    def send(self, frame, now=None):
        """Send an (height, width, 3) uint8 RGB array of exactly `size`."""
        if self.cam is None:
            return
        now = time.time() if now is None else now
        step = 1.0 / self.fps                    # on time: keep the rhythm; late: restart it, never burst
        self._next = self._next + step if self._next + step > now else now + step
        self.cam.send(frame)
        if self._last is not None and now > self._last:
            rate = 1.0 / (now - self._last)
            self.measured_fps = rate if not self.measured_fps else self.measured_fps + (rate - self.measured_fps) * 0.15
        self._last = now

    def stop(self):
        cam, self.cam = self.cam, None
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass


def fit_frame(surface, size, fill):
    """The picture shrunk to fit inside `size` (never stretched, never cropped), centred on a flat `fill`
    colour, as an (height, width, 3) uint8 array."""
    import pygame
    w, h = size
    sw, sh = surface.get_size()
    k = min(w / sw, h / sh)
    fit = (max(2, min(w, round(sw * k))), max(2, min(h, round(sh * k))))
    if fit != (sw, sh):
        surface = pygame.transform.smoothscale(surface, fit)
    canvas = pygame.Surface(size)
    canvas.fill(tuple(int(c) for c in fill))
    canvas.blit(surface, ((w - fit[0]) // 2, (h - fit[1]) // 2))
    return np.frombuffer(pygame.image.tostring(canvas, "RGB"), np.uint8).reshape(h, w, 3)
