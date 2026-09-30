"""Text-to-speech through whatever the operating system already has.

Windows: System.Speech (SAPI) via PowerShell; macOS: `say`; Linux: espeak-ng or
espeak. Nothing to install. Synthesis runs on a background thread and writes a
WAV/AIFF file that pygame can play and the lip-sync can analyse.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading

CACHE = os.path.join(tempfile.gettempdir(), "avatarkit_tts")
RATES = {"Slow": -3, "Normal": 0, "Fast": 3}      # SAPI steps; scaled for the other engines


class TTSError(RuntimeError):
    pass


def backend():
    """Name of the usable speech engine, or None."""
    if sys.platform.startswith("win") and shutil.which("powershell"):
        return "powershell"
    if sys.platform == "darwin" and shutil.which("say"):
        return "say"
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            return exe
    return None


def _run(cmd, env=None, timeout=60):
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout,
                         creationflags=flags)
    if res.returncode != 0:
        raise TTSError((res.stderr or res.stdout or "speech engine failed").strip()[:200])
    return res.stdout


def list_voices_info():
    """Installed voices as [(name, two-letter language)] (best effort; [] if not listable)."""
    b = backend()
    try:
        if b == "powershell":
            script = ("Add-Type -AssemblyName System.Speech; "
                      "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
                      "ForEach-Object { if ($_.Enabled) { $_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture.Name } }")
            out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], timeout=30)
            rows = [line.strip().split("|") for line in out.splitlines() if line.strip()]
            return [(r[0], (r[1] if len(r) > 1 else "en")[:2].lower()) for r in rows]
        if b == "say":
            rows = [line.split() for line in _run(["say", "-v", "?"]).splitlines() if line.strip()]
            return [(r[0], r[1][:2].lower() if len(r) > 1 else "en") for r in rows]
        if b in ("espeak-ng", "espeak"):
            rows = [line.split() for line in _run([b, "--voices"]).splitlines()[1:] if len(line.split()) > 3]
            return [(r[3], r[1].split("-")[0][:2].lower()) for r in rows]
    except (TTSError, OSError, subprocess.TimeoutExpired):
        pass
    return []


def list_voices():
    """Installed voice names."""
    return [name for name, _ in list_voices_info()]


def synthesize(text, voice="", rate="Normal"):
    """Speak `text` into an audio file (cached) and return its path."""
    b = backend()
    if b is None:
        raise TTSError("no speech engine found (Windows SAPI, macOS say or espeak)")
    steps = RATES.get(rate, 0)
    os.makedirs(CACHE, exist_ok=True)
    ext = ".aiff" if b == "say" else ".wav"
    key = hashlib.md5(f"{b}|{voice}|{steps}|{text}".encode("utf-8")).hexdigest()
    path = os.path.join(CACHE, key + ext)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    tmp = path + ".part" + ext
    if b == "powershell":
        env = dict(os.environ, AVATAR_TEXT=text, AVATAR_VOICE=voice or "", AVATAR_RATE=str(steps),
                   AVATAR_OUT=tmp)
        script = ("Add-Type -AssemblyName System.Speech; "
                  "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  "if ($env:AVATAR_VOICE) { $s.SelectVoice($env:AVATAR_VOICE) }; "
                  "$s.Rate = [int]$env:AVATAR_RATE; $s.SetOutputToWaveFile($env:AVATAR_OUT); "
                  "$s.Speak($env:AVATAR_TEXT); $s.Dispose()")
        _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], env=env)
    elif b == "say":
        cmd = ["say", "-r", str(int(175 * (1 + 0.12 * steps / 3))), "-o", tmp]
        if voice:
            cmd += ["-v", voice]
        _run(cmd + [text])
    else:
        cmd = [b, "-s", str(int(175 * (1 + 0.2 * steps / 3))), "-w", tmp]
        if voice:
            cmd += ["-v", voice]
        _run(cmd + [text])
    if not os.path.exists(tmp) or os.path.getsize(tmp) < 1000:
        raise TTSError("the speech engine produced no audio")
    os.replace(tmp, path)
    return path


class Job(threading.Thread):
    """Runs `synthesize` off the UI thread. Poll `done`, then read `path` or `error`."""

    def __init__(self, text, voice="", rate="Normal"):
        super().__init__(daemon=True)
        self.text, self.voice, self.rate = text, voice, rate
        self.path, self.error, self.done = None, None, False
        self.start()

    def run(self):
        try:
            self.path = synthesize(self.text, self.voice, self.rate)
        except (TTSError, OSError, subprocess.TimeoutExpired) as exc:
            self.error = str(exc)
        self.done = True


class VoiceList(threading.Thread):
    """Lists installed voices in the background (PowerShell start-up takes a second)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.voices, self.langs, self.done = [], {}, False
        self.start()

    def run(self):
        info = list_voices_info()
        self.voices = [name for name, _ in info]
        self.langs = dict(info)          # voice name -> two-letter language
        self.done = True
