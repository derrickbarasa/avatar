"""Encode rendered frames into a video (MP4 with audio) or an animated GIF."""
import os
import shutil
import subprocess


class RecordError(RuntimeError):
    pass


def ffmpeg_exe():
    """Path to an ffmpeg binary: the system one, else the one bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def can_make_mp4():
    return ffmpeg_exe() is not None


def write_mp4(path, frames, size, fps, audio=None, on_progress=None, total=None, audio_offset=0.0):
    """Pipe raw RGB frames (an iterable of bytes) into ffmpeg; muxes `audio` if given."""
    exe = ffmpeg_exe()
    if exe is None:
        raise RecordError("ffmpeg not found (pip install imageio-ffmpeg)")
    w, h = size
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-"]
    if audio:                       # -itsoffset delays the sound so it lines up with the lips
        cmd += ["-itsoffset", f"{audio_offset:.3f}", "-i", audio, "-c:a", "aac", "-b:a", "160k"]   # the video runs a beat past the sound
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", path]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    try:
        for i, frame in enumerate(frames):
            proc.stdin.write(frame)
            if on_progress:
                on_progress(i + 1, total)
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "ignore")
        if proc.wait() != 0:
            raise RecordError(err.strip()[-300:] or "ffmpeg failed")
    except OSError:                # ffmpeg quit early: Windows reports that as EINVAL, not EPIPE
        proc.kill()
        err = proc.stderr.read().decode("utf-8", "ignore").strip()[-300:]
        raise RecordError(err or "ffmpeg closed early")
    except BaseException:
        proc.kill()
        raise
    return path


def write_gif(path, frames, size, fps, max_width=480, on_progress=None, total=None):
    """Animated GIF (no sound) via Pillow, scaled down to keep the file small."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RecordError("Pillow is needed for GIF export") from exc
    w, h = size
    scale = min(1.0, max_width / w)
    out_size = (max(2, int(w * scale)), max(2, int(h * scale)))
    images = []
    for i, frame in enumerate(frames):
        img = Image.frombytes("RGB", size, frame)
        if scale < 1:
            img = img.resize(out_size, Image.LANCZOS)
        images.append(img.quantize(colors=128, dither=Image.NONE))
        if on_progress:
            on_progress(i + 1, total)
    if not images:
        raise RecordError("no frames were rendered")
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0,
                   optimize=False, disposal=2)
    return path


def encode(path, frames, size, fps, audio=None, on_progress=None, total=None, audio_offset=0.0):
    """Write `frames` to `path`; falls back to a GIF next to it if MP4 can't be made.

    Returns (actual_path, note) where note explains a fallback ("" if none).
    """
    if path.lower().endswith(".mp4") and can_make_mp4():
        return write_mp4(path, frames, size, fps, audio, on_progress, total, audio_offset), ""
    gif = os.path.splitext(path)[0] + ".gif"
    note = "" if path.lower().endswith(".gif") else "ffmpeg not found, so this is a silent GIF"
    return write_gif(gif, frames, size, fps, on_progress=on_progress, total=total), note
