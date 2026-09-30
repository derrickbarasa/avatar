"""Where avatars live: the library folder, `.avatar` files, autosave, bundles and dialogs.

Everything a person creates goes under one folder they can find and back up
(`~/Documents/AvatarStudio` by default, or `$AVATARKIT_HOME`):

    avatars/   saved .avatar files (with a thumbnail inside)
    exports/   PNG, GLB, VRM, OBJ, zip bundles, clips
    packs/     optional extra content (colours, presets, phrases)
    last.avatar   autosave, reopened on the next start
"""
import base64
import datetime
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass

from . import options as O

FORMAT = "avatarkit/avatar"
VERSION = 3
THUMB_SIZE = (168, 210)


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
def home():
    root = os.environ.get("AVATARKIT_HOME") or os.path.join(os.path.expanduser("~"), "Documents", "AvatarStudio")
    os.makedirs(root, exist_ok=True)
    return root


def folder(name):
    path = os.path.join(home(), name)
    os.makedirs(path, exist_ok=True)
    return path


def avatars_dir():
    return folder("avatars")


def exports_dir():
    return folder("exports")


def packs_dir():
    return folder("packs")


def slug(name, fallback="avatar"):
    s = re.sub(r"\s+", "_", re.sub(r"[^A-Za-z0-9._ -]+", "", name).strip())
    return s[:60] or fallback


def unique_path(directory, base, ext):
    """directory/base.ext, or base_2.ext, base_3.ext ... if that already exists."""
    path = os.path.join(directory, base + ext)
    n = 2
    while os.path.exists(path):
        path = os.path.join(directory, f"{base}_{n}{ext}")
        n += 1
    return path


def open_folder(path, select=None):
    """Show a folder (or reveal a file) in the system file manager."""
    try:
        if sys.platform.startswith("win"):
            if select and os.path.exists(select):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(select)])
            else:
                os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", select] if select else ["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# .avatar files
# ---------------------------------------------------------------------------
@dataclass
class AvatarInfo:
    path: str
    name: str
    modified: float
    thumbnail: bytes = b""


def _describe(state):
    return {key: opts[state[key]][0] for key, _, opts in O.OPTIONS if key not in ("preset",)}


def save_avatar(state, name, path=None, thumbnail_png=b""):
    """Write an .avatar file (JSON: readable option names + share code + optional thumbnail)."""
    name = name.strip() or "My avatar"
    path = path or os.path.join(avatars_dir(), slug(name) + ".avatar")
    data = {"format": FORMAT, "version": VERSION, "name": name,
            "saved": datetime.datetime.now().isoformat(timespec="seconds"),
            "code": O.encode_state(state), "options": _describe(state)}
    if thumbnail_png:
        data["thumbnail"] = base64.b64encode(thumbnail_png).decode("ascii")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)          # never leave a half-written file behind
    return path


def _apply(data, state):
    """Apply loaded avatar data to `state`; returns how many options were understood."""
    if data.get("code"):
        try:
            O.decode_state(data["code"], state)
            return len(O.CODE_KEYS)
        except ValueError:
            pass
    applied = 0
    for key, val in (data.get("options") or {}).items():
        try:
            O.set_by_name(state, key, val)
            applied += 1
        except ValueError:
            pass            # option renamed or removed in a newer/older version
    return applied


def load_avatar(path, state):
    """Load .avatar, legacy option-name .json, or a text file containing a share code.

    Returns the avatar's name. Raises ValueError with a readable message if the file
    is not an avatar.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Can't read {os.path.basename(path)}: {exc}") from exc
    raw = raw.strip()
    if raw.startswith(O.CODE_PREFIX):                          # a bare share code
        O.decode_state(raw.split()[0], state)
        return os.path.splitext(os.path.basename(path))[0]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{os.path.basename(path)} is not an avatar file") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{os.path.basename(path)} is not an avatar file")
    if "options" not in data and "code" not in data:           # legacy: plain {option: name}
        data = {"options": data}
    if data.get("format", FORMAT) != FORMAT:
        raise ValueError("That file is a different kind of JSON")
    if _apply(data, state) == 0:
        raise ValueError(f"{os.path.basename(path)} has no avatar options")
    return data.get("name") or os.path.splitext(os.path.basename(path))[0]


def list_avatars():
    """Saved avatars, newest first."""
    out = []
    for fn in os.listdir(avatars_dir()):
        if not fn.endswith(".avatar"):
            continue
        path = os.path.join(avatars_dir(), fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            thumb = base64.b64decode(data["thumbnail"]) if data.get("thumbnail") else b""
            out.append(AvatarInfo(path, data.get("name") or fn[:-7], os.path.getmtime(path), thumb))
        except (OSError, ValueError, KeyError):
            continue                                            # skip damaged files
    return sorted(out, key=lambda a: -a.modified)


def delete_avatar(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Autosave
# ---------------------------------------------------------------------------
def autosave_path():
    return os.path.join(home(), "last.avatar")


def autosave(state, name="Untitled"):
    try:
        save_avatar(state, name, autosave_path())
    except OSError:
        pass


def restore_autosave(state):
    """Load the autosave into `state`; returns its name or None."""
    path = autosave_path()
    if not os.path.exists(path):
        return None
    try:
        return load_avatar(path, state)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------
BUNDLE_README = """{name}
Made with Avatar Studio ({stamp})

  {base}.glb            posed 3D model with skeleton (Blender, Unity, Godot, three.js)
  {base}_skinned.glb    same, with a real skin so engines can animate it
  {base}.vrm            VRM 1.0 humanoid (VRoid-compatible viewers, VTubing apps)
  {base}.obj / .mtl     static mesh with the current pose baked in
  {base}.png            picture with the scene background
  {base}_transparent.png  picture with a transparent background
  {base}.avatar         open this in Avatar Studio to keep editing
  code.txt              the share code: paste it in Avatar Studio (Ctrl+V)

Share code: {code}
"""


def write_bundle(path, name, files, code):
    """Zip `files` ({archive_name: file_path}) with a README and the share code."""
    base = slug(name)
    stamp = datetime.date.today().isoformat()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in files.items():
            z.write(src, arc)
        z.writestr("README.txt", BUNDLE_README.format(name=name, base=base, stamp=stamp, code=code))
        z.writestr("code.txt", code + "\n")
    return path


def png_bytes(surface, size=None):
    """PNG bytes of a pygame surface (optionally scaled to `size`)."""
    import pygame
    if size:
        surface = pygame.transform.smoothscale(surface, size)
    buf = io.BytesIO()
    pygame.image.save(surface, buf, "x.png")
    return buf.getvalue()


def thumbnail_from(surface):
    """Crop a viewport grab to the thumbnail aspect ratio, centred, and scale it down."""
    import pygame
    w, h = surface.get_size()
    tw, th = THUMB_SIZE
    want = tw / th
    if w / h > want:
        cw, ch = int(h * want), h
    else:
        cw, ch = w, int(w / want)
    crop = surface.subsurface(pygame.Rect((w - cw) // 2, (h - ch) // 2, cw, ch)).copy()
    return png_bytes(crop, THUMB_SIZE)


# ---------------------------------------------------------------------------
# Native dialogs (optional: tkinter ships with most Python installs)
# ---------------------------------------------------------------------------
def _dialog(kind, **kw):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        fn = filedialog.asksaveasfilename if kind == "save" else filedialog.askopenfilename
        path = fn(**kw)
        root.destroy()
        return path or None
    except Exception:
        return None


def ask_save_path(default_name, ext=".avatar", title="Save avatar"):
    return _dialog("save", title=title, initialdir=avatars_dir(), initialfile=default_name + ext,
                   defaultextension=ext, filetypes=[("Avatar files", "*" + ext), ("All files", "*.*")])


def ask_open_path(title="Open avatar"):
    return _dialog("open", title=title, initialdir=avatars_dir(),
                   filetypes=[("Avatar files", "*.avatar *.json *.txt"), ("All files", "*.*")])
