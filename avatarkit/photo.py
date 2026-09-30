"""Suggest avatar options from a photo (skin tone, hair colour/length, facial hair).

This is a colour-and-position heuristic, not face recognition. If OpenCV is
installed (`pip install opencv-python-headless`) it is used to find the face;
otherwise the face is assumed to be in the centre of the picture.
"""
import numpy as np
import pygame

from . import options as O


def load_image(path):
    """Read an image file into an (H, W, 3) uint8 RGB array."""
    surf = pygame.image.load(path)
    return np.ascontiguousarray(pygame.surfarray.array3d(surf).transpose(1, 0, 2))


def _srgb_to_lab(rgb):
    c = np.asarray(rgb, float).reshape(-1, 3)
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (c @ m.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], -1)


def _dist(a, b, l_weight=1.0):
    d = _srgb_to_lab(a)[0] - _srgb_to_lab(b)[0]
    return float(np.sqrt((l_weight * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2))


def _patch(img, cx, cy, hw, hh):
    h, w, _ = img.shape
    x0, x1 = int(max(0, cx - hw)), int(min(w, cx + hw))
    y0, y1 = int(max(0, cy - hh)), int(min(h, cy + hh))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return np.median(img[y0:y1, x0:x1].reshape(-1, 3), axis=0) / 255.0


def detect_face(img):
    """Return ((x, y, w, h), method): OpenCV's face box if available, else a centred guess."""
    h, w, _ = img.shape
    why = "OpenCV not installed"
    try:
        import cv2
        if not hasattr(cv2, "CascadeClassifier"):
            why = "this OpenCV has no CascadeClassifier (use opencv-python-headless 4.x)"
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(max(40, w // 12),) * 2)
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                return (int(x), int(y), int(fw), int(fh)), "OpenCV"
            why = "OpenCV found no face"
    except ImportError:
        pass
    side = 0.38 * min(w, h)
    return (int(w / 2 - side / 2), int(h * 0.42 - side / 2), int(side), int(side)), f"centre guess ({why})"


def analyze(img, face_box=None):
    """Return ({option: value name}, notes list) inferred from an RGB image array.

    `face_box` (x, y, w, h) skips detection, for callers that already know where the face is.
    """
    (fx, fy, fw, fh), method = ((face_box, "given") if face_box else detect_face(img))
    notes = [f"face: {method}"]
    # Undo over/under-exposure: assume the brightest 5% of the picture should be near white.
    luma = (img[..., :3] @ np.array([0.299, 0.587, 0.114])) / 255.0
    gain = float(np.clip(0.92 / max(np.percentile(luma, 95), 0.05), 0.5, 2.2))
    if abs(gain - 1) > 0.08:
        img = np.clip(img.astype(float) * gain, 0, 255).astype(np.uint8)
        notes.append(f"exposure x{gain:.2f}")
    h, w, _ = img.shape
    result = {}

    def at(u, v, half=0.10):
        return _patch(img, fx + u * fw, fy + v * fh, half * fw, half * fh)

    cheeks = [p for p in (at(0.28, 0.58), at(0.72, 0.58), at(0.5, 0.22, 0.08)) if p is not None]
    skin = np.median(cheeks, axis=0)
    tones = [(n, c) for n, c in O.SKIN_TONES]
    result["skin"] = min(tones, key=lambda t: _dist(skin, t[1], 0.8))[0]

    corners = [p for p in (_patch(img, w * 0.06, h * 0.06, w * 0.05, h * 0.05),
                           _patch(img, w * 0.94, h * 0.06, w * 0.05, h * 0.05)) if p is not None]
    bg = np.median(corners, axis=0) if corners else np.array([0.8, 0.8, 0.8])
    # Look at a few strips just above the face and keep the one that stands out most.
    strips = [p for p in (_patch(img, fx + fw / 2, fy + v * fh, 0.30 * fw, 0.06 * fh)
                          for v in (-0.02, -0.10, -0.20)) if p is not None]
    if strips:
        stats = [(min(_dist(p, skin), _dist(p, bg)), p) for p in strips]
        best_score, hair_p = max(stats, key=lambda t: t[0])
        if best_score > 14:
            hsv_sat = (hair_p.max() - hair_p.min()) / max(hair_p.max(), 1e-3)
            natural = O.HAIR_COLORS[:8] if hsv_sat < 0.5 else O.HAIR_COLORS
            result["haircolor"] = min(natural, key=lambda t: _dist(hair_p, t[1], 0.8))[0]
            sides = [p for p in (_patch(img, fx - 0.25 * fw, fy + 0.95 * fh, 0.12 * fw, 0.15 * fh),
                                 _patch(img, fx + 1.25 * fw, fy + 0.95 * fh, 0.12 * fw, 0.15 * fh))
                     if p is not None]
            long_hair = len(sides) == 2 and all(_dist(p, bg) > 18 and _dist(p, hair_p) < 28 for p in sides)
            result["hair"] = "Long" if long_hair else "Short"
        elif all(_dist(p, skin) < 14 for p in strips):
            result["hair"] = "Bald"
        else:
            notes.append("hair colour is too close to the background to tell")
    else:
        notes.append("no room above the head to read hair")

    l_skin = _srgb_to_lab(skin)[0][0]
    chin, stache = at(0.5, 0.90, 0.12), at(0.5, 0.74, 0.10)
    dark = lambda p: p is not None and l_skin - _srgb_to_lab(p)[0][0] > 16
    if dark(chin):
        result["facial"] = "Beard"
    elif dark(stache):
        result["facial"] = "Mustache"
    else:
        result["facial"] = "None"
    return result, notes


def apply_photo(state, path):
    """Load `path`, update `state` in place, and return a one-line summary."""
    result, notes = analyze(load_image(path))
    for key, name in result.items():
        O.set_by_name(state, key, name)
    changed = ", ".join(f"{k}={v}" for k, v in result.items())
    return f"Photo: {changed} ({'; '.join(notes)})"
