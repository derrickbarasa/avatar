"""Hair styles, hats and facial hair (all fitted to the head surface)."""
import math

import numpy as np

from .head import MOUTH_Y, skull_ring
from .mathutil import catmull, mix, rot_x, rot_z, smoothstep
from .mesh import Mesh, ellipsoid, grid_faces, loft, tube, unit_sphere_z
from .strands import beard_margin, facial_strand_meshes, mustache_margin, strand_meshes

HAIR = dict(shine=40, spec=0.25, kind="hair")

HAIRLINES = {  # hairline height at azimuth 0, 40, 80, 110, 150, 180 degrees from the front
    "short": [0.36, 0.30, 0.20, 0.10, -0.05, -0.28],
    "long": [0.36, 0.30, 0.12, -0.02, -0.20, -0.35],
    "afro": [0.34, 0.30, 0.22, 0.15, 0.00, -0.25],
    "bangs": [0.27, 0.27, 0.12, -0.02, -0.20, -0.35],
    "undercut": [0.36, 0.34, 0.32, 0.30, 0.22, 0.05],     # only the top: the sides and back are shaved
}
for _name in ("buzz", "bun", "ponytail", "quiff", "mohawk", "curly", "pigtails", "braid"):
    HAIRLINES[_name] = HAIRLINES["short"]
HAIRLINES["bob"] = HAIRLINES["long"]
HAIRLINES["wavy"] = HAIRLINES["long"]
AZIMUTHS = np.radians([0, 40, 80, 110, 150, 180])


HAT_ANGLES = np.radians([0, 60, 110, 180])
HAT_LINES = {  # where a hat's lower edge sits at azimuth 0, 60, 110, 180 degrees from the front
    "beanie": [0.27, 0.20, -0.10, -0.18],
    "cap": [0.28, 0.26, 0.16, 0.10],
    "bucket": [0.13] * 4,
    "tophat": [0.20] * 4,
    "beret": [0.30, 0.25, 0.18, 0.12],
}


def hair_meshes(head, style, color, hat="none"):
    """Strands grown over a darker under-layer (scalp, hair mass) that hides any gaps.

    Under a hat there is no hair on the covered scalp: only what grows below the hat's edge is drawn."""
    if style == "bald":
        return []
    theta = np.abs(np.arctan2(head.X, head.Z))
    margin = head.Y - np.interp(theta, AZIMUTHS, HAIRLINES[style])
    if hat in HAT_LINES:
        margin = np.minimum(margin, np.interp(theta, HAT_ANGLES, HAT_LINES[hat]) - 0.03 - head.Y)
        if (margin > 0.03).sum() < 40:             # nothing left between the hairline and the hat
            return []
    strands = strand_meshes(head, style, color, margin)
    under = mix(color, (0, 0, 0), 0.38) if strands else color
    out = _under_layer(head, style, under, margin, hat != "none") + strands
    if style == "undercut":                    # the shaved sides: a dark shadow of stubble on the skin
        shaved = head.Y - np.interp(theta, AZIMUTHS, HAIRLINES["short"])
        out.append(head.shell(np.minimum(shaved, -margin + 0.05), 0.010, mix(color, (0, 0, 0), 0.3), 0.04,
                              opacity=0.55, **HAIR))
    return out


def _under_layer(head, style, color, margin, hatted=False):
    if style == "afro":
        return [afro_mesh(color)]
    top = np.clip(head.Y, 0, 1)
    if style in ("buzz", "mohawk"):
        thick = 0.014
    elif style == "curly":
        lumps = (np.sin(15 * head.X + 1) * np.sin(14 * head.Y) * np.sin(16 * head.Z + 2)
                 + np.sin(9 * head.X + head.Z * 7) * 0.5)
        thick = 0.075 + 0.035 * lumps
    else:
        thick = 0.05 + 0.03 * top ** 2
    out = [head.shell(margin, thick, color, 0.05, **HAIR)]
    if style == "bun":
        out.append(ellipsoid((0, 0.70, -0.22), (0.21, 0.21, 0.21), color, **HAIR))
    elif style == "long" and not hatted:          # (under a hat the strands that hid this slab are gone)
        out.append(hair_curtain(color, 1.35))
    elif style in ("bob", "bangs") and not hatted:
        out.append(hair_curtain(color, 0.7))
    elif style == "wavy" and not hatted:
        out.append(hair_curtain(color, 0.95))
    elif style == "pigtails":
        for s in (-1, 1):
            path = catmull([(s * 0.44, 0.10, -0.28), (s * 0.60, -0.05, -0.32), (s * 0.66, -0.55, -0.28),
                            (s * 0.64, -1.05, -0.20)], 20)
            radii = catmull([(0.06,), (0.10,), (0.09,), (0.03,)], 20)[:, 0]
            out.append(tube(path, radii, color, **HAIR))
            out.append(ellipsoid((s * 0.45, 0.10, -0.28), (0.07, 0.07, 0.06), (0.85, 0.25, 0.35), thin=True))
    elif style == "braid":
        path = catmull([(0, 0.26, -0.52), (0, 0.05, -0.74), (0, -0.60, -0.76), (0, -1.25, -0.70)], 30)
        radii = np.linspace(0.075, 0.04, 30)
        out.append(tube(path, radii, color, **HAIR))
        out.append(ellipsoid((0, 0.26, -0.52), (0.08, 0.08, 0.06), (0.85, 0.25, 0.35), thin=True))
        out.append(ellipsoid((0, -1.25, -0.70), (0.06, 0.05, 0.06), (0.85, 0.25, 0.35), thin=True))
    elif style == "ponytail":
        path = catmull([(0, 0.32, -0.50), (0, 0.18, -0.74), (0, -0.25, -0.80), (0, -0.78, -0.70)], 20)
        radii = catmull([(0.07,), (0.11,), (0.09,), (0.025,)], 20)[:, 0]
        out.append(tube(path, radii, color, **HAIR))
        out.append(ellipsoid((0, 0.30, -0.52), (0.09, 0.09, 0.06), (0.85, 0.25, 0.35), thin=True))
    elif style == "mohawk":
        ridge = catmull([(0, 0.50, 0.38), (0, 0.72, 0.12), (0, 0.70, -0.25), (0, 0.40, -0.55)], 24)
        radii = catmull([(0.05,), (0.10,), (0.10,), (0.05,)], 24)[:, 0]
        out.append(tube(ridge, radii, color, **HAIR))
    return out


def hat_meshes(head, kind, color):
    if kind == "none":
        return []
    theta = np.abs(np.arctan2(head.X, head.Z))
    keys = HAT_ANGLES
    top = np.clip(head.Y, 0, 1)
    fabric = dict(shine=8, spec=0.03, kind="cloth")
    if kind == "beanie":
        levels = HAT_LINES["beanie"]
        out = [head.shell(head.Y - np.interp(theta, keys, levels), 0.07 + 0.02 * top,
                          color, 0.03, **fabric)]
        phi = np.linspace(math.pi, 3 * math.pi, 40)  # start at the back so the seam is hidden
        y = np.interp(np.abs((phi + math.pi) % (2 * math.pi) - math.pi), keys, levels) + 0.005
        cuff = skull_ring(phi, y, 0.09)
        out.append(tube(cuff, np.full(len(phi), 0.05), mix(color, (0, 0, 0), 0.15), sides=10,
                        cap_ends=False, **fabric))
        return out
    if kind == "bucket":
        out = [head.shell(head.Y - 0.13, 0.075 + 0.02 * top, color, 0.03, **fabric)]
        out.append(loft(np.array([0.24, 0.18, 0.14]), np.array([0.62, 0.74, 0.80]), np.array([0.68, 0.80, 0.86]),
                        np.array([0.68, 0.80, 0.86]), color, cap_ends=False, **fabric))
        out.append(_hat_band(0.24, 0.085, mix(color, (0, 0, 0), 0.35), fabric))
        return out
    if kind == "tophat":
        ys = np.linspace(0.20, 0.98, 12)
        taper = 1 - 0.06 * (ys - 0.20) / 0.78
        out = [loft(ys, 0.60 * taper, 0.66 * taper, 0.66 * taper, color, **fabric),
               loft(np.array([0.21, 0.17]), np.array([0.62, 0.95]), np.array([0.68, 1.02]),
                    np.array([0.68, 1.02]), color, **fabric),
               _hat_band(0.30, 0.13, mix(color, (1, 1, 1), 0.5), fabric, radius=0.10)]
        return out
    if kind == "beret":
        levels = HAT_LINES["beret"]
        tilt = rot_x(math.radians(-8)) @ rot_z(math.radians(-12))
        return [head.shell(head.Y - np.interp(theta, keys, levels), 0.05 + 0.03 * top, color, 0.03, **fabric),
                ellipsoid((0.10, 0.50, -0.02), (0.56, 0.12, 0.56), color, rot=tilt, detail=28, **fabric),
                ellipsoid((0.10, 0.63, -0.02), (0.03, 0.04, 0.03), mix(color, (0, 0, 0), 0.25), **fabric)]
    if kind == "headband":
        phi = np.linspace(math.pi, 3 * math.pi, 48)
        ring = skull_ring(phi, 0.36, 0.062)                                          # sits on top of the hair's volume
        out = [tube(ring, np.full(len(phi), 0.03), color, sides=8, cap_ends=False, **fabric)]
        knot = skull_ring(np.array([math.pi * 0.62]), 0.36, 0.062)[0]                # a little knot at one side
        out.append(ellipsoid(knot + np.array([0.02, 0.0, 0.0]), (0.06, 0.06, 0.05), mix(color, (0, 0, 0), 0.2), **fabric))
        return out
    levels = HAT_LINES["cap"]
    return [head.shell(head.Y - np.interp(theta, keys, levels), 0.06 + 0.03 * top, color, 0.03, **fabric),
            ellipsoid((0, 0.27, 0.60), (0.29, 0.022, 0.30), color, rot=rot_x(math.radians(14)),
                      detail=28, **fabric),
            ellipsoid((0, 0.66, 0.0), (0.035, 0.03, 0.035), mix(color, (0, 0, 0), 0.2), **fabric)]


def _hat_band(y, thick, color, material, radius=0.0):
    """A ribbon around the hat just above the brim."""
    phi = np.linspace(math.pi, 3 * math.pi, 40)
    ring = skull_ring(phi, min(y, 0.6), 0.08 + radius)
    return tube(ring, np.full(len(phi), thick * 0.4), color, sides=10, cap_ends=False, **material)


def headphone_meshes(color):
    """Over-ear headphones: a band over the head and two cups with a coloured ring."""
    glossy = dict(shine=50, spec=0.3, kind="glossy")
    dark = (0.12, 0.12, 0.15)
    th = np.radians(np.linspace(-90, 90, 30))
    band = np.stack([0.60 * np.sin(th), 0.02 + 0.74 * np.cos(th), np.full(30, -0.02)], -1)
    out = [tube(band, np.full(30, 0.03), dark, sides=10, **glossy)]
    for s in (-1, 1):
        out.append(ellipsoid((s * 0.60, -0.02, -0.02), (0.085, 0.17, 0.15), dark, detail=20, **glossy))
        out.append(ellipsoid((s * 0.665, -0.02, -0.02), (0.02, 0.12, 0.11), color, detail=16, **glossy))
    return out


def afro_mesh(color):
    """A big lumpy ball set back and up so it frames the face."""
    p = unit_sphere_z(64, 33)
    lumps = 1 + 0.035 * np.sin(7 * p[:, 0] + 1) * np.sin(6 * p[:, 1]) * np.sin(8 * p[:, 2] + 2)
    center = np.array([0.0, 0.42, -0.20])
    verts = p * lumps[:, None] * (0.82, 0.66, 0.80) + center
    return Mesh(verts, grid_faces(64, 33), color, ref=center, **HAIR)


def hair_curtain(color, length):
    nu, nv = 48, 28
    phi = np.radians(np.linspace(-108, 108, nu))[None, :]
    t = np.linspace(0, 1, nv)[:, None]
    y_end = -length + 0.10 * min(1.0, length) * np.cos(3 * phi)  # wavy hem
    y = 0.05 + (y_end - 0.05) * t
    rx = 0.535 + 0.06 * np.sin(np.pi * 0.9 * t)  # fuller through the middle
    rz = 0.60 - 0.24 * smoothstep(0.0, -0.9, y)
    taper = 1 - 0.20 * smoothstep(0.6, 1.0, t)
    x = rx * np.sin(phi) * taper
    z = -rz * np.cos(phi) * taper
    p = np.stack([x, y, z], -1).reshape(-1, 3)
    ref = np.stack([np.zeros(len(p)), p[:, 1], np.zeros(len(p))], -1)
    return Mesh(p, grid_faces(nu, nv, wrap=False), color, ref=ref, **HAIR)


def facial_hair_meshes(head, style, color):
    """Moustache and beard grown as strands over a thin, darker skin-tone layer."""
    if style == "none":
        return []
    beard = style in ("beard", "stubble", "goatee")
    dark = mix(color, (0, 0, 0), 0.4)
    margin = np.full(head.Y.shape, -1.0)
    out = []
    if beard:
        margin = beard_margin(head)
        if style == "goatee":                                           # just the chin, under the lower lip
            margin = np.minimum(margin, np.minimum(0.17 - np.abs(head.X), (-0.405 - head.Y) * 1.5))
        shadow = 0.5 if style == "stubble" else 0.35
        out.append(head.shell(margin * 0.30, 0.012 if style == "stubble" else 0.02, dark, 0.03, opacity=shadow, **HAIR))
    out.append(head.shell(mustache_margin(head), 0.010, dark, 0.3, **HAIR))
    out += facial_strand_meshes(head, margin, color, style if beard else False)
    for m in out:
        m.jaw_follow = True          # the beard moves with the jaw
    return out
