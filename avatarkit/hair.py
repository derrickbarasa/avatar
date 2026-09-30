"""Hair styles, hats and facial hair (all fitted to the head surface)."""
import math

import numpy as np

from .head import MOUTH_Y
from .mathutil import catmull, mix, rot_x, smoothstep
from .mesh import Mesh, ellipsoid, grid_faces, tube, unit_sphere_z
from .strands import facial_strand_meshes, mustache_margin, strand_meshes

HAIR = dict(shine=40, spec=0.25, kind="hair")

HAIRLINES = {  # hairline height at azimuth 0, 40, 80, 110, 150, 180 degrees from the front
    "short": [0.36, 0.30, 0.20, 0.10, -0.05, -0.28],
    "long": [0.36, 0.30, 0.12, -0.02, -0.20, -0.35],
    "afro": [0.34, 0.30, 0.22, 0.15, 0.00, -0.25],
    "bangs": [0.27, 0.27, 0.12, -0.02, -0.20, -0.35],
}
for _name in ("buzz", "bun", "ponytail", "quiff", "mohawk", "curly"):
    HAIRLINES[_name] = HAIRLINES["short"]
HAIRLINES["bob"] = HAIRLINES["long"]
AZIMUTHS = np.radians([0, 40, 80, 110, 150, 180])


def hair_meshes(head, style, color):
    """Strands grown over a darker under-layer (scalp, hair mass) that hides any gaps."""
    if style == "bald":
        return []
    theta = np.abs(np.arctan2(head.X, head.Z))
    margin = head.Y - np.interp(theta, AZIMUTHS, HAIRLINES[style])
    strands = strand_meshes(head, style, color, margin)
    under = mix(color, (0, 0, 0), 0.38) if strands else color
    return _under_layer(head, style, under, margin) + strands


def _under_layer(head, style, color, margin):
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
    elif style == "long":
        out.append(hair_curtain(color, 1.35))
    elif style in ("bob", "bangs"):
        out.append(hair_curtain(color, 0.7))
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
    keys = np.radians([0, 60, 110, 180])
    top = np.clip(head.Y, 0, 1)
    fabric = dict(shine=8, spec=0.03, kind="cloth")
    if kind == "beanie":
        levels = [0.27, 0.20, -0.10, -0.18]
        out = [head.shell(head.Y - np.interp(theta, keys, levels), 0.07 + 0.02 * top,
                          color, 0.03, **fabric)]
        phi = np.linspace(math.pi, 3 * math.pi, 40)  # start at the back so the seam is hidden
        y = np.interp(np.abs((phi + math.pi) % (2 * math.pi) - math.pi), keys, levels) + 0.005
        half = np.sqrt(np.clip(1 - (y / 0.62) ** 2, 0, 1))
        cuff = np.stack([(0.5 * half + 0.09) * np.sin(phi), y, (0.56 * half + 0.09) * np.cos(phi)], -1)
        out.append(tube(cuff, np.full(len(phi), 0.05), mix(color, (0, 0, 0), 0.15), sides=10,
                        cap_ends=False, **fabric))
        return out
    levels = [0.28, 0.26, 0.16, 0.10]
    return [head.shell(head.Y - np.interp(theta, keys, levels), 0.06 + 0.03 * top, color, 0.03, **fabric),
            ellipsoid((0, 0.27, 0.60), (0.29, 0.022, 0.30), color, rot=rot_x(math.radians(14)),
                      detail=28, **fabric),
            ellipsoid((0, 0.66, 0.0), (0.035, 0.03, 0.035), mix(color, (0, 0, 0), 0.2), **fabric)]


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
    beard = style == "beard"
    dark = mix(color, (0, 0, 0), 0.4)
    margin = np.full(head.Y.shape, -1.0)
    out = []
    if beard:
        theta = np.abs(np.arctan2(head.X, head.Z))
        jaw = np.minimum(-0.22 - head.Y, (np.radians(85) - theta) * 0.5)
        mouth = np.sqrt((head.X / 0.19) ** 2 + ((head.Y - MOUTH_Y) / 0.065) ** 2) - 1
        margin = np.where(head.Y > -0.22, -1.0, np.minimum(jaw, mouth * 0.15))
        out.append(head.shell(margin * 0.35, 0.03, dark, 0.03, **HAIR))   # soft top edge
    out.append(head.shell(mustache_margin(head), 0.010, dark, 0.3, **HAIR))
    return out + facial_strand_meshes(head, margin, color, beard)
