"""The sculpted head and its face details (eyes, brows, lips, glasses, earrings)."""
import functools
import math

import numpy as np

from .mathutil import catmull, gauss, mix, rot_y, smoothstep, unit
from .mesh import Mesh, cap, ellipsoid, grid_faces, tube

MOUTH_Y = -0.36
EYE_X, EYE_Y = 0.20, 0.03
DARK = (0.10, 0.07, 0.07)
GOLD = (0.95, 0.78, 0.30)


class Head:
    """Sculpted head: an ellipsoid with a tapered jaw and gaussian features."""
    NU, NV = 128, 96

    def __init__(self, skin, jaw, nose):
        skin = np.asarray(skin, float)
        lat = np.linspace(-math.pi / 2, math.pi / 2, self.NV)[:, None]
        lon = np.linspace(0, 2 * math.pi, self.NU, endpoint=False)[None, :]
        dx = np.cos(lat) * np.sin(lon)
        dy = np.sin(lat) * np.ones_like(lon)
        dz = np.cos(lat) * np.cos(lon)
        x, y, z = 0.50 * dx, 0.62 * dy, 0.56 * dz
        low = np.clip(-dy, 0, 1) ** 1.6
        x, z = x * (1 - jaw * low), z * (1 - 0.5 * jaw * low)

        def g(cx, cy, sx, sy):
            return gauss(x, y, cx, cy, sx, sy)

        def pair(cx, cy, sx, sy):
            return g(cx, cy, sx, sy) + g(-cx, cy, sx, sy)

        d = 0.06 * nose * g(0, 0.0, 0.04, 0.14)                    # nose bridge
        d += 0.11 * nose * g(0, -0.20, 0.055, 0.05)                # nose tip
        d += 0.03 * nose * pair(0.08, -0.23, 0.04, 0.035)          # nostrils
        d -= 0.035 * pair(EYE_X, EYE_Y, 0.11, 0.075)               # eye sockets
        d += 0.03 * pair(0.2, 0.14, 0.15, 0.04)                    # brow ridge
        d += 0.035 * pair(0.30, -0.12, 0.13, 0.10)                 # cheekbones
        d += 0.03 * g(0, MOUTH_Y, 0.17, 0.06)                      # mouth area
        d += 0.03 * g(0, -0.56, 0.12, 0.05)                        # chin
        front = smoothstep(0.0, 0.5, dz)
        z = z + front * d

        blush = 0.22 * front * pair(0.30, -0.16, 0.11, 0.09)
        socket = 0.12 * pair(EYE_X, EYE_Y, 0.12, 0.07)
        target = skin * 0.75 + np.array([0.25, 0.05, 0.05])
        colors = skin * (1 - blush[..., None]) + target * blush[..., None]
        colors = colors * (1 - socket[..., None])

        self.P = np.stack([x, y, z], -1).reshape(-1, 3)
        self.X, self.Y, self.Z = self.P.T
        self.faces = grid_faces(self.NU, self.NV)
        self.mesh = Mesh(self.P, self.faces, skin, ref=(0, 0, 0), colors=colors,
                         per_face=False, shine=15, spec=0.05, kind="skin")
        front_mask = self.Z > 0.05
        self._fx, self._fy, self._fz = self.X[front_mask], self.Y[front_mask], self.Z[front_mask]

    def z_at(self, x, y):
        """Height of the face surface at (x, y)."""
        d2 = (self._fx - x) ** 2 + (self._fy - y) ** 2
        idx = np.argpartition(d2, 4)[:4]
        w = 1.0 / (d2[idx] + 1e-6)
        return float((self._fz[idx] * w).sum() / w.sum())

    def shell(self, margin, thickness, color, edge=0.05, **kw):
        """Offset copy of the head where margin > 0.

        The border fades out (alpha) and sinks into the scalp, so the edge
        is soft instead of following the mesh grid.
        """
        m = margin.ravel()
        t = np.broadcast_to(thickness, margin.shape).ravel() * smoothstep(0, edge, m) + 0.007
        verts = self.P + self.mesh.n * t[:, None]
        faces = self.faces[(m > 0)[self.faces].any(1)]
        alpha = smoothstep(0, 0.035, m)
        colors = np.column_stack([np.tile(color, (len(m), 1)), alpha])
        return Mesh(verts, faces, color, ref=(0, 0, 0), per_face=False, colors=colors, **kw)

    def on_face(self, xs, ys, lift=0.0):
        return np.array([[x, y, self.z_at(x, y) + lift] for x, y in zip(xs, ys)])


@functools.lru_cache(maxsize=4)
def cached_head(skin, jaw, nose):
    return Head(skin, jaw, nose)


def eye_meshes(head, side, size, iris, skin, lid_delta=0.0):
    """Eyeball (procedural iris in the shader), eyelid and lash line."""
    r = 0.095 * size
    x = side * EYE_X
    c = np.array([x, EYE_Y, head.z_at(x, EYE_Y) - 0.45 * r])
    ball = ellipsoid(c, (r,) * 3, (0.96, 0.95, 0.94), detail=40, shine=70, spec=0.4, kind="eye")
    ball.set(color2=tuple(iris), center=tuple(c))
    # Upper eyelid: a dome tilted back so its edge arches over the eye.
    tilt, half = math.radians(22 + lid_delta), math.radians(90)
    direction = np.array([0.0, math.cos(tilt), -math.sin(tilt)])
    lid = cap(c, r * 1.11, direction, half, skin, detail=22, shine=25, spec=0.12, kind="skin")
    # Dark lash line along the lid edge.
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(direction, e1)
    psi = np.linspace(0, 2 * math.pi, 240)[:, None]
    edge = direction * math.cos(half) + (e1 * np.cos(psi) + e2 * np.sin(psi)) * math.sin(half)
    edge = edge[edge[:, 2] > 0.25]
    edge = edge[np.argsort(edge[:, 0])][::4]
    lash = tube(c + edge * r * 1.113, np.full(len(edge), 0.0035), DARK, sides=6)
    for m in (lid, lash):
        m.anim = ("blink", tuple(c))
    return [ball, lid, lash]


def brow_mesh(head, side, thickness, color, dy=0.0, tilt=0.0):
    t = np.linspace(0, 1, 14)
    xs = side * (0.075 + 0.20 * t)
    ys = 0.185 + 0.045 * np.sin(t * math.pi * 0.85) - 0.02 * t
    ys = ys + dy - tilt * 0.05 * (1 - t) + tilt * 0.02 * t
    radii = 0.016 * thickness * (1.1 - 0.65 * t) * np.minimum(1, 0.3 + 6 * np.minimum(t, 1 - t))
    return tube(head.on_face(xs, ys, 0.004), radii, color, sides=8, shine=10, spec=0.02, kind="hair")


def lip_meshes(head, smile, width, skin, open_=0.0):
    w = 0.14 * width
    xs = np.linspace(-w, w, 33)
    u = xs / w
    corner = smile * 0.055 * u ** 2
    fall = np.sqrt(np.clip(1 - u ** 4, 0, 1))
    lip = mix(skin, (0.72, 0.30, 0.34), 0.38)
    drop = open_ * 0.05 * fall
    out = []
    if open_ > 0.05:
        zc = head.z_at(0, MOUTH_Y) - 0.015
        cy = MOUTH_Y + corner.mean() - 0.02 * open_
        out.append(ellipsoid((0, cy, zc), (w * 0.85, 0.01 + 0.035 * open_, 0.02),
                             (0.22, 0.05, 0.06), detail=20))
        out.append(ellipsoid((0, MOUTH_Y + corner.mean() + 0.012, zc + 0.008),
                             (w * 0.62, 0.011, 0.012), (0.96, 0.95, 0.92), detail=16))
    out.append(tube(head.on_face(xs, MOUTH_Y + corner, 0.002), 0.005 + 0.003 * fall,
                    (0.25, 0.10, 0.10), sides=6))
    out.append(tube(head.on_face(xs, MOUTH_Y + corner + 0.016 * fall, 0.0),
                    0.003 + 0.016 * fall, lip, sides=10, shine=35, spec=0.2, kind="glossy"))
    out.append(tube(head.on_face(xs, MOUTH_Y + corner - 0.018 * fall - drop, 0.0),
                    0.003 + 0.022 * fall, lip, sides=10, shine=35, spec=0.2, kind="glossy"))
    return out


def ear_meshes(side, skin):
    return [ellipsoid((side * 0.485, -0.09, -0.02), (0.05, 0.11, 0.075), skin,
                      rot=rot_y(-side * 0.35), detail=18, shine=25, spec=0.1, kind="skin")]


def earring_meshes(style):
    if style == "none":
        return []
    out = []
    for s in (-1, 1):
        lobe = (s * 0.505, -0.19, -0.02)
        if style == "studs":
            out.append(ellipsoid(lobe, (0.03, 0.03, 0.03), GOLD, detail=12, shine=80, spec=0.6,
                                 kind="glossy"))
        else:
            a = np.linspace(0, 2 * math.pi, 24)
            ring = np.stack([np.full(24, s * 0.515), -0.27 + 0.075 * np.cos(a),
                             -0.02 + 0.075 * np.sin(a)], -1)
            out.append(tube(ring, np.full(24, 0.008), GOLD, sides=6, cap_ends=False,
                            shine=80, spec=0.6, kind="glossy"))
    return out


def glasses_meshes(head, style):
    frame = (0.10, 0.10, 0.12)
    glossy = dict(shine=60, spec=0.3, kind="glossy")
    zg = max(head.z_at(s * (EYE_X + dx), EYE_Y + dy) for s in (-1, 1)
             for dx in (-0.14, 0, 0.14) for dy in (-0.1, 0, 0.1)) + 0.04
    ang = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    ang = np.append(ang, ang[0])
    if style == "round":
        rx = ry = 0.125
        ring = np.stack([rx * np.cos(ang), ry * np.sin(ang)], -1)
    else:
        rx, ry = 0.14, 0.10
        ring = np.stack([rx * np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** 0.5,
                         ry * np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** 0.5], -1)
    out = []
    for s in (-1, 1):
        pts = np.column_stack([s * EYE_X + ring[:, 0], EYE_Y + 0.01 + ring[:, 1],
                               np.full(len(ring), zg)])
        out.append(tube(pts, np.full(len(pts), 0.011), frame, sides=8, cap_ends=False, **glossy))
        if style == "sun":
            out.append(ellipsoid((s * EYE_X, EYE_Y + 0.01, zg), (rx - 0.005, ry - 0.005, 0.006),
                                 (0.05, 0.06, 0.09), detail=24, shine=90, spec=0.7, kind="glossy"))
        temple = catmull([(s * (EYE_X + rx), EYE_Y + 0.01, zg), (s * 0.43, EYE_Y + 0.02, 0.35),
                          (s * 0.53, EYE_Y, 0.10), (s * 0.54, 0.0, -0.03)], 12)
        out.append(tube(temple, np.full(12, 0.010), frame, sides=8, **glossy))
    bridge = catmull([(-(EYE_X - rx), EYE_Y + 0.04, zg), (0, EYE_Y + 0.055, zg + 0.015),
                      (EYE_X - rx, EYE_Y + 0.04, zg)], 10)
    out.append(tube(bridge, np.full(10, 0.010), frame, sides=8, **glossy))
    return out
