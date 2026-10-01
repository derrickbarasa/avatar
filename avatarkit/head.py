"""The sculpted head and its face details (eyes, brows, lips, glasses, earrings)."""
import functools
import math

import numpy as np

from .mathutil import catmull, gauss, mix, rot_y, smoothstep, unit
from .mesh import Mesh, cap, ellipsoid, grid_faces, merge, tube

MOUTH_Y = -0.36
EYE_X, EYE_Y = 0.20, 0.03
DARK = (0.10, 0.07, 0.07)
GOLD = (0.95, 0.78, 0.30)


HEAD_Y, HEAD_H = 0.02, 0.66          # the skull spans y = HEAD_Y +- HEAD_H (crown to chin)
SKULL_POWER = 2.5                     # cross-section exponent: 2 is an ellipse, higher is squarer (flatter sides)
# How each part of the skull departs from a plain ellipsoid, by height t (-1 chin .. +1 crown):
# t, width, front depth, back depth (multipliers of the ellipsoid's radii 0.50, 0.56, 0.56)
_SKULL_KEYS = np.array([
    (-1.00, 0.78, 1.20, 0.40),
    (-0.90, 0.90, 1.28, 0.50),     # chin: a firm, slightly forward jaw tip; the nape tucks in
    (-0.75, 0.97, 1.24, 0.68),
    (-0.55, 1.01, 1.14, 0.84),     # jaw line
    (-0.30, 1.00, 1.07, 0.98),
    (0.00, 1.00, 1.00, 1.03),      # ears / temples: the widest part
    (0.35, 0.99, 0.95, 1.04),
    (0.70, 0.93, 0.92, 1.00),      # forehead slopes back, the cranium stays full behind
    (1.00, 0.90, 0.90, 0.95)])


def skull_radii(y, jaw=0.0):
    """Half-width, front depth and back depth of the skull at height y (arrays), for a jaw value."""
    t = np.clip((np.asarray(y, float) - HEAD_Y) / HEAD_H, -1.0, 1.0)
    mod = np.stack([np.interp(t, _SKULL_KEYS[:, 0], _SKULL_KEYS[:, c]) for c in (1, 2, 3)], -1)
    round_off = np.sqrt(np.clip(1.0 - t * t, 0.0, 1.0))[..., None]
    low = np.clip(-t, 0, 1)[..., None] ** 1.6
    mod = mod * np.concatenate([1 - jaw * low, 1 - 0.5 * jaw * low, np.ones_like(low)], -1)
    return mod * round_off * np.array([0.50, 0.56, 0.56])


def skull_ring(phi, y, extra=0.0):
    """Points on the skull's horizontal section at height y (angle phi, 0 = straight ahead), pushed out by `extra`."""
    w, front, back = (a[..., 0] for a in np.split(skull_radii(y), 3, axis=-1))
    sx, cz = np.sin(phi), np.cos(phi)
    e = 2.0 / SKULL_POWER
    return np.stack([(w + extra) * np.sign(sx) * np.abs(sx) ** e, np.broadcast_to(np.asarray(y, float), sx.shape),
                     (np.where(cz > 0, front, back) + extra) * np.sign(cz) * np.abs(cz) ** e], -1)


class Head:
    """Sculpted head: a skull built from cross-sections (flat-sided cranium, jaw, chin) plus gaussian features."""
    NU, NV = 128, 96

    def __init__(self, skin, jaw, nose):
        skin = np.asarray(skin, float)
        lat = np.linspace(-math.pi / 2, math.pi / 2, self.NV)[:, None]
        lon = np.linspace(0, 2 * math.pi, self.NU, endpoint=False)[None, :]
        dz = np.cos(lat) * np.cos(lon)                              # for masks: how far round the front we are
        y = (HEAD_Y + HEAD_H * np.sin(lat)) * np.ones_like(lon)
        w, front, back = (a[..., 0] for a in np.split(skull_radii(y, jaw), 3, axis=-1))
        sx, cz = np.sin(lon) * np.ones_like(y), np.cos(lon) * np.ones_like(y)
        e = 2.0 / SKULL_POWER
        x = w * np.sign(sx) * np.abs(sx) ** e
        z = np.where(cz > 0, front, back) * np.sign(cz) * np.abs(cz) ** e

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
        self.mesh.jaw_follow = True          # the chin drops when the mouth opens
        front_mask = self.Z > 0.05
        self._fx, self._fy, self._fz = self.X[front_mask], self.Y[front_mask], self.Z[front_mask]

    def z_at(self, x, y):
        """Height of the face surface at (x, y)."""
        d2 = (self._fx - x) ** 2 + (self._fy - y) ** 2
        idx = np.argpartition(d2, 4)[:4]
        w = 1.0 / (d2[idx] + 1e-6)
        return float((self._fz[idx] * w).sum() / w.sum())

    def z_at_many(self, xs, ys):
        """Vectorised `z_at` for arrays of x and y (same shape)."""
        xs, ys = np.asarray(xs, np.float32).ravel(), np.asarray(ys, np.float32).ravel()
        out = np.empty(len(xs))
        fx, fy, fz = self._fx.astype(np.float32), self._fy.astype(np.float32), self._fz
        for a in range(0, len(xs), 400):
            d2 = (fx[None] - xs[a:a + 400, None]) ** 2 + (fy[None] - ys[a:a + 400, None]) ** 2
            idx = np.argpartition(d2, 4, axis=1)[:, :4]
            w = 1.0 / (np.take_along_axis(d2, idx, 1) + 1e-6)
            out[a:a + 400] = (fz[idx] * w).sum(1) / w.sum(1)
        return out

    def shell(self, margin, thickness, color, edge=0.05, opacity=1.0, **kw):
        """Offset copy of the head where margin > 0.

        The border fades out (alpha) and sinks into the scalp, so the edge
        is soft instead of following the mesh grid.
        """
        m = margin.ravel()
        t = np.broadcast_to(thickness, margin.shape).ravel() * smoothstep(0, edge, m) + 0.007
        verts = self.P + self.mesh.n * t[:, None]
        faces = self.faces[(m > 0)[self.faces].any(1)]
        alpha = smoothstep(0, 0.035, m) * opacity
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
    ball.anim = ("gaze", tuple(c))
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
    # A few lashes flicking out and up, longest at the outer corner.
    flicks = []
    for k in np.linspace(len(edge) * 0.10, len(edge) * 0.90, 7).astype(int):
        out = edge[k] / np.linalg.norm(edge[k])
        outer = (edge[k, 0] * side + 1.0) / 2                  # 0 at the nose side .. 1 at the temple
        length = r * (0.22 + 0.30 * outer)
        base = c + edge[k] * r * 1.113
        tip = base + (out * 0.75 + direction * 0.45) * length
        mid = base + (out * 0.9 + direction * 0.2) * length * 0.5
        flicks.append(tube(np.array([base, mid, tip]), np.array([0.0028, 0.0020, 0.0008]), DARK, sides=4))
    lash = merge([lash] + flicks)
    for m in (lid, lash):
        m.anim = ("blink", tuple(c))
    return [ball, lid, lash]


def brow_mesh(head, side, thickness, color, dy=0.0, tilt=0.0, scale=1.0):
    t = np.linspace(0, 1, 14)
    xs = side * (0.075 + 0.20 * t)
    ys = 0.185 + 0.045 * np.sin(t * math.pi * 0.85) - 0.02 * t
    ys = ys + dy - tilt * 0.05 * (1 - t) + tilt * 0.02 * t
    radii = scale * 0.016 * thickness * (1.1 - 0.65 * t) * np.minimum(1, 0.3 + 6 * np.minimum(t, 1 - t))
    return tube(head.on_face(xs, ys, 0.004), radii, color, sides=8, shine=10, spec=0.02, kind="hair")


class Mouth:
    """Lips that can be re-shaped every frame: open, wide (+) / pucker (-) and pressed.

    Shapes are quantised and cached, so speech re-uses a few dozen small meshes.
    """
    GX = np.linspace(-0.26, 0.26, 53)
    GY = np.linspace(MOUTH_Y - 0.17, MOUTH_Y + 0.10, 28)

    def __init__(self, head, smile, width, skin):
        self.smile, self.w0, self.skin = smile, 0.14 * width, tuple(skin)
        self.lip = mix(skin, (0.72, 0.30, 0.34), 0.38)
        if not hasattr(head, "_mouth_z"):
            gx, gy = np.meshgrid(self.GX, self.GY)
            head._mouth_z = head.z_at_many(gx, gy).reshape(gx.shape)
        self.zg = head._mouth_z
        self._cache = {}

    def z(self, x, y):
        """Face surface height at (x, y) by bilinear lookup in the precomputed grid."""
        fx = np.clip((np.asarray(x, float) - self.GX[0]) / (self.GX[1] - self.GX[0]), 0, len(self.GX) - 1.001)
        fy = np.clip((np.asarray(y, float) - self.GY[0]) / (self.GY[1] - self.GY[0]), 0, len(self.GY) - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        g = self.zg
        return (g[iy, ix] * (1 - tx) * (1 - ty) + g[iy, ix + 1] * tx * (1 - ty)
                + g[iy + 1, ix] * (1 - tx) * ty + g[iy + 1, ix + 1] * tx * ty)

    def meshes(self, open_=0.0, wide=0.0, press=0.0, smile=0.0):
        """Lips for a mouth shape; `smile` is added to the resting smile (negative = frown)."""
        key = (int(round(min(max(open_, 0.0), 1.0) * 24)), int(round(min(max(wide, -1.0), 1.0) * 12)),
               bool(press > 0.5), int(round(min(max(smile, -1.0), 1.0) * 10)))
        if key not in self._cache:
            self._cache[key] = self._build(key[0] / 24, key[1] / 12, key[2], key[3] / 10)
        return self._cache[key]

    def fixed_meshes(self, open_=0.0, wide=0.0, press=0.0, smile=0.0):
        """Lips for a mouth shape with the same parts and vertex layout whatever the shape (parts a shape
        would not draw are shrunk to a point). Morph targets need this; `meshes` does not keep one."""
        return self._build(min(max(open_, 0.0), 1.0), min(max(wide, -1.0), 1.0), press > 0.5, smile, fixed=True)

    def _build(self, o, wide, press, smile=0.0, fixed=False):
        w = self.w0 * (1 + 0.28 * wide)
        xs = np.linspace(-w, w, 33)
        u = xs / w
        corner = (self.smile + smile) * 0.055 * u ** 2 + 0.010 * wide * u ** 2
        fall = np.sqrt(np.clip(1 - u ** 4, 0, 1))
        r = max(0.0, -wide)
        thick = (1 + 0.55 * r) * (0.7 if press else 1.0)
        drop = o * 0.105 * fall ** 0.8
        rise = o * 0.016 * fall
        base = MOUTH_Y + corner

        def path(y, lift=0.0):
            return np.stack([xs, y, self.z(xs, y) + lift], -1)

        out = []
        if fixed:
            tiny = lambda shown: 1.0 if shown else 1e-3      # a hidden part collapses instead of vanishing
            zc = float(self.z(0.0, MOUTH_Y)) - 0.015
            span = float(drop.max())
            mean = float(corner.mean())
            cy = MOUTH_Y + mean - 0.45 * span
            k = tiny(o > 0.05)
            out.append(tube(path(base, 0.002), (0.005 + 0.003 * fall) * tiny(o <= 0.05), (0.25, 0.10, 0.10), sides=6))
            out.append(ellipsoid((0, cy, zc), (w * 0.88 * k, (0.008 + 0.56 * span) * k, 0.022 * k),
                                 (0.22, 0.05, 0.06), detail=20))
            out.append(ellipsoid((0, MOUTH_Y + mean + 0.4 * float(rise.max()) - 0.004, zc + 0.008),
                                 (w * 0.62 * (1 - 0.3 * r) * k, 0.011 * k, 0.012 * k), (0.96, 0.95, 0.92), detail=16))
            k = tiny(o > 0.45)
            out.append(ellipsoid((0, cy - 0.30 * span, zc + 0.002), (w * 0.40 * k, (0.010 + 0.014 * o) * k, 0.014 * k),
                                 (0.75, 0.32, 0.36), detail=14))
            k = tiny(o > 0.6)
            out.append(ellipsoid((0, cy - 0.85 * span, zc + 0.006), (w * 0.55 * (1 - 0.3 * r) * k, 0.009 * k, 0.011 * k),
                                 (0.94, 0.93, 0.90), detail=14))
        elif o > 0.05:
            zc = float(self.z(0.0, MOUTH_Y)) - 0.015
            span = float(drop.max())
            cy = MOUTH_Y + float(corner.mean()) - 0.45 * span
            out.append(ellipsoid((0, cy, zc), (w * 0.88, 0.008 + 0.56 * span, 0.022), (0.22, 0.05, 0.06), detail=20))
            out.append(ellipsoid((0, MOUTH_Y + float(corner.mean()) + 0.4 * float(rise.max()) - 0.004, zc + 0.008),
                                 (w * 0.62 * (1 - 0.3 * r), 0.011, 0.012), (0.96, 0.95, 0.92), detail=16))
            if o > 0.45:
                out.append(ellipsoid((0, cy - 0.30 * span, zc + 0.002), (w * 0.40, 0.010 + 0.014 * o, 0.014),
                                     (0.75, 0.32, 0.36), detail=14))
            if o > 0.6:
                out.append(ellipsoid((0, cy - 0.85 * span, zc + 0.006), (w * 0.55 * (1 - 0.3 * r), 0.009, 0.011),
                                     (0.94, 0.93, 0.90), detail=14))
        else:
            out.append(tube(path(base, 0.002), 0.005 + 0.003 * fall, (0.25, 0.10, 0.10), sides=6))
        out.append(tube(path(base + 0.016 * fall * thick + rise), (0.003 + 0.016 * fall) * thick, self.lip,
                        sides=10, shine=35, spec=0.2, kind="glossy"))
        out.append(tube(path(base - 0.018 * fall * thick - drop), (0.003 + 0.022 * fall) * thick, self.lip,
                        sides=10, shine=35, spec=0.2, kind="glossy"))
        return out


def lip_meshes(head, smile, width, skin, open_=0.0):
    """Static lips at a given openness (kept for tests and simple callers)."""
    return Mouth(head, smile, width, skin).meshes(open_)


def ear_meshes(side, skin):
    rot = rot_y(-side * 0.35)
    inner = tuple(c * 0.72 for c in skin)
    return [ellipsoid((side * 0.485, -0.09, -0.07), (0.05, 0.11, 0.075), skin,
                      rot=rot, detail=18, shine=25, spec=0.1, kind="skin"),
            ellipsoid((side * 0.522, -0.095, -0.055), (0.014, 0.068, 0.042), inner,
                      rot=rot, detail=14, shine=15, spec=0.05, kind="skin", thin=True)]


def nostril_meshes(head, skin):
    """Two small dark openings under the nose tip."""
    dark = tuple(c * 0.40 for c in skin)
    out = []
    for s in (-1, 1):
        x, y = s * 0.046, -0.243
        out.append(ellipsoid((x, y, head.z_at(x, y) - 0.004), (0.017, 0.011, 0.012), dark, detail=10,
                             kind="skin", thin=True))
    return out


def earring_meshes(style):
    if style == "none":
        return []
    out = []
    for s in (-1, 1):
        lobe = (s * 0.505, -0.19, -0.07)
        if style == "studs":
            out.append(ellipsoid(lobe, (0.03, 0.03, 0.03), GOLD, detail=12, shine=80, spec=0.6,
                                 kind="glossy"))
        else:
            a = np.linspace(0, 2 * math.pi, 24)
            ring = np.stack([np.full(24, s * 0.515), -0.27 + 0.075 * np.cos(a),
                             -0.07 + 0.075 * np.sin(a)], -1)
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
    elif style == "aviator":
        rx, ry = 0.14, 0.12
        sn = np.sin(ang)
        ring = np.stack([rx * np.cos(ang) * (1 - 0.12 * np.clip(sn, 0, 1)), np.where(sn > 0, 0.095, 0.13) * sn], -1)
        frame = (0.85, 0.70, 0.30)
    else:
        rx, ry = 0.14, 0.10
        ring = np.stack([rx * np.sign(np.cos(ang)) * np.abs(np.cos(ang)) ** 0.5,
                         ry * np.sign(np.sin(ang)) * np.abs(np.sin(ang)) ** 0.5], -1)
    out = []
    for s in (-1, 1):
        ring_s = ring.copy()
        if style == "cat":                      # outer corners sweep up
            outer = np.clip(s * ring[:, 0] / rx, 0, 1)
            ring_s[:, 1] += 0.055 * outer ** 2
            frame = (0.55, 0.08, 0.22)
        pts = np.column_stack([s * EYE_X + ring_s[:, 0], EYE_Y + 0.01 + ring_s[:, 1],
                               np.full(len(ring), zg)])
        out.append(tube(pts, np.full(len(pts), 0.011), frame, sides=8, cap_ends=False, **glossy))
        if style in ("sun", "aviator"):
            tint = (0.05, 0.06, 0.09) if style == "sun" else (0.10, 0.14, 0.12)
            out.append(ellipsoid((s * EYE_X, EYE_Y + 0.01, zg), (rx - 0.005, ry - 0.005, 0.006),
                                 tint, detail=24, shine=90, spec=0.7, kind="glossy"))
        temple = catmull([(s * (EYE_X + rx), EYE_Y + 0.01, zg), (s * 0.48, EYE_Y + 0.02, 0.35),
                          (s * 0.555, EYE_Y, 0.10), (s * 0.56, 0.0, -0.03)], 12)
        out.append(tube(temple, np.full(12, 0.010), frame, sides=8, **glossy))
    bridge = catmull([(-(EYE_X - rx), EYE_Y + 0.04, zg), (0, EYE_Y + 0.055, zg + 0.015),
                      (EYE_X - rx, EYE_Y + 0.04, zg)], 10)
    out.append(tube(bridge, np.full(10, 0.010), frame, sides=8, **glossy))
    return out
