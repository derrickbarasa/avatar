"""Real hair: thousands of thin, tapered strands grown from the scalp.

Each strand starts at a random point on the hair region of the head and is
integrated step by step: it leaves the scalp along a style-specific flow,
bends towards gravity, is pushed out of the head / neck / shoulders, and is
grouped into locks that share a wave phase and drift. Ponytail and bun use
explicit guide paths. The strands are merged into one mesh whose vertices
carry the strand tangent (for anisotropic hair highlights) and a per-strand
colour variation.
"""
import math
import zlib

import numpy as np

from .head import EYE_X, EYE_Y, MOUTH_Y
from .mathutil import catmull, smoothstep, unit
from .mesh import Mesh, grid_faces

DOWN = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])
HEAD_C = np.array([0.0, 0.02, 0.0])
HEAD_R = np.array([0.535, 0.665, 0.60])
CROWN = np.array([0.0, 0.45, -0.15])
SIDES = 3


# ---------------------------------------------------------------------------
# Sampling and physics
# ---------------------------------------------------------------------------
def sample_roots(head, margin, n, rng, keep=None, min_margin=0.02):
    """n random points (position, normal) on the hair region, uniform by area."""
    m = margin.ravel()
    faces = head.faces[(m > min_margin)[head.faces].all(1)]
    tri = head.P[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    want = int(n * (1.0 if keep is None else 2.2))
    idx = rng.choice(len(faces), want, p=area / area.sum())
    u = rng.random((want, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    w = np.stack([1 - u.sum(1), u[:, 0], u[:, 1]], 1)[..., None]
    f = faces[idx]
    pos = (head.P[f] * w).sum(1)
    nrm = unit((head.mesh.n[f] * w).sum(1) + 1e-9)
    if keep is not None:
        ok = keep(pos)
        pos, nrm = pos[ok][:n], nrm[ok][:n]
    return pos, nrm


def make_locks(root, scale=11):
    """Group nearby roots into locks (returns (unique ids, inverse index))."""
    cell = np.floor(root * scale).astype(np.int64)
    key = cell[:, 0] * 73856093 + cell[:, 1] * 19349663 + cell[:, 2] * 83492791
    return np.unique(key, return_inverse=True)


def collide(p):
    """Push points out of the head, neck and shoulders (in place)."""
    q = (p - HEAD_C) / HEAD_R
    e = (q * q).sum(1)
    inside = e < 1.0
    if inside.any():
        p[inside] = HEAD_C + (p[inside] - HEAD_C) / np.sqrt(np.maximum(e[inside], 1e-9))[:, None]
    y = p[:, 1]
    r = np.hypot(p[:, 0], p[:, 2])
    neck = (y < -0.45) & (y > -0.95) & (r < 0.31)
    if neck.any():
        f = 0.31 / np.maximum(r[neck], 1e-6)
        p[neck, 0] *= f
        p[neck, 2] *= f
    yy = np.clip(-0.72 - y, 0, None)
    a = np.minimum(0.30 + yy * 2.0, 0.93) + 0.03
    b = np.minimum(0.27 + yy * 0.35, 0.42) + 0.03
    k = (p[:, 0] / a) ** 2 + (p[:, 2] / b) ** 2
    body = (y < -0.70) & (k < 1.0)
    if body.any():
        s = np.sqrt(np.maximum(k[body], 1e-6))
        p[body, 0] /= s
        p[body, 2] /= s
    return p


def grow(root, nrm, d0, length, steps, rng, locks, gravity=0.3, wave=0.0, waves=2.0,
         tuck=0.0, jitter=0.15, floor=0.09, lift=None, collider=None):
    """Integrate strands. Returns paths (N, steps + 1, 3)."""
    n = len(root)
    uniq, inv = locks
    lock_dir = rng.normal(size=(len(uniq), 3))[inv]
    phase = (rng.random(len(uniq)) * 2 * math.pi)[inv]
    d = unit(d0 + lock_dir * jitter + rng.normal(size=(n, 3)) * jitter * 0.4)
    if lift is not None:   # keep hair lying on the head instead of springing away from it
        d = unit(d - nrm * np.clip((d * nrm).sum(1, keepdims=True) - lift, 0, None))
    p = root + nrm * 0.012
    pts = np.empty((steps + 1, n, 3))
    pts[0] = p
    step = (np.asarray(length, float) / steps)[:, None]
    for i in range(1, steps + 1):
        t = i / steps
        a = min(1.0, gravity * 1.6 * t ** 1.6 + floor)   # stiff at the root, drooping later
        d = unit(d * (1 - a) + DOWN * a)
        if tuck:
            rad = np.stack([p[:, 0], np.zeros(n), p[:, 2]], 1)
            d = unit(d - tuck * t * t * unit(rad + 1e-6))
        if wave:
            side = np.cross(d, UP)
            ns = np.linalg.norm(side, axis=1, keepdims=True)
            side = np.where(ns > 1e-4, side / np.maximum(ns, 1e-9), np.array([1.0, 0.0, 0.0]))
            d = unit(d + wave * np.cos(2 * math.pi * waves * t + phase)[:, None] * side)
        p = (collider or collide)(p + d * step)
        pts[i] = p
    return pts.transpose(1, 0, 2)


def smooth(paths, factor):
    """Catmull-Rom upsample every strand so long strands bend smoothly, not in corners."""
    if factor <= 1:
        return paths
    n, s1, _ = paths.shape
    t = np.linspace(0, s1 - 1, (s1 - 1) * factor + 1)
    i = np.minimum(t.astype(int), s1 - 2)
    u = (t - i)[None, :, None]
    p0 = paths[:, np.maximum(i - 1, 0)]
    p1, p2 = paths[:, i], paths[:, i + 1]
    p3 = paths[:, np.minimum(i + 2, s1 - 1)]
    return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


def curl(pts, radius, turns, rng, locks):
    """Add a helical offset that fades in from the root."""
    n, s1, _ = pts.shape
    t = np.linspace(0, 1, s1)
    tan = unit(np.gradient(pts, axis=1) + 1e-9)
    ref = np.where((np.abs(tan[..., 1]) < 0.9)[..., None], UP, np.array([1.0, 0.0, 0.0]))
    nrm = unit(np.cross(tan, ref))
    bin_ = np.cross(tan, nrm)
    ph = (rng.random(len(locks[0])) * 2 * math.pi)[locks[1]][:, None]
    ang = 2 * math.pi * turns * t[None, :] + ph
    amp = radius * smoothstep(0.0, 0.35, t)[None, :, None]
    return pts + amp * (np.cos(ang)[..., None] * nrm + np.sin(ang)[..., None] * bin_)


# ---------------------------------------------------------------------------
# Mesh from paths
# ---------------------------------------------------------------------------
def strand_mesh(paths, base_color, radius, rng, locks, flyaway=0.02, part="hair"):
    """Merge strand paths (N, S1, 3) into a single tube mesh."""
    n, s1, _ = paths.shape
    t = np.linspace(0, 1, s1)
    paths = paths.copy()
    fly = rng.random(n) < flyaway
    if fly.any():
        paths[fly] += rng.normal(size=(fly.sum(), 1, 3)) * 0.035 * t[None, :, None] ** 1.5
    r0 = radius * (0.85 + 0.35 * rng.random(n))
    r0[fly] *= 0.55
    # thick through most of the length, then a fine point (blunt cut ends looked coarse)
    taper = (1 - 0.30 * t) * (1 - 0.92 * smoothstep(0.55, 1.0, t) ** 1.3)
    radii = r0[:, None] * np.maximum(taper, 0.06)[None, :]
    # every lock lets its tips drift a little its own way, so the fall isn't a perfect curtain
    if part == "hair":
        lock_kick = rng.normal(size=(len(locks[0]), 3)) * np.array([1.0, 0.25, 1.0])
        paths += lock_kick[locks[1]][:, None, :] * (0.030 * t ** 3)[None, :, None]

    tan = unit(np.gradient(paths, axis=1) + 1e-9)
    ref = np.where((np.abs(tan[..., 1]) < 0.9)[..., None], UP, np.array([1.0, 0.0, 0.0]))
    nrm = unit(np.cross(tan, ref))
    bin_ = np.cross(tan, nrm)
    ang = np.linspace(0, 2 * math.pi, SIDES, endpoint=False)
    ring = paths[:, :, None, :] + radii[:, :, None, None] * (
        np.cos(ang)[None, None, :, None] * nrm[:, :, None, :]
        + np.sin(ang)[None, None, :, None] * bin_[:, :, None, :])
    verts = ring.reshape(-1, 3)

    base_faces = grid_faces(SIDES, s1)
    faces = (base_faces[None] + (np.arange(n) * s1 * SIDES)[:, None, None]).reshape(-1, 3)
    refs = np.repeat(paths.reshape(-1, 3), SIDES, axis=0)

    per_strand = 0.82 + 0.32 * rng.random(n)
    per_lock = (0.94 + 0.12 * rng.random(len(locks[0])))[locks[1]]
    shade = 0.70 + 0.42 * t                      # darker at the root, lighter at the tips
    col = (np.asarray(base_color)[None, None, :] * (per_strand * per_lock)[:, None, None]
           * shade[None, :, None])
    colors = np.repeat(np.clip(col, 0, 1), SIDES, axis=1).reshape(-1, 3)
    tangents = np.repeat(tan.reshape(-1, 3), SIDES, axis=0)

    mesh = Mesh(verts, faces, base_color, ref=refs, colors=colors, shine=40, spec=0.25,
                kind="hair")
    mesh.tangents = np.ascontiguousarray(tangents, np.float32)
    # (radius, position along the strand): lets the shader thicken strands at lower detail and sway them
    mesh.strand_aux = np.ascontiguousarray(np.stack([np.repeat(radii.reshape(-1), SIDES),
                                                     np.repeat(np.broadcast_to(t, (n, s1)).reshape(-1), SIDES)], 1),
                                           np.float32)
    mesh.strand = True
    mesh.n_strands, mesh.faces_per_strand = n, len(base_faces)   # faces are stored strand by strand
    mesh.part = part          # "hair", "facial" or "brows"
    return mesh


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _flow(root, nrm):
    v = root - CROWN
    tang = v - nrm * (v * nrm).sum(1, keepdims=True)
    return unit(tang + 1e-9) * 0.7 + nrm * 0.3


def _surface_flow(root, nrm):
    """Direction hair lies on the scalp: down the surface, spreading outward at the crown."""
    horiz = unit(np.stack([root[:, 0], np.zeros(len(root)), root[:, 2]], 1) + 1e-9)
    want = DOWN + 0.5 * horiz
    return unit(want - nrm * (want * nrm).sum(1, keepdims=True) + 1e-9)


def _front(root):
    return smoothstep(0.15, 0.5, root[:, 2] / 0.56)


def _forward_down(root, lean=0.0):
    return unit(np.stack([root[:, 0] * 0.6, np.full(len(root), -0.3 + lean), np.full(len(root), 0.9)], 1))


def _sweep(root):
    sgn = np.where(root[:, 0] >= 0, 1.0, -1.0)
    return unit(np.stack([sgn * 1.0, np.full(len(root), 0.1), np.full(len(root), -0.35)], 1))


def _hem_length(root, y_end, rng):
    return ((root[:, 1] - y_end) * 1.2 + 0.3) * (0.94 + 0.12 * rng.random(len(root)))


def _group_short(head, margin, rng, steps, n=2000, keep=None):
    root, nrm = sample_roots(head, margin, n, rng, keep)
    d0 = unit(_flow(root, nrm) * (1 - 0.6 * _front(root))[:, None]
              + _forward_down(root) * (0.6 * _front(root))[:, None])
    locks = make_locks(root)
    length = 0.17 * (0.85 + 0.3 * rng.random(len(root)))
    paths = grow(root, nrm, d0, length, steps, rng, locks, gravity=0.6, jitter=0.25, lift=0.45)
    return paths, locks, 0.0095


def _group_curly(head, margin, rng, steps):
    root, nrm = sample_roots(head, margin, 1700, rng)
    d0 = unit(_flow(root, nrm) * 0.4 + nrm * 0.6)
    locks = make_locks(root)
    length = 0.30 * (0.85 + 0.3 * rng.random(len(root)))
    paths = grow(root, nrm, d0, length, steps, rng, locks, gravity=0.3, jitter=0.5)
    return curl(paths, 0.03, 4, rng, locks), locks, 0.011


def _group_long(head, margin, rng, steps, y_end, n, tuck, wave, keep=None, sweep=True):
    root, nrm = sample_roots(head, margin, n, rng, keep)
    flow = _surface_flow(root, nrm)
    f = smoothstep(-0.25, 0.15, root[:, 2])[:, None]   # the whole front half parts and sweeps sideways
    d0 = unit(flow * (1 - f) + _sweep(root) * f) if sweep else flow
    locks = make_locks(root, 8)
    length = _hem_length(root, y_end, rng)
    paths = grow(root, nrm, d0, length, steps, rng, locks, gravity=0.6, wave=wave, tuck=tuck,
                 jitter=0.25, floor=0.22, lift=-0.05)
    return smooth(paths, 3), locks, 0.0135


def _group_fringe(head, margin, rng, steps):
    keep = lambda p: (p[:, 2] > 0.25) & (p[:, 1] > 0.30)
    root, nrm = sample_roots(head, margin, 800, rng, keep, min_margin=0.05)
    d0 = _forward_down(root, lean=-0.05)
    locks = make_locks(root)
    length = 0.32 * (0.9 + 0.2 * rng.random(len(root)))
    paths = grow(root, nrm, d0, length, steps, rng, locks, gravity=0.5, jitter=0.1, lift=0.3)
    return smooth(paths, 3), locks, 0.011


def _group_quiff(head, margin, rng, steps):
    keep = lambda p: (p[:, 2] > 0.15) & (p[:, 1] > 0.28)
    root, nrm = sample_roots(head, margin, 900, rng, keep, min_margin=0.05)
    d0 = unit(np.stack([root[:, 0] * 0.4, np.full(len(root), 0.9), np.full(len(root), 0.5)], 1))
    locks = make_locks(root)
    length = 0.42 * (0.85 + 0.3 * rng.random(len(root)))
    paths = grow(root, nrm, d0, length, steps, rng, locks, gravity=0.25, jitter=0.12)
    return paths, locks, 0.0105


def _group_mohawk(head, margin, rng, steps):
    keep = lambda p: (np.abs(p[:, 0]) < 0.11) & (p[:, 1] > 0.30)
    root, nrm = sample_roots(head, margin, 1000, rng, keep, min_margin=0.05)
    d0 = unit(np.stack([root[:, 0] * 3.0, np.full(len(root), 0.9), np.full(len(root), -0.1)], 1))
    locks = make_locks(root)
    length = 0.33 * (0.9 + 0.2 * rng.random(len(root)))
    paths = grow(root, nrm, d0, length, steps, rng, locks, gravity=0.08, jitter=0.12)
    return paths, locks, 0.011


def _group_afro(head, margin, rng, steps):
    root, nrm = sample_roots(head, margin, 3600, rng, min_margin=0.0)
    d0 = unit(nrm * 0.9 + rng.normal(size=nrm.shape) * 0.7)
    locks = make_locks(root)
    length = 0.40 * (0.8 + 0.4 * rng.random(len(root)))
    paths = grow(root, nrm, d0, length, steps, rng, locks, gravity=0.05, jitter=0.8)
    return curl(paths, 0.045, 4, rng, locks), locks, 0.015


def _smooth_pick(x):
    return x * x * (3 - 2 * x)


def _group_ponytail(head, margin, rng, steps):
    root, _ = sample_roots(head, margin, 2200, rng)
    n = len(root)
    tie = np.array([0.0, 0.30, -0.52])
    curve = catmull([(0.0, 0.30, -0.52), (0.0, 0.18, -0.74), (0.0, -0.25, -0.80), (0.0, -0.78, -0.70)], 80)
    off = rng.normal(size=(n, 2)) * 0.032
    u_end = 0.86 + 0.14 * rng.random(n)
    seg = 0.28
    pts = np.empty((n, steps + 1, 3))
    for i in range(steps + 1):
        t = i / steps
        if t < seg:
            a = _smooth_pick(t / seg)
            tgt = tie + np.stack([off[:, 0], off[:, 1], np.zeros(n)], 1) * 0.6
            pts[:, i] = root * (1 - a) + tgt * a
        else:
            u = (t - seg) / (1 - seg) * u_end
            c = np.stack([np.interp(u, np.linspace(0, 1, len(curve)), curve[:, k]) for k in range(3)], 1)
            prof = (0.9 - 0.5 * u + 0.35 * np.sin(math.pi * u))[:, None]
            pts[:, i] = c + np.stack([off[:, 0], np.zeros(n), off[:, 1]], 1) * prof * 2.2
        pts[:, i] = collide(pts[:, i])
    return pts, make_locks(root), 0.011


def _bundle_paths(root, steps, tie, curve_pts, offsets, side=None, seg=0.28):
    """Strands that run from their roots to a tie point, then follow a guide curve in a bundle.

    `offsets(u, n)` returns the (n, 3) cross-section offset of each strand at curve position u.
    """
    n = len(root)
    curve = catmull(curve_pts, 100)
    grid = np.linspace(0, 1, len(curve))
    pts = np.empty((n, steps + 1, 3))
    sign = np.ones(n) if side is None else side
    u_end = 0.86 + 0.14 * np.random.default_rng(7).random(n)
    for i in range(steps + 1):
        t = i / steps
        if t < seg:
            a = _smooth_pick(t / seg)
            target = tie * np.stack([sign, np.ones(n), np.ones(n)], 1) + offsets(0.0, n) * 0.5
            pts[:, i] = root * (1 - a) + target * a
        else:
            u = (t - seg) / (1 - seg) * u_end
            c = np.stack([np.interp(u, grid, curve[:, k]) for k in range(3)], 1)
            c[:, 0] *= sign
            pts[:, i] = c + offsets(u, n)
        pts[:, i] = collide(pts[:, i])
    return pts


def _group_pigtails(head, margin, rng, steps):
    root, _ = sample_roots(head, margin, 2600, rng)
    n = len(root)
    side = np.where(root[:, 0] >= 0, 1.0, -1.0)
    off = rng.normal(size=(n, 2)) * 0.028
    prof = lambda u: (0.9 - 0.45 * u + 0.3 * np.sin(math.pi * u))[:, None] if np.ndim(u) else (0.9 - 0.45 * u + 0.3 * math.sin(math.pi * u))

    def offsets(u, count):
        p = prof(np.full(count, u)) if np.ndim(u) == 0 else prof(u)
        return np.stack([off[:, 0], np.zeros(count), off[:, 1]], 1) * p * 1.9

    paths = _bundle_paths(root, steps, np.array([0.44, 0.10, -0.28]),
                          [(0.44, 0.10, -0.28), (0.60, -0.05, -0.32), (0.66, -0.55, -0.28), (0.64, -1.05, -0.20)],
                          offsets, side)
    return paths, make_locks(root), 0.010


def _group_braid(head, margin, rng, steps):
    root, _ = sample_roots(head, margin, 2600, rng)
    n = len(root)
    strand = rng.integers(0, 3, n)                       # three sub-strands woven around each other
    phase = strand * 2 * math.pi / 3
    scatter = rng.normal(size=(n, 2)) * 0.011
    turns = 7.0

    def offsets(u, count):
        u = np.full(count, u) if np.ndim(u) == 0 else u
        ang = phase + 2 * math.pi * turns * u
        radius = 0.052 * (1.0 - 0.45 * u)
        return np.stack([np.cos(ang) * radius + scatter[:, 0], np.zeros(count),
                         np.sin(ang) * radius * 0.8 + scatter[:, 1]], 1)

    paths = _bundle_paths(root, steps, np.array([0.0, 0.26, -0.52]),
                          [(0.0, 0.26, -0.52), (0.0, 0.05, -0.74), (0.0, -0.60, -0.76), (0.0, -1.25, -0.70)],
                          offsets)
    return paths, make_locks(root), 0.0095


def _group_bun(head, margin, rng, steps):
    root, _ = sample_roots(head, margin, 2600, rng)
    n = len(root)
    centre = np.array([0.0, 0.70, -0.22])
    radius = 0.22
    az = np.arctan2(root[:, 2] - centre[2], root[:, 0] - centre[0])
    turns = 1.3 + 0.5 * rng.random(n)
    rad = radius * (1.0 + 0.05 * rng.random(n))
    th0 = 1.7
    seg = 0.35
    pts = np.empty((n, steps + 1, 3))
    for i in range(steps + 1):
        t = i / steps
        if t < seg:
            a = _smooth_pick(t / seg)
            entry = centre + rad[:, None] * np.stack(
                [math.sin(th0) * np.cos(az), np.full(n, math.cos(th0)), math.sin(th0) * np.sin(az)], 1)
            pts[:, i] = root * (1 - a) + entry * a
        else:
            u = (t - seg) / (1 - seg)
            th = th0 - (th0 - 0.25) * u
            ph = az + turns * 2 * math.pi * u
            pts[:, i] = centre + rad[:, None] * np.stack(
                [np.sin(th) * np.cos(ph), np.cos(th) * np.ones(n), np.sin(th) * np.sin(ph)], 1)
        pts[:, i] = collide(pts[:, i])
    return pts, make_locks(root), 0.010


SWING = {"pigtails": 1.0, "braid": 0.9, "long": 1.0, "bob": 0.8, "bangs": 0.8, "ponytail": 1.0, "curly": 0.3, "afro": 0.15,
         "short": 0.25, "quiff": 0.2, "mohawk": 0.15, "bun": 0.05}


def strand_meshes(head, style, color, margin):
    """Strand mesh(es) for a hairstyle, or [] for styles without strands."""
    if style in ("bald", "buzz"):
        return []
    rng = np.random.default_rng(zlib.crc32(style.encode()))
    groups = []
    if style == "short":
        groups = [_group_short(head, margin, rng, 8)]
    elif style == "curly":
        groups = [_group_curly(head, margin, rng, 18)]
    elif style == "long":
        groups = [_group_long(head, margin, rng, 14, -1.28, 2800, 0.0, 0.10)]
    elif style == "bob":
        groups = [_group_long(head, margin, rng, 11, -0.78, 2400, 0.9, 0.07)]
    elif style == "bangs":
        no_fringe = lambda p: ~((p[:, 2] > 0.25) & (p[:, 1] > 0.30))
        groups = [_group_long(head, margin, rng, 11, -0.78, 2200, 0.9, 0.07, keep=no_fringe),
                  _group_fringe(head, margin, rng, 11)]
    elif style == "ponytail":
        groups = [_group_ponytail(head, margin, rng, 18)]
    elif style == "bun":
        groups = [_group_bun(head, margin, rng, 14)]
    elif style == "quiff":
        rest = lambda p: ~((p[:, 2] > 0.15) & (p[:, 1] > 0.28))
        groups = [_group_short(head, margin, rng, 10, 1500, keep=rest), _group_quiff(head, margin, rng, 10)]
    elif style == "mohawk":
        groups = [_group_mohawk(head, margin, rng, 10)]
    elif style == "afro":
        groups = [_group_afro(head, margin, rng, 16)]
    elif style == "pigtails":
        groups = [_group_pigtails(head, margin, rng, 18)]
    elif style == "braid":
        groups = [_group_braid(head, margin, rng, 34)]
    if not groups:
        return []
    paths = np.concatenate([g[0] for g in groups])
    offsets = np.cumsum([0] + [len(g[1][0]) for g in groups[:-1]])
    inv = np.concatenate([g[1][1] + o for g, o in zip(groups, offsets)])
    radius = float(np.mean([g[2] for g in groups]))
    mesh = strand_mesh(paths, color, radius, rng, (np.arange(inv.max() + 1), inv))
    mesh.swing = SWING.get(style, 0.3)
    return [mesh]


# ---------------------------------------------------------------------------
# Facial hair and eyebrows: strands that hug the skin
# ---------------------------------------------------------------------------
class Skin:
    """Keeps points outside the head mesh by pushing them along the nearest vertex normal."""

    def __init__(self, head, region):
        idx = np.flatnonzero(region)
        self.v = head.P[idx].astype(np.float32)
        self.n = head.mesh.n[idx].astype(np.float32)
        self.v2 = (self.v ** 2).sum(1)
        self.vt = np.ascontiguousarray(-2 * self.v.T)   # so |p - v|^2 needs one small matmul

    def nearest(self, p):
        p = np.asarray(p, np.float32)
        out = np.empty(len(p), np.int64)
        for a in range(0, len(p), 1500):           # chunked so the distance matrix stays small
            d = p[a:a + 1500] @ self.vt + self.v2[None, :]
            out[a:a + 1500] = d.argmin(1)
        return out

    def normals(self, p):
        return self.n[self.nearest(p)].astype(float)

    def push(self, p, offset):
        j = self.nearest(p)
        rel = p - self.v[j]
        h = (rel * self.n[j]).sum(1)
        low = h < offset
        p[low] += self.n[j][low] * (offset - h[low])[:, None]
        return p


def _face_region(head):
    return (head.Z > -0.15) & (head.Y < 0.02) & (head.Y > -0.7)


def _tangent(vec, nrm):
    return unit(vec - nrm * (vec * nrm).sum(1, keepdims=True) + 1e-9)


def mustache_margin(head):
    """> 0 inside the moustache area above the upper lip (shape of head.Y)."""
    m = 1 - np.sqrt((head.X / 0.20) ** 2 + ((head.Y + 0.305) / 0.04) ** 2)
    return np.where(head.Z > 0.25, m, -1.0)


def facial_strand_meshes(head, beard_margin, color, with_beard):
    """Beard (optional) and mustache strands. `beard_margin` > 0 marks the beard area."""
    rng = np.random.default_rng(zlib.crc32(b"facial"))
    skin = Skin(head, _face_region(head))
    groups = []
    steps = 6

    if with_beard:
        fade = lambda p: rng.random(len(p)) < smoothstep(-0.215, -0.34, p[:, 1]) ** 0.7   # thin out up the cheeks
        root, nrm = sample_roots(head, beard_margin, 2800, rng, keep=fade, min_margin=0.02)
        towards_chin = np.stack([-np.sign(root[:, 0]) * 0.35, np.full(len(root), 0.0),
                                 np.full(len(root), 0.12)], 1)
        d0 = _tangent(DOWN + towards_chin, nrm)
        longer = smoothstep(-0.35, -0.62, root[:, 1])
        length = (0.09 + 0.10 * longer) * (0.85 + 0.3 * rng.random(len(root)))
        locks = make_locks(root, 16)
        paths = grow(root, nrm, d0, length, steps, rng, locks, gravity=0.5, wave=0.05, jitter=0.22,
                     floor=0.25, collider=lambda p: skin.push(p, 0.007))
        groups.append((paths, locks, 0.0055))

    root, nrm = sample_roots(head, mustache_margin(head), 800, rng, min_margin=0.0)
    sgn = np.where(root[:, 0] >= 0, 1.0, -1.0)
    d0 = _tangent(np.stack([sgn * 0.9, np.full(len(root), -0.45), np.full(len(root), 0.15)], 1), nrm)
    length = 0.10 * (0.8 + 0.4 * rng.random(len(root)))
    locks = make_locks(root, 24)
    paths = grow(root, nrm, d0, length, steps, rng, locks, gravity=0.45, wave=0.04, jitter=0.15,
                 floor=0.2, collider=lambda p: skin.push(p, 0.008))
    groups.append((paths, locks, 0.0062))

    paths = np.concatenate([g[0] for g in groups])
    offsets = np.cumsum([0] + [len(g[1][0]) for g in groups[:-1]])
    inv = np.concatenate([g[1][1] + o for g, o in zip(groups, offsets)])
    radius = float(np.mean([g[2] for g in groups]))
    return [strand_mesh(paths, color, radius, rng, (np.arange(inv.max() + 1), inv), flyaway=0.01,
                        part="facial")]


def brow_curve(t, side, dy, tilt):
    """Brow centre line and its direction (inner end t=0 to outer end t=1)."""
    x = side * (0.075 + 0.20 * t)
    y = 0.185 + 0.045 * np.sin(t * math.pi * 0.85) - 0.02 * t + dy - tilt * 0.05 * (1 - t) + tilt * 0.02 * t
    dx = side * 0.20 * np.ones_like(t)
    dyd = 0.045 * math.pi * 0.85 * np.cos(t * math.pi * 0.85) - 0.02 + tilt * 0.05 + tilt * 0.02
    return x, y, dx, dyd


def brow_strand_meshes(head, thickness, color, dy=0.0, tilt=0.0):
    """Eyebrow hairs, one mesh per side ([right, left]), lying along the brow, thicker at the inner end."""
    skin = Skin(head, (head.Z > 0.0) & (np.abs(head.Y - 0.18) < 0.3))
    out = []
    for side in (-1, 1):
        rng = np.random.default_rng(zlib.crc32(b"brows") + side)
        per = int(240 * thickness)
        t = rng.random(per)
        x, y, dx, dyd = brow_curve(t, side, dy, tilt)
        width = 0.016 * thickness * (1.1 - 0.65 * t)
        y = y + rng.uniform(-1, 1, per) * width * 0.9
        z = head.z_at_many(x, y)
        root = np.stack([x, y, z], 1)
        d0 = unit(np.stack([dx, dyd + 0.03 * (1 - t), np.zeros(per)], 1) + 1e-9)
        nrm = skin.normals(root)
        d0 = _tangent(d0 + rng.normal(size=d0.shape) * 0.12, nrm)
        locks = make_locks(root, 40)
        length = 0.045 * (0.75 + 0.5 * rng.random(per))
        paths = grow(root, nrm, d0, length, 4, rng, locks, gravity=0.05, jitter=0.1, floor=0.0,
                     collider=lambda p: skin.push(p, 0.004))
        mesh = strand_mesh(paths, color, 0.0034, rng, (np.arange(per), np.arange(per)),
                           flyaway=0.0, part="brows")
        mesh.side = side
        out.append(mesh)
    return out
