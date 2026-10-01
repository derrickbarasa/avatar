"""Body, clothes, hands and accessories, split into rig nodes."""
import math

import numpy as np

from .head import GOLD
from .mathutil import catmull, mix, rot_z
from .mesh import ellipsoid, limb, loft, merge, ring_mesh, tube
from .rig import Rig

LIFT = 0.30  # the body is built with y=0 at the head centre, then raised by this much
CLOTH = dict(shine=12, spec=0.04, kind="cloth")


def torso_keys(bt):
    sh, chest, waist, hip, bust, _ = bt
    mid = (chest + waist) / 2
    return np.array([  # y, half-width, front depth, back depth
        (-1.02, 0.30, 0.26, 0.30),                       # base of the neck
        (-1.075, 0.42 * sh, 0.28, 0.33),                 # the trapezius slopes out to the shoulder...
        (-1.15, 0.60 * sh, 0.30, 0.37),
        (-1.26, 0.80 * sh, 0.33, 0.40),
        (-1.40, 0.93 * sh, 0.37, 0.40),                  # ...the shoulder tip
        (-1.62, 0.91 * sh * chest, 0.44 + bust * 0.7, 0.42),   # upper chest, shoulder blades
        (-1.95, 0.84 * sh * chest, 0.49 + bust, 0.40),         # chest
        (-2.40, 0.70 * mid, 0.38 + bust * 0.4, 0.34),
        (-3.00, 0.60 * waist, 0.34, 0.27),               # waist: a gentle belly in front, the small of the back
        (-3.50, 0.72 * hip, 0.37, 0.41),
        (-3.90, 0.80 * hip, 0.38, 0.50),                 # hips; the buttocks push the back out
        (-4.15, 0.76 * hip, 0.37, 0.46),
        (-4.30, 0.55 * hip, 0.30, 0.34)])


class Torso:
    """Torso silhouette lookups shared by the skin and every garment on top."""

    def __init__(self, body_type, build):
        dense = catmull(torso_keys(body_type), 240)
        self.negy = -dense[:, 0]
        self.a, self.bf, self.bb = dense[:, 1] * build, dense[:, 2], dense[:, 3]

    def dims(self, ys, grow=0.0):
        ys = np.asarray(ys, float)
        a = np.interp(-ys, self.negy, self.a) + grow
        return a, np.interp(-ys, self.negy, self.bf) + grow, np.interp(-ys, self.negy, self.bb) + grow

    def section(self, y0, y1, n, grow, color, **kw):
        ys = np.linspace(y0, y1, n)
        a, bf, bb = self.dims(ys, grow)
        return loft(ys, a, bf, bb, color, **kw)

    def front_z(self, ys, xs=0.0, grow=0.0):
        a, bf, _ = self.dims(ys, grow)
        return bf * np.clip(1 - (np.abs(xs) / a) ** 2.4, 0, 1) ** (1 / 2.4)


def neck_mesh(skin, lift=0.0):
    """The neck; `lift` raises the top (the head moves up with it) for a longer or shorter neck."""
    neck = catmull([(0, -0.28 + lift, -0.08), (0, -0.55 + lift * 0.5, -0.06), (0, -0.85, -0.02)], 10)
    return tube(neck, np.linspace(0.255, 0.30, 10), skin, kind="skin")


FINGER_CODES = ("th", "ix", "md", "rg", "pk")      # thumb, index, middle, ring, pinky


def hand_parts(wrist, side, skin, size=1.0):
    """Palm mesh plus each finger as (code, proximal mesh, distal mesh, knuckle, joint) pieces.

    Fingers hang down from the wrist with the palm facing the body; the proximal piece bends
    about the knuckle and the distal piece about the middle joint, so a fist can really curl.
    The palm starts as wide as the forearm's wrist and widens to the knuckles; knuckles, joints and
    fingertips are rounded, and the thumb lies along the front edge of the palm.
    """
    wx, wy, wz = wrist
    k = 1.5 * size  # hand scale, so the hands match the forearm width
    sk = dict(kind="skin")

    def ball(centre, r):
        return ellipsoid(centre, (r,) * 3, skin, detail=10, **sk)

    # palm: thickness (x) and width (z) per section, from the wrist to the knuckles
    ys = np.array([-0.04, 0.03, 0.10, 0.165, 0.205]) * k          # starts a little inside the forearm
    pts = np.column_stack([np.full(5, wx), wy - ys, np.full(5, wz)])
    palm = merge([limb(pts, np.array([0.057, 0.052, 0.050, 0.047, 0.040]) * k,
                       np.array([0.079, 0.083, 0.089, 0.093, 0.090]) * k, skin, sides=16, **sk),
                  ellipsoid((wx, wy - 0.085 * k, wz + 0.055 * k), (0.040 * k, 0.078 * k, 0.050 * k), skin,
                            detail=12, **sk)])                       # the pad at the base of the thumb
    fingers = []
    for code, dz, length in zip(FINGER_CODES[1:], (-0.062, -0.021, 0.021, 0.060), (0.125, 0.145, 0.135, 0.105)):
        base = np.array([wx, wy - 0.20 * k, wz + dz * k])
        mid = base + np.array([0.0, -0.55 * length * k, 0.0])
        tip = base + np.array([0.0, -length * k, 0.0])
        r0, r1, r2 = 0.0225 * k, 0.0185 * k, 0.0135 * k
        prox = merge([tube(np.linspace(base, mid, 5), np.linspace(r0, r1, 5), skin, sides=10, thin=True, **sk),
                      ball(base, r0)])                                # the knuckle
        dist = merge([tube(np.linspace(mid, tip, 5), np.linspace(r1, r2, 5), skin, sides=10, thin=True, **sk),
                      ball(mid, r1), ball(tip, r2)])                  # the joint and a rounded tip
        fingers.append((code, prox, dist, base, mid))
    t0 = np.array([wx, wy - 0.085 * k, wz + 0.080 * k])
    t1 = np.array([wx - side * 0.012 * k, wy - 0.150 * k, wz + 0.118 * k])
    t2 = np.array([wx - side * 0.024 * k, wy - 0.215 * k, wz + 0.128 * k])
    prox = merge([tube(np.linspace(t0, t1, 5), np.linspace(0.028, 0.023, 5) * k, skin, sides=10, thin=True, **sk),
                  ball(t0, 0.028 * k)])
    dist = merge([tube(np.linspace(t1, t2, 5), np.linspace(0.023, 0.0165, 5) * k, skin, sides=10, thin=True, **sk),
                  ball(t1, 0.023 * k), ball(t2, 0.0165 * k)])
    fingers.insert(0, ("th", prox, dist, t0, t1))
    return palm, fingers


def hand_meshes(wrist, side, skin):
    """All hand meshes as one flat list (for callers that don't need the joints)."""
    palm, fingers = hand_parts(wrist, side, skin)
    return [palm] + [m for _, p, d, _, _ in fingers for m in (p, d)]


# Foot sections from heel to toe: z, half-width, top (below the ankle point), toe-out sway.
_FOOT = np.array([(-0.20, 0.050, -0.20),
                  (-0.15, 0.080, -0.10),
                  (-0.05, 0.098, 0.000),
                  (0.08, 0.098, -0.075),     # the instep falls away in front of the ankle
                  (0.22, 0.108, -0.17),
                  (0.36, 0.118, -0.25),      # ball of the foot
                  (0.48, 0.100, -0.30),
                  (0.56, 0.060, -0.33),
                  (0.59, 0.020, -0.35)])


def foot_mesh(ax, ay, radius, color, grow=0.0, sole=False, **kw):
    """A foot standing on the ground below the ankle at (ax, ay): heel, arch, ball and toes as one smooth shape.

    `grow` fattens it (a shoe over the foot); `sole` makes only the thin slab under it."""
    keys = np.array(_FOOT)
    stations = catmull(keys, 18)
    z, half, top = stations[:, 0], np.maximum(stations[:, 1], 0.012), stations[:, 2]
    floor = -0.38
    if sole:
        top = np.full_like(top, floor + 0.07)
    top = np.maximum(top, floor + 0.06)
    cy = ay + (top + floor) / 2
    ry = (top - floor) / 2 + grow * (0.6 if not sole else 0.2)
    rx = half * radius + grow
    ang = np.linspace(0, 2 * math.pi, 18, endpoint=False)
    ring = np.stack([ax + rx[:, None] * np.cos(ang)[None], cy[:, None] + ry[:, None] * np.sin(ang)[None],
                     np.broadcast_to(z[:, None], (len(z), len(ang))) + (grow if sole else 0) * np.sign(z - 0.2)[:, None] * 0.5],
                    -1)
    axis = np.stack([np.full(len(z), ax), cy, z], -1)
    return ring_mesh(ring, axis, color, **kw)


def _bump(t, c, w):
    """Smooth hump centred on c that is exactly zero further than w away."""
    return np.clip(1.0 - ((t - c) / w) ** 2, 0.0, 1.0) ** 2


def limb_curve(keys, m, t0, t1, humps):
    """Catmull curve through the limb keys with muscle shape added to the radius (joints stay round).

    Rows are x, y, z, half-width, depth / width, forward offset of the section (see `limb_mesh`).
    """
    out = catmull(keys, m, t0, t1)
    t = np.linspace(t0, t1, m)
    out[:, 3] *= 1.0 + sum(a * _bump(t, c, w) for a, c, w in humps)
    return out


def limb_mesh(curve, color, grow=0.0, **kw):
    """An arm or leg (or the sleeve / trouser over it, with `grow`: a number or one per point) from a limb curve."""
    r = curve[:, 3] + grow
    return limb(curve[:, :3], r, curve[:, 3] * curve[:, 4] + grow, color, off=curve[:, 5], **kw)


def limb_joint(key, color, grow=0.0, flat=1.0, drop=0.0, **kw):
    """The ball that rounds off a joint, shaped like the limb's section there (`flat` < 1 squashes it
    and `drop` lowers it: a shoulder that follows the slope of the collar bone)."""
    r = key[3] + grow
    return ellipsoid(key[:3] + np.array([0.0, -drop, key[5]]), (r, r * flat, key[3] * key[4] + grow), color,
                     **kw).set(joint=True)


ARM_HUMPS = ((0.22, 1.0, 0.95), (0.27, 2.7, 0.7))     # biceps / deltoid, forearm belly
LEG_HUMPS = ((0.13, 0.95, 0.9), (0.32, 2.8, 0.8))     # thigh, calf


def arm_curve(keys, m, t0, t1):
    return limb_curve(keys, m, t0, t1, ARM_HUMPS)


def leg_curve(keys, m, t0, t1):
    return limb_curve(keys, m, t0, t1, LEG_HUMPS)


def arm_keys(s, build, radius, height, shoulders=1.0):
    # x, y, z, half-width, depth / width, forward offset.  The forearm flattens toward the wrist (the hand
    # hangs with its palm to the thigh, so it is wider front to back than side to side).
    keys = np.array([(0.98, -1.36, 0.00, 0.190, 1.00, 0.0),    # shoulder
                     (1.05, -2.05, 0.02, 0.162, 1.06, 0.0),    # biceps
                     (1.10, -2.85, 0.06, 0.122, 1.00, 0.0),    # elbow
                     (1.14, -3.50, 0.16, 0.108, 1.18, 0.0),    # forearm: bends a little forward
                     (1.17, -4.15, 0.25, 0.078, 1.38, 0.0)])   # wrist
    keys[:, 0] *= s * build
    # wider shoulders move the shoulder joint out with the torso, easing back toward the hand
    keys[:, 0] += s * build * 0.9 * (shoulders - 1.0) * np.array([1.0, 1.0, 0.6, 0.35, 0.2])
    keys[:, 3] *= radius
    keys[:, 1] = -1.36 + (keys[:, 1] + 1.36) * height ** 0.8
    return keys


def leg_keys(s, build, radius, height, hips=1.0):
    keys = np.array([(0.360, -3.95, 0.000, 0.300, 1.05, 0.00),    # hip
                     (0.385, -4.75, 0.015, 0.262, 1.14, 0.02),    # thigh: deeper than wide
                     (0.400, -5.60, 0.030, 0.198, 1.00, 0.00),    # knee
                     (0.395, -6.40, 0.000, 0.182, 1.08, -0.03),   # calf bulges to the back
                     (0.380, -7.25, 0.000, 0.098, 1.28, 0.02)])   # ankle
    keys[:, 0] *= s * build
    keys[:, 0] += s * build * 0.6 * 0.36 * (hips - 1.0) * np.array([1.0, 1.0, 0.6, 0.3, 0.1])
    keys[:, 3] *= radius
    keys[:, 1] = -3.95 + (keys[:, 1] + 3.95) * height
    return keys


def build_body(v):
    """Rig with the body, clothes and accessories (everything below the neck)."""
    rig = Rig()
    skin = tuple(v["skin"])
    top, pants, shoestyle = v["top"], v["pants"], v["shoestyle"]
    tcol, pcol, scol = v["topcolor"], v["pantscolor"], v["shoes"]
    style = top                                # what was chosen; polo and sweater reuse the tee / long-sleeve cut
    dress = top == "dress"
    if dress:                                  # a dress is a short-sleeved top with a long flared skirt
        top, pants, pcol = "tee", "skirt", tcol
    elif top == "polo":
        top = "tee"
    elif top == "turtleneck":
        top = "long"
    elif top == "sweater":
        top = "long"
    body_type, build, height = v["bodytype"], v["build"], v["height"]
    shoulders, hips = v["shoulders"], v["hips"]
    body_type = (body_type[0] * shoulders, body_type[1], body_type[2], body_type[3] * hips,
                 body_type[4], body_type[5])
    radius = math.sqrt(build) * body_type[5]
    torso = Torso(body_type, build)
    sk = dict(kind="skin")
    bulk = {"hoodie": 0.07, "jacket": 0.06, "sweater": 0.055}.get(style, 0.035)
    hem = -3.72 if bulk < 0.05 else -3.85
    if dress:
        hem = -3.4
    pkind = "denim" if pants == "jeans" else "cloth"
    cut_grow, cut_flare = {"jeans": (0.035, 0.05), "leggings": (0.010, 0.0), "wide": (0.075, 0.15)}.get(pants, (0.035, 0.05))

    def garment(mesh, chest=False):
        """Apply the top's print (the emblem only belongs on the chest piece)."""
        mesh.pattern = v["pattern"] if (v["pattern"] != 4 or chest) else 0
        mesh.color2 = tuple(v["patterncolor"])
        return mesh

    rig.add_node("root", (0, -3.65, 0))
    rig.add_node("torso", (0, -3.0, 0), "root")
    root, trunk = rig["root"], rig["torso"]

    # --- torso and what goes over it -------------------------------------
    # Skin torso is split at the waist: the lower half stays with the pelvis, under the
    # trousers, so twisting the upper body never exposes skin at the hips.
    trunk.add(torso.section(-1.02, -3.35, 30, 0.0, skin, **sk))
    root.add(torso.section(-2.65, -4.28, 16, 0.0, skin, **sk))  # overlaps the upper half
    shirt = torso.section(-1.02, hem, 30, bulk, tcol, cap_ends=False, **CLOTH)
    trunk.add(garment(shirt, chest=True))
    # A slightly smaller copy of the shirt's lower half stays with the pelvis, so leaning or
    # twisting never opens a gap between the shirt and the trousers.
    root.add(garment(torso.section(-2.65, hem, 14, bulk - 0.004, tcol, cap_ends=False, **CLOTH), chest=True))
    if top == "hoodie":
        phi = np.radians(np.linspace(-115, 115, 24))
        hood = np.stack([0.40 * np.sin(phi), -1.05 + 0.04 * np.cos(phi), 0.02 - 0.32 * np.cos(phi)], -1)
        trunk.add(tube(hood, np.full(len(phi), 0.11), tcol, sides=12, **CLOTH))
        for s in (-1, 1):
            cord = catmull([(s * 0.16, -1.10, 0.30), (s * 0.15, -1.5, 0.42), (s * 0.13, -1.95, 0.45)], 8)
            trunk.add(tube(cord, np.full(8, 0.013), (0.92, 0.92, 0.92), sides=6, thin=True, **CLOTH))
    if top == "jacket":
        ys = np.linspace(-1.08, hem + 0.03, 24)
        zf = torso.front_z(ys, 0.0, bulk) + 0.004
        trunk.add(tube(np.column_stack([np.zeros_like(ys), ys, zf]), np.full(len(ys), 0.011),
                       (0.82, 0.82, 0.85), sides=6, shine=60, spec=0.4, thin=True, kind="glossy"))
        phi = np.linspace(0, 2 * math.pi, 32)
        collar = np.stack([0.34 * np.sin(phi), np.full(len(phi), -1.05), 0.30 * np.cos(phi)], -1)
        trunk.add(tube(collar, np.full(len(phi), 0.06), mix(tcol, (0, 0, 0), 0.15), sides=10,
                       cap_ends=False, **CLOTH))
    trim = mix(tcol, (0, 0, 0), 0.18)
    if style in ("polo", "sweater"):
        phi = np.linspace(0, 2 * math.pi, 36)
        polo = style == "polo"
        ring = np.stack([(0.36 if polo else 0.385) * np.sin(phi), np.full(len(phi), -1.035),
                         0.335 * np.cos(phi)], -1)
        trunk.add(tube(ring, np.full(len(phi), 0.052 if polo else 0.085), trim, sides=8, cap_ends=False, **CLOTH))
        if polo:                                  # placket down the chest with two buttons
            ys = np.linspace(-1.08, -1.75, 10)
            zf = torso.front_z(ys, 0.0, bulk) + 0.004
            trunk.add(tube(np.column_stack([np.zeros_like(ys), ys, zf]), np.full(len(ys), 0.02), trim, sides=6,
                           thin=True, **CLOTH))
            for yb in (-1.30, -1.55):
                zb = float(torso.front_z(np.array([yb]), 0.0, bulk)[0]) + 0.02
                trunk.add(ellipsoid((0, yb, zb), (0.03, 0.03, 0.015), (0.95, 0.95, 0.92), detail=10))
        else:                                     # sweater: ribbed band at the hem
            a, bf, bb = torso.dims(np.array([hem]), bulk + 0.01)
            ring = np.stack([a[0] * np.sin(phi), np.full(len(phi), hem + 0.02),
                             np.where(np.cos(phi) > 0, bf[0], bb[0]) * np.cos(phi)], -1)
            root.add(tube(ring, np.full(len(phi), 0.055), trim, sides=8, cap_ends=False, **CLOTH))
    if style == "turtleneck":                       # a rolled collar that hugs the neck
        ys = np.linspace(-1.12, -0.76, 8)
        trunk.add(loft(ys, np.full(8, 0.30), np.full(8, 0.245), np.full(8, 0.365), tcol, cap_ends=False, **CLOTH))
        roll = np.linspace(0, 2 * math.pi, 28, endpoint=False)
        trunk.add(tube(np.stack([0.30 * np.sin(roll), np.full(28, -0.78), -0.06 + 0.30 * np.cos(roll)], -1),
                       np.full(28, 0.045), trim, sides=8, cap_ends=False, **CLOTH))
    if pants == "skirt":
        hem_y = -5.45 if dress else -5.05
        ys = np.linspace(-3.2, hem_y, 16)
        t = (-ys - 3.2) / (-hem_y - 3.2)
        ta, tf, tb = torso.dims(np.minimum(ys, -3.2), bulk + 0.03)       # starts at the waist, never narrower than the hips
        flare = t ** 1.3
        a = np.maximum(ta[0] + (0.62 if dress else 0.45) * flare, ta)
        depth = np.maximum(max(tf[0], tb[0]) + (0.50 if dress else 0.38) * flare, np.maximum(tf, tb))
        skirt = loft(ys, a, depth, depth, pcol, cap_ends=False, **CLOTH)
        root.add(garment(skirt) if dress else skirt)
    else:
        root.add(torso.section(-3.25, -4.30, 12, 0.03, pcol, cap_ends=False, shine=12, spec=0.04, kind=pkind))

    # --- extras on the torso ------------------------------------------------
    if v["scarf"] != "none":
        phi = np.linspace(0, 2 * math.pi, 36)
        ring = np.stack([0.37 * np.sin(phi), np.full(len(phi), -1.02), 0.34 * np.cos(phi)], -1)
        trunk.add(tube(ring, np.full(len(phi), 0.10), v["scarfcolor"], sides=10, cap_ends=False, **CLOTH))
        tail = catmull([(0.16, -1.0, 0.33), (0.21, -1.6, 0.45), (0.18, -2.3, 0.47)], 12)
        trunk.add(tube(tail, np.linspace(0.085, 0.06, 12), v["scarfcolor"], sides=10, **CLOTH))
    if v["necklace"] != "none":
        phi = np.radians(np.linspace(-80, 80, 32))
        ys = -1.05 - 0.30 * np.cos(phi) ** 2
        xs = 0.36 * np.sin(phi)
        zs = torso.front_z(ys, xs, bulk) + 0.012
        chain = np.stack([xs, ys, zs], -1)
        trunk.add(tube(chain, np.full(len(phi), 0.008), GOLD, sides=6, thin=True,
                       shine=80, spec=0.6, kind="glossy"))
        if v["necklace"] == "pendant":
            trunk.add(ellipsoid((0, ys[16] - 0.04, zs[16] + 0.015), (0.035, 0.05, 0.02), GOLD,
                                detail=14, shine=80, spec=0.6, kind="glossy"))
    if v["neckwear"] != "none":
        nc = v["neckcolor"]
        y0 = -1.10
        z0 = float(torso.front_z(y0, 0.0, bulk)) + 0.035
        knot = ellipsoid((0, y0, z0), (0.045, 0.05, 0.035), nc, detail=14, **CLOTH)
        if v["neckwear"] == "bowtie":
            trunk.add(knot)
            for s in (-1, 1):
                trunk.add(ellipsoid((s * 0.115, y0, z0 - 0.005), (0.10, 0.06, 0.03), nc,
                                    rot=rot_z(s * -0.25), detail=16, **CLOTH))
        else:
            zt = float(torso.front_z(-1.62, 0.0, bulk)) + 0.025
            trunk.add(knot)
            trunk.add(ellipsoid((0, -1.62, zt), (0.075, 0.46, 0.022), nc, detail=20, **CLOTH))
    if v["bag"] != "none":
        c = v["bagcolor"]
        strap_c = mix(c, (0, 0, 0), 0.35)
        top_y, bot_y = -1.45, -3.10
        _, _, bb_mid = torso.dims(np.array([-1.9, -2.2]), bulk)
        z_mid = -(float(bb_mid.max()) + 0.0)                      # the bag rests on the shoulder blades

        def bag_part(y0, y1, half_w, depth, color, dz=0.0, n=9, **kw):
            """A rounded slab between two heights, as a loft (its ends are rounded off, not cut flat)."""
            ys = np.linspace(y0, y1, n)
            u = np.linspace(-1.0, 1.0, n)
            round_ends = np.maximum(1.0 - np.abs(u) ** 6, 0.0) ** 0.5 * 0.85 + 0.15
            m = loft(ys, half_w * (0.8 + 0.2 * round_ends), depth * round_ends, depth * round_ends, color, **kw)
            m.v[:, 2] += z_mid - depth + dz                      # sits against the back, bulging outward
            return m

        depth = 0.24
        trunk.add(bag_part(top_y, bot_y, 0.40, depth, c, **CLOTH))
        trunk.add(bag_part(top_y, top_y - 0.55, 0.415, depth + 0.03, mix(c, (0, 0, 0), 0.12), **CLOTH))    # the lid flap
        trunk.add(bag_part(bot_y + 0.15, bot_y + 0.85, 0.30, 0.07, strap_c, dz=-depth - 0.04, **CLOTH))    # a pocket
        for s_ in (-1, 1):
            xs = np.array([0.30, 0.40, 0.50, 0.52, 0.50, 0.55, 0.62, 0.50, 0.38]) * s_
            ys = np.array([-1.62, -1.30, -1.15, -1.40, -1.85, -2.25, -2.50, -2.80, -2.90])
            back = np.array([True, True, True, False, False, False, False, True, True])
            zf = torso.front_z(ys, xs, bulk) + 0.035                                           # on the chest
            zb = -(torso.dims(ys, bulk)[2] * np.clip(1 - (np.abs(xs) / torso.dims(ys, bulk)[0]) ** 2.4, 0, 1)
                   ** (1 / 2.4)) - 0.035                                                        # on the back
            zs = np.where(back, zb, zf)
            zs[2] = 0.0                                                                         # over the shoulder
            strap = catmull(np.column_stack([xs, ys, zs]), 30)
            trunk.add(tube(strap, np.full(30, 0.045), strap_c, sides=8, **CLOTH))

    # --- arms, hands, sleeves ---------------------------------------------------
    for name, s in (("L", 1), ("R", -1)):
        ak = arm_keys(s, build, radius, height, shoulders)
        legk = leg_keys(s, build, radius, height, hips)
        arm, fore, hand = (rig.add_node("arm" + name, ak[0, :3], "torso"),
                           rig.add_node("fore" + name, ak[2, :3], "arm" + name),
                           rig.add_node("hand" + name, ak[4, :3], "fore" + name))
        thigh, shin, foot = (rig.add_node("thigh" + name, legk[0, :3], "root"),
                             rig.add_node("shin" + name, legk[2, :3], "thigh" + name),
                             rig.add_node("foot" + name, legk[4, :3], "shin" + name))
        up, lo = arm_curve(ak, 16, 0.0, 2.0), arm_curve(ak, 18, 2.0, 4.0)
        arm.add(limb_mesh(up, skin, **sk), limb_joint(ak[0], skin, flat=0.82, drop=0.04, **sk))
        fore.add(limb_mesh(lo, skin, **sk), limb_joint(ak[2], skin, **sk))
        palm, fingers = hand_parts(ak[4, :3], s, skin, radius)
        hand.add(palm)
        for code, prox, dist, knuckle, joint in fingers:
            pn = rig.add_node(f"{code}1{name}", knuckle, "hand" + name)
            dn = rig.add_node(f"{code}2{name}", joint, f"{code}1{name}")
            pn.add(prox)
            dn.add(dist)
        grow = bulk - 0.005
        if top != "tank":
            if top == "tee":
                sl = arm_curve(ak, 14, 0.0, 1.7)
                arm.add(garment(limb_mesh(sl, tcol, grow, cap_ends=False, **CLOTH)))
            else:
                sl = arm_curve(ak, 16, 0.0, 2.0)
                arm.add(garment(limb_mesh(sl, tcol, grow, cap_ends=False, **CLOTH)))
                sl2 = arm_curve(ak, 18, 2.0, 3.95)
                fore.add(garment(limb_mesh(sl2, tcol, grow, cap_ends=False, **CLOTH)),
                         limb_joint(ak[2], tcol, grow, **CLOTH))
            arm.add(garment(limb_joint(ak[0], tcol, grow, flat=0.82, drop=0.04, **CLOTH)))
            if style == "sweater":                # ribbed cuff at the wrist
                end = arm_curve(ak, 3, 3.9, 3.95)[-1]
                ang = np.linspace(0, 2 * math.pi, 20)
                r_cuff = end[3] + grow + 0.012
                cuff = np.stack([end[0] + r_cuff * np.cos(ang), np.full(len(ang), end[1]),
                                 end[2] + (end[3] * end[4] + grow + 0.012) * np.sin(ang)], -1)
                fore.add(tube(cuff, np.full(len(ang), 0.035), trim, sides=8, cap_ends=False, **CLOTH))
        if v["watch"] != "none" and s == 1:
            w, r0 = ak[4, :3], ak[4, 3] + 0.03
            a = np.linspace(0, 2 * math.pi, 22)
            band = np.stack([w[0] + r0 * np.cos(a), np.full(len(a), w[1] + 0.07),
                             w[2] + (ak[4, 3] * ak[4, 4] + 0.03) * np.sin(a)], -1)
            fore.add(tube(band, np.full(len(a), 0.022), (0.12, 0.12, 0.14), sides=8, cap_ends=False, **CLOTH))
            fore.add(ellipsoid((w[0] + r0 + 0.02, w[1] + 0.07, w[2]), (0.03, 0.06, 0.06),
                               (0.78, 0.79, 0.83), detail=16, shine=80, spec=0.6, kind="glossy"))
            fore.add(ellipsoid((w[0] + r0 + 0.038, w[1] + 0.07, w[2]), (0.02, 0.048, 0.048),
                               (0.05, 0.06, 0.09), detail=16, shine=90, spec=0.7, kind="glossy"))

        # --- legs, trousers, shoes -------------------------------------------------
        up, lo = leg_curve(legk, 18, 0.0, 2.0), leg_curve(legk, 20, 2.0, 4.0)
        thigh.add(limb_mesh(up, skin, **sk), limb_joint(legk[0], skin, **sk))
        shin.add(limb_mesh(lo, skin, **sk), limb_joint(legk[2], skin, **sk))
        trouser = dict(shine=12, spec=0.04, kind=pkind)
        if pants in ("jeans", "leggings", "wide"):
            tu = leg_curve(legk, 18, 0.0, 2.0)
            thigh.add(limb_mesh(tu, pcol, cut_grow, cap_ends=False, **trouser), limb_joint(legk[0], pcol, cut_grow, **trouser))
            t_end = 2.9 if shoestyle == "boots" else 3.95                      # tucked into boots
            tl = leg_curve(legk, 20, 2.0, t_end)
            flare = cut_flare * np.clip((np.linspace(2.0, t_end, 20) - 3.0) / 0.95, 0, 1) ** 1.5   # the hem hangs loose
            shin.add(limb_mesh(tl, pcol, cut_grow + flare, cap_ends=False, **trouser),
                     limb_joint(legk[2], pcol, cut_grow, **trouser))
        elif pants == "shorts":
            tu = leg_curve(legk, 14, 0.0, 1.7)
            thigh.add(limb_mesh(tu, pcol, 0.035, cap_ends=False, **trouser), limb_joint(legk[0], pcol, 0.035, **trouser))
        ax, ay = legk[4, 0], legk[4, 1]
        if shoestyle == "barefoot":
            foot.add(foot_mesh(ax, ay, radius, skin, **sk))
        else:
            leather = shoestyle == "boots"
            skind = "glossy" if leather else "cloth"
            foot.add(foot_mesh(ax, ay, radius, scol, grow=0.035, shine=30, spec=0.15, kind=skind))
            sole = (0.12, 0.10, 0.10) if leather else (0.93, 0.93, 0.93)
            foot.add(foot_mesh(ax, ay, radius, sole, grow=0.05, sole=True, shine=20, spec=0.1))
            if leather:
                shaft = leg_curve(legk, 12, 2.8, 3.95)
                shin.add(limb_mesh(shaft, scol, 0.055, cap_ends=True, shine=30, spec=0.15, kind="glossy"))
    rig.ground_y = float(legk[4, 1] - 0.38 + LIFT)

    # Raise the whole body so the shoulders sit close under the head.
    for node in rig.nodes.values():
        node.pivot[1] += LIFT
        for m in node.meshes:
            m.v[:, 1] += LIFT
    return rig
