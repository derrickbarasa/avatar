"""Body, clothes, hands and accessories, split into rig nodes."""
import math

import numpy as np

from .head import GOLD
from .mathutil import catmull, mix
from .mesh import ellipsoid, loft, tube
from .rig import Rig

LIFT = 0.30  # the body is built with y=0 at the head centre, then raised by this much
CLOTH = dict(shine=12, spec=0.04, kind="cloth")


def torso_keys(bt):
    sh, chest, waist, hip, bust, _ = bt
    mid = (chest + waist) / 2
    return np.array([  # y, half-width, front depth, back depth
        (-1.02, 0.30, 0.27, 0.27),
        (-1.15, 0.72 * sh, 0.34, 0.34),
        (-1.32, 0.90 * sh, 0.40, 0.38),
        (-1.80, 0.86 * sh * chest, 0.42 + bust, 0.40),
        (-2.40, 0.72 * mid, 0.37 + bust * 0.4, 0.36),
        (-3.00, 0.66 * waist, 0.33, 0.32),
        (-3.60, 0.74 * hip, 0.37, 0.37),
        (-4.15, 0.76 * hip, 0.38, 0.38),
        (-4.30, 0.55 * hip, 0.30, 0.30)])


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


def neck_mesh(skin):
    neck = catmull([(0, -0.28, 0.02), (0, -0.55, 0.03), (0, -0.85, 0.0)], 10)
    return tube(neck, np.linspace(0.235, 0.27, 10), skin, kind="skin")


def hand_meshes(wrist, side, skin):
    """Palm, four fingers and a thumb, hanging down with the palm facing the body."""
    wx, wy, wz = wrist
    k = 1.25  # hand scale, so the hands match the forearm width
    sk = dict(kind="skin")
    out = [ellipsoid((wx, wy - 0.10 * k, wz), (0.045 * k, 0.095 * k, 0.075 * k), skin, detail=16, **sk)]
    for dz, length in zip((-0.05, -0.017, 0.017, 0.05), (0.12, 0.14, 0.13, 0.10)):
        t = np.linspace(0, 1, 5)
        path = np.stack([wx - side * 0.03 * k * t ** 2, wy - 0.17 * k - length * k * t,
                         np.full(5, wz + dz * k)], -1)
        out.append(tube(path, np.linspace(0.02, 0.013, 5) * k, skin, sides=8, thin=True, **sk))
    thumb = np.array([(wx - side * 0.01, wy - 0.06 * k, wz + 0.06 * k),
                      (wx - side * 0.02 * k, wy - 0.11 * k, wz + 0.10 * k),
                      (wx - side * 0.03 * k, wy - 0.17 * k, wz + 0.115 * k)])
    out.append(tube(catmull(thumb, 6), np.linspace(0.024, 0.015, 6) * k, skin, sides=8,
                    thin=True, **sk))
    return out


def arm_keys(s, build, radius, height):
    keys = np.array([(0.98, -1.36, 0.00, 0.190),   # shoulder
                     (1.05, -2.05, 0.02, 0.170),   # biceps
                     (1.10, -2.85, 0.05, 0.155),   # elbow
                     (1.13, -3.50, 0.08, 0.140),   # forearm
                     (1.16, -4.15, 0.10, 0.115)])  # wrist
    keys[:, 0] *= s * build
    keys[:, 3] *= radius
    keys[:, 1] = -1.36 + (keys[:, 1] + 1.36) * height ** 0.8
    return keys


def leg_keys(s, build, radius, height):
    keys = np.array([(0.360, -3.95, 0.000, 0.300),  # hip
                     (0.385, -4.75, 0.015, 0.270),  # thigh
                     (0.400, -5.60, 0.030, 0.215),  # knee
                     (0.395, -6.40, 0.000, 0.200),  # calf
                     (0.380, -7.25, 0.000, 0.150)])  # ankle
    keys[:, 0] *= s * build
    keys[:, 3] *= radius
    keys[:, 1] = -3.95 + (keys[:, 1] + 3.95) * height
    return keys


def build_body(v):
    """Rig with the body, clothes and accessories (everything below the neck)."""
    rig = Rig()
    skin = tuple(v["skin"])
    top, pants, shoestyle = v["top"], v["pants"], v["shoestyle"]
    tcol, pcol, scol = v["topcolor"], v["pantscolor"], v["shoes"]
    body_type, build, height = v["bodytype"], v["build"], v["height"]
    radius = math.sqrt(build) * body_type[5]
    torso = Torso(body_type, build)
    sk = dict(kind="skin")
    bulk = {"hoodie": 0.07, "jacket": 0.06}.get(top, 0.035)
    hem = -3.72 if bulk < 0.05 else -3.85
    pkind = "denim" if pants == "jeans" else "cloth"

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
    if pants == "skirt":
        ys = np.linspace(-3.2, -5.05, 16)
        t = (-ys - 3.2) / 1.85
        a = (0.70 * build + 0.10) + 0.45 * t ** 1.3
        depth = 0.44 + 0.30 * t ** 1.3
        root.add(loft(ys, a, depth, depth, pcol, cap_ends=False, **CLOTH))
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
    if v["bag"] != "none":
        c = v["bagcolor"]
        strap_c = mix(c, (0, 0, 0), 0.35)
        z_back = -(float(torso.dims(-2.1)[2]) + bulk + 0.24)
        trunk.add(ellipsoid((0, -2.15, z_back), (0.50, 0.70, 0.26), c, detail=26, **CLOTH))
        trunk.add(ellipsoid((0, -2.55, z_back - 0.18), (0.30, 0.22, 0.10), strap_c, detail=16, **CLOTH))
        for s in (-1, 1):
            strap = catmull([(s * 0.42, -1.35, -0.42), (s * 0.50, -1.16, -0.03), (s * 0.44, -1.55, 0.32),
                             (s * 0.62, -2.25, 0.12), (s * 0.48, -2.7, -0.40)], 26)
            trunk.add(tube(strap, np.full(26, 0.05), strap_c, sides=8, **CLOTH))

    # --- arms, hands, sleeves ---------------------------------------------------
    for name, s in (("L", 1), ("R", -1)):
        ak = arm_keys(s, build, radius, height)
        legk = leg_keys(s, build, radius, height)
        arm, fore, hand = (rig.add_node("arm" + name, ak[0, :3], "torso"),
                           rig.add_node("fore" + name, ak[2, :3], "arm" + name),
                           rig.add_node("hand" + name, ak[4, :3], "fore" + name))
        thigh, shin, foot = (rig.add_node("thigh" + name, legk[0, :3], "root"),
                             rig.add_node("shin" + name, legk[2, :3], "thigh" + name),
                             rig.add_node("foot" + name, legk[4, :3], "shin" + name))
        up, lo = catmull(ak, 16, 0.0, 2.0), catmull(ak, 18, 2.0, 4.0)
        arm.add(tube(up[:, :3], up[:, 3], skin, **sk), ellipsoid(ak[0, :3], (ak[0, 3],) * 3, skin, **sk).set(joint=True))
        fore.add(tube(lo[:, :3], lo[:, 3], skin, **sk), ellipsoid(ak[2, :3], (ak[2, 3],) * 3, skin, **sk).set(joint=True))
        hand.add(hand_meshes(ak[4, :3], s, skin))
        grow = bulk - 0.005
        if top != "tank":
            if top == "tee":
                sl = catmull(ak, 14, 0.0, 1.7)
                arm.add(garment(tube(sl[:, :3], sl[:, 3] + grow, tcol, cap_ends=False, **CLOTH)))
            else:
                sl = catmull(ak, 16, 0.0, 2.0)
                arm.add(garment(tube(sl[:, :3], sl[:, 3] + grow, tcol, cap_ends=False, **CLOTH)))
                sl2 = catmull(ak, 18, 2.0, 3.95)
                fore.add(garment(tube(sl2[:, :3], sl2[:, 3] + grow, tcol, cap_ends=False, **CLOTH)),
                         ellipsoid(ak[2, :3], (ak[2, 3] + grow,) * 3, tcol, **CLOTH).set(joint=True))
            arm.add(garment(ellipsoid(ak[0, :3], (ak[0, 3] + grow,) * 3, tcol, **CLOTH).set(joint=True)))
        if v["watch"] != "none" and s == 1:
            w, r0 = ak[4, :3], ak[4, 3] + 0.03
            a = np.linspace(0, 2 * math.pi, 22)
            band = np.stack([w[0] + r0 * np.cos(a), np.full(len(a), w[1] + 0.07), w[2] + r0 * np.sin(a)], -1)
            fore.add(tube(band, np.full(len(a), 0.022), (0.12, 0.12, 0.14), sides=8, cap_ends=False, **CLOTH))
            fore.add(ellipsoid((w[0] + r0 + 0.02, w[1] + 0.07, w[2]), (0.03, 0.06, 0.06),
                               (0.78, 0.79, 0.83), detail=16, shine=80, spec=0.6, kind="glossy"))
            fore.add(ellipsoid((w[0] + r0 + 0.038, w[1] + 0.07, w[2]), (0.02, 0.048, 0.048),
                               (0.05, 0.06, 0.09), detail=16, shine=90, spec=0.7, kind="glossy"))

        # --- legs, trousers, shoes -------------------------------------------------
        up, lo = catmull(legk, 18, 0.0, 2.0), catmull(legk, 20, 2.0, 4.0)
        thigh.add(tube(up[:, :3], up[:, 3], skin, **sk), ellipsoid(legk[0, :3], (legk[0, 3],) * 3, skin, **sk).set(joint=True))
        shin.add(tube(lo[:, :3], lo[:, 3], skin, **sk), ellipsoid(legk[2, :3], (legk[2, 3],) * 3, skin, **sk).set(joint=True))
        trouser = dict(shine=12, spec=0.04, kind=pkind)
        if pants == "jeans":
            tu = catmull(legk, 18, 0.0, 2.0)
            thigh.add(tube(tu[:, :3], tu[:, 3] + 0.035, pcol, cap_ends=False, **trouser),
                      ellipsoid(legk[0, :3], (legk[0, 3] + 0.035,) * 3, pcol, **trouser).set(joint=True))
            tl = catmull(legk, 20, 2.0, 2.9 if shoestyle == "boots" else 3.95)  # tucked into boots
            shin.add(tube(tl[:, :3], tl[:, 3] + 0.035, pcol, cap_ends=False, **trouser),
                     ellipsoid(legk[2, :3], (legk[2, 3] + 0.035,) * 3, pcol, **trouser).set(joint=True))
        elif pants == "shorts":
            tu = catmull(legk, 14, 0.0, 1.7)
            thigh.add(tube(tu[:, :3], tu[:, 3] + 0.035, pcol, cap_ends=False, **trouser),
                      ellipsoid(legk[0, :3], (legk[0, 3] + 0.035,) * 3, pcol, **trouser).set(joint=True))
        ax, ay = legk[4, 0], legk[4, 1]
        if shoestyle == "barefoot":
            foot.add(ellipsoid((ax, ay - 0.31, 0.12), (0.13 * radius, 0.07, 0.27), skin, **sk))
        else:
            leather = shoestyle == "boots"
            skind = "glossy" if leather else "cloth"
            foot.add(ellipsoid((ax, ay - 0.25, 0.16), (0.15 * radius + 0.02, 0.12, 0.33), scol,
                               shine=30, spec=0.15, kind=skind))
            sole = (0.12, 0.10, 0.10) if leather else (0.93, 0.93, 0.93)
            foot.add(ellipsoid((ax, ay - 0.345, 0.16), (0.155 * radius + 0.025, 0.035, 0.335), sole,
                               shine=20, spec=0.1))
            if leather:
                shaft = catmull(legk, 12, 2.8, 3.95)
                shin.add(tube(shaft[:, :3], shaft[:, 3] + 0.055, scol, cap_ends=True,
                              shine=30, spec=0.15, kind="glossy"))
    rig.ground_y = float(legk[4, 1] - 0.38 + LIFT)

    # Raise the whole body so the shoulders sit close under the head.
    for node in rig.nodes.values():
        node.pivot[1] += LIFT
        for m in node.meshes:
            m.v[:, 1] += LIFT
    return rig
