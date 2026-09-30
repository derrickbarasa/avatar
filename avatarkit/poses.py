"""Pose and idle-animation functions: time -> joint angles in degrees.

Conventions (avatar faces +z, its left side is +x):
  arms:   rz = +-abduction (side * degrees), rx = -forward swing
  elbows: forearm rx = -flex (hand comes forward), rz continues the arm's plane
  legs:   thigh rx = -forward swing, shin rx = +knee bend (foot goes back)
  head:   rx = nod down, ry = turn
"""
import math

SIDES = (("L", 1), ("R", -1))


def _idle(t, animate):
    if not animate:
        return {}
    return {"head": (1.5 * math.sin(0.7 * t), 4.0 * math.sin(0.5 * t), 0.8 * math.sin(0.4 * t)),
            "torso": (0.0, 1.2 * math.sin(0.5 * t), 0.0)}


def _relaxed(t, animate):
    p = _idle(t, animate)
    sway = math.sin(1.6 * t) if animate else 0.0
    for name, s in SIDES:
        p["arm" + name] = (-2.0, 0.0, s * (5.0 + 1.2 * sway))
        p["fore" + name] = (-8.0, 0.0, 0.0)
    return p


def _apose(t, animate):
    p = _idle(t, animate)
    for name, s in SIDES:
        p["arm" + name] = (0.0, 0.0, s * 38.0)
        p["fore" + name] = (-5.0, 0.0, 0.0)
    return p


def _wave(t, animate):
    p = _relaxed(t, animate)
    wag = math.sin(7 * t) if animate else 0.0
    p["armR"] = (0.0, 0.0, -105.0)
    p["foreR"] = (0.0, 0.0, -(28.0 + 22.0 * wag))
    p["handR"] = (0.0, 0.0, -8.0 * wag)
    return p


def _hips(t, animate):
    p = _idle(t, animate)
    for name, s in SIDES:
        p["arm" + name] = (0.0, 0.0, s * 40.0)
        p["fore" + name] = (0.0, 0.0, -s * 112.0)
    return p


def _cheer(t, animate):
    p = _idle(t, animate)
    bob = math.sin(5 * t) if animate else 0.0
    for name, s in SIDES:
        p["arm" + name] = (0.0, 0.0, s * (152.0 + 4.0 * bob))
        p["fore" + name] = (0.0, 0.0, s * 8.0)
    p["head"] = (-6.0, 0.0, 0.0)
    return p


def _walk(t, animate):
    ph = 2 * math.pi * t * 0.9 if animate else 0.0
    p = _idle(t, animate)
    swing = math.sin(ph)
    for name, s in SIDES:
        leg_ph = ph if s > 0 else ph + math.pi
        leg = math.sin(leg_ph)   # > 0 while this leg is forward
        p["thigh" + name] = (-28.0 * leg, 0.0, s * 1.5)
        p["shin" + name] = (8.0 + 40.0 * max(0.0, math.cos(leg_ph)), 0.0, 0.0)
        p["foot" + name] = (-6.0 * leg, 0.0, 0.0)
        p["arm" + name] = (24.0 * leg, 0.0, s * 4.0)
        p["fore" + name] = (-14.0 - 10.0 * max(0.0, leg), 0.0, 0.0)
    p["torso"] = (0.0, -6.0 * swing, 0.0)
    p["root_dy"] = 0.05 * abs(math.cos(ph))
    return p


def _dance(t, animate):
    ph = 2 * math.pi * t * 1.6 if animate else 0.0
    beat, half = math.sin(ph), math.sin(ph / 2)
    p = {"root_dy": 0.07 * abs(math.sin(ph / 2 * 2)),
         "torso": (2.0 * abs(beat), 8.0 * half, 6.0 * beat),
         "head": (4.0 * abs(beat) - 2.0, -10.0 * half, -5.0 * beat)}
    for name, s in SIDES:
        pump = 0.5 + 0.5 * math.sin(ph + (0 if s > 0 else math.pi))
        p["arm" + name] = (-10.0 * pump, 0.0, s * (28.0 + 55.0 * pump))
        p["fore" + name] = (-25.0 - 30.0 * pump, 0.0, s * 15.0 * math.sin(2 * ph))
        bend = max(0.0, math.sin(ph + (0 if s > 0 else math.pi)))
        p["thigh" + name] = (-14.0 * bend, 0.0, s * 4.0)
        p["shin" + name] = (10.0 + 26.0 * bend, 0.0, 0.0)
    return p


POSE_FUNCS = {"relaxed": _relaxed, "apose": _apose, "wave": _wave, "hips": _hips,
              "cheer": _cheer, "walk": _walk, "dance": _dance}


# ---------------------------------------------------------------------------
# Hand shapes: bend (degrees) of each finger's two joints, thumb first
# ---------------------------------------------------------------------------
FINGER_CODES = ("th", "ix", "md", "rg", "pk")
HAND_SHAPES = {
    "open": [(4, 4), (3, 3), (3, 3), (3, 3), (3, 3)],
    "relaxed": [(14, 10), (16, 18), (20, 24), (24, 28), (28, 30)],
    "fist": [(32, 28), (85, 90), (90, 95), (92, 95), (88, 90)],
    "point": [(30, 26), (2, 2), (90, 95), (92, 95), (88, 90)],
    "thumbs_up": [(2, 2), (88, 92), (92, 96), (94, 96), (90, 92)],
}
# which shapes each pose uses (left, right); anything else is relaxed
POSE_HANDS = {"wave": ("relaxed", "open"), "cheer": ("open", "open"), "hips": ("fist", "fist"),
              "apose": ("relaxed", "relaxed"), "dance": ("open", "open")}


def hand_joints(side_name, side, shape_a, shape_b=None, t=0.0):
    """Finger joint angles for one hand, blending shape_a -> shape_b by t (0..1)."""
    a = HAND_SHAPES[shape_a]
    b = HAND_SHAPES[shape_b or shape_a]
    out = {}
    for code, (a1, a2), (b1, b2) in zip(FINGER_CODES, a, b):
        out[f"{code}1{side_name}"] = (0.0, 0.0, -side * (a1 + (b1 - a1) * t))
        out[f"{code}2{side_name}"] = (0.0, 0.0, -side * (a2 + (b2 - a2) * t))
    return out


def set_hand(pose, side_name, side, shape_a, shape_b=None, t=0.0):
    pose.update(hand_joints(side_name, side, shape_a, shape_b, t))


def pose_at(name, t=0.0, animate=True):
    """Joint dict {node_name: (rx, ry, rz), 'root_dy': float} for a pose at time t."""
    pose = POSE_FUNCS[name](t, animate)
    left, right = POSE_HANDS.get(name, ("relaxed", "relaxed"))
    set_hand(pose, "L", 1, left)
    set_hand(pose, "R", -1, right)
    return pose


def blink_angle(t):
    """Eyelid blink in degrees (0 = open) for time t; a quick blink every ~4 s."""
    phase = (t % 4.3) / 0.16
    return 58.0 * math.sin(math.pi * phase) if phase < 1.0 else 0.0


def _hash01(k):
    """Deterministic pseudo-random number in [0, 1) for an integer key."""
    return (math.sin(k * 12.9898 + 78.233) * 43758.5453) % 1.0


def gaze_at(t, speaking=False):
    """Eye direction (pitch, yaw) in degrees: small darting glances every ~1.6 s.

    While speaking the eyes hold the viewer's gaze more and glance less.
    """
    period, blend = 1.6, 0.12
    k, frac = int(t // period), (t % period)
    amp = 0.35 if speaking else 1.0

    def target(i):
        return ((_hash01(i) - 0.5) * 9.0 * amp, (_hash01(i + 100) - 0.5) * 16.0 * amp)

    a, b = target(k - 1), target(k)
    u = min(1.0, frac / blend)
    u = u * u * (3 - 2 * u)
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def talk_overlay(pose, mouth, t, gestures=True):
    """Add speaking body language to a pose: head beats, raised brows and hand gestures.

    `mouth` is a speech.MouthShape (energy = how animated, beat = a stressed syllable).
    Arm gestures only apply when the arms hang at the sides, so raised-arm poses stay intact.
    """
    e, b = mouth.energy, mouth.beat
    head = pose.get("head", (0.0, 0.0, 0.0))
    pose["head"] = (head[0] + 9.0 * b + 1.5 * e * math.sin(2.1 * t),
                    head[1] + 4.0 * e * math.sin(1.1 * t), head[2] + 2.0 * e * math.sin(0.9 * t))
    pose["brows"] = (-2.2 * e - 5.0 * b, 0.0, 0.0)
    torso = pose.get("torso", (0.0, 0.0, 0.0))
    pose["torso"] = (torso[0], torso[1] + 2.0 * e * math.sin(0.8 * t), torso[2])
    hanging = all(abs(pose.get("arm" + n, (0, 0, 0))[2]) < 15 for n, _ in SIDES)
    if gestures and hanging:
        for i, (name, s) in enumerate(SIDES):
            g = e * (0.55 + 0.45 * math.sin(2.3 * t + i * 2.4))
            arm = pose.get("arm" + name, (0.0, 0.0, 0.0))
            fore = pose.get("fore" + name, (0.0, 0.0, 0.0))
            hand = pose.get("hand" + name, (0.0, 0.0, 0.0))
            pose["arm" + name] = (arm[0] - 22.0 * g, arm[1], arm[2] + s * 10.0 * g)
            pose["fore" + name] = (fore[0] - 42.0 * g - 6.0 * b, fore[1], fore[2])
            pose["hand" + name] = (hand[0], hand[1], hand[2] + 14.0 * e * math.sin(6.0 * t + i))
            set_hand(pose, name, s, "relaxed", "open", min(1.0, 1.4 * g))
    return pose


# ---------------------------------------------------------------------------
# Emotes: short one-shot animations layered over the current pose
# ---------------------------------------------------------------------------
EMOTES = [("Nod", "nod", 1.3), ("Shake head", "shake", 1.3), ("Shrug", "shrug", 1.7),
          ("Bow", "bow", 2.0), ("Clap", "clap", 2.6), ("Laugh", "laugh", 2.4),
          ("Think", "think", 2.8), ("Point", "point", 2.0), ("Say hi", "hello", 2.6)]
EMOTE_DURATION = {key: dur for _, key, dur in EMOTES}


def _smooth(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _blend(pose, joint, target, w):
    cur = pose.get(joint, (0.0, 0.0, 0.0))
    pose[joint] = tuple(c + (t - c) * w for c, t in zip(cur, target))


def emote_overlay(pose, name, u):
    """Blend emote `name` at progress u (0..1) into `pose`; returns extras for face and eyes.

    extras: mouth=(open, wide, press, smile) or None, brow_raise, brow_tilt, gaze or None, lid.
    """
    w = _smooth(u / 0.15) * _smooth((1 - u) / 0.15)
    sin = lambda cycles: math.sin(2 * math.pi * cycles * u)
    ex = {"mouth": None, "brow_raise": 0.0, "brow_tilt": 0.0, "gaze": None, "lid": 0.0}
    head = pose.get("head", (0.0, 0.0, 0.0))
    if name == "nod":
        _blend(pose, "head", (head[0] + 15 * sin(2), head[1], head[2]), w)
    elif name == "shake":
        _blend(pose, "head", (head[0], head[1] + 26 * sin(2.5), head[2]), w)
        ex["brow_raise"] = 0.2 * w
    elif name == "shrug":
        for n, s in SIDES:
            _blend(pose, "arm" + n, (0.0, 0.0, s * 16), w)
            _blend(pose, "fore" + n, (-55.0, 0.0, 0.0), w)
        _blend(pose, "head", (head[0] + 4, head[1], 7.0), w)
        ex.update(mouth=(0.0, 0.3, 1.0, -0.25), brow_raise=0.7 * w, brow_tilt=-0.4 * w)
        for n, sd in SIDES:
            set_hand(pose, n, sd, "relaxed", "open", w)
    elif name == "bow":
        _blend(pose, "torso", (30.0, 0.0, 0.0), w)
        _blend(pose, "head", (14.0, 0.0, 0.0), w)
        for n, s in SIDES:
            _blend(pose, "arm" + n, (-12.0, 0.0, s * 6), w)
        ex["mouth"] = (0.0, 0.2, 0.0, 0.4)
    elif name == "clap":
        beat = 0.5 + 0.5 * sin(5)
        for n, s in SIDES:
            _blend(pose, "arm" + n, (-38.0, 0.0, s * (4 + 10 * beat)), w)
            _blend(pose, "fore" + n, (-95.0, 0.0, -s * (6 - 8 * beat)), w)
        ex.update(mouth=(0.25, 0.4, 0.0, 0.9), brow_raise=0.4 * w)
        for n, sd in SIDES:
            set_hand(pose, n, sd, "relaxed", "open", w)
    elif name == "laugh":
        buzz = sin(7)
        _blend(pose, "head", (head[0] - 8 + 4 * buzz, head[1], head[2]), w)
        _blend(pose, "torso", (-5 + 3 * buzz, 0.0, 0.0), w)
        ex.update(mouth=(0.5 + 0.3 * buzz, 0.5, 0.0, 0.9), brow_raise=0.5 * w, lid=6 * w)
    elif name == "think":
        _blend(pose, "armR", (-62.0, 0.0, -10.0), w)
        _blend(pose, "foreR", (-128.0, 0.0, 0.0), w)
        _blend(pose, "head", (head[0] - 4, head[1] + 12, 9.0), w)
        ex.update(mouth=(0.0, -0.3, 1.0, -0.1), brow_raise=0.3 * w, brow_tilt=0.3 * w,
                  gaze=(-7 * w, 10 * w))
        set_hand(pose, "R", -1, "relaxed", "fist", w)
    elif name == "point":
        _blend(pose, "armR", (-82.0, 0.0, -14.0), w)
        _blend(pose, "foreR", (-6.0, 0.0, 0.0), w)
        _blend(pose, "head", (head[0], head[1] - 10, head[2]), w)
        ex.update(mouth=(0.1, 0.5, 0.0, 0.6), gaze=(0.0, -9 * w))
        set_hand(pose, "R", -1, "relaxed", "point", w)
    elif name == "hello":
        _blend(pose, "armR", (0.0, 0.0, -105.0), w)
        _blend(pose, "foreR", (0.0, 0.0, -(28.0 + 20.0 * sin(3))), w)
        ex.update(mouth=(0.3, 0.5, 0.0, 0.8), brow_raise=0.4 * w)
        set_hand(pose, "R", -1, "relaxed", "open", w)
    return ex
