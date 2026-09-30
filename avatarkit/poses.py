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


def pose_at(name, t=0.0, animate=True):
    """Joint dict {node_name: (rx, ry, rz), 'root_dy': float} for a pose at time t."""
    return POSE_FUNCS[name](t, animate)


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
    return pose
