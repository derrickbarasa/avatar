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


POSE_FUNCS = {"relaxed": _relaxed, "apose": _apose, "wave": _wave, "hips": _hips,
              "cheer": _cheer, "walk": _walk}


def pose_at(name, t=0.0, animate=True):
    """Joint dict {node_name: (rx, ry, rz), 'root_dy': float} for a pose at time t."""
    return POSE_FUNCS[name](t, animate)


def blink_angle(t):
    """Eyelid blink in degrees (0 = open) for time t; a quick blink every ~4 s."""
    phase = (t % 4.3) / 0.16
    return 58.0 * math.sin(math.pi * phase) if phase < 1.0 else 0.0
