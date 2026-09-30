"""Small math helpers shared by the mesh builders."""
import math

import numpy as np


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def gauss(x, y, cx, cy, sx, sy):
    return np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rot_to(direction):
    """Rotation matrix taking +z onto `direction`."""
    z = np.array([0.0, 0.0, 1.0])
    d = unit(direction)
    c = float(z @ d)
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(z, d)
    s2 = float(axis @ axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s2)


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_matrix(rx, ry, rz):
    """Rotation for degrees (rx, ry, rz), composed as Ry * Rx * Rz.

    This matches the order the renderer applies glRotatef(ry), (rx), (rz).
    """
    return rot_y(math.radians(ry)) @ rot_x(math.radians(rx)) @ rot_z(math.radians(rz))


def matrix_to_quat(m):
    """Unit quaternion (x, y, z, w) from a 3x3 rotation matrix."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    q = np.array(q, float)
    return tuple(q / np.linalg.norm(q))


def catmull(keys, m, t0=0.0, t1=None):
    """Sample a Catmull-Rom spline through `keys` (K, D) at m points.

    t0/t1 are positions in key units (0 .. K-1); by default the whole curve.
    """
    keys = np.asarray(keys, float)
    k = len(keys)
    t = np.linspace(t0, (k - 1) if t1 is None else t1, m)
    i = np.minimum(t.astype(int), k - 2)
    u = (t - i)[:, None]
    p0 = keys[np.maximum(i - 1, 0)]
    p1, p2 = keys[i], keys[i + 1]
    p3 = keys[np.minimum(i + 2, k - 1)]
    return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


def mix(a, b, t):
    return tuple(float(x) for x in np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t)
