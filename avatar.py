#!/usr/bin/env python3
"""Procedural 3D avatar creator.

The character is generated in code from smooth meshes (a sculpted head with
nose, brows, cheeks and jaw; eyes with lids; lips; hair styles; a full body
with clothes) and can be customised live from the side panel.

    python avatar.py
    python avatar.py --shot out.png --view face --set hair=Long --set skin=Tan
"""
import argparse
import functools
import json
import math
import os
import random
import sys
import time

import numpy as np
import pygame
from OpenGL.GL import *  # noqa: F401,F403
from OpenGL.GLU import gluPerspective

W, H = 1000, 720
PANEL_W = 320
VIEW_W = W - PANEL_W
FPS = 60
EXPORT_DIR = "exports"

# ---------------------------------------------------------------------------
# Options shown in the customizer: (key, label, [(name, value), ...])
# ---------------------------------------------------------------------------
SKIN_TONES = [
    ("Porcelain", (0.96, 0.80, 0.69)),
    ("Light", (0.90, 0.72, 0.55)),
    ("Medium", (0.78, 0.58, 0.40)),
    ("Tan", (0.62, 0.44, 0.29)),
    ("Brown", (0.45, 0.31, 0.20)),
    ("Deep", (0.30, 0.20, 0.14)),
]
HAIR_COLORS = [
    ("Black", (0.06, 0.05, 0.05)),
    ("Dark brown", (0.22, 0.14, 0.09)),
    ("Brown", (0.38, 0.24, 0.13)),
    ("Auburn", (0.55, 0.22, 0.12)),
    ("Ginger", (0.72, 0.36, 0.14)),
    ("Blonde", (0.85, 0.68, 0.36)),
    ("Platinum", (0.92, 0.88, 0.78)),
    ("Gray", (0.60, 0.60, 0.62)),
    ("Pink", (0.90, 0.42, 0.62)),
    ("Blue", (0.20, 0.35, 0.80)),
]
EYE_COLORS = [
    ("Brown", (0.30, 0.17, 0.08)),
    ("Hazel", (0.50, 0.35, 0.15)),
    ("Amber", (0.70, 0.45, 0.10)),
    ("Blue", (0.20, 0.42, 0.72)),
    ("Green", (0.25, 0.55, 0.35)),
    ("Gray", (0.45, 0.52, 0.58)),
    ("Dark", (0.12, 0.08, 0.06)),
]
CLOTH_COLORS = [
    ("White", (0.93, 0.93, 0.94)),
    ("Black", (0.10, 0.10, 0.12)),
    ("Red", (0.78, 0.15, 0.17)),
    ("Orange", (0.93, 0.55, 0.16)),
    ("Yellow", (0.95, 0.80, 0.22)),
    ("Green", (0.20, 0.58, 0.36)),
    ("Teal", (0.13, 0.55, 0.60)),
    ("Blue", (0.20, 0.38, 0.78)),
    ("Purple", (0.50, 0.28, 0.70)),
    ("Gray", (0.50, 0.52, 0.56)),
]
PANTS_COLORS = [
    ("Denim", (0.20, 0.28, 0.48)),
    ("Black", (0.10, 0.10, 0.12)),
    ("Khaki", (0.68, 0.60, 0.42)),
    ("Gray", (0.40, 0.42, 0.46)),
    ("Navy", (0.10, 0.14, 0.28)),
    ("Olive", (0.35, 0.40, 0.22)),
]
SHOE_COLORS = [
    ("White", (0.92, 0.92, 0.92)),
    ("Black", (0.08, 0.08, 0.09)),
    ("Brown", (0.36, 0.22, 0.12)),
    ("Red", (0.75, 0.15, 0.15)),
]
BACKGROUNDS = [
    ("Studio", ((0.86, 0.88, 0.92), (0.62, 0.66, 0.74))),
    ("Sunset", ((0.98, 0.80, 0.62), (0.55, 0.36, 0.55))),
    ("Mint", ((0.82, 0.94, 0.90), (0.48, 0.70, 0.68))),
    ("Night", ((0.20, 0.22, 0.34), (0.05, 0.06, 0.12))),
]

OPTIONS = [
    ("skin", "Skin tone", SKIN_TONES),
    ("face", "Face shape", [("Round", 0.12), ("Oval", 0.28), ("Pointed", 0.42)]),
    ("nose", "Nose", [("Small", 0.8), ("Medium", 1.0), ("Large", 1.3)]),
    ("mouth", "Mouth", [("Neutral", (0.0, 1.0)), ("Smile", (0.6, 1.0)),
                        ("Big smile", (1.0, 1.15)), ("Small", (0.1, 0.8))]),
    ("eyes", "Eye color", EYE_COLORS),
    ("eyesize", "Eye size", [("Small", 0.85), ("Medium", 1.0), ("Large", 1.2)]),
    ("brows", "Eyebrows", [("Thin", 0.7), ("Normal", 1.0), ("Thick", 1.5)]),
    ("hair", "Hair style", [("Bald", "bald"), ("Buzz", "buzz"), ("Short", "short"),
                            ("Long", "long"), ("Bob", "bob"), ("Bun", "bun"),
                            ("Ponytail", "ponytail"), ("Quiff", "quiff"),
                            ("Mohawk", "mohawk"), ("Afro", "afro")]),
    ("haircolor", "Hair color", HAIR_COLORS),
    ("facial", "Facial hair", [("None", "none"), ("Mustache", "mustache"), ("Beard", "beard")]),
    ("glasses", "Glasses", [("None", "none"), ("Round", "round"), ("Square", "square"),
                            ("Sunglasses", "sun")]),
    ("hat", "Hat", [("None", "none"), ("Beanie", "beanie"), ("Cap", "cap")]),
    ("hatcolor", "Hat color", CLOTH_COLORS),
    ("build", "Build", [("Slim", 0.9), ("Average", 1.0), ("Broad", 1.12)]),
    ("top", "Top", [("T-shirt", "tee"), ("Long sleeve", "long"), ("Tank top", "tank"),
                       ("Hoodie", "hoodie"), ("Jacket", "jacket")]),
    ("topcolor", "Top color", CLOTH_COLORS),
    ("pants", "Bottoms", [("Jeans", "jeans"), ("Shorts", "shorts"), ("Skirt", "skirt")]),
    ("pantscolor", "Bottoms color", PANTS_COLORS),
    ("shoes", "Shoes", SHOE_COLORS),
    ("bg", "Background", BACKGROUNDS),
]
COLOR_KEYS = {"skin", "eyes", "haircolor", "hatcolor", "topcolor", "pantscolor", "shoes"}
DEFAULT_STATE = {"skin": 1, "face": 1, "nose": 1, "mouth": 1, "eyes": 0, "eyesize": 1,
                 "brows": 1, "hair": 2, "haircolor": 2, "facial": 0, "glasses": 0,
                 "hat": 0, "hatcolor": 2,
                 "build": 1, "top": 0, "topcolor": 7, "pants": 0, "pantscolor": 0,
                 "shoes": 0, "bg": 0}

VIEWS = {"face": (-0.02, 3.3), "bust": (-0.8, 6.5), "full": (-3.4, 15.0)}
VIEW_ORDER = ["bust", "face", "full"]

BODY_LIFT = 0.30
MOUTH_Y = -0.36
EYE_X, EYE_Y = 0.20, 0.03


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
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


def catmull(keys, m, frac=1.0):
    """Sample a Catmull-Rom spline through `keys` (K, D) at m points."""
    keys = np.asarray(keys, float)
    k = len(keys)
    t = np.linspace(0, (k - 1) * frac, m)
    i = np.minimum(t.astype(int), k - 2)
    u = (t - i)[:, None]
    p0 = keys[np.maximum(i - 1, 0)]
    p1, p2 = keys[i], keys[i + 1]
    p3 = keys[np.minimum(i + 2, k - 1)]
    return 0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


def mix(a, b, t):
    return tuple(np.asarray(a) * (1 - t) + np.asarray(b) * t)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
class Mesh:
    """Triangle mesh with smooth normals, drawn with vertex arrays.

    `ref` is a point (or per-vertex points) inside the shape; triangles are
    wound so their normals point away from it, which keeps lighting correct.
    """

    def __init__(self, verts, faces, color, ref, shine=20.0, spec=0.08,
                 colors=None, per_face=True, thin=False):
        v = np.asarray(verts, float).reshape(-1, 3)
        f = np.asarray(faces, np.int64).reshape(-1, 3)
        ref = np.broadcast_to(np.asarray(ref, float), v.shape)
        tri = v[f]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        outward = np.einsum("ij,ij->i", fn, tri.mean(1) - ref[f].mean(1))
        flip = outward < 0 if per_face else np.full(len(f), outward.sum() < 0)
        f = f.copy()
        f[flip] = f[flip][:, [0, 2, 1]]
        fn[flip] *= -1
        n = np.zeros_like(v)
        for a in range(3):
            for k in range(3):
                n[:, a] += np.bincount(f[:, k], weights=fn[:, a], minlength=len(v))
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.v = np.ascontiguousarray(v, np.float32)
        self.n = np.ascontiguousarray(n, np.float32)
        self.f = np.ascontiguousarray(f.ravel(), np.uint32)
        self.color = tuple(float(c) for c in color)
        self.colors = None if colors is None else np.ascontiguousarray(
            np.clip(colors, 0, 1).reshape(len(v), -1), np.float32)
        self.shine, self.spec = shine, spec
        self.outline, self.thin = False, thin
        self.ov, self.ow = None, 0.0

    def outlined(self):
        """Mark for the inked-outline pass (a darkened, inflated back-face shell)."""
        self.outline = not self.thin
        return self

    def outline_verts(self, width):
        if self.ov is None or abs(self.ow - width) > 0.05 * width:
            self.ov = np.ascontiguousarray(self.v + self.n * width, np.float32)
            self.ow = width
        return self.ov

    def draw(self):
        glVertexPointer(3, GL_FLOAT, 0, self.v)
        glNormalPointer(GL_FLOAT, 0, self.n)
        if self.colors is not None:
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(self.colors.shape[1], GL_FLOAT, 0, self.colors)
        else:
            glColor3f(*self.color)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (self.spec, self.spec, self.spec, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, self.shine)
        glDrawElements(GL_TRIANGLES, len(self.f), GL_UNSIGNED_INT, self.f)
        if self.colors is not None:
            glDisableClientState(GL_COLOR_ARRAY)


def grid_faces(nu, nv, wrap=True):
    i = np.arange(nu if wrap else nu - 1)
    j = np.arange(nv - 1)
    ii, jj = np.meshgrid(i, j)
    i1 = (ii + 1) % nu
    a, b = jj * nu + ii, jj * nu + i1
    c, d = (jj + 1) * nu + ii, (jj + 1) * nu + i1
    return np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                           np.stack([b, d, c], -1).reshape(-1, 3)])


def unit_sphere_z(nu, nv, theta_max=math.pi):
    th = np.linspace(0, theta_max, nv)[:, None]
    ph = np.linspace(0, 2 * math.pi, nu, endpoint=False)[None, :]
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                     np.cos(th) * np.ones_like(ph)], -1).reshape(-1, 3)


def ellipsoid(center, radii, color, rot=None, detail=24, **kw):
    p = unit_sphere_z(detail, detail // 2 + 1) * np.asarray(radii, float)
    if rot is not None:
        p = p @ rot.T
    center = np.asarray(center, float)
    return Mesh(p + center, grid_faces(detail, detail // 2 + 1), color, ref=center, **kw)


def cap(center, radius, direction, half_angle, color, detail=16, **kw):
    """Spherical cap of angular radius `half_angle` around `direction`."""
    p = unit_sphere_z(detail * 2, detail, half_angle) @ rot_to(direction).T * radius
    center = np.asarray(center, float)
    return Mesh(p + center, grid_faces(detail * 2, detail), color, ref=center, **kw)


def ring_mesh(ring, axis, color, cap_ends=True, **kw):
    """Mesh from a stack of rings (n, sides, 3) around axis points (n, 3)."""
    n, sides, _ = ring.shape
    verts = [ring.reshape(-1, 3)]
    refs = [np.repeat(axis, sides, axis=0)]
    faces = [grid_faces(sides, n)]
    if cap_ends:
        i = np.arange(sides)
        for end, inward in ((0, 1), (n - 1, n - 2)):
            base = sum(len(x) for x in verts)
            verts += [ring[end], axis[end][None]]
            refs += [np.repeat(axis[inward][None], sides + 1, axis=0)]
            faces.append(np.stack([np.full(sides, base + sides), base + i,
                                   base + (i + 1) % sides], -1))
    return Mesh(np.concatenate(verts), np.concatenate(faces), color,
                ref=np.concatenate(refs), **kw)


def tube(points, radii, color, sides=14, **kw):
    """Round tube along a polyline with a per-point radius."""
    p = np.asarray(points, float)
    r = np.asarray(radii, float)
    t = unit(np.gradient(p, axis=0))
    seed = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    nrm = np.zeros_like(p)
    nrm[0] = unit(seed - t[0] * (seed @ t[0]))
    for i in range(1, len(p)):
        nrm[i] = unit(nrm[i - 1] - t[i] * (nrm[i - 1] @ t[i]))
    binorm = np.cross(t, nrm)
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring = p[:, None] + r[:, None, None] * (np.cos(ang)[None, :, None] * nrm[:, None]
                                            + np.sin(ang)[None, :, None] * binorm[:, None])
    return ring_mesh(ring, p, color, **kw)


def loft(ys, a, b, color, sides=36, power=2.4, **kw):
    """Elliptic cross-sections stacked along y (half-widths a and b)."""
    ph = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ex = np.sign(np.cos(ph)) * np.abs(np.cos(ph)) ** (2 / power)
    ez = np.sign(np.sin(ph)) * np.abs(np.sin(ph)) ** (2 / power)
    ys = np.asarray(ys, float)
    ring = np.stack([a[:, None] * ex[None], np.repeat(ys[:, None], sides, 1),
                     b[:, None] * ez[None]], -1)
    axis = np.stack([np.zeros_like(ys), ys, np.zeros_like(ys)], -1)
    return ring_mesh(ring, axis, color, **kw)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
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
        d += 0.03 * nose * pair(0.08, -0.23, 0.04, 0.035)         # nostrils
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
                         per_face=False, shine=15, spec=0.05)
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


# ---------------------------------------------------------------------------
# Face details
# ---------------------------------------------------------------------------
DARK = (0.10, 0.07, 0.07)


def eye_meshes(head, side, size, iris, skin):
    r = 0.095 * size
    x = side * EYE_X
    c = np.array([x, EYE_Y, head.z_at(x, EYE_Y) - 0.45 * r])
    fwd = (side * 0.05, 0.0, 1.0)
    out = [ellipsoid(c, (r,) * 3, (0.96, 0.95, 0.94), detail=36, shine=70, spec=0.4),
           cap(c, r * 1.012, fwd, 0.62, np.array(iris) * 0.45, detail=18, shine=70, spec=0.4),
           cap(c, r * 1.022, fwd, 0.52, iris, detail=18, shine=70, spec=0.4),
           cap(c, r * 1.032, fwd, 0.24, (0.02, 0.02, 0.03), detail=12, shine=70, spec=0.4),
           ellipsoid(c + r * unit((0.3, 0.38, 0.9)), (0.15 * r,) * 3, (1, 1, 1),
                     detail=10, shine=90, spec=0.6)]
    # Upper eyelid: a dome tilted back so its edge arches over the eye.
    tilt, half = math.radians(22), math.radians(90)
    direction = np.array([0.0, math.cos(tilt), -math.sin(tilt)])
    out.append(cap(c, r * 1.11, direction, half, skin, detail=22, shine=25, spec=0.12))
    # Dark lash line along the lid edge.
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(direction, e1)
    psi = np.linspace(0, 2 * math.pi, 240)[:, None]
    edge = direction * math.cos(half) + (e1 * np.cos(psi) + e2 * np.sin(psi)) * math.sin(half)
    edge = edge[edge[:, 2] > 0.25]
    edge = edge[np.argsort(edge[:, 0])][::4]
    out.append(tube(c + edge * r * 1.113, np.full(len(edge), 0.0035), DARK, sides=6))
    return out


def brow_mesh(head, side, thickness, color):
    t = np.linspace(0, 1, 14)
    xs = side * (0.075 + 0.20 * t)
    ys = 0.185 + 0.045 * np.sin(t * math.pi * 0.85) - 0.02 * t
    radii = 0.016 * thickness * (1.1 - 0.65 * t) * np.minimum(1, 0.3 + 6 * np.minimum(t, 1 - t))
    return tube(head.on_face(xs, ys, 0.004), radii, color, sides=8, shine=10, spec=0.02)


def lip_meshes(head, smile, width, skin):
    w = 0.14 * width
    xs = np.linspace(-w, w, 33)
    u = xs / w
    corner = smile * 0.055 * u ** 2
    fall = np.sqrt(np.clip(1 - u ** 4, 0, 1))
    lip = mix(skin, (0.72, 0.30, 0.34), 0.38)
    line = tube(head.on_face(xs, MOUTH_Y + corner, 0.002), 0.005 + 0.003 * fall,
                (0.25, 0.10, 0.10), sides=6)
    upper = tube(head.on_face(xs, MOUTH_Y + corner + 0.016 * fall, 0.0),
                 0.003 + 0.016 * fall, lip, sides=10, shine=35, spec=0.2)
    lower = tube(head.on_face(xs, MOUTH_Y + corner - 0.018 * fall, 0.0),
                 0.003 + 0.022 * fall, lip, sides=10, shine=35, spec=0.2)
    return [line, upper, lower]


def glasses_meshes(head, style):
    frame = (0.10, 0.10, 0.12)
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
        out.append(tube(pts, np.full(len(pts), 0.011), frame, sides=8, cap_ends=False,
                        shine=60, spec=0.3))
        if style == "sun":
            out.append(ellipsoid((s * EYE_X, EYE_Y + 0.01, zg), (rx - 0.005, ry - 0.005, 0.006),
                                 (0.05, 0.06, 0.09), detail=24, shine=90, spec=0.7))
        temple = catmull([(s * (EYE_X + rx), EYE_Y + 0.01, zg), (s * 0.43, EYE_Y + 0.02, 0.35),
                          (s * 0.53, EYE_Y, 0.10), (s * 0.54, 0.0, -0.03)], 12)
        out.append(tube(temple, np.full(12, 0.010), frame, sides=8, shine=60, spec=0.3))
    bridge = catmull([(-(EYE_X - rx), EYE_Y + 0.04, zg), (0, EYE_Y + 0.055, zg + 0.015),
                      (EYE_X - rx, EYE_Y + 0.04, zg)], 10)
    out.append(tube(bridge, np.full(10, 0.010), frame, sides=8, shine=60, spec=0.3))
    return out


# ---------------------------------------------------------------------------
# Hair
# ---------------------------------------------------------------------------
HAIRLINES = {  # hairline height at azimuth 0, 40, 80, 110, 150, 180 degrees from the front
    "short": [0.36, 0.30, 0.20, 0.10, -0.05, -0.28],
    "long": [0.36, 0.30, 0.12, -0.02, -0.20, -0.35],
    "afro": [0.34, 0.30, 0.22, 0.15, 0.00, -0.25],
}
for _name in ("buzz", "bun", "ponytail", "quiff", "mohawk"):
    HAIRLINES[_name] = HAIRLINES["short"]
HAIRLINES["bob"] = HAIRLINES["long"]


def hair_meshes(head, style, color):
    if style == "bald":
        return []
    theta = np.abs(np.arctan2(head.X, head.Z))
    line = np.interp(theta, np.radians([0, 40, 80, 110, 150, 180]), HAIRLINES[style])
    margin = head.Y - line
    top = np.clip(head.Y, 0, 1)
    if style == "afro":
        return [afro_mesh(color)]
    thick = 0.014 if style in ("buzz", "mohawk") else 0.05 + 0.03 * top ** 2
    out = [head.shell(margin, thick, color, 0.05, shine=40, spec=0.25)]
    hair = dict(shine=40, spec=0.25)
    if style == "bun":
        out.append(ellipsoid((0, 0.66, -0.22), (0.2, 0.2, 0.2), color, **hair))
    elif style == "long":
        out.append(hair_curtain(color, 1.35))
    elif style == "bob":
        out.append(hair_curtain(color, 0.7))
    elif style == "ponytail":
        path = catmull([(0, 0.32, -0.50), (0, 0.18, -0.74), (0, -0.25, -0.80), (0, -0.78, -0.70)], 20)
        radii = catmull([(0.07,), (0.11,), (0.09,), (0.025,)], 20)[:, 0]
        out.append(tube(path, radii, color, **hair))
        out.append(ellipsoid((0, 0.30, -0.52), (0.09, 0.09, 0.06), (0.85, 0.25, 0.35), thin=True))
    elif style == "quiff":
        out.append(ellipsoid((0, 0.68, 0.28), (0.30, 0.17, 0.32), color,
                             rot=rot_x(math.radians(-15)), detail=28, **hair))
    elif style == "mohawk":
        ridge = catmull([(0, 0.50, 0.38), (0, 0.72, 0.12), (0, 0.70, -0.25), (0, 0.40, -0.55)], 24)
        radii = catmull([(0.05,), (0.10,), (0.10,), (0.05,)], 24)[:, 0]
        out.append(tube(ridge, radii, color, **hair))
    return out


def hat_meshes(head, kind, color):
    if kind == "none":
        return []
    theta = np.abs(np.arctan2(head.X, head.Z))
    keys = np.radians([0, 60, 110, 180])
    top = np.clip(head.Y, 0, 1)
    fabric = dict(shine=8, spec=0.03)
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
    return Mesh(verts, grid_faces(64, 33), color, ref=center, shine=30, spec=0.15)


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
    return Mesh(p, grid_faces(nu, nv, wrap=False), color, ref=ref, shine=40, spec=0.25)


def facial_hair_meshes(head, style, color):
    if style == "none":
        return []
    out = []
    xs = np.linspace(0, 1, 12)
    for s in (-1, 1):
        x = s * 0.17 * xs
        y = -0.285 - 0.02 * xs
        radii = 0.030 * (1 - 0.6 * xs) * np.minimum(1, 0.3 + 6 * (1 - xs))
        out.append(tube(head.on_face(x, y, 0.012), radii, color, sides=8, shine=30, spec=0.15))
    if style == "beard":
        theta = np.abs(np.arctan2(head.X, head.Z))
        jaw = np.minimum(-0.22 - head.Y, (np.radians(85) - theta) * 0.5)
        mouth = np.sqrt((head.X / 0.19) ** 2 + ((head.Y - MOUTH_Y) / 0.065) ** 2) - 1
        margin = np.minimum(jaw, mouth * 0.15)
        margin = np.where(head.Y > -0.22, -1.0, margin)
        out.append(head.shell(margin, 0.04, color, 0.05, shine=30, spec=0.15))
    return out


# ---------------------------------------------------------------------------
# Body and clothes
# ---------------------------------------------------------------------------
TORSO_KEYS = np.array([  # y, half-width, half-depth
    (-1.02, 0.30, 0.27), (-1.15, 0.72, 0.34), (-1.32, 0.90, 0.40), (-1.80, 0.86, 0.42),
    (-2.40, 0.72, 0.37), (-3.00, 0.66, 0.33), (-3.60, 0.74, 0.37), (-4.15, 0.76, 0.38),
    (-4.30, 0.55, 0.30)])


def torso_section(y0, y1, build, grow, n, color, **kw):
    dense = catmull(TORSO_KEYS, 200)
    ys = np.linspace(y0, y1, n)
    a = np.interp(-ys, -dense[:, 0], dense[:, 1]) * build + grow
    b = np.interp(-ys, -dense[:, 0], dense[:, 2]) + grow
    return loft(ys, a, b, color, **kw)


def hand_meshes(wrist, side, skin):
    """Palm, four fingers and a thumb, hanging down with the palm facing the body."""
    wx, wy, wz = wrist
    k = 1.25  # hand scale, so the hands match the forearm width
    out = [ellipsoid((wx, wy - 0.10 * k, wz), (0.045 * k, 0.095 * k, 0.075 * k), skin, detail=16)]
    for dz, length in zip((-0.05, -0.017, 0.017, 0.05), (0.12, 0.14, 0.13, 0.10)):
        t = np.linspace(0, 1, 5)
        path = np.stack([wx - side * 0.03 * k * t ** 2, wy - 0.17 * k - length * k * t,
                         np.full(5, wz + dz * k)], -1)
        out.append(tube(path, np.linspace(0.02, 0.013, 5) * k, skin, sides=8, thin=True))
    thumb = np.array([(wx - side * 0.01, wy - 0.06 * k, wz + 0.06 * k),
                      (wx - side * 0.02 * k, wy - 0.11 * k, wz + 0.10 * k),
                      (wx - side * 0.03 * k, wy - 0.17 * k, wz + 0.115 * k)])
    out.append(tube(catmull(thumb, 6), np.linspace(0.024, 0.015, 6) * k, skin, sides=8, thin=True))
    return out


def body_meshes(v):
    skin, top, pants = np.array(v["skin"]), v["top"], v["pants"]
    tcol, pcol, scol = v["topcolor"], v["pantscolor"], v["shoes"]
    b = v["build"]
    rb = math.sqrt(b)
    out = []
    neck = catmull([(0, -0.28, 0.02), (0, -0.55, 0.03), (0, -0.85, 0.0)], 10)
    out.append(tube(neck, np.linspace(0.235, 0.27, 10), skin))
    bulk = {"hoodie": 0.07, "jacket": 0.06}.get(top, 0.035)
    hem = -3.72 if bulk < 0.05 else -3.85
    cloth = dict(shine=12, spec=0.04)
    out.append(torso_section(-1.02, -4.28, b, 0.0, 40, skin))
    out.append(torso_section(-1.02, hem, b, bulk, 30, tcol, cap_ends=False, **cloth))
    if top == "hoodie":
        phi = np.radians(np.linspace(-115, 115, 24))
        hood = np.stack([0.40 * np.sin(phi), -1.05 + 0.04 * np.cos(phi), 0.02 - 0.32 * np.cos(phi)], -1)
        out.append(tube(hood, np.full(len(phi), 0.11), tcol, sides=12, **cloth))
        for s in (-1, 1):
            cord = catmull([(s * 0.16, -1.10, 0.30), (s * 0.15, -1.5, 0.42), (s * 0.13, -1.95, 0.45)], 8)
            out.append(tube(cord, np.full(8, 0.013), (0.92, 0.92, 0.92), sides=6, thin=True))
    if top == "jacket":
        ys = np.linspace(-1.08, hem + 0.03, 24)
        dense = catmull(TORSO_KEYS, 200)
        zf = np.interp(-ys, -dense[:, 0], dense[:, 2]) + bulk + 0.004
        out.append(tube(np.column_stack([np.zeros_like(ys), ys, zf]), np.full(len(ys), 0.011),
                        (0.82, 0.82, 0.85), sides=6, shine=60, spec=0.4, thin=True))
        phi = np.linspace(0, 2 * math.pi, 32)
        collar = np.stack([0.34 * np.sin(phi), np.full(len(phi), -1.05), 0.30 * np.cos(phi)], -1)
        out.append(tube(collar, np.full(len(phi), 0.06), mix(tcol, (0, 0, 0), 0.15), sides=10,
                        cap_ends=False, **cloth))
    if pants == "skirt":
        ys = np.linspace(-3.2, -5.05, 16)
        t = (-ys - 3.2) / 1.85
        out.append(loft(ys, (0.70 * b + 0.10) + 0.45 * t ** 1.3, 0.44 + 0.30 * t ** 1.3, pcol,
                        cap_ends=False, **cloth))
    else:
        out.append(torso_section(-3.25, -4.30, b, 0.03, 12, pcol, cap_ends=False, **cloth))

    for s in (-1, 1):
        arm_keys = np.array([(s * 0.98 * b, -1.36, 0.0, 0.19 * rb),
                             (s * 1.10 * b, -2.85, 0.05, 0.155 * rb),
                             (s * 1.16 * b, -4.15, 0.10, 0.115 * rb)])
        arm = catmull(arm_keys, 30)
        out.append(tube(arm[:, :3], arm[:, 3], skin))
        wrist = arm[-1, :3]
        out += hand_meshes(wrist, s, skin)
        if top == "tank":
            out.append(ellipsoid(arm_keys[0, :3], (0.19 * rb,) * 3, skin))
        else:
            frac = 0.42 if top == "tee" else 0.985
            grow = bulk - 0.005
            sleeve = catmull(arm_keys, max(6, int(30 * frac)), frac)
            out.append(tube(sleeve[:, :3], sleeve[:, 3] + grow, tcol, cap_ends=False, **cloth))
            out.append(ellipsoid(arm_keys[0, :3], (0.19 * rb + grow,) * 3, tcol, **cloth))

        leg_keys = np.array([(s * 0.36 * b, -3.95, 0.0, 0.30 * rb),
                             (s * 0.40 * b, -5.60, 0.03, 0.23 * rb),
                             (s * 0.38 * b, -7.25, 0.0, 0.15 * rb)])
        leg = catmull(leg_keys, 36)
        out.append(tube(leg[:, :3], leg[:, 3], skin))
        if pants != "skirt":
            frac = 0.985 if pants == "jeans" else 0.42
            trouser = catmull(leg_keys, max(6, int(36 * frac)), frac)
            out.append(tube(trouser[:, :3], trouser[:, 3] + 0.035, pcol, cap_ends=False, **cloth))
        out.append(ellipsoid((s * 0.38 * b, -7.50, 0.16), (0.15 * rb + 0.02, 0.12, 0.33), scol,
                             shine=30, spec=0.15))
        out.append(ellipsoid((s * 0.38 * b, -7.595, 0.16), (0.155 * rb + 0.025, 0.035, 0.335),
                             (0.93, 0.93, 0.93), shine=20, spec=0.1))
    return out


@functools.lru_cache(maxsize=4)
def cached_head(skin, jaw, nose):
    return Head(skin, jaw, nose)


def build_avatar(v):
    """Turn resolved option values into a list of meshes."""
    skin, hair = v["skin"], v["haircolor"]
    smile, mouth_w = v["mouth"]
    head = cached_head(skin, v["face"], v["nose"])
    meshes = [head.mesh]
    for s in (-1, 1):
        meshes += eye_meshes(head, s, v["eyesize"], v["eyes"], skin)
        meshes.append(brow_mesh(head, s, v["brows"], mix(hair, DARK, 0.25)))
        meshes.append(ellipsoid((s * 0.485, -0.09, -0.02), (0.05, 0.11, 0.075), skin,
                                rot=rot_y(-s * 0.35), detail=18, shine=25, spec=0.1))
    meshes += lip_meshes(head, smile, mouth_w, skin)
    style, hat = v["hair"], v["hat"]
    if style == "afro":
        hat = "none"
    elif hat != "none" and style in ("quiff", "bun", "mohawk"):
        style = "short"  # keep tall styles from poking through the hat
    hair_parts = hair_meshes(head, style, hair) + hat_meshes(head, hat, v["hatcolor"])
    meshes += hair_parts
    meshes += facial_hair_meshes(head, v["facial"], hair)
    if v["glasses"] != "none":
        meshes += glasses_meshes(head, v["glasses"])
    body = body_meshes(v)
    for m in body[1:]:  # everything but the neck is raised so the shoulders sit closer to the head
        m.v[:, 1] += BODY_LIFT
    for m in [head.mesh] + hair_parts + body:
        m.outlined()
    return meshes + body


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def resolve(state):
    return {key: opts[state[key]][1] for key, _, opts in OPTIONS}


def randomize(state):
    for key, _, opts in OPTIONS:
        if key != "bg":
            state[key] = random.randrange(len(opts))


def set_by_name(state, key, name):
    for k, _, opts in OPTIONS:
        if k == key:
            for i, (n, _) in enumerate(opts):
                if n.lower() == name.lower():
                    state[k] = i
                    return
            raise SystemExit(f"Unknown value '{name}' for '{key}'. "
                             f"Choices: {', '.join(n for n, _ in opts)}")
    raise SystemExit(f"Unknown option '{key}'. Options: {', '.join(k for k, _, _ in OPTIONS)}")


def save_state(state, path):
    data = {key: opts[state[key]][0] for key, _, opts in OPTIONS}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_state(state, path):
    with open(path) as f:
        data = json.load(f)
    for key, name in data.items():
        try:
            set_by_name(state, key, name)
        except SystemExit:
            pass  # ignore options that no longer exist


def export_obj(meshes, path):
    """Write the avatar as Wavefront OBJ + MTL (one material per mesh)."""
    base = os.path.splitext(path)[0]
    with open(base + ".mtl", "w") as mtl:
        for i, m in enumerate(meshes):
            mtl.write(f"newmtl m{i}\nKd {m.color[0]:.3f} {m.color[1]:.3f} {m.color[2]:.3f}\n\n")
    with open(path, "w") as obj:
        obj.write(f"mtllib {os.path.basename(base)}.mtl\n")
        offset = 1
        for i, m in enumerate(meshes):
            obj.write(f"o part{i}\nusemtl m{i}\n")
            obj.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in m.v)
            obj.writelines(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in m.n)
            tri = m.f.reshape(-1, 3) + offset
            obj.writelines(f"f {a}//{a} {b}//{b} {c}//{c}\n" for a, b, c in tri)
            offset += len(m.v)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
class Camera:
    def __init__(self, view):
        self.ty, self.dist = VIEWS[view]
        self.target_ty, self.target_dist = self.ty, self.dist
        self.yaw, self.pitch = 20.0, 4.0

    def set_view(self, view):
        self.target_ty, self.target_dist = VIEWS[view]

    def update(self):
        self.ty += (self.target_ty - self.ty) * 0.15
        self.dist += (self.target_dist - self.dist) * 0.15


def setup_gl():
    glEnable(GL_MULTISAMPLE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHTING)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.30, 0.30, 0.34, 1.0))
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.96, 0.90, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
    glEnable(GL_LIGHT2)  # cool rim light from behind
    glLightfv(GL_LIGHT2, GL_DIFFUSE, (0.35, 0.42, 0.60, 1.0))
    glLightfv(GL_LIGHT2, GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_DIFFUSE, (0.35, 0.40, 0.50, 1.0))
    glLightfv(GL_LIGHT1, GL_SPECULAR, (0.0, 0.0, 0.0, 1.0))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)


def ortho(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, width, height, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def draw_background(top, bottom):
    ortho(1, 1)
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glBegin(GL_QUADS)
    glColor3f(*top)
    glVertex2f(0, 0)
    glVertex2f(1, 0)
    glColor3f(*bottom)
    glVertex2f(1, 1)
    glVertex2f(0, 1)
    glEnd()
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)


def draw_shadow():
    glDisable(GL_LIGHTING)
    glDepthMask(GL_FALSE)
    glBegin(GL_TRIANGLE_FAN)
    glColor4f(0, 0, 0, 0.28)
    glVertex3f(0, -7.34, 0.05)
    glColor4f(0, 0, 0, 0.0)
    for k in range(33):
        a = 2 * math.pi * k / 32
        glVertex3f(2.1 * math.cos(a), -7.34, 0.05 + 1.3 * math.sin(a))
    glEnd()
    glDepthMask(GL_TRUE)
    glEnable(GL_LIGHTING)


def draw_outlines(meshes, width):
    """Ink outline: back faces pushed outwards, drawn in a dark tone of the mesh color."""
    glDisable(GL_LIGHTING)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_FRONT)
    for m in meshes:
        if m.outline:
            glColor3f(*(c * 0.3 for c in m.color))
            glVertexPointer(3, GL_FLOAT, 0, m.outline_verts(width))
            glDrawElements(GL_TRIANGLES, len(m.f), GL_UNSIGNED_INT, m.f)
    glDisable(GL_CULL_FACE)
    glEnable(GL_LIGHTING)


def draw_scene(meshes, cam, bg):
    glViewport(0, 0, VIEW_W, H)
    draw_background(*bg)
    glClear(GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(35, VIEW_W / H, 0.1, 100)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glLightfv(GL_LIGHT0, GL_POSITION, (-0.4, 0.7, 1.0, 0.0))
    glLightfv(GL_LIGHT1, GL_POSITION, (0.8, 0.1, 0.6, 0.0))
    glLightfv(GL_LIGHT2, GL_POSITION, (0.2, 0.5, -1.0, 0.0))
    glTranslatef(0, 0, -cam.dist)
    glRotatef(cam.pitch, 1, 0, 0)
    glRotatef(cam.yaw, 0, 1, 0)
    glTranslatef(0, -cam.ty, 0)
    draw_shadow()
    draw_outlines(meshes, 0.0022 * cam.dist)
    for m in meshes:
        m.draw()


def grab_viewport():
    data = glReadPixels(0, 0, VIEW_W, H, GL_RGB, GL_UNSIGNED_BYTE)
    surf = pygame.image.frombuffer(bytes(data), (VIEW_W, H), "RGB")
    return pygame.transform.flip(surf, False, True)


# ---------------------------------------------------------------------------
# Side panel (pygame surface uploaded as a texture)
# ---------------------------------------------------------------------------
ROW0, ROW_H = 62, 25


class Panel:
    def __init__(self):
        self.font = pygame.font.SysFont("segoeui,arial,helvetica", 16)
        self.bold = pygame.font.SysFont("segoeui,arial,helvetica", 22, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial,helvetica", 13)
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def render(self, state, selected, message):
        s = pygame.Surface((PANEL_W, H), pygame.SRCALPHA)
        s.fill((26, 28, 36, 255))
        s.blit(self.bold.render("Avatar Creator", True, (240, 240, 245)), (16, 14))
        for row, (key, label, opts) in enumerate(OPTIONS):
            y = ROW0 + row * ROW_H
            if row == selected:
                pygame.draw.rect(s, (52, 58, 78), (6, y - 2, PANEL_W - 12, ROW_H - 2), border_radius=6)
            s.blit(self.font.render(label, True, (170, 176, 192)), (16, y + 1))
            name, val = opts[state[key]]
            x = 150
            s.blit(self.font.render("‹", True, (120, 128, 150)), (x, y))
            if key in COLOR_KEYS:
                pygame.draw.rect(s, [int(c * 255) for c in val], (x + 16, y + 3, 14, 14),
                                 border_radius=3)
                x += 20
            s.blit(self.font.render(name, True, (235, 236, 242)), (x + 18, y + 1))
            s.blit(self.font.render("›", True, (120, 128, 150)), (PANEL_W - 24, y))
        help_lines = ["↑/↓ select   ←/→ change   (or click)",
                      "R random    V view    Space spin",
                      "S screenshot   E export OBJ",
                      "K save   L load   Drag: orbit   Wheel: zoom"]
        y = ROW0 + len(OPTIONS) * ROW_H + 12
        for line in help_lines:
            s.blit(self.small.render(line, True, (130, 136, 156)), (16, y))
            y += 18
        if message:
            s.blit(self.small.render(message, True, (140, 210, 160)), (16, H - 26))
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, PANEL_W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     pygame.image.tostring(s, "RGBA", False))

    def draw(self):
        glViewport(0, 0, W, H)
        ortho(W, H)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        for u, v, x, y in ((0, 0, VIEW_W, 0), (1, 0, W, 0), (1, 1, W, H), (0, 1, VIEW_W, H)):
            glTexCoord2f(u, v)
            glVertex2f(x, y)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def timestamp_path(ext):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    return os.path.join(EXPORT_DIR, time.strftime("avatar_%Y%m%d_%H%M%S") + ext)


def main():
    ap = argparse.ArgumentParser(description="3D avatar creator")
    ap.add_argument("--shot", help="render one frame to this PNG and exit")
    ap.add_argument("--view", choices=VIEWS, default="bust")
    ap.add_argument("--yaw", type=float, default=20.0)
    ap.add_argument("--seed", type=int, help="randomize using this seed")
    ap.add_argument("--set", action="append", default=[], metavar="OPTION=VALUE")
    args = ap.parse_args()

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
    pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Avatar Creator")
    setup_gl()

    state = dict(DEFAULT_STATE)
    if args.seed is not None:
        random.seed(args.seed)
        randomize(state)
    for item in args.set:
        key, _, name = item.partition("=")
        set_by_name(state, key, name)

    panel = Panel()
    cam = Camera(args.view)
    cam.yaw = args.yaw
    selected, message, message_until = 0, "", 0.0
    meshes = build_avatar(resolve(state))
    dirty, spin, dragging, want_shot = True, args.shot is None, False, False
    view_idx = VIEW_ORDER.index(args.view) if args.view in VIEW_ORDER else 0
    clock = pygame.time.Clock()
    frames = 0

    def notify(text):
        nonlocal message, message_until, dirty
        message, message_until, dirty = text, time.time() + 3, True

    def change(row, step):
        nonlocal dirty, meshes
        key, _, opts = OPTIONS[row]
        state[key] = (state[key] + step) % len(opts)
        meshes = build_avatar(resolve(state))
        dirty = True

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_UP:
                    selected, dirty = (selected - 1) % len(OPTIONS), True
                elif e.key == pygame.K_DOWN:
                    selected, dirty = (selected + 1) % len(OPTIONS), True
                elif e.key == pygame.K_LEFT:
                    change(selected, -1)
                elif e.key == pygame.K_RIGHT:
                    change(selected, 1)
                elif e.key == pygame.K_r:
                    randomize(state)
                    meshes, dirty = build_avatar(resolve(state)), True
                elif e.key == pygame.K_v:
                    view_idx = (view_idx + 1) % len(VIEW_ORDER)
                    cam.set_view(VIEW_ORDER[view_idx])
                elif e.key == pygame.K_SPACE:
                    spin = not spin
                elif e.key == pygame.K_s:
                    want_shot = True
                elif e.key == pygame.K_e:
                    path = timestamp_path(".obj")
                    export_obj(meshes, path)
                    notify(f"Exported {path}")
                elif e.key == pygame.K_k:
                    save_state(state, "avatar.json")
                    notify("Saved avatar.json")
                elif e.key == pygame.K_l and os.path.exists("avatar.json"):
                    load_state(state, "avatar.json")
                    meshes = build_avatar(resolve(state))
                    notify("Loaded avatar.json")
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if e.pos[0] < VIEW_W:
                    dragging, spin = True, False
                else:
                    row = (e.pos[1] - ROW0 + 2) // ROW_H
                    if 0 <= row < len(OPTIONS):
                        selected, dirty = row, True
                        x = e.pos[0] - VIEW_W
                        if x < 170:
                            change(row, -1)
                        elif x > 260:
                            change(row, 1)
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = False
            elif e.type == pygame.MOUSEMOTION and dragging:
                cam.yaw += e.rel[0] * 0.5
                cam.pitch = max(-60, min(60, cam.pitch + e.rel[1] * 0.4))
            elif e.type == pygame.MOUSEWHEEL:
                cam.target_dist = max(1.5, min(30, cam.target_dist * 0.9 ** e.y))

        if spin:
            cam.yaw += 0.4
        cam.update()
        if message and time.time() > message_until:
            message, dirty = "", True
        if dirty:
            panel.render(state, selected, message)
            dirty = False

        bg = BACKGROUNDS[state["bg"]][1]
        draw_scene(meshes, cam, bg)
        if want_shot:
            path = timestamp_path(".png")
            pygame.image.save(grab_viewport(), path)
            notify(f"Saved {path}")
            want_shot = False
        panel.draw()
        pygame.display.flip()
        clock.tick(FPS)

        frames += 1
        if args.shot and frames >= 30:
            pygame.image.save(pygame.image.frombuffer(
                bytes(glReadPixels(0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE)), (W, H), "RGB"), args.shot)
            # glReadPixels is bottom-up; flip in place
            img = pygame.transform.flip(pygame.image.load(args.shot), False, True)
            pygame.image.save(img, args.shot)
            break
    pygame.quit()


if __name__ == "__main__":
    main()
