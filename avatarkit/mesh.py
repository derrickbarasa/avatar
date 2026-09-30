"""Triangle meshes and the primitive builders (no OpenGL in here)."""
import math

import numpy as np

from .mathutil import rot_to, unit


class Mesh:
    """Triangle mesh with smooth normals.

    `ref` is a point (or per-vertex points) inside the shape; triangles are
    wound so their normals point away from it, which keeps lighting correct.

    Material fields used by the shader: `kind` (plain, skin, hair, cloth,
    denim, eye, glossy), `pattern` / `color2` (cloth prints), `freckle`, and
    `center` (object-space eye centre). `anim` is ("blink", pivot) for lids.
    """

    def __init__(self, verts, faces, color, ref, shine=20.0, spec=0.08,
                 colors=None, per_face=True, thin=False, kind="plain"):
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
        self.kind, self.thin = kind, thin
        self.outline, self.joint = False, False
        self.pattern, self.color2, self.freckle = 0, (1.0, 1.0, 1.0), 0.0
        self.center = (0.0, 0.0, 0.0)
        self.anim = None
        self.tangents, self.strand = None, False   # hair strands: per-vertex direction
        self.n_strands, self.faces_per_strand = 0, 0
        self.strand_aux, self.swing = None, 0.0   # per-vertex (radius, t along strand); sway amount 0..1
        self.jaw_follow = False                    # moves with the jaw when the mouth opens
        self.jaw_shift = 0.0                       # how far the head was moved (keeps the jaw region right)

    def outlined(self):
        self.outline = not (self.joint or self.strand)
        return self

    def set(self, **attrs):
        for k, val in attrs.items():
            setattr(self, k, val)
        return self


def strand_faces(mesh, fraction=1.0):
    """Face indices of the first `fraction` of a strand mesh's strands (all faces otherwise).

    Strands are stored one after another and were rooted at random points, so a prefix
    is an evenly thinned head of hair: used for level-of-detail and smaller exports.
    """
    if not mesh.strand or fraction >= 1.0 or not mesh.n_strands:
        return mesh.f
    keep = max(1, int(mesh.n_strands * fraction))
    return mesh.f[:keep * mesh.faces_per_strand * 3]


def transformed(mesh, scale, origin, shift):
    """A copy of `mesh` scaled about `origin` then moved by `shift` (the original is left alone: many
    meshes are cached and shared between avatars)."""
    o, sh = np.asarray(origin, np.float32), np.asarray(shift, np.float32)
    out = Mesh.__new__(Mesh)
    out.__dict__.update(mesh.__dict__)
    out.v = np.ascontiguousarray((mesh.v - o) * scale + o + sh, np.float32)
    if mesh.strand_aux is not None:
        aux = mesh.strand_aux.copy()
        aux[:, 0] *= scale
        out.strand_aux = aux
    move = lambda p: tuple(float(x) for x in (np.asarray(p, np.float32) - o) * scale + o + sh)
    out.center = move(mesh.center)
    if mesh.anim:
        out.anim = (mesh.anim[0], move(mesh.anim[1]))
    out.jaw_shift = mesh.jaw_shift + float(sh[1])
    return out


def merge(meshes, color=None):
    """Concatenate meshes that share one material into a single mesh."""
    meshes = [m for m in meshes if m is not None]
    out = meshes[0]
    verts = np.concatenate([m.v for m in meshes])
    norms = np.concatenate([m.n for m in meshes])
    offs = np.cumsum([0] + [len(m.v) for m in meshes[:-1]])
    faces = np.concatenate([m.f + o for m, o in zip(meshes, offs)]).astype(np.uint32)
    res = Mesh.__new__(Mesh)
    res.__dict__.update(out.__dict__)
    res.v, res.n, res.f = verts, norms, faces
    if any(m.colors is not None for m in meshes):
        def cols(m):
            if m.colors is not None:
                c = m.colors
                return c if c.shape[1] == 4 else np.column_stack([c, np.ones(len(c))])
            return np.column_stack([np.tile(m.color, (len(m.v), 1)), np.ones(len(m.v))])
        res.colors = np.ascontiguousarray(np.concatenate([cols(m) for m in meshes]), np.float32)
    if color is not None:
        res.color = tuple(color)
    return res


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


def loft(ys, a, bf, bb, color, sides=36, power=2.4, **kw):
    """Cross-sections stacked along y: half-width a, front depth bf, back depth bb."""
    ph = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ex = np.sign(np.cos(ph)) * np.abs(np.cos(ph)) ** (2 / power)
    ez = np.sign(np.sin(ph)) * np.abs(np.sin(ph)) ** (2 / power)
    ys = np.asarray(ys, float)
    depth = np.where(ez[None] > 0, np.asarray(bf)[:, None], np.asarray(bb)[:, None])
    ring = np.stack([np.asarray(a)[:, None] * ex[None], np.repeat(ys[:, None], sides, 1),
                     depth * ez[None]], -1)
    axis = np.stack([np.zeros_like(ys), ys, np.zeros_like(ys)], -1)
    return ring_mesh(ring, axis, color, **kw)
