"""A tiny rigid-part skeleton: nodes with pivots that hold meshes."""
import numpy as np

from .mathutil import euler_matrix


class Node:
    def __init__(self, name, pivot, parent=None):
        self.name = name
        self.pivot = np.asarray(pivot, float)
        self.parent = parent
        self.children = []
        self.meshes = []
        if parent is not None:
            parent.children.append(self)

    def add(self, *meshes):
        for m in meshes:
            if isinstance(m, (list, tuple)):
                self.meshes.extend(m)
            elif m is not None:
                self.meshes.append(m)


class Rig:
    """Rest-pose geometry lives in world coordinates; a node rotates its meshes
    (and children) about its pivot. Pose values are degrees (rx, ry, rz)."""

    def __init__(self):
        self.nodes = {}
        self.root = None
        self.ground_y = -7.3
        self.mouth = None          # head.Mouth, set by the builder
        self.base_open = 0.0       # resting openness from the chosen expression
        self._mouth_key = None

    def add_node(self, name, pivot, parent=None):
        node = Node(name, pivot, self.nodes[parent] if parent else None)
        self.nodes[name] = node
        if self.root is None:
            self.root = node
        return node

    def set_mouth(self, open_=0.0, wide=0.0, press=0.0):
        """Reshape the lips (speech adds to the expression's resting openness)."""
        if self.mouth is None:
            return
        meshes = self.mouth.meshes(min(1.0, self.base_open + open_), wide, press)
        if meshes is not self._mouth_key:
            self.nodes["mouth"].meshes = meshes
            self._mouth_key = meshes

    def __getitem__(self, name):
        return self.nodes[name]

    def all_meshes(self):
        return [m for n in self.nodes.values() for m in n.meshes]

    def world_matrices(self, pose=None):
        """4x4 matrix per node for a pose (rest pose if `pose` is None)."""
        pose = pose or {}
        out = {}

        def visit(node, parent_m):
            rx, ry, rz = pose.get(node.name, (0.0, 0.0, 0.0))
            local = np.eye(4)
            local[:3, :3] = euler_matrix(rx, ry, rz)
            local[:3, 3] = node.pivot - local[:3, :3] @ node.pivot
            if node is self.root:
                local[1, 3] += pose.get("root_dy", 0.0)
            m = parent_m @ local
            out[node.name] = m
            for c in node.children:
                visit(c, m)

        visit(self.root, np.eye(4))
        return out

    def baked_meshes(self, pose=None):
        """Yield (mesh, vertices, normals) with the pose applied to the geometry."""
        mats = self.world_matrices(pose)
        for name, node in self.nodes.items():
            m = mats[name]
            for mesh in node.meshes:
                v = mesh.v @ m[:3, :3].T + m[:3, 3]
                n = mesh.n @ m[:3, :3].T
                yield mesh, v.astype(np.float32), n.astype(np.float32)
