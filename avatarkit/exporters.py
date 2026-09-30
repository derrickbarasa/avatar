"""Export the avatar as Wavefront OBJ (pose baked in) or glTF binary (.glb, with rig nodes)."""
import json
import os
import struct

import numpy as np

from .mathutil import euler_matrix, matrix_to_quat
from .mesh import strand_faces

def compact(verts, normals, faces, colors=None):
    """Drop vertices no triangle uses (hair/hat shells keep the whole head grid).

    Unused vertices carry zero-length normals, which strict glTF loaders reject.
    """
    used = np.unique(faces)
    remap = np.zeros(len(verts), np.uint32)
    remap[used] = np.arange(len(used), dtype=np.uint32)
    return (verts[used], normals[used], remap[faces].astype(np.uint32),
            None if colors is None else colors[used])


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------


def export_obj(rig, path, pose=None, strand_fraction=1.0):
    """Write OBJ + MTL, one material per mesh, with the given pose applied."""
    base = os.path.splitext(path)[0]
    parts = list(rig.baked_meshes(pose))
    with open(base + ".mtl", "w") as mtl:
        for i, (m, _, _) in enumerate(parts):
            mtl.write(f"newmtl m{i}\nKd {m.color[0]:.3f} {m.color[1]:.3f} {m.color[2]:.3f}\n\n")
    with open(path, "w") as obj:
        obj.write(f"mtllib {os.path.basename(base)}.mtl\n")
        offset = 1
        for i, (m, v, n) in enumerate(parts):
            v, n, f, _ = compact(v, n, strand_faces(m, strand_fraction))
            obj.write(f"o part{i}\nusemtl m{i}\n")
            obj.writelines(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in v)
            obj.writelines(f"vn {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in n)
            tri = f.reshape(-1, 3) + offset
            obj.writelines(f"f {a}//{a} {b}//{b} {c}//{c}\n" for a, b, c in tri)
            offset += len(v)


# ---------------------------------------------------------------------------
# glTF binary
# ---------------------------------------------------------------------------
FLOAT, UINT = 5126, 5125
ARRAY_BUFFER, ELEMENT_BUFFER = 34962, 34963


class _Buffer:
    def __init__(self):
        self.data = bytearray()
        self.views, self.accessors = [], []

    def add(self, array, target, comp_type, kind, minmax=False):
        raw = np.ascontiguousarray(array).tobytes()
        while len(self.data) % 4:
            self.data.append(0)
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        self.data += raw
        acc = {"bufferView": len(self.views) - 1, "componentType": comp_type,
               "count": int(len(array)), "type": kind}
        if minmax:
            acc["min"] = [float(x) for x in array.min(axis=0)]
            acc["max"] = [float(x) for x in array.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def export_glb(rig, path, pose=None, strand_fraction=1.0):
    """Write a .glb whose node tree mirrors the rig, posed like the current view.

    Materials are plain colours (procedural patterns, freckles and the shader
    iris are render-only). Vertex colours (skin blush, soft hair edges) are kept.
    """
    pose = pose or {}
    buf = _Buffer()
    materials, mat_index = [], {}
    nodes, mesh_defs, node_index = [], [], {}

    def material(m):
        rough = 0.35 if m.kind in ("glossy", "eye") else 0.85
        has_alpha = m.colors is not None and m.colors.shape[1] == 4
        key = (m.color if m.colors is None else "vc", rough, has_alpha)
        if key not in mat_index:
            base = [1.0, 1.0, 1.0, 1.0] if m.colors is not None else [*m.color, 1.0]
            mat = {"pbrMetallicRoughness": {"baseColorFactor": base, "metallicFactor": 0.0,
                                            "roughnessFactor": rough}, "doubleSided": True}
            if has_alpha:
                mat["alphaMode"] = "BLEND"
            materials.append(mat)
            mat_index[key] = len(materials) - 1
        return mat_index[key]

    def visit(node, parent_pivot):
        idx = len(nodes)
        node_index[node.name] = idx
        rx, ry, rz = pose.get(node.name, (0.0, 0.0, 0.0))
        trans = node.pivot - parent_pivot
        if parent_pivot is None or node.parent is None:
            trans = node.pivot + np.array([0.0, pose.get("root_dy", 0.0), 0.0])
        entry = {"name": node.name, "translation": [float(x) for x in trans]}
        if rx or ry or rz:
            entry["rotation"] = [float(x) for x in matrix_to_quat(euler_matrix(rx, ry, rz))]
        nodes.append(entry)
        prims = []
        for m in node.meshes:
            v, n, f, cols = compact(m.v - node.pivot.astype(np.float32), m.n,
                                    strand_faces(m, strand_fraction), m.colors)
            attrs = {"POSITION": buf.add(v.astype(np.float32), ARRAY_BUFFER, FLOAT, "VEC3", minmax=True),
                     "NORMAL": buf.add(n, ARRAY_BUFFER, FLOAT, "VEC3")}
            if cols is not None:
                attrs["COLOR_0"] = buf.add(cols, ARRAY_BUFFER, FLOAT,
                                           "VEC4" if cols.shape[1] == 4 else "VEC3")
            prims.append({"attributes": attrs, "material": material(m),
                          "indices": buf.add(f, ELEMENT_BUFFER, UINT, "SCALAR")})
        if prims:
            mesh_defs.append({"name": node.name, "primitives": prims})
            entry["mesh"] = len(mesh_defs) - 1
        kids = [visit(c, node.pivot) for c in node.children]
        if kids:
            entry["children"] = kids
        return idx

    root = visit(rig.root, np.zeros(3))
    doc = {"asset": {"version": "2.0", "generator": "avatarkit"},
           "scene": 0, "scenes": [{"nodes": [root]}], "nodes": nodes, "meshes": mesh_defs,
           "materials": materials, "accessors": buf.accessors, "bufferViews": buf.views,
           "buffers": [{"byteLength": len(buf.data)}]}
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    data = bytes(buf.data) + b"\0" * (-len(buf.data) % 4)
    total = 12 + 8 + len(js) + 8 + len(data)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(data), b"BIN\0") + data)


# ---------------------------------------------------------------------------
# Skinned glTF and VRM 1.0
# ---------------------------------------------------------------------------
USHORT = 5123
# rig node -> VRM humanoid bone (the avatar's left side is +x; VRM 1.0 also faces +z)
VRM_BONES = {"root": "hips", "torso": "spine", "head": "head",
             "thighL": "leftUpperLeg", "shinL": "leftLowerLeg", "footL": "leftFoot",
             "thighR": "rightUpperLeg", "shinR": "rightLowerLeg", "footR": "rightFoot",
             "armL": "leftUpperArm", "foreL": "leftLowerArm", "handL": "leftHand",
             "armR": "rightUpperArm", "foreR": "rightLowerArm", "handR": "rightHand"}
VRM_HEIGHT_M = 1.65          # the avatar is about 8.4 units tall


def _write_glb(path, doc, buf):
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    data = bytes(buf.data) + b"\0" * (-len(buf.data) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(data)))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(data), b"BIN\0") + data)


def export_skinned(rig, path, pose=None, name="Avatar", vrm=False, scale=1.0, feet_on_ground=False,
                   strand_fraction=1.0):
    """Skinned glTF: bones are the rig nodes (posed like `pose`), every mesh is bound
    to its own bone with weight 1, and the file is written in the bind (rest) pose.

    With vrm=True it also carries VRM 1.0 metadata and the humanoid bone map, in metres.
    """
    pose = pose or {}
    buf = _Buffer()
    materials, mat_index = [], {}
    nodes, node_index = [], {}
    shift = np.array([0.0, -rig.ground_y if feet_on_ground else 0.0, 0.0])

    def material(m):
        rough = 0.35 if m.kind in ("glossy", "eye") else 0.85
        has_alpha = m.colors is not None and m.colors.shape[1] == 4
        key = (m.color if m.colors is None else "vc", rough, has_alpha)
        if key not in mat_index:
            base = [1.0, 1.0, 1.0, 1.0] if m.colors is not None else [*m.color, 1.0]
            mat = {"pbrMetallicRoughness": {"baseColorFactor": base, "metallicFactor": 0.0,
                                            "roughnessFactor": rough}, "doubleSided": True}
            if has_alpha:
                mat["alphaMode"] = "BLEND"
            materials.append(mat)
            mat_index[key] = len(materials) - 1
        return mat_index[key]

    def world_pivot(node):
        return (node.pivot + shift) * scale

    def visit(node, parent):
        idx = len(nodes)
        node_index[node.name] = idx
        here = world_pivot(node)
        trans = here - (world_pivot(parent) if parent else 0.0)
        entry = {"name": node.name, "translation": [float(x) for x in trans]}
        rx, ry, rz = pose.get(node.name, (0.0, 0.0, 0.0))
        if rx or ry or rz:
            entry["rotation"] = [float(x) for x in matrix_to_quat(euler_matrix(rx, ry, rz))]
        nodes.append(entry)
        kids = [visit(c, node) for c in node.children]
        if kids:
            entry["children"] = kids
        return idx

    root_bone = visit(rig.root, None)
    joint_nodes = [node_index[n] for n in rig.nodes]
    joint_of = {name: i for i, name in enumerate(rig.nodes)}

    ibm = np.zeros((len(rig.nodes), 4, 4), np.float32)
    for bone, node in rig.nodes.items():
        m = np.eye(4, dtype=np.float32)
        m[:3, 3] = -world_pivot(node)          # inverse of a pure translation
        ibm[joint_of[bone]] = m.T              # glTF matrices are column-major
    ibm_acc = buf.add(ibm, None, FLOAT, "MAT4")

    prims = []
    for bone, node in rig.nodes.items():
        for m in node.meshes:
            v, n, f, cols = compact((m.v + shift.astype(np.float32)) * scale, m.n,
                                    strand_faces(m, strand_fraction), m.colors)
            attrs = {"POSITION": buf.add(v.astype(np.float32), ARRAY_BUFFER, FLOAT, "VEC3", minmax=True),
                     "NORMAL": buf.add(n, ARRAY_BUFFER, FLOAT, "VEC3")}
            if cols is not None:
                attrs["COLOR_0"] = buf.add(cols, ARRAY_BUFFER, FLOAT, "VEC4" if cols.shape[1] == 4 else "VEC3")
            joints = np.zeros((len(v), 4), np.uint16)
            joints[:, 0] = joint_of[bone]
            weights = np.zeros((len(v), 4), np.float32)
            weights[:, 0] = 1.0
            attrs["JOINTS_0"] = buf.add(joints, ARRAY_BUFFER, USHORT, "VEC4")
            attrs["WEIGHTS_0"] = buf.add(weights, ARRAY_BUFFER, FLOAT, "VEC4")
            prims.append({"attributes": attrs, "material": material(m),
                          "indices": buf.add(f, ELEMENT_BUFFER, UINT, "SCALAR")})

    mesh_node = len(nodes)
    nodes.append({"name": name + "_mesh", "mesh": 0, "skin": 0})
    doc = {"asset": {"version": "2.0", "generator": "avatarkit"},
           "scene": 0, "scenes": [{"nodes": [root_bone, mesh_node]}], "nodes": nodes,
           "meshes": [{"name": name, "primitives": prims}],
           "skins": [{"name": name + "_skin", "joints": joint_nodes, "skeleton": root_bone,
                      "inverseBindMatrices": ibm_acc}],
           "materials": materials, "accessors": buf.accessors, "bufferViews": buf.views,
           "buffers": [{"byteLength": len(buf.data)}]}
    if vrm:
        bones = {VRM_BONES[n]: {"node": node_index[n]} for n in VRM_BONES if n in node_index}
        doc["extensionsUsed"] = ["VRMC_vrm"]
        doc["extensions"] = {"VRMC_vrm": {
            "specVersion": "1.0",
            "meta": {"name": name, "version": "1.0", "authors": ["Avatar Studio"],
                     "copyrightInformation": "Created with Avatar Studio",
                     "licenseUrl": "https://vrm.dev/licenses/1.0/",
                     "avatarPermission": "everyone", "commercialUsage": "personalNonProfit",
                     "creditNotation": "unnecessary", "modification": "allowModification"},
            "humanoid": {"humanBones": bones}}}
    _write_glb(path, doc, buf)


def export_vrm(rig, path, pose=None, name="Avatar", strand_fraction=0.35):
    """VRM 1.0 humanoid (metres, feet on the ground). Expressions/blendshapes are not included.

    Hair strands are thinned to `strand_fraction` because avatar apps prefer small meshes."""
    export_skinned(rig, path, pose, name=name, vrm=True, scale=VRM_HEIGHT_M / 8.4, feet_on_ground=True,
                   strand_fraction=strand_fraction)
