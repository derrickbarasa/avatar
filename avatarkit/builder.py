"""Assemble a full rig (body + head + hair + accessories) from resolved options."""
import functools

import numpy as np

from . import hair as hair_mod
from . import strands as strands_mod
from . import head as head_mod
from .body import build_body, neck_mesh
from .mathutil import mix
from .mesh import transformed

_hair = functools.lru_cache(maxsize=16)(hair_mod.hair_meshes)
_hat = functools.lru_cache(maxsize=16)(hair_mod.hat_meshes)
_facial = functools.lru_cache(maxsize=16)(hair_mod.facial_hair_meshes)
_glasses = functools.lru_cache(maxsize=8)(head_mod.glasses_meshes)
_brows = functools.lru_cache(maxsize=16)(strands_mod.brow_strand_meshes)


def build_avatar(v):
    """Turn resolved option values into a Rig of meshes."""
    skin, hair = tuple(v["skin"]), tuple(v["haircolor"])
    m_smile, mouth_w = v["mouth"]
    e_smile, e_open, brow_dy, brow_tilt, lid_l, lid_r = v["expression"]
    smile = max(-1.0, min(1.3, m_smile + e_smile))

    rig = build_body(v)
    for node in rig.nodes.values():
        for m in node.meshes:
            m.outlined()
    rig["torso"].add(neck_mesh(skin, v["neck"]).outlined())

    head = head_mod.cached_head(skin, v["face"], v["nose"])
    head.mesh.freckle = float(v["freckles"])
    node = rig.add_node("head", (0.0, -0.55, 0.03), "torso")
    node.add(head.mesh.outlined())
    brow_color = mix(hair, head_mod.DARK, 0.25)
    rig.add_node("brows", (0.0, 0.19, -0.35), "head")      # pivots behind the brow: raises them
    brow_nodes = {}
    for s in (-1, 1):                                        # each brow tilts about its own centre
        bz = head.z_at(s * 0.175, 0.19)
        brow_nodes[s] = rig.add_node("browL" if s > 0 else "browR", (s * 0.175, 0.19, bz), "brows")
    for s in (-1, 1):
        node.add(head_mod.eye_meshes(head, s, v["eyesize"], v["eyes"], skin,
                                     lid_l if s > 0 else lid_r))
        brow_nodes[s].add(head_mod.brow_mesh(head, s, v["brows"], brow_color, brow_dy, brow_tilt,
                                             scale=0.5))   # thin base under the brow hairs
        node.add(head_mod.ear_meshes(s, skin))
    for mesh in _brows(head, v["brows"], brow_color, brow_dy, brow_tilt):
        brow_nodes[mesh.side].add(mesh)
    rig.mouth = head_mod.Mouth(head, smile, mouth_w, skin)
    rig.base_open = e_open
    rig.add_node("mouth", node.pivot, "head")
    rig.set_mouth(0.0)
    node.add(head_mod.nostril_meshes(head, skin))
    node.add(head_mod.earring_meshes(v["earrings"]))

    style, hat = v["hair"], v["hat"]
    if style == "afro":
        hat = "none"
    elif hat != "none" and style in ("quiff", "bun", "mohawk", "curly"):
        style = "short"  # keep tall styles from poking through the hat
    parts = list(_hair(head, style, hair)) + list(_hat(head, hat, tuple(v["hatcolor"])))
    node.add([m.outlined() for m in parts])
    node.add(_facial(head, v["facial"], hair))
    if v["glasses"] != "none":
        node.add(_glasses(head, v["glasses"]))
    if v["headphones"]:
        node.add(hair_mod.headphone_meshes(tuple(v["hatcolor"])))
    apply_head_proportions(rig, node.pivot, v["headsize"], v["neck"])
    return rig


def apply_head_proportions(rig, origin, scale, lift):
    """Resize the head about the neck and raise it (a longer neck). Works on copies: the head parts
    are cached and shared between avatars."""
    if scale == 1.0 and lift == 0.0:
        return
    shift = np.array([0.0, lift, 0.0])
    for name in ("head", "brows", "browL", "browR"):
        node = rig[name]
        node.meshes = [transformed(m, scale, origin, shift) for m in node.meshes]
        if name != "head":                                   # rotation pivots of the parts inside the head
            node.pivot = (node.pivot - origin) * scale + origin + shift
    rig.head_fx = (scale, np.asarray(origin, float), shift)
    rig._mouth_key = None                                    # re-place the lips that were already set
    rig.set_mouth(0.0)
