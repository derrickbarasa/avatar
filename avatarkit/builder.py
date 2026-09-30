"""Assemble a full rig (body + head + hair + accessories) from resolved options."""
import functools

import numpy as np

from . import hair as hair_mod
from . import strands as strands_mod
from . import head as head_mod
from .body import build_body, neck_mesh
from .mathutil import mix

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
    rig["torso"].add(neck_mesh(skin).outlined())

    head = head_mod.cached_head(skin, v["face"], v["nose"])
    head.mesh.freckle = float(v["freckles"])
    node = rig.add_node("head", (0.0, -0.55, 0.03), "torso")
    node.add(head.mesh.outlined())
    brow_color = mix(hair, head_mod.DARK, 0.25)
    for s in (-1, 1):
        node.add(head_mod.eye_meshes(head, s, v["eyesize"], v["eyes"], skin,
                                     lid_l if s > 0 else lid_r))
        node.add(head_mod.brow_mesh(head, s, v["brows"], brow_color, brow_dy, brow_tilt,
                                    scale=0.5))   # thin base under the brow hairs
        node.add(head_mod.ear_meshes(s, skin))
    node.add(_brows(head, v["brows"], brow_color, brow_dy, brow_tilt))
    node.add(head_mod.lip_meshes(head, smile, mouth_w, skin, e_open))
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
    return rig
