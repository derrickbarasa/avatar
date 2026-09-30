"""Tests that don't need a window or GPU: geometry, options, exporters, poses, photo."""
import json
import math
import os
import random
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from avatarkit import options as O  # noqa: E402
from avatarkit.builder import build_avatar  # noqa: E402
from avatarkit.exporters import export_glb, export_obj  # noqa: E402
from avatarkit.mathutil import euler_matrix, matrix_to_quat  # noqa: E402
from avatarkit.poses import POSE_FUNCS, blink_angle, pose_at  # noqa: E402


def default_rig(**changes):
    state = dict(O.DEFAULT_STATE)
    for k, v in changes.items():
        O.set_by_name(state, k, v)
    return build_avatar(O.resolve(state))


class OptionTests(unittest.TestCase):
    def test_defaults_are_valid_indices(self):
        for key, _, opts in O.OPTIONS:
            self.assertLess(O.DEFAULT_STATE[key], len(opts), key)

    def test_every_tab_key_exists_once(self):
        keys = [k for _, ks in O.TABS for k in ks]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), set(O.OPTION_KEYS))

    def test_share_code_round_trip(self):
        rng = random.Random(1)
        for _ in range(20):
            a, b = dict(O.DEFAULT_STATE), dict(O.DEFAULT_STATE)
            O.randomize(a, rng)
            O.decode_state(O.encode_state(a), b)
            self.assertEqual(O.encode_state(a), O.encode_state(b))

    def test_share_code_rejects_garbage(self):
        state = dict(O.DEFAULT_STATE)
        for bad in ("", "hello", "AV2-abc", O.encode_state(state)[:-1], "AV2-" + "z" * len(O.CODE_KEYS)):
            with self.assertRaises(ValueError):
                O.decode_state(bad, state)

    def test_save_and_load(self):
        state = dict(O.DEFAULT_STATE)
        O.set_by_name(state, "hair", "Mohawk")
        O.set_by_name(state, "skin", "Deep")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.json")
            O.save_state(state, path)
            loaded = dict(O.DEFAULT_STATE)
            O.load_state(loaded, path)
        self.assertEqual(O.resolve(loaded)["hair"], "mohawk")
        self.assertEqual(loaded["skin"], state["skin"])

    def test_unknown_names_raise(self):
        with self.assertRaises(ValueError):
            O.set_by_name(dict(O.DEFAULT_STATE), "hair", "Nope")
        with self.assertRaises(ValueError):
            O.set_by_name(dict(O.DEFAULT_STATE), "nope", "x")

    def test_presets_only_use_valid_values(self):
        for name in O.PRESETS:
            state = dict(O.DEFAULT_STATE)
            O.apply_preset(state, name)
            build_avatar(O.resolve(state))


class BuildTests(unittest.TestCase):
    def test_every_option_value_builds(self):
        for key, _, opts in O.OPTIONS:
            for i in range(len(opts)):
                state = dict(O.DEFAULT_STATE)
                state[key] = i
                rig = build_avatar(O.resolve(state))
                self.assertTrue(rig.all_meshes(), (key, i))

    def test_random_avatars_build_with_finite_geometry(self):
        rng = random.Random(7)
        for _ in range(15):
            state = dict(O.DEFAULT_STATE)
            O.randomize(state, rng)
            for m in build_avatar(O.resolve(state)).all_meshes():
                self.assertTrue(np.isfinite(m.v).all())
                self.assertTrue(np.isfinite(m.n).all())
                self.assertLess(int(m.f.max()), len(m.v))

    def test_rig_has_expected_nodes(self):
        rig = default_rig()
        for name in ("root", "torso", "head", "armL", "armR", "foreL", "handR",
                     "thighL", "shinR", "footL"):
            self.assertIn(name, rig.nodes)
        self.assertEqual(rig.root.name, "root")

    def test_height_moves_the_ground(self):
        self.assertLess(default_rig(height="Tall").ground_y, default_rig(height="Short").ground_y)

    def test_hat_replaces_tall_hair_and_ignores_afro(self):
        # Just needs to build without the styles poking through / crashing.
        for hair in ("Quiff", "Bun", "Mohawk", "Curly", "Afro", "Ponytail"):
            default_rig(hair=hair, hat="Beanie")
            default_rig(hair=hair, hat="Cap")


class PoseTests(unittest.TestCase):
    def test_all_poses_produce_finite_angles(self):
        for name in POSE_FUNCS:
            for t in (0.0, 0.37, 1.9):
                for animate in (True, False):
                    for key, val in pose_at(name, t, animate).items():
                        vals = val if isinstance(val, tuple) else (val,)
                        self.assertTrue(all(math.isfinite(x) for x in vals), (name, key))

    def test_static_pose_ignores_time(self):
        self.assertEqual(pose_at("wave", 0.0, False), pose_at("wave", 5.0, False))

    def test_blink_is_open_most_of_the_time(self):
        angles = [blink_angle(t / 100) for t in range(0, 900)]
        self.assertEqual(angles[0], 0.0)
        self.assertGreater(max(angles), 30)
        self.assertGreater(sum(a == 0 for a in angles) / len(angles), 0.9)

    def test_quaternion_matches_matrix(self):
        m = euler_matrix(20, -35, 50)
        x, y, z, w = matrix_to_quat(m)
        back = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        np.testing.assert_allclose(back, m, atol=1e-9)

    def test_baked_pose_moves_the_hand(self):
        rig = default_rig()
        mats = rig.world_matrices(pose_at("cheer", 0.0, False))
        hand_rest = rig["handL"].meshes[0].v.mean(0)
        m = mats["handL"]
        hand_now = hand_rest @ m[:3, :3].T + m[:3, 3]
        self.assertGreater(hand_now[1], hand_rest[1] + 2.0)  # arm raised well above rest


class CameraTests(unittest.TestCase):
    def test_zoom_persists_and_view_change_resets_it(self):
        from avatarkit.render import Camera
        cam = Camera("bust")
        for _ in range(80):
            cam.update(-7.3)
        base = cam.dist
        cam.zoom(3)
        for _ in range(80):
            cam.update(-7.3)
        self.assertLess(cam.dist, base * 0.85)
        cam.set_view("full")
        for _ in range(120):
            cam.update(-7.3)
        self.assertGreater(cam.dist, base)  # the full-body view sits further back

    def test_taller_avatar_gets_a_further_full_view(self):
        from avatarkit.render import Camera
        self.assertGreater(Camera.target("full", -8.0)[1], Camera.target("full", -7.0)[1])


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_obj_and_mtl(self):
        rig = default_rig()
        path = os.path.join(self.tmp.name, "a.obj")
        export_obj(rig, path, pose_at("wave", 0.0, False))
        with open(path) as f:
            text = f.read()
        self.assertTrue(text.startswith("mtllib a.mtl"))
        self.assertGreater(text.count("\nv "), 10000)
        self.assertGreater(text.count("\nf "), 10000)
        with open(os.path.join(self.tmp.name, "a.mtl")) as f:
            self.assertIn("newmtl m0", f.read())

    def parse_glb(self, path):
        with open(path, "rb") as f:
            raw = f.read()
        magic, version, length = struct.unpack("<4sII", raw[:12])
        self.assertEqual((magic, version, length), (b"glTF", 2, len(raw)))
        jlen, jtype = struct.unpack("<I4s", raw[12:20])
        self.assertEqual(jtype, b"JSON")
        doc = json.loads(raw[20:20 + jlen])
        off = 20 + jlen
        blen, btype = struct.unpack("<I4s", raw[off:off + 8])
        self.assertEqual(btype, b"BIN\0")
        self.assertEqual(off + 8 + blen, len(raw))
        return doc, raw[off + 8:]

    def test_glb_structure(self):
        rig = default_rig(hair="Long", top="Hoodie", glasses="Round")
        path = os.path.join(self.tmp.name, "a.glb")
        export_glb(rig, path, pose_at("hips", 0.0, False))
        doc, blob = self.parse_glb(path)
        self.assertEqual(doc["buffers"][0]["byteLength"], len(blob))
        self.assertEqual(len(doc["nodes"]), len(rig.nodes))
        names = {n["name"] for n in doc["nodes"]}
        self.assertEqual(names, set(rig.nodes))
        for view in doc["bufferViews"]:
            self.assertLessEqual(view["byteOffset"] + view["byteLength"], len(blob))
            self.assertEqual(view["byteOffset"] % 4, 0)
        n_prims = sum(len(m["primitives"]) for m in doc["meshes"])
        self.assertEqual(n_prims, len(rig.all_meshes()))
        for mesh in doc["meshes"]:
            for prim in mesh["primitives"]:
                pos = doc["accessors"][prim["attributes"]["POSITION"]]
                self.assertEqual(pos["type"], "VEC3")
                self.assertEqual(len(pos["min"]), 3)
                idx = doc["accessors"][prim["indices"]]
                self.assertEqual(idx["count"] % 3, 0)
        # every node except the root is somebody's child
        children = {c for n in doc["nodes"] for c in n.get("children", [])}
        self.assertEqual(len(children), len(doc["nodes"]) - 1)

    def test_glb_loads_in_trimesh_if_available(self):
        try:
            import trimesh
        except ImportError:
            self.skipTest("trimesh not installed")
        rig = default_rig()
        path = os.path.join(self.tmp.name, "a.glb")
        export_glb(rig, path)
        scene = trimesh.load(path)
        self.assertGreater(len(scene.geometry), 20)
        top = max(scene.bounds[1][1], 0)
        self.assertGreater(top, 0.3)


class PhotoTests(unittest.TestCase):
    def synthetic(self, skin, hair, bg, beard=None):
        img = np.zeros((300, 300, 3), np.uint8)
        img[:] = (np.array(bg) * 255).astype(np.uint8)
        img[60:250, 100:200] = (np.array(skin) * 255).astype(np.uint8)   # face
        img[30:80, 90:210] = (np.array(hair) * 255).astype(np.uint8)     # hair above
        if beard is not None:
            img[195:250, 100:200] = (np.array(beard) * 255).astype(np.uint8)
        return img

    def test_reads_skin_hair_and_beard(self):
        from avatarkit import photo
        img = self.synthetic(skin=(0.62, 0.44, 0.29), hair=(0.06, 0.05, 0.05), bg=(0.9, 0.9, 0.95),
                             beard=(0.10, 0.08, 0.06))
        result, notes = photo.analyze(img, face_box=(100, 60, 100, 190))
        self.assertEqual(result["skin"], "Tan")
        self.assertEqual(result["haircolor"], "Black")
        self.assertEqual(result["facial"], "Beard")

    def test_no_beard_and_light_skin(self):
        from avatarkit import photo
        img = self.synthetic(skin=(0.96, 0.80, 0.69), hair=(0.85, 0.68, 0.36), bg=(0.3, 0.4, 0.6))
        result, _ = photo.analyze(img, face_box=(100, 60, 100, 190))
        self.assertEqual(result["skin"], "Porcelain")
        self.assertEqual(result["haircolor"], "Blonde")
        self.assertEqual(result["facial"], "None")


if __name__ == "__main__":
    unittest.main()
