"""Saving and exporting: library files, undo history, bundles, skinned GLB and VRM."""
import json
import os
import struct
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from avatarkit import options as O  # noqa: E402
from avatarkit import store  # noqa: E402
from avatarkit.builder import build_avatar  # noqa: E402
from avatarkit.exporters import export_glb, export_obj, export_skinned, export_vrm  # noqa: E402
from avatarkit.history import History  # noqa: E402
from avatarkit.poses import pose_at  # noqa: E402


class LibraryCase(unittest.TestCase):
    """Runs each test against its own empty library folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self._old = os.environ.get("AVATARKIT_HOME")
        os.environ["AVATARKIT_HOME"] = self.tmp.name
        self.addCleanup(self.restore_env)

    def restore_env(self):
        if self._old is None:
            os.environ.pop("AVATARKIT_HOME", None)
        else:
            os.environ["AVATARKIT_HOME"] = self._old

    def state(self, **changes):
        st = dict(O.DEFAULT_STATE)
        for k, v in changes.items():
            O.set_by_name(st, k, v)
        return st


class StoreTests(LibraryCase):
    def test_folders_are_created_under_the_home(self):
        for fn in (store.avatars_dir, store.exports_dir, store.packs_dir):
            self.assertTrue(fn().startswith(self.tmp.name))
            self.assertTrue(os.path.isdir(fn()))

    def test_save_and_load_round_trip(self):
        original = self.state(hair="Mohawk", skin="Deep", glasses="Round", pose="Wave")
        path = store.save_avatar(original, "Punk Rock!")
        self.assertTrue(path.endswith("Punk_Rock.avatar"))
        loaded = dict(O.DEFAULT_STATE)
        self.assertEqual(store.load_avatar(path, loaded), "Punk Rock!")
        self.assertEqual(O.resolve(loaded)["hair"], "mohawk")
        self.assertEqual(O.encode_state(loaded), O.encode_state(original))

    def test_file_is_readable_json_with_option_names(self):
        path = store.save_avatar(self.state(hair="Bob"), "Readable")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["format"], store.FORMAT)
        self.assertEqual(data["options"]["hair"], "Bob")
        self.assertTrue(data["code"].startswith(O.CODE_PREFIX))

    def test_option_names_still_load_if_the_code_is_stale(self):
        path = store.save_avatar(self.state(hair="Afro"), "Stale")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["code"] = "AV2-broken"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        st = dict(O.DEFAULT_STATE)
        store.load_avatar(path, st)
        self.assertEqual(O.resolve(st)["hair"], "afro")

    def test_legacy_json_and_share_code_files_open(self):
        legacy = os.path.join(self.tmp.name, "old.json")
        with open(legacy, "w") as f:
            json.dump({"hair": "Long", "skin": "Tan"}, f)
        st = dict(O.DEFAULT_STATE)
        store.load_avatar(legacy, st)
        self.assertEqual(O.resolve(st)["hair"], "long")
        code_file = os.path.join(self.tmp.name, "code.txt")
        open(code_file, "w").write(O.encode_state(self.state(hair="Quiff")) + "\n")
        st2 = dict(O.DEFAULT_STATE)
        store.load_avatar(code_file, st2)
        self.assertEqual(O.resolve(st2)["hair"], "quiff")

    def test_bad_files_give_readable_errors(self):
        for name, content in (("empty.avatar", ""), ("junk.avatar", "not json"), ("list.avatar", "[1, 2]"),
                              ("other.avatar", json.dumps({"format": "something/else", "options": {"a": 1}})),
                              ("nothing.avatar", json.dumps({"options": {"nonsense": "x"}}))):
            path = os.path.join(self.tmp.name, name)
            with open(path, "w") as f:
                f.write(content)
            with self.assertRaises(ValueError, msg=name):
                store.load_avatar(path, dict(O.DEFAULT_STATE))
        with self.assertRaises(ValueError):
            store.load_avatar(os.path.join(self.tmp.name, "missing.avatar"), dict(O.DEFAULT_STATE))

    def test_saving_is_atomic_and_overwrites_the_same_name(self):
        store.save_avatar(self.state(hair="Bob"), "Same")
        store.save_avatar(self.state(hair="Long"), "Same")
        files = os.listdir(store.avatars_dir())
        self.assertEqual(files, ["Same.avatar"])            # no .tmp left behind, no duplicate
        st = dict(O.DEFAULT_STATE)
        store.load_avatar(os.path.join(store.avatars_dir(), "Same.avatar"), st)
        self.assertEqual(O.resolve(st)["hair"], "long")

    def test_list_delete_and_damaged_files(self):
        a = store.save_avatar(self.state(), "First")
        os.utime(a, (1000, 1000))
        store.save_avatar(self.state(hair="Bob"), "Second")
        with open(os.path.join(store.avatars_dir(), "broken.avatar"), "w") as f:
            f.write("{oops")
        names = [x.name for x in store.list_avatars()]
        self.assertEqual(names, ["Second", "First"])          # newest first, damaged file skipped
        self.assertTrue(store.delete_avatar(a))
        self.assertFalse(store.delete_avatar(a))
        self.assertEqual([x.name for x in store.list_avatars()], ["Second"])

    def test_unique_path_and_slug(self):
        d = store.exports_dir()
        first = store.unique_path(d, "me", ".png")
        with open(first, "w") as f:
            f.write("x")
        self.assertTrue(store.unique_path(d, "me", ".png").endswith("me_2.png"))
        self.assertEqual(store.slug("A  b/c\\d:e?"), "A_bcde")
        self.assertEqual(store.slug("   "), "avatar")
        self.assertLessEqual(len(store.slug("x" * 300)), 60)

    def test_autosave_restores_and_ignores_garbage(self):
        st = self.state(hair="Curly", haircolor="Ginger")
        store.autosave(st, "Working")
        restored = dict(O.DEFAULT_STATE)
        self.assertEqual(store.restore_autosave(restored), "Working")
        self.assertEqual(O.resolve(restored)["hair"], "curly")
        with open(store.autosave_path(), "w") as f:
            f.write("garbage")
        self.assertIsNone(store.restore_autosave(dict(O.DEFAULT_STATE)))

    def test_thumbnails_are_cropped_and_stored_in_the_file(self):
        import pygame
        surf = pygame.Surface((700, 720))
        surf.fill((200, 60, 60))
        png = store.thumbnail_from(surf)
        img = pygame.image.load(__import__("io").BytesIO(png))
        self.assertEqual(img.get_size(), store.THUMB_SIZE)
        path = store.save_avatar(self.state(), "With picture", thumbnail_png=png)
        info = store.list_avatars()[0]
        self.assertEqual(info.thumbnail, png)
        self.assertEqual(info.path, path)

    def test_bundle_contains_files_readme_and_code(self):
        src = os.path.join(self.tmp.name, "model.glb")
        with open(src, "wb") as f:
            f.write(b"glTF")
        code = O.encode_state(self.state())
        path = store.write_bundle(os.path.join(self.tmp.name, "b.zip"), "My Hero", {"my.glb": src}, code)
        with zipfile.ZipFile(path) as z:
            self.assertEqual(set(z.namelist()), {"my.glb", "README.txt", "code.txt"})
            self.assertIn(code, z.read("README.txt").decode())
            self.assertEqual(z.read("code.txt").decode().strip(), code)
            self.assertIsNone(z.testzip())


class HistoryTests(unittest.TestCase):
    def test_undo_redo_walks_the_edits(self):
        st = dict(O.DEFAULT_STATE)
        h = History(st)
        for value in (1, 2, 3):
            st["hair"] = value
            self.assertTrue(h.commit(st))
        self.assertEqual(h.undo()["hair"], 2)
        self.assertEqual(h.undo()["hair"], 1)
        self.assertEqual(h.redo()["hair"], 2)
        self.assertTrue(h.can_undo and h.can_redo)

    def test_no_change_no_snapshot_and_new_edit_clears_redo(self):
        st = dict(O.DEFAULT_STATE)
        h = History(st)
        self.assertFalse(h.commit(st))
        st["hair"] = 5
        h.commit(st)
        h.undo()
        st2 = dict(O.DEFAULT_STATE)
        st2["hair"] = 7
        h.commit(st2)
        self.assertFalse(h.can_redo)
        self.assertIsNone(h.redo())

    def test_limit_and_empty_undo(self):
        st = dict(O.DEFAULT_STATE)
        h = History(st, limit=5)
        self.assertIsNone(h.undo())
        for i in range(20):
            st["hair"] = i % 12
            st["skin"] = i % 6
            h.commit(st)
        self.assertLessEqual(len(h.items), 5)
        for _ in range(10):
            h.undo()
        self.assertFalse(h.can_undo)


class ModelExportTests(LibraryCase):
    def parse(self, path):
        with open(path, "rb") as f:
            raw = f.read()
        jlen = struct.unpack("<I", raw[12:16])[0]
        doc = json.loads(raw[20:20 + jlen])
        blob = raw[28 + jlen:]
        return doc, blob

    def rig(self, **changes):
        return build_avatar(O.resolve(self.state(**changes)))

    def test_skinned_glb_binds_every_mesh_to_a_bone(self):
        rig = self.rig(hair="Bob")
        path = os.path.join(self.tmp.name, "s.glb")
        export_skinned(rig, path, pose_at("wave", 0.3, True), name="Hero", strand_fraction=0.3)
        doc, blob = self.parse(path)
        skin = doc["skins"][0]
        self.assertEqual(len(skin["joints"]), len(rig.nodes))
        ibm = doc["accessors"][skin["inverseBindMatrices"]]
        self.assertEqual((ibm["type"], ibm["count"]), ("MAT4", len(rig.nodes)))
        self.assertEqual(sum(len(m["primitives"]) for m in doc["meshes"]), len(rig.all_meshes()))
        for prim in doc["meshes"][0]["primitives"]:
            self.assertIn("JOINTS_0", prim["attributes"])
            joints = doc["accessors"][prim["attributes"]["JOINTS_0"]]
            weights = doc["accessors"][prim["attributes"]["WEIGHTS_0"]]
            self.assertEqual(joints["count"], weights["count"])
        # a mesh node references the skin and the bone tree is in the scene
        self.assertTrue(any(n.get("skin") == 0 for n in doc["nodes"]))

    def test_inverse_bind_matrices_undo_each_bones_rest_position(self):
        rig = self.rig()
        path = os.path.join(self.tmp.name, "s.glb")
        export_skinned(rig, path)
        doc, blob = self.parse(path)
        skin = doc["skins"][0]
        acc = doc["accessors"][skin["inverseBindMatrices"]]
        view = doc["bufferViews"][acc["bufferView"]]
        mats = np.frombuffer(blob, np.float32, acc["count"] * 16, view["byteOffset"]).reshape(-1, 4, 4)
        for i, name in enumerate(rig.nodes):
            np.testing.assert_allclose(mats[i][3, :3], -rig[name].pivot, atol=1e-4)   # column-major translation

    def test_vrm_is_a_metric_humanoid_standing_on_the_ground(self):
        rig = self.rig(hair="Short")
        path = os.path.join(self.tmp.name, "a.vrm")
        export_vrm(rig, path, name="Hero")
        doc, blob = self.parse(path)
        self.assertIn("VRMC_vrm", doc["extensionsUsed"])
        vrm = doc["extensions"]["VRMC_vrm"]
        self.assertEqual(vrm["specVersion"], "1.0")
        bones = vrm["humanoid"]["humanBones"]
        for required in ("hips", "spine", "head", "leftUpperArm", "rightUpperArm", "leftLowerArm",
                         "rightLowerArm", "leftHand", "rightHand", "leftUpperLeg", "rightUpperLeg",
                         "leftLowerLeg", "rightLowerLeg", "leftFoot", "rightFoot"):
            self.assertIn(required, bones)
            self.assertLess(bones[required]["node"], len(doc["nodes"]))
        self.assertEqual(vrm["meta"]["name"], "Hero")
        # geometry is in metres with the feet at y = 0
        lows, highs = [], []
        for prim in doc["meshes"][0]["primitives"]:
            pos = doc["accessors"][prim["attributes"]["POSITION"]]
            lows.append(pos["min"][1])
            highs.append(pos["max"][1])
        self.assertAlmostEqual(min(lows), 0.0, delta=0.06)
        self.assertAlmostEqual(max(highs), 1.65, delta=0.2)

    def accessor_floats(self, doc, blob, index):
        """Accessor contents as an (N, 3) array, applying a sparse substitution if there is one."""
        acc = doc["accessors"][index]
        out = np.zeros((acc["count"], 3), np.float32)
        if "bufferView" in acc:
            view = doc["bufferViews"][acc["bufferView"]]
            out = np.frombuffer(blob, np.float32, acc["count"] * 3, view["byteOffset"]).reshape(-1, 3)
        return out

    def test_vrm_has_expression_presets_backed_by_morph_targets(self):
        rig = self.rig(hair="Short")
        path = os.path.join(self.tmp.name, "a.vrm")
        export_vrm(rig, path)
        doc, blob = self.parse(path)
        presets = doc["extensions"]["VRMC_vrm"]["expressions"]["preset"]
        for name in ("happy", "angry", "sad", "surprised", "relaxed", "aa", "ih", "ou", "ee", "oh",
                     "blink", "blinkLeft", "blinkRight"):
            self.assertIn(name, presets)
        mesh = doc["meshes"][0]
        self.assertEqual(mesh["extras"]["targetNames"], list(presets))
        for bind in (b for p in presets.values() for b in p["morphTargetBinds"]):
            self.assertEqual(doc["nodes"][bind["node"]]["mesh"], 0)
            self.assertLess(bind["index"], len(presets))
        # every primitive carries every target (the glTF rule), each with one offset per vertex
        for prim in mesh["primitives"]:
            self.assertEqual(len(prim["targets"]), len(presets))
            count = doc["accessors"][prim["attributes"]["POSITION"]]["count"]
            for target in prim["targets"]:
                self.assertEqual(doc["accessors"][target["POSITION"]]["count"], count)

    def test_vrm_mouth_and_blink_targets_move_the_face(self):
        rig = self.rig(hair="Short")
        path = os.path.join(self.tmp.name, "a.vrm")
        export_vrm(rig, path)
        doc, blob = self.parse(path)
        names = doc["meshes"][0]["extras"]["targetNames"]
        reach = {n: 0.0 for n in names}
        for prim in doc["meshes"][0]["primitives"]:
            for name, target in zip(names, prim["targets"]):
                reach[name] = max(reach[name], float(np.abs(self.accessor_floats(doc, blob, target["POSITION"])).max()))
        for name in names:
            self.assertGreater(reach[name], 0.001, name)            # every expression changes something
            self.assertLess(reach[name], 0.2, name)                 # ... by a believable amount (metres)
        self.assertGreater(reach["aa"], reach["relaxed"])           # a wide-open mouth moves further

    def test_mouth_shapes_share_one_vertex_layout(self):
        from avatarkit.head import cached_head, Mouth
        mouth = Mouth(cached_head((0.9, 0.7, 0.6), 0.0, 0.0), 0.0, 1.0, (0.9, 0.7, 0.6))
        shapes = [mouth.fixed_meshes(*s) for s in ((0, 0, 0, 0), (1, 0, 0, 0), (0.5, 0.5, 0, 0.5), (0.2, -1, 1, -0.5))]
        for shape in shapes[1:]:
            self.assertEqual([m.v.shape for m in shape], [m.v.shape for m in shapes[0]])
            self.assertEqual([len(m.f) for m in shape], [len(m.f) for m in shapes[0]])

    def test_left_side_bones_are_on_positive_x(self):
        rig = self.rig()
        path = os.path.join(self.tmp.name, "a.vrm")
        export_vrm(rig, path)
        doc, _ = self.parse(path)
        bones = doc["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"]
        left = doc["nodes"][bones["leftUpperArm"]["node"]]["translation"][0]
        right = doc["nodes"][bones["rightUpperArm"]["node"]]["translation"][0]
        self.assertGreater(left, 0)
        self.assertLess(right, 0)

    def test_thinning_strands_makes_a_smaller_file(self):
        rig = self.rig(hair="Long")
        full, thin = (os.path.join(self.tmp.name, n) for n in ("full.glb", "thin.glb"))
        export_glb(rig, full, strand_fraction=1.0)
        export_glb(rig, thin, strand_fraction=0.25)
        self.assertLess(os.path.getsize(thin), os.path.getsize(full) * 0.6)

    def test_obj_has_no_zero_normals(self):
        rig = self.rig(hair="Bob")
        path = os.path.join(self.tmp.name, "a.obj")
        export_obj(rig, path, strand_fraction=0.2)
        with open(path) as f:
            normals = [line.split()[1:] for line in f if line.startswith("vn ")]
        self.assertGreater(len(normals), 1000)
        lengths = np.linalg.norm(np.array(normals, float), axis=1)
        self.assertGreater(lengths.min(), 0.9)


if __name__ == "__main__":
    unittest.main()
