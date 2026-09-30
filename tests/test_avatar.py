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


class StrandTests(unittest.TestCase):
    STYLES = ["Short", "Curly", "Long", "Bob", "Bangs", "Bun", "Ponytail", "Quiff", "Mohawk", "Afro"]

    def strand_meshes(self, hair):
        rig = default_rig(hair=hair)
        return [m for m in rig["head"].meshes if m.strand and m.part == "hair"]

    def test_every_style_grows_real_strands(self):
        for hair in self.STYLES:
            meshes = self.strand_meshes(hair)
            self.assertEqual(len(meshes), 1, hair)
            m = meshes[0]
            self.assertGreater(len(m.v), 10000, hair)          # thousands of strands
            self.assertTrue(np.isfinite(m.v).all(), hair)
            self.assertIsNotNone(m.tangents)
            self.assertEqual(len(m.tangents), len(m.v))
            np.testing.assert_allclose(np.linalg.norm(m.tangents, axis=1), 1.0, atol=1e-3)
            self.assertEqual(m.colors.shape, (len(m.v), 3))

    def test_bald_and_buzz_have_no_strands(self):
        for hair in ("Bald", "Buzz"):
            self.assertEqual(self.strand_meshes(hair), [])

    def test_strands_stay_out_of_the_head(self):
        from avatarkit.strands import HEAD_C, HEAD_R
        for hair in ("Long", "Bob", "Short", "Ponytail"):
            v = self.strand_meshes(hair)[0].v.astype(float)
            e = (((v - HEAD_C) / (HEAD_R * 0.92)) ** 2).sum(1)
            self.assertLess((e < 1.0).mean(), 0.02, hair)     # <2% of vertices dip inside

    def test_long_hair_reaches_the_shoulders_and_bob_stops_higher(self):
        low = lambda hair: self.strand_meshes(hair)[0].v[:, 1].min()
        self.assertLess(low("Long"), -1.0)
        self.assertGreater(low("Bob"), low("Long") + 0.3)

    def test_strand_hair_is_deterministic(self):
        a = self.strand_meshes("Curly")[0].v
        from avatarkit import builder
        builder._hair.cache_clear()
        b = self.strand_meshes("Curly")[0].v
        np.testing.assert_array_equal(a, b)

    def test_strands_have_no_outline_pass(self):
        for m in self.strand_meshes("Long"):
            self.assertFalse(m.outline)


class FacialStrandTests(unittest.TestCase):
    def head_strands(self, **changes):
        rig = default_rig(**changes)
        return [m for m in rig["head"].meshes if m.strand]

    def test_beard_mustache_and_brows_are_strands(self):
        none = self.head_strands(hair="Bald")                       # brows only
        stache = self.head_strands(hair="Bald", facial="Mustache")
        beard = self.head_strands(hair="Bald", facial="Beard")
        self.assertEqual(len(none), 1)
        self.assertEqual(len(stache), 2)
        self.assertEqual(len(beard), 2)
        vertex_count = lambda ms: sum(len(m.v) for m in ms)
        self.assertGreater(vertex_count(beard), vertex_count(stache) * 3)

    def test_brow_thickness_changes_hair_count(self):
        thin = self.head_strands(hair="Bald", brows="Thin")[0]
        thick = self.head_strands(hair="Bald", brows="Thick")[0]
        self.assertGreater(len(thick.v), len(thin.v) * 1.5)

    def test_facial_strands_stay_outside_the_skin(self):
        from avatarkit.head import cached_head
        from avatarkit.strands import Skin
        rig = default_rig(hair="Bald", facial="Beard")
        head = cached_head(tuple(O.resolve(dict(O.DEFAULT_STATE))["skin"]), 0.28, 1.0)
        skin = Skin(head, np.ones(len(head.P), bool))
        beard = [m for m in rig["head"].meshes if m.strand][-1]
        v = beard.v.astype(float)[::7]
        j = skin.nearest(v)
        depth = ((v - skin.v[j]) * skin.n[j]).sum(1)
        self.assertGreater(np.percentile(depth, 2), -0.01)   # (almost) nothing dips into the face

    def test_beard_hangs_below_the_chin(self):
        beard = self.head_strands(hair="Bald", facial="Beard")[-1]
        self.assertLess(beard.v[:, 1].min(), -0.62)


class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        pygame.font.init()

    def model(self, tab=0, **kw):
        from avatarkit.ui import UIModel
        return UIModel(dict(O.DEFAULT_STATE), tab=tab, code=O.encode_state(O.DEFAULT_STATE), **kw)

    def collect(self, ui, tab, size=(1000, 720)):
        """Every choose-action reachable in a tab by scrolling from top to bottom."""
        seen = set()
        ui.render(self.model(tab), size)
        for offset in range(0, ui.max_scroll + 80, 60):
            ui.scroll[tab] = offset
            ui.render(self.model(tab), size)
            seen |= {a for _, a in ui.widgets if a[0] == "choose"}
        return seen

    def test_every_option_value_is_reachable_by_scrolling(self):
        from avatarkit.ui import UI
        ui = UI()
        for tab, (_, keys) in enumerate(O.TABS):
            seen = self.collect(ui, tab)
            wanted = {("choose", k, j) for k in keys for j in range(len(O.choices(k)))}
            self.assertEqual(seen, wanted, O.TABS[tab][0])

    def test_controls_stay_inside_the_card(self):
        from avatarkit.ui import UI
        ui = UI()
        for tab in range(len(O.TABS)):
            ui.render(self.model(tab), (1000, 720))
            for rect, action in ui.widgets:
                if action[0] in ("choose", "tab", "button", "copy"):
                    self.assertTrue(ui.panel_rect.contains(rect), (tab, action, rect))

    def test_clicks_map_to_actions(self):
        from avatarkit.ui import UI
        ui = UI()
        ui.render(self.model(0), (1000, 720))
        tabs = {a: r for r, a in ui.widgets if a[0] == "tab"}
        self.assertEqual(ui.hit(tabs[("tab", 3)].center), ("tab", 3))
        self.assertIsNone(ui.hit((20, 20)))                       # empty scene area
        names = {a[1] for _, a in ui.widgets if a[0] == "button"}
        self.assertEqual(names, {"random", "photo", "png", "glb"})
        views = {a[1] for _, a in ui.widgets if a[0] == "view"}
        self.assertEqual(views, {"bust", "face", "full"})

    def test_hover_only_asks_for_a_redraw_when_the_target_changes(self):
        from avatarkit.ui import UI
        ui = UI()
        ui.render(self.model(0), (1000, 720))
        tab = next(r for r, a in ui.widgets if a == ("tab", 2))
        self.assertTrue(ui.hover(tab.center))
        self.assertFalse(ui.hover((tab.centerx + 1, tab.centery)))   # same control
        self.assertTrue(ui.hover((5, 5)))                           # left it

    def test_scrolling_is_clamped_and_wheel_ignored_off_panel(self):
        from avatarkit.ui import UI
        ui = UI()
        ui.render(self.model(0), (1000, 500))
        self.assertGreater(ui.max_scroll, 0)
        self.assertFalse(ui.scroll_by(3, (10, 10)))
        for _ in range(60):
            ui.scroll_by(-3, ui.panel_rect.center)
        ui.render(self.model(0), (1000, 500))
        self.assertEqual(ui.scroll[0], ui.max_scroll)
        for _ in range(80):
            ui.scroll_by(3, ui.panel_rect.center)
        ui.render(self.model(0), (1000, 500))
        self.assertEqual(ui.scroll[0], 0)

    def test_keyboard_focus_scrolls_into_view(self):
        from avatarkit.ui import UI
        ui = UI()
        last = len(O.TABS[0][1]) - 1
        ui.render(self.model(0, focus=0), (1000, 500))
        ui.focus_changed()
        ui.render(self.model(0, focus=last), (1000, 500))
        self.assertGreater(ui.scroll[0], 0)

    def test_toast_and_shortcut_sheet_render(self):
        from avatarkit.ui import UI
        ui = UI()
        ui.render(self.model(0), (1000, 720))
        base = bytes(ui.surface.get_view("0"))
        ui.render(self.model(0, toast="Saved a.png", toast_alpha=1.0), (1000, 720))
        with_toast = bytes(ui.surface.get_view("0"))
        ui.render(self.model(0, help=True), (1000, 720))
        with_help = bytes(ui.surface.get_view("0"))
        self.assertNotEqual(base, with_toast)
        self.assertNotEqual(base, with_help)


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

    def test_glb_normals_are_unit_length_and_vertices_all_used(self):
        # Regression: hair/hat shells used to export unused head vertices with zero normals,
        # which the Khronos glTF validator rejects.
        rig = default_rig(hair="Long", hat="Beanie", facial="Beard")
        path = os.path.join(self.tmp.name, "a.glb")
        export_glb(rig, path)
        doc, blob = self.parse_glb(path)
        for mesh in doc["meshes"]:
            for prim in mesh["primitives"]:
                acc = doc["accessors"][prim["attributes"]["NORMAL"]]
                view = doc["bufferViews"][acc["bufferView"]]
                normals = np.frombuffer(blob, np.float32, acc["count"] * 3, view["byteOffset"]).reshape(-1, 3)
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-3)
                idx = doc["accessors"][prim["indices"]]
                iview = doc["bufferViews"][idx["bufferView"]]
                indices = np.frombuffer(blob, np.uint32, idx["count"], iview["byteOffset"])
                self.assertEqual(len(np.unique(indices)), acc["count"])

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
    def test_real_portrait_is_stable_across_exposure(self):
        try:
            import cv2
            from skimage import data
        except ImportError:
            self.skipTest("opencv / scikit-image not installed")
        if not hasattr(cv2, "CascadeClassifier"):
            self.skipTest("OpenCV without CascadeClassifier")
        from avatarkit import photo
        img = data.astronaut()
        skins = set()
        for gain in (0.6, 1.0, 1.3):
            result, notes = photo.analyze(np.clip(img * gain, 0, 255).astype(np.uint8))
            skins.add(result["skin"])
            self.assertEqual(result["facial"], "None")
            self.assertEqual(result["hair"], "Short")
        self.assertEqual(skins, {"Porcelain"})

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
