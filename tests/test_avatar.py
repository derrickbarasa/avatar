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
        for bad in ("", "hello", "AV2-abc", O.encode_state(state)[:O.LEGACY_CODE_LEN + 3], "AV2-" + "z" * len(O.CODE_KEYS)):
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
        from avatarkit.head import cached_head
        head = cached_head((0.9, 0.7, 0.6), 0.28, 1.0)           # the default avatar's head
        rng = np.random.default_rng(0)
        for hair in ("Long", "Bob", "Short", "Ponytail", "Curly"):
            v = self.strand_meshes(hair)[0].v.astype(float)
            v = v[rng.choice(len(v), 3000, replace=False)]
            near = ((v[:, None, :] - head.P[None]) ** 2).sum(2).argmin(1)
            depth = ((v - head.P[near]) * head.mesh.n[near]).sum(1)     # < 0: under the skin
            self.assertLess((depth < -0.03).mean(), 0.01, hair)

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
        rig = default_rig(**changes)          # brows live in their own node so they can be raised
        return [m for n in ("head", "brows", "browL", "browR") for m in rig[n].meshes if m.strand]

    def test_beard_mustache_and_brows_are_strands(self):
        none = self.head_strands(hair="Bald")                       # brows only
        stache = self.head_strands(hair="Bald", facial="Mustache")
        beard = self.head_strands(hair="Bald", facial="Beard")
        self.assertEqual(len(none), 2)              # one brow mesh per side
        self.assertEqual(len(stache), 3)
        self.assertEqual(len(beard), 3)
        vertex_count = lambda ms: sum(len(m.v) for m in ms)
        self.assertGreater(vertex_count(beard), vertex_count(stache) * 3)

    def test_brow_thickness_changes_hair_count(self):
        thin = sum(len(m.v) for m in self.head_strands(hair="Bald", brows="Thin"))
        thick = sum(len(m.v) for m in self.head_strands(hair="Bald", brows="Thick"))
        self.assertGreater(thick, thin * 1.5)

    def test_facial_strands_stay_outside_the_skin(self):
        from avatarkit.head import cached_head
        from avatarkit.strands import Skin
        rig = default_rig(hair="Bald", facial="Beard")
        head = cached_head(tuple(O.resolve(dict(O.DEFAULT_STATE))["skin"]), 0.28, 1.0)
        skin = Skin(head, np.ones(len(head.P), bool))
        beard = [m for m in rig["head"].meshes if m.strand and m.part == "facial"][0]
        v = beard.v.astype(float)[::7]
        j = skin.nearest(v)
        depth = ((v - skin.v[j]) * skin.n[j]).sum(1)
        self.assertGreater(np.percentile(depth, 2), -0.01)   # (almost) nothing dips into the face

    def test_beard_hangs_below_the_chin(self):
        beard = [m for m in self.head_strands(hair="Bald", facial="Beard") if m.part == "facial"][0]
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


class LimbShapeTests(unittest.TestCase):
    def test_muscle_shaping_keeps_joints_round_and_bulges_between(self):
        from avatarkit import body
        keys = body.arm_keys(1, 1.0, 1.0, 1.0)
        plain = body.catmull(keys, 41, 0.0, 4.0)
        shaped = body.arm_curve(keys, 41, 0.0, 4.0)
        for k in (0, 20, 40):                                   # shoulder, elbow, wrist
            self.assertAlmostEqual(shaped[k, 3], plain[k, 3], places=6)
        self.assertGreater(shaped[10, 3], plain[10, 3] * 1.05)   # biceps
        self.assertGreater(shaped[27, 3], plain[27, 3] * 1.05)   # forearm
        np.testing.assert_allclose(shaped[:, :3], plain[:, :3])


def extent(rig, node, axis):
    """(min, max) of a node's vertices along an axis (0 = x, 1 = y)."""
    verts = np.concatenate([m.v for m in rig[node].meshes])
    return float(verts[:, axis].min()), float(verts[:, axis].max())


class ProportionTests(unittest.TestCase):
    def test_shoulders_and_hips_change_the_silhouette(self):
        narrow = default_rig(shoulders="Narrow", hips="Narrow")
        wide = default_rig(shoulders="Wide", hips="Wide")
        span = lambda rig, node: np.subtract(*extent(rig, node, 0)[::-1])
        self.assertGreater(span(wide, "torso"), span(narrow, "torso") * 1.1)
        self.assertGreater(span(wide, "root"), span(narrow, "root") * 1.05)

    def test_head_size_scales_the_head_and_everything_on_it(self):
        small, large = default_rig(headsize="Small", hair="Short"), default_rig(headsize="Large", hair="Short")
        height = lambda rig: np.subtract(*extent(rig, "head", 1)[::-1])
        self.assertGreater(height(large), height(small) * 1.15)
        base = default_rig(hair="Short")
        self.assertEqual(len(base["head"].meshes), len(large["head"].meshes))      # nothing lost

    def test_a_long_neck_raises_the_head_and_the_lips_follow(self):
        short, long_ = default_rig(neck="Short"), default_rig(neck="Long")
        self.assertGreater(extent(long_, "head", 1)[1], extent(short, "head", 1)[1] + 0.12)
        lips = lambda rig: np.concatenate([m.v for m in rig["mouth"].meshes])[:, 1].mean()
        self.assertGreater(lips(long_), lips(short) + 0.12)
        long_.set_mouth(0.8)                                                        # still follows when talking
        self.assertGreater(np.concatenate([m.v for m in long_["mouth"].meshes])[:, 1].mean(), lips(short))

    def test_resizing_never_changes_the_cached_shared_meshes(self):
        a = default_rig(hair="Long")
        before = extent(a, "head", 1)
        default_rig(hair="Long", headsize="Large", neck="Long")
        self.assertEqual(extent(default_rig(hair="Long"), "head", 1), before)

    def test_share_codes_carry_the_new_options_and_old_codes_still_load(self):
        state = dict(O.DEFAULT_STATE)
        for key, name in (("shoulders", "Wide"), ("hips", "Narrow"), ("headsize", "Large"), ("neck", "Long")):
            O.set_by_name(state, key, name)
        again = dict(O.DEFAULT_STATE)
        O.decode_state(O.encode_state(state), again)
        for key in ("shoulders", "hips", "headsize", "neck"):
            self.assertEqual(again[key], state[key], key)
        old = O.encode_state(dict(O.DEFAULT_STATE))[:len(O.CODE_PREFIX) + 38]
        O.decode_state(old, again)                       # 38-option codes from before the proportions
        self.assertEqual(again["headsize"], O.DEFAULT_STATE["headsize"])

    def test_quality_is_a_view_setting_not_part_of_the_avatar(self):
        self.assertNotIn("quality", O.CODE_KEYS)
        self.assertIn("quality", O.NON_BUILD_KEYS)


class ClipTests(unittest.TestCase):
    def test_clips_cover_the_pose_and_every_emote(self):
        from avatarkit import clips, poses
        found = clips.all_clips("walk")
        self.assertEqual(len(found), 1 + len(poses.EMOTES))
        for name, times, frames in found:
            self.assertEqual(len(times), len(frames), name)
            self.assertEqual(times[0], 0.0)
            self.assertTrue(all(b > a for a, b in zip(times, times[1:])), name)

    def test_animated_glb_has_matching_tracks(self):
        from avatarkit import clips
        from avatarkit.exporters import export_skinned
        rig = default_rig(hair="Bald")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.glb")
            export_skinned(rig, path, pose_at("relaxed"), clips=clips.all_clips("walk"))
            data = open(path, "rb").read()
        doc = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
        self.assertEqual(len(doc["animations"]), 10)
        for anim in doc["animations"]:
            for ch in anim["channels"]:
                sampler = anim["samplers"][ch["sampler"]]
                self.assertEqual(doc["accessors"][sampler["input"]]["count"],
                                 doc["accessors"][sampler["output"]]["count"])
                self.assertLess(ch["target"]["node"], len(doc["nodes"]))
        walk = doc["animations"][0]
        paths = {ch["target"]["path"] for ch in walk["channels"]}
        self.assertEqual(paths, {"rotation", "translation"})              # limbs swing, the body bobs

    def test_quaternions_stay_on_one_side_so_playback_never_flips(self):
        from avatarkit import clips
        from avatarkit.exporters import _animations, _Buffer
        rig = default_rig(hair="Bald")
        buf = _Buffer()
        index = {n: i for i, n in enumerate(rig.nodes)}
        out = _animations(buf, rig, index, [clips.pose_clip("dance")], 1.0, np.zeros(3))
        self.assertTrue(out)
        for ch, samp in zip(out[0]["channels"], out[0]["samplers"]):
            if ch["target"]["path"] == "rotation":
                acc = buf.accessors[samp["output"]]
                view = buf.views[acc["bufferView"]]
                q = np.frombuffer(bytes(buf.data[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]),
                                  np.float32).reshape(-1, 4)
                self.assertTrue(np.all(np.einsum("ij,ij->i", q[1:], q[:-1]) >= 0))


class RealismTests(unittest.TestCase):
    """The body is shaped like a person, not built from balls and tubes."""

    def test_the_head_is_not_a_sphere(self):
        from avatarkit.head import cached_head
        head = cached_head((0.9, 0.7, 0.6), 0.28, 1.0)
        width = lambda y: np.ptp(head.X[np.abs(head.Y - y) < 0.03])
        self.assertLess(width(-0.50), 0.8 * width(0.05))          # the jaw is narrower than the cranium
        depth = lambda y: head.Z[np.abs(head.Y - y) < 0.03].max() - head.Z[np.abs(head.Y - y) < 0.03].min()
        self.assertLess(depth(-0.55), 0.7 * depth(0.1))           # ... and the chin is shallower than the skull
        back = head.Z[np.abs(head.Y - 0.2) < 0.03].min()
        front = head.Z[(np.abs(head.Y - 0.35) < 0.03) & (np.abs(head.X) < 0.05)].max()
        self.assertLess(back, -front)                              # the skull reaches further back than the brow

    def test_limb_sections_are_ellipses_that_can_be_offset(self):
        from avatarkit.mesh import limb
        line = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, -2.0, 0.0]])
        m = limb(line, np.full(3, 0.1), np.full(3, 0.3), (1, 1, 1), off=np.full(3, 0.05), sides=24)
        self.assertAlmostEqual(float(np.ptp(m.v[:, 0])), 0.2, places=2)           # side to side
        self.assertAlmostEqual(float(np.ptp(m.v[:, 2])), 0.6, places=2)           # front to back
        self.assertAlmostEqual(float((m.v[:, 2].max() + m.v[:, 2].min()) / 2), 0.05, places=2)

    def test_arms_and_legs_vary_along_their_length(self):
        rig = default_rig(top="Tank top", pants="Shorts")
        for name in ("armL", "foreL", "thighL", "shinL"):
            mesh = rig[name].meshes[0]
            ys = np.unique(np.round(mesh.v[:, 1], 3))
            widths = np.array([np.ptp(mesh.v[np.isclose(mesh.v[:, 1], y, atol=1e-3), 0]) for y in ys])
            self.assertGreater(widths.max(), 1.3 * widths[widths > 0].min(), name)      # taper and muscle, not a tube

    def test_calf_bulges_backwards_and_forearm_flattens_toward_the_wrist(self):
        rig = default_rig(top="Tank top", pants="Shorts")
        calf = rig["shinL"].meshes[0].v
        centre = (calf[:, 2].max() + calf[:, 2].min()) / 2
        self.assertLess(calf[calf[:, 1] > calf[:, 1].min() + 0.5][:, 2].min(), centre)
        fore = rig["foreL"].meshes[0].v
        low = fore[fore[:, 1] < fore[:, 1].min() + 0.15]
        self.assertGreater(np.ptp(low[:, 2]), 1.15 * np.ptp(low[:, 0]))          # wider front to back than across

    def test_feet_join_the_ankle_and_stand_on_the_ground(self):
        for shoes in ("Barefoot", "Sneakers", "Boots"):
            rig = default_rig(shoestyle=shoes)
            foot = rig["footL"]
            top = max(m.v[:, 1].max() for m in foot.meshes)
            bottom = min(m.v[:, 1].min() for m in foot.meshes)
            self.assertGreater(top, foot.pivot[1] - 0.05, shoes)       # reaches up to the ankle: no gap under the shin
            self.assertAlmostEqual(bottom, rig.ground_y, delta=0.1, msg=shoes)

    def test_beard_covers_sideburns_and_chin_but_not_the_lips(self):
        from avatarkit.head import MOUTH_Y
        rig = default_rig(facial="Beard")
        beard = [m for m in rig["head"].meshes if m.strand and m.part == "facial"][0]
        roots = beard.v.reshape(beard.n_strands, -1, 3)[:, 0].astype(float)
        self.assertGreater(((np.abs(roots[:, 0]) > 0.38) & (roots[:, 1] > -0.25)).sum(), 100)    # sideburns
        self.assertGreater((roots[:, 1] < -0.52).sum(), 300)                                        # chin and jaw
        on_lips = ((roots[:, 0] / 0.17) ** 2 + ((roots[:, 1] - MOUTH_Y) / 0.05) ** 2 < 1) & (roots[:, 2] > 0.3)
        self.assertLess((on_lips & (roots[:, 1] < MOUTH_Y - 0.01)).sum(), 20)    # (the moustache sits above the lips)
        self.assertLess(roots[:, 1].max(), 0.1)                     # nothing grows up the forehead

    def test_hair_under_a_hat_is_only_what_hangs_below_its_edge(self):
        bare = default_rig(hair="Short")["head"].meshes
        hatted = default_rig(hair="Short", hat="Cap")["head"].meshes
        crown = lambda meshes: max(m.v.reshape(m.n_strands, -1, 3)[:, 0, 1].max() for m in meshes
                                   if m.strand and m.part == "hair")
        self.assertGreater(crown(bare), 0.5)
        for m in hatted:
            if m.strand and m.part == "hair":
                roots = m.v.reshape(m.n_strands, -1, 3)[:, 0]
                self.assertLess(roots[:, 1].max(), 0.3)            # no roots up on the covered crown

    def test_torso_has_a_waist_a_chest_and_buttocks(self):
        from avatarkit.body import Torso
        torso = Torso(O.resolve(dict(O.DEFAULT_STATE))["bodytype"], 1.0)
        a, front, back = torso.dims(np.array([-1.95, -3.0, -3.9]))        # chest, waist, hips
        self.assertLess(a[1], 0.85 * a[0])                                  # the waist is narrower than the chest
        self.assertGreater(front[0], front[1] + 0.1)                        # the chest stands out in front
        self.assertGreater(back[2], front[2] + 0.05)                        # the buttocks push the back out
        shoulder, _, _ = torso.dims(np.array([-1.15, -1.40]))
        self.assertLess(shoulder[0], 0.7 * shoulder[1])                     # the trapezius slopes out to the shoulder

    def test_skirt_and_dress_cover_the_hips_for_every_body_type(self):
        for body in ("Masculine", "Neutral", "Feminine"):
            for top, pants in (("Dress", "Jeans"), ("T-shirt", "Skirt")):
                root = default_rig(bodytype=body, top=top, pants=pants)["root"]
                skin, skirt = root.meshes[0].v, root.meshes[-1].v        # the pelvis, and the skirt added last
                for axis in (0, 2):                                      # across and front to back
                    self.assertGreaterEqual(np.ptp(skirt[:, axis]), np.ptp(skin[:, axis]), (body, top, axis))

    def test_long_hair_rests_on_the_shoulders_instead_of_passing_through_the_shirt(self):
        from avatarkit.body import LIFT
        from avatarkit.strands import _TORSO, TORSO_POWER
        for hair in ("Long", "Bob"):
            v = default_rig(hair=hair)["head"].meshes
            v = np.concatenate([m.v for m in v if m.strand and m.part == "hair"]).astype(float)
            v = v[v[:, 1] < -0.75]
            self.assertGreater(len(v), 1000, hair)                        # plenty of hair down there
            a, front, back = _TORSO.dims(v[:, 1] - LIFT, -0.03)       # tips may drift a few mm after the collision
            k = (np.abs(v[:, 0]) / a) ** TORSO_POWER + (np.abs(v[:, 2]) / np.where(v[:, 2] > 0, front, back)) ** TORSO_POWER
            self.assertLess((k < 1.0).mean(), 0.01, hair)                 # <1% of it is more than 3 cm inside the body

    def test_hand_has_a_palm_that_widens_to_the_knuckles_and_tapering_fingers(self):
        from avatarkit.body import FINGER_CODES, hand_parts
        palm, fingers = hand_parts((1.0, -4.0, 0.2), 1, (0.9, 0.7, 0.6))
        self.assertEqual([f[0] for f in fingers], list(FINGER_CODES))
        wrist = palm.v[np.abs(palm.v[:, 1] - -3.97) < 0.04]
        knuckle = palm.v[np.abs(palm.v[:, 1] - (-4.0 - 0.165 * 1.5)) < 0.04]
        self.assertGreater(np.ptp(knuckle[:, 2]), 0.9 * np.ptp(wrist[:, 2]))      # the width holds out to the knuckles
        for code, prox, dist, knuckle_at, joint_at in fingers:
            self.assertGreater(dist.v[:, 1].min(), -4.0 - 0.6)                    # a sensible length
            self.assertLess(dist.v[:, 1].min(), joint_at[1])                      # it reaches past its joint
        thumb = fingers[0]
        self.assertGreater(thumb[3][2], 0.2 + 0.05)                               # the thumb starts on the front edge

    def test_hands_scale_with_the_body(self):
        from avatarkit.body import hand_parts
        small = hand_parts((0, 0, 0), 1, (0.9, 0.7, 0.6), 0.8)[0]
        big = hand_parts((0, 0, 0), 1, (0.9, 0.7, 0.6), 1.2)[0]
        self.assertGreater(np.ptp(big.v[:, 1]), 1.3 * np.ptp(small.v[:, 1]))

    def test_new_options_were_appended_so_old_share_codes_still_mean_the_same(self):
        for key, old_last in (("facial", "beard"), ("hair", "braid"), ("hat", "tophat"), ("top", "sweater"),
                              ("pants", "skirt")):
            values = [v for _, v in O.choices(key)]
            self.assertGreater(len(values), values.index(old_last) + 1, key)      # something was added after the old last
        st = dict(O.DEFAULT_STATE)
        st.update(hair=13, facial=2, hat=4, top=7, pants=2)                       # the highest values a v1 code could hold
        again = dict(O.DEFAULT_STATE)
        O.decode_state(O.encode_state(st), again)
        self.assertEqual({k: again[k] for k in st}, st)
        self.assertEqual([n for n, _ in O.choices("hair")][:14], ["Bald", "Buzz", "Short", "Curly", "Long", "Bob", "Bangs",
                                                                  "Bun", "Ponytail", "Quiff", "Mohawk", "Afro", "Pigtails", "Braid"])

    def test_stubble_is_short_and_dense_and_a_goatee_stays_on_the_chin(self):
        def roots(kind):
            m = [m for m in default_rig(facial=kind)["head"].meshes if m.strand and m.part == "facial"][0]
            v = m.v.reshape(m.n_strands, -1, 3).astype(float)
            return v[:, 0], np.linalg.norm(v[:, -1] - v[:, 0], axis=1)
        beard_roots, beard_len = roots("Beard")
        stub_roots, stub_len = roots("Stubble")
        goat_roots, _ = roots("Goatee")
        self.assertLess(np.median(stub_len), 0.4 * np.median(beard_len))
        self.assertGreater(len(stub_roots), len(beard_roots))
        chin = goat_roots[goat_roots[:, 1] < -0.3]                                # (the moustache is above the lips)
        self.assertLess(np.abs(chin[:, 0]).max(), 0.3)                            # nothing out along the jaw
        self.assertGreater(len(beard_roots[np.abs(beard_roots[:, 0]) > 0.35]), 100)

    def test_undercut_has_hair_only_on_top_and_wavy_is_long(self):
        top = default_rig(hair="Undercut")["head"].meshes
        roots = np.concatenate([m.v.reshape(m.n_strands, -1, 3)[:, 0] for m in top if m.strand and m.part == "hair"])
        self.assertGreater(roots[:, 1].min(), 0.0)                                # nothing grows low on the sides or back
        wavy = np.concatenate([m.v for m in default_rig(hair="Wavy")["head"].meshes if m.strand and m.part == "hair"])
        self.assertLess(wavy[:, 1].min(), -0.8)

    def test_beret_and_headband_sit_on_the_head_and_a_turtleneck_covers_the_neck(self):
        for hat in ("Beret", "Headband"):
            rig = default_rig(hat=hat, hair="Bald")
            parts = [m for m in rig["head"].meshes if m.color == O.resolve(dict(O.DEFAULT_STATE))["hatcolor"]
                     or m.kind == "cloth"]
            self.assertTrue(parts, hat)
        plain = default_rig(top="Long sleeve")["torso"].meshes
        neck = default_rig(top="Turtleneck")["torso"].meshes
        self.assertGreater(len(neck), len(plain))                                 # the collar is extra geometry
        self.assertGreater(max(m.v[:, 1].max() for m in neck if m.kind == "cloth"),
                           max(m.v[:, 1].max() for m in plain if m.kind == "cloth") + 0.15)

    def test_leggings_hug_and_wide_trousers_hang_loose(self):
        def hem_width(pants):
            leg = [m for m in default_rig(pants=pants, shoestyle="Barefoot")["shinL"].meshes
                   if m.kind in ("denim", "cloth")][0].v
            return np.ptp(leg[leg[:, 1] < leg[:, 1].min() + 0.25, 0])
        self.assertLess(hem_width("Leggings"), hem_width("Jeans"))
        self.assertGreater(hem_width("Wide"), hem_width("Jeans"))

    def test_hair_rests_on_a_thick_hoodie_without_going_through_it(self):
        from avatarkit.body import LIFT
        from avatarkit.strands import _TORSO, TORSO_POWER
        v = default_rig(hair="Long", top="Hoodie")["head"].meshes
        v = np.concatenate([m.v for m in v if m.strand and m.part == "hair"]).astype(float)
        v = v[v[:, 1] < -0.75]
        a, front, back = _TORSO.dims(v[:, 1] - LIFT, 0.07 - 0.03)                   # the hoodie, less the usual 3 cm drift
        k = (np.abs(v[:, 0]) / a) ** TORSO_POWER + (np.abs(v[:, 2]) / np.where(v[:, 2] > 0, front, back)) ** TORSO_POWER
        self.assertLess((k < 1.0).mean(), 0.01)

    def test_ears_have_a_rim_a_bowl_and_a_lobe_and_flare_outward(self):
        from avatarkit.head import ear_meshes
        for side in (-1, 1):
            parts = ear_meshes(side, (0.9, 0.7, 0.6))
            self.assertEqual(len(parts), 4)
            outer = max(abs(m.v[:, 0]).max() for m in parts)
            self.assertGreater(outer, 0.5)                                         # sticks out past the skull's side
            self.assertTrue(all(np.sign(m.v[:, 0].mean()) == side for m in parts))


if __name__ == "__main__":
    unittest.main()
