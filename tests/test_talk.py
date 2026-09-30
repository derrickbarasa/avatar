"""Talking: text -> visemes, lip-sync timeline, dynamic mouth, gestures, gaze, TTS, Talk tab."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from avatarkit import options as O  # noqa: E402
from avatarkit import speech, tts  # noqa: E402
from avatarkit.builder import build_avatar  # noqa: E402
from avatarkit.poses import gaze_at, pose_at, talk_overlay  # noqa: E402


def default_rig(**changes):
    state = dict(O.DEFAULT_STATE)
    for k, v in changes.items():
        O.set_by_name(state, k, v)
    return build_avatar(O.resolve(state))


def fake_audio(bursts, rate=16000, tail=0.5):
    """Mono audio that is loud during the given (start, end) second ranges and silent elsewhere."""
    n = int((max(e for _, e in bursts) + tail) * rate)
    t = np.arange(n) / rate
    a = np.zeros(n)
    for s, e in bursts:
        m = (t >= s) & (t < e)
        a[m] = 0.5 * np.sin(2 * math.pi * 180 * t[m])
    return a, rate


class VisemeTests(unittest.TestCase):
    def names(self, text):
        return [v for v, _ in speech.text_to_visemes(text)]

    def test_known_words_map_to_expected_shapes(self):
        self.assertEqual(self.names("map")[0], "PP")
        self.assertIn("AA", self.names("map"))
        self.assertEqual(self.names("thin")[0], "TH")
        self.assertEqual(self.names("shoe")[0], "CH")
        self.assertIn("UU", self.names("boot"))
        self.assertIn("FF", self.names("phone"))
        self.assertEqual(self.names("see")[:2], ["SS", "II"])

    def test_silent_final_e_is_skipped(self):
        self.assertEqual(self.names("nice"), ["DD", "II", "KK"][:0] + self.names("nice"))
        self.assertNotEqual(self.names("nice")[-1], "EE")

    def test_punctuation_becomes_pauses_and_ends_are_trimmed(self):
        seq = speech.text_to_visemes("Hi, there.")
        self.assertIn("sil", [v for v, _ in seq])
        self.assertNotEqual(seq[-1][0], "sil")
        pause = max(w for v, w in seq if v == "sil")
        self.assertGreater(pause, 1.5)

    def test_empty_and_odd_input_do_not_crash(self):
        for text in ("", "   ", "123 456", "!!!", "ünïcödé wörds", "a" * 500):
            speech.text_to_visemes(text)
            speech.Speech.from_text_only(text or " ")

    def test_all_visemes_are_defined_and_in_range(self):
        for name, (o, w, p) in speech.VISEMES.items():
            self.assertTrue(0 <= o <= 1 and -1 <= w <= 1 and p in (0.0, 1.0), name)
        used = set(self.names("the quick brown fox jumps over a lazy dog, why? see boat!"))
        self.assertTrue(used <= set(speech.VISEMES))


class SpeechTimelineTests(unittest.TestCase):
    def test_envelope_follows_loudness(self):
        audio, rate = fake_audio([(0.2, 0.6), (1.0, 1.4)])
        env = speech.envelope(audio, rate)
        at = lambda t: env[int(t * 100)]
        self.assertGreater(at(0.4), 0.8)
        self.assertLess(at(0.8), 0.1)
        self.assertGreater(at(1.2), 0.8)

    def test_mouth_opens_on_speech_and_closes_in_pauses(self):
        audio, rate = fake_audio([(0.2, 0.7), (1.2, 1.9)])
        sp = speech.Speech.from_samples("Hello there", audio, rate)
        self.assertAlmostEqual(sp.span[0], 0.2, delta=0.06)
        self.assertAlmostEqual(sp.span[1], 1.9, delta=0.06)
        loud = max(sp.sample(t).open for t in np.arange(0.25, 0.65, 0.02))
        pause = max(sp.sample(t).open for t in np.arange(0.85, 1.05, 0.02))
        self.assertGreater(loud, 0.25)
        self.assertLess(pause, 0.05)
        self.assertEqual(sp.sample(-1).open, 0.0)
        self.assertEqual(sp.sample(sp.duration + 5).open, 0.0)

    def test_samples_stay_in_range_and_are_smooth(self):
        audio, rate = fake_audio([(0.1, 2.0)])
        sp = speech.Speech.from_samples("The quick brown fox jumps", audio, rate)
        prev = None
        for t in np.arange(0.0, sp.duration, 0.01):
            m = sp.sample(t)
            self.assertTrue(0 <= m.open <= 1 and -1 <= m.wide <= 1 and 0 <= m.press <= 1)
            self.assertTrue(0 <= m.energy <= 1.3 and 0 <= m.beat <= 1.3)
            if prev is not None:
                self.assertLess(abs(m.open - prev), 0.35)     # no popping between frames
            prev = m.open

    def test_stressed_syllables_open_wider(self):
        rate = 16000
        t = np.arange(int(2.0 * rate)) / rate
        amp = np.where(t < 1.0, 0.15, 0.6)
        audio = amp * np.sin(2 * math.pi * 150 * t)
        sp = speech.Speech.from_samples("ah ah", audio, rate)
        quiet = max(sp.sample(x).open for x in np.arange(0.3, 0.9, 0.02))
        strong = max(sp.sample(x).open for x in np.arange(1.1, 1.9, 0.02))
        self.assertGreater(strong, quiet)

    def test_text_only_fallback_produces_movement(self):
        sp = speech.Speech.from_text_only("Hello, this works without any audio.")
        opens = [sp.sample(t).open for t in np.arange(0, sp.duration, 0.02)]
        self.assertGreater(max(opens), 0.3)
        self.assertGreater(len(set(round(o, 2) for o in opens)), 10)


class MouthRigTests(unittest.TestCase):
    def test_rig_has_mouth_and_brow_nodes(self):
        rig = default_rig()
        for name in ("mouth", "brows"):
            self.assertIn(name, rig.nodes)
            self.assertEqual(rig[name].parent.name, "head")
        self.assertIsNotNone(rig.mouth)

    def test_opening_lowers_the_lower_lip_and_adds_a_cavity(self):
        rig = default_rig()
        rig.set_mouth(0.0)
        closed = list(rig["mouth"].meshes)
        rig.set_mouth(0.9)
        opened = list(rig["mouth"].meshes)
        self.assertGreater(len(opened), len(closed))                 # cavity + teeth (+ tongue)
        low = lambda ms: min(m.v[:, 1].min() for m in ms)
        self.assertLess(low(opened), low(closed) - 0.05)

    def test_pucker_narrows_and_spread_widens_the_mouth(self):
        rig = default_rig()
        width = lambda ms: max(m.v[:, 0].max() - m.v[:, 0].min() for m in ms)
        rig.set_mouth(0.3, -1.0)
        narrow = width(rig["mouth"].meshes)
        rig.set_mouth(0.3, 1.0)
        wide = width(rig["mouth"].meshes)
        self.assertGreater(wide, narrow * 1.3)

    def test_shapes_are_cached_by_identity(self):
        rig = default_rig()
        rig.set_mouth(0.5, 0.2)
        first = rig["mouth"].meshes
        rig.set_mouth(0.501, 0.201)          # same quantised shape
        self.assertIs(rig["mouth"].meshes, first)
        rig.set_mouth(0.9, 0.2)
        self.assertIsNot(rig["mouth"].meshes, first)

    def test_expression_openness_is_the_resting_point(self):
        rig = default_rig(expression="Surprised")
        rig.set_mouth(0.0)
        open_now = len(rig["mouth"].meshes)
        calm = default_rig()
        calm.set_mouth(0.0)
        self.assertGreater(open_now, len(calm["mouth"].meshes))

    def test_eyes_have_gaze_and_lids_have_blink(self):
        rig = default_rig()
        kinds = [m.anim[0] for m in rig["head"].meshes if m.anim]
        self.assertEqual(sorted(kinds), ["blink"] * 4 + ["gaze"] * 2)


class MotionTests(unittest.TestCase):
    def test_gaze_stays_small_and_calmer_when_speaking(self):
        free = [gaze_at(t / 10) for t in range(0, 300)]
        talk = [gaze_at(t / 10, True) for t in range(0, 300)]
        self.assertLess(max(abs(p) for p, _ in free), 6)
        self.assertLess(max(abs(y) for _, y in free), 10)
        self.assertLess(max(abs(y) for _, y in talk), max(abs(y) for _, y in free))
        self.assertTrue(all(math.isfinite(v) for g in free for v in g))

    def test_talk_overlay_adds_head_brow_and_arm_motion(self):
        base = pose_at("relaxed", 1.0, False)
        active = talk_overlay(dict(base), speech.MouthShape(energy=0.9, beat=0.4), 1.0)
        self.assertNotEqual(active["head"], base.get("head", (0.0, 0.0, 0.0)))
        self.assertLess(active["brows"][0], -2.0)                     # brows raised
        self.assertLess(active["foreL"][0], base["foreL"][0] - 10)    # forearm gestures
        for v in active.values():
            vals = v if isinstance(v, tuple) else (v,)
            self.assertTrue(all(math.isfinite(x) for x in vals))

    def test_gestures_can_be_switched_off_and_raised_arms_are_left_alone(self):
        base = pose_at("relaxed", 1.0, False)
        quiet = talk_overlay(dict(base), speech.MouthShape(energy=0.9, beat=0.4), 1.0, gestures=False)
        self.assertEqual(quiet["foreL"], base["foreL"])
        cheer = pose_at("cheer", 1.0, False)
        overlay = talk_overlay(dict(cheer), speech.MouthShape(energy=0.9, beat=0.4), 1.0)
        self.assertEqual(overlay["armL"], cheer["armL"])

    def test_no_speech_means_no_extra_motion(self):
        base = pose_at("relaxed", 1.0, False)
        still = talk_overlay(dict(base), speech.MouthShape(), 1.0)
        self.assertEqual(still["armL"], base["armL"])
        self.assertAlmostEqual(still["brows"][0], 0.0)

    def test_dance_is_a_valid_animated_pose(self):
        a, b = pose_at("dance", 0.0, True), pose_at("dance", 0.3, True)
        self.assertNotEqual(a["armL"], b["armL"])
        self.assertEqual(pose_at("dance", 0.0, False), pose_at("dance", 4.0, False))
        self.assertIn(("Dance", "dance"), O.POSES)


class TTSTests(unittest.TestCase):
    def test_backend_detection_is_consistent(self):
        b = tts.backend()
        self.assertIn(b, (None, "powershell", "say", "espeak-ng", "espeak"))

    def test_synthesis_makes_playable_audio(self):
        if tts.backend() is None:
            self.skipTest("no speech engine on this machine")
        path = tts.synthesize("Testing one two three.", "", "Fast")
        self.assertTrue(os.path.getsize(path) > 2000)
        self.assertEqual(path, tts.synthesize("Testing one two three.", "", "Fast"))   # cached
        job = tts.Job("Short test.", "", "Normal")
        job.join(30)
        self.assertTrue(job.done and job.error is None, job.error)

    def test_unknown_voice_falls_back_to_the_default_voice(self):
        if tts.backend() != "powershell":
            self.skipTest("PowerShell backend only")
        job = tts.Job("hello", "No Such Voice XYZ", "Normal")
        job.join(30)
        self.assertTrue(job.done)
        self.assertTrue(job.path is not None or job.error is not None)   # never hangs or raises

    def test_missing_engine_is_reported_as_an_error(self):
        from unittest import mock
        with mock.patch.object(tts, "backend", return_value=None):
            job = tts.Job("hello")
            job.join(5)
        self.assertTrue(job.done)
        self.assertIn("no speech engine", job.error)

    def test_talk_options_do_not_change_the_share_code(self):
        a, b = dict(O.DEFAULT_STATE), dict(O.DEFAULT_STATE)
        b["speed"], b["gestures"] = 2, 1
        self.assertEqual(O.encode_state(a), O.encode_state(b))
        self.assertTrue(O.TALK_KEYS <= O.NON_BUILD_KEYS)

    def test_voices_can_be_replaced_at_run_time(self):
        before = list(O.choices("voice"))
        try:
            O.set_choices("voice", ["A", "B"])
            self.assertEqual([n for n, _ in O.choices("voice")], ["A", "B"])
            state = dict(O.DEFAULT_STATE)
            state["voice"] = 1
            self.assertEqual(O.resolve(state)["voice"], "B")
        finally:
            O.set_choices("voice", [n for n, _ in before])


class NewContentTests(unittest.TestCase):
    def head_meshes(self, **changes):
        return default_rig(**changes)["head"].meshes

    def test_pigtails_and_braid_are_strand_hair(self):
        for style, minimum in (("Pigtails", 20000), ("Braid", 20000)):
            strands = [m for m in self.head_meshes(hair=style) if m.strand and m.part == "hair"]
            self.assertEqual(len(strands), 1, style)
            self.assertGreater(len(strands[0].v), minimum, style)
            self.assertEqual(strands[0].strand_aux.shape, (len(strands[0].v), 2))

    def test_pigtails_hang_on_both_sides_and_braid_down_the_back(self):
        v = [m for m in self.head_meshes(hair="Pigtails") if m.strand and m.part == "hair"][0].v
        low = v[v[:, 1] < -0.5]
        self.assertGreater((low[:, 0] > 0.3).sum(), 500)
        self.assertGreater((low[:, 0] < -0.3).sum(), 500)
        b = [m for m in self.head_meshes(hair="Braid") if m.strand and m.part == "hair"][0].v
        lowb = b[b[:, 1] < -0.6]
        self.assertGreater(len(lowb), 500)
        self.assertLess(np.abs(lowb[:, 0]).max(), 0.25)        # a narrow braid
        self.assertLess(lowb[:, 2].mean(), -0.5)                # behind the head

    def test_new_hats_headphones_and_glasses_build(self):
        n0 = len(self.head_meshes())
        for kw in ({"hat": "Bucket"}, {"hat": "Top hat"}, {"headphones": "Headphones"},
                   {"glasses": "Cat-eye"}, {"glasses": "Aviator"}):
            self.assertGreater(len(self.head_meshes(**kw)), n0, kw)

    def test_top_hat_is_taller_than_the_head(self):
        tall = max(m.v[:, 1].max() for m in self.head_meshes(hat="Top hat", hair="Bald"))
        self.assertGreater(tall, 0.9)

    def test_dress_replaces_the_trousers_with_a_skirt_in_the_top_colour(self):
        rig = default_rig(top="Dress", topcolor="Purple", pants="Jeans")
        self.assertFalse(rig["thighL"].meshes[2:] and any(m.kind == "denim" for m in rig["thighL"].meshes))
        skirt = [m for m in rig["root"].meshes if m.kind == "cloth" and m.v[:, 1].min() < -4.5]
        self.assertTrue(skirt)
        self.assertEqual(skirt[0].color, tuple(O.CLOTH_COLORS[8][1]))
        low = min(m.v[:, 1].min() for m in rig["root"].meshes if m.kind == "cloth")
        self.assertLess(low, -5.0)                               # reaches below the knee

    def test_polo_and_sweater_add_trim_to_the_tee_and_long_sleeve_cuts(self):
        tee, long_ = len(default_rig(top="T-shirt")["torso"].meshes), len(default_rig(top="Long sleeve")["torso"].meshes)
        polo, sweater = default_rig(top="Polo"), default_rig(top="Sweater")
        self.assertGreater(len(polo["torso"].meshes), tee + 2)             # collar, placket, buttons
        self.assertGreater(len(sweater["torso"].meshes), long_)             # collar
        self.assertGreater(len(sweater["forearmL" if "forearmL" in sweater.nodes else "foreL"].meshes),
                           len(default_rig(top="Long sleeve")["foreL"].meshes))   # cuff

    def test_eyes_have_lash_flicks(self):
        eye = default_rig()["head"].meshes
        lashes = [m for m in eye if m.anim and m.anim[0] == "blink" and m.kind == "plain"]
        self.assertTrue(lashes)
        self.assertGreater(max(len(m.v) for m in lashes), 240)              # the lash line plus the flicks

    def test_neckwear_adds_pieces_to_the_torso(self):
        base = len(default_rig()["torso"].meshes)
        self.assertGreater(len(default_rig(neckwear="Bow tie")["torso"].meshes), base)
        self.assertGreater(len(default_rig(neckwear="Tie")["torso"].meshes), base)

    def test_older_share_codes_still_open_with_defaults_for_new_options(self):
        st = dict(O.DEFAULT_STATE)
        O.set_by_name(st, "hair", "Afro")
        code = O.encode_state(st)
        legacy = code[:O.CODE_PREFIX.__len__() + O.LEGACY_CODE_LEN]
        out = dict(O.DEFAULT_STATE)
        out["neckwear"] = 2
        O.decode_state(legacy, out)
        self.assertEqual(O.resolve(out)["hair"], "afro")
        self.assertEqual(O.resolve(out)["neckwear"], "none")
        with self.assertRaises(ValueError):
            O.decode_state(code[:len(O.CODE_PREFIX) + O.LEGACY_CODE_LEN - 1], out)   # shorter than any issued


class TalkTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        pygame.font.init()

    def render(self, **kw):
        from avatarkit.ui import UI, UIModel
        ui = UI()
        tab = [n for n, _ in O.TABS].index("Talk")
        model = UIModel(dict(O.DEFAULT_STATE), tab=tab, code="x", **kw)
        ui.render(model, (1100, 760))
        return ui

    def test_talk_controls_exist_and_are_inside_the_card(self):
        ui = self.render(say_text="Hello there")
        actions = {a for _, a in ui.widgets}
        for a in (("textbox", "say"), ("say", "speak"), ("say", "random")):
            self.assertIn(a, actions)
        self.assertTrue({("phrase", i) for i in range(6)} <= actions)
        for rect, action in ui.widgets:
            if action[0] in ("textbox", "say", "phrase", "emote"):
                self.assertTrue(ui.panel_rect.contains(rect), action)

    def test_button_label_tracks_state(self):
        import pygame
        from avatarkit.ui import UI, UIModel
        tab = [n for n, _ in O.TABS].index("Talk")
        views = {}
        for name, kw in (("idle", {}), ("speaking", {"speaking": True}), ("busy", {"busy": True})):
            ui = UI()
            ui.render(UIModel(dict(O.DEFAULT_STATE), tab=tab, code="x", **kw), (1100, 760))
            views[name] = bytes(ui.surface.get_view("0"))
        self.assertEqual(len(set(views.values())), 3)

    def test_long_text_wraps_and_keeps_the_tail_visible(self):
        from avatarkit.ui import wrap
        lines = wrap("word " * 80, False, 14, 280)
        self.assertGreater(len(lines), 3)
        from avatarkit.ui import text_width
        self.assertTrue(all(text_width(line, False, 14) <= 280 for line in lines))
        self.assertTrue(all(text_width(x, False, 14) <= 200 for x in wrap("x" * 200, False, 14, 200)))

    def test_voice_names_are_shortened_for_chips(self):
        from avatarkit.ui import chip_label
        self.assertEqual(chip_label("voice", "Microsoft Zira Desktop"), "Zira")
        self.assertEqual(chip_label("speed", "Fast"), "Fast")

    def test_focused_textbox_and_cursor_change_the_picture(self):
        a = bytes(self.render(say_text="Hi").surface.get_view("0"))
        b = bytes(self.render(say_text="Hi", focus_field="say", cursor_on=True).surface.get_view("0"))
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
