"""Content packs and lip-sync in other languages."""
import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from avatarkit import options as O  # noqa: E402
from avatarkit import packs, speech, ui  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


class PackCase(unittest.TestCase):
    """Snapshots the shared option tables and restores them, so packs never leak between tests."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.saved = {
            "lists": {name: list(lst) for name, lst in packs.COLOR_GROUPS.items()},
            "presets": copy.deepcopy(O.PRESETS),
            "preset_choices": list(O.choices("preset")),
            "phrases": list(ui.PHRASES),
            "bgs": list(O.BACKGROUNDS),
        }
        self.addCleanup(self.restore)

    def restore(self):
        for name, lst in packs.COLOR_GROUPS.items():
            lst[:] = self.saved["lists"][name]
        O.PRESETS.clear()
        O.PRESETS.update(self.saved["presets"])
        O.choices("preset")[:] = self.saved["preset_choices"]
        ui.PHRASES[:] = self.saved["phrases"]
        O.BACKGROUNDS[:] = self.saved["bgs"]

    def write(self, name, data):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(data if isinstance(data, str) else json.dumps(data))
        return path


class PackTests(PackCase):
    def test_the_bundled_example_pack_loads_and_works(self):
        pack = packs.read_pack(os.path.join(ROOT, "examples", "packs", "neon.json"))
        summary = packs.apply_pack(pack)
        self.assertIn("Neon", summary)
        names = [n for n, _ in O.HAIR_COLORS]
        self.assertIn("Neon teal", names)
        self.assertIn("Rave", O.PRESETS)
        self.assertIn(("Rave", "Rave"), O.choices("preset"))
        self.assertIn("Club", [n for n, _ in O.BACKGROUNDS])
        self.assertIn("Rave on", [p[0] for p in ui.PHRASES])
        state = dict(O.DEFAULT_STATE)
        O.apply_preset(state, "Rave")
        self.assertEqual(O.resolve(state)["haircolor"], tuple(dict(O.HAIR_COLORS)["Neon teal"]))

    def test_pack_colours_can_be_chosen_and_encoded_in_share_codes(self):
        packs.apply_pack(packs.read_pack(self.write("a.json", {"colors": {"hair": [["Mint", [0.4, 1, 0.7]]]}})))
        state = dict(O.DEFAULT_STATE)
        O.set_by_name(state, "haircolor", "Mint")
        again = dict(O.DEFAULT_STATE)
        O.decode_state(O.encode_state(state), again)
        self.assertEqual(O.resolve(again)["haircolor"], (0.4, 1.0, 0.7))

    def test_colours_accept_0_255_and_0_1(self):
        self.assertEqual(packs._rgb([255, 0, 128]), (1.0, 0.0, 128 / 255))
        self.assertEqual(packs._rgb([0.5, 0.25, 1]), (0.5, 0.25, 1.0))
        for bad in ([1, 2], "red", [0, 0, "x"], [0, 0, 400], [-1, 0, 0], [True, 0, 0], None):
            with self.assertRaises(packs.PackError, msg=repr(bad)):
                packs._rgb(bad)

    def test_bad_packs_are_reported_and_never_stop_the_good_ones(self):
        self.write("good.json", {"name": "Good", "colors": {"eyes": [["Teal eyes", [0, 200, 200]]]}})
        self.write("broken.json", "{not json")
        self.write("list.json", "[1, 2, 3]")
        self.write("group.json", {"colors": {"socks": [["A", [0, 0, 0]]]}})
        self.write("preset.json", {"presets": {"Bad": {"hair": "Not A Style"}}})
        self.write("ignored.txt", "hello")
        summaries, errors = packs.load_all(self.tmp.name)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(len(errors), 4)
        self.assertIn("Teal eyes", [n for n, _ in O.EYE_COLORS])
        self.assertNotIn("Bad", O.PRESETS)

    def test_duplicates_and_limits(self):
        many = [[f"C{i}", [i / 60, 0, 0]] for i in range(60)]
        packs.apply_pack(packs.read_pack(self.write("m.json", {"colors": {"pants": many + many}})))
        self.assertLessEqual(len(O.PANTS_COLORS), packs.MAX_OPTIONS)
        names = [n.lower() for n, _ in O.PANTS_COLORS]
        self.assertEqual(len(names), len(set(names)))
        packs.apply_pack(packs.read_pack(self.write("d.json", {"colors": {"hair": [["black", [0, 0, 0]]]}})))
        self.assertEqual([n.lower() for n, _ in O.HAIR_COLORS].count("black"), 1)     # not added twice

    def test_long_names_are_trimmed_and_phrases_validated(self):
        pack = packs.read_pack(self.write("p.json", {"phrases": [["A very long label indeed", "x" * 500]]}))
        label, text = pack["phrases"][0]
        self.assertLessEqual(len(label), 14)
        self.assertLessEqual(len(text), 240)
        with self.assertRaises(packs.PackError):
            packs.read_pack(self.write("q.json", {"phrases": [["", "empty label"]]}))

    def test_load_all_on_an_empty_folder(self):
        self.assertEqual(packs.load_all(self.tmp.name), ([], []))


class LanguageTests(unittest.TestCase):
    def names(self, text, lang):
        return [v for v, _ in speech.text_to_visemes(text, lang)]

    def test_spanish_is_read_letter_by_letter(self):
        self.assertEqual(self.names("hola", "es"), ["OO", "DD", "AA"])            # silent h
        self.assertEqual(self.names("mucho", "es")[:2], ["PP", "UU"])
        self.assertIn("CH", self.names("mucho", "es"))
        self.assertEqual(self.names("queso", "es")[0], "KK")
        self.assertNotIn("UU", self.names("queso", "es"))                         # silent u in que
        self.assertEqual(self.names("vaca", "es")[0], "PP")                       # v sounds like b
        self.assertEqual(self.names("cielo", "es")[0], "SS")

    def test_accents_and_enye_are_understood(self):
        self.assertEqual(self.names("niño", "es"), self.names("nino", "es"))
        self.assertEqual(self.names("café", "en"), self.names("cafe", "en"))
        self.assertTrue(self.names("Ça va très bien", "fr"))

    def test_italian_and_portuguese_differ_where_they_should(self):
        self.assertEqual(self.names("ciao", "it")[0], "CH")
        self.assertEqual(self.names("cena", "es")[0], "SS")
        self.assertEqual(self.names("vino", "it")[0], "FF")

    def test_unknown_language_falls_back_to_english_rules(self):
        self.assertEqual(self.names("thin", "xx"), self.names("thin", "en"))
        self.assertEqual(self.names("thin", "de")[0], "TH")

    def test_language_reaches_the_lip_sync_timeline(self):
        es = speech.Speech.from_text_only("hola amigo", lang="es")
        en = speech.Speech.from_text_only("hola amigo", lang="en")
        self.assertEqual(es.lang, "es")
        self.assertNotEqual([s[0] for s in es.segments], [s[0] for s in en.segments])

    def test_every_language_produces_valid_shapes(self):
        for lang in ("en", "es", "it", "pt", "fr", "de", "xx"):
            for text in ("¡Hola! ¿Cómo estás?", "Ich möchte Käse.", "", "1234"):
                for v, w in speech.text_to_visemes(text, lang):
                    self.assertIn(v, speech.VISEMES)
                    self.assertGreater(w, 0)


if __name__ == "__main__":
    unittest.main()
