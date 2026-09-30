"""Content packs: drop a .json file into the library's `packs` folder to add your own
colours, outfit presets, quick phrases and backgrounds. Nothing is executed; a pack is
plain data that is checked before it is used.

    {
      "name": "Neon",
      "colors": {"hair": [["Teal", [0.1, 0.8, 0.8]]], "cloth": [["Neon pink", [255, 20, 147]]]},
      "presets": {"Rave": {"hair": "Mohawk", "haircolor": "Teal", "top": "Hoodie", "topcolor": "Neon pink"}},
      "phrases": [["Rave on", "Turn the music up!"]],
      "backgrounds": [["Club", [[0.1, 0.0, 0.2], [0.4, 0.0, 0.5]]]]
    }

Colour groups: skin, hair, eyes, cloth (tops, hats, scarves, bags, patterns), pants, shoes.
"""
import json
import os

from . import options as O
from . import store, ui

COLOR_GROUPS = {"skin": O.SKIN_TONES, "hair": O.HAIR_COLORS, "eyes": O.EYE_COLORS, "cloth": O.CLOTH_COLORS,
                "pants": O.PANTS_COLORS, "shoes": O.SHOE_COLORS}
MAX_OPTIONS = 36            # share codes use one base-36 digit per option
MAX_NAME = 24


class PackError(ValueError):
    pass


def _rgb(value):
    """Accept [r, g, b] as 0..1 floats or 0..255 numbers."""
    if not (isinstance(value, (list, tuple)) and len(value) == 3
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)):
        raise PackError(f"not an [r, g, b] colour: {value!r}")
    top = max(value)
    rgb = [float(x) / 255.0 if top > 1.0 else float(x) for x in value]
    if not all(0.0 <= x <= 1.0 for x in rgb):
        raise PackError(f"colour out of range: {value!r}")
    return tuple(rgb)


def _name(value, limit=MAX_NAME):
    if not isinstance(value, str) or not value.strip():
        raise PackError(f"bad name: {value!r}")
    return value.strip()[:limit]


def read_pack(path):
    """Parse and validate one pack file into plain data; raises PackError with a readable reason."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackError(f"{os.path.basename(path)}: {exc}") from exc
    if not isinstance(data, dict):
        raise PackError(f"{os.path.basename(path)}: a pack must be a JSON object")
    pack = {"name": str(data.get("name") or os.path.splitext(os.path.basename(path))[0])[:40],
            "colors": {}, "presets": {}, "phrases": [], "backgrounds": []}
    try:
        for group, entries in (data.get("colors") or {}).items():
            if group not in COLOR_GROUPS:
                raise PackError(f"unknown colour group '{group}' (use {', '.join(COLOR_GROUPS)})")
            pack["colors"][group] = [(_name(n), _rgb(c)) for n, c in entries]
        for name, options in (data.get("presets") or {}).items():
            if not isinstance(options, dict):
                raise PackError(f"preset '{name}' must be an object")
            pack["presets"][_name(name)] = {str(k): str(v) for k, v in options.items()}
        for label, text in data.get("phrases") or []:
            pack["phrases"].append((_name(label, 14), str(text).strip()[:240]))
        for name, pair in data.get("backgrounds") or []:
            if len(pair) != 2:
                raise PackError(f"background '{name}' needs a top and a bottom colour")
            pack["backgrounds"].append((_name(name), (_rgb(pair[0]), _rgb(pair[1]))))
    except (TypeError, ValueError) as exc:
        raise PackError(f"{os.path.basename(path)}: {exc}") from exc
    return pack


def apply_pack(pack):
    """Add a validated pack's content to the option tables. Returns a short summary string."""
    added = []
    for group, entries in pack["colors"].items():
        target = COLOR_GROUPS[group]
        existing = {n.lower() for n, _ in target}
        for name, rgb in entries:
            if name.lower() not in existing and len(target) < MAX_OPTIONS:
                target.append((name, rgb))
                existing.add(name.lower())
                added.append(name)
    for name, options in pack["presets"].items():
        probe = dict(O.DEFAULT_STATE)
        try:
            for key, value in options.items():
                O.set_by_name(probe, key, value)
        except ValueError as exc:
            raise PackError(f"preset '{name}': {exc}") from exc
        if name not in O.PRESETS:
            O.add_preset(name, options)
            added.append(name)
    known = {label.lower() for label, _ in ui.PHRASES}
    for label, text in pack["phrases"]:
        if label.lower() not in known and text:
            ui.PHRASES.append((label, text))
            known.add(label.lower())
            added.append(label)
    have = {n.lower() for n, _ in O.BACKGROUNDS}
    for name, pair in pack["backgrounds"]:
        if name.lower() not in have:
            O.BACKGROUNDS.append((name, pair))
            have.add(name.lower())
            added.append(name)
    return f"{pack['name']}: {len(added)} new"


def load_all(directory=None):
    """Load every *.json pack. Returns (summaries, errors); a bad pack never stops the others."""
    directory = directory or store.packs_dir()
    summaries, errors = [], []
    for fn in sorted(os.listdir(directory)):
        if not fn.lower().endswith(".json"):
            continue
        try:
            summaries.append(apply_pack(read_pack(os.path.join(directory, fn))))
        except PackError as exc:
            errors.append(str(exc))
    return summaries, errors
