"""Customizer options, state handling, presets and share codes."""
import json
import random

SKIN_TONES = [
    ("Porcelain", (0.96, 0.80, 0.69)),
    ("Light", (0.90, 0.72, 0.55)),
    ("Medium", (0.78, 0.58, 0.40)),
    ("Tan", (0.62, 0.44, 0.29)),
    ("Brown", (0.45, 0.31, 0.20)),
    ("Deep", (0.30, 0.20, 0.14)),
]
HAIR_COLORS = [
    ("Black", (0.06, 0.05, 0.05)),
    ("Dark brown", (0.22, 0.14, 0.09)),
    ("Brown", (0.38, 0.24, 0.13)),
    ("Auburn", (0.55, 0.22, 0.12)),
    ("Ginger", (0.72, 0.36, 0.14)),
    ("Blonde", (0.85, 0.68, 0.36)),
    ("Platinum", (0.92, 0.88, 0.78)),
    ("Gray", (0.60, 0.60, 0.62)),
    ("Pink", (0.90, 0.42, 0.62)),
    ("Blue", (0.20, 0.35, 0.80)),
]
EYE_COLORS = [
    ("Brown", (0.30, 0.17, 0.08)),
    ("Hazel", (0.50, 0.35, 0.15)),
    ("Amber", (0.70, 0.45, 0.10)),
    ("Blue", (0.20, 0.42, 0.72)),
    ("Green", (0.25, 0.55, 0.35)),
    ("Gray", (0.45, 0.52, 0.58)),
    ("Dark", (0.12, 0.08, 0.06)),
]
CLOTH_COLORS = [
    ("White", (0.93, 0.93, 0.94)),
    ("Black", (0.10, 0.10, 0.12)),
    ("Red", (0.78, 0.15, 0.17)),
    ("Orange", (0.93, 0.55, 0.16)),
    ("Yellow", (0.95, 0.80, 0.22)),
    ("Green", (0.20, 0.58, 0.36)),
    ("Teal", (0.13, 0.55, 0.60)),
    ("Blue", (0.20, 0.38, 0.78)),
    ("Purple", (0.50, 0.28, 0.70)),
    ("Gray", (0.50, 0.52, 0.56)),
]
PANTS_COLORS = [
    ("Denim", (0.20, 0.28, 0.48)),
    ("Black", (0.10, 0.10, 0.12)),
    ("Khaki", (0.68, 0.60, 0.42)),
    ("Gray", (0.40, 0.42, 0.46)),
    ("Navy", (0.10, 0.14, 0.28)),
    ("Olive", (0.35, 0.40, 0.22)),
]
SHOE_COLORS = [
    ("White", (0.92, 0.92, 0.92)),
    ("Black", (0.08, 0.08, 0.09)),
    ("Brown", (0.36, 0.22, 0.12)),
    ("Red", (0.75, 0.15, 0.15)),
]
BACKGROUNDS = [
    ("Studio", ((0.86, 0.88, 0.92), (0.62, 0.66, 0.74))),
    ("Sunset", ((0.98, 0.80, 0.62), (0.55, 0.36, 0.55))),
    ("Mint", ((0.82, 0.94, 0.90), (0.48, 0.70, 0.68))),
    ("Night", ((0.20, 0.22, 0.34), (0.05, 0.06, 0.12))),
]

# Expression: (smile, mouth_open, brow_dy, brow_tilt, lid_left, lid_right).
# lid_* are degrees added to the eyelid tilt (negative = more closed).
EXPRESSIONS = [
    ("Neutral", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("Happy", (0.5, 0.25, 0.01, 0.0, -6.0, -6.0)),
    ("Surprised", (0.0, 0.9, 0.05, -0.2, 20.0, 20.0)),
    ("Sad", (-0.5, 0.0, 0.0, -0.7, -5.0, -5.0)),
    ("Angry", (-0.3, 0.0, -0.03, 0.7, -8.0, -8.0)),
    ("Wink", (0.6, 0.3, 0.01, 0.0, 0.0, -50.0)),
]
# Body type: (shoulders, chest, waist, hips, bust, limb thickness)
BODY_TYPES = [
    ("Masculine", (1.06, 1.0, 1.02, 0.94, 0.0, 1.05)),
    ("Neutral", (1.0, 1.0, 1.0, 1.0, 0.0, 1.0)),
    ("Feminine", (0.90, 1.0, 0.86, 1.10, 0.05, 0.92)),
]
PATTERNS = [("None", 0), ("Stripes", 1), ("Dots", 2), ("Plaid", 3), ("Emblem", 4)]
POSES = [("Relaxed", "relaxed"), ("A-pose", "apose"), ("Wave", "wave"),
         ("Hands on hips", "hips"), ("Cheer", "cheer"), ("Walk", "walk"), ("Dance", "dance")]

# key, label, [(name, value)...], default index
_DEFS = [
    ("skin", "Skin tone", SKIN_TONES, 1),
    ("freckles", "Freckles", [("None", 0.0), ("Light", 0.5), ("Heavy", 1.0)], 0),
    ("face", "Face shape", [("Round", 0.12), ("Oval", 0.28), ("Pointed", 0.42)], 1),
    ("nose", "Nose", [("Small", 0.8), ("Medium", 1.0), ("Large", 1.3)], 1),
    ("mouth", "Mouth", [("Neutral", (0.0, 1.0)), ("Smile", (0.6, 1.0)),
                        ("Big smile", (1.0, 1.15)), ("Small", (0.1, 0.8))], 1),
    ("expression", "Expression", EXPRESSIONS, 0),
    ("eyes", "Eye color", EYE_COLORS, 0),
    ("eyesize", "Eye size", [("Small", 0.85), ("Medium", 1.0), ("Large", 1.2)], 1),
    ("brows", "Eyebrows", [("Thin", 0.7), ("Normal", 1.0), ("Thick", 1.5)], 1),
    ("facial", "Facial hair", [("None", "none"), ("Mustache", "mustache"), ("Beard", "beard")], 0),
    ("glasses", "Glasses", [("None", "none"), ("Round", "round"), ("Square", "square"),
                            ("Sunglasses", "sun"), ("Cat-eye", "cat"), ("Aviator", "aviator")], 0),
    ("earrings", "Earrings", [("None", "none"), ("Studs", "studs"), ("Hoops", "hoops")], 0),
    ("hair", "Hair style", [("Bald", "bald"), ("Buzz", "buzz"), ("Short", "short"),
                            ("Curly", "curly"), ("Long", "long"), ("Bob", "bob"),
                            ("Bangs", "bangs"), ("Bun", "bun"), ("Ponytail", "ponytail"),
                            ("Quiff", "quiff"), ("Mohawk", "mohawk"), ("Afro", "afro"),
                            ("Pigtails", "pigtails"), ("Braid", "braid")], 2),
    ("haircolor", "Hair color", HAIR_COLORS, 2),
    ("hat", "Hat", [("None", "none"), ("Beanie", "beanie"), ("Cap", "cap"), ("Bucket", "bucket"),
                    ("Top hat", "tophat")], 0),
    ("hatcolor", "Hat color", CLOTH_COLORS, 2),
    ("bodytype", "Body type", BODY_TYPES, 1),
    ("build", "Build", [("Slim", 0.9), ("Average", 1.0), ("Broad", 1.12)], 1),
    ("height", "Height", [("Short", 0.93), ("Average", 1.0), ("Tall", 1.08)], 1),
    ("top", "Top", [("T-shirt", "tee"), ("Long sleeve", "long"), ("Tank top", "tank"),
                    ("Hoodie", "hoodie"), ("Jacket", "jacket"), ("Dress", "dress")], 0),
    ("topcolor", "Top color", CLOTH_COLORS, 7),
    ("pattern", "Top pattern", PATTERNS, 0),
    ("patterncolor", "Pattern color", CLOTH_COLORS, 0),
    ("pants", "Bottoms", [("Jeans", "jeans"), ("Shorts", "shorts"), ("Skirt", "skirt")], 0),
    ("pantscolor", "Bottoms color", PANTS_COLORS, 0),
    ("shoestyle", "Shoes", [("Sneakers", "sneakers"), ("Boots", "boots"),
                            ("Barefoot", "barefoot")], 0),
    ("shoes", "Shoe color", SHOE_COLORS, 0),
    ("necklace", "Necklace", [("None", "none"), ("Chain", "chain"), ("Pendant", "pendant")], 0),
    ("scarf", "Scarf", [("None", "none"), ("Scarf", "scarf")], 0),
    ("scarfcolor", "Scarf color", CLOTH_COLORS, 2),
    ("watch", "Watch", [("None", "none"), ("Watch", "watch")], 0),
    ("bag", "Bag", [("None", "none"), ("Backpack", "backpack")], 0),
    ("bagcolor", "Bag color", CLOTH_COLORS, 3),
    ("pose", "Pose", POSES, 0),
    ("animate", "Animation", [("On", True), ("Off", False)], 0),
    ("bg", "Background", BACKGROUNDS, 0),
    ("voice", "Voice", [("Default", "")], 0),
    ("speed", "Speech speed", [("Slow", "Slow"), ("Normal", "Normal"), ("Fast", "Fast")], 1),
    ("gestures", "Gestures", [("On", True), ("Off", False)], 0),
    ("shadows", "Shadows", [("On", True), ("Off", False)], 0),
    ("physics", "Hair physics", [("On", True), ("Off", False)], 0),
    ("headphones", "Headphones", [("None", False), ("Headphones", True)], 0),
    ("neckwear", "Neckwear", [("None", "none"), ("Bow tie", "bowtie"), ("Tie", "tie")], 0),
    ("neckcolor", "Neckwear color", CLOTH_COLORS, 2),
]

OPTIONS = [(k, label, opts) for k, label, opts, _ in _DEFS]
DEFAULT_STATE = {k: d for k, _, _, d in _DEFS}
OPTION_KEYS = [k for k, _, _ in OPTIONS]
_BY_KEY = {k: (label, opts) for k, label, opts in OPTIONS}

TABS = [
    ("Face", ["skin", "freckles", "face", "nose", "mouth", "expression", "eyes", "eyesize",
              "brows", "facial", "glasses", "earrings"]),
    ("Hair", ["hair", "haircolor", "hat", "hatcolor"]),
    ("Body", ["bodytype", "build", "height"]),
    ("Outfit", ["top", "topcolor", "pattern", "patterncolor", "pants", "pantscolor",
                "shoestyle", "shoes"]),
    ("Extras", ["necklace", "neckwear", "neckcolor", "scarf", "scarfcolor", "headphones", "watch", "bag",
                "bagcolor"]),
    ("Scene", ["pose", "animate", "shadows", "physics", "bg", "preset"]),
    ("Talk", ["voice", "speed", "gestures"]),
    ("Library", []),
]
COLOR_KEYS = {"skin", "eyes", "haircolor", "hatcolor", "topcolor", "patterncolor",
              "pantscolor", "shoes", "scarfcolor", "bagcolor", "neckcolor"}

# Outfit presets: only the listed options change, so the face stays yours.
PRESETS = {
    "Casual": {"hair": "Short", "hat": "None", "top": "T-shirt", "topcolor": "Blue",
               "pattern": "None", "pants": "Jeans", "pantscolor": "Denim", "shoestyle": "Sneakers",
               "shoes": "White", "glasses": "None", "scarf": "None", "bag": "None",
               "necklace": "None", "watch": "None"},
    "Streetwear": {"hair": "Quiff", "hat": "Cap", "hatcolor": "Black", "top": "Hoodie",
                   "topcolor": "Gray", "pattern": "None", "pants": "Jeans", "pantscolor": "Black",
                   "shoestyle": "Sneakers", "shoes": "Red", "glasses": "Sunglasses",
                   "bag": "Backpack", "bagcolor": "Black", "scarf": "None", "watch": "Watch"},
    "Business": {"hair": "Short", "hat": "None", "top": "Jacket", "topcolor": "Gray",
                 "pattern": "None", "pants": "Jeans", "pantscolor": "Navy", "shoestyle": "Boots",
                 "shoes": "Brown", "glasses": "Square", "necklace": "None", "watch": "Watch",
                 "scarf": "None", "bag": "None"},
    "Punk": {"hair": "Mohawk", "haircolor": "Pink", "hat": "None", "top": "Jacket",
             "topcolor": "Black", "pattern": "None", "pants": "Jeans", "pantscolor": "Black",
             "shoestyle": "Boots", "shoes": "Black", "glasses": "Sunglasses",
             "earrings": "Hoops", "necklace": "Chain", "scarf": "None", "bag": "None"},
    "Sporty": {"hair": "Ponytail", "hat": "Cap", "hatcolor": "Teal", "top": "Tank top",
               "topcolor": "Orange", "pattern": "Stripes", "patterncolor": "White",
               "pants": "Shorts", "pantscolor": "Black", "shoestyle": "Sneakers", "shoes": "White",
               "glasses": "None", "watch": "Watch", "scarf": "None", "bag": "None"},
    "Cozy": {"hair": "Long", "hat": "Beanie", "hatcolor": "Red", "top": "Hoodie",
             "topcolor": "Purple", "pattern": "None", "pants": "Jeans", "pantscolor": "Gray",
             "shoestyle": "Boots", "shoes": "Brown", "scarf": "Scarf", "scarfcolor": "Yellow",
             "glasses": "None", "bag": "None", "necklace": "None", "watch": "None"},
    "Retro": {"hair": "Afro", "haircolor": "Black", "hat": "None", "top": "Long sleeve",
              "topcolor": "Orange", "pattern": "Stripes", "patterncolor": "Yellow",
              "pants": "Jeans", "pantscolor": "Khaki", "shoestyle": "Boots", "shoes": "Brown",
              "glasses": "Sunglasses", "necklace": "Pendant", "scarf": "None", "bag": "None"},
    "Formal": {"hair": "Bangs", "hat": "None", "top": "Long sleeve", "topcolor": "Black",
               "pattern": "None", "pants": "Skirt", "pantscolor": "Black", "shoestyle": "Boots",
               "shoes": "Black", "glasses": "None", "necklace": "Pendant", "earrings": "Studs",
               "scarf": "None", "bag": "None", "watch": "None"},
}
_DEFS.append(("preset", "Outfit preset", [(n, n) for n in PRESETS], 0))
OPTIONS.append(("preset", "Outfit preset", _DEFS[-1][2]))
DEFAULT_STATE["preset"] = 0
_BY_KEY["preset"] = ("Outfit preset", _DEFS[-1][2])
OPTION_KEYS = [k for k, _, _ in OPTIONS]

# Options that don't change the meshes, so no rebuild is needed when they change.
TALK_KEYS = {"voice", "speed", "gestures"}
VIEW_KEYS = {"shadows", "physics"}          # how it is drawn, not what the avatar is
NON_BUILD_KEYS = {"pose", "animate", "bg", "preset"} | TALK_KEYS | VIEW_KEYS
NO_RANDOM = {"pose", "animate", "bg", "preset"} | TALK_KEYS | VIEW_KEYS
CODE_KEYS = [k for k in OPTION_KEYS if k not in ("preset", "animate") and k not in TALK_KEYS | VIEW_KEYS]
CODE_PREFIX = "AV2-"
LEGACY_CODE_LEN = 35          # codes made before the newest options were added are shorter


def set_choices(key, names):
    """Replace an option's choices at run time (the installed voices are only known then)."""
    opts = [(n, n) for n in names] or [("Default", "")]
    for i, (k, lab, _) in enumerate(OPTIONS):
        if k == key:
            OPTIONS[i] = (k, lab, opts)
    _BY_KEY[key] = (_BY_KEY[key][0], opts)


def add_preset(name, mapping):
    """Register an outfit preset at run time (used by content packs)."""
    PRESETS[name] = dict(mapping)
    _BY_KEY["preset"][1].append((name, name))


def label(key):
    return _BY_KEY[key][0]


def choices(key):
    return _BY_KEY[key][1]


def resolve(state):
    return {key: opts[state[key] % len(opts)][1] for key, _, opts in OPTIONS}


def randomize(state, rng=random):
    for key, _, opts in OPTIONS:
        if key not in NO_RANDOM:
            state[key] = rng.randrange(len(opts))


def set_by_name(state, key, name):
    if key not in _BY_KEY:
        raise ValueError(f"Unknown option '{key}'. Options: {', '.join(OPTION_KEYS)}")
    for i, (n, _) in enumerate(_BY_KEY[key][1]):
        if n.lower() == str(name).lower():
            state[key] = i
            return
    raise ValueError(f"Unknown value '{name}' for '{key}'. "
                     f"Choices: {', '.join(n for n, _ in _BY_KEY[key][1])}")


def apply_preset(state, name):
    for key, val in PRESETS[name].items():
        set_by_name(state, key, val)
    state["preset"] = [n for n in PRESETS].index(name)


def save_state(state, path):
    data = {key: opts[state[key]][0] for key, _, opts in OPTIONS if key != "preset"}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_state(state, path):
    with open(path) as f:
        data = json.load(f)
    for key, name in data.items():
        try:
            set_by_name(state, key, name)
        except ValueError:
            pass  # ignore options that no longer exist


_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def encode_state(state):
    """Compact share code, e.g. AV2-1a0..., one base-36 digit per option."""
    return CODE_PREFIX + "".join(_DIGITS[state[k]] for k in CODE_KEYS)


def decode_state(code, state):
    """Apply a share code to `state`. Raises ValueError if it isn't valid."""
    code = code.strip()
    if not code.startswith(CODE_PREFIX):
        raise ValueError("Not an avatar code")
    digits = code[len(CODE_PREFIX):].lower()
    if not (LEGACY_CODE_LEN <= len(digits) <= len(CODE_KEYS)) or any(c not in _DIGITS for c in digits):
        raise ValueError("Avatar code has the wrong length")
    new = {}
    for key, ch in zip(CODE_KEYS, digits):
        idx = _DIGITS.index(ch)
        if idx >= len(_BY_KEY[key][1]):
            raise ValueError(f"Invalid value for {key}")
        new[key] = idx
    for key in CODE_KEYS[len(digits):]:       # options the code predates keep their defaults
        new[key] = DEFAULT_STATE[key]
    state.update(new)
