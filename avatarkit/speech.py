"""Lip-sync: turn a sentence and its audio into mouth shapes over time.

Text is converted to a rough viseme sequence (mouth shapes for English
sounds), stretched over the part of the audio where someone is actually
speaking, and modulated by the loudness of the audio so the mouth closes in
pauses and opens wider on stressed syllables. The same audio envelope drives
gesture energy and head beats. No audio device is needed to build or test it.
"""
import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np

# viseme -> (mouth open 0..1, wideness -1 pucker .. +1 spread, lips pressed 0/1)
VISEMES = {
    "sil": (0.00, 0.00, 0.0),
    "PP": (0.00, -0.10, 1.0),     # p b m
    "FF": (0.14, 0.00, 0.0),      # f v
    "TH": (0.20, 0.05, 0.0),      # th
    "DD": (0.24, 0.10, 0.0),      # t d n l
    "KK": (0.30, 0.00, 0.0),      # k g
    "CH": (0.25, -0.55, 0.0),     # ch sh j
    "SS": (0.14, 0.65, 0.0),      # s z
    "RR": (0.30, -0.30, 0.0),     # r
    "AA": (0.95, 0.05, 0.0),      # a
    "EE": (0.50, 0.55, 0.0),      # e
    "II": (0.32, 0.75, 0.0),      # i
    "OO": (0.62, -0.75, 0.0),     # o
    "UU": (0.34, -1.00, 0.0),     # u w
}
VOWELS = {"AA", "EE", "II", "OO", "UU"}

_DIGRAPHS = [("th", "TH"), ("sh", "CH"), ("ch", "CH"), ("zh", "CH"), ("ph", "FF"), ("wh", "UU"),
             ("ng", "KK"), ("ck", "KK"), ("qu", "KK"), ("oo", "UU"), ("ou", "OO"), ("ow", "OO"),
             ("ee", "II"), ("ea", "II"), ("ie", "II"), ("ai", "EE"), ("ay", "EE"), ("ei", "EE"),
             ("oa", "OO"), ("oi", "OO"), ("au", "AA"), ("aw", "AA")]
_SINGLE = {"a": "AA", "e": "EE", "i": "II", "o": "OO", "u": "UU", "y": "II",
           "p": "PP", "b": "PP", "m": "PP", "f": "FF", "v": "FF",
           "t": "DD", "d": "DD", "n": "DD", "l": "DD",
           "k": "KK", "g": "KK", "c": "KK", "q": "KK", "x": "KK",
           "s": "SS", "z": "SS", "r": "RR", "w": "UU", "j": "CH", "h": "KK"}
_PAUSE = {",": 2.0, ";": 2.5, ":": 2.5, ".": 3.5, "!": 3.5, "?": 3.5, "-": 1.2}


_PHONEMIC = {"es", "it", "pt"}      # spelling is close to pronunciation: letters map straight to shapes
_PH_SINGLE = {"a": "AA", "e": "EE", "i": "II", "o": "OO", "u": "UU", "y": "II", "w": "UU",
              "b": "PP", "p": "PP", "m": "PP", "f": "FF", "d": "DD", "t": "DD", "n": "DD", "l": "DD",
              "r": "RR", "g": "KK", "k": "KK", "q": "KK", "j": "KK", "s": "SS", "z": "SS", "x": "SS"}


def _strip_accents(text):
    """é -> e, ñ -> n, ü -> u ... (the mouth shape is the same)."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def _phonemic_visemes(text, lang):
    out = []
    for word in re.findall(r"[a-z']+|[,;:.!?\-]", _strip_accents(text.lower())):
        if word in _PAUSE:
            out.append(("sil", _PAUSE[word]))
            continue
        w, i = word.replace("'", ""), 0
        while i < len(w):
            pair, ch, nxt = w[i:i + 2], w[i], w[i + 1:i + 2]
            if pair == "ch":
                out.append(("CH", 1.2))
                i += 2
            elif pair in ("ll", "rr", "lh", "nh", "gn"):
                out.append(("II" if pair == "ll" else "RR" if pair == "rr" else "DD", 1.2))
                i += 2
            elif ch == "h":
                i += 1                                           # silent
            elif ch == "c":
                out.append(("SS" if nxt in ("e", "i") and lang != "it" else "CH" if nxt in ("e", "i") else "KK", 1.0))
                i += 1
            elif ch == "v":
                out.append(("PP" if lang == "es" else "FF", 1.0))
                i += 1
            elif ch == "u" and w[i - 1:i] in ("q", "g") and nxt in ("e", "i"):
                i += 1                                           # the u in que / gui is silent
            else:
                v = _PH_SINGLE.get(ch)
                if v:
                    out.append((v, 2.0 if v in VOWELS else 1.0))
                i += 1
        out.append(("sil", 0.5))
    while out and out[-1][0] == "sil":
        out.pop()
    return out


def text_to_visemes(text, lang="en"):
    """Rough grapheme -> viseme sequence: list of (viseme, weight).

    Weights are relative durations: vowels are long, consonants short, and punctuation
    becomes a pause ("sil"). `lang` is a two-letter code; Spanish, Italian and Portuguese get
    their own phonemic rules, everything else uses the English rules with accents removed.
    """
    if lang in _PHONEMIC:
        return _phonemic_visemes(text, lang)
    text = _strip_accents(text)
    out = []
    for word in re.findall(r"[a-z']+|[,;:.!?\-]", text.lower()):
        if word in _PAUSE:
            out.append(("sil", _PAUSE[word]))
            continue
        w = word.replace("'", "")
        i = 0
        while i < len(w):
            pair = w[i:i + 2]
            match = next((v for d, v in _DIGRAPHS if d == pair), None)
            if match:
                out.append((match, 2.0 if match in VOWELS else 1.2))
                i += 2
                continue
            ch = w[i]
            silent_e = ch == "e" and i == len(w) - 1 and len(w) > 2 and w[i - 1] not in "aeiou"
            if ch == "c" and w[i + 1:i + 2] in ("e", "i", "y"):
                v = "SS"
            else:
                v = _SINGLE.get(ch)
            if v and not silent_e:
                out.append((v, 2.0 if v in VOWELS else 1.0))
            i += 1
        out.append(("sil", 0.5))
    while out and out[-1][0] == "sil":
        out.pop()
    return out


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def envelope(samples, rate, fps=100):
    """Loudness (0..1) per 1/fps second from mono float samples."""
    hop = max(1, int(rate / fps))
    n = len(samples) // hop
    if n == 0:
        return np.zeros(1)
    frames = np.asarray(samples[:n * hop], float).reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(1))
    ref = np.percentile(rms, 95) if rms.max() > 0 else 1.0
    env = np.clip(rms / max(ref, 1e-9), 0.0, 1.3)
    kernel = np.array([0.25, 0.5, 0.25])
    return np.convolve(env, kernel, mode="same")


# emotion -> (smile, brow raise 0..1, brow tilt -1 sad .. +1 angry, eyelid closing in degrees)
EMOTIONS = {
    "neutral": (0.0, 0.0, 0.0, 0.0),
    "happy": (0.45, 0.25, 0.0, 3.0),
    "excited": (0.75, 0.80, 0.0, -6.0),
    "sad": (-0.55, 0.15, -0.8, 5.0),
    "angry": (-0.25, -0.35, 0.9, 7.0),
    "surprised": (0.05, 1.00, 0.0, -12.0),
    "curious": (0.05, 0.55, 0.3, -3.0),
}
_POSITIVE = {"love", "great", "amazing", "wonderful", "happy", "thanks", "thank", "glad", "nice", "good",
             "awesome", "fun", "beautiful", "perfect", "yay", "congratulations", "welcome", "hello", "hi",
             "excellent", "fantastic", "cool", "enjoy", "please"}
_SAD = {"sad", "sorry", "unfortunately", "miss", "lost", "cry", "alone", "tired", "sick", "bad", "worst",
        "goodbye", "farewell", "lonely", "disappointed", "regret"}
_ANGRY = {"angry", "hate", "furious", "annoyed", "stupid", "terrible", "awful", "never", "stop", "damn",
          "mad", "unacceptable", "ridiculous"}
_SURPRISE = {"wow", "whoa", "really", "unbelievable", "incredible", "seriously", "omg", "surprise"}
_QUESTION = {"why", "what", "how", "who", "where", "when", "which"}


def classify(sentence):
    """Pick an emotion for one sentence from its words and punctuation."""
    words = set(re.findall(r"[a-z']+", sentence.lower()))
    bangs = sentence.count("!")
    score = {"happy": len(words & _POSITIVE), "sad": len(words & _SAD), "angry": len(words & _ANGRY),
             "surprised": len(words & _SURPRISE)}
    best = max(score, key=score.get)
    if score[best] == 0:
        if sentence.strip().endswith("?") or words & _QUESTION:
            return "curious"
        return "excited" if bangs else "neutral"
    if best == "happy" and bangs:
        return "excited"
    if best == "surprised" and (bangs or sentence.strip().endswith("?")):
        return "surprised"
    return best


def emotion_track(text):
    """[(start_fraction, end_fraction, emotion)] over the text, one entry per sentence."""
    spans, total = [], max(len(text), 1)
    for m in re.finditer(r"[^.!?]+[.!?]*", text):
        chunk = m.group(0)
        if chunk.strip():
            spans.append((m.start() / total, m.end() / total, classify(chunk)))
    return spans or [(0.0, 1.0, "neutral")]


@dataclass
class MouthShape:
    open: float = 0.0
    wide: float = 0.0
    press: float = 0.0
    energy: float = 0.0     # smoothed loudness: how animated the speaker is
    beat: float = 0.0       # onset strength: a stressed syllable just started


class Speech:
    """A sentence with a lip-sync timeline. `sample(t)` gives the mouth at time t (seconds)."""
    FPS = 100

    def __init__(self, text, duration, env, lang="en"):
        self.text = text
        self.lang = lang
        self.duration = float(duration)
        self.env = np.asarray(env, float)
        self.energy = np.convolve(self.env, np.ones(15) / 15, mode="same")
        lag = np.concatenate([np.full(8, self.env[0]), self.env[:-8]]) if len(self.env) > 8 else self.env
        self.beats = np.convolve(np.clip(self.env - lag, 0, 1), np.ones(6) / 6, mode="same")
        self._build_timeline()

    # ---- construction -----------------------------------------------------------------
    @classmethod
    def from_samples(cls, text, samples, rate, lang="en"):
        return cls(text, len(samples) / rate, envelope(samples, rate, cls.FPS), lang)

    @classmethod
    def from_text_only(cls, text, chars_per_second=14.0, lang="en"):
        """No audio available: a plausible loudness pattern timed from the text alone."""
        visemes = text_to_visemes(text, lang) or [("sil", 1.0)]
        total = sum(w for _, w in visemes)
        duration = max(1.0, len(text) / chars_per_second)
        env = np.zeros(int(duration * cls.FPS) + 1)
        t = 0.0
        for v, w in visemes:
            d = duration * w / total
            if v != "sil":
                env[int(t * cls.FPS):max(int(t * cls.FPS) + 1, int((t + d) * cls.FPS))] = 0.75 if v in VOWELS else 0.4
            t += d
        return cls(text, duration, np.convolve(env, np.ones(5) / 5, mode="same"), lang)

    def _build_timeline(self):
        """Stretch the viseme sequence over the span where the audio is loud enough."""
        loud = np.flatnonzero(self.env > 0.08)
        t0 = loud[0] / self.FPS if len(loud) else 0.0
        t1 = (loud[-1] + 1) / self.FPS if len(loud) else self.duration
        seq = text_to_visemes(self.text, self.lang) or [("sil", 1.0)]
        weights = np.array([w for _, w in seq])
        edges = t0 + (t1 - t0) * np.concatenate([[0], np.cumsum(weights)]) / weights.sum()
        self.segments = [(seq[i][0], edges[i], edges[i + 1]) for i in range(len(seq))]
        self.centers = np.array([(a + b) / 2 for _, a, b in self.segments])
        self.halves = np.array([(b - a) / 2 for _, a, b in self.segments])
        self.params = np.array([VISEMES[v] for v, _, _ in self.segments])
        self.span = (t0, t1)
        # emotion timeline: sentences share the speaking span in proportion to their length
        spans = emotion_track(self.text)
        self.emotion_centers = np.array([t0 + (t1 - t0) * (a + b) / 2 for a, b, _ in spans])
        self.emotion_halves = np.array([(t1 - t0) * (b - a) / 2 for a, b, _ in spans])
        self.emotion_params = np.array([EMOTIONS[e] for _, _, e in spans])
        self.emotions = [e for _, _, e in spans]

    # ---- sampling -------------------------------------------------------------------------
    def _at(self, arr, t):
        i = int(np.clip(t * self.FPS, 0, len(arr) - 1))
        return float(arr[i])

    def sample(self, t):
        """Mouth shape at time t: visemes blended by proximity, gated by loudness."""
        if t < 0 or t > self.duration:
            return MouthShape()
        width = 0.55 * self.halves + 0.03            # coarticulation: neighbours overlap
        w = np.exp(-(((t - self.centers) / width) ** 2))
        total = w.sum()
        if total < 1e-6:
            return MouthShape(energy=self._at(self.energy, t))
        o, wide, press = (self.params * w[:, None]).sum(0) / total
        env = self._at(self.env, t)
        gate = float(smoothstep(0.03, 0.20, env))     # close the mouth in pauses
        loud = 0.55 + 0.75 * min(env, 1.2)            # open wider on stressed syllables
        return MouthShape(open=min(1.0, o * gate * loud), wide=float(wide * gate),
                     press=float(press * (1 - gate * 0.7)),
                     energy=self._at(self.energy, t), beat=self._at(self.beats, t))

    def emotion(self, t):
        """(smile, brow_raise, brow_tilt, lid) at time t: sentence emotions cross-faded smoothly."""
        if t < -0.2 or t > self.duration + 0.3:
            return EMOTIONS["neutral"]
        width = 0.75 * self.emotion_halves + 0.25
        w = np.exp(-(((t - self.emotion_centers) / width) ** 2))
        fade_out = float(smoothstep(self.span[1] + 0.25, self.span[1] - 0.1, t))     # relax after the last word
        fade_in = float(smoothstep(self.span[0] - 0.3, self.span[0], t))
        if w.sum() < 1e-6:
            return EMOTIONS["neutral"]
        blend = (self.emotion_params * w[:, None]).sum(0) / max(w.sum(), 1.0)
        return tuple(float(v) * fade_in * fade_out for v in blend)
