"""The interface: a floating side card with tabs, colour swatches and option chips,
a view switcher over the 3D scene, toasts and a shortcut sheet.

Everything is drawn with pygame onto one transparent overlay that is uploaded
as a texture. `UI.render` builds the overlay (no OpenGL needed, so it can be
tested); `UI.draw` uploads and composites it.
"""
import functools
import io
import os
from dataclasses import dataclass, field

import pygame

from . import options as O
from .poses import EMOTES

PANEL_W = 360
MARGIN = 14
RESERVED = PANEL_W + 2 * MARGIN          # width taken from the 3D view

COL = dict(
    panel=(24, 26, 36), panel2=(33, 36, 50), chip=(43, 47, 65), chip_hover=(58, 63, 87),
    line=(47, 51, 68), text=(238, 240, 248), muted=(146, 152, 175), faint=(104, 110, 134),
    accent=(99, 102, 241), accent_hover=(122, 125, 255), ok=(94, 214, 158), danger=(240, 96, 110),
)
VIEWS = [("bust", "Bust"), ("face", "Face"), ("full", "Full")]
SHORTCUTS = [
    ("Tab / Shift+Tab", "switch tab"), ("↑ ↓  ← →", "focus option / change it"),
    ("R", "randomize"), ("[  ]", "previous / next outfit preset"),
    ("V", "cycle view"), ("Space", "toggle spin"), ("Drag / wheel", "orbit / zoom"),
    ("Ctrl+S / Ctrl+O", "save / open an avatar"), ("Ctrl+Z / Ctrl+Y", "undo / redo"),
    ("S / G", "save PNG / transparent PNG"), ("E / O", "export GLB / OBJ"),
    ("C / Ctrl+V", "copy / paste share code"), ("P", "import from a photo"),
    ("T / Enter", "open Talk / speak the text"), ("Esc", "stop speaking"), ("F11", "full screen on / off"),
    ("M / W / F", "microphone / virtual webcam / face tracking"),
    ("Drop a file", "open an .avatar file or a photo"), ("H", "show or hide this sheet"),
]

PHRASES = [
    ("Hello!", "Hello! I'm your new avatar. Nice to meet you."),
    ("About me", "I'm a procedural character, built from thousands of tiny strands of hair, and now I can talk."),
    ("Joke", "Why did the avatar go to school? To get a little more depth!"),
    ("Thanks", "Thank you so much for making me look so good."),
    ("Excited", "Wow! That is amazing. I can't believe it actually works!"),
    ("Goodbye", "It was great talking to you. See you next time!"),
]
LABELS = {"__say": "SAY SOMETHING", "__phrases": "QUICK PHRASES", "__name": "AVATAR NAME",
          "__mine": "MY AVATARS", "__export": "EXPORT / DOWNLOAD", "__edit": "EDIT", "__emotes": "EMOTES",
          "__live": "LIVE"}
EXPORTS = [("png", "PNG"), ("transparent", "Transparent PNG"), ("glb", "GLB"), ("skinned", "Skinned GLB"),
           ("animated", "Animated GLB"), ("vrm", "VRM"), ("obj", "OBJ"), ("bundle", "Bundle .zip"), ("clip", "Talking clip")]


@dataclass
class UIModel:
    state: dict
    tab: int = 0
    focus: int = 0
    code: str = ""
    view: str = "bust"
    spin: bool = False
    toast: str = ""
    toast_alpha: float = 0.0
    help: bool = False
    say_text: str = ""
    name_text: str = ""
    focus_field: str = ""         # "say", "name" or "" (which text box has the keyboard)
    cursor_on: bool = False
    speaking: bool = False
    busy: bool = False            # a voice is being prepared
    avatars: list = field(default_factory=list)      # store.AvatarInfo, newest first
    can_undo: bool = False
    can_redo: bool = False
    mic_on: bool = False          # lip-sync from the microphone
    webcam_on: bool = False       # sending to the virtual webcam
    webcam_fps: int = 0           # how many frames per second it is really delivering
    track_on: bool = False        # the avatar follows your face


# ---------------------------------------------------------------------------
# Fonts and cached, anti-aliased shapes
# ---------------------------------------------------------------------------
def _font(bold, size):
    win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    names = (("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf") if bold
             else ("SegUIVar.ttf", "segoeui.ttf", "arial.ttf"))
    for n in names:
        path = os.path.join(win, n)
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("segoeui,arial,helvetica,sans", size, bold=bold)


@functools.lru_cache(maxsize=None)
def font(bold, size):
    pygame.font.init()
    return _font(bold, size)


@functools.lru_cache(maxsize=1024)
def text(s, bold, size, color):
    return font(bold, size).render(s, True, color)


def text_width(s, bold, size):
    return font(bold, size).size(s)[0]


def ellipsize(s, bold, size, width):
    if text_width(s, bold, size) <= width:
        return s
    while len(s) > 1 and text_width(s + "…", bold, size) > width:
        s = s[:-1]
    return s + "…"


@functools.lru_cache(maxsize=512)
def pill(w, h, r, color, ring=None, ring_w=2):
    """Anti-aliased rounded rectangle (drawn 4x and scaled down)."""
    s = 4
    surf = pygame.Surface((w * s, h * s), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0, 0, w * s, h * s), border_radius=r * s)
    if ring:
        pygame.draw.rect(surf, ring, (0, 0, w * s, h * s), width=ring_w * s, border_radius=r * s)
    return pygame.transform.smoothscale(surf, (w, h))


@functools.lru_cache(maxsize=256)
def disc(d, color, ring=None, ring_w=3):
    s = 4
    surf = pygame.Surface((d * s, d * s), pygame.SRCALPHA)
    pygame.draw.circle(surf, color, (d * s // 2, d * s // 2), d * s // 2)
    if ring:
        pygame.draw.circle(surf, ring, (d * s // 2, d * s // 2), d * s // 2, width=ring_w * s)
    return pygame.transform.smoothscale(surf, (d, d))


@functools.lru_cache(maxsize=16)
def shadow(w, h, r, spread=26):
    """Soft drop shadow: a small blurred-by-upscaling rounded rect."""
    k = 4
    small = pygame.Surface(((w + 2 * spread) // k + 1, (h + 2 * spread) // k + 1), pygame.SRCALPHA)
    pygame.draw.rect(small, (0, 0, 0, 120), (spread // k, spread // k, w // k, h // k),
                     border_radius=max(1, r // k))
    return pygame.transform.smoothscale(small, (w + 2 * spread, h + 2 * spread))


@functools.lru_cache(maxsize=16)
def fade(w, h, color, top):
    """Vertical gradient from `color` to transparent (used to soften scrolled content)."""
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(h):
        a = int(255 * (1 - y / h) ** 1.5) if top else int(255 * (y / h) ** 1.5)
        pygame.draw.line(surf, color + (a,), (0, y), (w, y))
    return surf


def wrap(s, bold, size, width):
    """Word-wrap `s` to `width` px (long words are split)."""
    lines = []
    for para in s.split("\n"):
        line = ""
        for word in para.split(" "):
            test = (line + " " + word).strip() if line else word
            if text_width(test, bold, size) <= width:
                line = test
                continue
            if line:
                lines.append(line)
            while text_width(word, bold, size) > width and len(word) > 1:
                cut = len(word) - 1
                while cut > 1 and text_width(word[:cut], bold, size) > width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            line = word
        lines.append(line)
    return lines


def chip_label(key, name):
    if key == "voice":
        return name.replace("Microsoft ", "").replace(" Desktop", "")
    return name


def rgb(c):
    return tuple(int(round(x * 255)) for x in c)


# ---------------------------------------------------------------------------
class UI:
    def __init__(self):
        self.widgets = []            # (rect, action) in window coordinates
        self.mouse = (-1, -1)
        self.hover_action = None
        self.scroll = {}
        self.focus_key = None
        self.surface = None
        self.dirty = True
        self.tex = None
        self.tex_size = None
        self.panel_rect = pygame.Rect(0, 0, 0, 0)
        self.uploaded = False
        self.tab = 0
        self.max_scroll = 0
        self._thumbs = {}

    # ---- interaction ---------------------------------------------------------------
    def action_at(self, pos):
        for rect, action in reversed(self.widgets):
            if rect.collidepoint(pos):
                return action
        return None

    def hover(self, pos):
        """Track the pointer; returns True if the hovered control changed (needs a redraw)."""
        self.mouse = pos
        act = self.action_at(pos)
        if act != self.hover_action:
            self.hover_action = act
            self.dirty = True
            return True
        return False

    def hit(self, pos):
        return self.action_at(pos)

    def over_panel(self, pos):
        return self.panel_rect.collidepoint(pos)

    def scroll_by(self, notches, pos):
        """Wheel over the panel scrolls its content; returns True if it was consumed."""
        if not self.over_panel(pos):
            return False
        cur = self.scroll.get(self.tab, 0)
        self.scroll[self.tab] = max(0, min(self.max_scroll, cur - notches * 44))
        self.dirty = True
        return True

    def focus_changed(self):
        self.focus_key = "pending"

    # ---- layout ---------------------------------------------------------------------
    @staticmethod
    def layout(keys, state, width):
        """Position labels, chips and swatches relative to the content origin."""
        items, sections, y = [], [], 0
        for key in keys:
            top = y
            items.append(("label", pygame.Rect(0, y, width, 20), key))
            y += 26
            opts = O.choices(key)
            if key in O.COLOR_KEYS:
                x = 0
                for j, (name, c) in enumerate(opts):
                    if x + 30 > width:
                        x, y = 0, y + 38
                    items.append(("swatch", pygame.Rect(x, y, 30, 30), (key, j)))
                    x += 38
                y += 38
            else:
                x = 0
                for j, (name, _) in enumerate(opts):
                    w = text_width(chip_label(key, name), False, 13) + 26
                    if x + w > width:
                        x, y = 0, y + 34
                    items.append(("chip", pygame.Rect(x, y, w, 28), (key, j)))
                    x += w + 6
                y += 34
            y += 14
            sections.append((key, top, y - 14))
        return items, y, sections

    @staticmethod
    def _chips(items, y, width, entries):
        """Wrap action chips; entries are (action, label, style). Returns the new y."""
        x = 0
        for act, label, style in entries:
            w = text_width(label, False, 13) + 26
            if x + w > width:
                x, y = 0, y + 34
            items.append(("achip", pygame.Rect(x, y, w, 28), (act, label, style)))
            x += w + 6
        return y + 34

    def layout_talk(self, width, m):
        """Talk tab: text box, Speak / Random line buttons, quick phrases."""
        items = [("label", pygame.Rect(0, 0, width, 20), "__say"),
                 ("textbox", pygame.Rect(0, 26, width, 96), "say")]
        y = 26 + 96 + 10
        half = (width - 8) // 2
        speak = "Preparing..." if m.busy else ("Stop" if m.speaking else "Speak")
        items.append(("abutton", pygame.Rect(0, y, half, 38), (("say", "speak"), speak,
                                                              "muted" if m.busy else "primary")))
        items.append(("abutton", pygame.Rect(half + 8, y, half, 38), (("say", "random"), "Random line", "secondary")))
        y += 38 + 22
        items.append(("label", pygame.Rect(0, y, width, 20), "__phrases"))
        y = self._chips(items, y + 26, width, [(("phrase", i), name, "chip") for i, (name, _) in enumerate(PHRASES)])
        items.append(("label", pygame.Rect(0, y + 8, width, 20), "__emotes"))
        y = self._chips(items, y + 34, width, [(("emote", key), name, "chip") for name, key, _ in EMOTES])
        items.append(("label", pygame.Rect(0, y + 8, width, 20), "__live"))
        y = self._chips(items, y + 34, width, [
            (("live", "mic"), "Microphone  " + ("on" if m.mic_on else "off"), "primary" if m.mic_on else "chip"),
            (("live", "webcam"), "Virtual webcam  " + (f"on, {m.webcam_fps} fps" if m.webcam_on and m.webcam_fps else
                                                       "on" if m.webcam_on else "off"),
             "primary" if m.webcam_on else "chip"),
            (("live", "track"), "Face tracking  " + ("on" if m.track_on else "off"), "primary" if m.track_on else "chip")])
        return items, y + 14

    def layout_library(self, width, m):
        """Library tab: name, save/open, saved avatars, export buttons, edit history."""
        items = [("label", pygame.Rect(0, 0, width, 20), "__name"),
                 ("textbox", pygame.Rect(0, 26, width, 44), "name")]
        y = 26 + 44 + 10
        third = (width - 16) // 3
        for i, (act, label, style) in enumerate([(("lib", "save"), "Save", "primary"),
                                                 (("lib", "saveas"), "Save as...", "secondary"),
                                                 (("lib", "open"), "Open...", "secondary")]):
            items.append(("abutton", pygame.Rect(i * (third + 8), y, third, 36), (act, label, style)))
        y += 36 + 22
        items.append(("label", pygame.Rect(0, y, width, 20), "__mine"))
        y += 26
        if not m.avatars:
            items.append(("note", pygame.Rect(0, y, width, 40), "Nothing saved yet. Press Save to keep this avatar."))
            y += 50
        else:
            cw = (width - 16) // 3
            ch = int(cw * 1.25) + 26
            for i, info in enumerate(m.avatars):
                col, row = i % 3, i // 3
                items.append(("card", pygame.Rect(col * (cw + 8), y + row * (ch + 8), cw, ch), info))
            y += ((len(m.avatars) + 2) // 3) * (ch + 8)
        y += 14
        items.append(("label", pygame.Rect(0, y, width, 20), "__export"))
        y = self._chips(items, y + 26, width, [(("export", k), lab, "chip") for k, lab in EXPORTS])
        y = self._chips(items, y, width, [(("lib", "folder"), "Open folder", "chip"),
                                          (("lib", "reveal"), "Show last export", "chip"),
                                          (("lib", "packs"), "Content packs", "chip")])
        y += 14
        items.append(("label", pygame.Rect(0, y, width, 20), "__edit"))
        y = self._chips(items, y + 26, width, [
            (("edit", "undo"), "Undo", "chip" if m.can_undo else "muted"),
            (("edit", "redo"), "Redo", "chip" if m.can_redo else "muted")])
        return items, y + 14

    # ---- rendering --------------------------------------------------------------------
    def render(self, m, size):
        W, H = size
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        self.widgets = []
        self.tab = m.tab
        px, py, pw, ph = W - PANEL_W - MARGIN, MARGIN, PANEL_W, H - 2 * MARGIN
        self.panel_rect = pygame.Rect(px, py, pw, ph)
        hov = self.hover_action

        surf.blit(shadow(pw, ph, 18), (px - 26, py - 20))
        surf.blit(pill(pw, ph, 18, COL["panel"]), (px, py))
        x0, inner = px + 22, pw - 44

        # header
        surf.blit(text("Avatar Studio", True, 22, COL["text"]), (x0, py + 16))
        surf.blit(text("Create, animate, talk and export", False, 12, COL["muted"]), (x0, py + 46))

        # tabs: two rows of four
        cols, gap, tab_h = 4, 4, 28
        tw = (inner - gap * (cols - 1)) // cols
        ty = py + 72
        for i, (name, _) in enumerate(O.TABS):
            r = pygame.Rect(x0 + (i % cols) * (tw + gap), ty + (i // cols) * (tab_h + 6), tw, tab_h)
            act = ("tab", i)
            if i == m.tab:
                surf.blit(pill(r.w, r.h, 9, COL["accent"]), r)
            elif hov == act:
                surf.blit(pill(r.w, r.h, 9, COL["chip"]), r)
            label = text(name, True, 12, COL["text"] if i == m.tab or hov == act else COL["muted"])
            surf.blit(label, label.get_rect(center=r.center))
            self.widgets.append((r, act))
        tab_rows = (len(O.TABS) + cols - 1) // cols

        # scrolling content
        top = ty + tab_rows * (tab_h + 6) + 8
        footer_h = 146
        clip = pygame.Rect(px + 6, top, pw - 12, py + ph - footer_h - top)
        keys = O.TABS[m.tab][1]
        items, total_h, sections = self.layout(keys, m.state, inner - 10)
        tab_name = O.TABS[m.tab][0]
        if tab_name in ("Talk", "Library"):
            pre, y_pre = (self.layout_talk if tab_name == "Talk" else self.layout_library)(inner - 10, m)
            items = pre + [(k, r.move(0, y_pre), d) for k, r, d in items]
            sections = [(k, a + y_pre, b + y_pre) for k, a, b in sections]
            total_h += y_pre
        self.max_scroll = max(0, total_h - clip.h)
        focus_key = keys[min(m.focus, len(keys) - 1)] if keys else None
        sc = self.scroll.get(m.tab, 0)
        if self.focus_key == "pending":               # keep the keyboard focus visible
            for key, s_top, s_bot in sections:
                if key == focus_key:
                    if s_top < sc:
                        sc = s_top
                    elif s_bot > sc + clip.h:
                        sc = s_bot - clip.h
            self.focus_key = None
        sc = max(0, min(self.max_scroll, sc))
        self.scroll[m.tab] = sc
        ox, oy = x0, clip.y - sc

        surf.set_clip(clip)
        for key, s_top, s_bot in sections:
            if key == focus_key:
                surf.blit(pill(3, s_bot - s_top, 1, COL["accent"]), (px + 10, oy + s_top))
        for kind, rect, data in items:
            r = rect.move(ox, oy)
            if r.bottom < clip.top or r.top > clip.bottom:
                continue
            vis = r.clip(clip)
            if kind == "textbox":
                self._textbox(surf, r, m, data, hov)
                self.widgets.append((vis, ("textbox", data)))
            elif kind in ("abutton", "achip"):
                act, label, style = data
                self._action(surf, r, act, label, style, hov, chip=(kind == "achip"))
                if style != "muted":
                    self.widgets.append((vis, act))
            elif kind == "card":
                self._card(surf, r, data, hov, clip)
            elif kind == "note":
                for i, line in enumerate(wrap(data, False, 13, r.w)):
                    surf.blit(text(line, False, 13, COL["faint"]), (r.x, r.y + i * 18))
            elif kind == "label" and data in LABELS:
                suffix = f"  ({len(m.avatars)})" if data == "__mine" and m.avatars else ""
                surf.blit(text(LABELS[data] + suffix, True, 11, COL["muted"]), (r.x, r.y + 2))
            elif kind == "label":
                key = data
                surf.blit(text(O.label(key).upper(), True, 11, COL["text"] if key == focus_key else COL["muted"]),
                          (r.x, r.y + 2))
                val = O.choices(key)[m.state[key]][0]
                v = text(val, False, 12, COL["faint"])
                surf.blit(v, (r.right - v.get_width(), r.y + 1))
            else:
                key, j = data
                act = ("choose", key, j)
                selected = m.state[key] == j
                hovered = hov == act and clip.collidepoint(self.mouse)
                if kind == "swatch":
                    color = rgb(O.choices(key)[j][1])
                    if selected:
                        surf.blit(disc(30, COL["panel"], COL["accent_hover"], 3), r)
                        surf.blit(disc(20, color), (r.x + 5, r.y + 5))
                    else:
                        d, o = (28, 1) if hovered else (26, 2)
                        surf.blit(disc(d, color), (r.x + o, r.y + o))
                        if hovered:
                            surf.blit(disc(30, (0, 0, 0, 0), COL["muted"], 2), r)
                else:
                    name = chip_label(key, O.choices(key)[j][0])
                    bg = COL["accent"] if selected else (COL["chip_hover"] if hovered else COL["chip"])
                    surf.blit(pill(r.w, r.h, 10, bg), r)
                    label = text(name, False, 13, COL["text"] if selected or hovered else (200, 204, 220))
                    surf.blit(label, label.get_rect(center=r.center))
                if vis.w > 0 and vis.h > 0:
                    self.widgets.append((vis, act))
        surf.set_clip(None)
        if self.max_scroll:
            surf.blit(fade(clip.w, 18, COL["panel"], True), (clip.x, clip.y))
            surf.blit(fade(clip.w, 22, COL["panel"], False), (clip.x, clip.bottom - 22))
            track_h = clip.h - 8
            thumb_h = max(30, int(track_h * clip.h / total_h))
            thumb_y = clip.y + 4 + int((track_h - thumb_h) * sc / self.max_scroll)
            surf.blit(pill(4, thumb_h, 2, COL["line"]), (px + pw - 12, thumb_y))

        # footer: share code, action buttons, hint
        fy = py + ph - footer_h
        pygame.draw.line(surf, COL["line"], (x0, fy + 6), (x0 + inner, fy + 6))
        copy_r = pygame.Rect(x0 + inner - 58, fy + 20, 58, 30)
        box = pygame.Rect(x0, fy + 20, inner - 66, 30)
        surf.blit(pill(box.w, box.h, 9, COL["panel2"]), box)
        surf.blit(text(m.code, False, 11, COL["muted"]), (box.x + 10, box.y + 8))
        hc = hov == ("copy",)
        surf.blit(pill(copy_r.w, copy_r.h, 9, COL["chip_hover"] if hc else COL["chip"]), copy_r)
        cl = text("Copy", True, 12, COL["text"])
        surf.blit(cl, cl.get_rect(center=copy_r.center))
        self.widgets.append((copy_r, ("copy",)))

        buttons = [("Random", "random", True), ("Photo", "photo", False),
                   ("PNG", "png", False), ("GLB", "glb", False)]
        bw = (inner - 3 * 8) // 4
        for i, (label, name, primary) in enumerate(buttons):
            r = pygame.Rect(x0 + i * (bw + 8), fy + 64, bw, 40)
            act = ("button", name)
            hv = hov == act
            color = (COL["accent_hover"] if hv else COL["accent"]) if primary else (
                COL["chip_hover"] if hv else COL["chip"])
            surf.blit(pill(r.w, r.h, 12, color), r)
            t = text(label, True, 13, COL["text"])
            surf.blit(t, t.get_rect(center=r.center))
            self.widgets.append((r, act))
        hint = text("Press H for keyboard shortcuts", False, 11, COL["faint"])
        surf.blit(hint, hint.get_rect(midtop=(px + pw // 2, fy + 114)))

        # view switcher floating over the scene
        vcx = (W - RESERVED) // 2
        seg_w, bar_h = 64, 40
        labels = [(name, key) for key, name in VIEWS]
        bar_w = seg_w * len(labels) + 86 + 12
        bar = pygame.Rect(vcx - bar_w // 2, H - 66, bar_w, bar_h)
        surf.blit(shadow(bar.w, bar.h, 20, 14), (bar.x - 14, bar.y - 8))
        surf.blit(pill(bar.w, bar.h, 20, COL["panel"] + (235,)), bar)
        for i, (name, key) in enumerate(labels):
            r = pygame.Rect(bar.x + 6 + i * seg_w, bar.y + 5, seg_w, bar_h - 10)
            act = ("view", key)
            if m.view == key:
                surf.blit(pill(r.w, r.h, 15, COL["accent"]), r)
            elif hov == act:
                surf.blit(pill(r.w, r.h, 15, COL["chip"]), r)
            t = text(name, True, 13, COL["text"] if m.view == key or hov == act else COL["muted"])
            surf.blit(t, t.get_rect(center=r.center))
            self.widgets.append((r, act))
        r = pygame.Rect(bar.x + 6 + len(labels) * seg_w + 8, bar.y + 5, 74, bar_h - 10)
        act = ("toggle", "spin")
        if hov == act:
            surf.blit(pill(r.w, r.h, 15, COL["chip"]), r)
        surf.blit(disc(9, COL["ok"] if m.spin else COL["faint"]), (r.x + 10, r.centery - 4))
        t = text("Spin", True, 13, COL["text"] if m.spin or hov == act else COL["muted"])
        surf.blit(t, (r.x + 26, r.centery - t.get_height() // 2))
        self.widgets.append((r, act))

        # toast
        if m.toast and m.toast_alpha > 0:
            label = ellipsize(m.toast, False, 13, max(200, W - RESERVED - 120))
            t = text(label, False, 13, COL["text"])
            tw_, th_ = t.get_width() + 44, 38
            r = pygame.Rect(vcx - tw_ // 2, H - 118, tw_, th_)
            a = int(255 * min(1.0, m.toast_alpha))
            toast = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
            toast.blit(pill(r.w, r.h, 19, COL["panel2"] + (240,)), (0, 0))
            toast.blit(disc(8, COL["ok"]), (14, r.h // 2 - 4))
            toast.blit(t, (30, r.h // 2 - t.get_height() // 2))
            toast.set_alpha(a)
            surf.blit(toast, r)

        # shortcut sheet
        if m.help:
            veil = pygame.Surface((W - RESERVED, H), pygame.SRCALPHA)
            veil.fill((10, 11, 16, 150))
            surf.blit(veil, (0, 0))
            cw, chh = 440, 64 + len(SHORTCUTS) * 25
            card = pygame.Rect(vcx - cw // 2, max(12, (H - chh) // 2), cw, chh)
            surf.blit(shadow(cw, chh, 18), (card.x - 26, card.y - 20))
            surf.blit(pill(cw, chh, 18, COL["panel"]), card)
            surf.blit(text("Keyboard shortcuts", True, 18, COL["text"]), (card.x + 24, card.y + 18))
            for i, (k, desc) in enumerate(SHORTCUTS):
                y = card.y + 56 + i * 25
                surf.blit(pill(text_width(k, True, 12) + 16, 22, 7, COL["chip"]), (card.x + 24, y))
                surf.blit(text(k, True, 12, COL["text"]), (card.x + 32, y + 3))
                surf.blit(text(desc, False, 13, COL["muted"]), (card.x + 200, y + 2))

        self.surface = surf
        self.dirty = False
        self.uploaded = False

    # ---- pieces ------------------------------------------------------------------------------
    def _action(self, surf, r, act, label, style, hov, chip):
        hv = hov == act
        if style == "primary":
            color = COL["accent_hover"] if hv else COL["accent"]
            fg = COL["text"]
        elif style == "muted":
            color, fg = COL["panel2"], COL["faint"]
        else:
            color, fg = (COL["chip_hover"] if hv else COL["chip"]), (COL["text"] if hv else (200, 204, 220))
        surf.blit(pill(r.w, r.h, 10 if chip else 12, color), r)
        t = text(label, not chip, 13, fg)
        surf.blit(t, t.get_rect(center=r.center))

    def _thumb(self, info, w, h):
        """Thumbnail scaled to fill w x h (cached by file and modification time)."""
        key = (info.path, info.modified, w, h)
        if key not in self._thumbs:
            surf = None
            if info.thumbnail:
                try:
                    raw = pygame.image.load(io.BytesIO(info.thumbnail))
                    sw, sh = raw.get_size()
                    scale = max(w / sw, h / sh)
                    big = pygame.transform.smoothscale(raw, (max(w, int(sw * scale)), max(h, int(sh * scale))))
                    surf = big.subsurface(pygame.Rect((big.get_width() - w) // 2, (big.get_height() - h) // 2, w, h)).copy()
                except (pygame.error, ValueError):
                    surf = None
            self._thumbs[key] = surf
            if len(self._thumbs) > 64:
                self._thumbs.pop(next(iter(self._thumbs)))
        return self._thumbs[key]

    def _card(self, surf, r, info, hov, clip):
        act, dele = ("avatar", info.path), ("delete", info.path)
        hovered = hov in (act, dele) and clip.collidepoint(self.mouse)
        surf.blit(pill(r.w, r.h, 12, COL["chip_hover"] if hovered else COL["panel2"]), r)
        th = r.h - 28
        img_rect = pygame.Rect(r.x + 5, r.y + 5, r.w - 10, th - 5)
        thumb = self._thumb(info, img_rect.w, img_rect.h)
        if thumb is not None:
            mask = pill(img_rect.w, img_rect.h, 8, (255, 255, 255, 255))
            clipped = pygame.Surface(thumb.get_size(), pygame.SRCALPHA)
            clipped.blit(thumb, (0, 0))
            clipped.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            surf.blit(clipped, img_rect)
        else:
            surf.blit(pill(img_rect.w, img_rect.h, 8, COL["chip"]), img_rect)
            surf.blit(disc(28, COL["faint"]), (img_rect.centerx - 14, img_rect.centery - 14))
        name = text(ellipsize(info.name, True, 12, r.w - 14), True, 12, COL["text"])
        surf.blit(name, (r.x + 8, r.bottom - 22))
        vis = r.clip(clip)
        if vis.w > 0 and vis.h > 0:
            self.widgets.append((vis, act))
            if hovered:
                x = pygame.Rect(r.right - 27, r.y + 7, 20, 20).clip(clip)
                if x.w > 0:
                    surf.blit(disc(20, COL["danger"] if hov == dele else (0, 0, 0, 170)), (r.right - 27, r.y + 7))
                    t = text("x", True, 13, COL["text"])
                    surf.blit(t, t.get_rect(center=(r.right - 17, r.y + 16)))
                    self.widgets.append((x, dele))

    def _textbox(self, surf, r, m, field_name, hov):
        """Text field: multi-line for the speech box, one line for the name."""
        focused = m.focus_field == field_name
        content = m.say_text if field_name == "say" else m.name_text
        border = COL["accent"] if focused else (COL["chip_hover"] if hov == ("textbox", field_name) else None)
        surf.blit(pill(r.w, r.h, 12, COL["panel2"], border, 2) if border else pill(r.w, r.h, 12, COL["panel2"]), r)
        multi = r.h > 60
        if multi:
            lines = wrap(content, False, 14, r.w - 28) if content else []
            shown = lines[-3:] if lines else []
            placeholder = "Type what the avatar should say..."
        else:
            shown = [content[-int(max(1, (r.w - 28) / 7.5)):]] if content else []
            placeholder = "Name your avatar"
        if not content and not focused:
            surf.blit(text(placeholder, False, 14, COL["faint"]), (r.x + 14, r.y + (12 if multi else 12)))
        for i, line in enumerate(shown):
            surf.blit(text(line, False, 14, COL["text"]), (r.x + 14, r.y + 12 + i * 22))
        if focused and m.cursor_on:
            last = shown[-1] if shown else ""
            cx = r.x + 15 + text_width(last, False, 14)
            cy = r.y + 12 + max(0, len(shown) - 1) * 22
            pygame.draw.line(surf, COL["accent_hover"], (cx, cy + 1), (cx, cy + 18), 2)
        if multi:
            count = text(f"{len(content)}/240", False, 11, COL["faint"])
            surf.blit(count, (r.right - count.get_width() - 12, r.bottom - 20))

    # ---- OpenGL ---------------------------------------------------------------------------
    def draw(self, size):
        from OpenGL.GL import (GL_LINEAR, GL_QUADS, GL_RGBA, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                               GL_TEXTURE_MIN_FILTER, GL_UNSIGNED_BYTE, GL_DEPTH_TEST, glBegin,
                               glBindTexture, glColor4f, glDisable, glEnable, glEnd, glGenTextures,
                               glLoadIdentity, glMatrixMode, glOrtho, glTexCoord2f, glTexImage2D,
                               glTexParameteri, glTexSubImage2D, glUseProgram, glVertex2f,
                               glViewport, GL_MODELVIEW, GL_PROJECTION)
        w, h = size
        if self.tex is None:
            self.tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        if not self.uploaded:
            data = pygame.image.tostring(self.surface, "RGBA", False)
            if self.tex_size != (w, h):
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
                self.tex_size = (w, h)
            else:
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, data)
            self.uploaded = True
        glUseProgram(0)
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        for u, v, x, y in ((0, 0, 0, 0), (1, 0, w, 0), (1, 1, w, h), (0, 1, 0, h)):
            glTexCoord2f(u, v)
            glVertex2f(x, y)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
