"""The customizer side panel: tabs, option rows and buttons, drawn with pygame
onto a texture and hit-tested by the app."""
import pygame
from OpenGL.GL import *  # noqa: F401,F403

from . import options as O

PANEL_W = 320
TAB_Y, TAB_H = 44, 26
ROW0, ROW_H = 84, 26
BUTTONS = [("Random", "random"), ("Photo", "photo"), ("PNG", "png"), ("GLB", "glb")]
BG = (26, 28, 36)
ACCENT = (86, 130, 255)


def _fonts():
    names = "segoeui,arial,helvetica"
    return (pygame.font.SysFont(names, 15), pygame.font.SysFont(names, 21, bold=True),
            pygame.font.SysFont(names, 12), pygame.font.SysFont(names, 13, bold=True))


class Panel:
    def __init__(self):
        self.font, self.title, self.small, self.tab_font = _fonts()
        self.tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.height = 0

    # ---- layout ---------------------------------------------------------------
    @staticmethod
    def tab_rect(i):
        w = (PANEL_W - 16) / len(O.TABS)
        return pygame.Rect(8 + int(i * w), TAB_Y, int(w) - 2, TAB_H)

    @staticmethod
    def row_rect(i):
        return pygame.Rect(6, ROW0 + i * ROW_H, PANEL_W - 12, ROW_H - 2)

    @staticmethod
    def button_rects(height):
        w = (PANEL_W - 16 - 3 * 6) // len(BUTTONS)
        return [pygame.Rect(8 + i * (w + 6), height - 122, w, 28) for i in range(len(BUTTONS))]

    def hit(self, x, y, tab, height):
        """Translate a click at panel coordinates into an action tuple (or None)."""
        for i in range(len(O.TABS)):
            if self.tab_rect(i).collidepoint(x, y):
                return ("tab", i)
        for i, (_, name) in enumerate(BUTTONS):
            if self.button_rects(height)[i].collidepoint(x, y):
                return ("button", name)
        for i in range(len(O.TABS[tab][1])):
            if self.row_rect(i).collidepoint(x, y):
                side = -1 if x < 168 else (1 if x > PANEL_W - 64 else 0)
                return ("row", i, side)
        return None

    # ---- drawing ----------------------------------------------------------------
    def render(self, state, tab, row, message, code, height):
        self.height = height
        s = pygame.Surface((PANEL_W, height), pygame.SRCALPHA)
        s.fill(BG + (255,))
        s.blit(self.title.render("Avatar Creator", True, (240, 240, 245)), (16, 10))
        for i, (name, _) in enumerate(O.TABS):
            r = self.tab_rect(i)
            active = i == tab
            pygame.draw.rect(s, ACCENT if active else (44, 48, 62), r, border_radius=6)
            txt = self.tab_font.render(name, True, (255, 255, 255) if active else (160, 166, 184))
            s.blit(txt, txt.get_rect(center=r.center))
        keys = O.TABS[tab][1]
        for i, key in enumerate(keys):
            r = self.row_rect(i)
            if i == row:
                pygame.draw.rect(s, (52, 58, 78), r, border_radius=6)
            s.blit(self.font.render(O.label(key), True, (170, 176, 192)), (16, r.y + 3))
            name, val = O.choices(key)[state[key]]
            x = 150
            s.blit(self.font.render("‹", True, (120, 128, 150)), (x, r.y + 2))
            if key in O.COLOR_KEYS:
                pygame.draw.rect(s, [int(c * 255) for c in val], (x + 16, r.y + 5, 14, 14), border_radius=3)
                x += 20
            s.blit(self.font.render(name, True, (235, 236, 242)), (x + 18, r.y + 3))
            s.blit(self.font.render("›", True, (120, 128, 150)), (PANEL_W - 24, r.y + 2))
        y = ROW0 + len(keys) * ROW_H + 10
        s.blit(self.small.render("Share code (C copies, Ctrl+V pastes)", True, (120, 126, 146)), (16, y))
        s.blit(self.small.render(code, True, (190, 196, 215)), (16, y + 16))
        for r, (label, _) in zip(self.button_rects(height), BUTTONS):
            pygame.draw.rect(s, (44, 48, 62), r, border_radius=7)
            txt = self.tab_font.render(label, True, (225, 228, 238))
            s.blit(txt, txt.get_rect(center=r.center))
        y = height - 82
        for line in ("Tab: switch tab   ↑/↓ select   ←/→ change",
                     "R random  V view  Space spin  [ ] presets",
                     "S PNG  G transparent  E GLB  O OBJ  P photo",
                     "K/L save/load   Drag: orbit   Wheel: zoom"):
            s.blit(self.small.render(line, True, (118, 124, 144)), (16, y))
            y += 15
        if message:
            s.blit(self.small.render(message[:52], True, (140, 210, 160)), (16, height - 18))
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, PANEL_W, height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     pygame.image.tostring(s, "RGBA", False))

    def draw(self, size):
        w, h = size
        glUseProgram(0)
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glColor4f(1, 1, 1, 1)
        x0 = w - PANEL_W
        glBegin(GL_QUADS)
        for u, v, x, y in ((0, 0, x0, 0), (1, 0, w, 0), (1, 1, w, h), (0, 1, x0, h)):
            glTexCoord2f(u, v)
            glVertex2f(x, y)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
