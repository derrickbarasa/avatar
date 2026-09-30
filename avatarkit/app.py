"""The application: window, input handling, animation loop and command line."""
import argparse
import math
import os
import random
import sys
import time

import pygame

from . import options as O
from .builder import build_avatar
from .exporters import export_glb, export_obj
from .photo import apply_photo
from .poses import blink_angle, pose_at
from .render import Camera, Renderer
from .ui import PANEL_W, Panel

EXPORT_DIR = "exports"
VIEW_ORDER = ["bust", "face", "full"]
MIN_SIZE = (900, 560)


def stamp(ext):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    return os.path.join(EXPORT_DIR, time.strftime("avatar_%Y%m%d_%H%M%S")
                        + f"_{int(time.time() * 1000) % 1000:03d}" + ext)


def pick_file():
    """Ask for an image with a native dialog; None if cancelled or unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a photo", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")])
        root.destroy()
        return path or None
    except Exception:
        return None


def clipboard_get():
    try:
        raw = pygame.scrap.get(pygame.SCRAP_TEXT)
        return raw.decode("utf-8", "ignore").strip("\0 \r\n") if raw else ""
    except Exception:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            text = root.clipboard_get()
            root.destroy()
            return text.strip()
        except Exception:
            return ""


def clipboard_put(text):
    try:
        pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
        return True
    except Exception:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()
            return True
        except Exception:
            return False


def open_window(size):
    pygame.init()
    for attr, val in ((pygame.GL_MULTISAMPLEBUFFERS, 1), (pygame.GL_MULTISAMPLESAMPLES, 4),
                      (pygame.GL_ALPHA_SIZE, 8), (pygame.GL_DEPTH_SIZE, 24)):  # 16-bit depth z-fights clothes
        pygame.display.gl_set_attribute(attr, val)
    pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption("Avatar Creator")
    try:
        from pygame._sdl2.video import Window
        Window.from_display_module().minimum_size = MIN_SIZE
    except Exception:
        pass  # older pygame: the layout just clamps to MIN_SIZE
    try:
        pygame.scrap.init()
    except Exception:
        pass


class App:
    def __init__(self, args):
        open_window(args.size)
        self.args = args
        self.renderer, self.panel = Renderer(), Panel()
        self.state = dict(O.DEFAULT_STATE)
        self.setup_state(args)
        self.cam = Camera(args.view)
        self.cam.yaw = args.yaw
        self.tab, self.row = [n.lower() for n, _ in O.TABS].index(args.tab), 0
        self.message, self.message_until = "", 0.0
        self.dirty = True
        self.spin = args.shot is None
        self.dragging = False
        self.want = None   # pending capture: "png" or "transparent"
        self.rig = None
        self.rebuild()

    # ---- state ---------------------------------------------------------------------
    def setup_state(self, args):
        if args.load:
            O.load_state(self.state, args.load)
        if args.code:
            O.decode_state(args.code, self.state)
        if args.seed is not None:
            O.randomize(self.state, random.Random(args.seed))
        if args.preset:
            O.apply_preset(self.state, args.preset)
        if args.photo:
            print(apply_photo(self.state, args.photo))
        for item in args.set:
            key, _, name = item.partition("=")
            O.set_by_name(self.state, key, name)

    def rebuild(self):
        self.rig = build_avatar(O.resolve(self.state))
        self.dirty = True

    def notify(self, text):
        self.message, self.message_until, self.dirty = text, time.time() + 4, True

    def change(self, key, step):
        n = len(O.choices(key))
        self.state[key] = (self.state[key] + step) % n
        if key == "preset":
            O.apply_preset(self.state, O.choices("preset")[self.state["preset"]][0])
        if key not in O.NON_BUILD_KEYS or key == "preset":
            self.rebuild()
        self.dirty = True

    def randomize(self):
        O.randomize(self.state)
        self.rebuild()

    # ---- actions ---------------------------------------------------------------------
    def current_pose(self, t):
        animate = O.resolve(self.state)["animate"]
        name = O.resolve(self.state)["pose"]
        return pose_at(name, t, animate), (blink_angle(t) if animate else 0.0)

    def save_png(self, transparent):
        self.want = "transparent" if transparent else "png"

    def export(self, kind):
        t = self.clock_time()
        pose, _ = self.current_pose(t)
        path = stamp("." + kind)
        (export_glb if kind == "glb" else export_obj)(self.rig, path, pose)
        self.notify(f"Exported {path}")

    def choose_photo(self):
        path = pick_file()
        if not path:
            self.notify("Photo import cancelled (or tkinter unavailable)")
            return
        try:
            self.notify(apply_photo(self.state, path))
            self.rebuild()
        except Exception as exc:
            self.notify(f"Could not read photo: {exc}")

    def copy_code(self):
        code = O.encode_state(self.state)
        self.notify("Code copied to clipboard" if clipboard_put(code) else code)

    def paste_code(self):
        try:
            O.decode_state(clipboard_get(), self.state)
            self.rebuild()
            self.notify("Loaded avatar from clipboard code")
        except ValueError as exc:
            self.notify(f"Clipboard: {exc}")

    # ---- input -------------------------------------------------------------------------
    def size(self):
        w, h = pygame.display.get_window_size()
        return max(w, MIN_SIZE[0]), max(h, MIN_SIZE[1])

    def rows(self):
        return O.TABS[self.tab][1]

    def set_tab(self, i):
        self.tab, self.row, self.dirty = i % len(O.TABS), 0, True

    def on_key(self, e):
        mods = pygame.key.get_mods()
        ctrl, shift = mods & pygame.KMOD_CTRL, mods & pygame.KMOD_SHIFT
        k = e.key
        if k == pygame.K_ESCAPE:
            return False
        if k == pygame.K_TAB:
            self.set_tab(self.tab + (-1 if shift else 1))
        elif k == pygame.K_UP:
            self.row, self.dirty = (self.row - 1) % len(self.rows()), True
        elif k == pygame.K_DOWN:
            self.row, self.dirty = (self.row + 1) % len(self.rows()), True
        elif k in (pygame.K_LEFT, pygame.K_RIGHT):
            self.change(self.rows()[self.row], -1 if k == pygame.K_LEFT else 1)
        elif k == pygame.K_v and ctrl:
            self.paste_code()
        elif k == pygame.K_r:
            self.randomize()
        elif k == pygame.K_v:
            self.cam.set_view(VIEW_ORDER[(VIEW_ORDER.index(self.cam.view) + 1) % 3])
        elif k == pygame.K_SPACE:
            self.spin = not self.spin
        elif k == pygame.K_s:
            self.save_png(False)
        elif k == pygame.K_g:
            self.save_png(True)
        elif k == pygame.K_e:
            self.export("glb")
        elif k == pygame.K_o:
            self.export("obj")
        elif k == pygame.K_p:
            self.choose_photo()
        elif k == pygame.K_c:
            self.copy_code()
        elif k == pygame.K_LEFTBRACKET:
            self.change("preset", -1)
        elif k == pygame.K_RIGHTBRACKET:
            self.change("preset", 1)
        elif k == pygame.K_k:
            O.save_state(self.state, "avatar.json")
            self.notify("Saved avatar.json")
        elif k == pygame.K_l and os.path.exists("avatar.json"):
            O.load_state(self.state, "avatar.json")
            self.rebuild()
            self.notify("Loaded avatar.json")
        return True

    def on_click(self, pos):
        w, h = self.size()
        if pos[0] < w - PANEL_W:
            self.dragging, self.spin = True, False
            return
        hit = self.panel.hit(pos[0] - (w - PANEL_W), pos[1], self.tab, h)
        if not hit:
            return
        if hit[0] == "tab":
            self.set_tab(hit[1])
        elif hit[0] == "row":
            self.row, self.dirty = hit[1], True
            if hit[2]:
                self.change(self.rows()[hit[1]], hit[2])
        elif hit[1] == "random":
            self.randomize()
        elif hit[1] == "photo":
            self.choose_photo()
        elif hit[1] == "png":
            self.save_png(False)
        elif hit[1] == "glb":
            self.export("glb")

    def handle(self, e):
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            return self.on_key(e)
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            self.on_click(e.pos)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.dragging = False
        elif e.type == pygame.MOUSEMOTION and self.dragging:
            self.cam.yaw += e.rel[0] * 0.5
            self.cam.pitch = max(-60, min(60, self.cam.pitch + e.rel[1] * 0.4))
        elif e.type == pygame.MOUSEWHEEL:
            self.cam.zoom(e.y)
        elif e.type == pygame.VIDEORESIZE:
            self.dirty = True
        return True

    # ---- frame ---------------------------------------------------------------------------
    def clock_time(self):
        return self.args.time if self.args.shot is not None else pygame.time.get_ticks() / 1000.0

    def draw(self):
        w, h = self.size()
        view = (w - PANEL_W, h)
        t = self.clock_time()
        pose, blink = self.current_pose(t)
        bg = O.resolve(self.state)["bg"]
        if self.spin:
            self.cam.yaw += 0.4
        self.cam.update(self.rig.ground_y)
        self.renderer.draw_scene(self.rig, pose, self.cam, bg, view, blink)
        if self.want:
            transparent = self.want == "transparent"
            if transparent:
                self.renderer.draw_scene(self.rig, pose, self.cam, bg, view, blink, transparent=True)
            path = stamp(".png")
            pygame.image.save(self.renderer.grab(view, alpha=transparent), path)
            self.want = None
            self.notify(f"Saved {path}")
        if self.message and time.time() > self.message_until:
            self.message, self.dirty = "", True
        if self.dirty or self.panel.height != h:
            self.panel.render(self.state, self.tab, self.row, self.message,
                              O.encode_state(self.state), h)
            self.dirty = False
        self.panel.draw((w, h))
        return view

    def run(self):
        clock = pygame.time.Clock()
        running, frames = True, 0
        while running:
            for e in pygame.event.get():
                running = self.handle(e) and running
            view = self.draw()
            pygame.display.flip()
            clock.tick(60)
            frames += 1
            if self.args.shot and frames >= 3:
                self.save_shot(view)
                break
        pygame.quit()

    def save_shot(self, view):
        w, h = self.size()
        if self.args.transparent or self.args.bare:
            t = self.clock_time()
            pose, blink = self.current_pose(t)
            bg = O.resolve(self.state)["bg"]
            self.renderer.draw_scene(self.rig, pose, self.cam, bg, view, blink,
                                     transparent=self.args.transparent)
            surf = self.renderer.grab(view, alpha=self.args.transparent)
        else:
            surf = self.renderer.grab((w, h))
        pygame.image.save(surf, self.args.shot)
        print("saved", self.args.shot)


def make_sheet(args):
    """Render N random avatars into one contact-sheet PNG."""
    open_window(args.size)
    renderer = Renderer()
    w, h = args.size
    view = (w - PANEL_W, h)
    n = args.sheet
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cw, ch = 260, 300
    sheet = pygame.Surface((cols * cw, rows * ch))
    bg = O.BACKGROUNDS[0][1]
    for i in range(n):
        state = dict(O.DEFAULT_STATE)
        O.randomize(state, random.Random((args.seed or 0) + i))
        state["pose"] = 0
        rig = build_avatar(O.resolve(state))
        cam = Camera(args.view)
        cam.yaw = args.yaw
        for _ in range(80):
            cam.update(rig.ground_y)
        renderer.draw_scene(rig, pose_at("relaxed", 0.0, False), cam, bg, view)
        thumb = pygame.transform.smoothscale(renderer.grab(view), (cw, ch))
        sheet.blit(thumb, ((i % cols) * cw, (i // cols) * ch))
        pygame.display.flip()
    pygame.image.save(sheet, args.shot or "sheet.png")
    print("saved", args.shot or "sheet.png")
    pygame.quit()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="3D avatar creator")
    ap.add_argument("--shot", help="render one frame to this PNG and exit")
    ap.add_argument("--view", choices=VIEW_ORDER, default="bust")
    ap.add_argument("--yaw", type=float, default=20.0)
    ap.add_argument("--tab", choices=[n.lower() for n, _ in O.TABS], default="face",
                    help="side-panel tab to show first")
    ap.add_argument("--time", type=float, default=0.0, help="animation time for --shot")
    ap.add_argument("--seed", type=int, help="randomize using this seed")
    ap.add_argument("--set", action="append", default=[], metavar="OPTION=VALUE")
    ap.add_argument("--preset", choices=list(O.PRESETS), help="apply an outfit preset")
    ap.add_argument("--code", help="load an avatar share code")
    ap.add_argument("--load", help="load an avatar JSON file")
    ap.add_argument("--photo", help="suggest options from a photo")
    ap.add_argument("--bare", action="store_true", help="--shot: viewport only, no side panel")
    ap.add_argument("--transparent", action="store_true", help="--shot: transparent background")
    ap.add_argument("--sheet", type=int, metavar="N", help="render N random avatars to --shot")
    ap.add_argument("--size", default="1000x720", help="window size, e.g. 1200x800")
    args = ap.parse_args(argv)
    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
    except ValueError:
        ap.error("--size must look like 1000x720")
    args.size = (max(w, MIN_SIZE[0]), max(h, MIN_SIZE[1]))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.sheet:
            make_sheet(args)
        else:
            App(args).run()
    except ValueError as exc:
        sys.exit(str(exc))
