"""The application: window, input handling, animation loop and command line."""
import argparse
import math
import os
import random
import sys
import time

import pygame

from . import options as O
from . import speech as speech_mod
from . import tts
from .builder import build_avatar
from .exporters import export_glb, export_obj
from .photo import apply_photo
from .poses import blink_angle, gaze_at, pose_at, talk_overlay
from .render import Camera, Renderer
from .ui import PHRASES, RESERVED, UI, UIModel

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
        self.renderer, self.ui = Renderer(), UI()
        self.help, self.ui_key = False, None
        self.speech, self.speech_t0, self.job, self.sound = None, 0.0, None, None
        self.spin_before = False
        self.say_text, self.say_focus = PHRASES[0][1], False
        self.voices_job = None if args.shot else tts.VoiceList()
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
        if args.say:
            self.say_text = args.say[:240]
            if args.shot:
                self.speak_now_blocking()
            else:
                self.speak()

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
        self.set_value(key, (self.state[key] + step) % len(O.choices(key)))

    def set_value(self, key, idx):
        self.state[key] = idx
        if key == "preset":
            O.apply_preset(self.state, O.choices("preset")[idx][0])
        if key not in O.NON_BUILD_KEYS or key == "preset":
            self.rebuild()
        self.dirty = True

    def randomize(self):
        O.randomize(self.state)
        self.rebuild()

    # ---- actions ---------------------------------------------------------------------
    def current_pose(self, t):
        """(pose, blink, gaze) for time t, with speech body language and lips applied."""
        r = O.resolve(self.state)
        animate = r["animate"]
        pose = pose_at(r["pose"], t, animate)
        mouth = self.mouth_now()
        if mouth is not None:
            talk_overlay(pose, mouth, t, r["gestures"])
            self.rig.set_mouth(mouth.open, mouth.wide, mouth.press)
        else:
            self.rig.set_mouth(0.0)
        blink = blink_angle(t) if animate else 0.0
        gaze = gaze_at(t, mouth is not None) if animate else (0.0, 0.0)
        return pose, blink, gaze

    # ---- speech --------------------------------------------------------------------------
    @property
    def speaking(self):
        return self.speech is not None

    def mouth_now(self):
        """Current MouthShape while speaking (or the --viseme override), else None."""
        if self.args.viseme:
            o, w, p = speech_mod.VISEMES[self.args.viseme]
            return speech_mod.MouthShape(open=o, wide=w, press=p)
        if self.speech is None:
            return None
        tt = self.clock_time() - self.speech_t0
        if tt > self.speech.duration + 0.3:
            self.stop_speech()
            return None
        return self.speech.sample(tt)

    def speak(self):
        """Speak the text box (a second press stops it). Voice synthesis runs in the background."""
        if self.speech is not None:
            self.stop_speech()
            return
        if self.job is not None and not self.job.done:
            return
        text = self.say_text.strip()
        if not text:
            self.notify("Type something for the avatar to say")
            return
        r = O.resolve(self.state)
        self.job = tts.Job(text, r["voice"], r["speed"])
        self.notify("Preparing voice...")
        self.dirty = True

    def speak_now_blocking(self):
        """For --shot: synthesize on this thread so a frame at --time shows the right lips."""
        text = self.say_text.strip()
        r = O.resolve(self.state)
        try:
            self.begin_speech(text, tts.synthesize(text, r["voice"], r["speed"]), None)
        except tts.TTSError as exc:
            self.begin_speech(text, None, str(exc))

    def begin_speech(self, text, path, error):
        """Analyse the audio for lip-sync and start playing it (falls back to lips-only)."""
        sp, note = None, ""
        if path:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                sound = pygame.mixer.Sound(path)
                arr = pygame.sndarray.array(sound).astype("float32") / 32768.0
                mono = arr.mean(1) if arr.ndim == 2 else arr
                sp = speech_mod.Speech.from_samples(text, mono, pygame.mixer.get_init()[0])
                if self.args.shot is None:
                    sound.play()
                self.sound = sound
            except (pygame.error, ValueError) as exc:
                note = f"Audio unavailable ({exc}); showing lips only"
        else:
            note = f"No voice ({error}); showing lips only"
        if sp is None:
            sp = speech_mod.Speech.from_text_only(text)
        if note:
            self.notify(note)
        else:
            self.message = ""          # drop "Preparing voice..."
        self.speech = sp
        self.speech_t0 = 0.0 if self.args.shot is not None else self.clock_time()
        self.spin_before, self.spin = self.spin, False      # face the viewer while talking
        self.dirty = True

    def stop_speech(self):
        try:
            pygame.mixer.stop()
        except pygame.error:
            pass
        if self.speech is not None:
            self.spin = self.spin_before
        self.speech, self.sound, self.dirty = None, None, True
        if self.rig:
            self.rig.set_mouth(0.0)

    def poll_background(self):
        if self.job is not None and self.job.done:
            job, self.job = self.job, None
            self.begin_speech(job.text, job.path, job.error)
        if self.voices_job is not None and self.voices_job.done:
            if self.voices_job.voices:
                O.set_choices("voice", self.voices_job.voices)
            self.voices_job, self.dirty = None, True

    def say_key(self, e):
        """Keys typed while the text box has focus."""
        k, ctrl = e.key, pygame.key.get_mods() & pygame.KMOD_CTRL
        if k == pygame.K_ESCAPE:
            self.say_focus = False
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.say_focus = False
            self.speak()
        elif k == pygame.K_BACKSPACE:
            self.say_text = self.say_text[:-1]
        elif k == pygame.K_v and ctrl:
            self.say_text = (self.say_text + clipboard_get().replace("\r", " ").replace("\n", " "))[:240]
        elif e.unicode and e.unicode.isprintable() and len(self.say_text) < 240:
            self.say_text += e.unicode
        self.dirty = True
        return True

    def save_png(self, transparent):
        self.want = "transparent" if transparent else "png"

    def export(self, kind):
        t = self.clock_time()
        pose, _, _ = self.current_pose(t)
        path = stamp("." + kind)
        (export_glb if kind == "glb" else export_obj)(self.rig, path, pose)
        self.notify(f"Exported {os.path.basename(path)}")

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
        if self.say_focus:
            return self.say_key(e)
        if k == pygame.K_ESCAPE:
            if self.help:
                self.help, self.dirty = False, True
                return True
            if self.speech is not None or self.job is not None:
                self.stop_speech()
                self.job = None
                return True
            return False
        if k == pygame.K_t:
            self.set_tab([n for n, _ in O.TABS].index("Talk"))
            self.say_focus = True
            return True
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.speak()
            return True
        if k in (pygame.K_h, pygame.K_F1):
            self.help, self.dirty = not self.help, True
        elif k == pygame.K_TAB:
            self.set_tab(self.tab + (-1 if shift else 1))
        elif k == pygame.K_UP:
            self.row, self.dirty = (self.row - 1) % len(self.rows()), True
            self.ui.focus_changed()
        elif k == pygame.K_DOWN:
            self.row, self.dirty = (self.row + 1) % len(self.rows()), True
            self.ui.focus_changed()
        elif k in (pygame.K_LEFT, pygame.K_RIGHT):
            self.change(self.rows()[self.row], -1 if k == pygame.K_LEFT else 1)
        elif k == pygame.K_v and ctrl:
            self.paste_code()
        elif k == pygame.K_r:
            self.randomize()
        elif k == pygame.K_v:
            self.cam.set_view(VIEW_ORDER[(VIEW_ORDER.index(self.cam.view) + 1) % 3])
            self.dirty = True
        elif k == pygame.K_SPACE:
            self.spin, self.dirty = not self.spin, True
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
        if self.help:
            self.help, self.dirty = False, True
            return
        hit = self.ui.hit(pos)
        self.say_focus = hit == ("textbox",)
        self.dirty = True
        if hit is None:
            if pos[0] < w - RESERVED:
                self.dragging, self.spin, self.dirty = True, False, True
            return
        kind = hit[0]
        if kind == "tab":
            self.set_tab(hit[1])
        elif kind == "choose":
            self.row = self.rows().index(hit[1]) if hit[1] in self.rows() else self.row
            self.set_value(hit[1], hit[2])
        elif kind == "view":
            self.cam.set_view(hit[1])
            self.dirty = True
        elif kind == "toggle":
            self.spin, self.dirty = not self.spin, True
        elif kind == "copy":
            self.copy_code()
        elif kind == "say":
            if hit[1] == "speak":
                self.speak()
            else:
                self.say_text = random.choice(PHRASES)[1]
        elif kind == "phrase":
            self.say_text = PHRASES[hit[1]][1]
            if self.speech is not None:
                self.stop_speech()
            self.speak()
        elif kind == "textbox":
            self.say_focus = True
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
        elif e.type == pygame.MOUSEMOTION:
            self.ui.hover(e.pos)
            if self.dragging:
                self.cam.yaw += e.rel[0] * 0.5
                self.cam.pitch = max(-60, min(60, self.cam.pitch + e.rel[1] * 0.4))
        elif e.type == pygame.MOUSEWHEEL:
            if not self.ui.scroll_by(e.y, pygame.mouse.get_pos()):
                self.cam.zoom(e.y)
        elif e.type == pygame.VIDEORESIZE:
            self.dirty = True
        return True

    # ---- frame ---------------------------------------------------------------------------
    def clock_time(self):
        return self.args.time if self.args.shot is not None else pygame.time.get_ticks() / 1000.0

    def draw(self):
        w, h = self.size()
        view = (w - RESERVED, h)
        t = self.clock_time()
        self.poll_background()
        pose, blink, gaze = self.current_pose(t)
        bg = O.resolve(self.state)["bg"]
        if self.spin:
            self.cam.yaw += 0.4
        elif self.speech is not None and not self.dragging and self.args.shot is None:
            self.cam.yaw += (round(self.cam.yaw / 360) * 360 - self.cam.yaw) * 0.08   # turn to face us
        self.cam.update(self.rig.ground_y)
        self.renderer.draw_scene(self.rig, pose, self.cam, bg, (w, h), blink, reserved=RESERVED, gaze=gaze)
        if self.want:
            transparent = self.want == "transparent"
            if transparent:
                self.renderer.draw_scene(self.rig, pose, self.cam, bg, (w, h), blink, transparent=True,
                                         reserved=RESERVED, gaze=gaze)
            path = stamp(".png")
            pygame.image.save(self.renderer.grab(view, alpha=transparent), path)
            self.want = None
            self.notify(f"Saved {os.path.basename(path)}")
        left = self.message_until - time.time()
        if self.message and left <= 0:
            self.message = ""
        alpha = 0.0 if not self.message else min(1.0, left / 0.5)
        busy = self.job is not None and not self.job.done
        cursor = self.say_focus and int(time.time() * 2) % 2 == 0
        model = UIModel(self.state, self.tab, self.row, O.encode_state(self.state), self.cam.view,
                        self.spin, self.message, alpha, self.help, self.say_text, self.say_focus,
                        cursor, self.speech is not None, busy)
        key = (self.tab, self.row, self.cam.view, self.spin, self.message, round(alpha, 2),
               self.help, (w, h), tuple(self.state.values()), self.say_text, self.say_focus, cursor,
               self.speech is not None, busy, len(O.choices("voice")))
        if key != self.ui_key or self.ui.dirty:
            self.ui.render(model, (w, h))
            self.ui_key = key
        self.ui.draw((w, h))
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
            pose, blink, gaze = self.current_pose(t)
            bg = O.resolve(self.state)["bg"]
            self.renderer.draw_scene(self.rig, pose, self.cam, bg, (w, h), blink,
                                     transparent=self.args.transparent, reserved=RESERVED, gaze=gaze)
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
    view = (w - RESERVED, h)
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
        renderer.draw_scene(rig, pose_at("relaxed", 0.0, False), cam, bg, (w, h), reserved=RESERVED)
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
    ap.add_argument("--say", help="make the avatar say this (with --shot: lips at --time)")
    ap.add_argument("--viseme", choices=sorted(speech_mod.VISEMES), help="hold one mouth shape")
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
