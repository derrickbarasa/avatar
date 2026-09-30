"""The application: window, input, animation loop, library, exports and command line."""
import argparse
import math
import os
import random
import sys
import tempfile
import time

import pygame

from . import options as O
from . import packs, recorder
from . import speech as speech_mod
from . import clips, store, tts
from .builder import build_avatar
from .exporters import export_glb, export_obj, export_skinned, export_vrm
from .history import History
from .photo import apply_photo
from .physics import HairSwing
from .poses import EMOTE_DURATION, blink_angle, emote_overlay, gaze_at, pose_at, talk_overlay
from .render import Camera, Renderer
from .ui import PHRASES, RESERVED, UI, UIModel

VIEW_ORDER = ["bust", "face", "full"]
PACK_MESSAGES = ([], [])            # (loaded summaries, errors) from the packs folder
MIN_SIZE = (900, 560)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
MODEL_EXPORTS = {"glb": (".glb", "GLB"), "skinned": ("_skinned.glb", "Skinned GLB"),
                 "animated": ("_animated.glb", "Animated GLB"),
                 "vrm": (".vrm", "VRM"), "obj": (".obj", "OBJ")}


# ---------------------------------------------------------------------------
# small OS helpers
# ---------------------------------------------------------------------------
def pick_photo():
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
    wanted = int(os.environ.get("AVATARKIT_MSAA", "8"))     # 0, 2, 4 or 8 samples per pixel
    for samples in [s for s in (8, 4, 2, 0) if s <= wanted]:  # smoothest edges the graphics card allows
        for attr, val in ((pygame.GL_MULTISAMPLEBUFFERS, 1 if samples else 0), (pygame.GL_MULTISAMPLESAMPLES, samples),
                          (pygame.GL_ALPHA_SIZE, 8), (pygame.GL_DEPTH_SIZE, 24)):  # 16-bit depth z-fights clothes
            pygame.display.gl_set_attribute(attr, val)
        try:
            pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
            break
        except pygame.error:
            if not samples:
                raise
    pygame.display.set_caption("Avatar Studio")
    try:
        from pygame._sdl2.video import Window
        Window.from_display_module().minimum_size = MIN_SIZE
    except Exception:
        pass  # older pygame: the layout just clamps to MIN_SIZE
    try:
        pygame.scrap.init()
    except Exception:
        pass


def even(n):
    return max(2, int(n) // 2 * 2)


# ---------------------------------------------------------------------------
class App:
    def __init__(self, args):
        open_window(args.size)
        self.args = args
        self.renderer, self.ui = Renderer(), UI()
        self.help, self.ui_key = False, None
        self.speech, self.speech_t0, self.job, self.sound = None, 0.0, None, None
        self.spin_before = False
        self.say_text, self.name_text, self.focus_field = PHRASES[0][1], "My avatar", ""
        self.voices_job = None if args.shot else tts.VoiceList()
        self.voice_langs = {}
        self.pending = []                 # capture requests, served right after the scene is drawn
        self.avatars, self.avatars_stale = [], True
        self.last_export = None
        self.export_dir = args.out
        self.autosave_due = None
        self.emote = None
        self.hair, self.jaw, self.last_t = HairSwing(), 0.0, None
        self.rig_cache = {}
        self.message, self.message_until = "", 0.0
        self.dirty, self.dragging = True, False
        self.spin = args.shot is None

        self.state = dict(O.DEFAULT_STATE)
        explicit = any([args.load, args.code, args.seed is not None, args.preset, args.photo, args.set])
        if args.shot is None and not args.fresh and not explicit:
            restored = store.restore_autosave(self.state)
            if restored:
                self.name_text = restored
        for line in PACK_MESSAGES[0]:
            print("pack:", line)
        self.setup_state(args)
        if args.name:
            self.name_text = args.name
        self.history = History(self.state)
        self.cam = Camera(args.view)
        self.cam.yaw = args.yaw
        self.tab, self.row = [n.lower() for n, _ in O.TABS].index(args.tab), 0
        self.rig = None
        self.rebuild()
        if args.emote:
            self.emote = (args.emote, 0.0 if args.shot else self.clock_time())
        if args.say:
            self.say_text = args.say[:240]
            if args.shot:
                self.speak_now_blocking()
            else:
                self.speak()

    # ---- state ---------------------------------------------------------------------
    def setup_state(self, args):
        if args.load:
            self.name_text = store.load_avatar(args.load, self.state)
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
        key = tuple(v for k, v in self.state.items() if k not in O.NON_BUILD_KEYS)
        rig = self.rig_cache.pop(key, None)
        if rig is None:
            rig = build_avatar(O.resolve(self.state))
        self.rig_cache[key] = rig                      # most recently used goes last
        while len(self.rig_cache) > 3:
            self.rig_cache.pop(next(iter(self.rig_cache)))
        self.rig = rig
        rig.set_mouth(0.0)
        self.dirty = True

    def notify(self, text):
        self.message, self.message_until, self.dirty = text, time.time() + 4, True

    def commit(self):
        """Record the current options for undo and schedule an autosave."""
        if self.history.commit(self.state):
            self.autosave_due = time.time() + 1.0

    def change(self, key, step):
        self.set_value(key, (self.state[key] + step) % len(O.choices(key)))

    def set_value(self, key, idx):
        self.state[key] = idx
        if key == "preset":
            O.apply_preset(self.state, O.choices("preset")[idx][0])
        if key not in O.NON_BUILD_KEYS or key == "preset":
            self.rebuild()
        self.commit()
        self.dirty = True

    def randomize(self):
        O.randomize(self.state)
        self.rebuild()
        self.commit()

    def apply_snapshot(self, snap):
        self.state.clear()
        self.state.update(snap)
        self.rebuild()
        self.autosave_due = time.time() + 1.0

    def undo(self):
        snap = self.history.undo()
        self.notify("Nothing to undo" if snap is None else "Undo")
        if snap is not None:
            self.apply_snapshot(snap)

    def redo(self):
        snap = self.history.redo()
        self.notify("Nothing to redo" if snap is None else "Redo")
        if snap is not None:
            self.apply_snapshot(snap)

    # ---- library ---------------------------------------------------------------------
    def avatar_name(self):
        return self.name_text.strip() or "My avatar"

    def refresh_avatars(self):
        self.avatars, self.avatars_stale, self.dirty = store.list_avatars(), False, True

    def save_avatar(self, dialog=False):
        """Save to the library (the thumbnail is grabbed from the next drawn frame)."""
        name, path = self.avatar_name(), None
        if dialog:
            path = store.ask_save_path(store.slug(name))
            if not path:
                self.notify("Save cancelled")
                return
        self.pending.append(("save", name, path))

    def open_dialog(self):
        path = store.ask_open_path()
        if path:
            self.load_path(path)
        else:
            self.notify("Open cancelled")

    def load_path(self, path):
        """Open an .avatar / .json / share-code file, or import a photo dropped on the window."""
        if path.lower().endswith(IMAGE_EXTS):
            try:
                self.notify(apply_photo(self.state, path))
                self.rebuild()
                self.commit()
            except Exception as exc:
                self.notify(f"Could not read photo: {exc}")
            return
        try:
            self.name_text = store.load_avatar(path, self.state)
        except ValueError as exc:
            self.notify(str(exc))
            return
        self.rebuild()
        self.commit()
        self.notify(f"Opened {self.avatar_name()}")

    def delete_avatar(self, path):
        name = next((a.name for a in self.avatars if a.path == path), os.path.basename(path))
        self.notify(f"Deleted {name}" if store.delete_avatar(path) else "Could not delete that file")
        self.refresh_avatars()

    def autosave_tick(self, force=False):
        if self.args.shot is not None:
            return
        if force or (self.autosave_due is not None and time.time() >= self.autosave_due):
            store.autosave(self.state, self.avatar_name())
            self.autosave_due = None

    # ---- exports -----------------------------------------------------------------------
    def out_dir(self):
        if self.export_dir:
            os.makedirs(self.export_dir, exist_ok=True)
            return self.export_dir
        return store.exports_dir()

    def out_path(self, ext):
        return store.unique_path(self.out_dir(), store.slug(self.avatar_name()), ext)

    def done_export(self, path, label=None):
        self.last_export = path
        self.notify(f"Exported {os.path.basename(path)}" + (f" ({label})" if label else ""))

    def export_named(self, kind):
        """Start an export: pictures and bundles wait for a drawn frame, models are written now."""
        if kind in ("png", "transparent", "bundle"):
            self.pending.append((kind,))
        elif kind == "clip":
            self.export_clip()
        elif kind in MODEL_EXPORTS:
            self.export_model(kind)

    def export_model(self, kind):
        ext, label = MODEL_EXPORTS[kind]
        pose = self.live_state(self.clock_time())[0]
        path = self.out_path(ext)
        name = self.avatar_name()
        if kind == "glb":
            export_glb(self.rig, path, pose, strand_fraction=0.5)
        elif kind == "skinned":
            export_skinned(self.rig, path, pose, name=name, strand_fraction=0.5)
        elif kind == "animated":
            export_skinned(self.rig, path, pose, name=name, strand_fraction=0.5,
                           clips=clips.all_clips(O.resolve(self.state)["pose"]))
        elif kind == "vrm":
            export_vrm(self.rig, path, pose, name=name)
        else:
            export_obj(self.rig, path, pose, strand_fraction=0.5)
        self.done_export(path, label)

    def make_bundle(self, png_surface, transparent_surface, pose):
        """Zip everything: models, pictures, the editable .avatar file and the share code."""
        name = self.avatar_name()
        base = store.slug(name)
        with tempfile.TemporaryDirectory() as tmp:
            files = {}

            def add(arc, writer):
                p = os.path.join(tmp, arc)
                writer(p)
                files[arc] = p

            add(base + ".glb", lambda p: export_glb(self.rig, p, pose, strand_fraction=0.5))
            add(base + "_skinned.glb", lambda p: export_skinned(
                self.rig, p, pose, name=name, strand_fraction=0.5,
                clips=clips.all_clips(O.resolve(self.state)["pose"])))       # bones + the pose loop and emotes
            add(base + ".vrm", lambda p: export_vrm(self.rig, p, pose, name=name))
            add(base + ".obj", lambda p: export_obj(self.rig, p, pose, strand_fraction=0.4))
            files[base + ".mtl"] = os.path.join(tmp, base + ".mtl")
            add(base + ".png", lambda p: pygame.image.save(png_surface, p))
            add(base + "_transparent.png", lambda p: pygame.image.save(transparent_surface, p))
            thumb = store.thumbnail_from(png_surface)
            add(base + ".avatar", lambda p: store.save_avatar(self.state, name, p, thumb))
            path = store.write_bundle(self.out_path(".zip"), name, files, O.encode_state(self.state))
        self.done_export(path, "bundle")

    def process_pending(self, view, size, pose, blink, gaze, lid, bg):
        """Serve capture requests using the frame that was just drawn."""
        while self.pending:
            kind, *rest = self.pending.pop(0)
            grab = lambda alpha=False: self.renderer.grab(view, alpha=alpha)
            # pictures are drawn 3x larger off-screen and shrunk: clean edges, and 2x the window's pixels
            shot = lambda alpha=False: self.renderer.capture(
                self.rig, pose, self.cam, bg, size, view, scale=2, supersample=3, alpha=alpha, blink=blink,
                gaze=gaze, lid=lid, reserved=RESERVED, **dict(self.look(), quality=1.0))   # pictures: always best
            if kind == "png":
                path = self.out_path(".png")
                pygame.image.save(shot(), path)
                self.done_export(path, "picture")
            elif kind in ("transparent", "bundle"):
                solid = shot()
                clear = shot(True)
                if kind == "transparent":
                    path = self.out_path("_transparent.png")
                    pygame.image.save(clear, path)
                    self.done_export(path, "transparent picture")
                else:
                    self.make_bundle(solid, clear, pose)
            elif kind == "save":
                name, path = rest
                thumb = store.thumbnail_from(grab())
                saved = store.save_avatar(self.state, name, path, thumb)
                self.name_text = name
                self.refresh_avatars()
                self.notify(f"Saved {name} to your library" if path is None else f"Saved {os.path.basename(saved)}")

    def export_clip(self, path=None):
        """Render the current line of speech to MP4 (with audio) or a GIF."""
        text = (self.say_text or PHRASES[0][1]).strip()
        r = O.resolve(self.state)
        audio = None
        try:
            audio = tts.synthesize(text, r["voice"], r["speed"])
        except tts.TTSError as exc:
            self.notify(f"No voice ({exc}); the clip will be silent")
        speech = self.analyse(text, audio, self.voice_lang(r["voice"]))
        fps, lead, tail = 30, 0.25, 0.6
        total = int((speech.duration + lead + tail) * fps)
        w, h = self.size()
        view = (w - RESERVED, h)
        scale = min(1.0, 720 / h)
        out_size = (even(view[0] * scale), even(h * scale))
        bg = r["bg"]
        yaw, self.cam.yaw = self.cam.yaw, round(self.cam.yaw / 360) * 360
        for _ in range(60):
            self.cam.update(self.rig.ground_y)

        def frames():
            for i in range(total):
                t = i / fps
                pose, blink, gaze, lid = self.frame_state(t, speech, t - lead)
                self.hair.update(1 / fps, pose, r["physics"])
                self.renderer.draw_scene(self.rig, pose, self.cam, bg, (w, h), blink, reserved=RESERVED,
                                         gaze=gaze, lid=lid, **self.look())
                surf = self.renderer.grab(view)
                if surf.get_size() != out_size:
                    surf = pygame.transform.smoothscale(surf, out_size)
                yield pygame.image.tostring(surf, "RGB")

        def progress(i, n):
            if i % 8 == 0 or i == n:
                self.notify(f"Rendering clip... {int(100 * i / n)}%")
                self.ui.dirty = True
                self.present()
                pygame.event.pump()

        target = path or self.out_path(".mp4")
        try:
            done, note = recorder.encode(target, frames(), out_size, fps, audio, progress, total, lead)
        except recorder.RecordError as exc:
            self.notify(f"Clip failed: {exc}")
            done = None
        finally:
            self.cam.yaw = yaw
            self.rig.set_mouth(0.0)
        if done:
            self.done_export(done, note or "clip")
        return done

    # ---- speech -----------------------------------------------------------------------
    @property
    def speaking(self):
        return self.speech is not None

    def voice_lang(self, voice):
        """Two-letter language of a voice (for the lip-sync rules); English if unknown."""
        return self.voice_langs.get(voice, "en")

    def analyse(self, text, path, lang="en"):
        """Lip-sync timeline for `text`, from the audio file if it can be read."""
        if path:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                arr = pygame.sndarray.array(pygame.mixer.Sound(path)).astype("float32") / 32768.0
                mono = arr.mean(1) if arr.ndim == 2 else arr
                return speech_mod.Speech.from_samples(text, mono, pygame.mixer.get_init()[0], lang)
            except (pygame.error, ValueError):
                pass
        return speech_mod.Speech.from_text_only(text, lang=lang)

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
        self.job.lang = self.voice_lang(r["voice"])
        self.notify("Preparing voice...")
        self.dirty = True

    def speak_now_blocking(self):
        """For --shot: synthesize on this thread so a frame at --time shows the right lips."""
        text = self.say_text.strip()
        r = O.resolve(self.state)
        lang = self.voice_lang(r["voice"])
        try:
            self.begin_speech(text, tts.synthesize(text, r["voice"], r["speed"]), None, lang)
        except tts.TTSError as exc:
            self.begin_speech(text, None, str(exc), lang)

    def begin_speech(self, text, path, error, lang="en"):
        """Start speaking: play the audio and run the lip-sync timeline (lips only if no audio)."""
        note = ""
        sp = self.analyse(text, path, lang) if path else None
        if sp is not None and self.args.shot is None:
            try:
                sound = pygame.mixer.Sound(path)
                sound.play()
                self.sound = sound
            except pygame.error as exc:
                note = f"Audio unavailable ({exc}); showing lips only"
        if not path:
            note = f"No voice ({error}); showing lips only"
            sp = speech_mod.Speech.from_text_only(text, lang=lang)
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
            self.begin_speech(job.text, job.path, job.error, getattr(job, "lang", "en"))
        if self.voices_job is not None and self.voices_job.done:
            if self.voices_job.voices:
                O.set_choices("voice", self.voices_job.voices)
                self.voice_langs = self.voices_job.langs
            self.voices_job, self.dirty = None, True

    # ---- animation state -----------------------------------------------------------------
    def frame_state(self, t, speech=None, tt=None, emote=None):
        """(pose, blink, gaze, lid) at time t: the chosen pose plus idle life, speech and emotes."""
        r = O.resolve(self.state)
        animate = r["animate"]
        pose = pose_at(r["pose"], t, animate)
        mouth = None
        if self.args.viseme:
            o, w, p = speech_mod.VISEMES[self.args.viseme]
            mouth = speech_mod.MouthShape(open=o, wide=w, press=p)
        elif speech is not None and tt is not None:
            mouth = speech.sample(tt)
        smile = raise_ = tilt = lid = 0.0
        if mouth is not None:
            talk_overlay(pose, mouth, t, r["gestures"])
            if speech is not None and tt is not None:
                smile, raise_, tilt, lid = speech.emotion(tt)
        gaze = gaze_at(t, mouth is not None) if animate else (0.0, 0.0)
        mouth_args = (mouth.open, mouth.wide, mouth.press) if mouth else (0.0, 0.0, 0.0)
        if emote is not None:
            ex = emote_overlay(pose, *emote)
            if ex["mouth"]:
                mouth_args, smile = ex["mouth"][:3], smile + ex["mouth"][3]
            raise_, tilt, lid = max(raise_, ex["brow_raise"]), tilt + ex["brow_tilt"], lid + ex["lid"]
            gaze = ex["gaze"] or gaze
        if raise_ or tilt:
            b = pose.get("brows", (0.0, 0.0, 0.0))
            pose["brows"] = (b[0] - 6.0 * raise_, b[1], b[2])
            pose["browL"], pose["browR"] = (0.0, 0.0, 12.0 * tilt), (0.0, 0.0, -12.0 * tilt)
        self.rig.set_mouth(*mouth_args, smile=smile)
        self.jaw = 0.105 * min(1.0, self.rig.base_open + mouth_args[0])
        return pose, (blink_angle(t) if animate else 0.0), gaze, lid

    def look(self):
        """Extra draw parameters: jaw drop, hair sway and whether shadows are on."""
        r = O.resolve(self.state)
        return {"jaw": self.jaw, "swing": self.hair.swing, "shadows": r["shadows"], "quality": r["quality"]}

    def live_state(self, t):
        speech = tt = emote = None
        if self.speech is not None:
            tt = t - self.speech_t0
            if tt > self.speech.duration + 0.3:
                self.stop_speech()
                tt = None
            else:
                speech = self.speech
        if self.emote:
            name, t0 = self.emote
            u = (t - t0) / EMOTE_DURATION[name]
            if u >= 1.0:
                self.emote = None
            else:
                emote = (name, max(0.0, u))
        return self.frame_state(t, speech, tt, emote)

    # ---- text fields ---------------------------------------------------------------------
    def field_key(self, e):
        """Keys typed while a text box has the keyboard."""
        k, ctrl = e.key, pygame.key.get_mods() & pygame.KMOD_CTRL
        field, limit = self.focus_field, (240 if self.focus_field == "say" else 48)
        text = self.say_text if field == "say" else self.name_text
        if k == pygame.K_ESCAPE:
            self.focus_field = ""
        elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.focus_field = ""
            if field == "say":
                self.speak()
            else:
                self.save_avatar()
        elif k == pygame.K_BACKSPACE:
            text = text[:-1]
        elif k == pygame.K_v and ctrl:
            text = (text + clipboard_get().replace("\r", " ").replace("\n", " "))[:limit]
        elif e.unicode and e.unicode.isprintable() and len(text) < limit:
            text += e.unicode
        if field == "say":
            self.say_text = text
        elif field == "name":
            self.name_text = text
            self.autosave_due = time.time() + 1.0
        self.dirty = True
        return True

    # ---- misc actions -------------------------------------------------------------------------
    def choose_photo(self):
        path = pick_photo()
        if not path:
            self.notify("Photo import cancelled (or tkinter unavailable)")
            return
        self.load_path(path)

    def copy_code(self):
        code = O.encode_state(self.state)
        self.notify("Code copied to clipboard" if clipboard_put(code) else code)

    def paste_code(self):
        try:
            O.decode_state(clipboard_get(), self.state)
            self.rebuild()
            self.commit()
            self.notify("Loaded avatar from clipboard code")
        except ValueError as exc:
            self.notify(f"Clipboard: {exc}")

    def open_packs(self):
        ok = store.open_folder(store.packs_dir())
        self.notify("Opened the packs folder" if ok else f"Packs go in {store.packs_dir()}")

    def open_exports(self, reveal=False):
        ok = store.open_folder(self.out_dir(), self.last_export if reveal else None)
        self.notify("Opened the exports folder" if ok else f"Exports are in {self.out_dir()}")

    # ---- input --------------------------------------------------------------------------------
    def size(self):
        w, h = pygame.display.get_window_size()
        return max(w, MIN_SIZE[0]), max(h, MIN_SIZE[1])

    def rows(self):
        return O.TABS[self.tab][1]

    def set_tab(self, i):
        self.tab, self.row, self.dirty = i % len(O.TABS), 0, True
        self.focus_field = ""
        if O.TABS[self.tab][0] == "Library":
            self.refresh_avatars()

    def on_key(self, e):
        mods = pygame.key.get_mods()
        ctrl, shift = mods & pygame.KMOD_CTRL, mods & pygame.KMOD_SHIFT
        k = e.key
        if self.focus_field:
            return self.field_key(e)
        if ctrl:
            if k == pygame.K_s:
                self.save_avatar(dialog=bool(shift))
            elif k == pygame.K_o:
                self.open_dialog()
            elif k == pygame.K_z:
                self.redo() if shift else self.undo()
            elif k == pygame.K_y:
                self.redo()
            elif k == pygame.K_v:
                self.paste_code()
            return True
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
            self.focus_field = "say"
            return True
        if k in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.speak()
            return True
        if k in (pygame.K_h, pygame.K_F1):
            self.help, self.dirty = not self.help, True
        elif k == pygame.K_TAB:
            self.set_tab(self.tab + (-1 if shift else 1))
        elif k == pygame.K_UP and self.rows():
            self.row, self.dirty = (self.row - 1) % len(self.rows()), True
            self.ui.focus_changed()
        elif k == pygame.K_DOWN and self.rows():
            self.row, self.dirty = (self.row + 1) % len(self.rows()), True
            self.ui.focus_changed()
        elif k in (pygame.K_LEFT, pygame.K_RIGHT) and self.rows():
            self.change(self.rows()[self.row], -1 if k == pygame.K_LEFT else 1)
        elif k == pygame.K_r:
            self.randomize()
        elif k == pygame.K_v:
            self.cam.set_view(VIEW_ORDER[(VIEW_ORDER.index(self.cam.view) + 1) % 3])
            self.dirty = True
        elif k == pygame.K_SPACE:
            self.spin, self.dirty = not self.spin, True
        elif k == pygame.K_s:
            self.export_named("png")
        elif k == pygame.K_g:
            self.export_named("transparent")
        elif k == pygame.K_e:
            self.export_named("glb")
        elif k == pygame.K_o:
            self.export_named("obj")
        elif k == pygame.K_b:
            self.export_named("bundle")
        elif k == pygame.K_p:
            self.choose_photo()
        elif k == pygame.K_c:
            self.copy_code()
        elif k == pygame.K_LEFTBRACKET:
            self.change("preset", -1)
        elif k == pygame.K_RIGHTBRACKET:
            self.change("preset", 1)
        elif k == pygame.K_k:
            self.save_avatar()
        elif k == pygame.K_l:
            self.open_dialog()
        return True

    def on_click(self, pos):
        w, h = self.size()
        if self.help:
            self.help, self.dirty = False, True
            return
        hit = self.ui.hit(pos)
        self.focus_field = hit[1] if hit and hit[0] == "textbox" else ""
        self.dirty = True
        if hit is None:
            if pos[0] < w - RESERVED:
                self.dragging, self.spin = True, False
            return
        kind = hit[0]
        if kind == "tab":
            self.set_tab(hit[1])
        elif kind == "choose":
            self.row = self.rows().index(hit[1]) if hit[1] in self.rows() else self.row
            self.set_value(hit[1], hit[2])
        elif kind == "view":
            self.cam.set_view(hit[1])
        elif kind == "toggle":
            self.spin = not self.spin
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
        elif kind == "emote":
            self.emote = (hit[1], self.clock_time())
        elif kind == "lib":
            {"save": self.save_avatar, "open": self.open_dialog, "folder": self.open_exports,
             "saveas": lambda: self.save_avatar(dialog=True), "packs": self.open_packs,
             
             "reveal": lambda: self.open_exports(reveal=True)}[hit[1]]()
        elif kind == "avatar":
            self.load_path(hit[1])
        elif kind == "delete":
            self.delete_avatar(hit[1])
        elif kind == "export":
            self.export_named(hit[1])
        elif kind == "edit":
            self.undo() if hit[1] == "undo" else self.redo()
        elif kind == "button":
            if hit[1] == "random":
                self.randomize()
            elif hit[1] == "photo":
                self.choose_photo()
            elif hit[1] == "png":
                self.export_named("png")
            elif hit[1] == "glb":
                self.export_named("glb")

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
        elif e.type == pygame.DROPFILE:
            self.load_path(e.file)
        return True

    # ---- frame ---------------------------------------------------------------------------
    def clock_time(self):
        return self.args.time if self.args.shot is not None else pygame.time.get_ticks() / 1000.0

    def ui_model(self):
        left = self.message_until - time.time()
        if self.message and left <= 0:
            self.message = ""
        alpha = 0.0 if not self.message else min(1.0, left / 0.5)
        busy = self.job is not None and not self.job.done
        cursor = bool(self.focus_field) and int(time.time() * 2) % 2 == 0
        model = UIModel(self.state, self.tab, self.row, O.encode_state(self.state), self.cam.view, self.spin,
                        self.message, alpha, self.help, self.say_text, self.name_text, self.focus_field,
                        cursor, self.speech is not None, busy, self.avatars, self.history.can_undo,
                        self.history.can_redo)
        key = (self.tab, self.row, self.cam.view, self.spin, self.message, round(alpha, 2), self.help,
               tuple(self.state.values()), self.say_text, self.name_text, self.focus_field, cursor,
               self.speech is not None, busy, len(O.choices("voice")), len(self.avatars),
               tuple(a.modified for a in self.avatars), self.history.can_undo, self.history.can_redo)
        return model, key

    def present(self):
        """Draw the interface over whatever is on screen and flip (also used while rendering a clip)."""
        w, h = self.size()
        model, key = self.ui_model()
        self.ui.render(model, (w, h))
        self.ui.draw((w, h))
        pygame.display.flip()

    def draw(self):
        w, h = self.size()
        view = (w - RESERVED, h)
        t = self.clock_time()
        self.poll_background()
        if self.avatars_stale and O.TABS[self.tab][0] == "Library":
            self.refresh_avatars()
        self.autosave_tick()
        pose, blink, gaze, lid = self.live_state(t)
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        self.hair.update(dt, pose, O.resolve(self.state)["physics"])
        bg = O.resolve(self.state)["bg"]
        if self.spin:
            self.cam.yaw += 0.4
        elif self.speech is not None and not self.dragging and self.args.shot is None:
            self.cam.yaw += (round(self.cam.yaw / 360) * 360 - self.cam.yaw) * 0.08   # turn to face us
        self.cam.update(self.rig.ground_y)
        self.renderer.draw_scene(self.rig, pose, self.cam, bg, (w, h), blink, reserved=RESERVED, gaze=gaze,
                                 lid=lid, **self.look())
        if self.pending:
            self.process_pending(view, (w, h), pose, blink, gaze, lid, bg)
        model, key = self.ui_model()
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
            if self.args.shot is not None and frames >= 3:
                self.finish_shot(view)
                break
        self.autosave_tick(force=self.args.shot is None)
        self.stop_speech()
        pygame.quit()

    def finish_shot(self, view):
        """Non-interactive modes: save the picture and/or requested exports, then exit."""
        args = self.args
        for kind in args.export:
            self.export_named(kind)
        if args.clip:
            self.export_clip(args.clip)
        while self.pending:                                     # let the drawn frame serve them
            self.draw()
        if args.save:
            self.pending.append(("save", args.save, None))
            self.draw()
        w, h = self.size()
        if args.shot:
            if args.transparent or args.bare:
                t = self.clock_time()
                pose, blink, gaze, lid = self.live_state(t)
                surf = self.renderer.capture(
                    self.rig, pose, self.cam, O.resolve(self.state)["bg"], (w, h), view, scale=1, supersample=3,
                    alpha=args.transparent, blink=blink, gaze=gaze, lid=lid, reserved=RESERVED,
                    **dict(self.look(), quality=1.0))
            else:
                surf = self.renderer.grab((w, h))              # the window as you see it, card included
            pygame.image.save(surf, args.shot)
            print("saved", args.shot)
        if self.last_export and args.export:
            print("exported", self.last_export)


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


def list_library():
    for a in store.list_avatars():
        print(f"{a.name}\t{a.path}")
    print(f"({store.avatars_dir()})")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Avatar Studio: create, animate, talk and export 3D avatars")
    ap.add_argument("--shot", help="render one frame to this PNG and exit")
    ap.add_argument("--view", choices=VIEW_ORDER + ["hands"], default="bust")
    ap.add_argument("--yaw", type=float, default=20.0)
    ap.add_argument("--tab", choices=[n.lower() for n, _ in O.TABS], default="face",
                    help="side-panel tab to show first")
    ap.add_argument("--time", type=float, default=0.0, help="animation time for --shot")
    ap.add_argument("--say", help="make the avatar say this (with --shot: lips at --time)")
    ap.add_argument("--viseme", choices=sorted(speech_mod.VISEMES), help="hold one mouth shape")
    ap.add_argument("--emote", choices=sorted(EMOTE_DURATION), help="play an emote (with --shot: at --time)")
    ap.add_argument("--seed", type=int, help="randomize using this seed")
    ap.add_argument("--set", action="append", default=[], metavar="OPTION=VALUE")
    ap.add_argument("--preset", choices=list(O.PRESETS), help="apply an outfit preset")
    ap.add_argument("--code", help="load an avatar share code")
    ap.add_argument("--load", help="load an .avatar file (or option JSON)")
    ap.add_argument("--photo", help="suggest options from a photo")
    ap.add_argument("--name", help="avatar name used for saving and file names")
    ap.add_argument("--save", metavar="NAME", help="save the avatar to your library (with --shot / --export)")
    ap.add_argument("--export", action="append", default=[], metavar="KIND",
                    choices=["png", "transparent", "glb", "skinned", "animated", "vrm", "obj", "bundle"],
                    help="write an export and exit (repeatable): png transparent glb skinned vrm obj bundle")
    ap.add_argument("--clip", metavar="FILE", help="render the --say line to an .mp4 (or .gif) and exit")
    ap.add_argument("--out", help="folder for exports (default: your library's exports folder)")
    ap.add_argument("--fresh", action="store_true", help="don't reopen the autosaved avatar")
    ap.add_argument("--list", action="store_true", help="list saved avatars and exit")
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
    if (args.export or args.clip or args.save) and args.shot is None:
        args.shot = ""                       # exports are non-interactive: render, write, exit
    return args


def main(argv=None):
    PACK_MESSAGES_LOAD = packs.load_all()          # before parsing, so pack presets work as --preset
    PACK_MESSAGES[0][:], PACK_MESSAGES[1][:] = PACK_MESSAGES_LOAD
    args = parse_args(argv)
    if args.list:
        list_library()
        return
    try:
        if args.sheet:
            make_sheet(args)
        else:
            App(args).run()
    except ValueError as exc:
        sys.exit(str(exc))
