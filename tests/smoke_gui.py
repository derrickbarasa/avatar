"""Manual GUI smoke test (needs a display and OpenGL): not collected by unittest.

    python tests/smoke_gui.py

Drives the real app through its event handlers: tabs, chips, typing, saving to a temporary
library, reopening, undo/redo, every export format, a talking clip and speech.
"""
import os
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
home = tempfile.mkdtemp()
os.environ["AVATARKIT_HOME"] = home
os.chdir(tempfile.mkdtemp())

import pygame  # noqa: E402

from avatarkit import app, options as O, store  # noqa: E402

a = app.App(app.parse_args(["--size", "1200x780", "--fresh"]))


def frame(n=1):
    for _ in range(n):
        a.draw()
        pygame.display.flip()


def key(k, uni="", mod=0):
    pygame.key.set_mods(mod)
    a.handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=uni))
    pygame.key.set_mods(0)
    frame()


def find(action):
    for _ in range(25):                          # scroll the panel until the control is on screen
        rect = next((r for r, act in a.ui.widgets if act == action), None)
        if rect is not None:
            return rect
        a.ui.scroll_by(-2, a.ui.panel_rect.center)
        frame()
    return None


def click(action):
    frame()
    a.ui.scroll[a.tab] = 0
    frame()
    rect = find(action)
    assert rect is not None, f"no widget for {action}"
    a.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=rect.center, rel=(0, 0), buttons=(0, 0, 0)))
    a.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center))
    a.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, button=1, pos=rect.center))
    frame(2)


def tab(name):
    click(("tab", [n for n, _ in O.TABS].index(name)))


def type_text(text):
    for ch in text:
        key(ord(ch), ch)


frame(3)
print("home:", home)

# -- customise, then save through the Library tab ---------------------------------------
tab("Hair")
click(("choose", "hair", 4))                    # Long
click(("choose", "haircolor", 5))               # Blonde
tab("Library")
click(("textbox", "name"))
assert a.focus_field == "name"
a.name_text = ""
type_text("Sky Walker")
assert a.name_text == "Sky Walker"
click(("lib", "save"))
frame(3)
assert os.path.exists(os.path.join(store.avatars_dir(), "Sky_Walker.avatar")), os.listdir(store.avatars_dir())
assert [x.name for x in a.avatars] == ["Sky Walker"], a.avatars
assert a.avatars[0].thumbnail, "the thumbnail should be stored inside the file"
print("saved:", a.message)

# -- change things, save a second avatar, then reopen the first from the gallery ----------
tab("Hair")
click(("choose", "hair", 10))                   # Mohawk
click(("choose", "haircolor", 8))               # Pink
tab("Library")
a.name_text = "Punk"
key(pygame.K_s, "s", pygame.KMOD_CTRL)          # Ctrl+S
frame(3)
assert {x.name for x in a.avatars} == {"Sky Walker", "Punk"}
first = next(x for x in a.avatars if x.name == "Sky Walker")
click(("avatar", first.path))
assert O.resolve(a.state)["hair"] == "long" and a.name_text == "Sky Walker"
print("reopened:", a.message)

# -- undo / redo ---------------------------------------------------------------------------
before = O.encode_state(a.state)
tab("Body")
click(("choose", "height", 2))
assert O.encode_state(a.state) != before
key(pygame.K_z, "z", pygame.KMOD_CTRL)
assert O.encode_state(a.state) == before, "Ctrl+Z should undo the height change"
key(pygame.K_y, "y", pygame.KMOD_CTRL)
assert O.resolve(a.state)["height"] > 1.05
tab("Library")
click(("edit", "undo"))
assert O.resolve(a.state)["height"] == 1.0

# -- delete ----------------------------------------------------------------------------------
punk = next(x for x in a.avatars if x.name == "Punk")
click(("delete", punk.path)) if any(act == ("delete", punk.path) for _, act in a.ui.widgets) else a.delete_avatar(punk.path)
assert [x.name for x in a.avatars] == ["Sky Walker"]

# -- every export ----------------------------------------------------------------------------
exports = store.exports_dir()
for kind in ("png", "transparent", "glb", "skinned", "vrm", "obj", "bundle"):
    n_before = len(os.listdir(exports))
    click(("export", kind))
    frame(3)
    assert len(os.listdir(exports)) > n_before, f"{kind} wrote nothing: {a.message}"
    print("export", kind, "->", os.path.basename(a.last_export))
bundle = next(f for f in os.listdir(exports) if f.endswith(".zip"))
with zipfile.ZipFile(os.path.join(exports, bundle)) as z:
    names = z.namelist()
    assert z.testzip() is None
    for want in ("Sky_Walker.glb", "Sky_Walker_skinned.glb", "Sky_Walker.vrm", "Sky_Walker.obj",
                 "Sky_Walker.png", "Sky_Walker_transparent.png", "Sky_Walker.avatar", "README.txt", "code.txt"):
        assert want in names, (want, names)
    print("bundle:", len(names), "files,", sum(i.file_size for i in z.infolist()) // 1024, "KB")
clear = pygame.image.load(os.path.join(exports, next(f for f in os.listdir(exports) if f.endswith("_transparent.png"))))
assert clear.get_at((3, 3))[3] == 0, "the transparent picture should have a clear corner"

# -- talking clip ------------------------------------------------------------------------------
a.say_text = "Hello there. This clip has sound."
t0 = time.time()
click(("export", "clip"))
clip = a.last_export
assert clip and clip.endswith((".mp4", ".gif")) and os.path.getsize(clip) > 20000, (clip, a.message)
print("clip:", os.path.basename(clip), os.path.getsize(clip) // 1024, "KB in %.1fs" % (time.time() - t0))

# -- autosave + restore ----------------------------------------------------------------------------
a.commit()
a.autosave_tick(force=True)
fresh = dict(O.DEFAULT_STATE)
assert store.restore_autosave(fresh) == "Sky Walker" and O.encode_state(fresh) == O.encode_state(a.state)

# -- drop a file onto the window -----------------------------------------------------------------------
a.handle(pygame.event.Event(pygame.DROPFILE, file=os.path.join(store.avatars_dir(), "Sky_Walker.avatar")))
assert "Opened" in a.message, a.message
a.handle(pygame.event.Event(pygame.DROPFILE, file=os.path.join(home, "nothing.avatar")))
assert "Can't read" in a.message, a.message

# -- speech with the live loop -----------------------------------------------------------------------------
tab("Talk")
click(("emote", "nod"))
assert a.emote and a.emote[0] == "nod"
a.say_text = "Testing one two three."
click(("say", "speak"))
t0 = time.time()
while not a.speaking and time.time() - t0 < 20:
    frame()
    time.sleep(0.03)
assert a.speaking
key(pygame.K_ESCAPE)
assert not a.speaking
print("smoke_gui ok")
