"""Frame-time benchmark (needs a display and OpenGL): not collected by unittest.

    python tests/bench_gui.py

Draws the busiest avatar (long hair, beard, walking) with each quality setting and prints the
average milliseconds per frame, GPU work included.
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["AVATARKIT_HOME"] = tempfile.mkdtemp()
os.chdir(tempfile.mkdtemp())

import pygame  # noqa: E402
from OpenGL.GL import glFinish  # noqa: E402

from avatarkit import app, options as O  # noqa: E402

a = app.App(app.parse_args(["--size", "1280x800", "--fresh", "--set", "hair=Long", "--set", "facial=Beard",
                            "--set", "pose=Walk"]))


def measure(frames=25, rounds=4):
    """Best of several rounds: other programs on the machine only ever make a round slower."""
    for _ in range(25):
        a.draw()
        pygame.display.flip()
    best = float("inf")
    for _ in range(rounds):
        glFinish()
        start = time.perf_counter()
        for _ in range(frames):
            a.draw()
            pygame.display.flip()
        glFinish()
        best = min(best, (time.perf_counter() - start) / frames * 1000)
    return best


for view in ("bust", "full"):
    a.cam.set_view(view)
    for _ in range(80):
        a.cam.update(a.rig.ground_y)
    for name, opts in (("High", {"shadows": "On", "quality": "High"}), ("Balanced", {"quality": "Balanced"}),
                       ("Fast", {"quality": "Fast"}), ("shadows off", {"shadows": "Off", "quality": "Fast"})):
        for key, val in opts.items():
            O.set_by_name(a.state, key, val)
        print(f"{view:5s} {name:14s} {measure():6.1f} ms")
pygame.quit()
