"""Animation clips sampled from the pose system, for exporting as animated glTF/VRM.

A clip is (name, times, poses): a list of key times in seconds and a pose dict for each.
"""
from . import poses

FPS = 30
LOOP_SECONDS = {"walk": 1.6, "dance": 3.2}       # one full cycle; other poses idle for a few seconds


def _sample(name, seconds, make_pose, fps):
    count = max(2, int(round(seconds * fps)) + 1)
    times = [i / fps for i in range(count)]
    return name, times, [make_pose(t) for t in times]


def pose_clip(pose_name, label=None, fps=FPS):
    """The current pose with its idle motion (breathing, sway) or its walk / dance cycle."""
    seconds = LOOP_SECONDS.get(pose_name, 4.0)
    return _sample(label or pose_name.capitalize(), seconds, lambda t: poses.pose_at(pose_name, t, True), fps)


def emote_clip(emote, base_pose="relaxed", fps=FPS):
    """One emote (nod, shrug, clap...) played over `base_pose`."""
    label = next(lab for lab, key, _ in poses.EMOTES if key == emote)
    seconds = poses.EMOTE_DURATION[emote]

    def make(t):
        pose = poses.pose_at(base_pose, t, True)
        poses.emote_overlay(pose, emote, min(1.0, t / seconds))
        return pose

    return _sample(label, seconds, make, fps)


def all_clips(pose_name, fps=FPS):
    """The current pose plus every emote: what the animated exports carry."""
    return [pose_clip(pose_name, fps=fps)] + [emote_clip(key, fps=fps) for _, key, _ in poses.EMOTES]
