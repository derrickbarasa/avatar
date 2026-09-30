"""Secondary motion: hair that lags behind the head and swings back (a damped spring)."""
import math


class HairSwing:
    """Sway of the hair in the head's side-to-side (x) and fore-aft (z) directions.

    The hair is pulled towards a target that opposes how fast the head is turning, nodding
    or bobbing; a spring with damping chases that target, so it lags, overshoots a little
    and settles. `swing` is (x, 0, z) in world units at the tip of the longest hair.
    """
    STIFFNESS, DAMPING, LIMIT = 70.0, 7.5, 0.22
    YAW_GAIN, ROLL_GAIN, PITCH_GAIN, BOB_GAIN = 0.0045, 0.0035, 0.0055, 0.9

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = self.z = self.vx = self.vz = 0.0
        self.prev = None

    @property
    def swing(self):
        return (self.x, 0.0, self.z)

    def update(self, dt, pose, enabled=True):
        head = pose.get("head", (0.0, 0.0, 0.0))
        torso = pose.get("torso", (0.0, 0.0, 0.0))
        cur = (head[0] + torso[0], head[1] + torso[1], head[2] + torso[2], pose.get("root_dy", 0.0))
        if not enabled:
            self.reset()
            return self.swing
        if self.prev is None or dt <= 0:
            self.prev = cur
            return self.swing
        dt = min(dt, 0.1)
        pitch_rate, yaw_rate, roll_rate, bob_rate = ((c - p) / dt for c, p in zip(cur, self.prev))
        self.prev = cur
        target_x = -(self.YAW_GAIN * yaw_rate + self.ROLL_GAIN * roll_rate)
        target_z = -(self.PITCH_GAIN * pitch_rate) - self.BOB_GAIN * bob_rate * 0.05
        steps = 4
        h = dt / steps
        for _ in range(steps):                # small sub-steps keep the spring stable at low frame rates
            self.vx += (self.STIFFNESS * (target_x - self.x) - self.DAMPING * self.vx) * h
            self.vz += (self.STIFFNESS * (target_z - self.z) - self.DAMPING * self.vz) * h
            self.x += self.vx * h
            self.z += self.vz * h
        lim = self.LIMIT
        self.x = max(-lim, min(lim, self.x))
        self.z = max(-lim, min(lim, self.z))
        if not all(math.isfinite(v) for v in (self.x, self.z)):
            self.reset()
        return self.swing
