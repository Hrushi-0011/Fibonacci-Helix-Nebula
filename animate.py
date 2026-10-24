import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# ---------- Settings ----------
SEED = 42
POINTS_PER_STRAND = 2500
HAZE_POINTS = 1500
TURNS = 8
HEIGHT = 12.0
GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))

FRAMES = 720
BREATH_AMP = 0.06
BREATH_CYCLES = 3

CORE_ALPHA = 0.7        # brightness of the sharp points
GLOW_ALPHA = 0.10       # brightness of the soft halo layer
GLOW_SIZE = 120         # halo size (core size is 3)

rng = np.random.default_rng(SEED)

# ---------- Palettes ----------
FIRE = LinearSegmentedColormap.from_list(
    "fire", ["#2a0a6b", "#7b2cbf", "#c13584", "#ff7a59", "#ffd98a"])
ICE = LinearSegmentedColormap.from_list(
    "ice", ["#0a2a6b", "#1565c0", "#2ec4d6", "#9af0e0", "#ffffff"])


class Strand:
    def __init__(self, n, phase, cmap):
        self.phase = phase
        self.cmap = cmap
        self.t0 = rng.random(n)
        self.disc_a = np.arange(n) * GOLDEN_ANGLE
        self.disc_r = np.sqrt(rng.random(n))
        self.noise_z = rng.normal(0, 0.6, n)

    def positions(self, frame):
        t = (self.t0 + frame / FRAMES) % 1.0
        breath = 1 + BREATH_AMP * np.sin(2 * np.pi * BREATH_CYCLES * frame / FRAMES)
        theta = TURNS * 2 * np.pi * t + self.phase
        radius = (1.5 + 2.0 * t) * breath
        spread = 0.10 + 0.70 * t**1.5

        x = radius * np.cos(theta) + spread * self.disc_r * np.cos(self.disc_a)
        y = radius * np.sin(theta) + spread * self.disc_r * np.sin(self.disc_a)
        z = (t - 0.5) * HEIGHT + spread * self.noise_z
        return x, y, z, t

    def colors(self, t, alpha):
        """RGBA per particle: color from palette, opacity fades at both ends."""
        rgba = self.cmap(t)                         # shape (n, 4)
        fade = np.sin(np.pi * t) ** 0.5             # 0 at t=0 and t=1, 1 in the middle
        rgba[:, 3] = alpha * fade
        return rgba


def make_haze(n):
    r = 5.5 * np.sqrt(rng.random(n)) * (0.4 + rng.random(n))
    a = rng.random(n) * 2 * np.pi
    z = rng.normal(0, HEIGHT / 3.5, n)
    return r * np.cos(a), r * np.sin(a), z


# ---------- Build the scene ----------
strands = [Strand(POINTS_PER_STRAND, 0, FIRE),
           Strand(POINTS_PER_STRAND, np.pi, ICE)]

fig = plt.figure(figsize=(9, 9), facecolor="black")
ax = fig.add_subplot(111, projection="3d", facecolor="black")
ax.set_box_aspect((1, 1, 1.3))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-7, 7)
ax.set_axis_off()
plt.tight_layout()

hx, hy, hz = make_haze(HAZE_POINTS)
ax.scatter(hx, hy, hz, c="#6a4cff", s=1, alpha=0.12, linewidths=0)

layers = []   # (strand, glow_scatter, core_scatter)
for strand in strands:
    x, y, z, t = strand.positions(0)
    glow = ax.scatter(x, y, z, color=strand.colors(t, GLOW_ALPHA),
                      s=GLOW_SIZE, linewidths=0)     # drawn first = underneath
    core = ax.scatter(x, y, z, color=strand.colors(t, CORE_ALPHA),
                      s=3, linewidths=0)             # drawn second = on top
    layers.append((strand, glow, core))


# ---------- Animation ----------
def update(frame):
    for strand, glow, core in layers:
        x, y, z, t = strand.positions(frame)
        glow._offsets3d = (x, y, z)
        core._offsets3d = (x, y, z)
        glow.set_facecolor(strand.colors(t, GLOW_ALPHA))
        core.set_facecolor(strand.colors(t, CORE_ALPHA))
    progress = frame / FRAMES
    ax.view_init(elev=20 + 6 * np.sin(2 * np.pi * progress),
                 azim=35 + 360 * progress)
    return []


anim = FuncAnimation(fig, update, frames=FRAMES, interval=33, blit=False)
plt.show()