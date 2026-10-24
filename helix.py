import numpy as np
import matplotlib.pyplot as plt

SEED = 42
POINTS_PER_STRAND = 10000
HAZE_POINTS = 3000
TURNS = 21
HEIGHT = 12.0
GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))

#Reproducible, random number generator
rng = np.random.default_rng(SEED)

def make_strand(n, phase):
    """ One helix strand as an array of shape (n, 4): x, y, z, t."""
    t = np.sort(rng.random(n)) 
    theta = TURNS * 2 * np.pi * t + phase
    radius = 1.5 + 2.0 * t

    #centerline of the helix
    cx = radius * np.cos(theta)
    cy = radius * np.sin(theta)
    cz = (t - 0.5) * HEIGHT

    #cloud around the centerline
    spread = 0.05 + 1.5 * t**2
    i = np.arange(n)
    disc_r = spread * np.sqrt(rng.random(n))
    disc_a = i * GOLDEN_ANGLE
    x = cx + disc_r * np.cos(disc_a)
    y = cy + disc_r * np.sin(disc_a)
    z = cz + rng.normal(0, spread * 0.6, n)

    return np.column_stack([x, y, z, t])


def make_haze(n):
    """Diffuze background cloud around the whole helix."""
    t = rng.random(n)
    r = 5.5 * np.sqrt(rng.random(n)) * (0.4 + t)
    a = rng.random(n) * 2 * np.pi
    z = rng.normal(0, HEIGHT / 3.5, n)
    x = r * np.cos(a)
    y = r * np.sin(a)
    return np.column_stack([x, y, z, t, r, a])


strand_a = make_strand(POINTS_PER_STRAND, phase=0)
strand_b = make_strand(POINTS_PER_STRAND, phase=np.pi * 2/3)
haze = make_haze(HAZE_POINTS)

#plot
fig = plt.figure(figsize=(9, 9), facecolor="black")
ax = fig.add_subplot(111, projection="3d", facecolor="black")

ax.scatter(*strand_a[:, :3].T, c=strand_a[:, 3], s=3, cmap="plasma", alpha=0.55, linewidth=0)
ax.scatter(*strand_b[:, :3].T, c=strand_b[:, 3], s=3, cmap="cool", alpha=0.55, linewidth=0)
ax.scatter(*haze[:, :3].T, c="#6a4cff", s=1, alpha=0.12, linewidth=0)

ax.set_box_aspect((1, 1, 1.3))
ax.view_init(elev=20, azim=35)
ax.set_axis_off()
plt.tight_layout()
plt.show()

