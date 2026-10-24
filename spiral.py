import numpy as np
import matplotlib.pyplot as plt

N = 2000
golden_angle = np.pi * (3 - np.sqrt(5))

i = np.arange(N)
r = np.sqrt(i)
theta = i * golden_angle

x = r * np.cos(theta)
y = r * np.sin(theta)

fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
ax.set_facecolor("black")
ax.scatter(x, y, s=4, c=i, cmap="plasma")   
ax.set_aspect("equal")
ax.axis("off")
plt.show()
