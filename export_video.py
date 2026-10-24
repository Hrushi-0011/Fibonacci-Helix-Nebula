import time
import numpy as np
import imageio_ffmpeg
import nebula_bloom as nb          # reuse the shaders, camera and bloom pipeline

QUICK_TEST = False                 # True = 4 s test, False = 12 s preview clip

if QUICK_TEST:
    WIDTH, HEIGHT, FPS, SECONDS, OUT = 1280, 720, 30, 4, "test.mp4"
else:
    WIDTH, HEIGHT, FPS, SECONDS, OUT = 1280, 720, 30, 12, "nebula_loop.mp4"

LOOP_FRAMES = int(FPS * nb.LOOP_SECONDS)    # frames in one full seamless loop
N_FRAMES = int(FPS * SECONDS)               # frames we actually render

nb.window.set_visible(False)                # we only need its OpenGL context
fbo = nb.ctx.simple_framebuffer((WIDTH, HEIGHT), components=3)

writer = imageio_ffmpeg.write_frames(
    OUT, (WIDTH, HEIGHT), fps=FPS, quality=8, macro_block_size=1)
writer.send(None)                           # start the encoder

print(f"Rendering {N_FRAMES} frames to {OUT} ...")
t0 = time.perf_counter()
for i in range(N_FRAMES):
    nb.render_frame(i / LOOP_FRAMES, WIDTH, HEIGHT, fbo)
    raw = fbo.read(components=3)                              # bytes, bottom row first
    frame = np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH, 3)[::-1]   # flip upright
    writer.send(frame.tobytes())
    if i % 30 == 0:
        done = (i + 1) / N_FRAMES
        elapsed = time.perf_counter() - t0
        print(f"{done:6.1%}   {elapsed:5.0f}s elapsed   ~{elapsed / done - elapsed:4.0f}s left")

writer.close()
print(f"Saved {OUT} in {time.perf_counter() - t0:.0f}s")