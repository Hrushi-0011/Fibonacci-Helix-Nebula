import time
import numpy as np
import moderngl
import pyglet

# ---------- Settings ----------
SEED = 42
POINTS_PER_STRAND = 60_000
HAZE_POINTS = 20_000
STAR_COUNT = 8_000
LOOP_SECONDS = 24.0
BRIGHTNESS = 0.30            # nebula particle brightness
BASE_SIZE = 3.0
GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))
CAM_DISTANCE = 17.0

BLOOM_THRESHOLD = 0.50      # only light brighter than this blooms
BLOOM_STRENGTH = 0.2        # how much bloom is added back
EXPOSURE = 1.2               # overall brightness after tone mapping
STAR_BRIGHTNESS = 1.0

# ---------- Nebula shaders (same as Step 5) ----------
VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform float time;
uniform float point_scale;
uniform float brightness;

in float a_t0;
in float a_disc_a;
in float a_disc_r;
in float a_noise_z;
in float a_strand;
in float a_size;

out vec3 v_color;
out float v_alpha;

const float PI = 3.14159265;
const float TURNS = 8.0;
const float HEIGHT = 12.0;

const vec3 FIRE[5] = vec3[5](
    vec3(0.165, 0.039, 0.420), vec3(0.482, 0.173, 0.749),
    vec3(0.757, 0.208, 0.518), vec3(1.000, 0.478, 0.349),
    vec3(1.000, 0.851, 0.541));
const vec3 ICE[5] = vec3[5](
    vec3(0.039, 0.165, 0.420), vec3(0.082, 0.396, 0.753),
    vec3(0.180, 0.769, 0.839), vec3(0.604, 0.941, 0.878),
    vec3(1.000, 1.000, 1.000));

vec3 palette(float t, float strand) {
    float x = clamp(t, 0.0, 1.0) * 4.0;
    int i = int(min(floor(x), 3.0));
    float f = x - float(i);
    if (strand < 0.5) return mix(FIRE[i], FIRE[i + 1], f);
    return mix(ICE[i], ICE[i + 1], f);
}

void main() {
    vec3 p;
    float fade = 1.0;
    float size = a_size;

    if (a_strand > 1.5) {
        p = vec3(a_disc_r * cos(a_disc_a), a_disc_r * sin(a_disc_a), a_noise_z);
        v_color = vec3(0.42, 0.30, 1.0);
        fade = 0.12;
        size *= 5.0;
    } else {
        float t = fract(a_t0 + time);
        float breath = 1.0 + 0.06 * sin(2.0 * PI * 3.0 * time);
        float phase = a_strand * PI;
        float theta = TURNS * 2.0 * PI * t + phase;
        float radius = (1.5 + 2.0 * t) * breath;
        float spread = 0.10 + 0.70 * pow(t, 1.5);

        p = vec3(
            radius * cos(theta) + spread * a_disc_r * cos(a_disc_a),
            radius * sin(theta) + spread * a_disc_r * sin(a_disc_a),
            (t - 0.5) * HEIGHT + spread * a_noise_z);

        v_color = palette(t, a_strand);
        fade = sqrt(sin(PI * t));
    }

    gl_Position = mvp * vec4(p, 1.0);
    gl_PointSize = size * point_scale / gl_Position.w;
    v_alpha = fade * brightness;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
in float v_alpha;
out vec4 f_color;

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float d = dot(c, c);
    if (d > 1.0) discard;
    float glow = exp(-d * 4.0);
    f_color = vec4(v_color * glow * v_alpha, 1.0);
}
"""

# ---------- Star shaders ----------
STAR_VS = """
#version 330
uniform mat4 mvp;
uniform float progress;
uniform float star_scale;
uniform float star_brightness;

in vec3 a_pos;
in float a_size;
in float a_phase;
in float a_temp;

out vec3 v_color;
out float v_alpha;

const float PI = 3.14159265;

void main() {
    gl_Position = mvp * vec4(a_pos, 1.0);
    gl_PointSize = a_size * star_scale;
    // whole-number frequency => twinkle loops seamlessly with the animation
    float freq = 4.0 + floor(a_phase * 8.0);
    float twinkle = 0.65 + 0.35 * sin(2.0 * PI * (freq * progress + a_phase));
    v_color = mix(vec3(0.65, 0.78, 1.0), vec3(1.0, 0.82, 0.62), a_temp);
    v_alpha = star_brightness * twinkle * (a_size / 3.0);
}
"""

STAR_FS = """
#version 330
in vec3 v_color;
in float v_alpha;
out vec4 f_color;

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float d = dot(c, c);
    if (d > 1.0) discard;
    f_color = vec4(v_color * exp(-d * 3.0) * v_alpha, 1.0);
}
"""

# ---------- Post-processing shaders ----------
QUAD_VS = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

DOWN_FS = """
#version 330
uniform sampler2D src;
uniform vec2 texel;          // size of one source pixel in uv units
uniform float threshold;
in vec2 uv;
out vec4 f_color;
void main() {
    vec3 c = 0.25 * (texture(src, uv + texel * vec2(-0.5, -0.5)).rgb
                   + texture(src, uv + texel * vec2( 0.5, -0.5)).rgb
                   + texture(src, uv + texel * vec2(-0.5,  0.5)).rgb
                   + texture(src, uv + texel * vec2( 0.5,  0.5)).rgb);
    c = max(c - vec3(threshold), vec3(0.0));
    f_color = vec4(c, 1.0);
}
"""

BLUR_FS = """
#version 330
uniform sampler2D src;
uniform vec2 dir;            // (1/w, 0) for horizontal, (0, 1/h) for vertical
in vec2 uv;
out vec4 f_color;
const float W[5] = float[5](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
void main() {
    vec3 c = texture(src, uv).rgb * W[0];
    for (int i = 1; i < 5; i++) {
        c += texture(src, uv + dir * float(i)).rgb * W[i];
        c += texture(src, uv - dir * float(i)).rgb * W[i];
    }
    f_color = vec4(c, 1.0);
}
"""

COMPOSITE_FS = """
#version 330
uniform sampler2D scene;
uniform sampler2D bloom0;
uniform sampler2D bloom1;
uniform sampler2D bloom2;
uniform float bloom_strength;
uniform float exposure;
in vec2 uv;
out vec4 f_color;
void main() {
    vec3 c = texture(scene, uv).rgb;
    vec3 b = texture(bloom0, uv).rgb + texture(bloom1, uv).rgb + texture(bloom2, uv).rgb;
    c += b * bloom_strength;

    c = 1.0 - exp(-c * exposure);                 // tone mapping: squeeze HDR into 0..1

    vec2 p = uv * 2.0 - 1.0;
    c *= 1.0 - 0.25 * dot(p, p);                  // vignette: darker corners

    float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    c += (n - 0.5) / 255.0;                       // dither: hides colour banding
    f_color = vec4(c, 1.0);
}
"""


# ---------- Data and camera math ----------
def build_particles(rng):
    def strand(n, sid):
        return np.column_stack([
            rng.random(n),
            np.mod(np.arange(n) * GOLDEN_ANGLE, 2 * np.pi),
            np.sqrt(rng.random(n)),
            rng.normal(0, 0.6, n),
            np.full(n, sid),
            rng.uniform(0.6, 1.6, n),
        ])

    n = HAZE_POINTS
    haze = np.column_stack([
        np.zeros(n),
        rng.random(n) * 2 * np.pi,
        5.5 * np.sqrt(rng.random(n)) * (0.4 + rng.random(n)),
        rng.normal(0, 12.0 / 3.5, n),
        np.full(n, 2.0),
        rng.uniform(0.6, 1.6, n),
    ])
    return np.vstack([strand(POINTS_PER_STRAND, 0.0),
                      strand(POINTS_PER_STRAND, 1.0),
                      haze]).astype("f4")


def build_stars(rng):
    """Random points on a big sphere (radius 60). Columns: x y z size phase temp."""
    n = STAR_COUNT
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)       # unit vectors = random directions
    pos = d * 60.0
    size = 1.5 + 3.5 * rng.random(n) ** 3               # mostly small, a few big
    return np.column_stack([pos, size, rng.random(n), rng.random(n)]).astype("f4")


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0]], dtype="f4")


def look_at(eye, target, up):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype="f4")
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


# ---------- Window and GPU setup ----------
window = pyglet.window.Window(1280, 720, "Fibonacci Helix Nebula",
                              resizable=True, vsync=True)
ctx = moderngl.create_context()
rng = np.random.default_rng(SEED)

neb_prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
neb_vbo = ctx.buffer(build_particles(rng).tobytes())
neb_vao = ctx.vertex_array(neb_prog, [
    (neb_vbo, "1f 1f 1f 1f 1f 1f", "a_t0", "a_disc_a", "a_disc_r",
     "a_noise_z", "a_strand", "a_size")])

star_prog = ctx.program(vertex_shader=STAR_VS, fragment_shader=STAR_FS)
star_vbo = ctx.buffer(build_stars(rng).tobytes())
star_vao = ctx.vertex_array(star_prog, [
    (star_vbo, "3f 1f 1f 1f", "a_pos", "a_size", "a_phase", "a_temp")])

# a full-screen rectangle (two triangles) used by every post-processing pass
quad_buf = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())


def make_pass(fragment_src):
    prog = ctx.program(vertex_shader=QUAD_VS, fragment_shader=fragment_src)
    return prog, ctx.vertex_array(prog, [(quad_buf, "2f", "in_pos")])


down_prog, down_vao = make_pass(DOWN_FS)
blur_prog, blur_vao = make_pass(BLUR_FS)
comp_prog, comp_vao = make_pass(COMPOSITE_FS)

ctx.enable(moderngl.PROGRAM_POINT_SIZE)
ctx.disable(moderngl.DEPTH_TEST)
ctx.blend_func = moderngl.ONE, moderngl.ONE


def make_texture(size):
    tex = ctx.texture(size, 4, dtype="f2")          # 16-bit float: can hold values above 1.0 (HDR)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


class Targets:
    """Off-screen textures: the HDR scene plus 3 shrinking bloom levels."""

    def __init__(self, w, h):
        self.size = (w, h)
        self.scene = make_texture((w, h))
        self.scene_fbo = ctx.framebuffer(color_attachments=[self.scene])
        self.levels = []                             # (size, tex_a, tex_b, fbo_a, fbo_b)
        for i in range(3):
            s = (max(w >> (i + 1), 1), max(h >> (i + 1), 1))   # half, quarter, eighth
            a, b = make_texture(s), make_texture(s)
            self.levels.append((s, a, b,
                                ctx.framebuffer(color_attachments=[a]),
                                ctx.framebuffer(color_attachments=[b])))

    def release(self):
        for s, a, b, fa, fb in self.levels:
            for obj in (fa, fb, a, b):
                obj.release()
        self.scene_fbo.release()
        self.scene.release()


targets = None
start = time.perf_counter()


def render_frame(progress, w, h, output):
    """Render one complete frame at loop position `progress` (0..1)
    into the framebuffer `output` (the window or an off-screen buffer)."""
    global targets
    if targets is None or targets.size != (w, h):
        if targets is not None:
            targets.release()
        targets = Targets(w, h)
    T = targets

    # ----- camera -----
    az = 0.6 + 2 * np.pi * progress
    el = np.radians(20 + 6 * np.sin(2 * np.pi * progress))
    eye = CAM_DISTANCE * np.array([np.cos(el) * np.cos(az),
                                   np.cos(el) * np.sin(az),
                                   np.sin(el)], dtype="f4")
    view = look_at(eye, np.zeros(3, "f4"), np.array([0, 0, 1], "f4"))
    proj = perspective(45, w / h, 0.1, 150.0)
    mvp_bytes = (proj @ view).T.astype("f4").tobytes()

    # ----- pass 1: stars + nebula into the HDR texture -----
    T.scene_fbo.use()
    T.scene_fbo.clear(0.0, 0.0, 0.01)
    ctx.enable(moderngl.BLEND)

    star_prog["mvp"].write(mvp_bytes)
    star_prog["progress"].value = progress
    star_prog["star_scale"].value = h / 720
    star_prog["star_brightness"].value = STAR_BRIGHTNESS
    star_vao.render(moderngl.POINTS)

    neb_prog["mvp"].write(mvp_bytes)
    neb_prog["time"].value = progress
    neb_prog["brightness"].value = BRIGHTNESS
    neb_prog["point_scale"].value = BASE_SIZE * CAM_DISTANCE * h / 720
    neb_vao.render(moderngl.POINTS)

    ctx.disable(moderngl.BLEND)

    # ----- pass 2: bloom -----
    src = T.scene
    for i, (size, a, b, fa, fb) in enumerate(T.levels):
        fa.use()
        src.use(0)
        tw, th = src.size
        down_prog["src"].value = 0
        down_prog["texel"].value = (1.0 / tw, 1.0 / th)
        down_prog["threshold"].value = BLOOM_THRESHOLD if i == 0 else 0.0
        down_vao.render(moderngl.TRIANGLE_STRIP)

        fb.use()
        a.use(0)
        blur_prog["src"].value = 0
        blur_prog["dir"].value = (1.0 / size[0], 0.0)
        blur_vao.render(moderngl.TRIANGLE_STRIP)

        fa.use()
        b.use(0)
        blur_prog["dir"].value = (0.0, 1.0 / size[1])
        blur_vao.render(moderngl.TRIANGLE_STRIP)

        src = a

    # ----- pass 3: composite into `output` -----
    output.use()
    ctx.viewport = (0, 0, w, h)
    T.scene.use(0)
    T.levels[0][1].use(1)
    T.levels[1][1].use(2)
    T.levels[2][1].use(3)
    comp_prog["scene"].value = 0
    comp_prog["bloom0"].value = 1
    comp_prog["bloom1"].value = 2
    comp_prog["bloom2"].value = 3
    comp_prog["bloom_strength"].value = BLOOM_STRENGTH
    comp_prog["exposure"].value = EXPOSURE
    comp_vao.render(moderngl.TRIANGLE_STRIP)


def render_frame(progress, w, h, output):
    """Render one complete frame at loop position `progress` (0..1)
    into the framebuffer `output` (the window or an off-screen buffer)."""
    global targets
    if targets is None or targets.size != (w, h):
        if targets is not None:
            targets.release()
        targets = Targets(w, h)
    T = targets

    # ----- camera -----
    az = 0.6 + 2 * np.pi * progress
    el = np.radians(20 + 6 * np.sin(2 * np.pi * progress))
    eye = CAM_DISTANCE * np.array([np.cos(el) * np.cos(az),
                                   np.cos(el) * np.sin(az),
                                   np.sin(el)], dtype="f4")
    view = look_at(eye, np.zeros(3, "f4"), np.array([0, 0, 1], "f4"))
    proj = perspective(45, w / h, 0.1, 150.0)
    mvp_bytes = (proj @ view).T.astype("f4").tobytes()

    # ----- pass 1: stars + nebula into the HDR texture -----
    T.scene_fbo.use()
    T.scene_fbo.clear(0.0, 0.0, 0.01)
    ctx.enable(moderngl.BLEND)

    star_prog["mvp"].write(mvp_bytes)
    star_prog["progress"].value = progress
    star_prog["star_scale"].value = h / 720
    star_prog["star_brightness"].value = STAR_BRIGHTNESS
    star_vao.render(moderngl.POINTS)

    neb_prog["mvp"].write(mvp_bytes)
    neb_prog["time"].value = progress
    neb_prog["brightness"].value = BRIGHTNESS
    neb_prog["point_scale"].value = BASE_SIZE * CAM_DISTANCE * h / 720
    neb_vao.render(moderngl.POINTS)

    ctx.disable(moderngl.BLEND)

    # ----- pass 2: bloom -----
    src = T.scene
    for i, (size, a, b, fa, fb) in enumerate(T.levels):
        fa.use()
        src.use(0)
        tw, th = src.size
        down_prog["src"].value = 0
        down_prog["texel"].value = (1.0 / tw, 1.0 / th)
        down_prog["threshold"].value = BLOOM_THRESHOLD if i == 0 else 0.0
        down_vao.render(moderngl.TRIANGLE_STRIP)

        fb.use()
        a.use(0)
        blur_prog["src"].value = 0
        blur_prog["dir"].value = (1.0 / size[0], 0.0)
        blur_vao.render(moderngl.TRIANGLE_STRIP)

        fa.use()
        b.use(0)
        blur_prog["dir"].value = (0.0, 1.0 / size[1])
        blur_vao.render(moderngl.TRIANGLE_STRIP)

        src = a

    # ----- pass 3: composite into `output` -----
    output.use()
    ctx.viewport = (0, 0, w, h)
    T.scene.use(0)
    T.levels[0][1].use(1)
    T.levels[1][1].use(2)
    T.levels[2][1].use(3)
    comp_prog["scene"].value = 0
    comp_prog["bloom0"].value = 1
    comp_prog["bloom1"].value = 2
    comp_prog["bloom2"].value = 3
    comp_prog["bloom_strength"].value = BLOOM_STRENGTH
    comp_prog["exposure"].value = EXPOSURE
    comp_vao.render(moderngl.TRIANGLE_STRIP)


@window.event
def on_draw():
    w, h = window.get_framebuffer_size()
    if w < 16 or h < 16:                     # minimised window
        return
    progress = ((time.perf_counter() - start) / LOOP_SECONDS) % 1.0
    render_frame(progress, w, h, ctx.screen)


if __name__ == "__main__":                   # only run the window when launched directly
    pyglet.app.run(1 / 60)