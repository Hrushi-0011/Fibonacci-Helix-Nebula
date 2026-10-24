import time
import numpy as np
import moderngl
import pyglet

# ---------- Settings ----------
SEED = 42
POINTS_PER_STRAND = 60_000
HAZE_POINTS = 20_000
LOOP_SECONDS = 24.0
BRIGHTNESS = 0.50           # lower this if the center blows out to white
BASE_SIZE = 3.0              # sprite size in pixels at 720p
GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))
CAM_DISTANCE = 17.0

VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform float time;          // loop progress, 0 -> 1
uniform float point_scale;
uniform float brightness;

in float a_t0;
in float a_disc_a;
in float a_disc_r;
in float a_noise_z;
in float a_strand;           // 0 = fire, 1 = ice, 2 = haze
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
        // haze: static cloud, drawn big and faint
        p = vec3(a_disc_r * cos(a_disc_a), a_disc_r * sin(a_disc_a), a_noise_z);
        v_color = vec3(0.42, 0.30, 1.0);
        fade = 0.12;
        size *= 5.0;
    } else {
        float t = fract(a_t0 + time);                 // drift, wraps at 1
        float breath = 1.0 + 0.06 * sin(2.0 * PI * 3.0 * time);
        float phase = a_strand * PI;                  // 0 or PI
        float theta = TURNS * 2.0 * PI * t + phase;
        float radius = (1.5 + 2.0 * t) * breath;
        float spread = 0.10 + 0.70 * pow(t, 1.5);

        p = vec3(
            radius * cos(theta) + spread * a_disc_r * cos(a_disc_a),
            radius * sin(theta) + spread * a_disc_r * sin(a_disc_a),
            (t - 0.5) * HEIGHT + spread * a_noise_z);

        v_color = palette(t, a_strand);
        fade = sqrt(sin(PI * t));                     // invisible at the ends
    }

    gl_Position = mvp * vec4(p, 1.0);
    gl_PointSize = size * point_scale / gl_Position.w;   // nearer = bigger
    v_alpha = fade * brightness;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
in float v_alpha;
out vec4 f_color;

void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;   // -1..1 across the sprite
    float d = dot(c, c);                  // squared distance from center
    if (d > 1.0) discard;                 // round, not square
    float glow = exp(-d * 4.0);           // soft gaussian falloff
    f_color = vec4(v_color * glow * v_alpha, 1.0);
}
"""


def build_particles(rng):
    """Returns float32 array, one row per particle, 6 columns:
    t0, disc_a, disc_r, noise_z, strand_id, size"""

    def strand(n, sid):
        return np.column_stack([
            rng.random(n),
            np.mod(np.arange(n) * GOLDEN_ANGLE, 2 * np.pi),  # keep angles small
            np.sqrt(rng.random(n)),
            rng.normal(0, 0.6, n),
            np.full(n, sid),
            rng.uniform(0.6, 1.6, n),
        ])

    n = HAZE_POINTS
    haze = np.column_stack([
        np.zeros(n),
        rng.random(n) * 2 * np.pi,                           # angle
        5.5 * np.sqrt(rng.random(n)) * (0.4 + rng.random(n)),  # final radius
        rng.normal(0, 12.0 / 3.5, n),                        # height
        np.full(n, 2.0),
        rng.uniform(0.6, 1.6, n),
    ])
    return np.vstack([strand(POINTS_PER_STRAND, 0.0),
                      strand(POINTS_PER_STRAND, 1.0),
                      haze]).astype("f4")


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


window = pyglet.window.Window(1280, 720, "Fibonacci Helix Nebula",
                              resizable=True, vsync=True)
ctx = moderngl.create_context()          # attaches to pyglet's OpenGL context

prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
data = build_particles(np.random.default_rng(SEED))
vbo = ctx.buffer(data.tobytes())
vao = ctx.vertex_array(prog, [
    (vbo, "1f 1f 1f 1f 1f 1f", "a_t0", "a_disc_a", "a_disc_r",
     "a_noise_z", "a_strand", "a_size")])

ctx.enable(moderngl.PROGRAM_POINT_SIZE)     # shader sets the point size
ctx.disable(moderngl.DEPTH_TEST)
ctx.enable(moderngl.BLEND)
ctx.blend_func = moderngl.ONE, moderngl.ONE   # additive blending

start = time.perf_counter()


@window.event
def on_draw():
    w, h = window.get_framebuffer_size()
    ctx.screen.use()
    ctx.viewport = (0, 0, w, h)
    ctx.clear(0.0, 0.0, 0.02)

    progress = ((time.perf_counter() - start) / LOOP_SECONDS) % 1.0

    az = 0.6 + 2 * np.pi * progress
    el = np.radians(20 + 6 * np.sin(2 * np.pi * progress))
    eye = CAM_DISTANCE * np.array([np.cos(el) * np.cos(az),
                                   np.cos(el) * np.sin(az),
                                   np.sin(el)], dtype="f4")
    view = look_at(eye, np.zeros(3, "f4"), np.array([0, 0, 1], "f4"))
    proj = perspective(45, w / h, 0.1, 100.0)
    mvp = proj @ view

    prog["mvp"].write(mvp.T.astype("f4").tobytes())
    prog["time"].value = progress
    prog["brightness"].value = BRIGHTNESS
    prog["point_scale"].value = BASE_SIZE * CAM_DISTANCE * h / 720
    vao.render(moderngl.POINTS)


print("starting loop")
pyglet.app.run(1 / 60)
print("loop ended")