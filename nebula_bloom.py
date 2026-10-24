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

BLOOM_THRESHOLD = 0.35       # only light brighter than this blooms
BLOOM_STRENGTH = 0.55        # how much bloom is added back
EXPOSURE = 1.2               # overall brightness after tone mapping
STAR_BRIGHTNESS = 0.9

PUSH_IN = 0.18               # camera moves 18% closer at the closest point
DOF_STRENGTH = 1.5           # 0 = everything sharp, higher = blurrier off-focus
DUST = 0.12                  # size of the tiny drifting wobble

TOUCH_RADIUS = 0.22          # size of the disturbed area (screen height = 2.0)
TOUCH_STRENGTH = 0.30        # how far particles get thrown
TOUCH_LIFETIME = 4.0         # seconds before a touch is forgotten
MAX_TOUCHES = 8              # must match the array size in the shader

# ---------- Nebula shaders ----------
VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform float time;
uniform float point_scale;
uniform float brightness;
uniform float focus;         // distance to the sharp plane
uniform float dof;           // depth-of-field strength
uniform float dust;          // wobble size
uniform vec4 touches[8];     // per touch: x, y (screen, -1..1), age in seconds (-1 = unused)
uniform float aspect;        // window width / height
uniform float touch_radius;
uniform float touch_strength;

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
const float TOUCH_PEAK = 0.30;   // seconds until a touch reaches maximum scatter

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

    // dust: small wobble, whole-number frequencies so it loops seamlessly
    float ph = a_disc_a * 3.0 + a_t0 * 40.0;
    p += dust * vec3(sin(2.0 * PI * 2.0 * time + ph),
                     sin(2.0 * PI * 3.0 * time + ph * 1.3),
                     sin(2.0 * PI * 2.0 * time + ph * 0.7));

    gl_Position = mvp * vec4(p, 1.0);

    // touch: particles near a click are pushed away on screen, then settle back
    vec2 ndc = gl_Position.xy / gl_Position.w;
    float ra = a_disc_a * 7.0 + a_t0 * 50.0;
    vec2 rnd = vec2(cos(ra), sin(ra));           // each particle's own scatter direction
    vec2 push = vec2(0.0);
    float lit = 0.0;
    for (int k = 0; k < 8; k++) {
        float age = touches[k].z;
        if (age < 0.0) continue;                 // unused slot
        vec2 d = (ndc - touches[k].xy) * vec2(aspect, 1.0);
        float dist2 = dot(d, d);
        float influence = exp(-dist2 / (touch_radius * touch_radius));   // 1 at the click, 0 far away
        float x = age / TOUCH_PEAK;
        float env = x * exp(1.0 - x);            // 0 -> 1 at the peak -> back to 0
        vec2 radial = d / (sqrt(dist2) + 0.0001);
        push += (radial + 0.8 * rnd) * influence * env;
        lit += influence * env;
    }
    push.x /= aspect;                            // back from aspect-corrected units
    gl_Position.xy += push * touch_strength * gl_Position.w;

    // depth of field: far from the focal plane = bigger, dimmer dot
    float coc = abs(gl_Position.w - focus) / max(focus, 0.001);
    float blur = 1.0 + dof * coc;
    gl_PointSize = size * blur * (1.0 + 0.6 * lit) * point_scale / gl_Position.w;
    v_alpha = fade * brightness * (1.0 + 1.5 * lit) / (blur * blur);
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
uniform vec2 texel;
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
uniform vec2 dir;
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

    c = 1.0 - exp(-c * exposure);                 // tone mapping

    // colour grade: richer colour, cool shadows, warm highlights
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(luma), c, 1.15);
    c = mix(c * vec3(0.92, 0.97, 1.08),
            c * vec3(1.08, 1.00, 0.92),
            smoothstep(0.2, 0.8, luma));

    vec2 p = uv * 2.0 - 1.0;
    c *= 1.0 - 0.25 * dot(p, p);                  // vignette

    float n = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    c += (n - 0.5) / 255.0;                       // dither
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
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pos = d * 60.0
    size = 1.5 + 3.5 * rng.random(n) ** 3
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
    tex = ctx.texture(size, 4, dtype="f2")
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
        self.levels = []
        for i in range(3):
            s = (max(w >> (i + 1), 1), max(h >> (i + 1), 1))
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


def eye_from_angles(az, el, dist):
    """Camera position on a sphere around the origin (z is up)."""
    return (dist * np.array([np.cos(el) * np.cos(az),
                             np.cos(el) * np.sin(az),
                             np.sin(el)])).astype("f4")


def auto_eye(progress):
    """The cinematic camera: slow orbit, gentle sway, push-in."""
    az = 0.6 + 2 * np.pi * progress
    el = np.radians(20 + 6 * np.sin(2 * np.pi * progress))
    dist = CAM_DISTANCE * (1 - PUSH_IN * 0.5 * (1 - np.cos(2 * np.pi * progress)))
    return eye_from_angles(az, el, dist)


def render_frame(progress, w, h, output, eye=None, touches=None):
    """Render one complete frame at loop position `progress` (0..1)
    into the framebuffer `output` (the window or an off-screen buffer).
    `eye` overrides the cinematic camera; `touches` is a (MAX_TOUCHES, 4) array."""
    global targets
    if targets is None or targets.size != (w, h):
        if targets is not None:
            targets.release()
        targets = Targets(w, h)
    T = targets

    # ----- camera -----
    if eye is None:
        eye = auto_eye(progress)
    dist = float(np.linalg.norm(eye))                # also the depth-of-field focus
    if touches is None:
        touches = np.full((MAX_TOUCHES, 4), -1.0, "f4")   # all slots unused
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
    neb_prog["focus"].value = float(dist)
    neb_prog["dof"].value = DOF_STRENGTH
    neb_prog["dust"].value = DUST
    neb_prog["touches"].write(np.asarray(touches, "f4").tobytes())
    neb_prog["aspect"].value = w / h
    neb_prog["touch_radius"].value = TOUCH_RADIUS
    neb_prog["touch_strength"].value = TOUCH_STRENGTH
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


# ---------- Interaction (mouse + keyboard) ----------
HELP = """
Controls
  click ................ scatter the particles around the cursor
  double-click ......... toggle FREE VIEW (cinematic camera pauses)
  drag (free view) ..... orbit to any angle
  scroll (free view) ... zoom in / out
  1 top   2 bottom   3 front   4 side   5 diagonal
  R .................... back to the cinematic camera
  Esc .................. quit
"""
EL_LIMIT = np.radians(89)        # stop just short of the poles (look_at breaks at exactly 90)
DOUBLE_CLICK_SECONDS = 0.40
CLICK_MAX_DRAG = 6               # pixels: more than this counts as a drag, not a click


class Interaction:
    """Everything the user can change: camera mode, camera angles, active touches."""

    def __init__(self):
        self.touches = []                    # (screen_x, screen_y, start_time)
        self.free_mode = False
        self.blend = 0.0                     # 0 = cinematic camera, 1 = free camera
        self.az, self.el, self.dist = 0.6, np.radians(20), CAM_DISTANCE
        self.goal = None                     # (az, el) a preset view is gliding toward
        self.last_eye = auto_eye(0.0)
        self.last_frame = start
        self.moved = 0.0                     # pixels dragged since the button went down
        self.last_click_time = -10.0
        self.last_click_xy = (0, 0)

    # --- camera mode ---
    def enter_free(self):
        if self.free_mode:
            return
        e = self.last_eye                    # start exactly where the camera is now
        self.dist = float(np.linalg.norm(e))
        self.el = float(np.clip(np.arcsin(e[2] / self.dist), -EL_LIMIT, EL_LIMIT))
        self.az = float(np.arctan2(e[1], e[0]))
        self.blend = 1.0
        self.free_mode = True

    def leave_free(self):
        self.free_mode = False
        self.goal = None

    def camera(self, progress, dt):
        """Camera position for this frame."""
        if self.goal is not None:            # glide toward a preset view
            k = min(1.0, dt * 5.0)
            self.az += ((self.goal[0] - self.az + np.pi) % (2 * np.pi) - np.pi) * k
            self.el += (self.goal[1] - self.el) * k
        target = 1.0 if self.free_mode else 0.0
        self.blend += float(np.clip(target - self.blend, -dt, dt))   # 1-second transition
        s = self.blend * self.blend * (3 - 2 * self.blend)           # smoothstep
        eye_a = auto_eye(progress)
        if s <= 0.0:
            return eye_a
        eye_f = eye_from_angles(self.az, self.el, self.dist)
        if s >= 1.0:
            return eye_f
        da, df = np.linalg.norm(eye_a), np.linalg.norm(eye_f)
        d = eye_a / da * (1 - s) + eye_f / df * s                    # blend directions...
        d /= np.linalg.norm(d)
        return (d * (da * (1 - s) + df * s)).astype("f4")            # ...and distances

    # --- touches ---
    def add_touch(self, x, y, now):
        nx = x / window.width * 2 - 1        # pixels -> -1..1 screen coordinates
        ny = y / window.height * 2 - 1
        self.touches.append((nx, ny, now))
        self.touches = self.touches[-MAX_TOUCHES:]


ui = Interaction()


@window.event
def on_mouse_press(x, y, button, modifiers):
    ui.moved = 0.0


@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    ui.moved += abs(dx) + abs(dy)
    if ui.free_mode and (buttons & pyglet.window.mouse.LEFT):
        ui.goal = None                       # the user takes over from any preset glide
        ui.az -= dx * 0.006
        ui.el = float(np.clip(ui.el - dy * 0.006, -EL_LIMIT, EL_LIMIT))


@window.event
def on_mouse_release(x, y, button, modifiers):
    if button != pyglet.window.mouse.LEFT or ui.moved > CLICK_MAX_DRAG:
        return                               # it was a drag, not a click
    now = time.perf_counter()
    near = abs(x - ui.last_click_xy[0]) + abs(y - ui.last_click_xy[1]) < 12
    if now - ui.last_click_time < DOUBLE_CLICK_SECONDS and near:
        if ui.free_mode:
            ui.leave_free()
        else:
            ui.enter_free()
        ui.last_click_time = -10.0           # so a third click doesn't toggle again
    else:
        ui.add_touch(x, y, now)
        ui.last_click_time = now
        ui.last_click_xy = (x, y)


@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    if ui.free_mode:
        ui.dist = float(np.clip(ui.dist * 0.9 ** scroll_y, 5.0, 45.0))


@window.event
def on_key_press(symbol, modifiers):
    key = pyglet.window.key
    if symbol == key.ESCAPE:                 # defining on_key_press replaces pyglet's default Esc
        window.close()
        return
    if symbol == key.R:
        ui.leave_free()
        return
    if symbol in (key._1, key._2, key._3, key._4, key._5):
        ui.enter_free()
        presets = {key._1: (ui.az, EL_LIMIT),         # top: looking straight down the helix
                   key._2: (ui.az, -EL_LIMIT),        # bottom
                   key._3: (-np.pi / 2, 0.0),         # front
                   key._4: (0.0, 0.0),                # side
                   key._5: (0.6, 0.35)}               # diagonal
        ui.goal = presets[symbol]


@window.event
def on_draw():
    w, h = window.get_framebuffer_size()
    if w < 16 or h < 16:                     # minimised window
        return
    now = time.perf_counter()
    dt = min(now - ui.last_frame, 0.1)       # cap so a stall doesn't make the camera jump
    ui.last_frame = now
    progress = ((now - start) / LOOP_SECONDS) % 1.0

    eye = ui.camera(progress, dt)
    ui.last_eye = eye

    ui.touches = [t for t in ui.touches if now - t[2] < TOUCH_LIFETIME]
    touch_data = np.full((MAX_TOUCHES, 4), -1.0, "f4")
    for i, (tx, ty, t0) in enumerate(ui.touches):
        touch_data[i] = (tx, ty, now - t0, 0.0)          # third value = age in seconds

    render_frame(progress, w, h, ctx.screen, eye=eye, touches=touch_data)


if __name__ == "__main__":                   # only run the window when launched directly
    window.set_caption("Fibonacci Helix Nebula  |  click: scatter  |  double-click: free view")
    print(HELP)
    pyglet.app.run(1 / 60)