"""OpenGL rendering: cel-shaded materials with procedural detail, cast shadows, inked
outlines, skeleton drawing, level-of-detail for hair, camera and screenshots."""
import math

import numpy as np
import pygame
from OpenGL.GL import *  # noqa: F401,F403
from OpenGL.GL import shaders
from OpenGL.GLU import gluPerspective

KINDS = {"plain": 0.0, "skin": 1.0, "hair": 2.0, "cloth": 3.0, "denim": 4.0, "eye": 5.0,
         "glossy": 6.0}
FOV = 35.0
KEY_LIGHT = np.array([-0.4, 0.7, 1.0])          # eye space: the key light travels with the camera
SHADOW_SIZE = 2048
SHADOW_STRAND_LOD = 0.35          # fraction of hair strands that cast shadows

# Vertex offsets shared by the main, outline and shadow passes so they always agree:
# strands thicken when fewer are drawn, sway with the head, and the jaw drops when talking.
DISPLACE = """
uniform float uStrandScale;
uniform vec3 uSwing;          // hair sway (side, unused, fore-aft)
uniform float uSwingScale;
uniform float uJaw;           // how far the lower face drops (world units), face meshes only
uniform float uJawShift;      // how far a raised head has moved the jaw region
vec4 displaced(vec4 v) {
    vec2 aux = gl_MultiTexCoord1.xy;   // strands: (radius, distance along the strand 0..1)
    v.xyz += gl_Normal * (aux.x * (uStrandScale - 1.0));
    v.xyz += vec3(uSwing.x, -abs(uSwing.x) * 0.25, uSwing.z) * (aux.y * aux.y * uSwingScale);
    float jaw = smoothstep(-0.33, -0.46, v.y - uJawShift) * smoothstep(-0.15, 0.2, v.z);
    v.y -= uJaw * jaw;
    v.z -= uJaw * jaw * 0.25;
    return v;
}
"""

VERTEX = """
#version 120
uniform mat4 uLightVP;
uniform mat4 uAoVP;
uniform mat4 uInvView;
varying vec4 vAO;
""" + DISPLACE + """
centroid varying vec3 vN;
centroid varying vec3 vP;
centroid varying vec3 vObj;
centroid varying vec3 vT;
centroid varying vec4 vColor;
varying vec4 vLight;
void main() {
    vec4 pos = displaced(gl_Vertex);
    vec4 ep = gl_ModelViewMatrix * pos;
    vLight = uLightVP * (uInvView * ep);       // this point as seen from the key light
    vAO = uAoVP * (uInvView * ep);             // ... and from straight above (sky light)
    vP = ep.xyz;
    vN = gl_NormalMatrix * gl_Normal;
    vT = gl_NormalMatrix * gl_MultiTexCoord0.xyz;   // strand direction (0,1,0 for other meshes)
    vObj = gl_Vertex.xyz;
    vColor = gl_Color;
    gl_Position = gl_ProjectionMatrix * ep;
}
"""

SHADOW_LOOKUP = """
uniform sampler2DShadow uShadow;   // depth texture with hardware compare: every tap is already a 2x2 filter
uniform float uShadowOn;
uniform float uShadowTexel;
varying vec4 vLight;
float shadowAmount(float ndl, float biasScale, float radius) {
    vec3 p = vLight.xyz / vLight.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    float bias = (0.0007 + 0.0032 * (1.0 - clamp(ndl, 0.0, 1.0))) * biasScale;
    float lit = 0.0, wsum = 0.0;
    for (int i = -1; i <= 1; i++) {                                  // radius 1 = 3x3 taps, 0 = a single tap
        for (int j = -1; j <= 1; j++) {
            if (abs(float(i)) > radius || abs(float(j)) > radius) continue;
            float w = 1.0 - 0.25 * float(i * i + j * j);           // soft, round kernel: no banding
            lit += w * shadow2D(uShadow, vec3(p.xy + vec2(float(i), float(j)) * uShadowTexel * 1.5,
                                              p.z - bias)).r;
            wsum += w;
        }
    }
    return lit / wsum;
}
"""

FRAGMENT = """
#version 120
centroid varying vec3 vN;
centroid varying vec3 vP;
centroid varying vec3 vObj;
centroid varying vec3 vT;
centroid varying vec4 vColor;
uniform float uKind;
uniform float uSpec;
uniform float uShine;
uniform vec3 uColor2;
uniform float uPattern;
uniform float uFreckle;
uniform vec3 uCenter;
uniform float uStrand;
uniform mat4 uInvView;
uniform sampler2DShadow uAoMap;
uniform float uAoOn;
uniform float uAoTexel;
uniform float uTaps;          // shadow filter radius: 1 = 3x3 taps, 0 = one
varying vec4 vAO;
""" + SHADOW_LOOKUP + """
// Ambient occlusion: how much of the sky is open above this point (a wide, soft shadow from overhead).
float skyOpen(vec3 nEye, float biasScale, float radius) {
    vec3 p = vAO.xyz / vAO.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    vec3 nw = mat3(uInvView) * nEye;
    float bias = (0.0025 + 0.010 * (1.0 - clamp(nw.y, 0.0, 1.0))) * biasScale;
    float lit = 0.0, wsum = 0.0;
    float spread = radius < 1.5 ? 6.0 : 3.0;                         // fewer taps are spread wider
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            if (abs(float(i)) > radius || abs(float(j)) > radius) continue;
            float w = max(1.0 - 0.09 * float(i * i + j * j), 0.0);
            lit += w * shadow2D(uAoMap, vec3(p.xy + vec2(float(i), float(j)) * uAoTexel * spread,
                                             p.z - bias)).r;
            wsum += w;
        }
    }
    return lit / wsum;
}

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float hash3(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float disc(float r, float radius) { return 1.0 - smoothstep(radius - 0.012, radius, r); }

void main() {
    vec3 N = normalize(vN);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(-vP);
    vec3 base = vColor.rgb;
    float kind = uKind;
    bool isSkin = kind > 0.5 && kind < 1.5;
    bool isHair = kind > 1.5 && kind < 2.5;
    bool isCloth = kind > 2.5 && kind < 4.5;
    bool isEye = kind > 4.5 && kind < 5.5;
    bool isGlossy = kind > 5.5;
    float spec = uSpec;

    // ---- procedural surface detail --------------------------------------
    if (isEye) {
        vec3 p = normalize(vObj - uCenter);
        float ang = acos(clamp(p.z, -1.0, 1.0));
        float a = atan(p.y, p.x);
        float irisR = 0.54;
        float pupilR = 0.23;
        vec3 col = vec3(0.96, 0.95, 0.94) * (1.0 - 0.12 * smoothstep(0.5, 1.4, ang));
        col += vec3(0.10, 0.0, 0.0) * 0.4 * smoothstep(0.9, 1.5, ang);
        float streak = vnoise(vec2(a * 7.0, ang * 9.0));
        vec3 iris = uColor2 * (0.70 + 0.65 * streak);
        iris = mix(iris * 1.35, iris * 0.75, smoothstep(0.15, 0.50, ang));
        iris = mix(iris, uColor2 * 0.22, smoothstep(irisR - 0.10, irisR, ang));
        col = mix(col, iris, 1.0 - smoothstep(irisR - 0.015, irisR, ang));
        col = mix(col, vec3(0.02), 1.0 - smoothstep(pupilR - 0.015, pupilR, ang));
        col = mix(col, vec3(1.0), smoothstep(0.972, 0.984, dot(p, normalize(vec3(0.30, 0.38, 0.90)))));
        base = col;
    }
    if (isSkin && uFreckle > 0.0) {
        float mask = (1.0 - smoothstep(0.24, 0.34, abs(vObj.x)))
                   * smoothstep(-0.28, -0.18, vObj.y) * (1.0 - smoothstep(0.02, 0.12, vObj.y))
                   * step(0.25, vObj.z);
        vec3 q = vObj * 52.0;
        vec3 id = floor(q);
        vec3 jitter = (vec3(hash3(id + 1.7), hash3(id + 3.1), hash3(id + 5.3)) - 0.5) * 0.5;
        float d = length(fract(q) - 0.5 - jitter);
        float dots = step(1.0 - 0.30 * uFreckle, hash3(id)) * (1.0 - smoothstep(0.16, 0.26, d));
        base = mix(base, base * vec3(0.62, 0.45, 0.35), dots * mask * 0.9);
    }
    if (isCloth) {
        if (kind > 3.5) {
            base *= 0.97 + 0.06 * vnoise(vec2((vObj.x + vObj.z) * 40.0, vObj.y * 20.0));
        }
        // soft fabric folds: bend the normal along slowly wandering horizontal creases
        float fold = sin(vObj.y * 22.0 + 2.2 * sin(vObj.x * 7.0 + vObj.z * 6.0));
        N = normalize(N + normalize(vT) * fold * 0.018);
        base *= 1.0 - 0.025 * max(-fold, 0.0);
        if (uPattern > 0.5) {
            float th = atan(vObj.x, vObj.z);
            float pat = 0.0;
            if (uPattern < 1.5) {
                pat = step(0.5, fract(vObj.y * 7.0));
            } else if (uPattern < 2.5) {
                vec2 g = vec2(th * 5.5, vObj.y * 7.0);
                pat = 1.0 - smoothstep(0.20, 0.26, length(fract(g) - 0.5));
            } else if (uPattern < 3.5) {
                pat = max(step(0.72, fract(vObj.y * 6.0)), step(0.72, fract(th * 2.6)));
            } else {
                float r = length(vObj.xy - vec2(0.0, -1.5));
                pat = (disc(r, 0.26) * (1.0 - disc(r, 0.19)) + disc(r, 0.10)) * step(0.0, vObj.z);
            }
            base = mix(base, uColor2, pat);
        }
    }
    if (isHair && uStrand < 0.5) {
        float th = atan(vObj.x, vObj.z);
        float strand = hash(vec2(floor(th * 64.0), 3.0));
        base *= 0.80 + 0.30 * strand * 0.6 + 0.35 * vnoise(vec2(th * 30.0, vObj.y * 4.0)) * 0.6;
    }

    // ---- lighting: soft toon bands, tinted shadows, fill and rim ---------
    vec3 L0 = normalize(gl_LightSource[0].position.xyz);
    vec3 L1 = normalize(gl_LightSource[1].position.xyz);
    vec3 L2 = normalize(gl_LightSource[2].position.xyz);
    float ndl = dot(N, L0);
    // cast shadows from the key light; thin strands need a bigger bias or they shadow themselves
    float sh = uShadowOn > 0.5 ? shadowAmount(ndl, uStrand > 0.5 ? 4.0 : 1.0, uStrand > 0.5 ? 0.0 : uTaps) : 1.0;
    float lit = (isSkin ? smoothstep(-0.50, 0.65, ndl) : smoothstep(-0.15, 0.35, ndl)) * sh;
    float hi = (isSkin ? smoothstep(0.10, 1.00, ndl) * 0.7 : smoothstep(0.25, 0.95, ndl)) * sh;
    vec3 shadowTint = isSkin ? vec3(0.80, 0.58, 0.60) : vec3(0.62, 0.60, 0.78);
    vec3 c = base * mix(shadowTint * 0.72, vec3(0.92), lit);
    c = mix(c, base * vec3(1.0, 0.97, 0.92) * 1.05, hi);
    c += base * vec3(0.32, 0.38, 0.50) * 0.22 * max(dot(N, L1), 0.0);
    c += base * mix(vec3(0.05, 0.04, 0.07), vec3(0.10, 0.11, 0.14), N.y * 0.5 + 0.5);
    if (isSkin) {
        c += base * vec3(0.30, 0.06, 0.02) * smoothstep(0.35, 0.0, abs(ndl - 0.08)) * 0.5 * mix(0.4, 1.0, sh);
    }
    if (uAoOn > 0.5) {
        float open = skyOpen(N, uStrand > 0.5 ? 3.0 : 1.0, uStrand > 0.5 ? 0.0 : 2.0);
        c *= mix(uStrand > 0.5 ? 0.88 : 0.76, 1.0, open);
    }
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    c += vec3(0.30, 0.36, 0.55) * rim * smoothstep(-0.3, 0.5, dot(N, L2)) * 0.5;

    vec3 H = normalize(L0 + V);
    float nh = max(dot(N, H), 0.0);
    if (isHair) {
        vec3 T = normalize(vT);
        float th = dot(T, H);
        float st = sqrt(max(1.0 - th * th, 0.0));
        float kk = (pow(st, 60.0) * 0.6 + pow(st, 14.0) * 0.18) * smoothstep(0.0, 0.3, ndl) * sh;
        c += mix(base, vec3(1.0), 0.55) * kk * 0.55;
    } else if (isGlossy || isEye) {
        c += vec3(1.0) * smoothstep(0.55, 0.70, pow(nh, uShine * 0.5)) * spec * (isEye ? 0.8 : 1.4) * sh;
    } else {
        c += vec3(1.0) * smoothstep(0.45, 0.55, pow(nh, uShine)) * spec * 1.6 * sh;
    }
    gl_FragColor = vec4(c, vColor.a);
}
"""

OUTLINE_VERTEX = """
#version 120
uniform float uPx;
uniform float uViewH;
""" + DISPLACE + """
void main() {
    vec4 ep = gl_ModelViewMatrix * displaced(gl_Vertex);
    vec3 n = normalize(gl_NormalMatrix * gl_Normal);
    float wpp = abs(ep.z) * 2.0 / (gl_ProjectionMatrix[1][1] * uViewH);
    ep.xyz += n * (uPx * wpp);
    gl_Position = gl_ProjectionMatrix * ep;
}
"""

OUTLINE_FRAGMENT = """
#version 120
uniform vec3 uOutline;
void main() { gl_FragColor = vec4(uOutline, 1.0); }
"""

DEPTH_VERTEX = """
#version 120
""" + DISPLACE + """
void main() { gl_Position = gl_ModelViewProjectionMatrix * displaced(gl_Vertex); }
"""

DEPTH_FRAGMENT = """
#version 120
void main() { gl_FragColor = vec4(1.0); }
"""

GROUND_VERTEX = """
#version 120
uniform mat4 uLightVP;
uniform mat4 uInvView;
varying vec4 vLight;
varying vec3 vObj;
void main() {
    vec4 ep = gl_ModelViewMatrix * gl_Vertex;
    vLight = uLightVP * (uInvView * ep);
    vObj = gl_Vertex.xyz;
    gl_Position = gl_ProjectionMatrix * ep;
}
"""

GROUND_FRAGMENT = """
#version 120
varying vec3 vObj;
""" + SHADOW_LOOKUP + """
void main() {
    float sh = uShadowOn > 0.5 ? shadowAmount(1.0, 1.0, 1.0) : 1.0;
    float fall = 1.0 - smoothstep(2.2, 6.0, length(vObj.xz));
    gl_FragColor = vec4(0.05, 0.06, 0.12, (1.0 - sh) * 0.34 * fall);
}
"""


# ---------------------------------------------------------------------------
# Matrices (column-vector convention, like OpenGL)
# ---------------------------------------------------------------------------
def translate(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def rotate(deg, axis):
    a, c, s = math.radians(deg), math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = np.eye(4)
    if axis == "x":
        m[1:3, 1:3] = [[c, -s], [s, c]]
    else:
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def look_at(eye, center, up):
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def ortho(half, near, far):
    m = np.eye(4)
    m[0, 0] = m[1, 1] = 1.0 / half
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


class Camera:
    def __init__(self, view="bust"):
        self.view = view
        self.zoom_mul = 1.0
        self.ty, self.dist = self.target(view, -7.3)
        self.yaw, self.pitch = 20.0, 4.0
        self.tx = 0.0                 # sideways look-at offset (used for close-ups)

    @staticmethod
    def target(view, ground_y):
        if view == "face":
            return -0.02, 3.3
        if view == "bust":
            return -0.8, 6.5
        if view == "hands":            # for checking hand shapes (not in the V cycle)
            return -3.9, 4.6
        span = (1.45 - ground_y) * 1.08
        return (1.45 + ground_y) / 2, span / (2 * math.tan(math.radians(FOV / 2)))

    def set_view(self, view):
        self.view, self.zoom_mul = view, 1.0

    def update(self, ground_y):
        ty, dist = self.target(self.view, ground_y)
        self.ty += (ty - self.ty) * 0.15
        self.dist += (max(1.5, dist * self.zoom_mul) - self.dist) * 0.15

    def zoom(self, steps):
        self.zoom_mul = max(0.25, min(3.0, self.zoom_mul * 0.9 ** steps))

    def matrix(self):
        """World -> eye transform (the same one draw_scene loads into OpenGL)."""
        return (translate(0, 0, -self.dist) @ rotate(self.pitch, "x") @ rotate(self.yaw, "y")
                @ translate(-self.tx, -self.ty, 0))


class ShadowMap:
    """Depth-only framebuffer the key light renders into."""

    def __init__(self, size=SHADOW_SIZE):
        self.size, self.ok = size, False
        try:
            self.tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, size, size, 0, GL_DEPTH_COMPONENT,
                         GL_FLOAT, None)
            for name, val in ((GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                              (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE),
                              (GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE),
                              (GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)):
                glTexParameteri(GL_TEXTURE_2D, name, val)
            self.fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.tex, 0)
            glDrawBuffer(GL_NONE)
            glReadBuffer(GL_NONE)
            self.ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glDrawBuffer(GL_BACK)
            glReadBuffer(GL_BACK)
        except Exception:                      # very old drivers: run without shadows
            self.ok = False
            try:
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
            except Exception:
                pass


class Renderer:
    """Owns the shader programs and draws a rig for a given pose and time."""

    def __init__(self):
        compile_ = shaders.compileShader

        def program(vs, fs):
            return shaders.compileProgram(compile_(vs, GL_VERTEX_SHADER), compile_(fs, GL_FRAGMENT_SHADER),
                                          validate=False)

        self.prog, self.oprog = program(VERTEX, FRAGMENT), program(OUTLINE_VERTEX, OUTLINE_FRAGMENT)
        self.dprog, self.gprog = program(DEPTH_VERTEX, DEPTH_FRAGMENT), program(GROUND_VERTEX, GROUND_FRAGMENT)
        disp = ("uStrandScale", "uSwing", "uSwingScale", "uJaw", "uJawShift")
        loc = lambda p, names: {n: glGetUniformLocation(p, n) for n in names}
        self.u = loc(self.prog, ("uKind", "uSpec", "uShine", "uColor2", "uPattern", "uFreckle", "uCenter",
                                 "uStrand", "uLightVP", "uInvView", "uShadow", "uShadowOn", "uShadowTexel",
                                 "uAoVP", "uAoMap", "uAoOn", "uAoTexel", "uTaps") + disp)
        self.ou = loc(self.oprog, ("uPx", "uViewH", "uOutline") + disp)
        self.du = loc(self.dprog, disp)
        self.gu = loc(self.gprog, ("uLightVP", "uInvView", "uShadow", "uShadowOn", "uShadowTexel"))
        self.blink, self.gaze, self.lid, self.jaw = 0.0, (0.0, 0.0), 0.0, 0.0
        self.swing = (0.0, 0.0, 0.0)
        self.lod, self.strand_scale = 1.0, 1.0
        self.light_vp = np.eye(4)
        self.inv_view = np.eye(4)
        self.view_h = 720.0
        self.shadow = ShadowMap()
        self.ao = ShadowMap(512)
        self.ao_vp = np.eye(4)
        self.target_fbo = 0            # where the scene is drawn (an off-screen buffer for hi-res captures)
        self.px_scale = 1.0            # outline width multiplier when supersampling
        self.quality = 1.0
        self.offscreen = None
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        # Lights are stored in eye space (set with an identity modelview): key, fill, rim.
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, (*KEY_LIGHT, 0.0))
        glLightfv(GL_LIGHT1, GL_POSITION, (0.8, 0.1, 0.6, 0.0))
        glLightfv(GL_LIGHT2, GL_POSITION, (0.2, 0.5, -1.0, 0.0))

    # ---- scene ---------------------------------------------------------------
    def draw_scene(self, rig, pose, cam, bg, size, blink=0.0, transparent=False, reserved=0,
                   gaze=(0.0, 0.0), lid=0.0, jaw=0.0, swing=(0.0, 0.0, 0.0), shadows=True, quality=1.0):
        """Draw into the whole window; `reserved` px on the right (the UI card) are kept free by
        shifting the lens, so the avatar stays centred in the remaining area.

        gaze (pitch, yaw) turns the eyes; lid closes the lids extra (expressions); jaw is how far the
        lower face drops (talking); swing is the hair sway; shadows toggles the cast shadows;
        quality 1.0 / 0.6 / 0.3 (High / Balanced / Fast) thins the hair and, below 1, drops ambient occlusion.
        """
        self.quality = quality
        w, h = size
        self.view_h = float(h)
        self.blink, self.gaze, self.lid, self.jaw, self.swing = blink, gaze, lid, jaw, swing
        # Fewer strands at a distance (drawn thicker) keeps long hair light on the GPU.
        self.lod = float(np.interp(cam.dist, [3.3, 6.5, 10.0, 15.0], [1.0, 0.85, 0.65, 0.5])) * (0.4 + 0.6 * quality)
        self.strand_scale = 1.0 / math.sqrt(self.lod)
        view = cam.matrix()
        self.inv_view = np.linalg.inv(view)
        use_shadow = bool(shadows and self.shadow.ok)
        if use_shadow:
            self._shadow_pass(rig, pose, view, cam)
        glViewport(0, 0, w, h)
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not transparent:
            self._background(*bg)
        glClear(GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glTranslatef(-reserved / max(w, 1), 0.0, 0.0)
        gluPerspective(FOV, w / max(h, 1), 0.5, 80)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixf(view.T.astype(np.float32))
        if use_shadow:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self.shadow.tex)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, self.ao.tex if self.ao.ok else 0)
            glActiveTexture(GL_TEXTURE0)
        self._ground(rig.ground_y, use_shadow)
        self._outline_pass(rig, pose)
        self._main_pass(rig, pose, use_shadow)

    def _light_matrices(self, view, rig):
        """Orthographic key-light camera framing the avatar (the light is fixed to the camera)."""
        to_world = view[:3, :3].T
        direction = to_world @ (KEY_LIGHT / np.linalg.norm(KEY_LIGHT))
        top = 0.9
        center = np.array([0.0, (top + rig.ground_y) / 2, 0.0])
        radius = (top - rig.ground_y) / 2 + 1.6
        up = np.array([0.0, 1.0, 0.0]) if abs(direction[1]) < 0.95 else np.array([0.0, 0.0, 1.0])
        eye = center + direction * radius * 2
        return look_at(eye, center, up), ortho(radius, radius * 0.4, radius * 3.6)

    def _sky_matrices(self, rig):
        """Orthographic camera looking down from above (slightly forward): the sky-light occluders."""
        d = np.array([0.18, 1.0, 0.35])
        d /= np.linalg.norm(d)
        top = 0.9
        center = np.array([0.0, (top + rig.ground_y) / 2, 0.0])
        radius = (top - rig.ground_y) / 2 + 1.6
        eye = center + d * radius * 2
        return look_at(eye, center, np.array([0.0, 0.0, -1.0])), ortho(radius, radius * 0.4, radius * 3.6)

    def _shadow_pass(self, rig, pose, view, cam):
        light_view, light_proj = self._light_matrices(view, rig)
        self.light_vp = light_proj @ light_view
        self._depth_pass(self.shadow, rig, pose, light_view, light_proj)
        if self.ao.ok and self.quality >= 1.0:
            sky_view, sky_proj = self._sky_matrices(rig)
            self.ao_vp = sky_proj @ sky_view
            self._depth_pass(self.ao, rig, pose, sky_view, sky_proj)

    def _depth_pass(self, smap, rig, pose, light_view, light_proj):
        glBindFramebuffer(GL_FRAMEBUFFER, smap.fbo)
        glViewport(0, 0, smap.size, smap.size)
        glClear(GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(light_proj.T.astype(np.float32))
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(light_view.T.astype(np.float32))
        glUseProgram(self.dprog)
        self._set_displacement(self.du)
        self._walk(rig.root, pose, self._draw_depth)
        glUseProgram(0)
        glBindFramebuffer(GL_FRAMEBUFFER, self.target_fbo)

    def _set_displacement(self, u, jaw=None):
        glUniform1f(u["uStrandScale"], self.strand_scale)
        glUniform3f(u["uSwing"], *self.swing)
        glUniform1f(u["uSwingScale"], 0.0)
        glUniform1f(u["uJaw"], 0.0)

    def _background(self, top, bottom):
        glUseProgram(0)
        self._ortho(1, 1)
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_QUADS)
        glColor3f(*top)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        glColor3f(*bottom)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def _ground(self, ground_y, use_shadow):
        """A soft contact blob plus (with shadows on) the avatar's real cast shadow."""
        glUseProgram(0)
        glDepthMask(GL_FALSE)
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0, 0, 0, 0.20 if use_shadow else 0.28)
        glVertex3f(0, ground_y, 0.05)
        glColor4f(0, 0, 0, 0.0)
        for k in range(33):
            a = 2 * math.pi * k / 32
            glVertex3f(2.1 * math.cos(a), ground_y, 0.05 + 1.3 * math.sin(a))
        glEnd()
        if use_shadow:
            glUseProgram(self.gprog)
            g = self.gu
            glUniformMatrix4fv(g["uLightVP"], 1, GL_TRUE, self.light_vp.astype(np.float32))
            glUniformMatrix4fv(g["uInvView"], 1, GL_TRUE, self.inv_view.astype(np.float32))
            glUniform1i(g["uShadow"], 1)
            glUniform1f(g["uShadowOn"], 1.0)
            glUniform1f(g["uShadowTexel"], 1.0 / self.shadow.size)
            glBegin(GL_QUADS)
            for x, z in ((-7, -7), (7, -7), (7, 7), (-7, 7)):
                glVertex3f(x, ground_y + 0.002, z)
            glEnd()
            glUseProgram(0)
        glDepthMask(GL_TRUE)

    @staticmethod
    def _ortho(w, h):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    # ---- passes ------------------------------------------------------------------
    def _outline_pass(self, rig, pose):
        glUseProgram(self.oprog)
        glUniform1f(self.ou["uViewH"], self.view_h)
        self._set_displacement(self.ou)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        self._walk(rig.root, pose, self._draw_outline)
        glDisable(GL_CULL_FACE)

    def _main_pass(self, rig, pose, use_shadow):
        glUseProgram(self.prog)
        u = self.u
        glUniformMatrix4fv(u["uLightVP"], 1, GL_TRUE, self.light_vp.astype(np.float32))
        glUniformMatrix4fv(u["uInvView"], 1, GL_TRUE, self.inv_view.astype(np.float32))
        glUniform1i(u["uShadow"], 1)
        glUniform1f(u["uShadowOn"], 1.0 if use_shadow else 0.0)
        glUniform1f(u["uShadowTexel"], 1.0 / self.shadow.size)
        glUniformMatrix4fv(u["uAoVP"], 1, GL_TRUE, self.ao_vp.astype(np.float32))
        glUniform1i(u["uAoMap"], 2)
        glUniform1f(u["uAoOn"], 1.0 if (use_shadow and self.ao.ok and self.quality >= 1.0) else 0.0)
        glUniform1f(u["uAoTexel"], 1.0 / self.ao.size)
        glUniform1f(u["uTaps"], 1.0 if self.quality >= 1.0 else 0.0)
        self._set_displacement(u)
        self._walk(rig.root, pose, self._draw_mesh)
        glUseProgram(0)

    def _walk(self, node, pose, draw):
        glPushMatrix()
        rx, ry, rz = pose.get(node.name, (0.0, 0.0, 0.0))
        if node.parent is None:
            glTranslatef(0, pose.get("root_dy", 0.0), 0)
        if rx or ry or rz:
            px, py, pz = node.pivot
            glTranslatef(px, py, pz)
            glRotatef(ry, 0, 1, 0)
            glRotatef(rx, 1, 0, 0)
            glRotatef(rz, 0, 0, 1)
            glTranslatef(-px, -py, -pz)
        for m in node.meshes:
            if m.anim:
                kind, (px, py, pz) = m.anim
                glPushMatrix()
                glTranslatef(px, py, pz)
                if kind == "blink":
                    glRotatef(self.blink + self.lid, 1, 0, 0)
                else:                     # "gaze": turn the eyeball; the iris follows on the sphere
                    glRotatef(self.gaze[1], 0, 1, 0)
                    glRotatef(self.gaze[0], 1, 0, 0)
                glTranslatef(-px, -py, -pz)
                draw(m)
                glPopMatrix()
            else:
                draw(m)
        for child in node.children:
            self._walk(child, pose, draw)
        glPopMatrix()

    # ---- drawing one mesh --------------------------------------------------------------
    def _count(self, m, lod=None):
        """Index count to draw: hair strands are thinned out with distance."""
        lod = self.lod if lod is None else lod
        if m.strand and m.n_strands and lod < 1.0:
            return max(1, int(m.n_strands * lod)) * m.faces_per_strand * 3
        return len(m.f)

    def _mesh_displacement(self, u, m):
        glUniform1f(u["uSwingScale"], m.swing if m.strand else 0.0)
        glUniform1f(u["uJaw"], self.jaw if m.jaw_follow else 0.0)
        glUniform1f(u["uJawShift"], m.jaw_shift if m.jaw_follow else 0.0)
        glUniform1f(u["uStrandScale"], self.strand_scale if m.strand else 1.0)

    def _arrays(self, m):
        glVertexPointer(3, GL_FLOAT, 0, m.v)
        glNormalPointer(GL_FLOAT, 0, m.n)
        if m.strand_aux is not None:
            glClientActiveTexture(GL_TEXTURE1)
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, 0, m.strand_aux)
            glClientActiveTexture(GL_TEXTURE0)
        else:
            glMultiTexCoord2f(GL_TEXTURE1, 0.0, 0.0)

    @staticmethod
    def _release(m):
        if m.strand_aux is not None:
            glClientActiveTexture(GL_TEXTURE1)
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glClientActiveTexture(GL_TEXTURE0)

    def _draw_depth(self, m):
        # shadows are soft, so the shadow pass draws far fewer (thicker) strands than the picture
        lod = min(self.lod, SHADOW_STRAND_LOD)
        self._mesh_displacement(self.du, m)
        if m.strand:
            glUniform1f(self.du["uStrandScale"], 1.0 / math.sqrt(lod))
        self._arrays(m)
        glDrawElements(GL_TRIANGLES, self._count(m, lod), GL_UNSIGNED_INT, m.f)
        self._release(m)

    def _draw_outline(self, m):
        if not m.outline:
            return
        glUniform1f(self.ou["uPx"], (1.0 if m.thin else 2.2) * self.px_scale)
        glUniform3f(self.ou["uOutline"], *(c * 0.30 for c in m.color))
        self._mesh_displacement(self.ou, m)
        self._arrays(m)
        glDrawElements(GL_TRIANGLES, self._count(m), GL_UNSIGNED_INT, m.f)
        self._release(m)

    def _draw_mesh(self, m):
        u = self.u
        glUniform1f(u["uKind"], KINDS[m.kind])
        glUniform1f(u["uSpec"], m.spec)
        glUniform1f(u["uShine"], m.shine)
        glUniform3f(u["uColor2"], *m.color2)
        glUniform1f(u["uPattern"], float(m.pattern))
        glUniform1f(u["uFreckle"], m.freckle)
        glUniform3f(u["uCenter"], *m.center)
        glUniform1f(u["uStrand"], 1.0 if m.strand else 0.0)
        self._mesh_displacement(u, m)
        self._arrays(m)
        if m.tangents is not None:
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(3, GL_FLOAT, 0, m.tangents)
        else:
            glMultiTexCoord3f(GL_TEXTURE0, 0.0, 1.0, 0.0)
        if m.colors is not None:
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(m.colors.shape[1], GL_FLOAT, 0, m.colors)
        else:
            glColor3f(*m.color)
        glDrawElements(GL_TRIANGLES, self._count(m), GL_UNSIGNED_INT, m.f)
        if m.colors is not None:
            glDisableClientState(GL_COLOR_ARRAY)
        if m.tangents is not None:
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        self._release(m)

    # ---- capture ---------------------------------------------------------------------
    def _offscreen(self, w, h):
        if self.offscreen and self.offscreen[0] == (w, h):
            return self.offscreen[1]
        self.release_offscreen()
        fbo = glGenFramebuffers(1)
        color, depth = glGenRenderbuffers(1), glGenRenderbuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glBindRenderbuffer(GL_RENDERBUFFER, color)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color)
        glBindRenderbuffer(GL_RENDERBUFFER, depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth)
        ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.offscreen = ((w, h), fbo if ok else None, color, depth)
        return fbo if ok else None

    def release_offscreen(self):
        if self.offscreen:
            _, fbo, color, depth = self.offscreen
            if fbo:
                glDeleteFramebuffers(1, [fbo])
            glDeleteRenderbuffers(2, [color, depth])
            self.offscreen = None

    def capture(self, rig, pose, cam, bg, size, view, scale=2, supersample=3, alpha=False, **kw):
        """Draw the scene `supersample` times larger off-screen and shrink it to `scale` x the view size:
        much smoother edges and hair than the live window. Returns a pygame surface.
        Falls back to the plain on-screen picture if the graphics card can't do it."""
        w, h = size
        vw, vh = view
        limit = int(glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE) or 4096)
        ss = supersample
        while ss > 1 and max(w, h) * ss > limit:
            ss -= 1
        target_w, target_h = int(vw * scale), int(vh * scale)
        big = max(ss, scale)
        while big > 1 and max(w, h) * big > limit:
            big -= 1
        fbo = self._offscreen(w * big, h * big) if big > 1 or scale > 1 else None
        if not fbo:
            self.release_offscreen()
            return self.grab(view, alpha=alpha)
        keep = kw.pop("reserved", 0)
        self.target_fbo, self.px_scale = fbo, float(big)
        try:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            self.draw_scene(rig, pose, cam, bg, (w * big, h * big), transparent=alpha, reserved=keep * big, **kw)
            surf = self.grab((vw * big, vh * big), alpha=alpha)
        finally:
            self.target_fbo, self.px_scale = 0, 1.0
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if surf.get_size() != (target_w, target_h):
            surf = pygame.transform.smoothscale(surf, (target_w, target_h))
        return surf

    @staticmethod
    def grab(size, alpha=False):
        w, h = size
        fmt = GL_RGBA if alpha else GL_RGB
        data = glReadPixels(0, 0, w, h, fmt, GL_UNSIGNED_BYTE)
        surf = pygame.image.frombuffer(bytes(data), (w, h), "RGBA" if alpha else "RGB")
        return pygame.transform.flip(surf, False, True)
