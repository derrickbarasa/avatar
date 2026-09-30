"""OpenGL rendering: cel-shaded materials with procedural detail, inked outlines,
skeleton drawing, camera and screenshots."""
import math

import numpy as np
import pygame
from OpenGL.GL import *  # noqa: F401,F403
from OpenGL.GL import shaders
from OpenGL.GLU import gluPerspective

KINDS = {"plain": 0.0, "skin": 1.0, "hair": 2.0, "cloth": 3.0, "denim": 4.0, "eye": 5.0,
         "glossy": 6.0}
FOV = 35.0

VERTEX = """
#version 120
centroid varying vec3 vN;
centroid varying vec3 vP;
centroid varying vec3 vObj;
centroid varying vec3 vT;
centroid varying vec4 vColor;
void main() {
    vec4 ep = gl_ModelViewMatrix * gl_Vertex;
    vP = ep.xyz;
    vN = gl_NormalMatrix * gl_Normal;
    vT = gl_NormalMatrix * gl_MultiTexCoord0.xyz;   // strand direction (0,1,0 for other meshes)
    vObj = gl_Vertex.xyz;
    vColor = gl_Color;
    gl_Position = gl_ProjectionMatrix * ep;
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
        base *= 0.96 + 0.06 * hash(floor(vObj.xy * 180.0));
        if (kind > 3.5) {
            base *= 0.95 + 0.10 * vnoise(vec2((vObj.x + vObj.z) * 70.0, vObj.y * 30.0));
        }
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
    float lit = smoothstep(-0.15, 0.35, ndl);
    float hi = smoothstep(0.25, 0.95, ndl);
    vec3 shadowTint = isSkin ? vec3(0.80, 0.58, 0.60) : vec3(0.62, 0.60, 0.78);
    vec3 c = base * mix(shadowTint * 0.72, vec3(0.92), lit);
    c = mix(c, base * vec3(1.0, 0.97, 0.92) * 1.05, hi);
    c += base * vec3(0.32, 0.38, 0.50) * 0.22 * max(dot(N, L1), 0.0);
    c += base * mix(vec3(0.05, 0.04, 0.07), vec3(0.10, 0.11, 0.14), N.y * 0.5 + 0.5);
    if (isSkin) {
        c += base * vec3(0.30, 0.06, 0.02) * smoothstep(0.35, 0.0, abs(ndl - 0.08)) * 0.5;
    }
    float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
    c += vec3(0.30, 0.36, 0.55) * rim * smoothstep(-0.3, 0.5, dot(N, L2)) * 0.5;

    vec3 H = normalize(L0 + V);
    float nh = max(dot(N, H), 0.0);
    if (isHair) {
        vec3 T = normalize(vT);
        float th = dot(T, H);
        float st = sqrt(max(1.0 - th * th, 0.0));
        float kk = (pow(st, 60.0) * 0.6 + pow(st, 14.0) * 0.18) * smoothstep(0.0, 0.3, ndl);
        c += mix(base, vec3(1.0), 0.55) * kk * 0.55;
    } else if (isGlossy || isEye) {
        c += vec3(1.0) * smoothstep(0.55, 0.70, pow(nh, uShine * 0.5)) * spec * (isEye ? 0.8 : 1.4);
    } else {
        c += vec3(1.0) * smoothstep(0.45, 0.55, pow(nh, uShine)) * spec * 1.6;
    }
    gl_FragColor = vec4(c, vColor.a);
}
"""

OUTLINE_VERTEX = """
#version 120
uniform float uPx;
uniform float uViewH;
void main() {
    vec4 ep = gl_ModelViewMatrix * gl_Vertex;
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


class Camera:
    def __init__(self, view="bust"):
        self.view = view
        self.zoom_mul = 1.0
        self.ty, self.dist = self.target(view, -7.3)
        self.yaw, self.pitch = 20.0, 4.0

    @staticmethod
    def target(view, ground_y):
        if view == "face":
            return -0.02, 3.3
        if view == "bust":
            return -0.8, 6.5
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


class Renderer:
    """Owns the shader programs and draws a rig for a given pose and time."""

    def __init__(self):
        self.prog = shaders.compileProgram(
            shaders.compileShader(VERTEX, GL_VERTEX_SHADER),
            shaders.compileShader(FRAGMENT, GL_FRAGMENT_SHADER), validate=False)
        self.oprog = shaders.compileProgram(
            shaders.compileShader(OUTLINE_VERTEX, GL_VERTEX_SHADER),
            shaders.compileShader(OUTLINE_FRAGMENT, GL_FRAGMENT_SHADER), validate=False)
        self.u = {n: glGetUniformLocation(self.prog, n) for n in
                  ("uKind", "uSpec", "uShine", "uColor2", "uPattern", "uFreckle", "uCenter", "uStrand")}
        self.ou = {n: glGetUniformLocation(self.oprog, n) for n in ("uPx", "uViewH", "uOutline")}
        self.blink = 0.0
        self.gaze = (0.0, 0.0)
        self.view_h = 720.0
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
        glLightfv(GL_LIGHT0, GL_POSITION, (-0.4, 0.7, 1.0, 0.0))
        glLightfv(GL_LIGHT1, GL_POSITION, (0.8, 0.1, 0.6, 0.0))
        glLightfv(GL_LIGHT2, GL_POSITION, (0.2, 0.5, -1.0, 0.0))

    # ---- scene ---------------------------------------------------------------
    def draw_scene(self, rig, pose, cam, bg, size, blink=0.0, transparent=False, reserved=0,
                   gaze=(0.0, 0.0)):
        """Draw into the whole window; `reserved` px on the right (the UI card) are kept free by
        shifting the lens, so the avatar stays centred in the remaining area."""
        w, h = size
        self.view_h = float(h)
        self.blink = blink
        self.gaze = gaze                 # (pitch, yaw) degrees for the eyeballs
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
        glTranslatef(0, 0, -cam.dist)
        glRotatef(cam.pitch, 1, 0, 0)
        glRotatef(cam.yaw, 0, 1, 0)
        glTranslatef(0, -cam.ty, 0)
        self._shadow(rig.ground_y)
        self._outline_pass(rig, pose)
        self._main_pass(rig, pose)

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

    @staticmethod
    def _shadow(ground_y):
        glUseProgram(0)
        glDepthMask(GL_FALSE)
        glBegin(GL_TRIANGLE_FAN)
        glColor4f(0, 0, 0, 0.28)
        glVertex3f(0, ground_y, 0.05)
        glColor4f(0, 0, 0, 0.0)
        for k in range(33):
            a = 2 * math.pi * k / 32
            glVertex3f(2.1 * math.cos(a), ground_y, 0.05 + 1.3 * math.sin(a))
        glEnd()
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
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)
        self._walk(rig.root, pose, self._draw_outline)
        glDisable(GL_CULL_FACE)

    def _main_pass(self, rig, pose):
        glUseProgram(self.prog)
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
                    glRotatef(self.blink, 1, 0, 0)
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

    def _arrays(self, m):
        glVertexPointer(3, GL_FLOAT, 0, m.v)
        glNormalPointer(GL_FLOAT, 0, m.n)

    def _draw_outline(self, m):
        if not m.outline:
            return
        glUniform1f(self.ou["uPx"], 1.0 if m.thin else 2.2)
        glUniform3f(self.ou["uOutline"], *(c * 0.30 for c in m.color))
        self._arrays(m)
        glDrawElements(GL_TRIANGLES, len(m.f), GL_UNSIGNED_INT, m.f)

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
        glDrawElements(GL_TRIANGLES, len(m.f), GL_UNSIGNED_INT, m.f)
        if m.colors is not None:
            glDisableClientState(GL_COLOR_ARRAY)
        if m.tangents is not None:
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    # ---- capture ---------------------------------------------------------------------
    @staticmethod
    def grab(size, alpha=False):
        w, h = size
        fmt = GL_RGBA if alpha else GL_RGB
        data = glReadPixels(0, 0, w, h, fmt, GL_UNSIGNED_BYTE)
        surf = pygame.image.frombuffer(bytes(data), (w, h), "RGBA" if alpha else "RGB")
        return pygame.transform.flip(surf, False, True)
