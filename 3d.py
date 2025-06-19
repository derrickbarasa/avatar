import pygame
import random
import math
from OpenGL.GL import *
from OpenGL.GL import (
    glEnable, GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0, GL_COLOR_MATERIAL, glLightfv, GL_POSITION, GL_DIFFUSE,
    glColorMaterial, GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, glClearColor, glTranslatef, glPushMatrix, glPopMatrix,
    glColor3fv, GL_LINE_LOOP, glBegin, glEnd, glVertex3f, GL_LINES, GL_LINE_STRIP,
    glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, glRotatef, glScalef
)
from OpenGL.GLU import *
from OpenGL.GLU import gluPerspective, gluNewQuadric, gluDeleteQuadric, gluSphere, gluCylinder

# Initialize Pygame and OpenGL
try:
    pygame.init()
    WIDTH, HEIGHT = 400, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("3D Avatar")
    print(f"OpenGL Version: {glGetString(GL_VERSION).decode('utf-8')}")
except (pygame.error, RuntimeError) as e:
    print(f"Error initializing Pygame/OpenGL: {e}")
    exit(1)

FPS = 60
clock = pygame.time.Clock()

# Colors
WHITE = (1, 1, 1)
BLACK = (0, 0, 0)
SKIN_COLORS = [(0.96, 0.84, 0.76), (0.82, 0.65, 0.49), (0.47, 0.33, 0.22)]
HAIR_COLORS = [(0, 0, 0), (0.55, 0.27, 0.07), (1, 0.84, 0)]
EYE_COLORS = [(0, 0, 1), (0, 0.5, 0), (0.65, 0.16, 0.16)]

# Randomize avatar features
skin_color = random.choice(SKIN_COLORS)
hair_color = random.choice(HAIR_COLORS)
eye_color = random.choice(EYE_COLORS)


# Global quadric
quadric = None

def setup():
    global quadric
    print("Entering setup...")
    try:
        glClearColor(*WHITE, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        gluPerspective(45, WIDTH / HEIGHT, 0.1, 50.0)
        glTranslatef(0, 0, -5)
        quadric = gluNewQuadric()
        if not quadric:
            raise RuntimeError("Failed to create quadric")
        print("Setup completed successfully")
    except Exception as e:
        print(f"Error in setup: {e}")
        exit(1)

def draw_sphere(radius, slices, stacks, color):
    global quadric
    if quadric is None:
        print("Error: Quadric not initialized")
        return
    try:
        glPushMatrix()
        glColor3fv(color)
        gluSphere(quadric, radius, slices, stacks)
        glPopMatrix()
    except Exception as e:
        print(f"Error in draw_sphere: {e}")

def draw_wireframe_quads(points, color):
    try:
        glPushMatrix()
        glColor3fv(color)
        glBegin(GL_LINE_LOOP)
        for x, y, z in points:
            glVertex3f(x, y, z)
        glEnd()
        glPopMatrix()
    except Exception as e:
        print(f"Error in draw_wireframe_quads: {e}")

def draw_avatar():
    try:
        print("Drawing avatar...")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glRotatef(pygame.time.get_ticks() * 0.05, 0, 1, 0)

        # Head
        draw_sphere(1, 32, 32, skin_color)

        # Hair
        glPushMatrix()
        glTranslatef(0, 0.2, 0)
        glScalef(1.1, 0.6, 1.1)
        draw_sphere(1, 32, 16, hair_color)
        glPopMatrix()

        # Eyes
        glPushMatrix()
        glTranslatef(-0.4, 0.2, 0.9)
        draw_sphere(0.15, 16, 16, eye_color)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(0.4, 0.2, 0.9)
        draw_sphere(0.15, 16, 16, eye_color)
        glPopMatrix()

        # Nose
        glPushMatrix()
        glTranslatef(0, 0, 1)
        glRotatef(90, 1, 0, 0)
        glColor3fv(skin_color)
        gluCylinder(quadric, 0.1, 0, 0.3, 16, 1)
        glPopMatrix()

        # Glasses
        draw_wireframe_quads([
            (-0.5, 0.3, 0.95), (-0.2, 0.3, 0.95),
            (-0.2, 0.1, 0.95), (-0.5, 0.1, 0.95)
        ], BLACK)
        draw_wireframe_quads([
            (0.2, 0.3, 0.95), (0.5, 0.3, 0.95),
            (0.5, 0.1, 0.95), (0.2, 0.1, 0.95)
        ], BLACK)
        glBegin(GL_LINES)
        glColor3fv(BLACK)
        glVertex3f(-0.2, 0.2, 0.95)
        glVertex3f(0.2, 0.2, 0.95)
        glVertex3f(-0.5, 0.2, 0.95)
        glVertex3f(-0.7, 0.2, 0.95)
        glVertex3f(0.5, 0.2, 0.95)
        glVertex3f(0.7, 0.2, 0.95)
        glEnd()

        # Ears
        glPushMatrix()
        glTranslatef(-1.1, 0, 0)
        glScalef(0.2, 0.4, 0.1)
        draw_sphere(1, 16, 16, skin_color)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(1.1, 0, 0)
        glScalef(0.2, 0.4, 0.1)
        draw_sphere(1, 16, 16, skin_color)
        glPopMatrix()

        # Mouth
        glBegin(GL_LINE_STRIP)
        glColor3fv(BLACK)
        for i in range(11):
            x = -0.3 + (0.6 * i) / 10
            y = -0.3 - 0.1 * math.sin(math.pi * i / 10)
            glVertex3f(x, y, 0.95)
        glEnd()

        # Mustache
        glBegin(GL_LINE_STRIP)
        glColor3fv(hair_color)
        for i in range(11):
            x = -0.3 + (0.6 * i) / 10
            y = -0.1 - 0.05 * math.sin(math.pi * i / 10)
            glVertex3f(x, y, 0.95)
        glEnd()

        glPopMatrix()
        print("Avatar drawn successfully")
    except Exception as e:
        print(f"Error in draw_avatar: {e}")

def update_loop():
    try:
        draw_avatar()
        pygame.display.flip()
        clock.tick(FPS)
    except Exception as e:
        print(f"Error in update_loop: {e}")

def cleanup():
    global quadric
    try:
        if quadric:
            gluDeleteQuadric(quadric)
        pygame.quit()
        print("Cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def main():
    setup()
    running = True
    while running:
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            update_loop()
        except Exception as e:
            print(f"Error in main loop: {e}")
            running = False
    cleanup()

if __name__ == "__main__":
    try:
        print("Starting main loop...")
        main()
    except Exception as e:
        print(f"Error running application: {e}")
        cleanup()