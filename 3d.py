import pygame
import random
from OpenGL.GL import (
    glEnable, glClearColor, glClear, glPushMatrix, glPopMatrix, glColor3fv, glBegin, glEnd,
    glVertex3f, glRotatef, glTranslatef, glScalef, glMaterialfv, glMaterialf, glLightfv,
    glColorMaterial, glGetIntegerv, GL_MULTISAMPLE, GL_DEPTH_TEST, GL_LIGHTING, GL_LIGHT0, GL_POSITION,
    GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR, GL_COLOR_MATERIAL, GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
    GL_SHININESS, GL_LINE_LOOP, GL_LINES, GL_MODELVIEW_STACK_DEPTH
)
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT
)
from OpenGL.GLU import gluPerspective, gluNewQuadric, gluQuadricNormals, gluSphere, gluCylinder, gluDeleteQuadric, GLU_SMOOTH

# Initialize Pygame and OpenGL
pygame.init()
WIDTH, HEIGHT = 400, 400
flags = pygame.OPENGL | pygame.DOUBLEBUF
screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
pygame.display.set_caption("3D Avatar")
glEnable(GL_MULTISAMPLE)

FPS = 60
clock = pygame.time.Clock()

# Colors
WHITE = (1, 1, 1)
BLACK = (0.0, 0.0, 0.0)
SKIN_COLORS = [
    (0.96, 0.80, 0.69),  # Light (Fitzpatrick I-II)
    (0.87, 0.72, 0.53),  # Light-medium (Fitzpatrick II-III)
    (0.76, 0.60, 0.42),  # Medium (Fitzpatrick III-IV)
    (0.60, 0.44, 0.29),  # Medium-dark (Fitzpatrick IV-V)
    (0.44, 0.32, 0.19),  # Dark (Fitzpatrick V-VI)
    (0.29, 0.20, 0.13),  # Deep (Fitzpatrick VI)
]
HAIR_COLORS = [
    (0.10, 0.07, 0.03),   # Black
    (0.36, 0.25, 0.20),   # Dark Brown
    (0.55, 0.36, 0.22),   # Medium Brown
    (0.76, 0.60, 0.42),   # Light Brown
    (0.93, 0.80, 0.69),   # Blonde
    (0.85, 0.62, 0.35),   # Dark Blonde
    (0.98, 0.82, 0.53),   # Light Blonde
    (0.80, 0.36, 0.22),   # Auburn/Red
    (0.72, 0.53, 0.40),   # Ash Brown
]
EYE_COLORS = [
    (0.13, 0.22, 0.30),  # Dark Brown
    (0.36, 0.25, 0.20),  # Medium Brown
    (0.55, 0.36, 0.22),  # Light Brown/Hazel
    (0.22, 0.36, 0.45),  # Blue
    (0.32, 0.45, 0.36),  # Green
    (0.50, 0.50, 0.36),  # Hazel/Amber
    (0.18, 0.24, 0.22),  # Gray
]

# Randomize avatar features
skin_color = random.choice(SKIN_COLORS)
hair_color = random.choice(HAIR_COLORS)
eye_color = random.choice(EYE_COLORS)
print(f"Selected colors: Skin={skin_color}, Hair={hair_color}, Eyes={eye_color}")

# Global variables
quadric = None

def setup():
    global quadric
    print("Entering setup...")
    try:
        glClearColor(*WHITE, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (0, 2, 3, 0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.5, 0.5, 0.5, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 80.0)
        gluPerspective(45, WIDTH / HEIGHT, 0.1, 50.0)
        glTranslatef(0, 0, -5)
        quadric = gluNewQuadric()
        if not quadric:
            raise RuntimeError("Failed to create quadric")
        gluQuadricNormals(quadric, GLU_SMOOTH)
        print("Setup completed successfully")
    except Exception as e:
        print(f"Error in setup: {e}")
        exit(1)

def draw_sphere(radius, slices, stacks, color):
    global quadric
    try:
        glPushMatrix()
        if quadric is None:
            print("Error: Quadric not initialized")
            glPopMatrix()
            return
        print(f"Drawing sphere with color: {color}")
        glColor3fv(color)
        gluSphere(quadric, radius, slices, stacks)
        glPopMatrix()
    except Exception as e:
        print(f"Error in draw_sphere: {e}")
        glPopMatrix()

def draw_cylinder(radius, height, slices, color):
    global quadric
    try:
        glPushMatrix()
        if quadric is None:
            print("Error: Quadric not initialized")
            glPopMatrix()
            return
        print(f"Drawing cylinder with color: {color}")
        glColor3fv(color)
        gluCylinder(quadric, radius, radius, height, slices, 1)
        glPopMatrix()
    except Exception as e:
        print(f"Error in draw_cylinder: {e}")
        glPopMatrix()

def draw_wireframe_quads(points, color):
    try:
        glPushMatrix()
        print(f"Drawing wireframe with color: {color}")
        glColor3fv(color)
        glBegin(GL_LINE_LOOP)
        for x, y, z in points:
            glVertex3f(x, y, z)
        glEnd()
        glPopMatrix()
    except Exception as e:
        print(f"Error in draw_wireframe_quads: {e}")
        glPopMatrix()
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
def draw_avatar():
    try:
        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))
        stack_depth = glGetIntegerv(GL_MODELVIEW_STACK_DEPTH)
        print(f"Initial stack depth: {stack_depth}")
        glPushMatrix()
        glRotatef(pygame.time.get_ticks() * 0.05, 0, 1, 0)

        # Head
        draw_sphere(1, 64, 64, (0.96, 0.80, 0.69))

        # Hair (covers top, above eyes and ears)
        glPushMatrix()
        # Raise hair higher on Y axis, and scale to not cover eyes/ears or forehead
        glTranslatef(0, 0.75, 0)  # Move hair even higher above head to reveal forehead
        glScalef(0.95, 0.55, 0.95)  # Make hair smaller and less tall/wide
        draw_sphere(1, 64, 64, BLACK)
        glPopMatrix()

        # Hair strands
        # The following block adds a "tuft" of hair on top of the head for extra detail.
        # Top tuft of hair
        #glPushMatrix()
        #glTranslatef(0, 0.2, 0)
        #Front strand
        #glTranslatef(0, 0, 1.0)
        #glRotatef(-20, 1, 0, 0)
        #draw_cylinder(0.03, 0.7, 32, hair_color)
        #glPopMatrix()
        #Left strand
        #glPushMatrix()
        #glTranslatef(-0.8, 0, 0.5)
        #glRotatef(-30, 1, 0, 0)
        #glRotatef(20, 0, 1, 0)
        #draw_cylinder(0.03, 0.6, 32, hair_color)
        #glPopMatrix()
        #Right strand
        #glPushMatrix()
        #glTranslatef(0.8, 0, 0.5)
        #glRotatef(-30, 1, 0, 0)
        #glRotatef(-20, 0, 1, 0)
        #draw_cylinder(0.03, 0.6, 32, hair_color)
        #glPopMatrix()
        #Back strand
        #glPopMatrix()

        # Eyes
        eye_y = 0.25
        eye_z = 0.78  # Slightly forward for more natural look
        eye_x_offset = 0.36  # Symmetrical spacing from center

        glPushMatrix()
        glTranslatef(-eye_x_offset, eye_y, eye_z)
        draw_sphere(0.18, 32, 32, WHITE)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(eye_x_offset, eye_y, eye_z)
        draw_sphere(0.18, 32, 32, WHITE)
        glPopMatrix()
        
        # Draw pupils
        pupil_radius = 0.07  # Smaller radius for pupils
        pupil_z_offset = 0.13  # Move pupils further forward so they're visible
        glPushMatrix()
        glTranslatef(-eye_x_offset, eye_y, eye_z + pupil_z_offset)
        draw_sphere(pupil_radius, 32, 32, BLACK)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(eye_x_offset, eye_y, eye_z + pupil_z_offset)
        draw_sphere(pupil_radius, 32, 32, BLACK)
        glPopMatrix()


        # Nose
        glPushMatrix()
        glTranslatef(0, 0.18, 0.98)
        glRotatef(90, 1, 0, 0)
        glColor3fv(skin_color)
        print(f"Drawing nose with color: {0.97, 0.80, 0.69}")        
        gluCylinder(quadric, 0.09, 0.03, 0.18, 32, 1)
        glPopMatrix()

        # Glasses
        #draw_wireframe_quads([
           # (-0.45, 0.35, 0.95), (-0.25, 0.35, 0.95),
           # (-0.25, 0.15, 0.95), (-0.45, 0.15, 0.95)
        #], BLACK)
       # draw_wireframe_quads([
        #    (0.25, 0.35, 0.95), (0.45, 0.35, 0.95),
         #   (0.45, 0.15, 0.95), (0.25, 0.15, 0.95)
        #], BLACK)
        #glBegin(GL_LINES)
        #glColor3fv(BLACK)
        #print(f"Drawing glasses lines with color: {BLACK}")
        #glVertex3f(-0.25, 0.25, 0.95)
       # glVertex3f(0.25, 0.25, 0.95)
       # glVertex3f(-0.45, 0.25, 0.95)
       # glVertex3f(-0.65, 0.25, 0.95)
       # glVertex3f(0.45, 0.25, 0.95)
        #glVertex3f(0.65, 0.25, 0.95)
        #glEnd()

        # Ears
        glPushMatrix()
        glTranslatef(-1.05, 0, 0)
        glScalef(0.18, 0.35, 0.08)
        draw_sphere(1, 32, 32, (0.96, 0.80, 0.69))
        glPopMatrix()
        glPushMatrix()
        glTranslatef(1.05, 0, 0)
        glScalef(0.18, 0.35, 0.08)
        draw_sphere(1, 32, 32, (0.96, 0.80, 0.69))
        glPopMatrix()

        # Mouth (upper and lower lips)
        lips_color = (0.85, 0.45, 0.55)
        # Upper lip
        glPushMatrix()
        glTranslatef(0, -0.22, 0.97)
        glRotatef(90, 1, 0, 0)
        glScalef(0.25, 0.04, 0.08)
        print(f"Drawing upper lip with color: {lips_color}")
        glColor3fv(lips_color)
        gluSphere(quadric, 1, 32, 32)
        glPopMatrix()
        # Lower lip
        glPushMatrix()
        glTranslatef(0, -0.28, 0.97)
        glRotatef(90, 1, 0, 0)
        glScalef(0.25, 0.04, 0.08)
        print(f"Drawing lower lip with color: {lips_color}")
        glColor3fv(lips_color)
        gluSphere(quadric, 1, 32, 32)
        glPopMatrix()

        # Mustache
        glPushMatrix()
        glTranslatef(0, -0.1, 0.95)
        glRotatef(90, 1, 0, 0)
        glScalef(0.25, 0.04, 0.04)
        glColor3fv(BLACK)
        print(f"Drawing mustache with color: {BLACK}")
        gluCylinder(quadric, 0.08, 0.08, 0.15, 32, 1)
        glPopMatrix()

        glPopMatrix()
        final_depth = glGetIntegerv(GL_MODELVIEW_STACK_DEPTH)
        print(f"Final stack depth: {final_depth}")
        if final_depth != stack_depth:
            print(f"Matrix stack imbalance detected! Initial: {stack_depth}, Final: {final_depth}")
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
        main()
    except Exception as e:
        print(f"Error running application: {e}")
        cleanup()
