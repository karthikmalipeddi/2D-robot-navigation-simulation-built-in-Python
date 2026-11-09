"""
=============================================================================
  MegaMinds Robotics Internship Assessment
  Module: arena.py
  ---------------------------------------------------------------------------
  Defines the simulation environment, static boundary limits, Start/Goal zones,
  obstacle geometries, and spatial collision resolution routines.
=============================================================================
"""

import math
from typing import List, Tuple
import pygame


# ---------------------------------------------------------------------------
# DISPLAY & ARENA DIMENSIONS
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60


# ---------------------------------------------------------------------------
# PALETTE & VISUAL STYLING
# ---------------------------------------------------------------------------
COL_BG                  = (18, 18, 24)        # Deep navy-black arena background
COL_GRID_LINE           = (30, 30, 42)        # Coordinate grid lines
COL_OBSTACLE            = (200, 60, 80)       # Solid obstacle core
COL_OBSTACLE_BRD        = (255, 90, 110)      # Obstacle perimeter highlight
COL_OBSTACLE_PADDED     = (255, 60, 80, 40)   # Faint transparent red debug safety margin
COL_OBSTACLE_PADDED_BRD = (255, 90, 110, 100) # Boundary outline for padded hitbox
COL_START_FILL          = (50, 180, 120)      # Deployment / Start zone fill
COL_START_BRD           = (80, 220, 150)      # Deployment zone border
COL_GOAL_FILL           = (60, 140, 255)      # Target / Goal zone fill
COL_GOAL_BRD            = (100, 180, 255)     # Target zone border
COL_ROBOT_BODY          = (255, 200, 60)      # Robot chassis fill
COL_ROBOT_BRD           = (255, 230, 130)     # Robot chassis border
COL_ROBOT_DIR           = (255, 255, 255)     # Heading orientation indicator
COL_SENSOR_HIT          = (255, 50, 50)       # Obstacle sensor contact marker
COL_PATH                = (120, 80, 255)      # Planned A* trajectory path
COL_WAYPOINT            = (180, 140, 255)     # Waypoint node marker
COL_HUD_TEXT            = (220, 220, 230)     # Telemetry readout text
COL_MODE_MANUAL         = (80, 220, 150)      # Telemetry manual mode badge
COL_MODE_AUTO           = (100, 180, 255)     # Telemetry autonomous mode badge
COL_GOAL_MSG            = (255, 215, 0)       # Mission completion gold text
COL_COLLISION_FLASH     = (255, 80, 80)       # Collision warning highlight


# ---------------------------------------------------------------------------
# ZONES & STATIC OBSTACLES
# ---------------------------------------------------------------------------
START_ZONE = pygame.Rect(30, 30, 60, 60)
GOAL_ZONE  = pygame.Rect(700, 500, 70, 70)

# Static rectangular obstacles strategically placed across the workspace
OBSTACLES: List[pygame.Rect] = [
    pygame.Rect(200, 100, 30, 200),   # Obstacle 1: Vertical barrier (west)
    pygame.Rect(350, 0,   30, 180),   # Obstacle 2: Vertical barrier (north-center)
    pygame.Rect(350, 280, 30, 200),   # Obstacle 3: Vertical barrier (mid-center)
    pygame.Rect(500, 150, 200, 30),   # Obstacle 4: Horizontal barrier (northeast)
    pygame.Rect(500, 350, 30, 220),   # Obstacle 5: Vertical barrier (southeast)
    pygame.Rect(600, 350, 170, 30),   # Obstacle 6: Horizontal barrier (east)
    pygame.Rect(100, 400, 200, 30),   # Obstacle 7: Horizontal barrier (southwest)
]


# ---------------------------------------------------------------------------
# COLLISION & REACTION DYNAMICS
# ---------------------------------------------------------------------------

def handle_boundary_collision(robot, safety_margin: float = 12.0):
    """
    Constrain robot kinematics within arena boundaries.

    When the robot perimeter touches an outer wall, its position is clamped
    and the velocity normal to that wall is instantaneously arrested,
    preserving tangential momentum.
    """
    r = robot.radius

    # Left boundary constraint
    if robot.x < r:
        robot.x = float(r)
        robot.vx = 0.0

    # Right boundary constraint
    if robot.x > SCREEN_WIDTH - r:
        robot.x = float(SCREEN_WIDTH - r)
        robot.vx = 0.0

    # Upper boundary constraint
    if robot.y < r:
        robot.y = float(r)
        robot.vy = 0.0

    # Lower boundary constraint
    if robot.y > SCREEN_HEIGHT - r:
        robot.y = float(SCREEN_HEIGHT - r)
        robot.vy = 0.0


def handle_obstacle_collisions(robot, obstacles: List[pygame.Rect]):
    """
    Resolve geometric penetration between circular robot chassis and rectangular obstacles.

    For each obstacle:
    1. Clamps robot center to the obstacle rectangle to compute the closest point.
    2. Measures penetration depth and displacement normal.
    3. If penetration occurs, executes an inelastic push-out along the contact
       normal and zeroes the normal velocity component to prevent boundary clipping.
    """
    for obs in obstacles:
        closest_x = max(obs.left, min(robot.x, obs.right))
        closest_y = max(obs.top, min(robot.y, obs.bottom))

        dx = robot.x - closest_x
        dy = robot.y - closest_y
        dist_sq = dx * dx + dy * dy

        if dist_sq < robot.radius * robot.radius:
            d = math.sqrt(dist_sq)

            # Trigger telemetry collision counter and flash timer
            if robot.collision_flash_timer <= 0:
                robot.collision_count += 1
                robot.collision_flash_timer = 0.3

            if d > 1e-4:
                nx = dx / d
                ny = dy / d
                penetration = robot.radius - d

                # Push-out along contact normal
                robot.x += nx * penetration
                robot.y += ny * penetration

                # Neutralize velocity component directed into obstacle surface
                dot = robot.vx * nx + robot.vy * ny
                if dot < 0.0:
                    robot.vx -= dot * nx
                    robot.vy -= dot * ny
            else:
                # Center is directly on obstacle boundary: separate along primary axis
                if robot.x >= obs.centerx:
                    robot.x = float(obs.right + robot.radius)
                else:
                    robot.x = float(obs.left - robot.radius)
                robot.vx = 0.0
                robot.vy = 0.0


# ---------------------------------------------------------------------------
# ARENA RENDERING ROUTINES
# ---------------------------------------------------------------------------

def draw_grid(surface: pygame.Surface, cell_size: int = 20):
    """Render background coordinate reference grid."""
    for x in range(0, SCREEN_WIDTH, cell_size):
        pygame.draw.line(surface, COL_GRID_LINE, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, cell_size):
        pygame.draw.line(surface, COL_GRID_LINE, (0, y), (SCREEN_WIDTH, y))


def draw_zones(surface: pygame.Surface, pulse_time: float):
    """
    Render Start and Goal deployment zones with subtle telemetry pulse animation.
    """
    # Start Zone
    pygame.draw.rect(surface, COL_START_FILL, START_ZONE, border_radius=6)
    pygame.draw.rect(surface, COL_START_BRD, START_ZONE, width=2, border_radius=6)

    # Goal Zone with radiant telemetry glow
    pulse = 0.5 + 0.5 * math.sin(pulse_time * 3.0)
    glow_alpha = int(40 + 60 * pulse)

    glow_surf = pygame.Surface(
        (GOAL_ZONE.width + 20, GOAL_ZONE.height + 20), pygame.SRCALPHA
    )
    glow_color = (COL_GOAL_FILL[0], COL_GOAL_FILL[1], COL_GOAL_FILL[2], glow_alpha)
    pygame.draw.rect(
        glow_surf, glow_color,
        (0, 0, GOAL_ZONE.width + 20, GOAL_ZONE.height + 20),
        border_radius=10
    )
    surface.blit(glow_surf, (GOAL_ZONE.x - 10, GOAL_ZONE.y - 10))

    pygame.draw.rect(surface, COL_GOAL_FILL, GOAL_ZONE, border_radius=6)
    pygame.draw.rect(surface, COL_GOAL_BRD, GOAL_ZONE, width=2, border_radius=6)


def draw_obstacles(surface: pygame.Surface,
                   safety_margin: int = 16,
                   show_padding_debug: bool = True):
    """
    Render static obstacles and transparent debug safety margin bounding boxes.
    """
    if show_padding_debug:
        debug_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for obs in OBSTACLES:
            padded_rect = obs.inflate(safety_margin * 2, safety_margin * 2)
            pygame.draw.rect(debug_surf, COL_OBSTACLE_PADDED, padded_rect, border_radius=6)
            pygame.draw.rect(debug_surf, COL_OBSTACLE_PADDED_BRD, padded_rect, width=1, border_radius=6)
        surface.blit(debug_surf, (0, 0))

    for obs in OBSTACLES:
        shadow = obs.move(3, 3)
        pygame.draw.rect(surface, (10, 10, 15), shadow, border_radius=4)
        pygame.draw.rect(surface, COL_OBSTACLE, obs, border_radius=4)
        pygame.draw.rect(surface, COL_OBSTACLE_BRD, obs, width=2, border_radius=4)


# ---------------------------------------------------------------------------
# OBJECT-ORIENTED ARENA ENVIRONMENT CLASS
# ---------------------------------------------------------------------------

class Arena:
    """
    Encapsulates environmental geometry, workspace bounds, and physical contact dynamics.
    """

    def __init__(self, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT):
        self.width = width
        self.height = height
        self.start_zone = START_ZONE
        self.goal_zone = GOAL_ZONE
        self.obstacles = OBSTACLES

    def resolve_boundary_constraints(self, robot):
        """Clamp robot within arena boundary constraints."""
        handle_boundary_collision(robot)

    def resolve_obstacle_collisions(self, robot):
        """Resolve circular robot collisions against static rectangular obstacles."""
        handle_obstacle_collisions(robot, self.obstacles)

    def render(self, surface: pygame.Surface, pulse_time: float,
               safety_margin: int = 16, show_debug_margins: bool = True):
        """Render all environmental layers onto target display surface."""
        draw_grid(surface)
        draw_zones(surface, pulse_time)
        draw_obstacles(surface, safety_margin=safety_margin, show_padding_debug=show_debug_margins)
