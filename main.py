
"""
=============================================================================
  MegaMinds Robotics Internship Assessment
  Module: main.py
  ---------------------------------------------------------------------------
  Entry point for the 2D Autonomous Robot Navigation Assessment.
  Coordinates Pygame initialization, real-time superloop execution,
  event routing, physics stepping, and corporate telemetry HUD rendering.
=============================================================================
"""

import sys
import time
import math
from typing import List, Tuple, Optional
import pygame

# Arena, environment geometry, bounds, and collision routines
from arena import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS,
    START_ZONE, GOAL_ZONE, OBSTACLES,
    COL_BG, COL_PATH, COL_WAYPOINT, COL_HUD_TEXT,
    COL_MODE_AUTO, COL_MODE_MANUAL, COL_COLLISION_FLASH, COL_GOAL_MSG,
    draw_grid, draw_zones, draw_obstacles,
    handle_boundary_collision, handle_obstacle_collisions
)

# Configuration space inflation and pathfinding
from pathfinder import (
    OBSTACLE_SAFETY_MARGIN,
    build_obstacle_grid
)

# Robot platform, kinematic physics, sensors, and arrival steering
from robot import (
    AutonomousRobot, AutoPilot,
    THRUST_FORCE,
    vec_length, vec_normalize, dist,
    cast_sensor_array, draw_sensors, draw_robot
)


# ---------------------------------------------------------------------------
# HUD & TELEMETRY RENDERING
# ---------------------------------------------------------------------------

def draw_hud(surface: pygame.Surface,
             font_title: pygame.font.Font,
             font_body: pygame.font.Font,
             font_small: pygame.font.Font,
             elapsed_time: float,
             collisions: int,
             dist_travelled: float,
             is_auto: bool,
             speed: float):
    """
    Render corporate telemetry dashboard displaying real-time metrics.
    """
    hud_width = 320
    hud_height = 205
    hud_surf = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)

    # Semi-transparent corporate panel background
    pygame.draw.rect(hud_surf, (12, 14, 20, 220), (0, 0, hud_width, hud_height), border_radius=8)
    pygame.draw.rect(hud_surf, (50, 60, 85, 180), (0, 0, hud_width, hud_height), width=1, border_radius=8)
    surface.blit(hud_surf, (12, 12))

    x_text = 24
    y_offset = 20

    # Assessment header title
    title_surf = font_title.render("MegaMinds Robot Navigation Assessment", True, (255, 215, 0))
    surface.blit(title_surf, (x_text, y_offset))
    y_offset += 24

    # Navigation Mode Badge
    mode_text = "AUTONOMOUS (A* + STEERING)" if is_auto else "MANUAL (KEYBOARD THRUST)"
    mode_color = COL_MODE_AUTO if is_auto else COL_MODE_MANUAL
    mode_surf = font_body.render(f"MODE: {mode_text}", True, mode_color)
    surface.blit(mode_surf, (x_text, y_offset))
    y_offset += 24

    # Elapsed Mission Time
    minutes = int(elapsed_time) // 60
    seconds = int(elapsed_time) % 60
    ms = int((elapsed_time % 1.0) * 100)
    time_str = f"ELAPSED TIME:  {minutes:02d}:{seconds:02d}.{ms:02d}"
    time_surf = font_body.render(time_str, True, COL_HUD_TEXT)
    surface.blit(time_surf, (x_text, y_offset))
    y_offset += 22

    # Collision Counter (highlights red on contact)
    col_color = COL_COLLISION_FLASH if collisions > 0 else COL_HUD_TEXT
    col_surf = font_body.render(f"COLLISIONS:    {collisions}", True, col_color)
    surface.blit(col_surf, (x_text, y_offset))
    y_offset += 22

    # Cumulative Odometer Distance
    dist_surf = font_body.render(f"DISTANCE:      {dist_travelled:.0f} px", True, COL_HUD_TEXT)
    surface.blit(dist_surf, (x_text, y_offset))
    y_offset += 22

    # Kinematic Speed
    speed_surf = font_body.render(f"SPEED:         {speed:.0f} px/s", True, COL_HUD_TEXT)
    surface.blit(speed_surf, (x_text, y_offset))
    y_offset += 22

    # Safety Margin Status Tag
    debug_tag = f"SAFETY MARGIN: +{OBSTACLE_SAFETY_MARGIN}px [DEBUG VISUAL ACTIVE]"
    debug_surf = font_small.render(debug_tag, True, (255, 130, 130))
    surface.blit(debug_surf, (x_text, y_offset))

    # Control Key Hints (bottom-center)
    hint_text = "ARROWS: Apply Thrust  |  SPACE: Toggle Mode  |  R: Reset Assessment  |  ESC: Exit"
    hint_surf = font_small.render(hint_text, True, (120, 130, 155))
    surface.blit(
        hint_surf,
        (SCREEN_WIDTH // 2 - hint_surf.get_width() // 2, SCREEN_HEIGHT - 22)
    )

    # Zone Identifier Labels
    start_lbl = font_small.render("START", True, (255, 255, 255))
    surface.blit(start_lbl, (
        START_ZONE.centerx - start_lbl.get_width() // 2,
        START_ZONE.centery - start_lbl.get_height() // 2
    ))

    goal_lbl = font_small.render("GOAL", True, (255, 255, 255))
    surface.blit(goal_lbl, (
        GOAL_ZONE.centerx - goal_lbl.get_width() // 2,
        GOAL_ZONE.centery - goal_lbl.get_height() // 2
    ))


def draw_path(surface: pygame.Surface,
              path: Optional[List[Tuple[float, float]]],
              path_index: int):
    """
    Render the planned A* trajectory path and waypoint centers.
    """
    if path is None or len(path) < 2:
        return

    path_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    for i in range(max(0, path_index - 1), len(path) - 1):
        p1 = (int(path[i][0]), int(path[i][1]))
        p2 = (int(path[i + 1][0]), int(path[i + 1][1]))
        alpha = 180 if i >= path_index else 60
        pygame.draw.line(
            path_surf,
            (COL_PATH[0], COL_PATH[1], COL_PATH[2], alpha),
            p1, p2, 2
        )
    surface.blit(path_surf, (0, 0))

    for i in range(path_index, len(path)):
        wx, wy = int(path[i][0]), int(path[i][1])
        pygame.draw.circle(surface, COL_WAYPOINT, (wx, wy), 3)
        pygame.draw.circle(surface, COL_PATH, (wx, wy), 3, 1)


def draw_assessment_complete(surface: pygame.Surface,
                             font_large: pygame.font.Font,
                             font_medium: pygame.font.Font,
                             elapsed_time: float,
                             collisions: int,
                             distance_val: float,
                             pulse_time: float):
    """
    Render evaluation success card upon reaching the target zone.
    """
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (0, 0))

    scale = 1.0 + 0.04 * math.sin(pulse_time * 5.0)
    msg = "ASSESSMENT COMPLETE - GOAL REACHED"
    msg_surf = font_large.render(msg, True, COL_GOAL_MSG)
    w = int(msg_surf.get_width() * scale)
    h = int(msg_surf.get_height() * scale)
    msg_scaled = pygame.transform.smoothscale(msg_surf, (w, h))

    surface.blit(msg_scaled, (
        SCREEN_WIDTH // 2 - w // 2,
        SCREEN_HEIGHT // 2 - h // 2 - 40
    ))

    minutes = int(elapsed_time) // 60
    seconds = int(elapsed_time) % 60
    stats = f"Time: {minutes:02d}:{seconds:02d}  |  Collisions: {collisions}  |  Trajectory: {distance_val:.0f} px"
    stats_surf = font_medium.render(stats, True, COL_HUD_TEXT)
    surface.blit(stats_surf, (
        SCREEN_WIDTH // 2 - stats_surf.get_width() // 2,
        SCREEN_HEIGHT // 2 + 25
    ))

    restart_hint = font_medium.render("Press 'R' to re-run assessment", True, (150, 160, 185))
    surface.blit(restart_hint, (
        SCREEN_WIDTH // 2 - restart_hint.get_width() // 2,
        SCREEN_HEIGHT // 2 + 60
    ))


# ---------------------------------------------------------------------------
# MAIN SUPERLOOP ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    """
    Main application loop for the MegaMinds Robot Navigation Assessment.

    Architecture follows the 3-phase real-time simulation pattern:
        Phase 1: Event Handling   (I/O processing & mode switching)
        Phase 2: State Update     (Kinematics, collision resolution, sensor casting)
        Phase 3: Rendering        (Environment, agents, sensors, telemetry)
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("MegaMinds Robot Navigation Assessment")
    clock = pygame.time.Clock()

    # Font initialization
    try:
        font_title = pygame.font.SysFont("consolas", 14, bold=True)
        font_body = pygame.font.SysFont("consolas", 13)
        font_small = pygame.font.SysFont("consolas", 11)
        font_large = pygame.font.SysFont("consolas", 30, bold=True)
        font_medium = pygame.font.SysFont("consolas", 16)
    except Exception:
        font_title = pygame.font.Font(None, 18)
        font_body = pygame.font.Font(None, 18)
        font_small = pygame.font.Font(None, 14)
        font_large = pygame.font.Font(None, 36)
        font_medium = pygame.font.Font(None, 20)

    # Initialize robot at deployment zone center
    robot = AutonomousRobot(START_ZONE.centerx, START_ZONE.centery)

    # Build obstacle grid with safety margin configuration space inflation
    obstacle_grid = build_obstacle_grid(OBSTACLES, OBSTACLE_SAFETY_MARGIN)

    # Instantiate autopilot steering controller
    autopilot = AutoPilot(obstacle_grid)

    # Simulation control variables
    is_autonomous = False
    start_time = time.time()
    goal_reached_time = None
    last_replan_time = 0.0
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)  # Cap timestep to avoid kinematic instability
        pulse_time = time.time()

        # ===========================================================
        # PHASE 1: EVENT HANDLING
        # ===========================================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    is_autonomous = not is_autonomous
                    if is_autonomous:
                        robot.needs_replan = True
                    else:
                        robot.path = None
                        robot.path_index = 0

                elif event.key == pygame.K_r:
                    robot.reset(START_ZONE.centerx, START_ZONE.centery)
                    start_time = time.time()
                    goal_reached_time = None
                    is_autonomous = False

        # ===========================================================
        # PHASE 2: STATE UPDATE
        # ===========================================================
        if not robot.reached_goal:
            thrust_x, thrust_y = 0.0, 0.0
            thrusting = False

            if not is_autonomous:
                # MANUAL MODE: Direct keyboard kinematic thrust
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    thrust_x -= THRUST_FORCE
                if keys[pygame.K_RIGHT]:
                    thrust_x += THRUST_FORCE
                if keys[pygame.K_UP]:
                    thrust_y -= THRUST_FORCE
                if keys[pygame.K_DOWN]:
                    thrust_y += THRUST_FORCE

                thrusting = (thrust_x != 0.0 or thrust_y != 0.0)
                if thrusting:
                    robot.heading = math.atan2(thrust_y, thrust_x)

            else:
                # AUTONOMOUS MODE: A* trajectory tracking with arrival steering
                curr_t = time.time()
                if robot.needs_replan and (curr_t - last_replan_time > 0.5):
                    new_path = autopilot.plan_path(robot, GOAL_ZONE)
                    if new_path is not None:
                        robot.path = new_path
                        robot.path_index = 0
                    robot.needs_replan = False
                    last_replan_time = curr_t

                thrust_x, thrust_y = autopilot.compute_thrust(robot, dt)
                thrusting = (vec_length(thrust_x, thrust_y) > 1.0)

            # Apply kinematic thrust (preserves momentum)
            robot.apply_thrust(thrust_x, thrust_y, dt)

            # Euler kinematic position update
            robot.update_position(dt)

            # Spatial boundary and obstacle contact resolution
            handle_boundary_collision(robot)
            handle_obstacle_collisions(robot, OBSTACLES)

            # Decrement collision visual indicator timer
            if robot.collision_flash_timer > 0:
                robot.collision_flash_timer -= dt

            # Update thrust visual exhaust trail
            thrust_dir = vec_normalize(thrust_x, thrust_y)
            robot.update_particles(dt, thrusting, thrust_dir)

            # Scan forward-looking multi-beam sensor array
            sensor_results = cast_sensor_array(
                robot.x, robot.y, robot.heading, OBSTACLES
            )

            # Reactive hazard avoidance re-plan trigger
            if is_autonomous and autopilot.check_obstacle_ahead(sensor_results):
                robot.needs_replan = True

            # Target arrival verification
            if robot.get_rect().colliderect(GOAL_ZONE):
                goal_cx, goal_cy = GOAL_ZONE.center
                if dist(robot.x, robot.y, goal_cx, goal_cy) < GOAL_ZONE.width / 2.0:
                    robot.reached_goal = True
                    goal_reached_time = time.time()

        else:
            sensor_results = []
            robot.update_particles(dt, False, (0.0, 0.0))

        # ===========================================================
        # PHASE 3: RENDERING
        # ===========================================================
        screen.fill(COL_BG)

        # 1. Environment grid, deployment zones, and obstacle barriers
        draw_grid(screen)
        draw_zones(screen, pulse_time)
        draw_obstacles(screen, safety_margin=OBSTACLE_SAFETY_MARGIN, show_padding_debug=True)

        # 2. Planned A* path with safe cell center nodes
        if is_autonomous and robot.path:
            draw_path(screen, robot.path, robot.path_index)

        # 3. Sensor range indicator: Translucent FOV cone and raycasts
        if not robot.reached_goal:
            draw_sensors(screen, robot, sensor_results)

        # 4. Robot platform with thruster effects and heading indicator
        draw_robot(screen, robot, pulse_time)

        # 5. Corporate telemetry dashboard HUD
        elapsed = (goal_reached_time or time.time()) - start_time
        speed = vec_length(robot.vx, robot.vy)
        draw_hud(
            screen, font_title, font_body, font_small,
            elapsed, robot.collision_count,
            robot.total_distance, is_autonomous, speed
        )

        # 6. Evaluation completion overlay
        if robot.reached_goal:
            draw_assessment_complete(
                screen, font_large, font_medium,
                elapsed, robot.collision_count,
                robot.total_distance, pulse_time
            )

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
