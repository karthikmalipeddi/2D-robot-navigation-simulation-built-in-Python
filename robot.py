"""
=============================================================================
  MegaMinds Robotics Internship Assessment
  Module: robot.py
  ---------------------------------------------------------------------------
  Implements the AutonomousRobot agent, kinematic motion physics with
  preserved momentum, Craig Reynolds arrival steering controller, and
  forward-looking raycast sensor payload.
=============================================================================
"""

import math
import random
from typing import List, Tuple, Optional
import pygame

from arena import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    COL_ROBOT_BODY, COL_ROBOT_BRD, COL_ROBOT_DIR,
    COL_SENSOR_HIT
)
from pathfinder import (
    PathFinder, astar_pathfind, grid_to_pixel, pixel_to_grid,
    GRID_ROWS, GRID_COLS
)


# ---------------------------------------------------------------------------
# ROBOT PHYSICAL & KINEMATIC SPECIFICATIONS
# ---------------------------------------------------------------------------
ROBOT_RADIUS = 12                 # Visual chassis radius in pixels
ROBOT_WIDTH = ROBOT_RADIUS * 2    # Physical chassis diameter: 24 pixels
ROBOT_MASS = 1.0                  # Unit mass: Force = Acceleration (F = m * a)
THRUST_FORCE = 200.0              # Manual pilot thrust acceleration (px/s^2)
MAX_SPEED = 300.0                 # Physical maximum velocity ceiling (px/s)

# Autonomous Autopilot Navigation Parameters
AUTO_THRUST_FORCE = 320.0         # Autopilot acceleration / counter-thrust capability (px/s^2)
AUTO_CRUISE_SPEED = 180.0         # Cruise speed for stable trajectory tracking (px/s)
BRAKE_DISTANCE_THRESHOLD = 28.0   # Distance threshold for inertia braking counter-thrust (px)
WAYPOINT_REACH_DIST = 6.0         # Strict center-point arrival tolerance (px)
OBSTACLE_DETECT_DIST = 45.0       # Proximity threshold triggering reactive re-pathing (px)

# Sensor Array Payload Specifications
SENSOR_RANGE = 120.0              # Maximum range of sensor raycasts (px)
SENSOR_FOV = 90.0                 # Field-of-view angular aperture (degrees)
NUM_SENSOR_RAYS = 9               # Number of discrete raycast channels


# ---------------------------------------------------------------------------
# VECTOR MATHEMATICS UTILITIES
# ---------------------------------------------------------------------------

def vec_length(vx: float, vy: float) -> float:
    """Return the Euclidean magnitude of a 2D vector."""
    return math.sqrt(vx * vx + vy * vy)


def vec_normalize(vx: float, vy: float) -> Tuple[float, float]:
    """Return unit direction vector. Returns (0.0, 0.0) for null vectors."""
    length = vec_length(vx, vy)
    if length < 1e-8:
        return (0.0, 0.0)
    return (vx / length, vy / length)


def vec_clamp_magnitude(vx: float, vy: float, max_mag: float) -> Tuple[float, float]:
    """Bound vector magnitude to max_mag while preserving angular direction."""
    mag = vec_length(vx, vy)
    if mag > max_mag and mag > 1e-8:
        scale = max_mag / mag
        return (vx * scale, vy * scale)
    return (vx, vy)


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between points (ax, ay) and (bx, by)."""
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


# ---------------------------------------------------------------------------
# RAYCAST SENSOR PAYLOAD
# ---------------------------------------------------------------------------

def cast_ray(origin_x: float, origin_y: float,
             angle_rad: float, max_dist: float,
             obstacles: List[pygame.Rect]) -> Tuple[float, float, float, bool]:
    """
    Project a discrete sensor ray from origin along angle_rad up to max_dist.

    Returns:
        (hit_x, hit_y, distance_to_contact, did_intersect)
    """
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    step = 4.0

    d = 0.0
    while d < max_dist:
        test_x = origin_x + dx * d
        test_y = origin_y + dy * d

        for obs in obstacles:
            if obs.collidepoint(test_x, test_y):
                return (test_x, test_y, d, True)

        if (test_x < 0 or test_x > SCREEN_WIDTH or
            test_y < 0 or test_y > SCREEN_HEIGHT):
            return (test_x, test_y, d, True)

        d += step

    end_x = origin_x + dx * max_dist
    end_y = origin_y + dy * max_dist
    return (end_x, end_y, max_dist, False)


def cast_sensor_array(robot_x: float, robot_y: float,
                      heading_rad: float,
                      obstacles: List[pygame.Rect]
                      ) -> List[Tuple[float, float, float, bool]]:
    """
    Cast an array of forward-looking sensor beams across the robot's FOV.

    Simulates a multi-beam LIDAR / ultrasonic range-finding sensor payload.
    """
    results = []
    half_fov = math.radians(SENSOR_FOV / 2.0)

    for i in range(NUM_SENSOR_RAYS):
        if NUM_SENSOR_RAYS > 1:
            t = i / (NUM_SENSOR_RAYS - 1)
            ray_angle = heading_rad - half_fov + t * 2.0 * half_fov
        else:
            ray_angle = heading_rad

        res = cast_ray(robot_x, robot_y, ray_angle, SENSOR_RANGE, obstacles)
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# AUTONOMOUS ROBOT CLASS
# ---------------------------------------------------------------------------

class AutonomousRobot:
    """
    Autonomous Mobile Robot platform featuring kinematic motion with preserved momentum.

    Kinematic Motion Model:
        Operates under Newtonian kinematic motion with preserved momentum (zero drag).
        Actuator thrust modifies the velocity vector directly:
            velocity += acceleration * dt
        Without counter-thrust, momentum persists indefinitely, requiring active
        inertial braking to decelerate and prevent overshoot into boundaries.
    """

    def __init__(self, x: float, y: float):
        self.radius = ROBOT_RADIUS
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.heading = 0.0

        # Telemetry metrics
        self.total_distance = 0.0
        self.collision_count = 0
        self.reached_goal = False

        # Autopilot state
        self.path: Optional[List[Tuple[float, float]]] = None
        self.path_index = 0
        self.needs_replan = True

        # Visualization effects
        self.collision_flash_timer = 0.0
        self.thrust_particles: List[dict] = []

    def reset(self, x: float, y: float):
        """Reinitialize robot kinematic state at specified coordinates."""
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.heading = 0.0
        self.total_distance = 0.0
        self.collision_count = 0
        self.reached_goal = False
        self.path = None
        self.path_index = 0
        self.needs_replan = True
        self.collision_flash_timer = 0.0
        self.thrust_particles.clear()

    def apply_thrust(self, ax: float, ay: float, dt: float):
        """
        Apply kinematic thrust acceleration to the velocity vector.

        Kinematic motion with preserved momentum:
            v_new = v_current + a * dt
        """
        self.vx += ax * dt
        self.vy += ay * dt
        self.vx, self.vy = vec_clamp_magnitude(self.vx, self.vy, MAX_SPEED)

    def update_position(self, dt: float):
        """
        Integrate kinematic motion forward in time via first-order Euler integration.

        Kinematic integration:
            p_new = p_current + v * dt
        """
        old_x, old_y = self.x, self.y
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.total_distance += dist(old_x, old_y, self.x, self.y)

        # Update heading orientation along current velocity vector
        speed = vec_length(self.vx, self.vy)
        if speed > 5.0:
            self.heading = math.atan2(self.vy, self.vx)

    def update_particles(self, dt: float, thrusting: bool,
                         thrust_dir: Tuple[float, float]):
        """Update exhaust particle trail for visual feedback of active thrust."""
        self.thrust_particles = [p for p in self.thrust_particles if p['life'] > 0]
        for p in self.thrust_particles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['life'] -= dt

        if thrusting and vec_length(*thrust_dir) > 0.1:
            for _ in range(2):
                spread = random.uniform(-0.4, 0.4)
                speed = random.uniform(30.0, 80.0)
                angle = math.atan2(-thrust_dir[1], -thrust_dir[0]) + spread
                self.thrust_particles.append({
                    'x': self.x,
                    'y': self.y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': random.uniform(0.2, 0.45),
                    'max_life': 0.45,
                    'radius': random.uniform(2.0, 4.5),
                })

    def get_rect(self) -> pygame.Rect:
        """Return the physical bounding box for goal area overlap testing."""
        return pygame.Rect(
            int(self.x - self.radius),
            int(self.y - self.radius),
            int(self.radius * 2),
            int(self.radius * 2)
        )


# Backward compatibility alias
Robot = AutonomousRobot


# ---------------------------------------------------------------------------
# CRAIG REYNOLDS ARRIVAL STEERING CONTROLLER (AUTOPILOT)
# ---------------------------------------------------------------------------

class AutoPilot:
    """
    Autonomous trajectory tracking controller implementing Craig Reynolds arrival
    steering behavior, Center-Point Targeting, and Inertia Braking counter-thrust.
    """

    def __init__(self, grid: List[List[int]]):
        self.grid = grid

    def plan_path(self, robot: AutonomousRobot, goal_rect: pygame.Rect
                  ) -> Optional[List[Tuple[float, float]]]:
        """
        Compute an optimal obstacle-free trajectory using A* search.

        Center-Point Targeting:
        Every intermediate node corresponds to the exact geometric center
        coordinate of a safe grid cell, preventing boundary clipping.
        """
        start_cell = pixel_to_grid(robot.x, robot.y)
        goal_cx = goal_rect.centerx
        goal_cy = goal_rect.centery
        goal_cell = pixel_to_grid(goal_cx, goal_cy)

        # Fallback if robot current cell is marked blocked in inflated grid
        if self.grid[start_cell[1]][start_cell[0]] == 1:
            best_dist = float('inf')
            best_cell = start_cell
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = start_cell[1] + dr, start_cell[0] + dc
                    if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS and self.grid[nr][nc] == 0:
                        d = math.hypot(dr, dc)
                        if d < best_dist:
                            best_dist = d
                            best_cell = (nc, nr)
            start_cell = best_cell

        cell_path = astar_pathfind(self.grid, start_cell, goal_cell)
        if cell_path is None:
            return None

        # Convert grid cell sequence to exact center-point coordinates
        return [grid_to_pixel(c, r) for c, r in cell_path]

    def compute_thrust(self, robot: AutonomousRobot, dt: float
                       ) -> Tuple[float, float]:
        """
        Compute steering acceleration force vector to follow the planned path.

        Mathematical Adjustments:
        1. Center-Point Targeting:
           Targets exact X/Y cell centers; strict arrival distance (WAYPOINT_REACH_DIST)
           ensures the robot aligns with cell centers before advancing.
        2. Inertia Braking:
           When approaching within BRAKE_DISTANCE_THRESHOLD, desired speed ramps down
           and counter-thrust directly cancels preserved momentum and lateral drift.
        """
        if robot.path is None or robot.path_index >= len(robot.path):
            # Target reached or path empty: apply complete counter-thrust to halt
            speed = vec_length(robot.vx, robot.vy)
            if speed > 0.5:
                nvx, nvy = vec_normalize(robot.vx, robot.vy)
                brake = min(AUTO_THRUST_FORCE, speed / dt) if dt > 0 else 0.0
                return (-nvx * brake, -nvy * brake)
            return (0.0, 0.0)

        # Target coordinates of the current safe cell center
        target_x, target_y = robot.path[robot.path_index]
        dx = target_x - robot.x
        dy = target_y - robot.y
        dist_to_target = dist(robot.x, robot.y, target_x, target_y)

        # Advance to next waypoint only when tightly centered on current cell
        if dist_to_target < WAYPOINT_REACH_DIST:
            robot.path_index += 1
            if robot.path_index >= len(robot.path):
                speed = vec_length(robot.vx, robot.vy)
                if speed > 0.5:
                    nvx, nvy = vec_normalize(robot.vx, robot.vy)
                    brake = min(AUTO_THRUST_FORCE, speed / dt) if dt > 0 else 0.0
                    return (-nvx * brake, -nvy * brake)
                return (0.0, 0.0)
            target_x, target_y = robot.path[robot.path_index]
            dx = target_x - robot.x
            dy = target_y - robot.y
            dist_to_target = dist(robot.x, robot.y, target_x, target_y)

        dir_x, dir_y = vec_normalize(dx, dy)

        # Velocity decomposition: parallel to target line and lateral (cross-track drift)
        v_parallel = robot.vx * dir_x + robot.vy * dir_y
        v_lateral_x = robot.vx - v_parallel * dir_x
        v_lateral_y = robot.vy - v_parallel * dir_y

        # Inertia Braking: scale target velocity down within distance threshold
        if dist_to_target < BRAKE_DISTANCE_THRESHOLD:
            target_speed = AUTO_CRUISE_SPEED * (dist_to_target / BRAKE_DISTANCE_THRESHOLD)
        else:
            target_speed = AUTO_CRUISE_SPEED

        desired_vx = dir_x * target_speed
        desired_vy = dir_y * target_speed

        # Steering force: (desired - current) naturally applies counter-thrust (-v)
        # Actively dampens lateral drift to eliminate sliding into walls
        steer_x = (desired_vx - robot.vx) - 1.2 * v_lateral_x
        steer_y = (desired_vy - robot.vy) - 1.2 * v_lateral_y

        steer_x, steer_y = vec_clamp_magnitude(
            steer_x, steer_y, AUTO_THRUST_FORCE * dt
        )

        ax = steer_x / dt if dt > 0 else 0.0
        ay = steer_y / dt if dt > 0 else 0.0

        return (ax, ay)

    def check_obstacle_ahead(self, sensor_results: List[Tuple]) -> bool:
        """
        Evaluate sensor raycasts for imminent obstacle hazards requiring re-planning.
        """
        for (_, _, hit_dist, did_hit) in sensor_results:
            if did_hit and hit_dist < OBSTACLE_DETECT_DIST:
                return True
        return False


# ---------------------------------------------------------------------------
# ROBOT & SENSOR RENDERING ROUTINES
# ---------------------------------------------------------------------------

def draw_sensors(surface: pygame.Surface, robot: AutonomousRobot,
                 sensor_results: List[Tuple]):
    """
    Render sensor visualization: Translucent detection cone, color-coded proximity rays,
    and contact impact markers.
    """
    if not sensor_results:
        return

    # 1. Translucent sensor FOV cone polygon
    cone_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    cone_points = [(int(robot.x), int(robot.y))]
    for (ex, ey, _, _) in sensor_results:
        cone_points.append((int(ex), int(ey)))
    cone_points.append((int(robot.x), int(robot.y)))

    if len(cone_points) >= 3:
        pygame.draw.polygon(cone_surface, (0, 255, 180, 22), cone_points)
    surface.blit(cone_surface, (0, 0))

    # 2. Individual sensor raycasts with proximity color transitions
    ray_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    for (ex, ey, hit_dist, did_hit) in sensor_results:
        if did_hit:
            t = max(0.0, min(1.0, hit_dist / SENSOR_RANGE))
            r = int(255 * (1.0 - t))
            g = int(200 * t)
            color = (r, g, 50, 85)
        else:
            color = (0, 200, 150, 40)

        pygame.draw.line(
            ray_surface, color,
            (int(robot.x), int(robot.y)),
            (int(ex), int(ey)), 1
        )

        if did_hit:
            pygame.draw.circle(surface, COL_SENSOR_HIT, (int(ex), int(ey)), 3)

    surface.blit(ray_surface, (0, 0))


def draw_robot(surface: pygame.Surface, robot: AutonomousRobot, pulse_time: float):
    """
    Render the robot chassis, thruster exhaust trails, collision flash, and heading indicator.
    """
    rx, ry = int(robot.x), int(robot.y)

    # Exhaust particle trail
    particle_surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    for p in robot.thrust_particles:
        life_ratio = max(0.0, p['life'] / p['max_life'])
        alpha = int(150 * life_ratio)
        radius = max(1, int(p['radius'] * life_ratio))
        pygame.draw.circle(
            particle_surf,
            (255, 160, 30, alpha),
            (int(p['x']), int(p['y'])),
            radius
        )
    surface.blit(particle_surf, (0, 0))

    # Collision warning flash effect
    if robot.collision_flash_timer > 0:
        flash_alpha = int(150 * (robot.collision_flash_timer / 0.3))
        flash_surf = pygame.Surface((ROBOT_RADIUS * 4, ROBOT_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(
            flash_surf,
            (255, 80, 80, flash_alpha),
            (ROBOT_RADIUS * 2, ROBOT_RADIUS * 2),
            ROBOT_RADIUS * 2
        )
        surface.blit(flash_surf, (rx - ROBOT_RADIUS * 2, ry - ROBOT_RADIUS * 2))

    # Subtle status glow ring
    glow_pulse = 0.7 + 0.3 * math.sin(pulse_time * 4.0)
    glow_radius = int(ROBOT_RADIUS + 4 * glow_pulse)
    glow_surf = pygame.Surface((glow_radius * 2 + 4, glow_radius * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(
        glow_surf,
        (255, 200, 60, 40),
        (glow_radius + 2, glow_radius + 2),
        glow_radius
    )
    surface.blit(glow_surf, (rx - glow_radius - 2, ry - glow_radius - 2))

    # Primary chassis body
    pygame.draw.circle(surface, COL_ROBOT_BODY, (rx, ry), ROBOT_RADIUS)
    pygame.draw.circle(surface, COL_ROBOT_BRD, (rx, ry), ROBOT_RADIUS, 2)

    # Heading directional nose indicator
    nose_len = ROBOT_RADIUS + 6
    nose_x = rx + int(math.cos(robot.heading) * nose_len)
    nose_y = ry + int(math.sin(robot.heading) * nose_len)
    pygame.draw.line(surface, COL_ROBOT_DIR, (rx, ry), (nose_x, nose_y), 2)
    pygame.draw.circle(surface, COL_ROBOT_DIR, (nose_x, nose_y), 3)
