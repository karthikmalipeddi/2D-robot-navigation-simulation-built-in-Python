"""
=============================================================================
  MegaMinds Robotics Internship Assessment
  Module: pathfinder.py
  ---------------------------------------------------------------------------
  Implements grid discretization, configuration-space obstacle inflation,
  admissible Octile heuristic estimation, and optimal A* pathfinding.
=============================================================================
"""

import math
import heapq
from typing import List, Tuple, Optional, Set
import pygame

from arena import SCREEN_WIDTH, SCREEN_HEIGHT, OBSTACLES


# ---------------------------------------------------------------------------
# GRID & NAVIGATION CONSTANTS
# ---------------------------------------------------------------------------
GRID_CELL_SIZE = 20
GRID_COLS = SCREEN_WIDTH // GRID_CELL_SIZE   # 40 columns
GRID_ROWS = SCREEN_HEIGHT // GRID_CELL_SIZE  # 30 rows

# Safety margin for obstacle inflation: at least half the robot's width (width = 24px -> half = 12px)
# Configured to 16px to guarantee clearance during diagonal traversal and high-momentum maneuvers
OBSTACLE_SAFETY_MARGIN = 16


# ---------------------------------------------------------------------------
# COORDINATE CONVERSION UTILITIES
# ---------------------------------------------------------------------------

def grid_to_pixel(col: int, row: int) -> Tuple[float, float]:
    """
    Convert discrete grid indices (col, row) to exact continuous X/Y cell center coordinates.

    Center-Point Targeting requires the robot to target cell midlines:
        x_center = col * cell_size + cell_size / 2.0
        y_center = row * cell_size + cell_size / 2.0
    """
    return (
        col * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0,
        row * GRID_CELL_SIZE + GRID_CELL_SIZE / 2.0
    )


def pixel_to_grid(px: float, py: float) -> Tuple[int, int]:
    """
    Convert continuous pixel coordinates into discrete grid cell indices (col, row),
    clamped to valid arena boundaries.
    """
    col = int(px // GRID_CELL_SIZE)
    row = int(py // GRID_CELL_SIZE)
    col = max(0, min(col, GRID_COLS - 1))
    row = max(0, min(row, GRID_ROWS - 1))
    return (col, row)


# ---------------------------------------------------------------------------
# CONFIGURATION-SPACE OBSTACLE INFLATION
# ---------------------------------------------------------------------------

def build_obstacle_grid(obstacles: List[pygame.Rect],
                        safety_margin: int = OBSTACLE_SAFETY_MARGIN) -> List[List[int]]:
    """
    Construct a 2D occupancy grid where 0 = navigable, 1 = impassable.

    Obstacle Inflation (Padding):
    Expands the logical bounding box of every static obstacle by the safety
    margin (>= half robot width) in all directions. Any grid cell intersecting
    the inflated bounding box is tagged as blocked. This establishes a protective
    buffer preventing chassis or corner clipping.
    """
    grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cell_rect = pygame.Rect(
                col * GRID_CELL_SIZE,
                row * GRID_CELL_SIZE,
                GRID_CELL_SIZE,
                GRID_CELL_SIZE
            )

            # Test intersection against inflated obstacle bounding boxes
            for obs in obstacles:
                inflated = obs.inflate(safety_margin * 2, safety_margin * 2)
                if cell_rect.colliderect(inflated):
                    grid[row][col] = 1
                    break

    # Block cells outside safe perimeter boundary
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cx = col * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
            cy = row * GRID_CELL_SIZE + GRID_CELL_SIZE // 2
            if (cx < safety_margin or cx > SCREEN_WIDTH - safety_margin or
                cy < safety_margin or cy > SCREEN_HEIGHT - safety_margin):
                grid[row][col] = 1

    return grid


# ---------------------------------------------------------------------------
# OCTILE HEURISTIC
# ---------------------------------------------------------------------------

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Admissible and consistent Octile distance heuristic for 8-connected grid navigation.

    Formula:
        h = max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)
    Guarantees optimal path convergence while accurately accounting for diagonal step costs.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy)


# ---------------------------------------------------------------------------
# A* SEARCH ALGORITHM
# ---------------------------------------------------------------------------

def astar_pathfind(grid: List[List[int]],
                   start: Tuple[int, int],
                   goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Execute deterministic A* search on the occupancy grid.

    Features:
    1. Min-heap priority queue ordered by total estimated cost f = g + h.
    2. Strict corner-cutting validation: Diagonal moves require both orthogonal
       flanking cells to be open, ensuring the robot chassis fits through passages.
    3. Deterministic path reconstruction via back-tracking pointers.

    Returns:
        Ordered list of (col, row) waypoints from start to goal, or None if blocked.
    """
    sr, sc = start[1], start[0]
    gr, gc = goal[1], goal[0]

    # Validate coordinate bounds
    if not (0 <= sr < GRID_ROWS and 0 <= sc < GRID_COLS):
        return None
    if not (0 <= gr < GRID_ROWS and 0 <= gc < GRID_COLS):
        return None

    # Verify start and goal locations are walkable
    if grid[sr][sc] == 1 or grid[gr][gc] == 1:
        return None

    # 8-Directional neighbor offsets: (dcol, drow, cost)
    SQRT2 = math.sqrt(2.0)
    neighbors = [
        (-1, -1, SQRT2), (0, -1, 1.0), (1, -1, SQRT2),
        (-1,  0, 1.0),                 (1,  0, 1.0),
        (-1,  1, SQRT2), (0,  1, 1.0), (1,  1, SQRT2),
    ]

    open_set: list = []
    counter = 0
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: dict = {}
    g_score: dict = {start: 0.0}
    closed_set: Set[Tuple[int, int]] = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct trajectory
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        if current in closed_set:
            continue
        closed_set.add(current)

        cx, cy = current
        for dcol, drow, cost in neighbors:
            nx, ny = cx + dcol, cy + drow

            if not (0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS):
                continue
            if grid[ny][nx] == 1:
                continue

            # Corner-cutting prevention: Ensure both adjacent cardinal cells are open
            if dcol != 0 and drow != 0:
                if grid[cy][cx + dcol] == 1 or grid[cy + drow][cx] == 1:
                    continue

            neighbor = (nx, ny)
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbor))

    return None


def simplify_path(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Optional utility to prune collinear waypoints from a discrete path.
    (Note: In Center-Point Targeting mode, full cell trajectories are preferred
    to keep the robot centered on safe cell midlines).
    """
    if len(path) <= 2:
        return path

    simplified = [path[0]]
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        d2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        if d1 != d2:
            simplified.append(path[i])
    simplified.append(path[-1])
    return simplified


# ---------------------------------------------------------------------------
# OBJECT-ORIENTED PATHFINDER CLASS
# ---------------------------------------------------------------------------

class PathFinder:
    """
    Maintains the global occupancy grid and coordinates A* trajectory planning.
    """

    def __init__(self, obstacles: List[pygame.Rect] = OBSTACLES,
                 safety_margin: int = OBSTACLE_SAFETY_MARGIN):
        self.safety_margin = safety_margin
        self.obstacles = obstacles
        self.grid = build_obstacle_grid(obstacles, safety_margin)

    def rebuild_grid(self, safety_margin: Optional[int] = None):
        """Recompute occupancy matrix if obstacles or safety margins change."""
        if safety_margin is not None:
            self.safety_margin = safety_margin
        self.grid = build_obstacle_grid(self.obstacles, self.safety_margin)

    def plan_trajectory(self, start_pos: Tuple[float, float],
                        goal_pos: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        Compute an optimal path from continuous start_pos (x, y) to goal_pos (x, y).

        Enforces Center-Point Targeting:
        Every intermediate node corresponds to the exact geometric center of a safe grid cell.
        """
        start_cell = pixel_to_grid(start_pos[0], start_pos[1])
        goal_cell = pixel_to_grid(goal_pos[0], goal_pos[1])

        # If start cell lies inside an inflated boundary, find nearest open cell
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

        # Convert discrete path cells into exact continuous center-point coordinates
        return [grid_to_pixel(c, r) for c, r in cell_path]
