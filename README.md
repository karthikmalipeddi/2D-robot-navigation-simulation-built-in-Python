
README.txt
100%
MegaMinds Robotics Internship – Technical Assessment
Candidate: [Your Name]
Date: November 2026

OVERVIEW
This is a 2D robot navigation simulation built in Python. The simulation evaluates an autonomous robot's ability to navigate an arena, avoid obstacles using raycast sensors, and reach a target goal using an A* pathfinding algorithm. The code is modularized into four packages to ensure production-level readability and maintainability.

SYSTEM REQUIREMENTS & SETUP
- Python Version: Python 3.8 or higher
- Required Libraries: Pygame

INSTALLATION
1. Open your terminal or command prompt.
2. Install the Pygame library by running: pip install pygame

RUN COMMAND
Navigate to the extracted folder containing the 4 Python files and run:
python main.py

CONTROLS
- SPACE: Toggle between Manual Mode and Autonomous (A*) Mode
- Arrow Keys: Apply directional thrust (Manual Mode only)
- R: Reset the simulation to starting positions
- ESC: Quit the application

IMPLEMENTED FEATURES
1. Modular Architecture: Code split across main.py, robot.py, arena.py, and pathfinder.py for clean object-oriented design.
2. Kinematic Physics: The robot utilizes Euler integration to simulate momentum and inertia. 
3. Autonomous Navigation: Uses A* (A-Star) grid-based pathfinding with an Octile distance heuristic for optimal routing.
4. Active Sensor Array: 9 forward-facing raycasts detect obstacles dynamically. If an obstacle breaches the sensor threshold during autonomous mode, it triggers real-time A* path recalculation.
5. Obstacle Inflation: Static obstacles are mathematically padded by the robot's radius in the pathfinding grid to prevent corner-clipping.
6. Arrival Steering: Implements Craig Reynolds' arrival behavior to apply counter-thrust (braking) as the robot nears waypoints, preventing momentum overshoot.
7. Live Telemetry: HUD displaying simulation mode, elapsed time, collision count, distance travelled, and speed.

KNOWN LIMITATIONS
- The braking threshold distance is currently a hardcoded value; at significantly higher maximum speeds, the robot may require dynamic threshold calculations to prevent collisions.
- The A* grid resolution is fixed at 20px cells.
Displaying README.txt.
