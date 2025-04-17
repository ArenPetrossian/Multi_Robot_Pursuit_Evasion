import numpy as np
from PIL import Image
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import cv2

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import rps.robotarium as robotarium
from datetime import datetime
import os
from scipy.ndimage import binary_dilation
import heapq
import imageio.v2 as imageio  # pip install imageio[ffmpeg]
import io

#### Set up constants for experiment ####

# Define which figures to be active
active_figures = [1, 5, 7]
            # 1: Robotarium figure (Always active)
            # 2: Original Map Graph
            # 3: Scaled Maze
            # 4: Inflated Maze for Pursuers
            # 5: A* Attractive Potential Field, Wall Repulsive Potential Field, Wall Raising Potential Field, Static Potential Field
            # 6: 3D Static Potential Field
            # 7: Pursuer Potential Field, Dynamic Potential Field
            # 8: 3D Pursuer Potential Field, 3D Dynamic Potential Field

# Record Video
video_flag = False  # Change to True to record video
fps = 10  # frames per second
ipf = 10  # iterations per frame

# Scale map down to ease computation
scale = 15

# How much to inflate walls for pursuer A*
robot_radius = int(0.06*500/scale)  # Number of grid cells to expand walls

# Exit locations
goals = [(-1.18, -0.95), (0.2, -0.95)]

# Load/Save Potential Field Arrays
load_from_csv = True
save_to_csv = False

# Number of iterations until code stops
iterations = 900 #900, 725

# Evader parameters
avoid_pursuers = True # Evader ignores pursuers if False
evader_speed_limit = 0.95

# Pursuer parameters
# Joseph Note - link chase_evader variable to the difference between uninformed search and directed A* once evader is found
chase_evader = False # Pursuers slowly drive upwards if False
pursuer_speed_limit = 0.33

# Figure saving
save_figures = False # Save pngs of active figures
save_figure_iterations = [175] # Iteration numbers to save figures
keep_figures_open = True # Keep code running so figures don't close

#### END VARIABLES ####



# Set up Robotarium object
pursuers = 3
evaders = 1
N = pursuers + evaders

initial_pursuers = np.array([
    [-1.18,  0.20, -1.4], 
    [-0.85, -0.85, 0.2], 
    [np.pi/2, np.pi/2, 0]])
#initial_pursuers = np.array([
#    [-1.18,  1.5, -1.4], 
#    [-0.85, -0.85, 0.2], 
#    [np.pi/2, np.pi/2, 0]])
initial_evaders = np.array([
    [1.45], 
    [0.2], 
    [np.pi]])

# y-coord
# x-coord
# ??? coord
patrol_goals = np.array([
    [16, 16, 48],
    [52, 87, 17]
])

patrol_returns = np.array([
    [48, 48, 48],
    [52, 17, 87]
])

initial_conditions = np.hstack((initial_pursuers, initial_evaders))
r = robotarium.Robotarium(number_of_robots=N, show_figure=(1 in active_figures), initial_conditions=initial_conditions)

# Plot the iteration and time in the lower left
iteration_caption = 'Iteration: 0'
start_time = datetime.now()
time_caption = 'Time Elapsed: 0.00'
iteration_label = plt.text(-1.5, -0.84, iteration_caption, fontsize=9, color='r', fontweight='bold')
time_label = plt.text(-1.5, -0.94, time_caption, fontsize=9, color='r', fontweight='bold')

# Import and scale the map image appropriately
# Define the path to the map image
map_img_path = os.path.join(os.path.dirname(__file__), 'MRS_Map2.png')
map_img = Image.open(map_img_path)
x_img = np.linspace(-1.6, 1.6, map_img.size[0])
y_img = np.linspace(1.0, -1.0, map_img.size[1])
plt.imshow(map_img, extent=[x_img.min(), x_img.max(), y_img.min(), y_img.max()])

# Make a binary array of 0 or 1 based on the image being white or not
map_np = np.array(map_img)
map = np.all(map_np == 255, axis=2).astype(int) # 0 is wall, 1 is free space


# Draw a semicircle on figure 1 at the locations of the goals
if 1 in active_figures:
    for goal in goals:
        plt.figure(1)
        semicircle = Wedge(goal, 0.25, 0, 180, color='g', alpha=0.25)
        plt.gca().add_patch(semicircle)

# Plot Original Map
if 2 in active_figures:
    plt.figure(num=2)
    graph = plt.imshow(map, cmap='gray')
    plt.title('Original Map Graph')
    plt.show()

# Scale map and inflate obstacles
maze = map
scaled_maze = maze[::scale, ::scale]
if 3 in active_figures:
    plt.figure(num=3)
    graph = plt.imshow(scaled_maze, cmap='gray')
    plt.title('Scaled Maze')
    plt.show()


def inflate_walls(grid, radius):
    """Expands walls by a given radius to account for the robot's size."""
    struct = np.ones((2 * radius + 1, 2 * radius + 1))  # Create a square kernel
    inflated_grid = binary_dilation(grid == 0, structure=struct).astype(int)
    return np.where(inflated_grid, 0, 1)  # Convert back to 0 (wall) and 1 (free space)

# Define A* algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 1:
                temp_g_score = g_score[current] + 1
                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None  # No path found, code currently assumes path found
# repulsive was 0.5
def potential_field(grid, goals, repulsive_scale=0.75, attractive_scale=1.0, influence_radius=8):
    rows, cols = grid.shape
    depth = len(goals)
    astar_attractive_potential = np.zeros((rows, cols, depth))
    wall_repulsive_potential = np.zeros((rows, cols))
    wall_raising_potential = np.zeros((rows, cols))
    # Calculate attractive potential
    for r in range(rows):
        print ("Deleting System32: ", int(r/rows*100), "%")
        for c in range(cols):
            if grid[r, c] == 1:
                for goal_index, goal in enumerate(goals):
                    dist_to_goal = len(astar(grid, (r, c), goal))
                    astar_attractive_potential[r, c, goal_index] += attractive_scale * dist_to_goal
    # Calculate repulsive potential
    for r in range(rows):
        print("Deleting System32 Again: ", int(r / rows * 100), "%")
        for c in range(cols):
            if grid[r, c] == 0:  # Wall
                # Find the closest open space
                min_dist = float('inf')
                closest_potential = 0
                # Could reduce loading time by only checking a certain radius around the wall
                for i in range(rows):
                    for j in range(cols):
                        if grid[i, j] == 1:  # Open space
                            dist = np.sqrt((r - i) ** 2 + (c - j) ** 2)
                            #wall_repulsive_potential[i, j] += repulsive_scale / dist
                            if dist <= influence_radius:
                                wall_repulsive_potential[i, j] += repulsive_scale * (1 + np.cos(np.pi * dist / influence_radius)) / 2
                            if dist < min_dist:
                                min_dist = dist
                                closest_potential = np.min(astar_attractive_potential[i, j, :]) + 93
                                wall_raising_potential[r, c] = closest_potential
    return astar_attractive_potential, wall_repulsive_potential, wall_raising_potential
    
def avoid_pursuers_potential_field(pursuer_loc, repulsive_scale=30.0, influence_radius=20):
    rows, cols = pursuer_loc.shape
    potential = np.zeros((rows, cols))
    # Calculate repulsive potential
    for r in range(rows):
        for c in range(cols):
            if pursuer_loc[r, c] == 0:
                for i in range(max(0, r - influence_radius), min(rows, r + influence_radius + 1)):
                    for j in range(max(0, c - influence_radius), min(cols, c + influence_radius + 1)):
                        dist = np.sqrt((r - i) ** 2 + (c - j) ** 2)
                        if dist != 0 and dist <= influence_radius:
                            #potential[i, j] += repulsive_scale * (np.cos(np.pi * dist / influence_radius))
                            potential[i, j] += repulsive_scale * (1 + np.cos(np.pi * dist / influence_radius)) / 2
    return potential

def get_next_position(potential, current_pos):
    rows, cols = potential.shape
    min_potential = float('inf')
    next_pos = current_pos
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for d in directions:
        neighbor = (current_pos[0] + d[0], current_pos[1] + d[1])
        if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
            if potential[neighbor] < min_potential:
                min_potential = potential[neighbor]
                next_pos = neighbor
    return next_pos


# Inflate walls so pursuers don't collide with them
inflated_maze = inflate_walls(scaled_maze, robot_radius)
if 4 in active_figures:
    plt.figure(num=4)
    graph = plt.imshow(inflated_maze, cmap='gray')
    plt.title('Inflated Maze for Pursuers')
    plt.show()


scaled_goals = [(int((goal[1]*-500 + 500)/scale), int((goal[0]*500 + 800)/scale)) for goal in goals]
csv_folder = os.path.join(os.path.dirname(__file__), 'csv_map2')
os.makedirs(csv_folder, exist_ok=True)
if load_from_csv:
    print("loading")
    astar_attractive_potential = np.zeros((scaled_maze.shape[0], scaled_maze.shape[1], len(scaled_goals)))
    for k in range(len(scaled_goals)):
        astar_attractive_potential[:, :, k] = np.loadtxt(os.path.join(csv_folder, f'astar_attractive_potential_{k}.csv'), delimiter=',')
    wall_repulsive_potential = np.loadtxt(os.path.join(csv_folder, 'wall_repulsive_potential.csv'), delimiter=',')
    wall_raising_potential = np.loadtxt(os.path.join(csv_folder, 'wall_raising_potential.csv'), delimiter=',')
elif save_to_csv:
    print("calculating and saving")
    astar_attractive_potential, wall_repulsive_potential, wall_raising_potential = potential_field(scaled_maze, scaled_goals)
    for k in range(astar_attractive_potential.shape[2]):
        np.savetxt(os.path.join(csv_folder, f'astar_attractive_potential_{k}.csv'), astar_attractive_potential[:, :, k], delimiter=',')
    np.savetxt(os.path.join(csv_folder, 'wall_repulsive_potential.csv'), wall_repulsive_potential, delimiter=',')
    np.savetxt(os.path.join(csv_folder, 'wall_raising_potential.csv'), wall_raising_potential, delimiter=',')
else:
    print("calculating")
    astar_attractive_potential, wall_repulsive_potential, wall_raising_potential = potential_field(scaled_maze, scaled_goals)


static_pot_field = np.min(astar_attractive_potential, axis=2) + wall_repulsive_potential + wall_raising_potential
print("deleted")

if 5 in active_figures:
    depth = astar_attractive_potential.shape[2]
    fig, axes = plt.subplots(depth, 3, num=5, figsize=(17, 5 * depth))
    for k in range(depth):
        # A* Attractive Potential Field for each depth
        axes[k, 0].imshow(astar_attractive_potential[:, :, k], cmap='hot')
        axes[k, 0].set_title(f'A* Attractive Potential Field (Goal {k+1})')
        plt.colorbar(axes[k, 0].imshow(astar_attractive_potential[:, :, k], cmap='hot'), ax=axes[k, 0])
        # Wall Repulsive Potential Field
        if k == 0:
            axes[k, 1].imshow(wall_repulsive_potential, cmap='hot')
            axes[k, 1].set_title('Wall Repulsive Potential Field')
            plt.colorbar(axes[k, 1].imshow(wall_repulsive_potential, cmap='hot'), ax=axes[k, 1])
            # Static Potential Field
            axes[k, 2].imshow(static_pot_field, cmap='hot')
            axes[k, 2].set_title('Static Potential Field')
            plt.colorbar(axes[k, 2].imshow(static_pot_field, cmap='hot'), ax=axes[k, 2])
        elif k == 1:
            # Wall Raising Potential Field
            axes[k, 1].imshow(wall_raising_potential, cmap='hot')
            axes[k, 1].set_title('Wall Raising Potential Field')
            plt.colorbar(axes[k, 1].imshow(wall_raising_potential, cmap='hot'), ax=axes[k, 1])
            fig.delaxes(axes[k, 2])
        else:
            # Remove empty plots
            fig.delaxes(axes[k, 1])
            fig.delaxes(axes[k, 2])
    plt.show()

if 6 in active_figures:
    # Plot 3 dimension colormap of potential field
    fig = plt.figure(num=6)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(static_pot_field.shape[1]), np.arange(static_pot_field.shape[0]))
    surf = ax.plot_surface(X, Y, static_pot_field, cmap='hot')
    ax.view_init(elev=-40, azim=109, roll=180) #-24 > -40
    ax.set_zlim(ax.get_zlim()[::-1])  # Invert the z-axis
    plt.title('3D Static Potential Field')
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()



# Grab tools for converting to single-integrator dynamics and ensuring safety
uni_barrier_cert = create_unicycle_barrier_certificate_with_boundary()
si_to_uni_dyn = create_si_to_uni_dynamics(linear_velocity_gain=0.5, angular_velocity_limit=np.pi/2)

# Gains for the transformation from single-integrator to unicycle dynamics
formation_control_gain = 10

# Initialize robots
xuni = r.get_poses()
x = xuni[:2, :]
r.set_velocities(np.arange(N), np.zeros((2, N)))
r.step()

# Initialize video
if video_flag:
    videos_folder = os.path.join(os.path.dirname(__file__), 'videos')
    os.makedirs(videos_folder, exist_ok=True)
    video_filename = f'MRS_Pur_Eva_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
    video_path = os.path.join(videos_folder, video_filename)
    vid = imageio.get_writer(video_path, fps=fps, codec="libx264")


# Evader needs some sort of logic that makes it loop around the pursuers
# Maybe check what move it wanted to make on the static map, and
# if the dynamic map move is in the opposite direction, then check for the
# closest walls to move away from or for the second best move

dx = np.zeros((2, N))
# Iterate for the previously specified number of iterations
for t in range(iterations):
    x = r.get_poses()
    # Take robots current positions
    eva_loc = x[:2, pursuers]
    pur_loc = x[:2, :pursuers]

    #only do this every 10 iterations
    if t % 10 == 0:
        # The x coordinate is the column and the y coordinate is the row
        eva_loc_scaled = (int((eva_loc[1]*-500 + 500)/scale), int((eva_loc[0]*500 + 800)/scale))
        pur_loc_scaled = (np.array((pur_loc[1, :]*-500 + 500)/scale, dtype=int), np.array((pur_loc[0, :]*500 + 800)/scale, dtype=int))

        if avoid_pursuers:
            pursuer_map = np.ones(scaled_maze.shape)
            for i in range(pursuers):
                pursuer_map[pur_loc_scaled[0][i], pur_loc_scaled[1][i]] = 0
            pursuer_pot_field = avoid_pursuers_potential_field(pursuer_map)
            #add evader potential field to pursuer potential field
            dynamic_pot_field = pursuer_pot_field + static_pot_field
            eva_next_pos = get_next_position(dynamic_pot_field, eva_loc_scaled)
            
            if t % 190 == 0 and 7 in active_figures:
                # Create two horizontal subplots for potential fields
                fig, (ax1, ax2) = plt.subplots(1, 2, num=7, figsize=(10, 5))
                ax1.imshow(pursuer_pot_field, cmap='hot')
                ax1.set_title('Pursuer Potential Field')

                ax2.imshow(dynamic_pot_field, cmap='hot')
                ax2.set_title('Dynamic Potential Field')
                plt.show()
            if t % 190 == 0 and 8 in active_figures:
                # Create two horizontal subplots for potential fields
                fig, (ax1, ax2) = plt.subplots(1, 2, num=8, figsize=(10, 5), subplot_kw={'projection': '3d'})
                X, Y = np.meshgrid(np.arange(pursuer_pot_field.shape[1]), np.arange(pursuer_pot_field.shape[0]))
                ax1.plot_surface(X, Y, pursuer_pot_field, cmap='hot')
                ax1.view_init(elev=-40, azim=109, roll=180) #-24 > -40
                ax1.set_zlim(ax1.get_zlim()[::-1])  # Invert the z-axis
                ax1.set_title('Pursuer Potential Field')

                X, Y = np.meshgrid(np.arange(dynamic_pot_field.shape[1]), np.arange(dynamic_pot_field.shape[0]))
                ax2.plot_surface(X, Y, dynamic_pot_field, cmap='hot')
                ax2.view_init(elev=-40, azim=109, roll=180) #-24 > -40
                ax2.set_zlim(ax2.get_zlim()[::-1])  # Invert the z-axis
                ax2.set_title('Dynamic Potential Field')
                plt.show()
        else:
            eva_next_pos = get_next_position(static_pot_field, eva_loc_scaled)

        if chase_evader:
            #Use A* to move towards the evader's position on the scaled maze
            pur_next_pos = np.zeros((2, pursuers))
            pur_vel = np.zeros((2, pursuers))
            for i in range(pursuers):
                pur_loc_r = pur_loc_scaled[0][i]
                pur_loc_c = pur_loc_scaled[1][i]
                pur_loc_scaled_again = (pur_loc_r, pur_loc_c)
                # The pursuers will use an inflated maze to avoid hitting walls
                # since they don't have the same potential field as the evader
                print(eva_loc_scaled)
                path = astar(inflated_maze, pur_loc_scaled_again, eva_loc_scaled)
                if path is not None:
                    #Create an array of pursuer next positions
                    pur_next_pos[0, i] = path[1][0]
                    pur_next_pos[1, i] = path[1][1]
                    pur_vel[0, i] = (pur_next_pos[1, i] - pur_loc_scaled_again[1])/500*scale
                    pur_vel[1, i] = (pur_next_pos[0, i] - pur_loc_scaled_again[0])/-500*scale
                #else:
                #    pur_next_pos[0, i] = pur_loc_scaled_again[0]
                #    pur_next_pos[1, i] = pur_loc_scaled_again[1]
            #if norms is not 0, normalize the velocity vector
            norms = np.linalg.norm(pur_vel, axis=0)
            non_zero_norms = norms != 0
            pur_vel[:, non_zero_norms] = pur_vel[:, non_zero_norms] * pursuer_speed_limit * r.max_linear_velocity / norms[non_zero_norms]
            dx[0, :pursuers] = pur_vel[0]
            dx[1, :pursuers] = pur_vel[1]
        # Search case. Change chase_evader to true if the evader comes within a certain radius of a pursuer
        else:
            # Currently remaining stationary for testing/"if evader is in certain radius"-ness
            #pur_vel = np.zeros((2, pursuers))
            #pur_vel[1, :] = 0.05

            pur_next_pos = np.zeros((2, pursuers))
            pur_vel = np.zeros((2, pursuers))

            dx[0, :pursuers] = pur_vel[0]
            dx[1, :pursuers] = pur_vel[1]
            
            for i in range(pursuers):
                pur_loc_r = pur_loc_scaled[0][i]
                pur_loc_c = pur_loc_scaled[1][i]
                pur_loc_scaled_again = (pur_loc_r, pur_loc_c)

                # Pursuers use same maze as in A* chase, but use goal points for patrol of the space
                patrol_goal_current = (patrol_goals[0][i], patrol_goals[1][i])

                # If a bot reaches its patrol goal, switch to second patrol goal
                if abs(patrol_goal_current[0] - pur_loc_scaled_again[0]) < 5 and abs(patrol_goal_current[1] - pur_loc_scaled_again[1]):
                    temp_1 = patrol_goals[0][i]
                    temp_2 = patrol_goals[1][i]
                    patrol_goals[0][i] = patrol_returns[0][i]
                    patrol_goals[1][i] = patrol_returns[1][i]
                    patrol_returns[0][i] = temp_1
                    patrol_returns[1][i] = temp_2

                path = astar(inflated_maze, pur_loc_scaled_again, patrol_goal_current)

                if path is not None:
                    #Create an array of pursuer next positions
                    pur_next_pos[0, i] = path[1][0]
                    pur_next_pos[1, i] = path[1][1]
                    pur_vel[0, i] = (pur_next_pos[1, i] - pur_loc_scaled_again[1])/500*scale
                    pur_vel[1, i] = (pur_next_pos[0, i] - pur_loc_scaled_again[0])/-500*scale

            # Normalize
            norms = np.linalg.norm(pur_vel, axis=0)
            non_zero_norms = norms != 0
            pur_vel[:, non_zero_norms] = pur_vel[:, non_zero_norms] * pursuer_speed_limit * r.max_linear_velocity / norms[non_zero_norms]
            dx[0, :pursuers] = pur_vel[0]
            dx[1, :pursuers] = pur_vel[1]

            # Change patrol point if previous one has been reached
        

            # "Intruder detected" condition - if any of the pursuers are within a certain distance of the pursuer, all of them begin A* chase
            for i in range(0, pur_loc.shape[1]):
                if np.linalg.norm(pur_loc[:, i] - eva_loc) < 0.75:
                    print("Found them!")
                    chase_evader = True
        
        #convert row, column to x, y
        # Aren note - since the map is scaled, using x and y becomes inaccurate with scale estimations
        #eva_next_loc = (((float(eva_next_pos[1])*scale - 800)/500), ((float(eva_next_pos[0])*scale - 500)/-500))

        # Plot this path on the maze
        #plt.figure(num=6)
        #graph = plt.imshow(scaled_maze, cmap='gray')
        #plt.plot(eva_loc_scaled[1], eva_loc_scaled[0], 'ro')
        #plt.plot(eva_next_pos[1], eva_next_pos[0], 'bo')

        eva_vel = np.zeros((2, 1))
        #path and start are given as row, column
        eva_vel[0] = (eva_next_pos[1] - eva_loc_scaled[1])/500*scale
        eva_vel[1] = (eva_next_pos[0] - eva_loc_scaled[0])/-500*scale
        #increase velocity vector to maximum speed
        eva_vel = eva_vel * r.max_linear_velocity / np.linalg.norm(eva_vel)
        dx[0, pursuers:] = eva_vel[0]
        dx[1, pursuers:] = eva_vel[1]
        #print (dx)


    # plot an arrow pointing to the next position on figure 1
    plt.figure(1)
    if 'arrow' in locals():
        arrow.remove()
    #current robot position is not perfectly in center of model
    arrow = plt.arrow(eva_loc[0] + (eva_vel[0,0]/4), eva_loc[1] + (eva_vel[1,0]/4), eva_vel[0,0], eva_vel[1,0], head_width=0.1, head_length=0.1, fc='b', ec='b')
    #Draw arrows for pursuers in red
    for i in range(pursuers):
        if 'arrow' + str(i) in locals():
            locals()['arrow' + str(i)].remove()
        locals()['arrow' + str(i)] = plt.arrow(pur_loc[0, i] + (pur_vel[0, i]/2), pur_loc[1, i] + (pur_vel[1, i]/2), pur_vel[0, i], pur_vel[1, i], head_width=0.1, head_length=0.1, fc='r', ec='r')

    # Update Iteration and Time marker
    iteration_caption = f'Iteration: {t}'
    time_caption = f'Time Elapsed: {(datetime.now() - start_time).total_seconds():0.2f}'
    iteration_label.set_text(iteration_caption)
    time_label.set_text(time_caption)

    # Avoid actuator errors
    norms = np.linalg.norm(dx, axis=0)
    threshold = evader_speed_limit * r.max_linear_velocity
    to_thresh = norms > threshold
    dx[:, to_thresh] = threshold * dx[:, to_thresh] / norms[to_thresh]

    # Transform the single-integrator dynamics to unicycle dynamics using a provided utility function
    dx_u = si_to_uni_dyn(dx, x)
    dx_u_bc = dx_u
    #dx_u_bc = uni_barrier_cert(dx_u, x)

    # Set velocities of agents 1:N
    r.set_velocities(np.arange(N), dx_u_bc)

    # Update the video
    if video_flag and (t % ipf == 0):
        # Don't ask why this works, just be glad it does
        # Capture the current frame from figure 1
        plt.figure(1)
        plt.draw()
        # Save the figure to a memory buffer instead of a file
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=200)
        buf.seek(0)
        # Convert buffer image to an array and write to video
        img = imageio.imread(buf)
        vid.append_data(img)



    # Save a png of all active figures at specified iterations
    if t in save_figure_iterations and save_figures:
        figures_folder = os.path.join(os.path.dirname(__file__), 'figures_map2')
        os.makedirs(figures_folder, exist_ok=True)
        for fig_num in active_figures:
            plt.figure(fig_num)
            plt.savefig(os.path.join(figures_folder, f'MRS_Pur_Eva_{fig_num}_iter_{t}.png'))

    # Send the previously set velocities to the agents. This function must be called!
    r.step()


if video_flag:
    vid.close()

r.call_at_scripts_end()

# Don't end the code until q is pressed on figure
while keep_figures_open:
    plt.pause(0.1)
    if plt.get_fignums():
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            if plt.waitforbuttonpress():
                plt.close('all')
                break
    else:
        break
