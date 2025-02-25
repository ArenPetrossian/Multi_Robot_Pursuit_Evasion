import numpy as np
from PIL import Image
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import cv2

import matplotlib.pyplot as plt
import rps.robotarium as robotarium
from datetime import datetime
import os
from scipy.ndimage import binary_dilation
import heapq

# Set up Robotarium object
pursuers = 3
evaders = 1
N = pursuers + evaders

initial_pursuers = np.zeros((3, pursuers))
initial_evaders = np.zeros((3, evaders))
for i in range(pursuers):
    initial_pursuers[0, i] = -1.35 + (i % 2) * 0.3
    initial_pursuers[1, i] = -0.77 + (i // 2) * 0.25
    initial_pursuers[2, i] = np.pi / 2
for i in range(evaders):
    initial_evaders[0, i] = 1.45 - (i % 3) * 0.1
    initial_evaders[1, i] = 0.2 + (i // 3) * 0.1
    initial_evaders[2, i] = np.pi

initial_conditions = np.hstack((initial_pursuers, initial_evaders))
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, initial_conditions=initial_conditions)

video_flag = False  # Change to True to record video
fps = 10  # frames per second
ipf = 30  # iterations per frame

# Plot the iteration and time in the lower left
iteration_caption = 'Iteration: 0'
start_time = datetime.now()
time_caption = 'Time Elapsed: 0.00'
iteration_label = plt.text(-1.5, -0.84, iteration_caption, fontsize=9, color='r', fontweight='bold')
time_label = plt.text(-1.5, -0.94, time_caption, fontsize=9, color='r', fontweight='bold')

# Import and scale the map image appropriately
# Define the path to the map image
map_img_path = os.path.join(os.path.dirname(__file__), 'MRS_Map.png')
map_img = Image.open(map_img_path)
x_img = np.linspace(-1.6, 1.6, map_img.size[0])
y_img = np.linspace(1.0, -1.0, map_img.size[1])
plt.imshow(map_img, extent=[x_img.min(), x_img.max(), y_img.min(), y_img.max()])

# Make a binary array of 0 or 1 based on the image being white or not
map_np = np.array(map_img)
map = np.all(map_np == 255, axis=2).astype(int) # 0 is wall, 1 is free space

# Plot Original Map
plt.figure(num=2)
graph = plt.imshow(map, cmap='gray')
plt.title('Original Map Graph')
plt.show()

# Scale map and inflate obstacles
maze = map
scale = 15
scaled_maze = maze[::scale, ::scale]
plt.figure(num=3)
graph = plt.imshow(scaled_maze, cmap='gray')
plt.title('Scaled Maze')
plt.show()

def potential_field(grid, goal, repulsive_scale=2.0, attractive_scale=1.0):
    rows, cols = grid.shape
    potential = np.zeros((rows, cols))
    # Calculate repulsive potential
    for r in range(rows):
        print ("Deleting System32: ", int(r/rows*100), "%")
        #print (r, rows)
        for c in range(cols):
            if grid[r, c] == 0:  # Wall
                for i in range(rows):
                    for j in range(cols):
                        dist = np.sqrt((r - i) ** 2 + (c - j) ** 2)
                        if dist != 0:
                            potential[i, j] += repulsive_scale / dist
    # Calculate attractive potential
    for r in range(rows):
        for c in range(cols):
            dist_to_goal = np.sqrt((r - goal[0]) ** 2 + (c - goal[1]) ** 2)
            potential[r, c] += attractive_scale * dist_to_goal
    return potential

def avoid_pursuers_potential_field(pursuer_loc, repulsive_scale=4.0):
    rows, cols = pursuer_loc.shape
    potential = np.zeros((rows, cols))
    # Calculate repulsive potential
    for r in range(rows):
        for c in range(cols):
            if pursuer_loc[r, c] == 0:
                for i in range(rows):
                    for j in range(cols):
                        dist = np.sqrt((r - i) ** 2 + (c - j) ** 2)
                        if dist != 0:
                            potential[i, j] += repulsive_scale / dist
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

goal = (-1.2, -0.9)
goal = (int((goal[1]*-500 + 500)/scale), int((goal[0]*500 + 800)/scale))

# Calculate static map potential field
print("loading")
static_pot_field = potential_field(scaled_maze, goal)
print("deleted")
plt.figure(num=4)
graph = plt.imshow(static_pot_field, cmap='hot')
plt.title('Static Potential Field')
plt.show()

# Plot 3 dimension colormap of potential field
fig = plt.figure(num=5)
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(static_pot_field.shape[1]), np.arange(static_pot_field.shape[0]))
ax.plot_surface(X, Y, static_pot_field, cmap='hot')
ax.view_init(elev=-24, azim=109, roll=180)
ax.set_zlim(ax.get_zlim()[::-1])  # Invert the z-axis
plt.title('3D Static Potential Field')
plt.show()



# Set up constants for experiment
iterations = 500

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
    vid = cv2.VideoWriter('MRS_Pur_Eva.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (800, 500))

dx = np.zeros((2, N))
# Iterate for the previously specified number of iterations
for t in range(iterations):
    x = r.get_poses()

    #only do this every 15 iterations
    if t % 15 == 0:
        # Take robots current positions
        eva_loc = x[:2, pursuers]
        pur_loc = x[:2, :pursuers]

        # The x coordinate is the column and the y coordinate is the row
        eva_loc_scaled = (int((eva_loc[1]*-500 + 500)/scale), int((eva_loc[0]*500 + 800)/scale))
        pur_loc_scaled = (np.array((pur_loc[1, :]*-500 + 500)/scale, dtype=int), np.array((pur_loc[0, :]*500 + 800)/scale, dtype=int))

        avoid_pursuers = True
        if avoid_pursuers:
            pursuer_map = np.ones(scaled_maze.shape)
            for i in range(pursuers):
                pursuer_map[pur_loc_scaled[0][i], pur_loc_scaled[1][i]] = 0
            pursuer_pot_field = avoid_pursuers_potential_field(pursuer_map, repulsive_scale=50.0)
            #add evader potential field to pursuer potential field
            dynamic_pot_field = pursuer_pot_field + static_pot_field
            eva_next_pos = get_next_position(dynamic_pot_field, eva_loc_scaled)
            if t % 30 == 0:
                # Create three horizontal subplots for potential fields
                fig = plt.figure(num=6, figsize=(10, 5))
                #ax1 = fig.add_subplot(131, projection='3d')
                #X, Y = np.meshgrid(np.arange(static_pot_field.shape[1]), np.arange(static_pot_field.shape[0]))
                #ax1.plot_surface(X, Y, static_pot_field, cmap='hot')
                #ax1.view_init(elev=-24, azim=109, roll=180)
                #ax1.set_zlim(ax1.get_zlim()[::-1])  # Invert the z-axis
                #ax1.set_title('Static Potential Field')

                ax2 = fig.add_subplot(121, projection='3d')
                X, Y = np.meshgrid(np.arange(pursuer_pot_field.shape[1]), np.arange(pursuer_pot_field.shape[0]))
                ax2.plot_surface(X, Y, pursuer_pot_field, cmap='hot')
                ax2.view_init(elev=-24, azim=109, roll=180)
                ax2.set_zlim(ax2.get_zlim()[::-1])  # Invert the z-axis
                ax2.set_title('Pursuer Potential Field')

                ax3 = fig.add_subplot(122, projection='3d')
                X, Y = np.meshgrid(np.arange(dynamic_pot_field.shape[1]), np.arange(dynamic_pot_field.shape[0]))
                ax3.plot_surface(X, Y, dynamic_pot_field, cmap='hot')
                ax3.view_init(elev=-24, azim=109, roll=180)
                ax3.set_zlim(ax3.get_zlim()[::-1])  # Invert the z-axis
                ax3.set_title('Dynamic Potential Field')
                plt.show()
        else:
            eva_next_pos = get_next_position(static_pot_field, eva_loc_scaled)

        #convert row, column to x, y
        # Aren note - since the map is scaled, using x and y becomes inaccurate with scale estimations
        #eva_next_loc = (((float(eva_next_pos[1])*scale - 800)/500), ((float(eva_next_pos[0])*scale - 500)/-500))

        # Plot this path on the maze
        #plt.figure(num=6)
        #graph = plt.imshow(scaled_maze, cmap='gray')
        #plt.plot(eva_loc_scaled[1], eva_loc_scaled[0], 'ro')
        #plt.plot(eva_next_pos[1], eva_next_pos[0], 'bo')

        eva_vel = np.zeros((2, 1))
        #eva_vel = eva_next_loc - eva_loc
        #path and start are given as row, column
        eva_vel[0] = (eva_next_pos[1] - eva_loc_scaled[1])/500*scale
        eva_vel[1] = (eva_next_pos[0] - eva_loc_scaled[0])/-500*scale
        #increase velocity vector to maximum speed
        eva_vel = eva_vel * r.max_linear_velocity / np.linalg.norm(eva_vel)
        #print (start)
        #print (eva_next_pos)
        #print (eva_vel)
        #print (np.linalg.norm(eva_vel))
        #print (r.max_linear_velocity)
        #eva_speed = np.linalg.norm(eva_vel)
        dx[0, pursuers:] = eva_vel[0]
        dx[1, pursuers:] = eva_vel[1]
        dx[1, :pursuers] = 0.05
        #print (dx)
        print (eva_vel)
        print (eva_loc)
        # plot an arrow pointing to the next position on figure 1
        plt.figure(1)
        if 'arrow' in locals():
            arrow.remove()
        #current robot position is not perfectly in center of model
        arrow = plt.arrow(eva_loc[0] + (eva_vel[0,0]/4), eva_loc[1] + (eva_vel[1,0]/4), eva_vel[0,0], eva_vel[1,0], head_width=0.1, head_length=0.1, fc='b', ec='b')


    # Update Iteration and Time marker
    iteration_caption = f'Iteration: {t}'
    time_caption = f'Time Elapsed: {(datetime.now() - start_time).total_seconds():0.2f}'
    iteration_label.set_text(iteration_caption)
    time_label.set_text(time_caption)

    # Avoid actuator errors
    norms = np.linalg.norm(dx, axis=0)
    threshold = 3 / 4 * r.max_linear_velocity
    to_thresh = norms > threshold
    dx[:, to_thresh] = threshold * dx[:, to_thresh] / norms[to_thresh]

    # Transform the single-integrator dynamics to unicycle dynamics using a provided utility function
    dx_u = si_to_uni_dyn(dx, x)
    dx_u_bc = uni_barrier_cert(dx_u, x)

    # Set velocities of agents 1:N
    r.set_velocities(np.arange(N), dx_u_bc)

    # Update the video
    if video_flag:
        # Capture the current frame from figure 1
        # Aren note - this doesn't work yet
        plt.figure(1)
        plt.draw()
        frame = np.frombuffer(plt.gcf().canvas.tostring_argb(), dtype=np.uint8)
        frame = frame.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vid.write(frame)

    #if iteration is 200, save all figures
    save_figures = False
    if t == 175 and save_figures:
        plt.figure(1)
        plt.savefig('MRS_Pur_Eva_1.png')
        plt.figure(2)
        plt.savefig('MRS_Pur_Eva_2.png')
        plt.figure(3)
        plt.savefig('MRS_Pur_Eva_3.png')
        plt.figure(4)
        plt.savefig('MRS_Pur_Eva_4.png')
        plt.figure(5)
        plt.savefig('MRS_Pur_Eva_5.png')
        plt.figure(6)
        plt.savefig('MRS_Pur_Eva_6.png')

    # Send the previously set velocities to the agents. This function must be called!
    r.step()

if video_flag:
    vid.release()

r.call_at_scripts_end()

# Don't end the code until q is pressed on a figure (figure 1)
while True:
    plt.pause(0.1)
    if plt.get_fignums():
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            if plt.waitforbuttonpress():
                plt.close('all')
                break
    else:
        break
