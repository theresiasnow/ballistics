# Plotting functions
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_bullet_trajectory(x_vals, y_vals, z_vals, t_eval):
    """
    Create an animated 3D plot of the bullet trajectory.

    Parameters:
    - x_vals (array): X positions (horizontal distance)
    - y_vals (array): Y positions (vertical distance/height)
    - z_vals (array): Z positions (wind drift)
    - t_eval (array): Time intervals used for the animation frames

    Returns:
    None
    """
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Setting labels
    ax.set_xlabel('X Distance (m)')
    ax.set_ylabel('Y Height (m)')
    ax.set_zlabel('Z Wind Drift (m)')

    # Limits for axes
    ax.set_xlim([0, max(x_vals)])
    ax.set_ylim([min(y_vals), max(y_vals)])

    # Modify the Z-axis limits to avoid identical low and high values
    z_min, z_max = min(z_vals), max(z_vals)
    if z_min == z_max:  # If there's no variation in z-values, add a buffer
        z_min -= 1
        z_max += 1
    ax.set_zlim([z_min, z_max])

    # Plotting an empty line which will be updated in the animation
    line, = ax.plot([], [], [], color='blue', label='Bullet Trajectory')
    scatter = ax.scatter([], [], [], color='red', label='Current Position', s=100)

    # Initialize function for the animation
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        scatter._offsets3d = ([], [], [])
        return line, scatter

    # Update function for the animation
    def update(frame):
        line.set_data(x_vals[:frame], y_vals[:frame])
        line.set_3d_properties(z_vals[:frame])
        scatter._offsets3d = (x_vals[frame:frame + 1], y_vals[frame:frame + 1], z_vals[frame:frame + 1])
        ax.view_init(elev=30, azim=45 + frame * 0.5)  # Adjust elevation and azimuth for better view
        return line, scatter

    # Add legend and show the plot
    ax.legend()
    plt.title("3D Bullet Trajectory Animation")
    plt.grid(True)  # Add grid lines

    # Create the animation using FuncAnimation
    ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=200, repeat=True)
    return ani, plt, fig