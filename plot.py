import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


def animate_bullet_trajectory(
        x_vals, y_vals, z_vals, t_eval,
        velocities=None,
        rotate=False,
        save_mp4=False,
        save_gif=False,
        filename_prefix="bullet_traj",
        zero_point=None  # NEW: expected zero distance in meters
):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Distance (m)', labelpad=15)
    ax.set_ylabel('Z Wind Drift (m)', labelpad=15)
    ax.set_zlabel('Y Height (m)', labelpad=15)

    ax.set_xlim([0, max(x_vals)])

    z_min, z_max = min(z_vals), max(z_vals)
    if z_min == z_max:
        z_min -= 0.01
        z_max += 0.01
    ax.set_ylim([z_min, z_max])

    y_min, y_max = min(y_vals), max(y_vals)
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    ax.set_zlim([y_min, y_max])

    ax.view_init(elev=25, azim=-60)

    line, = ax.plot([], [], [], color='blue', linewidth=2, label='Bullet Trajectory')
    scatter = ax.scatter([], [], [], color='red', s=80, label='Current Position')
    ax.scatter(x_vals[-1], z_vals[-1], y_vals[-1], color='green', s=60, label='Impact')

    # Label HUD
    hud_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, fontfamily='monospace')

    # === ZERO POINT marker ===
    if zero_point is not None:
        zero_index = np.argmin(np.abs(x_vals - zero_point))
        x_zero = x_vals[zero_index]
        y_zero = y_vals[zero_index]
        z_zero = z_vals[zero_index]
        ax.text(x_zero, z_zero, y_zero + 0.02, "Target", color='orange', fontsize=9)

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        scatter._offsets3d = ([], [], [])
        hud_text.set_text("")
        return line, scatter, hud_text

    def update(frame):
        line.set_data(x_vals[:frame], z_vals[:frame])
        line.set_3d_properties(y_vals[:frame])
        scatter._offsets3d = (x_vals[frame:frame + 1], z_vals[frame:frame + 1], y_vals[frame:frame + 1])
        if rotate:
            ax.view_init(elev=25, azim=-60 + frame * 0.4)

        t = t_eval[frame]
        dist = x_vals[frame]
        vel = velocities[frame] if velocities is not None else None

        label = f"Time: {t:.2f} s\nDistance: {dist:.1f} m"
        if vel is not None:
            label += f"\nVelocity: {vel:.1f} m/s"
        hud_text.set_text(label)

        return line, scatter, hud_text

    ax.legend(loc='upper right')
    plt.title("3D Bullet Trajectory Animation", pad=20)
    plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init,
                        blit=False, interval=60, repeat=True)

    if save_mp4:
        writer = FFMpegWriter(fps=30, bitrate=2400)
        ani.save(f"{filename_prefix}.mp4", writer=writer)
        print(f"Saved MP4 as {filename_prefix}.mp4")

    if save_gif:
        writer = PillowWriter(fps=10)
        ani.save(f"{filename_prefix}.gif", writer=writer)
        print(f"Saved GIF as {filename_prefix}.gif")

    return ani, plt, fig
