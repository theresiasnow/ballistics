# Plotting functions
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _segment_speeds(x_vals, y_vals, z_vals, t_eval):
    """Per-sample speed (m/s) derived from position deltas, length == inputs."""
    pos = np.column_stack([x_vals, y_vals, z_vals])
    dt = np.diff(t_eval)
    dt[dt == 0] = np.nan
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    # Pad to the same length as the inputs (repeat the last segment speed).
    return np.concatenate([seg, seg[-1:]]) if len(seg) else np.zeros(len(x_vals))


def animate_bullet_trajectory(
        x_vals, y_vals, z_vals, t_eval,
        velocities=None,
        rotate=True,
        save_mp4=False,
        save_gif=False,
        filename_prefix="bullet_traj",
        zero_point=None,
):
    """
    Animated 3D bullet trajectory, colored by velocity with reference context.

    The line is colored fast (red) -> slow (blue). The trajectory is also
    projected onto the back wall (drop vs range) and the floor (drift vs range)
    as faint shadows, and the muzzle, line-of-sight crossing (zero), and target
    are marked. A small HUD shows time / distance / velocity, and the camera can
    slowly orbit.

    Parameters:
    - x_vals (array): X positions, downrange distance (m)
    - y_vals (array): Y positions, height vs line of sight (m)
    - z_vals (array): Z positions, lateral wind drift (m)
    - t_eval (array): Time of each sample (s)
    - velocities (array, optional): Speed (m/s) at each sample. If omitted, it is
      derived from the position/time deltas.
    - rotate (bool): Slowly orbit the camera during the animation.
    - save_mp4 / save_gif (bool): Also save the animation to disk.
    - filename_prefix (str): Base filename for save_mp4 / save_gif.
    - zero_point (float | None): Expected zero distance (m); if given, marked and
      labelled instead of the auto-detected line-of-sight crossing.

    Returns:
    (ani, plt, fig)
    """
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    z_vals = np.asarray(z_vals, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    speeds = np.asarray(velocities, dtype=float) if velocities is not None \
        else _segment_speeds(x_vals, y_vals, z_vals, t_eval)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Downrange (m)", labelpad=15)
    ax.set_ylabel("Drop / rise (m)", labelpad=15)
    ax.set_zlabel("Wind drift (m)", labelpad=15)
    ax.set_title("3D Bullet Trajectory (colored by velocity)", pad=20)

    # Axis limits with a little padding so markers/labels are not clipped.
    def _lims(a, pad_frac=0.08, min_pad=0.02):
        lo, hi = float(np.min(a)), float(np.max(a))
        pad = max((hi - lo) * pad_frac, min_pad)
        return lo - pad, hi + pad

    x_lo, x_hi = 0.0, float(np.max(x_vals))
    y_lo, y_hi = _lims(y_vals)
    z_lo, z_hi = _lims(z_vals)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.view_init(elev=22, azim=45)

    # Color normalization across the whole flight (so colors are stable).
    cmap = plt.get_cmap("turbo")
    vmin, vmax = float(np.min(speeds)), float(np.max(speeds))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    norm = plt.Normalize(vmin=vmax, vmax=vmin)  # reversed: fast=warm, slow=cool

    def _colored_segments(n):
        """Line3DCollection for the first n points, colored by speed."""
        pts = np.column_stack([x_vals[:n], y_vals[:n], z_vals[:n]]).reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, cmap=cmap, norm=norm)
        lc.set_array(speeds[: max(n - 1, 0)])
        lc.set_linewidth(2.5)
        return lc

    # Faint shadow projections onto the back wall (drop) and floor (drift).
    ax.plot(x_vals, y_vals, zs=z_lo, zdir="z", color="0.6", lw=1, alpha=0.5)
    ax.plot(x_vals, np.full_like(x_vals, y_lo), z_vals, color="0.6", lw=1, alpha=0.5)

    # Reference markers: muzzle, zero, target.
    ax.scatter(x_vals[0], y_vals[0], z_vals[0], color="black", s=60, marker="^", label="Muzzle")
    ax.scatter(x_vals[-1], y_vals[-1], z_vals[-1], color="green", s=80, marker="*", label="Target")
    if zero_point is not None:
        zi = int(np.argmin(np.abs(x_vals - zero_point)))
        ax.scatter(x_vals[zi], y_vals[zi], z_vals[zi], color="orange", s=70, marker="o",
                   label=f"Zero ({zero_point:.0f} m)")
    else:
        # Zero = where the bullet falls back through the line of sight (y == 0).
        crossings = np.where(np.diff(np.sign(y_vals)) != 0)[0]
        if crossings.size:
            zi = int(crossings[-1])
            ax.scatter(x_vals[zi], 0.0, z_vals[zi], color="orange", s=70, marker="o",
                       label=f"Zero (~{x_vals[zi]:.0f} m)")

    # Colorbar for the velocity scale.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(speeds)
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1).set_label("Velocity (m/s)")

    # HUD readout (time / distance / velocity).
    hud_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=11,
                         fontfamily="monospace")

    line_holder = {"lc": None}
    marker = ax.scatter([], [], [], color="red", s=90, label="Bullet")

    def init():
        hud_text.set_text("")
        return (marker, hud_text)

    def update(frame):
        n = frame + 1
        if line_holder["lc"] is not None:
            line_holder["lc"].remove()
        lc = _colored_segments(n)
        ax.add_collection3d(lc)
        line_holder["lc"] = lc
        marker._offsets3d = (x_vals[frame:n], y_vals[frame:n], z_vals[frame:n])
        if rotate:
            ax.view_init(elev=22, azim=45 + frame * 0.4)  # slow orbit; needs blit=False

        label = f"Time: {t_eval[frame]:.2f} s\nDistance: {x_vals[frame]:.1f} m"
        label += f"\nVelocity: {speeds[frame]:.0f} m/s"
        hud_text.set_text(label)
        return (marker, lc, hud_text)

    ax.legend(loc="upper left")

    # blit=False is required: the orbiting camera (view_init) repaints the whole
    # axes each frame, which blitting would cache and break.
    ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init,
                        blit=False, interval=50, repeat=True)

    if save_mp4:
        ani.save(f"{filename_prefix}.mp4", writer=FFMpegWriter(fps=30, bitrate=2400))
        print(f"Saved MP4 as {filename_prefix}.mp4")
    if save_gif:
        ani.save(f"{filename_prefix}.gif", writer=PillowWriter(fps=10))
        print(f"Saved GIF as {filename_prefix}.gif")

    return ani, plt, fig
