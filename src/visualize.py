"""
visualize.py
------------
Plotting and animation utilities for FDTD simulation results.

Author: Ibar Romay (rowmatrix)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
from typing import Optional

from fdtd import FDTDSolver, GridConfig


# --------------------------------------------------------------------------- #
#  Color palette — calibrated for EM field visualization
# --------------------------------------------------------------------------- #
_CMAP_FIELD = "RdBu_r"   # diverging: negative=red, zero=white, positive=blue
_CMAP_AMP   = "inferno"  # amplitude/intensity maps


def _symmetric_clim(data: np.ndarray, percentile: float = 99.5) -> float:
    """Return symmetric color limits based on percentile of |data|."""
    return np.percentile(np.abs(data), percentile) + 1e-10


# --------------------------------------------------------------------------- #
#  Static field snapshot
# --------------------------------------------------------------------------- #
def plot_field(
    solver: FDTDSolver,
    step_index: int = -1,
    field: str = "Ez",
    figsize: tuple = (8, 7),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a single field snapshot from the solver's recorded history.

    Parameters
    ----------
    solver      : Completed FDTDSolver instance.
    step_index  : Index into solver.snapshots (-1 for last).
    field       : Field component name (currently only 'Ez' stored).
    figsize     : Figure size.
    title       : Plot title override.
    save_path   : If set, saves figure to this path.
    show        : Whether to call plt.show().
    """
    cfg = solver.config
    snap = solver.snapshots[step_index]
    t_ns = solver.snapshot_times[step_index] * 1e9

    x = np.linspace(0, cfg.Lx * 1e3, cfg.nx)  # mm
    y = np.linspace(0, cfg.Ly * 1e3, cfg.ny)  # mm

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    clim = _symmetric_clim(snap)
    im = ax.pcolormesh(
        y, x, snap,
        cmap=_CMAP_FIELD,
        vmin=-clim, vmax=clim,
        shading="auto",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Ez  [V/m]", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("y  [mm]", color="white", fontsize=11)
    ax.set_ylabel("x  [mm]", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    _title = title or f"Ez field — t = {t_ns:.2f} ns"
    ax.set_title(_title, color="white", fontsize=13, pad=10)

    # PML indicator bars
    n = cfg.pml_layers
    dx_mm = cfg.dx * 1e3
    dy_mm = cfg.dy * 1e3
    pml_kw = dict(linewidth=0, alpha=0.12, color="yellow")
    for rect in [
        Rectangle((0,            0),            n*dy_mm, cfg.Lx*1e3, **pml_kw),
        Rectangle((cfg.Ly*1e3 - n*dy_mm, 0),   n*dy_mm, cfg.Lx*1e3, **pml_kw),
        Rectangle((0,            0),            cfg.Ly*1e3, n*dx_mm, **pml_kw),
        Rectangle((0, cfg.Lx*1e3 - n*dx_mm),   cfg.Ly*1e3, n*dx_mm, **pml_kw),
    ]:
        ax.add_patch(rect)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig, ax


# --------------------------------------------------------------------------- #
#  Animation
# --------------------------------------------------------------------------- #
def animate_field(
    solver: FDTDSolver,
    interval: int = 40,
    repeat: bool = True,
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
    fps: int = 25,
) -> animation.FuncAnimation:
    """
    Create an animated GIF or MP4 of the Ez field evolution.

    Parameters
    ----------
    solver      : Completed FDTDSolver with recorded snapshots.
    interval    : Delay between frames [ms] for display.
    repeat      : Loop animation.
    save_path   : If set (e.g. 'wave.gif' or 'wave.mp4'), saves animation.
    fps         : Frames per second for saved file.
    """
    cfg = solver.config
    snaps = solver.snapshots
    times = solver.snapshot_times

    x = np.linspace(0, cfg.Lx * 1e3, cfg.nx)
    y = np.linspace(0, cfg.Ly * 1e3, cfg.ny)

    # Pre-compute global color limit across all frames
    all_data = np.stack(snaps)
    clim = _symmetric_clim(all_data, percentile=99.0)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    im = ax.pcolormesh(
        y, x, snaps[0],
        cmap=_CMAP_FIELD,
        vmin=-clim, vmax=clim,
        shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Ez  [V/m]", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("y  [mm]", color="white")
    ax.set_ylabel("x  [mm]", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    title = ax.set_title("Ez  —  t = 0.00 ns", color="white", fontsize=12)
    fig.tight_layout()

    def _update(frame_idx):
        im.set_array(snaps[frame_idx].ravel())
        t_ns = times[frame_idx] * 1e9
        title.set_text(f"Ez  —  t = {t_ns:.2f} ns")
        return im, title

    anim = animation.FuncAnimation(
        fig, _update,
        frames=len(snaps),
        interval=interval,
        repeat=repeat,
        blit=False,
    )

    if save_path:
        p = Path(save_path)
        writer = "pillow" if p.suffix == ".gif" else "ffmpeg"
        anim.save(save_path, writer=writer, fps=fps,
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
        print(f"Saved animation: {save_path}")

    return anim


# --------------------------------------------------------------------------- #
#  Intensity / energy density map
# --------------------------------------------------------------------------- #
def plot_intensity(
    solver: FDTDSolver,
    figsize: tuple = (8, 7),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot time-averaged intensity (|Ez|^2 summed over all recorded snapshots).
    Useful for visualizing standing wave patterns, resonances, or shadow zones.
    """
    cfg = solver.config
    intensity = np.zeros((cfg.nx, cfg.ny))
    for snap in solver.snapshots:
        intensity += snap**2
    intensity /= len(solver.snapshots)

    x = np.linspace(0, cfg.Lx * 1e3, cfg.nx)
    y = np.linspace(0, cfg.Ly * 1e3, cfg.ny)

    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    im = ax.pcolormesh(y, x, intensity, cmap=_CMAP_AMP, shading="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Ez|²  (time-avg)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("y  [mm]", color="white", fontsize=11)
    ax.set_ylabel("x  [mm]", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_title("Time-Averaged Intensity  |Ez|²", color="white", fontsize=13, pad=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig, ax