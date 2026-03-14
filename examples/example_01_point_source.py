"""
example_01_point_source.py
--------------------------
Sinusoidal point source radiating in free space.

Demonstrates:
    - Basic GridConfig and PointSource setup
    - PML absorption at boundaries
    - Circular wavefront propagation

Run:
    python examples/example_01_point_source.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fdtd import GridConfig, PointSource, FDTDSolver
from visualize import plot_field

# --------------------------------------------------------------------------- #
#  Grid: 200x200 mm, 1 mm cells, 800 steps, thicker PML
# --------------------------------------------------------------------------- #
cfg = GridConfig(
    nx=200, ny=200,
    dx=1e-3, dy=1e-3,
    nt=800,
    pml_layers=25,      # thicker PML = cleaner corner absorption
)

# 3 GHz source at grid center (longer wavelength = more cells/wavelength,
# less numerical dispersion, rounder wavefronts)
f0 = 3e9
src = PointSource(
    ix=cfg.nx // 2,
    iy=cfg.ny // 2,
    frequency=f0,
    amplitude=1.0,
)

print("=" * 55)
print("  Example 01 — Point Source in Free Space")
print("=" * 55)
print(f"  Grid    : {cfg.nx} x {cfg.ny} cells  ({cfg.Lx*1e3:.0f} x {cfg.Ly*1e3:.0f} mm)")
print(f"  dx/dy   : {cfg.dx*1e3:.1f} mm")
print(f"  dt      : {cfg.dt*1e12:.3f} ps")
print(f"  f0      : {f0/1e9:.1f} GHz")
print(f"  lambda  : {3e8/f0*1e3:.1f} mm  ({int(3e8/f0/cfg.dx)} cells/wavelength)")
print(f"  PML     : {cfg.pml_layers} layers")
print(f"  Steps   : {cfg.nt}")
print()

solver = FDTDSolver(cfg, sources=[src], record_every=5)
solver.run(verbose=True)

# --------------------------------------------------------------------------- #
#  Static snapshot
# --------------------------------------------------------------------------- #
plot_field(solver, step_index=-1,
           title="Point Source — Free Space  (t_final)",
           save_path="assets/ex01_snapshot.png",
           show=False)

# --------------------------------------------------------------------------- #
#  Animation — build frame by frame so timestamps render correctly in GIF
# --------------------------------------------------------------------------- #
print("\nBuilding animation frames...")

snaps = solver.snapshots
times = solver.snapshot_times

all_data = np.stack(snaps)
clim = float(np.percentile(np.abs(all_data), 99.5)) + 1e-10

frames_dir = "assets/_frames"
os.makedirs(frames_dir, exist_ok=True)

for i, (snap, t) in enumerate(zip(snaps, times)):
    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.pcolormesh(
        np.linspace(0, cfg.Ly * 1e3, cfg.ny),
        np.linspace(0, cfg.Lx * 1e3, cfg.nx),
        snap,
        cmap="RdBu_r", vmin=-clim, vmax=clim, shading="auto",
    )
    ax.set_title(f"Ez  —  t = {t*1e9:.2f} ns", color="white", fontsize=11)
    ax.set_xlabel("y  [mm]", color="white")
    ax.set_ylabel("x  [mm]", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    fig.tight_layout()
    fig.savefig(f"{frames_dir}/frame_{i:04d}.png", dpi=80,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    if i % 20 == 0:
        print(f"  frame {i+1}/{len(snaps)}")

# Stitch frames into GIF using Pillow
from PIL import Image as PILImage

frame_files = sorted(
    os.path.join(frames_dir, f)
    for f in os.listdir(frames_dir) if f.endswith(".png")
)
pil_frames = [PILImage.open(f).convert("RGB") for f in frame_files]
pil_frames[0].save(
    "assets/ex01_animation.gif",
    save_all=True,
    append_images=pil_frames[1:],
    duration=50,       # ms per frame (~20 fps)
    loop=0,
)

# Clean up temp frames
import shutil
shutil.rmtree(frames_dir)

print(f"\nAnimation saved: assets/ex01_animation.gif  ({len(pil_frames)} frames)")
print("Done. See assets/ for outputs.")