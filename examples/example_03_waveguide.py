"""
example_03_waveguide.py
------------------------
EM wave propagation in a 2D parallel-plate waveguide with PEC walls.

Demonstrates:
    - PEC walls via high conductivity (sigma >> 1)
    - Guided mode propagation above cutoff frequency
    - Standing wave pattern across the guide cross-section

Physical relevance: foundational for microwave waveguide design,
cavity resonators, and transmission line theory.

Run:
    python examples/example_03_waveguide.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import matplotlib
matplotlib.use("Agg")

from fdtd import GridConfig, PointSource, MaterialMap, FDTDSolver
from visualize import plot_field, plot_intensity

# --------------------------------------------------------------------------- #
#  Grid
# --------------------------------------------------------------------------- #
cfg = GridConfig(
    nx=60, ny=250,      # guide runs along y (long axis), walls along x (short axis)
    dx=0.5e-3, dy=0.5e-3,
    nt=1200,
    pml_layers=15,
)

# --------------------------------------------------------------------------- #
#  PEC walls: left and right columns (x=0 and x=nx) set to high conductivity
# --------------------------------------------------------------------------- #
WALL = 5  # wall thickness in cells

mat = MaterialMap(cfg.nx, cfg.ny)
mat.add_rectangle(0,           0, WALL,         cfg.ny, sigma=1e7)
mat.add_rectangle(cfg.nx-WALL, 0, cfg.nx,       cfg.ny, sigma=1e7)

# --------------------------------------------------------------------------- #
#  Source: point source near the bottom (y=pml+10), centered in x
# --------------------------------------------------------------------------- #
f0 = 8e9
src = PointSource(
    ix=cfg.nx // 2,
    iy=cfg.pml_layers + 10,
    frequency=f0,
    amplitude=1.0,
)

# Compute TE10 cutoff: guide width = nx minus wall cells, along x
a_m  = (cfg.nx - 2 * WALL) * cfg.dx
fc10 = 3e8 / (2 * a_m)

print("=" * 55)
print("  Example 03 — Parallel-Plate Waveguide")
print("=" * 55)
print(f"  Guide width : {a_m*1e3:.1f} mm")
print(f"  TE10 cutoff : {fc10/1e9:.2f} GHz")
print(f"  Source freq : {f0/1e9:.1f} GHz  ", end="")
print("(propagating)" if f0 > fc10 else "(below cutoff — evanescent!)")
print(f"  PML         : {cfg.pml_layers} layers")
print()

solver = FDTDSolver(cfg, materials=mat, sources=[src], record_every=5)
solver.run(verbose=True)

plot_field(solver, step_index=-1,
           title=f"Waveguide — Ez  (f={f0/1e9:.0f} GHz, fc={fc10/1e9:.2f} GHz)",
           save_path="assets/ex03_snapshot.png",
           show=False)

plot_intensity(solver,
               save_path="assets/ex03_intensity.png",
               show=False)

print("\nDone. See assets/ for outputs.")