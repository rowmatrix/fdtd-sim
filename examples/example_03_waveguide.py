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
    nx=250, ny=120,
    dx=0.5e-3, dy=0.5e-3,
    nt=1000,
    pml_layers=15,
)

# --------------------------------------------------------------------------- #
#  PEC walls: top and bottom rows set to very high conductivity
# --------------------------------------------------------------------------- #
WALL = 5  # wall thickness in cells

mat = MaterialMap(cfg.nx, cfg.ny)
mat.add_rectangle(0, 0,           cfg.nx, WALL,         sigma=1e7)
mat.add_rectangle(0, cfg.ny-WALL, cfg.nx, cfg.ny,       sigma=1e7)

# --------------------------------------------------------------------------- #
#  Source: point source near the left end, centered vertically in the guide
# --------------------------------------------------------------------------- #
f0 = 8e9
src = PointSource(
    ix=cfg.pml_layers + 10,
    iy=cfg.ny // 2,
    frequency=f0,
    amplitude=1.0,
)

# Compute TE10 cutoff for this guide
a_m  = (cfg.ny - 2 * WALL) * cfg.dy
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