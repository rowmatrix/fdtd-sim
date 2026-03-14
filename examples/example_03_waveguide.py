"""
example_03_waveguide.py
------------------------
EM wave propagation through a 2D parallel-plate waveguide (PEC walls).

Demonstrates:
    - PEC (perfect electric conductor) boundary conditions via sigma → ∞
    - Guided mode cutoff: only modes with lambda > 2a/m propagate
    - Standing wave pattern across the guide cross-section

Physical relevance: foundational for understanding waveguide design,
microwave cavity resonators, and transmission line theory.

Run:
    python examples/example_03_waveguide.py
"""

import sys
sys.path.insert(0, "../src")

import numpy as np
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
WALL_THICKNESS = 5  # cells

mat = MaterialMap(cfg.nx, cfg.ny)
mat.add_rectangle(
    x0=0, y0=0,
    x1=cfg.nx, y1=WALL_THICKNESS,
    sigma=1e7,  # ~PEC
)
mat.add_rectangle(
    x0=0, y0=cfg.ny - WALL_THICKNESS,
    x1=cfg.nx, y1=cfg.ny,
    sigma=1e7,
)

# --------------------------------------------------------------------------- #
#  Source: point source near the left end, centered vertically
# --------------------------------------------------------------------------- #
f0 = 8e9
src = PointSource(
    ix=cfg.pml_layers + 10,
    iy=cfg.ny // 2,
    frequency=f0,
    amplitude=1.0,
)

# Waveguide parameters
a_m = (cfg.ny - 2 * WALL_THICKNESS) * cfg.dy  # guide width in meters
fc_TE1 = 3e8 / (2 * a_m)  # TE10 cutoff frequency

print("=" * 55)
print("  Example 03 — Parallel-Plate Waveguide")
print("=" * 55)
print(f"  Guide width   : {a_m*1e3:.1f} mm")
print(f"  TE10 cutoff   : {fc_TE1/1e9:.2f} GHz")
print(f"  Source freq   : {f0/1e9:.1f} GHz  ", end="")
print("(propagating)" if f0 > fc_TE1 else "(below cutoff — evanescent!)")
print()

solver = FDTDSolver(cfg, materials=mat, sources=[src], record_every=5)
solver.run(verbose=True)

plot_field(solver, step_index=-1,
           title=f"Waveguide — Ez  (f0={f0/1e9:.0f} GHz, fc={fc_TE1/1e9:.2f} GHz)",
           save_path="assets/ex03_snapshot.png",
           show=False)

plot_intensity(solver,
               save_path="assets/ex03_intensity.png",
               show=False)

print("\nDone. See assets/ for outputs.")
