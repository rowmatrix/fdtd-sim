"""
example_02_dielectric_scattering.py
-------------------------------------
A plane wave (approximated by a line source) impinging on a dielectric cylinder.

Demonstrates:
    - MaterialMap: adding a circular dielectric scatterer (eps_r = 4)
    - Refraction, internal reflections, and shadow zone
    - Time-averaged intensity map to reveal the scattered field pattern

Physical analogy: similar to a signal scattering off a radome or dielectric lens
in RF/microwave systems — directly relevant to antenna and waveguide design.

Run:
    python examples/example_02_dielectric_scattering.py
"""

import sys
sys.path.insert(0, "../src")

from fdtd import GridConfig, PointSource, MaterialMap, FDTDSolver
from visualize import plot_field, plot_intensity, animate_field

# --------------------------------------------------------------------------- #
#  Grid
# --------------------------------------------------------------------------- #
cfg = GridConfig(
    nx=200, ny=200,
    dx=0.5e-3, dy=0.5e-3,
    nt=800,
    pml_layers=20,
)

# --------------------------------------------------------------------------- #
#  Materials: dielectric cylinder (eps_r = 4.0, like FR4 PCB material)
# --------------------------------------------------------------------------- #
mat = MaterialMap(cfg.nx, cfg.ny)
mat.add_circle(
    cx=cfg.nx // 2,
    cy=cfg.ny // 2,
    radius=30,
    eps_r=4.0,    # relative permittivity of FR4/glass
)

# --------------------------------------------------------------------------- #
#  Source: line of point sources on the left side → approximates plane wave
# --------------------------------------------------------------------------- #
f0 = 10e9  # 10 GHz
sources = [
    PointSource(
        ix=cfg.pml_layers + 5,
        iy=j,
        frequency=f0,
        amplitude=1.0,
    )
    for j in range(cfg.pml_layers + 5, cfg.ny - cfg.pml_layers - 5)
]

print("=" * 55)
print("  Example 02 — Dielectric Cylinder Scattering")
print("=" * 55)
print(f"  Scatterer : eps_r = 4.0, radius = 30 cells (15 mm)")
print(f"  Source    : line source → plane wave approximation")
print(f"  f0        : {f0/1e9:.0f} GHz")
print(f"  lambda    : {3e8/f0*1e3:.1f} mm in free space")
print()

solver = FDTDSolver(cfg, materials=mat, sources=sources, record_every=8)
solver.run(verbose=True)

plot_field(solver, step_index=-1,
           title="Dielectric Scattering — Ez (t_final)",
           save_path="assets/ex02_snapshot.png",
           show=False)

plot_intensity(solver,
               save_path="assets/ex02_intensity.png",
               show=False)

print("\nDone. See assets/ for outputs.")
