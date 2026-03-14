# fdtd-sim

**2D Finite-Difference Time-Domain (FDTD) electromagnetic field solver — built in Python.**

> *Solving Maxwell's curl equations on a staggered Yee grid, with PML absorbing boundaries, material support, and animated field visualization.*

<p align="center">
  <img src="assets/ex01_animation.gif" width="480" alt="Point source radiating in free space"/>
  <br>
  <em>3 GHz point source radiating circular wavefronts in free space. PML boundaries absorb outgoing waves cleanly.</em>
</p>

---

## What is FDTD?

The Finite-Difference Time-Domain method is a numerical technique for solving Maxwell's equations directly in the time domain. Instead of approximating solutions analytically, FDTD discretizes space and time on an interlaced grid (the **Yee lattice**) and steps the electric and magnetic fields forward in a leapfrog scheme:

```
∂Ez/∂t = (1/ε) [ ∂Hy/∂x - ∂Hx/∂y ] - (σ/ε) Ez
∂Hx/∂t = -(1/μ) [ ∂Ez/∂y ]
∂Hy/∂t =  (1/μ) [ ∂Ez/∂x ]
```

Because it operates in the time domain, a single simulation captures the full broadband frequency response of a structure — making it a workhorse for antenna design, radar cross-section analysis, photonics, and RF component modeling.

---

## Features

- **2D TM_z solver** — solves Ez, Hx, Hy on a Yee staggered grid
- **Convolutional PML** — polynomial-graded absorbing boundary conditions (no reflections from domain edges)
- **Material support** — spatially varying ε_r, μ_r, σ (dielectrics, conductors, lossy media)
- **Geometry primitives** — rectangles and circles via `MaterialMap`
- **Sources** — sinusoidal `PointSource` and broadband `GaussianPulseSource`
- **Visualization** — snapshot plots, time-averaged intensity maps, animated GIFs
- **Pure NumPy** — no compiled extensions required

---

## Installation

```bash
git clone https://github.com/rowmatrix/fdtd-sim.git
cd fdtd-sim
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, NumPy, Matplotlib

---

## Quick Start

```python
from fdtd import GridConfig, PointSource, FDTDSolver
from visualize import plot_field

# Define the grid: 200x200 mm, 1 mm cells, 600 time steps
cfg = GridConfig(nx=200, ny=200, dx=1e-3, dy=1e-3, nt=800, pml_layers=25)

# 3 GHz point source at grid center
src = PointSource(ix=100, iy=100, frequency=3e9)

# Run
solver = FDTDSolver(cfg, sources=[src], record_every=5)
solver.run()

# Plot final snapshot
plot_field(solver)

# Plot final snapshot (run with PYTHONPATH=src from repo root)
```

---

## Examples

| Script | Description |
|--------|-------------|
| `examples/example_01_point_source.py` | Circular wavefront from a 3 GHz CW source in free space |
| `examples/example_02_dielectric_scattering.py` | Plane wave scattering off a dielectric cylinder (ε_r = 4) |
| `examples/example_03_waveguide.py` | Guided mode propagation in a parallel-plate waveguide with PEC walls |

Run any example from the repo root:

```bash
python examples/example_01_point_source.py
```

---

## Project Structure

```
fdtd-sim/
├── src/
│   ├── fdtd.py         # Core solver: GridConfig, MaterialMap, Sources, FDTDSolver, PML
│   └── visualize.py    # Plotting: snapshots, intensity maps, animations
├── examples/
│   ├── example_01_point_source.py
│   ├── example_02_dielectric_scattering.py
│   └── example_03_waveguide.py
├── tests/
│   └── test_fdtd.py    # pytest unit tests (Courant, sources, PML, energy)
├── docs/
│   └── theory.md       # FDTD theory reference
├── assets/             # Output figures and animations
├── requirements.txt
└── README.md
```

---

## Physics Reference

### Courant Stability Condition (2D)

The time step is constrained by:

```
Δt ≤ Δx / (c₀ · √2)
```

Violating this condition causes exponential field growth and numerical instability. This solver enforces the Courant condition automatically via `GridConfig.dt`.

### Yee Grid (TM_z)

```
Hy(i+½, j)     Ez(i, j)     Hy(i+½, j)
                    ↕
    Hx(i, j+½) ← Ez  → Hx(i, j+½)
                    ↕
```

Ez is defined at cell centers. Hx at top/bottom faces, Hy at left/right faces.

### PML Absorbing Boundaries

This solver uses a polynomial-graded convolutional PML (CPML) following Taflove & Hagness (2005). Conductivity ramps up smoothly at domain edges:

```
σ(d) = σ_max · (d/L_pml)^m     [m = 4 by default]
```

The optimal `σ_max` is computed from the PML thickness and target reflectance (default: -80 dB).

---

## Roadmap

- [ ] 1D FDTD with analytic validation
- [ ] TF/SF (Total-Field/Scattered-Field) source for true plane wave injection
- [ ] Dispersive materials (Drude model for metals/plasma)
- [ ] Near-to-far field transformation (antenna patterns)
- [ ] 3D solver (TE + TM, full vector fields)
- [ ] GPU acceleration via CuPy

---

## References

1. Yee, K.S. (1966). Numerical solution of initial boundary value problems involving Maxwell's equations. *IEEE TAP*, 14(3), 302–307.
2. Taflove, A. & Hagness, S.C. (2005). *Computational Electrodynamics: The FDTD Method* (3rd ed.). Artech House.
3. Roden, J.A. & Gedney, S.D. (2000). Convolution PML (CPML). *Microwave and Optical Technology Letters*, 27(5).

---

## Author

**Ibar Romay** — [rowmatrix](https://github.com/rowmatrix) | [LinkedIn](https://linkedin.com/in/ibarromay)

*Physics-first. Mission-critical. Building intelligent systems with aerospace rigor.*