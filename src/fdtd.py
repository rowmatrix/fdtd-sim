"""
fdtd.py
-------
2D Finite-Difference Time-Domain (FDTD) electromagnetic field solver.

Implements the Yee algorithm for TM_z polarization (Ez, Hx, Hy) with
uniaxial PML (UPML) absorbing boundary conditions.

The UPML splits Ez into two components (Ezx, Ezy) so that x-directed
and y-directed absorption are applied independently. This avoids the
instability caused by summing PML conductivities in corner regions.

References:
    Yee, K. (1966). IEEE TAP 14(3), 302-307.
    Taflove & Hagness (2005). Computational Electrodynamics, Ch. 7. Artech House.
    Gedney, S.D. (1996). IEEE TAP 44(12) — UPML formulation.

Author: Ibar Romay (rowmatrix)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------- #
#  Physical constants
# --------------------------------------------------------------------------- #
C0   = 2.997924e8            # speed of light [m/s]
MU0  = 4 * np.pi * 1e-7     # permeability of free space [H/m]
EPS0 = 1.0 / (MU0 * C0**2)  # permittivity of free space [F/m]


# --------------------------------------------------------------------------- #
#  Grid configuration
# --------------------------------------------------------------------------- #
@dataclass
class GridConfig:
    """
    Spatial and temporal discretization.

    Parameters
    ----------
    nx, ny      : Grid dimensions (cells).
    dx, dy      : Cell size [m].
    nt          : Number of time steps.
    courant     : Courant number (must be < 1/sqrt(2) for 2D stability).
    pml_layers  : PML thickness in cells on each edge.
    """
    nx: int   = 200
    ny: int   = 200
    dx: float = 1e-3
    dy: float = 1e-3
    nt: int   = 500
    courant: float = 0.9 / np.sqrt(2)
    pml_layers: int = 20

    @property
    def dt(self) -> float:
        return self.courant * min(self.dx, self.dy) / C0

    @property
    def Lx(self) -> float:
        return self.nx * self.dx

    @property
    def Ly(self) -> float:
        return self.ny * self.dy


# --------------------------------------------------------------------------- #
#  Material map
# --------------------------------------------------------------------------- #
@dataclass
class MaterialMap:
    """
    Spatially varying material properties.

    Attributes
    ----------
    eps_r : relative permittivity (nx, ny)
    mu_r  : relative permeability (nx, ny)
    sigma : electric conductivity (nx, ny) [S/m]
    """
    nx: int
    ny: int
    eps_r: np.ndarray = field(init=False)
    mu_r:  np.ndarray = field(init=False)
    sigma: np.ndarray = field(init=False)

    def __post_init__(self):
        self.eps_r = np.ones((self.nx, self.ny))
        self.mu_r  = np.ones((self.nx, self.ny))
        self.sigma = np.zeros((self.nx, self.ny))

    def add_rectangle(self, x0, y0, x1, y1,
                      eps_r=1.0, mu_r=1.0, sigma=0.0):
        self.eps_r[x0:x1, y0:y1] = eps_r
        self.mu_r [x0:x1, y0:y1] = mu_r
        self.sigma[x0:x1, y0:y1] = sigma

    def add_circle(self, cx, cy, radius,
                   eps_r=1.0, mu_r=1.0, sigma=0.0):
        ix, iy = np.ogrid[:self.nx, :self.ny]
        mask = (ix - cx)**2 + (iy - cy)**2 <= radius**2
        self.eps_r[mask] = eps_r
        self.mu_r [mask] = mu_r
        self.sigma[mask] = sigma


# --------------------------------------------------------------------------- #
#  Sources
# --------------------------------------------------------------------------- #
@dataclass
class PointSource:
    """
    Sinusoidal soft source injected additively into Ez at (ix, iy).

    Parameters
    ----------
    ix, iy      : Grid indices.
    frequency   : Source frequency [Hz].
    amplitude   : Peak amplitude [V/m].
    delay       : Gaussian ramp-up delay [s] (default: 3 / frequency).
    """
    ix: int
    iy: int
    frequency: float
    amplitude: float = 1.0
    delay: Optional[float] = None

    def waveform(self, t: float) -> float:
        delay = self.delay if self.delay is not None else 3.0 / self.frequency
        ramp  = 1.0 - np.exp(-((t - delay) * self.frequency) ** 2)
        return self.amplitude * ramp * np.sin(2 * np.pi * self.frequency * t)


@dataclass
class GaussianPulseSource:
    """
    Broadband Gaussian pulse injected at (ix, iy).

    Parameters
    ----------
    ix, iy    : Grid indices.
    t0        : Pulse center time [s].
    spread    : Pulse width sigma [s].
    amplitude : Peak amplitude [V/m].
    """
    ix: int
    iy: int
    t0: float
    spread: float
    amplitude: float = 1.0

    def waveform(self, t: float) -> float:
        return self.amplitude * np.exp(-0.5 * ((t - self.t0) / self.spread) ** 2)


# --------------------------------------------------------------------------- #
#  PML conductivity profile
# --------------------------------------------------------------------------- #
def _pml_profile(N: int, d: float, n_layers: int, m: int = 3,
                 R0: float = 1e-6) -> np.ndarray:
    """
    Polynomial-graded PML conductivity profile along one axis.

    Parameters
    ----------
    N        : Total number of cells along this axis.
    d        : Cell size [m].
    n_layers : Number of PML cells on each end.
    m        : Grading polynomial order.
    R0       : Target theoretical reflectance (lower = more absorption).

    Returns shape (N,) with sigma = 0 in interior, ramping at both ends.
    """
    sigma_max = -(m + 1) * np.log(R0) / (2.0 * n_layers * d) * C0 * EPS0
    sigma = np.zeros(N)
    for i in range(n_layers):
        # depth goes from ~1 (outermost) to ~0 (innermost PML cell)
        depth = (n_layers - i - 0.5) / n_layers
        val   = sigma_max * depth ** m
        sigma[i]         = val
        sigma[N - 1 - i] = val
    return sigma


# --------------------------------------------------------------------------- #
#  Main FDTD solver  (UPML split-field formulation)
# --------------------------------------------------------------------------- #
class FDTDSolver:
    """
    2D TM_z FDTD solver with Uniaxial PML absorbing boundaries.

    Ez is split into two sub-components to apply x and y absorption
    independently — this is the key that makes corners stable.

        Ezx(i,j): absorbs in x-direction (updated from dHy/dx)
        Ezy(i,j): absorbs in y-direction (updated from dHx/dy)
        Ez      = Ezx + Ezy  (what we visualize and inject into)

    Hx and Hy use simple lossless leapfrog updates. PML absorption
    on the H fields is approximated via the Ez split, which is
    sufficient for most FDTD applications at this grid resolution.

    Parameters
    ----------
    config       : GridConfig
    materials    : MaterialMap (defaults to free space)
    sources      : list of PointSource or GaussianPulseSource
    record_every : Save a snapshot every N steps
    """

    def __init__(
        self,
        config: GridConfig,
        materials: Optional[MaterialMap] = None,
        sources: Optional[list] = None,
        record_every: int = 5,
    ):
        self.config       = config
        self.materials    = materials or MaterialMap(config.nx, config.ny)
        self.sources      = sources or []
        self.record_every = record_every

        nx, ny = config.nx, config.ny
        dt     = config.dt
        dx, dy = config.dx, config.dy

        # ---- Fields ------------------------------------------------------- #
        self.Hx  = np.zeros((nx, ny))
        self.Hy  = np.zeros((nx, ny))
        self.Ezx = np.zeros((nx, ny))   # Ez split: x-component
        self.Ezy = np.zeros((nx, ny))   # Ez split: y-component

        # ---- Material properties ------------------------------------------ #
        eps = self.materials.eps_r * EPS0  # (nx, ny)
        sig = self.materials.sigma          # (nx, ny) material conductivity

        # ---- PML conductivity profiles (1D, then broadcast to 2D) --------- #
        sx_1d = _pml_profile(nx, dx, config.pml_layers)   # (nx,)  varies along x
        sy_1d = _pml_profile(ny, dy, config.pml_layers)   # (ny,)  varies along y

        sx = sx_1d[:, np.newaxis] * np.ones((nx, ny))     # (nx, ny)
        sy = sy_1d[np.newaxis, :] * np.ones((nx, ny))     # (nx, ny)

        # ---- Ezx update coefficients (absorbs in x, driven by dHy/dx) ---- #
        # sig_x = material sigma + PML sx
        sig_x   = sig + sx
        denom_x = eps + sig_x * dt / 2.0
        self._cax  = (eps - sig_x * dt / 2.0) / denom_x   # (nx, ny)
        self._cbx  = (dt / dx) / denom_x                    # (nx, ny)

        # ---- Ezy update coefficients (absorbs in y, driven by dHx/dy) ---- #
        sig_y   = sig + sy
        denom_y = eps + sig_y * dt / 2.0
        self._cay  = (eps - sig_y * dt / 2.0) / denom_y   # (nx, ny)
        self._cby  = (dt / dy) / denom_y                    # (nx, ny)

        # ---- H update coefficients (lossless, free-space scalars) --------- #
        self._chx = dt / (MU0 * dy)   # for Hx update
        self._chy = dt / (MU0 * dx)   # for Hy update

        # ---- Snapshot storage --------------------------------------------- #
        self.snapshots:      list[np.ndarray] = []
        self.snapshot_times: list[float]      = []

    @property
    def Ez(self) -> np.ndarray:
        """Total Ez field = Ezx + Ezy."""
        return self.Ezx + self.Ezy

    # ----------------------------------------------------------------------- #
    def step(self, n: int):
        """Advance all fields by one Yee leapfrog time step."""
        t = n * self.config.dt

        # Snapshot of current Ez for H updates
        Ez = self.Ez

        # ---- 1. Update Hx (lossless) -------------------------------------- #
        # dHx/dt = -(1/mu0) * dEz/dy
        self.Hx[:, :-1] -= self._chx * (Ez[:, 1:] - Ez[:, :-1])

        # ---- 2. Update Hy (lossless) -------------------------------------- #
        # dHy/dt = +(1/mu0) * dEz/dx
        self.Hy[:-1, :] += self._chy * (Ez[1:, :] - Ez[:-1, :])

        # ---- 3. Update Ezx (x-absorption, driven by dHy/dx) -------------- #
        dHy_dx = self.Hy[1:-1, 1:-1] - self.Hy[:-2, 1:-1]
        self.Ezx[1:-1, 1:-1] = (
            self._cax[1:-1, 1:-1] * self.Ezx[1:-1, 1:-1]
            + self._cbx[1:-1, 1:-1] * dHy_dx
        )

        # ---- 4. Update Ezy (y-absorption, driven by dHx/dy) -------------- #
        dHx_dy = self.Hx[1:-1, 1:-1] - self.Hx[1:-1, :-2]
        self.Ezy[1:-1, 1:-1] = (
            self._cay[1:-1, 1:-1] * self.Ezy[1:-1, 1:-1]
            - self._cby[1:-1, 1:-1] * dHx_dy
        )

        # ---- 5. Inject sources into total Ez (soft/additive) -------------- #
        # Split evenly between Ezx and Ezy so Ez = Ezx + Ezy is correct
        for src in self.sources:
            val = src.waveform(t) / 2.0
            self.Ezx[src.ix, src.iy] += val
            self.Ezy[src.ix, src.iy] += val

        # ---- 6. Record snapshot of total Ez ------------------------------- #
        if n % self.record_every == 0:
            self.snapshots.append(self.Ez.copy())
            self.snapshot_times.append(t)

    # ----------------------------------------------------------------------- #
    def run(self, verbose: bool = True):
        """Run all nt time steps."""
        nt = self.config.nt
        for n in range(nt):
            self.step(n)
            if verbose and n % 100 == 0:
                peak = np.max(np.abs(self.Ez))
                print(f"  step {n:4d}/{nt}  |  "
                      f"t = {n * self.config.dt * 1e9:.2f} ns  |  "
                      f"|Ez|_max = {peak:.4f}")
        if verbose:
            print(f"\nDone. {len(self.snapshots)} snapshots recorded.")