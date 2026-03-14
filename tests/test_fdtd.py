"""
test_fdtd.py
------------
Unit tests for the core FDTD solver.

Tests cover:
    - Courant condition (stability)
    - Field initialization (all zeros)
    - Source injection (Ez at source location changes)
    - Energy conservation in lossless free space (approximate)
    - PML absorption (energy decreases after source stops)
    - Material assignment

Run:
    pytest tests/test_fdtd.py -v
"""

import sys
sys.path.insert(0, "../src")

import numpy as np
import pytest
from fdtd import (
    GridConfig, MaterialMap, PointSource,
    GaussianPulseSource, FDTDSolver,
    C0, MU0, EPS0,
)


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_grid():
    return GridConfig(nx=60, ny=60, dx=1e-3, dy=1e-3, nt=100, pml_layers=10)


@pytest.fixture
def free_space_solver(small_grid):
    src = PointSource(ix=30, iy=30, frequency=5e9)
    return FDTDSolver(small_grid, sources=[src], record_every=10)


# --------------------------------------------------------------------------- #
#  GridConfig tests
# --------------------------------------------------------------------------- #
class TestGridConfig:
    def test_courant_condition(self):
        """dt must satisfy Courant condition: c*dt <= dx/sqrt(2)."""
        cfg = GridConfig(nx=100, ny=100, dx=1e-3, dy=1e-3)
        assert C0 * cfg.dt <= cfg.dx / np.sqrt(2) * 1.01  # 1% tolerance

    def test_dimensions(self):
        cfg = GridConfig(nx=50, ny=80, dx=2e-3, dy=2e-3)
        assert abs(cfg.Lx - 0.1) < 1e-10
        assert abs(cfg.Ly - 0.16) < 1e-10

    def test_dt_positive(self):
        cfg = GridConfig()
        assert cfg.dt > 0


# --------------------------------------------------------------------------- #
#  MaterialMap tests
# --------------------------------------------------------------------------- #
class TestMaterialMap:
    def test_default_free_space(self):
        mat = MaterialMap(50, 50)
        assert np.all(mat.eps_r == 1.0)
        assert np.all(mat.mu_r  == 1.0)
        assert np.all(mat.sigma == 0.0)

    def test_add_rectangle(self):
        mat = MaterialMap(50, 50)
        mat.add_rectangle(10, 10, 30, 30, eps_r=4.0)
        assert mat.eps_r[20, 20] == 4.0
        assert mat.eps_r[5,  5 ] == 1.0   # outside region

    def test_add_circle(self):
        mat = MaterialMap(100, 100)
        mat.add_circle(cx=50, cy=50, radius=10, eps_r=9.0)
        assert mat.eps_r[50, 50] == 9.0   # center
        assert mat.eps_r[0,  0 ] == 1.0   # outside


# --------------------------------------------------------------------------- #
#  Source waveform tests
# --------------------------------------------------------------------------- #
class TestSources:
    def test_point_source_zero_at_t0(self):
        """Source waveform should be ~0 at t=0 (before ramp-up)."""
        src = PointSource(ix=0, iy=0, frequency=1e9)
        assert abs(src.waveform(0.0)) < 0.01

    def test_point_source_nonzero_later(self):
        src = PointSource(ix=0, iy=0, frequency=1e9)
        T = 1.0 / src.frequency
        # After several cycles the source should be active
        val = max(abs(src.waveform(t)) for t in np.linspace(5*T, 10*T, 100))
        assert val > 0.5

    def test_gaussian_pulse_peak(self):
        t0 = 1e-9
        sp = 0.2e-9
        src = GaussianPulseSource(ix=0, iy=0, t0=t0, spread=sp)
        assert abs(src.waveform(t0) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
#  Solver initialization tests
# --------------------------------------------------------------------------- #
class TestSolverInit:
    def test_fields_zero_at_start(self, free_space_solver):
        s = free_space_solver
        assert np.all(s.Ez == 0)
        assert np.all(s.Hx == 0)
        assert np.all(s.Hy == 0)

    def test_snapshot_list_empty_at_start(self, free_space_solver):
        assert len(free_space_solver.snapshots) == 0


# --------------------------------------------------------------------------- #
#  Solver run tests
# --------------------------------------------------------------------------- #
class TestSolverRun:
    def test_source_injects_field(self, free_space_solver):
        """Ez at source location must become non-zero after running."""
        s = free_space_solver
        s.step(0)
        s.step(1)
        s.step(2)
        ix, iy = s.sources[0].ix, s.sources[0].iy
        # After a few steps the field at the source should be non-zero
        assert abs(s.Ez[ix, iy]) > 0

    def test_snapshots_recorded(self, free_space_solver):
        """Snapshots should be recorded at the correct interval."""
        s = free_space_solver
        s.run(verbose=False)
        expected = s.config.nt // s.record_every
        assert len(s.snapshots) == expected

    def test_snapshot_shape(self, free_space_solver):
        s = free_space_solver
        s.run(verbose=False)
        nx, ny = s.config.nx, s.config.ny
        for snap in s.snapshots:
            assert snap.shape == (nx, ny)

    def test_energy_increase_with_source(self, small_grid):
        """Total field energy should grow while source is active."""
        src = PointSource(ix=30, iy=30, frequency=5e9)
        s = FDTDSolver(small_grid, sources=[src], record_every=10)
        s.run(verbose=False)
        energy_start = np.sum(s.snapshots[0]**2)
        energy_end   = np.sum(s.snapshots[-1]**2)
        # Energy should grow (source is continuously injecting)
        assert energy_end > energy_start

    def test_pml_absorbs_outgoing_waves(self):
        """
        After source is shut off (Gaussian pulse), energy should decay
        as PML absorbs outgoing waves.
        """
        cfg = GridConfig(nx=80, ny=80, dx=1e-3, dy=1e-3, nt=400, pml_layers=15)
        # Gaussian pulse: most energy emitted in first ~200 steps
        src = GaussianPulseSource(
            ix=40, iy=40,
            t0=50 * cfg.dt,
            spread=15 * cfg.dt,
        )
        s = FDTDSolver(cfg, sources=[src], record_every=10)
        s.run(verbose=False)
        # Energy in second half should be lower than in first half
        mid = len(s.snapshots) // 2
        e_early = np.sum(s.snapshots[mid // 2]**2)
        e_late  = np.sum(s.snapshots[-1]**2)
        assert e_late < e_early, "PML should absorb outgoing wave energy"
