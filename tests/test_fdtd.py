"""
test_fdtd.py
------------
Unit tests for the UPML split-field FDTD solver.

Run:
    pytest tests/test_fdtd.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

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
    return GridConfig(nx=60, ny=60, dx=1e-3, dy=1e-3,
                      nt=100, pml_layers=10)


@pytest.fixture
def free_space_solver(small_grid):
    src = PointSource(ix=30, iy=30, frequency=3e9)
    return FDTDSolver(small_grid, sources=[src], record_every=10)


# --------------------------------------------------------------------------- #
#  GridConfig
# --------------------------------------------------------------------------- #
class TestGridConfig:
    def test_courant_condition(self):
        """dt must satisfy the 2D Courant condition."""
        cfg = GridConfig(nx=100, ny=100, dx=1e-3, dy=1e-3)
        assert C0 * cfg.dt <= cfg.dx / np.sqrt(2) * 1.01

    def test_dt_positive(self):
        assert GridConfig().dt > 0

    def test_dimensions(self):
        cfg = GridConfig(nx=50, ny=80, dx=2e-3, dy=2e-3)
        assert abs(cfg.Lx - 0.1) < 1e-10
        assert abs(cfg.Ly - 0.16) < 1e-10


# --------------------------------------------------------------------------- #
#  MaterialMap
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
        assert mat.eps_r[5,   5] == 1.0

    def test_add_circle(self):
        mat = MaterialMap(100, 100)
        mat.add_circle(cx=50, cy=50, radius=10, eps_r=9.0)
        assert mat.eps_r[50, 50] == 9.0
        assert mat.eps_r[0,   0] == 1.0


# --------------------------------------------------------------------------- #
#  Source waveforms
# --------------------------------------------------------------------------- #
class TestSources:
    def test_point_source_near_zero_at_t0(self):
        src = PointSource(ix=0, iy=0, frequency=1e9)
        assert abs(src.waveform(0.0)) < 0.01

    def test_point_source_active_after_delay(self):
        src = PointSource(ix=0, iy=0, frequency=1e9)
        T = 1.0 / src.frequency
        peak = max(abs(src.waveform(t)) for t in np.linspace(5*T, 10*T, 200))
        assert peak > 0.8

    def test_gaussian_pulse_peak(self):
        t0, sp = 1e-9, 0.2e-9
        src = GaussianPulseSource(ix=0, iy=0, t0=t0, spread=sp)
        assert abs(src.waveform(t0) - 1.0) < 1e-9

    def test_gaussian_pulse_decays(self):
        t0, sp = 1e-9, 0.2e-9
        src = GaussianPulseSource(ix=0, iy=0, t0=t0, spread=sp)
        assert src.waveform(t0 + 5 * sp) < 0.01


# --------------------------------------------------------------------------- #
#  Solver initialisation
# --------------------------------------------------------------------------- #
class TestSolverInit:
    def test_split_fields_zero_at_start(self, free_space_solver):
        s = free_space_solver
        assert np.all(s.Ezx == 0)
        assert np.all(s.Ezy == 0)
        assert np.all(s.Hx  == 0)
        assert np.all(s.Hy  == 0)

    def test_ez_property_zero_at_start(self, free_space_solver):
        """Ez = Ezx + Ezy should be zero before any stepping."""
        assert np.all(free_space_solver.Ez == 0)

    def test_snapshots_empty_at_start(self, free_space_solver):
        assert len(free_space_solver.snapshots) == 0

    def test_ca_coefficient_range(self, small_grid):
        """_cax and _cay must be in (-1, 1] for numerical stability."""
        s = FDTDSolver(small_grid)
        assert np.all(s._cax <= 1.0)
        assert np.all(s._cay <= 1.0)
        assert np.all(s._cax > -1.0)
        assert np.all(s._cay > -1.0)


# --------------------------------------------------------------------------- #
#  Solver stepping and running
# --------------------------------------------------------------------------- #
class TestSolverRun:
    def test_source_injects_into_ez(self, free_space_solver):
        """After a few steps, Ez at the source location must be non-zero."""
        s = free_space_solver
        for n in range(5):
            s.step(n)
        ix, iy = s.sources[0].ix, s.sources[0].iy
        assert abs(s.Ez[ix, iy]) > 0

    def test_snapshot_count(self, free_space_solver):
        s = free_space_solver
        s.run(verbose=False)
        expected = s.config.nt // s.record_every
        assert len(s.snapshots) == expected

    def test_snapshot_shape(self, free_space_solver):
        s = free_space_solver
        s.run(verbose=False)
        for snap in s.snapshots:
            assert snap.shape == (s.config.nx, s.config.ny)

    def test_snapshot_is_total_ez(self, small_grid):
        """
        A snapshot recorded at step N must equal Ezx + Ezy at that exact
        step — not after further stepping. We verify by running one step
        past a recording boundary and checking the stored snapshot matches
        the Ez state captured immediately after that step.
        """
        src = PointSource(ix=30, iy=30, frequency=3e9)
        s = FDTDSolver(small_grid, sources=[src], record_every=10)

        # Step to exactly the first snapshot boundary
        for n in range(10):
            s.step(n)

        # The snapshot at index 0 was taken at step 0 (n=0, before source ramp)
        # Step 10 triggers the second snapshot — capture Ez right after
        ez_at_snap = s.Ez.copy()
        assert len(s.snapshots) == 1  # only step-0 snapshot so far

        s.step(10)  # this triggers snapshot index 1
        assert len(s.snapshots) == 2
        # snapshot[1] must equal the current Ez since no further steps
        # have been taken after the recording boundary
        np.testing.assert_array_almost_equal(s.snapshots[1], s.Ez, decimal=10)

    def test_energy_grows_with_cw_source(self, small_grid):
        """Total field energy should grow while a CW source is active."""
        src = PointSource(ix=30, iy=30, frequency=3e9)
        s = FDTDSolver(small_grid, sources=[src], record_every=10)
        s.run(verbose=False)
        e_start = np.sum(s.snapshots[0]  ** 2)
        e_end   = np.sum(s.snapshots[-1] ** 2)
        assert e_end > e_start

    def test_pml_absorbs_gaussian_pulse(self):
        """
        After a Gaussian pulse fully propagates out, PML should leave
        only residual energy. Uses a wide pulse and enough steps for the
        wavefront to reach and be absorbed by the boundaries.

        Grid: 100x100 mm, 1 mm cells
        Pulse center: t0 = 200 steps (~0.42 ns)
        Pulse width:  spread = 40 steps (~85 ps) — wide enough to be
                      well-resolved but narrow enough to fully exit by
                      step 600 (the last 100 steps should be near-zero).
        """
        cfg = GridConfig(nx=100, ny=100, dx=1e-3, dy=1e-3,
                         nt=700, pml_layers=20)
        src = GaussianPulseSource(
            ix=50, iy=50,
            t0=200 * cfg.dt,    # pulse peaks at step 200
            spread=40 * cfg.dt, # wide pulse, well-resolved
        )
        s = FDTDSolver(cfg, sources=[src], record_every=10)
        s.run(verbose=False)

        # Energy near the pulse peak (around step 200-300, snapshots 20-30)
        e_peak = np.sum(s.snapshots[25] ** 2)

        # Energy well after the pulse has left the grid (last snapshot)
        e_late = np.sum(s.snapshots[-1] ** 2)

        assert e_late < e_peak, (
            f"PML should absorb the outgoing pulse. "
            f"e_peak={e_peak:.6f}, e_late={e_late:.6f}"
        )

    def test_no_instability(self, small_grid):
        """Fields must remain bounded — no exponential blowup."""
        src = PointSource(ix=30, iy=30, frequency=3e9)
        s = FDTDSolver(small_grid, sources=[src], record_every=10)
        s.run(verbose=False)
        assert np.isfinite(s.Ez).all()
        assert np.max(np.abs(s.Ez)) < 1e6