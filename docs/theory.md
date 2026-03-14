# FDTD Theory Reference

A concise reference for the physics and numerics behind this solver.

---

## Maxwell's Curl Equations (TM_z, 2D)

In 2D with TM_z polarization (Ez, Hx, Hy non-zero; all ∂/∂z = 0):

```
∂Ez/∂t = (1/ε) [ ∂Hy/∂x - ∂Hx/∂y ] - (σ/ε) Ez

∂Hx/∂t = -(1/μ) [ ∂Ez/∂y ]

∂Hy/∂t =  (1/μ) [ ∂Ez/∂x ]
```

where ε = ε₀ · εᵣ, μ = μ₀ · μᵣ, σ = electric conductivity.

---

## Yee Grid

The Yee (1966) staggered grid interleaves E and H components in both space and time:

- **Ez(i, j)** sampled at integer grid points (cell centers)
- **Hx(i, j+½)** sampled at half-integer j (horizontal faces)
- **Hy(i+½, j)** sampled at half-integer i (vertical faces)
- **E** updated at integer time steps n
- **H** updated at half-integer time steps n+½

This staggering gives second-order accuracy in both space and time using only simple centered differences, with no matrix inversion required.

---

## Finite Difference Update Equations

### Ez update (at time step n+1):

```
Ez^(n+1)(i,j) = Ca(i,j) · Ez^n(i,j)
              + Cbx(i,j) · [ Hy^(n+½)(i+½,j) - Hy^(n+½)(i-½,j) ] / Δx
              - Cby(i,j) · [ Hx^(n+½)(i,j+½) - Hx^(n+½)(i,j-½) ] / Δy

Ca  = (1 - σΔt/2ε) / (1 + σΔt/2ε)
Cbx = (Δt/εΔx)    / (1 + σΔt/2ε)
Cby = (Δt/εΔy)    / (1 + σΔt/2ε)
```

### H update (lossless, no magnetic conductivity):

```
Hx^(n+½)(i,j+½) = Hx^(n-½)(i,j+½) - (Δt/μΔy) [ Ez^n(i,j+1) - Ez^n(i,j) ]

Hy^(n+½)(i+½,j) = Hy^(n-½)(i+½,j) + (Δt/μΔx) [ Ez^n(i+1,j) - Ez^n(i,j) ]
```

---

## Courant Stability Condition

The explicit FDTD scheme is conditionally stable. In 2D:

```
Δt ≤ Δx / (c₀ · √2)    [for Δx = Δy]
```

The Courant number S = c₀ · Δt / Δx must satisfy S ≤ 1/√2 ≈ 0.707.
This solver uses S = 0.9/√2 ≈ 0.636 by default.

---

## Spatial Resolution

To resolve wave propagation accurately, the grid spacing must be a small fraction of the shortest wavelength:

```
Δx ≤ λ_min / N_ppw
```

where N_ppw is the number of points per wavelength. Common practice:
- N_ppw ≥ 10 for adequate accuracy
- N_ppw ≥ 20 for good accuracy
- N_ppw ≥ 30 for high accuracy (dispersive materials, etc.)

For a 10 GHz signal in free space: λ = 30 mm → Δx ≤ 1.5 mm for N_ppw = 20.

---

## Perfectly Matched Layer (PML)

Bare truncation of the computational domain causes artificial reflections from the boundaries. The PML (Bérenger 1994, refined by Gedney 1996) avoids this by surrounding the domain with a lossy medium that is impedance-matched to the interior — causing no reflection at the interface, only absorption.

### Conductivity profile (polynomial grading):

```
σ(d) = σ_max · (d / d_PML)^m
```

where d is the depth into the PML layer, d_PML is its total thickness, and m is the grading order (typically 3–4).

### Optimal σ_max:

```
σ_max = -(m+1) · c₀ · ln(R₀) / (2 · d_PML)
```

where R₀ is the target theoretical reflectance (default: R₀ = 10⁻⁸, i.e., -160 dB round-trip).

---

## Source Injection

**Soft source** (additive): adds the source waveform to the existing field at the injection point. Allows backscattered fields to pass through the source location transparently. Used in examples 01–03.

**Hard source** (assignment): sets the field to the waveform value. Reflects backscattered waves. Appropriate only for isolated simulations.

For plane wave injection, a **Total-Field/Scattered-Field (TF/SF)** formulation is preferred (planned for roadmap).

---

## Numerical Dispersion

Even with the Yee scheme, the numerical phase velocity differs slightly from c₀ due to discretization:

```
v_phase / c₀ ≈ 1 - (π/N_ppw)² / 6  (for 1D, leading order)
```

This causes grid dispersion: different frequency components travel at slightly different speeds. Keeping N_ppw ≥ 20 keeps this error below ~0.4% for most applications.

---

## References

1. Yee, K.S. (1966). Numerical solution of initial boundary value problems involving Maxwell's equations. *IEEE TAP*, 14(3).
2. Bérenger, J.P. (1994). A perfectly matched layer for the absorption of electromagnetic waves. *J. Comput. Phys.*, 114(2).
3. Gedney, S.D. (1996). An anisotropic perfectly matched layer absorbing medium. *IEEE TAP*, 44(12).
4. Taflove, A. & Hagness, S.C. (2005). *Computational Electrodynamics: The FDTD Method* (3rd ed.). Artech House.
