# DPF spatial discretization: numerical diffusion and the van Leer scheme

This document explains improvement **#6** — replacing the first-order upwind
advection in the DPF conduits with a higher-order, low-dissipation **van Leer
(MUSCL) flux-limited** scheme — why it matters, how it works, and the
grid-convergence evidence. It complements `docs/NUMERICS.md` (which describes
the overall method-of-lines solve).

## The problem: numerical diffusion

The DPF model (paper Eq. 1) is advection + physical dispersion:

```
∂c/∂t = −u ∂c/∂z + D_ax ∂²c/∂z²,     D_ax = u·d/Pe,  Pe = 0.5
```

The original scheme used **first-order upwind** for the advection term. Upwind
is simple, positive (no oscillations) and gives a neat constant tridiagonal
Jacobian — but its "modified equation" reveals it behaves as if there were an
*extra* dispersion coefficient

```
D_num ≈ u·Δz / 2          (numerical / artificial diffusion)
```

on top of the physical `D_ax`. So the mesh silently makes the conduit more
dispersive than the model intends. The relative error is

```
D_num / D_ax = (u·Δz/2) / (u·d/Pe) = Pe·Δz / (2d) = Δz / (4d)   (Pe = 0.5)
```

which is large for **long, thin** conduits. For the sample loop (d = 0.5 mm,
L = 1324 mm) at the default 40 cells, `Δz = 33 mm`, so `D_num` is **~16× the
physical `D_ax`** — the grid, not the physics, sets the dispersion. The result
is a pulse that is far too broad and too short.

## The evidence (`convergence_study.py`)

Impulse response through the sample loop, sweeping the cell count. `area/in` is
tracer mass recovery (should be 1), `var` is the spread of the outlet response
(should converge to the true physical value):

```
scheme      n  area/in     MRT     var     peak
upwind     20   1.0000   16.69   12.12   0.1174
upwind     40   1.0000   16.89    6.40   0.1598   <- default cells
upwind    160   1.0000   17.04    1.96   0.2864
upwind    640   1.0000   17.07    0.83   0.4386   <- still not converged
vanleer    20   1.0000   17.11    3.48   0.2028
vanleer    40   1.0000   17.09    1.42   0.3154   <- default cells
vanleer   160   1.0000   17.09    0.50   0.5495   <- essentially converged
vanleer   640   1.0000   17.09    0.45   0.5903
```

Reading it:

- **Upwind is nowhere near grid-converged** — its variance is still falling and
  its peak still rising at 640 cells. At the default 40 cells its spread (6.40)
  is ~14× the converged physical value (~0.45).
- **van Leer converges by ~160 cells** (var 0.50 → 0.45, peak 0.55 → 0.59). At
  the default 40 cells its spread (1.42) is already ~4.5× closer to converged
  than upwind's.
- **Both conserve mass exactly** (`area/in = 1.0000`) and preserve the mean
  residence time (~17 s). The scheme changes the *shape* (dispersion), not the
  amount of tracer or when it arrives on average.

`convergence_study.png` plots peak and variance vs cell count for both schemes.

## How the van Leer scheme works

It is a conservative finite-volume scheme written with **total face fluxes**.
For each face `j` the flux is `Ftot[j] = advective − diffusive`, and every cell
is the flux difference `dc_i/dt = −(Ftot[i+1] − Ftot[i])/Δz`. Because
neighbouring cells share the same face flux, the sum telescopes and mass is
conserved exactly (this is what fixed an earlier non-conservative version).

- **Inlet face:** the Danckwerts boundary gives a total inlet flux `u·c_in(t)`.
- **Interior faces:** the advective part uses a **slope-limited linear
  reconstruction** of the concentration at the face from the upwind cell:

  ```
  c_face = c_up + ½ · φ(r) · (c_up − c_up-1),
  r = (c_down − c_up) / (c_up − c_up-1),
  φ(r) = (r + |r|) / (1 + |r|)          # van Leer limiter
  ```

  In smooth regions `φ ≈ 1` and the reconstruction is ~2nd-order (low
  diffusion); near a sharp front `φ → 0` and it falls back to first-order
  upwind, which is what prevents the spurious over/undershoots a naive
  central scheme would produce. The diffusive part is the usual central
  `D_ax·(c_j − c_{j-1})/Δz`.
- **Outlet face:** first-order upwind advection, no diffusive flux (Neumann).

The scheme is selected per call or globally:

```python
from rtd.units import DPF_SCHEME          # module default, = "vanleer"
dpf_outlet(..., scheme="vanleer")         # or "upwind"
```

`DPF_SCHEME = "vanleer"` is the default used everywhere (peripheral tubes and
the filter's V_I compartment).

## Cost and the Jacobian trade-off

The MUSCL stencil for cell `i` reaches cells `i−2 … i+1`, and the limiter is
**nonlinear**, so the exact constant tridiagonal Jacobian used by the upwind
scheme (improvement #10) no longer applies. Instead the van Leer path hands BDF
a **banded sparsity pattern** (bands −2, −1, 0, +1) and lets it build a grouped
finite-difference Jacobian — much cheaper than a dense one. In practice the run
time is essentially unchanged versus upwind (the band is narrow), while the
accuracy per cell is far higher. The upwind scheme (with its analytic Jacobian)
remains available via `scheme="upwind"` for comparison or speed-critical use.

## Practical guidance

- The default (`vanleer`, 40 cells) already removes most of the over-dispersion
  and is the recommended setting.
- For a **grid-independent** result on the thinnest conduits (the sample loop),
  use ~160 cells (`dpf_outlet(..., n_cells=160)`); the convergence table shows
  that is where van Leer has settled.
- Mean residence time and mass recovery are insensitive to the scheme; only the
  dispersion (peak height / spread) changes, which is exactly the quantity the
  higher-order scheme improves.
