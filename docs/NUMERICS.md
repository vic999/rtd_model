# How the equations are solved

This document explains the numerical solution of every model in the package —
the discretizations, boundary-condition handling, time integrator, and the way
the equipment train is chained together. It is meant to be read alongside the
source in `rtd/units.py`, `rtd/filter_model.py` and `rtd/simulate.py`.

Notation follows the paper (Chen et al., 2024). Internally everything is in a
consistent unit system: **volume in µL, length in mm, time in s**, so
`V̇[µL/s] = flow[mL/min] · 1000/60`. Concentrations are carried in molar units;
the models are linear in concentration, so absolute scaling does not affect RTD
shape.

---

## 1. Overall strategy: an operator chain

The ÄKTA + filter system is a **series of units with no downstream feedback**
(the flow only ever moves forward). That is the key structural fact the solver
exploits. Each unit is treated as an input→output operator

```
c_out(t) = Unit[ c_in(t) ]
```

and the whole train is solved one unit at a time (`rtd/simulate.py::run_train`):

```
c = injection_signal
for unit in train:
    c = unit.propagate(t_grid, c, flow)   # solve this unit with c as its inlet
    # c is now the outlet, i.e. the inlet of the next unit
```

Because there is no recycle, this sequential solution is exact — it is not an
operator-splitting approximation. Each `propagate` call solves that unit's own
differential equation on the shared time grid `t_grid`, using the previous
unit's computed outlet as a time-dependent inlet boundary condition.

The inlet signal handed to each unit is a sampled array. Inside a unit it is
turned into a continuous function of time by **linear interpolation** with flat
extrapolation at the ends (`_make_inlet`, `scipy.interpolate.interp1d`), so the
ODE solver can evaluate `c_in(t)` at any internal time step it chooses.

---

## 2. CST model — Eq. (2)

Governing equation for a single well-mixed tank of hold-up `V`:

```
dc/dt = (V̇ / V) · (c_in(t) − c)
```

This is one linear ODE. It is integrated directly with
`scipy.integrate.solve_ivp` (`rtd/units.py::cst_outlet`). The right-hand side is

```python
def rhs(t, y):
    return np.array([tau_inv * (cin(t) - y[0])])   # tau_inv = V̇/V
```

The outlet concentration is the tank concentration itself (a CST is perfectly
mixed, so outlet = interior). The analytic impulse response is a decaying
exponential with time constant `V/V̇`; `verify.py` confirms the numerical first
moment reproduces `V/V̇`.

---

## 3. DPF model — Eq. (1), method of lines

Governing PDE (advection + axial dispersion) for a conduit of length `L`:

```
∂c/∂t = −u ∂c/∂z + D_ax ∂²c/∂z²
```

with, as specified in the paper,

```
u    = V̇ / A                     (A taken as V/L so mean residence = V/V̇)
D_ax = u · d / Pe,   Pe = 0.5     (Péclet based on the physical diameter d)
```

### 3.1 Spatial discretization (finite volume)

The domain `[0, L]` is split into `n_cells = 40` equal cells of width
`dz = L/n_cells`; `c[i]` is the average concentration in cell `i`. The PDE is
converted to a system of 40 ODEs (method of lines) by approximating the two
spatial operators at each cell:

**Advection — first-order upwind.** Because flow is strictly forward (`u > 0`),
the concentration carried across a face comes from the upstream cell:

```
advection[0]   = u · (c_in(t) − c[0]) / dz      # inlet face uses c_in(t)
advection[i]   = u · (c[i-1]  − c[i]) / dz       # interior faces, upwind
```

Upwind is used deliberately: it is unconditionally stable and monotone (no
spurious oscillations / negative concentrations), at the cost of a small amount
of numerical diffusion. With `Pe = 0.5` the physical dispersion `D_ax` dominates
that numerical diffusion, so the tracer spreading is set by the model, not the
scheme.

**Dispersion — central difference.**

```
diffusion[i] = D_ax · (c[i+1] − 2·c[i] + c[i-1]) / dz²     (interior)
```

### 3.2 Boundary conditions

The paper prescribes **Danckwerts** conditions, which the discretization enforces
through the boundary diffusive fluxes rather than through ghost cells:

- **Inlet (z = 0):** `D_ax ∂c/∂z|₀ = u·(c(0,t) − c_in(t))`. The inlet diffusive
  flux is set equal to `u·(c[0] − c_in(t))`, so mass entering by dispersion is
  balanced against the advective jump at the boundary:

  ```python
  left_flux0 = u * (c[0] - cin(t))                  # = D_ax ∂c/∂z at inlet
  diffusion[0] = (D_ax*(c[1]-c[0])/dz - left_flux0) / dz
  ```

- **Outlet (z = L):** Neumann, `∂c/∂z|_L = 0`, i.e. no diffusive flux through the
  last face:

  ```python
  diffusion[-1] = D_ax * (c[-2] - c[-1]) / dz²
  ```

The cell right-hand side is `advection + diffusion`, and the outlet signal
returned to the next unit is the last cell, `c[-1](t)`.

`verify.py` checks that this scheme conserves tracer mass (area of the outlet
pulse ÷ area of the inlet pulse = 1.000) and reproduces the correct mean
residence time `V/V̇`.

---

## 4. Three-compartment filter — Eqs. (3)–(9)

A filter `propagate` call is itself a mini-chain of three sub-solves
(`rtd/filter_model.py::filter_outlet`):

```
c_in → [ V_I : DPF ] → [ V_wall : CST ] → [ V_O : permeate ] → c_out
```

- `V_I` (hollow spaces & headers) reuses the **DPF** solver of §3, with the
  compartment modelled as an equivalent cylinder of length `L = 10.8 cm` and
  diameter `d̄ = 2·√(V_I/(π·L))`.
- `V_wall` (fibre walls) reuses the **CST** solver of §2.
- `V_O` (permeate space) is the new part, solved as one coupled ODE system
  described next.

### 4.1 Permeate space: radial TIS × axial two-CST

Radial non-ideal mixing is a **tanks-in-series** cascade of `l` stages. Within
each radial stage, axial non-ideal mixing is **two interconnected CSTs**: a
through-flow tank `k1` (carries the net flow) and an exchange/dead-zone tank
`k2` (no net flow, exchanges with `k1` at flow `η·V̇`). The state vector stacks
both tanks of every stage:

```
y = [ c_k1(1), c_k2(1), c_k1(2), c_k2(2), …, c_k1(l), c_k2(l) ]     (length 2l)
```

For stage `j`, with `V_stage = V_O/l` and inlet `c_k1(j−1)` (the previous
stage's through-flow tank, or the compartment inlet for `j=1`):

```
ε·V_stage      · dc_k1(j)/dt = V̇·(c_k1(j-1) − c_k1(j)) + η·V̇·(c_k2(j) − c_k1(j))   (Eq. 3)
(1−ε)·V_stage  · dc_k2(j)/dt =                            η·V̇·(c_k1(j) − c_k2(j))   (Eq. 4)
```

The exchange terms are written in **antisymmetric (conservative) form** — the
mass `k1` loses to `k2` is exactly what `k2` gains — so the coupled pair
conserves tracer regardless of `ε` and `η`. These `2l` ODEs are assembled in a
single `rhs(t, y)` loop over stages and integrated together with `solve_ivp`.
The compartment outlet is `c_k1(l)`, the last through-flow tank.

Behavioural check: this structure is what produces the **pronounced exponential
tailing** through a filter (the dead-zone tank slowly bleeds tracer back into the
flow) and the `l`-stage cascade sharpens the front — both visible in the
reproduced Figure 3/4.

### 4.2 Film resistance: the time-dependent split ε(t) — Eqs. (5)–(9)

`ε` is the through-flow volume fraction. With film resistance active it is made
concentration-dependent via the Graetz–Lévêque scaling `k_m ∝ u^{1/3}`:

```
u   = V̇ / (π·d̄·L)                         side-area velocity of the equiv. cylinder
ε(t) = clip( 1 − u^{1/3}·(A₃/A_j)·(Δc_eq + α·Δc_k1)/Δc_max ,  0, 1 )    (Eqs. 8–9)
```

where `Δc_k1 = c_k1(j−1) − c_k1(j)` is the tracer driving force at the stage
inlet and `A₃/A_j` the surface-area ratio (Eq. 7).

**Implementation note (important).** Eq. 8 needs the displaced-buffer driving
force `Δc_eq` and a mass-transfer scale `k_m,eq` that the paper does not
tabulate; with the reported `Δc_max = 2.17e-7` and normalized concentrations the
raw argument is astronomically large, which makes `ε` bang-bang between 0 and 1
and the ODE violently stiff. So the code:

1. keeps the full functional form (`film_resistance=True`), but maps the
   argument through a smooth saturating function `1 − (1−ε_floor)·tanh(|arg|)`
   so `ε ∈ [ε_floor, 1]` and the ODE stays integrable. This reproduces the
   *qualitative* behaviour the paper describes: `ε → 1` near equilibrium
   (`Δc → 0`), `ε → ε_floor` at large driving force.
2. defaults the runnable figures to `film_resistance=False`, i.e. a **constant**
   `ε = ε_const = 0.85`. This is numerically clean and reproduces the RTD curve
   **shapes**; quantitative use of the film term would require the missing scale
   factor or a re-fit against experimental data.

This is the single place where the reconstruction cannot be made quantitative
from the paper alone, and it is flagged both here and in the code.

---

## 5. Time integration

All units use `scipy.integrate.solve_ivp` with:

- **Method `BDF`** — an implicit, variable-order backward-differentiation
  formula. It is chosen because (a) the semi-discretized DPF system is mildly
  stiff (dispersion couples neighbouring cells), and (b) the explicit/`LSODA`
  paths were observed to silently emit `NaN` on the long, flat post-pulse tails
  of slow experiments. BDF integrates those tails cleanly.
- **Tolerances** `rtol = 1e-6`, `atol = 1e-9`.
- **`t_eval = t_grid`** so output lands exactly on the shared sample grid that
  the next unit expects.

### 5.1 `max_step` — resolving narrow injections without over-solving

An adaptive integrator can legally take one giant step straight over a short
injection pulse and never "see" it, because the forcing enters only through
`c_in(t)`. To prevent that, `compute_max_step` (`rtd/units.py`) inspects the
inlet signal and caps the step:

- it finds the width of the narrowest **active** (non-zero) segment of `c_in`
  and limits `max_step` to a quarter of it, so a pulse is always sampled several
  times;
- elsewhere it allows a coarse step (`span/500`) so long flat regions are cheap.

`run_train` computes one `max_step` from the original (sharpest) injection and
reuses it for every unit, guaranteeing the pulse is resolved throughout the
train. This is what keeps a 260 µL bolus faithfully represented even when it
later passes through a filter with thousands of seconds of hold-up.

---

## 6. What "solved" means here — and what is not solved

- **Solved:** every curve is the output of numerically integrating the model
  ODEs/PDE above. `verify.py` demonstrates this by recovering emergent
  properties that are *not* coded anywhere — tracer mass conservation, the
  `V/V̇` first moment, unit steady-state gain, and monotonic peak-time ordering
  with filter size.
- **Not solved:** the **inverse / calibration** problem (paper Eq. 10). The
  package runs a *forward* simulation with the paper's already-calibrated
  parameters (`l=3, η=0.13, α=1.14, Δc_max=2.17e-7`). Re-fitting those
  parameters, and computing the Table 3 R² values, requires the experimental
  UV/conductivity traces, which are not available. The hook for that
  (`rtd.r2_score`, matching `sklearn`) is provided for when data exists.
