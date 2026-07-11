# Parameter & units audit

Every constant used by the model, its units, its value, and where it comes
from. Units are stated explicitly to remove ambiguity. The one genuinely
unresolved item (the film-resistance `Δc_max` scale) is flagged at the end.

Internal unit convention: **volume µL, length mm, time s**; flow is entered as
mL/min and converted to µL/s internally. Concentration is molar (mol/L) unless
a detector conversion is applied.

## Calibrated model parameters (from the paper)

| Symbol | Value | Units | Meaning | Source |
|--------|-------|-------|---------|--------|
| `l` | 3 | – | radial tanks-in-series in the permeate space | Chen et al. 2024, calibration (Sec. 3.1) |
| `η` (eta) | 0.13 | – | axial exchange-flow ratio between the two CSTs | same |
| `α` (alpha) | 1.14 | – | ratio of film mass-transfer coefficients (tracer/buffer) | same |
| `Δc_max` (dcmax) | 2.17e-7 | see note | film-resistance saturation scale | same — **unit ambiguity, see below** |
| `β` (beta) | 4.49 | g/M | antibody↔NaNO₃ concentration equivalence `c_mAb = β·c_NaNO₃` | same (Sec. 3.4) |

`l, η, α, Δc_max` are the four fitted parameters of Eq. 10; defaults live in
`rtd/filter_model.py` and `rtd/equipment.py::Filter`.

## Geometry & dispersion

| Symbol | Value | Units | Meaning | Source |
|--------|-------|-------|---------|--------|
| `Pe` | 0.5 | – | Péclet number for the DPF dispersion `D_ax = u·d/Pe` | paper, Sec. 2.2.1 |
| `A` | `V/L` | mm² | conduit cross-section, taken from tabulated hold-up so MRT = V/V̇ | this reconstruction (see `docs/NUMERICS.md`) |
| `d` | per unit | mm | conduit diameter (sets the dispersion length scale) | Table 1 |
| `L_equiv` | 108 | mm (10.8 cm) | equivalent-cylinder length for filter compartments | paper, Fig. 2 caption |
| `d̄` | `2√(V/(πL))` | mm | equivalent-cylinder diameter of a filter compartment | paper, Fig. 2 caption |
| `A₃` | 3 | cm² | reference (smallest) filter surface area in the A₃/Aⱼ ratio | Table 1 / Eq. 7 |
| filter `V_I, V_wall, V_O` | per size | µL | filter compartment hold-ups (3/10/100 cm²) | Table 1 (encoded in `equipment.FILTERS`) |
| peripheral volumes/lengths/diameters | per unit | µL / mm / mm | tubing, loop, mixer, valves, monitors | Table 1 (encoded in `equipment.PERIPHERAL`) |

## Detector constants (`rtd/detectors.py`)

| Symbol | Value | Units | Meaning | Source / status |
|--------|-------|-------|---------|-----------------|
| `COND_NANO3` | 121.5 | mS·cm⁻¹ per (mol/L) | NaNO₃ conductivity response (Kohlrausch, Eq. 12) | limiting molar conductivity of NaNO₃ ≈121.5 S·cm²/mol; gives 0.05 M → ~6.1 mS/cm, matching the measured baseline. **Physically grounded.** |
| `UV_NANO3` | 430 | mAU per (mol/L) | nitrate UV₂₈₀ response (Beer, Eq. 11) | **Illustrative calibration** — chosen to give paper-like magnitudes (tens–hundreds of mAU); scales magnitude only, not shape. Re-fit for quantitative UV. |
| UV path length | 2 | mm | UV monitor path length (folded into `UV_NANO3` here) | paper, Sec. 2.2.5 |
| cond. baseline | 0 | mS/cm | buffer background added to conductivity in the figures | illustrative (`rtd.experiments.COND_BASELINE`) |
| `COND_NANO3_L0`, `COND_NANO3_K` | 134, 97 | mS·cm²/mol-ish | NaNO₃ conductivity via Kohlrausch √c law `κ = c(Λ₀−K√c)` (`cond_nano3`) | **illustrative** — 0.05 M→5.6, 0.1 M→10.3, 0.5 M→33 mS/cm (both dilute values below the 11.9 buffer → "U") |
| `COND_BUFFER` | 11.9 | mS/cm per (unit) | tris-acetate buffer conductivity (multi-component, #2) | **illustrative** — set to the measured ~11.9 baseline |
| `UV_BUFFER` | 0 | mAU per (unit) | tris-acetate UV₂₈₀ (≈none) | physically ~0 |
| `UV_ANTIBODY` | 90 | mAU per (g/L) | IgG UV₂₈₀ response | **illustrative** |
| `COND_ANTIBODY` | 0.5 | mS/cm per (g/L) | IgG conductivity (small) | **illustrative** |

## Numerics (`rtd/units.py`)

| Symbol | Default | Units | Meaning | Source |
|--------|---------|-------|---------|--------|
| `n_cells` | 40 | – | DPF finite-volume cells | this reconstruction; convergence in `docs/DISCRETIZATION.md` (≈160 for grid independence) |
| `DPF_SCHEME` | "vanleer" | – | DPF advection scheme (van Leer MUSCL / upwind) | improvement #6 |
| BDF `rtol` / `atol` | 1e-6 / 1e-9 | – | time-integration tolerances | this reconstruction |
| `max_step` | adaptive | s | caps the solver step to resolve pulses/flow features | `compute_max_step` |

## Flow-profile parameters (`rtd/experiments.py`, illustrative for the figures)

| Symbol | Value | Units | Meaning | Source |
|--------|-------|-------|---------|--------|
| `PUMP_LAG_S` | 6 | s | pump start-up delay to set-point (gradient pattern) | paper, Sec. 4 ("~6-s delay") |
| `SAWTOOTH_PERIOD_S` | 15 | s | saw-tooth period at high flow | **illustrative** (paper used the measured trace; exact value not tabulated) |
| saw-tooth amplitude | 0.5–1.0 × set-point | mL/min | high-flow interruption depth | **illustrative** |

## The one unresolved item: `Δc_max` units

Eq. 8 combines `u^{1/3}·(A₃/Aⱼ)·(Δc_eq + α·Δc)/Δc_max`. For this to be
dimensionless, `Δc_max` must carry units of `(m/s)^{1/3}·(mol/L)` — but it is
also tied to a mass-transfer scale `k_m,eq` that the paper does **not** tabulate
(only the composite `k_m,eq·Δc_max = 2.17e-7` is reported, quoted as
"M/(s/m)"). With normalised concentrations the raw argument is enormous, so the
code uses a **saturated surrogate** (`tanh`) that reproduces the qualitative
behaviour (ε→1 at equilibrium, ε→floor at large driving force) but is **not**
quantitatively pinned. Resolving this needs either the paper's `k_m,eq` or a
re-fit against experimental data (ties into backlog items 5 and 3). This is the
single parameter whose absolute scale the reconstruction cannot currently
justify from first principles; everything else above is either measured,
tabulated, or an explicitly-flagged illustrative choice.
