# Plans: detector scaling, variable flow rate, and an improvement backlog

This document started as planning only. Items that have since been **built** are
marked inline. Nothing has been removed — the original plan text is kept for
context, with a status tag added.

Status legend: **✅ IMPLEMENTED** · **◐ PARTIAL** · **⬜ NOT YET** · **✖ CLOSED (won't do)**.

It covers three things you asked about:

1. Why UV and conductivity currently share the same scale, and the plan to fix
   it so the figures behave like the paper.  **✅ IMPLEMENTED**
2. A plan to remove the constant-flow-rate assumption and allow a varying (e.g.
   ramped) flow rate.  **✅ IMPLEMENTED**
3. A backlog of "good to have" improvements (partially implemented — see tags).

### Implementation summary (added after the plan was written)

- **✅ Detector model** — `rtd/detectors.py` (`beer_uv`, `kohlrausch_cond`);
  figures now plot mAU and mS/cm on separate axes.
- **✅ Variable flow** — `rtd/flow.py` (`Constant`, `Ramp`, `DelayedStep`,
  `Sawtooth`, `FromData`, `Piecewise`); solvers evaluate `V̇(t)`; volume-based
  injection; docs in `docs/FLOW_PROFILES.md`; demo in `demo_flow_profiles.py`.
- **✅ Flow on plots** — third axis in `run_figures.py` and `compare_data.py`.
- **✅ Auto configuration/parameter detection** — `rtd.data.detect_run_parameters`;
  `compare_data.py` now auto-detects flow (drives the model via `FromData`),
  transition edges, and pulse presence from any raw CSV (backlog item 9).
- **✅ Verification** — `verify.py` adds constant-flow regression and
  mass-conservation-under-varying-flow checks.
- **✅ Performance** — analytic sparse Jacobians, vectorised permeate RHS,
  cached basis responses, optional Numba, and a window-sizing fix (backlog
  item 10). Same results, roughly 2× faster.
- **✅ Higher-fidelity DPF** — van Leer (MUSCL) flux-limited advection, default
  scheme, with a grid-convergence study (backlog item 6). Cuts numerical
  diffusion; see `docs/DISCRETIZATION.md`.
- **✅ YAML experiments + CLI** — `experiments.yaml` + `rtd_cli.py`
  (list/figure/plot/csv) with `--help` (backlog item 14); injection-point
  modelling (item 12) and the parameter-provenance audit `docs/PARAMETERS.md`
  (item 13); item 7 closed as superseded.
- **✅ Multi-component tracers** — species-resolved propagation + per-species
  detectors; opposite-sign transition demo `TR1` (item 2); `docs/MULTICOMPONENT.md`.
- **✅ Test suite + CI** — `tests/` (pytest) + GitHub Actions (item 11).
- **✅ Packaging & typing** — `pyproject.toml`, `rtd` console entry point,
  `py.typed` (item 17).

---

## 1. UV and conductivity share the same scale — diagnosis & fix — ✅ IMPLEMENTED

> **Done:** `rtd/detectors.py` provides `beer_uv` (mAU) and `kohlrausch_cond`
> (mS/cm); `run_figures.py` plots them on separate axes. The plan below is the
> original write-up.

### What is happening (verified)

`run_train` propagates a single tracer **concentration** `c(t)` (mol/L) through
the train. In `run_figures.py` we then plot:

- left axis: `signals[uv_i]`  = concentration at the UV monitor U9-D
- right axis: `signals[cond_i]` = concentration at the conductivity monitor C9

These are the *same quantity* read ~1 CST (22 µL) apart. Measured directly:

```
UV signal  : max 0.3673        Cond signal: max 0.3601      Cond/UV peak ratio 0.98
```

So both twin axes span ~0–0.4 mol/L and look identical. The RTD math is correct;
what's missing is the **detector conversion layer**. The paper does not plot
concentration — it plots two physically different transducer signals:

- **UV absorbance (mAU)** via **Beer's law** (paper Eq. 11):
  `A_UV(t) = Σ_i ε_i · L_path · c_i(t)  (+ baseline)`
  with `ε_i` the molar absorptivity of species *i* at 280 nm and
  `L_path = 2 mm` (the UV monitor path length stated in the paper).
- **Conductivity (mS/cm)** via **Kohlrausch's law** (paper Eq. 12):
  `κ(t) = κ_buffer + Σ_i Λ_i · c_i(t)`
  with `Λ_i` the molar conductivity of species *i*.

For a **single tracer** (the C-series NaNO₃ pulses) both signals are
proportional to the same `c(t)`, so their *shapes* are identical — that part is
correct and matches the paper. The point is they must sit on **different y-axes
with different magnitudes and baselines** (UV ~0–200 mAU rising from 0;
conductivity ~11–15 mS/cm sitting on a buffer baseline), which only happens once
Beer/Kohlrausch scaling is applied. For **multi-component** runs (tris-acetate +
NaNO₃, i.e. the transition/combined experiments) the two signals also differ in
*shape*, because UV and conductivity weight the species differently — this is
exactly the opposite-sign behaviour seen in the real uploaded data.

### Fix plan

1. **New module `rtd/detectors.py`.**
   - `beer_uv(concentrations: dict[str, ndarray], eps: dict, L_path_mm, baseline=0.0)`
   - `kohlrausch_cond(concentrations: dict[str, ndarray], Lambda: dict, baseline)`
   Each takes one array per chemical species and returns the detector signal.

2. **Species-resolved propagation.** Because every unit is linear in
   concentration, propagate each species independently (or scale a single
   propagated tracer) and combine at the detector. For the single-tracer figures
   this is just one species; keep the current propagation and apply the scale.

3. **Constant provenance.** Populate `eps`, `Lambda`, `κ_buffer`, `L_path`
   from literature (NaNO₃ molar conductivity ≈ 121 S·cm²/mol at infinite
   dilution; nitrate UV absorptivity at 280 nm) or leave them as clearly-labelled
   calibration constants. For the *forward* figures these set only magnitude, not
   shape, so illustrative values that reproduce the paper's y-ranges are fine and
   must be documented as such.

4. **Plot changes in `run_figures.py`** (keep your new combined legend):
   - left axis label → `UV₂₈₀ (mAU)`, right axis label → `Conductivity (mS/cm)`,
   - feed `uv = beer_uv(...)`, `cond = kohlrausch_cond(...)` into the two axes,
   - let each axis autoscale to its own physical range.

5. **Validation.**
   - Single tracer: normalized UV and normalized conductivity shapes must
     coincide (they already do); only magnitudes/baselines differ.
   - Magnitudes land in the paper's ballpark (UV tens–hundreds of mAU,
     conductivity ~10–15 mS/cm).
   - Multi-component test (once species support exists): UV and conductivity
     show *different* shapes for the tris-acetate/NaNO₃ transition, reproducing
     the sign difference in the measured data.

Note: reading UV at U9-D and conductivity one CST later at C9 is physically
correct (the monitors are in series) and should be **kept** — it is the small
UV→conductivity lag visible in the paper's panels.

---

## 2. Removing the constant-flow-rate assumption — ✅ IMPLEMENTED

> **Done:** `rtd/flow.py` implements the `FlowProfile` abstraction
> (`Constant`, `Ramp`, `DelayedStep`, `Sawtooth`, `FromData`, `Piecewise`); all
> solvers evaluate `V̇(t)` inside the RHS; `pulse_inlet` is volume-based; and
> `compute_max_step` resolves flow features. Guide: `docs/FLOW_PROFILES.md`.
> The plan below is the original write-up.

Today every solver takes a scalar `flow_mL_min` and treats `V̇` as constant. The
paper instead models `V̇ = V̇(t)` — a delayed ramp on start-up (the pump reaches
setpoint ~6 s late) and a sawtooth at high flow when a second tracer is
introduced. Supporting a time-varying flow is a contained but cross-cutting
change.

### Physics — what depends on flow

| Model | Dependence on `V̇` | Effect of `V̇(t)` |
|-------|--------------------|-------------------|
| CST (Eq. 2) | `dc/dt = (V̇/V)(c_in − c)` | coefficient becomes time-dependent |
| DPF (Eq. 1) | `u = V̇/A`, `D_ax = u·d/Pe` | both advection and dispersion coefficients vary in time |
| Filter permeate (Eqs. 3–9) | `V̇` throughout; `u_side = V̇/(π d̄ L)`; film term | all terms and `ε(t)` vary |

The differential equations stay valid; the right-hand sides simply evaluate
`V̇(t)` at the current time. BDF handles the resulting non-autonomous system;
`max_step` must additionally resolve rapid flow changes.

### Design

1. **A `FlowProfile` abstraction** returning `V̇(t)` in mL/min:
   - `Constant(v)` — wraps a scalar (keeps every current call working);
   - `Ramp(v_final, t_start, t_ramp)` — linear rise (models the delayed
     setpoint / gradient flow increase);
   - `DelayedStep(v, lag=6.0)` — the ~6 s pump lead-in the paper describes;
   - `Sawtooth(...)` — the high-flow overpressure pattern (Fig. 3g, 4);
   - `FromData(t, v)` — interpolate the **measured System-flow column** (already
     present in the ÄKTA CSVs), so a real run can be reproduced with its true
     flow trace.

2. **Thread a flow callable** (instead of a scalar) through
   `cst_outlet`, `dpf_outlet`, `permeate_space_outlet`, `filter_outlet`,
   `Unit.propagate`, `Filter.propagate`, and `run_train`. Inside each `rhs`,
   compute `Vdot = flow(t)` and derive `u(t)`, `D_ax(t)`, `τ⁻¹(t)`.
   Accept a scalar too (auto-wrap as `Constant`) for backward compatibility.

3. **Injection under varying flow.** A loop pulse delivers a fixed **volume**
   (260 µL), not a fixed duration. The bolus lasts until the *cumulative*
   delivered volume reaches the loop volume:
   `∫_{t_start}^{t_end} V̇(t) dt = V_loop`.
   `injection.pulse_inlet` must therefore integrate the flow to find the pulse
   end, rather than using `V_loop / V̇`. Stepwise/gradient inputs stay
   time-based (they are pump *composition* changes, independent of flow).

4. **Step control.** Extend `compute_max_step` to also cap the step at a
   fraction of the shortest flow-change feature (e.g. `t_ramp/4`, or the pump
   lag), so a fast ramp is resolved.

5. **Time-window sizing** in `run_figures` should use a representative
   (steady-state) flow for choosing `t_end`, or integrate the delivered volume.

### Validation

- **Constant-flow limit:** `Constant(v)` must reproduce today's results exactly
  (regression test).
- **Mass conservation under varying flow:** the conserved quantity is *mass*,
  `∫ c_out·V̇ dt = ∫ c_in·V̇ dt`, **not** area under `c(t)` (area alone is only
  conserved at constant flow). Add this as an explicit check.
- **Real-run check:** drive the model with `FromData` using the measured flow
  column of the uploaded CSV and confirm the start-up/shut-down transients
  improve versus the constant-flow assumption.

### Why it matters

Variable flow is exactly what the paper invokes to explain the "peculiar" shapes
at 10 mL/min (delayed ramp, sawtooth). Adding it is also the prerequisite for
pushing R² on those high-flow experiments toward the paper's values.

---

## 3. Improvement backlog (good to have — not to be built now)

Roughly in priority order for matching the paper and hardening the code:

1. **✅ IMPLEMENTED — Detector model (Beer/Kohlrausch)** — the §1 fix, as its own
   module (`rtd/detectors.py`). Highest value: makes UV and conductivity
   physically distinct and paper-like.
2. **✅ IMPLEMENTED — Multi-component tracer support** — propagate tris-acetate
   and NaNO₃ separately and combine per detector.
   *Done:* an experiment can declare a `species:` list (each with
   baseline/step/pulse components); `simulate` propagates each species
   independently through the same train (superposition — valid at constant ε)
   with `run_train(c0=baseline)` so a background buffer starts pre-equilibrated,
   then forms UV and conductivity as per-species weighted sums
   (`detectors.uv_from_species` / `cond_from_species`, with `SPECIES_UV`/
   `SPECIES_COND`). The `TR1` demo reproduces the **opposite-sign** behaviour
   (conductivity drops while UV rises across the transition, both spike on the
   pulse); combined same-species experiments `V5/V6-2/V7` are also bundled.
   The C*/V* experiments now run against a default **equilibration buffer**
   (`defaults.background` in the YAML), so their conductivity matches the paper
   — a **U** for the V-series steps (buffer displaced by NaNO₃) and a pedestal
   for the C-series pulses — at no extra solve (the buffer is the analytic
   complement of the step). Write-up: `docs/MULTICOMPONENT.md`.
3. **⬜ NOT YET — Inverse calibration (Eq. 10)** — wrap the model in
   `scipy.optimize` to fit `(l, η, α, Δc_max)` plus the detector constants to a
   measured run and compute R²; this is what reproduces Table 3 and closes the
   ~0.87→0.96 gap.
4. **✅ IMPLEMENTED — Time-varying flow + pump ramp/delay/sawtooth** — the §2
   plan (`rtd/flow.py`, all profiles including `DelayedStep` and `Sawtooth`).
5. **⬜ NOT YET — Physically-scaled film resistance ε(t)** — replace the current
   saturated surrogate with the true Graetz–Lévêque scaling once `k_m,eq` (and
   the equilibration driving force) are pinned from data or literature.
6. **✅ IMPLEMENTED — Higher-fidelity DPF discretization** — a flux-limited
   **van Leer (MUSCL)** advection scheme (conservative total-flux form,
   Danckwerts inlet), selectable via `DPF_SCHEME` / `scheme=` and default for all
   conduits; first-order upwind remains available. Cuts numerical diffusion so
   the modelled dispersion matches the physical `Pe = 0.5`, mass and MRT
   preserved. Grid-convergence study in `convergence_study.py`; full write-up in
   `docs/DISCRETIZATION.md`. (A full discontinuous-Galerkin scheme was not
   needed — van Leer already reaches grid independence at ~160 cells where
   upwind is not converged at 640.)
7. **✖ CLOSED — WON'T DO (superseded) — Analytic transfer-function /
   convolution solver** for the linear CST/DPF subsystem — exact and much
   faster; only the (nonlinear) filter then needs ODE integration.
   *Closed because it is superseded by other work.* A convolution/transfer-
   function solve requires a **linear, time-invariant** subsystem, and two
   implemented changes break that premise for the default configuration:
   (a) the default DPF scheme is now the **van Leer flux limiter, which is
   nonlinear** (item 6), and (b) **variable flow** (item 4) makes even the
   linear units time-*varying*, so no single impulse-response kernel applies.
   It would only help the narrow special case of constant flow **and**
   `scheme="upwind"` **and** constant-ε filter, and the intended speed-up was
   already delivered by item 10 (analytic sparse Jacobians + cached basis
   responses). No further action.
8. **⬜ NOT YET — Batch reproduction of Table 3** — loaders for all experiment
   CSVs and an automated R² table across every C/V experiment.
9. **✅ IMPLEMENTED — Automated configuration & parameter detection** from a raw
   CSV — `rtd.data.detect_run_parameters` + generalised `compare_data.py`: the
   flow profile is read from the data (`FromData`) and drives the model, and the
   transition edges and pulse presence are detected automatically.
10. **✅ IMPLEMENTED — Performance** — analytic *sparse* Jacobian for BDF (CST
    1×1, DPF tridiagonal, constant-ε filter block; sparsity pattern otherwise),
    vectorised right-hand sides (permeate stage loop removed), caching of basis
    responses in `compare_data.py`, and optional Numba JIT of the DPF stencil
    (NumPy fallback when absent). Profiling also fixed a window-sizing bug (the
    representative flow was read during the pump ramp), which was the largest
    single speed-up. Results unchanged; `run_figures.py` ~60 s → ~34 s, heavy
    filter panels ~7–8 s → ~3 s.
11. **✅ IMPLEMENTED — Test suite + CI** — `verify.py` still runs the physics
    checks, and there is now a proper **pytest** suite in `tests/` covering
    mass conservation and MRT (CST + both DPF schemes), constant-flow
    regression, varying-flow mass conservation, **DPF grid convergence** (van
    Leer converges where upwind does not), single-species backward
    compatibility, **multi-component opposite-sign behaviour**, injection-point
    routing, and CLI/config plumbing. A **GitHub Actions** workflow
    (`.github/workflows/ci.yml`) runs `pytest` + `verify.py` on Python
    3.10/3.11/3.12.
12. **✅ IMPLEMENTED — Sample-pump vs loop injection** — model the two injection
    points correctly (the paper notes stepwise injection used the sample pump,
    not the loop).
    *Done:* (a) the two injection *mechanisms* are distinct — the loop pulse is
    delivered by **fixed volume** (`pulse_inlet` integrates the flow, correct
    under a ramp/saw-tooth) while the stepwise input is time-based
    (`step_inlet`); and (b) `build_train(..., inject_at=...)` now sets the
    injection *location*, dropping units upstream of it from the tracer path.
    Pulse experiments inject at `"Loop"` (traverse the 260 µL loop); stepwise
    experiments inject at `"5"` (the sample pump enters downstream of the loop,
    so the loop hold-up is not traversed). Wired through the YAML experiment
    config.
13. **✅ IMPLEMENTED — Units & parameter-provenance audit** — resolve the `Δc_max`
    unit ambiguity and document where every constant comes from.
    *Done:* a complete audit table of every parameter — value, **units**, meaning
    and provenance — is written in **`docs/PARAMETERS.md`** (calibrated model
    params, geometry/dispersion, detector constants, numerics, flow profiles).
    The `Δc_max` situation is explicitly analysed: only the composite
    `k_m,eq·Δc_max` is reported by the paper, so its absolute scale cannot be
    pinned without the un-tabulated `k_m,eq`; the code's saturated surrogate is
    documented as such. The *audit* (documenting units/provenance and isolating
    the one unresolvable value) is complete; actually resolving `Δc_max` needs
    external data and lives in items 5/3.
14. **✅ IMPLEMENTED — Experiment configuration files (YAML) + CLI** instead of
    hard-coded `FIG3/FIG4` dicts; a small CLI to run any experiment by name.
    *Assessment & delivery:* the hard-coded experiment lists were the main
    obstacle to reuse — every new experiment meant editing `run_figures.py`.
    Now all experiments (C* and V*) live in **`experiments.yaml`**; add one by
    appending a line, no code change. `rtd/experiments.py` loads them into
    `Experiment` objects and simulates them; `rtd/plots.py` renders them.
    `run_figures.py` was refactored to a thin wrapper (backward-compatible
    shims kept for `verify.py`). A new entry point **`rtd_cli.py`** provides:
    `list` (all experiments, optionally by figure), `figure` (build Figure 3/4
    grids), `plot` (per-experiment high-resolution PNG, DPI/size configurable
    with sensible defaults), and `csv` (export the full simulated data). All
    have `--help`; defaults come from the YAML `defaults:` block. This also
    completed the injection-location wiring for item 12 (pulse→loop,
    step→pump node).
15. **⬜ NOT YET — Temperature/viscosity dependence** — conductivity drifts
    ~2 %/°C and `D_ax` depends on viscosity; relevant if matching absolute
    magnitudes.
16. **⬜ NOT YET — Sensitivity / uncertainty analysis** — how RTD responds to each
    parameter, to guide calibration and report confidence bands.
17. **✅ IMPLEMENTED — Packaging & typing** — `pyproject.toml`, installable
    package, type hints and docstring coverage.
    *Done:* `pyproject.toml` (setuptools) makes the project pip-installable
    (`pip install -e .`), declares dependencies + optional extras (`speed` →
    numba, `dev` → pytest), configures pytest, and adds a **`rtd` console
    entry point** → `rtd_cli:main`. The `rtd` package ships a `py.typed`
    marker (PEP 561); the public API (dataclasses, `simulate`, `load_config`,
    detector/flow functions) carries type hints and docstrings. (Exhaustive
    annotation of every internal helper is not claimed, but the shipped types
    cover the public surface.)
