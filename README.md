# RTD model for continuous virus filtration — code reproduction

A from-scratch Python re-implementation of the mechanistic residence-time-distribution
(RTD) model described in:

> Chen, Y.-C., Recanati, G., De Mathia, F., Lin, D.-Q. & Jungbauer, A. (2024).
> *Residence time distribution in continuous virus filtration.*
> Biotechnology and Bioengineering 121:1876–1888. DOI: 10.1002/bit.28696

The paper's own code was not published, so this is an independent reconstruction
built directly from the equations, Table 1 (equipment geometry), Table 2
(experiment matrix) and the calibrated parameters reported in the text. It lets
you re-run the model and reproduce the shapes of the model curves in Figures 3
and 4.

## What is modelled

The ÄKTA + Planova BioEX system is represented as a **series of RTD units** (no
downstream feedback), so the whole train is solved unit-by-unit: the outlet of
one unit is the inlet of the next.

| Model | Paper eq. | Used for | File |
|-------|-----------|----------|------|
| Dispersed plug flow (DPF) | Eq. 1 | tubing, sample loop, nozzles, filter compartment `V_I` | `rtd/units.py` |
| Continuously stirred tank (CST) | Eq. 2 | mixer, connector, valves, monitors, filter wall `V_wall` | `rtd/units.py` |
| Three-compartment filter | Eqs. 3–9 | Planova BioEX filter (`V_I` → `V_wall` → `V_O`) | `rtd/filter_model.py` |

The DPF PDE is solved by a finite-volume **method of lines** (upwind advection +
central diffusion) with the Danckwerts inlet boundary condition and a Neumann
outlet, using `Pe = 0.5` and `Dax = u·d/Pe` as stated in the paper. The permeate
space `V_O` is a **radial tanks-in-series cascade** of `l` stages, each an
**axial pair of interconnected CSTs** (a through-flow tank `k1` and an exchange /
dead-zone tank `k2` coupled at flow ratio `η`), with a film-resistance term that
makes the through-flow volume fraction `ε` time-dependent.

> **How each equation is actually discretized and integrated is documented in
> [`docs/NUMERICS.md`](docs/NUMERICS.md)** — spatial schemes, boundary-condition
> handling, the BDF time integrator, the `max_step` logic, and how the units are
> chained.

Calibrated parameters from the paper are the defaults:

```
l = 3,  η = 0.13,  α = 1.14,  Δc_max = 2.17e-7   (β = 4.49 g/M for antibody)
```

## Layout

```
rtd_model/
├── rtd/
│   ├── units.py         # DPF + CST solvers, adaptive max_step helper
│   ├── filter_model.py  # three-compartment virus filter (Eqs. 3–9)
│   ├── equipment.py     # Table 1 geometry + train assembly (Fig. 1 connections)
│   ├── injection.py     # pulse / stepwise / combined inlet signals
│   ├── flow.py          # FlowProfile: Constant/Ramp/DelayedStep/Sawtooth/FromData
│   ├── detectors.py     # Beer's-law UV (mAU) + Kohlrausch conductivity (mS/cm)
│   ├── data.py          # ÄKTA UNICORN CSV loader + event detection
│   └── simulate.py      # run_train() + r2_score()
├── run_figures.py       # reproduces Figure 3 (pulse) and Figure 4 (stepwise)
├── compare_data.py      # compare model vs a real ÄKTA run; infer configuration
├── demo_flow_profiles.py# shows variable flow reshaping the RTD response
├── verify.py            # physics checks (mass balance, MRT, gain, flow, ordering)
├── docs/
│   ├── NUMERICS.md      # how every equation is discretized and solved
│   ├── FLOW_PROFILES.md # variable flow: how it works and how to configure it
│   └── PLANS.md         # design plans + improvement backlog
├── requirements.txt
└── README.md
```

## Detector signals (UV and conductivity)

The RTD solvers produce a tracer **concentration** `c(t)`. `rtd/detectors.py`
converts that to the two instrument signals the paper actually plots:

- `beer_uv(c)` → UV₂₈₀ absorbance in **mAU** (Beer's law, Eq. 11)
- `kohlrausch_cond(c)` → conductivity in **mS/cm** (Kohlrausch's law, Eq. 12)

For a single tracer both share the same *shape* but sit on different-magnitude
axes; for multiple species they also differ in shape. The molar constants
(`UV_NANO3`, `COND_NANO3`) are documented in `detectors.py`; the conductivity one
comes from NaNO₃'s molar conductivity, the UV one is an illustrative calibration
(magnitude only) pending a fit. `run_figures.py` plots UV, conductivity and flow
on three separate axes.

## Plot style

`run_figures.py` has a `PLOT_STYLE` flag (default `"paper"`):

- `"paper"` — the Chen et al. (2024) layout: three stacked sub-panels per
  experiment sharing the time axis — UV (mAU, blue) on top, flow rate (mL/min,
  orange) in the middle, conductivity (mS/cm, green) at the bottom.
- `"overlay"` — compact: UV, conductivity and flow overlaid on three y-axes of a
  single panel per experiment.

Set the constant in the file, or override per run on the command line:

```bash
python3 run_figures.py            # uses PLOT_STYLE (default "paper")
python3 run_figures.py overlay    # force the overlay layout
```

### X-axis auto-focus

The full window is always simulated (tails and mass balance stay intact) but the
plot is cropped to where the signal actually lives — like the paper's tight
x-axes — instead of trailing a long flat tail. It's controlled by flags near the
top of `run_figures.py`:

- `FOCUS_ENABLED` (default `True`) — turn the auto-crop on/off.
- `FOCUS_FRAC` (default `0.02`) — a signal counts as "back to baseline" below
  this fraction of its peak.
- `FOCUS_MARGIN` (default `0.15`) — head-room added to the right of the active
  region.

Disable it, or force a specific limit per experiment:

```bash
python3 run_figures.py paper nofocus     # show the whole simulated window
```

```python
# In a FIG3/FIG4 entry, add "xmax" to override the auto-crop for that panel:
("C1  (bypass, 1 mL/min)", dict(connection="bypass", flow=1.0, xmax=60.0)),
```

## Variable flow rate

Flow may be a constant number **or** a `FlowProfile` (`Constant`, `Ramp`,
`DelayedStep`, `Sawtooth`, `FromData`, `Piecewise`). The solvers evaluate
`V̇(t)` inside the equations, so ramps and saw-tooth flows are handled correctly,
and a loop pulse is delivered by **volume** (not fixed duration) under varying
flow. Passing a number reproduces the old results exactly. See
[`docs/FLOW_PROFILES.md`](docs/FLOW_PROFILES.md) for the full configuration guide
and `demo_flow_profiles.py` for a worked example.

## How to run

```bash
pip install -r requirements.txt
python3 run_figures.py     # -> figure3_calibration.png, figure4_validation.png
python3 verify.py          # -> prints PASS/FAIL on the conservation checks
```

## Validation against real instrument data

`compare_data.py` compares the model against a real ÄKTA UNICORN export
(`20230208_260Microliters0.5MPulseson50mMNaNO3Transition_FDM 002.csv`) — a
combined stepwise + pulse run (260 µL / 0.5 M NaNO₃ pulse on a 50 mM NaNO₃
transition, 1 mL/min).

```bash
python3 compare_data.py            # uses the bundled CSV, or pass a path
```

What it does:

1. `rtd/data.py` parses the interleaved UV / conductivity / flow curves and
   resamples them onto a common time grid (s).
2. The two independent inputs are reconstructed — a **gradient ramp** for the
   buffer transition and a **rectangular 260 µL pulse** — and each is propagated
   through a candidate train to give basis response shapes.
3. Each detector is fit as `signal = a·(transition response) + b·(pulse
   response) + c` by least squares: `a, b, c` are the Beer / Kohlrausch
   calibration constants (units + baseline), so **shape agreement is what R²
   measures**, not amplitude.
4. It repeats for bypass / connector / 3·10·100 cm² filter and reports which
   reproduces the data best.

**Result on the supplied file:** the run is a **no-filter** configuration —
bypass and connector both give R² ≈ 0.87 (UV and conductivity), while every
filter configuration fits markedly worse (0.47–0.73). This is consistent with
the paper's C-series connector/bypass experiments at 1 mL/min.

Two honest observations, both visible in `data_comparison.png`:

- The buffer transition in the data is a **programmed gradient** (a slow ramp),
  not a step, so it is driven by the pump programme and only weakly probes the
  RTD. It is modelled as a ramp for context.
- The **pulse** is the genuine RTD test, and the measured pulse is **broader and
  more tailed** than the ideal model with published parameters predicts. This is
  the exact small-scale tailing limitation the authors themselves report for
  these experiments ("experimental profiles revealed more pronounced tailing
  that did not fully match the simulated signals… dominating at very small
  scale"). Closing that gap to the paper's ~0.96 needs the **inverse
  calibration** (Eq. 10) — fitting the dispersion / mixing parameters to this
  data rather than using the published values — plus the pump-ramp refinement.

## Important caveat about "verifying the results"

The published R² values (Table 3) compare the model to those measurements, so
they cannot be recomputed here. What this code does instead is a **forward
simulation** with the paper's calibrated parameters, reproducing the *model*
curves (the solid lines in Figures 3 and 4) and confirming the model is
self-consistent (`verify.py`).

To go further and reproduce the R² numbers, drop the experimental UV /
conductivity CSVs into the project and:

1. read each experiment's UV and conductivity trace,
2. simulate the matching train with `run_train`,
3. convert simulated concentration to signal (Beer's law for UV, Kohlrausch's
   law for conductivity — Eqs. 11–12), and
4. call `rtd.r2_score(experiment, simulation)`.

The `r2_score` function is already provided and matches
`sklearn.metrics.r2_score` (the paper's metric).

## Modelling assumptions & where this departs from the paper

These are documented so the reconstruction is auditable:

- **Flow rate.** Time-varying flow is supported (`rtd/flow.py`): the solvers
  evaluate `V̇(t)` inside the equations, and a measured flow trace can be
  replayed (`FromData`). The reproduced Figures 3/4 now drive each experiment
  with the **paper's flow pattern** (`experiment_flow` in `run_figures.py`): a
  delayed pump ramp to the set-point at low flow (the "gradient pattern",
  Fig 3a,b) and a saw-tooth at 10 mL/min (flow interruption, Fig 3g, 4a–c,f,i,l).
  The saw-tooth amplitude/period are illustrative (the paper used the measured
  flow trace; exact values are not tabulated). See `docs/FLOW_PROFILES.md`.
- **Film-resistance ε(t) (Eqs. 5–9).** This is the most under-specified part of
  the paper: Eq. 8 needs the equilibration-buffer driving force `Δc_eq` and the
  mass-transfer scale `k_m,eq`, neither of which is tabulated, and with the
  reported `Δc_max` and normalised concentrations the raw argument is enormous.
  The full expression is implemented (`film_resistance=True`) but saturated so
  it stays in `[ε_floor, 1]` and the ODE stays non-stiff; it reproduces the
  *qualitative* behaviour (ε→1 near equilibrium, ε→floor at large Δc). The
  runnable figures default to `film_resistance=False` with a constant
  through-flow fraction `ε_const = 0.85`, which is numerically stable and
  reproduces the RTD curve **shapes** (peak position and the pronounced filter
  tailing). Quantitative use of the film term needs the missing scale factor.
- **Equivalent-cylinder geometry.** Filter compartments use the paper's
  `d̄ = 2√(V/(πL))` with `L = 10.8 cm`; peripheral DPF units take `A = V/L` from
  the tabulated hold-up so the mean residence time is exactly `V/V̇`.
- **Detector read-out.** UV is read at the U9-D monitor and conductivity one CST
  later at C9, so the conductivity trace is slightly delayed vs. UV — as seen in
  the bypass/connector panels.
- **Solver.** `scipy.integrate.solve_ivp` with the implicit **BDF** method
  (LSODA was found to silently emit NaNs on long flat tails). `max_step` is
  chosen adaptively to resolve narrow injection pulses without over-resolving
  the rest of the run.

## Extending

- Different experiment: call `build_train(connection, surface_cm2=...)` with
  `connection` in `{"bypass", "connector", "filter"}`, then `run_train(...)`.
- Turn the film-resistance term on: pass `film_resistance=True` to `Filter`.
- Calibrate parameters against your own data: wrap `run_train` in
  `scipy.optimize.minimize` over `(l, η, α, Δc_max)`, exactly as Eq. 10.
