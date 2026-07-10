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
│   └── simulate.py      # run_train() + r2_score()
├── run_figures.py       # reproduces Figure 3 (pulse) and Figure 4 (stepwise)
├── verify.py            # physics checks (mass balance, MRT, gain, ordering)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python3 run_figures.py     # -> figure3_calibration.png, figure4_validation.png
python3 verify.py          # -> prints PASS/FAIL on the conservation checks
```

## Important caveat about "verifying the results"

You confirmed the **raw experimental tracer time series are not available**.
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

- **Constant flow rate.** The paper adds a pump ramp/sawtooth to capture the
  6-second delay of the flow controller (visible as the peculiar shapes at
  10 mL/min). Here flow is held at the setpoint; this affects only fine detail
  of the start-up/flow-change transients, not the RTD structure.
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
