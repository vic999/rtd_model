# Flow profiles — how they work and how to configure them

The model now supports a **time-varying flow rate** `V̇(t)`. Anywhere a flow
rate is accepted (`run_train`, `cst_outlet`, `dpf_outlet`, `filter_outlet`,
`Unit.propagate`, `Filter.propagate`, `pulse_inlet`) you can pass **either**:

- a plain number — mL/min, constant (the original behaviour, unchanged), or
- a **FlowProfile** — a callable `t[s] → V̇[mL/min]`.

Everything is in `rtd/flow.py` and re-exported from `rtd`.

## Why time-varying flow matters

The solvers evaluate `V̇(t)` *inside* the differential equations, so a changing
flow correctly changes every flow-dependent quantity in real time:

| Model | quantity that becomes time-dependent |
|-------|--------------------------------------|
| CST   | `τ⁻¹(t) = V̇(t)/V` |
| DPF   | `u(t) = V̇(t)/A` and `D_ax(t) = u(t)·d/Pe` |
| Filter permeate | `V̇(t)` in Eqs. 3–4, `u_side(t)` in the film term |

This is what reproduces the paper's start-up ramp (the pump reaches its set-point
~6 s late) and the high-flow saw-tooth, and it lets you replay a *measured* flow
trace.

## The profiles

All are callables; all accept a scalar or array `t` (seconds) and return
mL/min. Import from `rtd`.

### `Constant(v)`
Flat flow. `Constant(1.0)` is identical to passing the number `1.0`
(there is a regression test asserting this).

### `Ramp(v_final, t_start=0.0, t_ramp=30.0, v_start=0.0)`
Linear ramp: holds `v_start` until `t_start`, rises linearly to `v_final` over
`t_ramp` seconds, then holds `v_final`.

```python
from rtd import Ramp
flow = Ramp(v_final=10.0, t_start=0.0, t_ramp=6.0, v_start=0.0)  # 0 → 10 in 6 s
```

Use it for a gradient flow increase or a soft start-up.

### `DelayedStep(v, lag=6.0, t_start=0.0, v_start=0.0)`
A step to `v` reached after a short linear lead-in of `lag` seconds — the
paper's ~6 s pump delay. Equivalent to `Ramp(v, t_start, lag, v_start)`.

```python
from rtd import DelayedStep
flow = DelayedStep(10.0, lag=6.0)     # reaches 10 mL/min 6 s after t=0
```

### `Sawtooth(v_base, v_peak, period, t_start=0.0)`
Periodic saw-tooth between `v_base` and `v_peak`; each cycle ramps up over
`period` seconds then resets. Reproduces the high-flow interruption/recovery
pattern (paper Fig. 3g, 4).

```python
from rtd import Sawtooth
flow = Sawtooth(v_base=8.0, v_peak=12.0, period=20.0)
```

### `FromData(t, v)`
Interpolate a measured flow trace — e.g. the *System flow* column of an ÄKTA
CSV, so a real run is driven by its true flow.

```python
import numpy as np
from rtd import FromData
from rtd.data import load_unicorn_csv
d = load_unicorn_csv("run.csv")
flow = FromData(d["flow_t"], d["flow"])       # t already in seconds
```

### `Piecewise([(t_from, profile), ...])`
Concatenate profiles over time windows; the last profile whose `t_from ≤ t`
applies. Good for start-up → hold → shut-down.

```python
from rtd import Piecewise, Ramp, Constant
flow = Piecewise([
    (0.0,   Ramp(10.0, t_ramp=6.0)),   # ramp up
    (600.0, Constant(10.0)),           # steady
    (3600.0, Ramp(0.0, t_start=3600.0, t_ramp=6.0, v_start=10.0)),  # ramp down
])
```

## Using a profile

Just pass it wherever a flow goes:

```python
from rtd import build_train, run_train, Ramp, beer_uv
from rtd.injection import pulse_inlet
import numpy as np

seq, names, uv_i, cond_i = build_train("connector")
t = np.linspace(0, 300, 2000)
flow = Ramp(2.0, t_ramp=40, v_start=0.2)

c_in = pulse_inlet(t, 260.0, flow, c_tracer=0.5, t_start=5.0)  # volume-based
signals, _ = run_train(seq, t, c_in, flow, read_indices=[uv_i, cond_i])
uv = beer_uv(signals[uv_i])
```

`demo_flow_profiles.py` runs exactly this for four profiles and plots the result.

## Injection under varying flow (important)

A sample loop delivers a fixed **volume** (260 µL), not a fixed duration.
`pulse_inlet` therefore integrates the flow and ends the pulse when the
*delivered volume* reaches the loop volume:

```
∫_{t_start}^{t_end} V̇(t) dt = V_loop
```

At constant flow this is exactly the old `width = V_loop / V̇`. Under a ramp the
pulse is physically shorter in time when the flow is higher. Stepwise / gradient
inputs stay time-based (they are pump *composition* changes, not volume
deliveries), so `step_inlet` is unchanged.

## Configuration guidance / gotchas

- **Units:** profiles return **mL/min**; time is **seconds**. Internally the
  solvers convert to µL/s.
- **`max_step`:** `compute_max_step` now also inspects the flow profile and caps
  the step so a fast ramp or saw-tooth edge is resolved. If you build a very
  sharp custom profile, make sure the time grid `t` samples it finely enough.
- **Time-window sizing:** helper code that picks a simulation end time from a
  "mean residence time" uses a *representative* (set-point) flow —
  `representative_flow(flow, t_grid)` returns the max of the sampled profile. For
  a slow start-up ramp you may want a longer window than the set-point implies.
- **Zero flow:** avoid `V̇(t)=0` for sustained periods — `τ⁻¹` and `u` go to
  zero (nothing moves) and residence times diverge. Ramps starting from a small
  positive `v_start` (e.g. 0.2) are safer than starting from exactly 0 if the
  injection happens during the ramp.
- **Backward compatibility:** every existing call that passes a number keeps
  working and produces identical results (`Constant` regression test in
  `verify.py`).

## What is conserved under varying flow

At constant flow the area under `c(t)` is conserved. **Under varying flow the
conserved quantity is mass**, `∫ c·V̇ dt`, not area under `c`. `verify.py`
checks `∫c_out·V̇dt / ∫c_in·V̇dt = 1` for a ramp profile.
