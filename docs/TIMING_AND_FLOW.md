# Protocol timing & event-driven flow (aligning with the paper)

Status: **✅ IMPLEMENTED.** This makes the reproduced panels line up with the
paper's time axis and flow trace while staying flexible. It follows the V1
diagnosis below. As-built notes are flagged **[as built]** where the
implementation differs from the original proposal.

**[as built]** Summary of what shipped:
- `Experiment.timing` — an optional `{t_on, t_off, t_end, t_pulse}` block per
  experiment; when omitted, the residence-time auto-sizing is used. (The global
  `time_mode` from the proposal was dropped as unnecessary — per-experiment
  `timing` is the switch.)
- `rtd.flow.RampWithDips` + `experiments.build_flow(...)` — the flow is a ramp
  to a constant set-point with a triangular dip toward 0 at each event time
  (transition / pulse), applied only at high flow (≥10 mL/min). `flow` may be a
  scalar or a spec dict (`constant` / `ramp` / `sawtooth` / `from_data` /
  `ramp_dips`).
- `experiments.yaml` — `V1`, `V3`, `V5` carry the paper's `10 / 255 / 300 s`
  timing; the rest stay on auto (their residence-time windows already scale
  sensibly). Tests: `test_protocol_timing_override`, `test_event_driven_flow_dip`.

## The two mismatches (V1 example)

Comparing the reproduced V1 (bypass, 0.05 M, 10 mL/min) with paper Fig. 4a:

| aspect | paper | current model | why |
|--------|-------|---------------|-----|
| time axis | 0–300 s, step held ~245 s | 0–~19 s, step held ~8.7 s | window auto-sized to residence time |
| flow | ramp to a **constant** 10, single **dip toward 0** at shut-down (~255 s) | **continuous saw-tooth** 5↔10, period 15 s, never below 5 | `experiment_flow(≥10)` returns a `Sawtooth(0.5·setpoint … setpoint)` |

Concrete numbers (V1): hold-up 481 µL ÷ 166.5 µL/s → mean residence ≈ **2.9 s**;
auto timing gives `t_on=1.4 s`, `t_off=10.1 s`, `t_end=18.8 s`. The paper's
timing is a protocol choice (~10 / 255 / 300 s), unrelated to the 2.9 s
residence time.

## Guiding principle

Two clocks are being conflated:

- **RTD clock** — the residence time (~3 s); what the *physics* needs to develop
  the transition. Good for a compact, exploratory view.
- **Protocol clock** — the operator's on/off/run times (~10 / 255 / 300 s); what
  the *paper* plots.

Design goal: **default to the RTD clock, allow the protocol clock as an opt-in
override**, and **drive the flow perturbations from protocol events** (not as a
continuous oscillation).

## Part 1 — Timing as an optional override

Add an optional `timing` block per experiment. When present it pins the
protocol times; when absent, fall back to today's residence-time auto-sizing.
**[as built]** the global `time_mode` default was dropped — the per-experiment
`timing` block *is* the opt-in (omit it for auto).

```yaml
- name: V1
  ...
  timing:                  # optional; omit -> auto (residence-time sizing)
    t_on:  10              # step starts (s)
    t_off: 255             # step ends   (s)
    t_end: 300             # total run   (s)
    # t_pulse: 130         # (combined runs) when the pulse fires
```

For pulse experiments the analogous fields are `t_pulse` and `t_end`; for the
combined runs, `t_on`/`t_off` plus `t_pulse` (pulse fired during the plateau).

`simulate()` logic:

```
if experiment has a timing block (or time_mode == protocol):
    use t_on / t_off / t_end (and t_pulse) verbatim
else:
    derive from mean_res as today (t_on = 0.5·mean_res, ...)
```

This yields the paper's 300 s window with the long plateau when you want a 1:1
comparison, and the compact ~19 s view otherwise — no hard-coding of either.

## Part 2 — Event-driven flow (replaces the continuous saw-tooth)

`experiment_flow(setpoint)` currently hard-codes "≥10 mL/min → continuous
saw-tooth". Replace it with a profile that mirrors the instrument:

**Default profile ("ramp + event dips"):**
1. ramp from 0 to the set-point over the pump lag (~6 s) at start-up;
2. hold **constant** at the set-point;
3. apply a brief **interruption toward ~0** at each *flow-change event* — the
   step-off / valve switch (`t_off`) and any pulse introduction (`t_pulse`) —
   then recover.

Built as a `Piecewise` from the same event times used for the concentration
input, so the dip lands exactly at the transition, drops to ~0 (not 5), and is
*localized* rather than continuous. This reproduces the paper's "saw-tooth at
high flow" as the transient it actually is.

**Per-experiment override** (a small flow spec that maps onto the existing
`FlowProfile` classes — nothing new to learn):

```yaml
flow: 10                                          # scalar -> default ramp+dips
flow: {type: constant, setpoint: 10}
flow: {type: ramp, setpoint: 10, lag: 6}
flow: {type: sawtooth, base: 0, peak: 10, period: 15}   # if a run truly oscillates
flow: {type: from_data, csv: "run.csv"}           # replay the measured trace
```

`from_data` (already available as `rtd.flow.FromData`, used by `compare_data.py`)
is the highest-fidelity path: drive V1 with its real recorded flow.

**Controlling the dips.** The interruption need not go all the way to 0, and can
be switched off entirely, via three per-experiment fields:

```yaml
flow_dips: true       # (default) add interruption dips at transitions; false = none
step_flow_dip: 0.0    # bottom flow (mL/min) of the step-off / valve-switch dip
pulse_flow_dip: 2.5   # bottom flow (mL/min) of the pulse dip (e.g. V5: dip to 2.5, not 0)
```

Each event is an `(time, depth)` pair passed to `RampWithDips`, so different
events can dip to different levels. For V5 the pulse fires essentially at
step-off (`t_pulse≈t_off`), so both dips are set to 2.5 and merge into a single
notch bottoming at 2.5. Setting `flow_dips: false` (e.g. for the V8
approximation) removes the automatic shut-down dip so the plateau stays flat.

### Why event-driven

- The paper's flow dips are caused by valve/tracer **events**, so tying them to
  `t_off` / `t_pulse` is physically correct and automatically consistent with
  the signal input.
- Configuring flow events **independently** of the signal timing would risk
  them drifting out of sync — a subtle bug this design avoids by construction.

## Implementation (as built)

The changes that shipped (all backward-compatible):

1. **`rtd/experiments.py`**
   - `Experiment`: added optional `timing` (dict); `flow` may be a scalar *or* a
     spec dict.
   - `simulate()`: computes `t_on/t_off/t_end/t_pulse` from `timing` when given,
     else from `mean_res`; builds the flow profile from the flow spec (default =
     ramp + event dips using those times); flow-interruption `events` are
     `[t_off]` for a step and `[t_pulse]` for a plateau pulse.
   - Helpers `flow_setpoint(flow)` and `build_flow(flow, setpoint, events)`
     → `FlowProfile` (constant / ramp / sawtooth / from_data / ramp_dips).
2. **`rtd/flow.py`** — added `RampWithDips` (ramp to constant + triangular dips,
   `dip_to=0`, `dip_width=6 s`); exported from `rtd`.
3. **`experiments.yaml`** — `V1`, `V3`, `V5` carry the paper's `10 / 255 / 300 s`
   `timing`; `V5` also sets `t_pulse: 130`. Other experiments stay on auto.
   **[as built]** no global `time_mode`.
4. **`tests/test_rtd.py`** — `test_protocol_timing_override` (window pinned by
   `t_end`, plateau spans the protocol) and `test_event_driven_flow_dip`
   (constant on the plateau, dips to ~0 at `t_off`).

### Paper protocol times

| exp | t_on | t_off | t_end | notes |
|-----|------|-------|-------|-------|
| V1, V3 | 10 | 255 | 300 | bypass step — **set** |
| V5 | 10 | 255 | 300 | + pulse at 130 s — **set** |
| V2-*, V4-*, V6-*, V7, V8 | — | — | — | left on **auto** (residence-time windows already scale with flow; add a `timing` block to pin exact protocol times) |

The filter experiments at 0.35/1 mL/min physically take much longer than
bypass; `auto` already captures that scaling, so exact protocol times are only
needed for a pixel-for-pixel match.

## Backward compatibility & trade-offs

- Everything is **optional**. With no `timing` and a scalar `flow`, behaviour is
  the current compact auto view — nothing breaks.
- Keep **`auto` as the global default**: the residence-time window is genuinely
  better for *seeing* the RTD compactly. Treat protocol-matching as opt-in
  (per experiment or `time_mode: protocol`).
- The continuous-saw-tooth default is **dropped** in favour of ramp + event
  dips; a true continuous saw-tooth remains available via
  `flow: {type: sawtooth, ...}` for any run that needs it.
- Highest fidelity for a specific run is always `flow: {type: from_data}` plus a
  matching `timing` block (or driving the whole thing from the measured CSV).

## Related items

- Ties into backlog #12 (injection points) and the `FromData` work already used
  in `compare_data.py`.
- V8's true twin-filter continuous mode is still a separate feature (see
  `docs/MULTICOMPONENT.md`); protocol timing here does not model the valve
  switching, only the single-cycle approximation's window.
