# `experiments.yaml` schema reference

`experiments.yaml` is the single source of truth for what the model runs. The
CLI (`rtd_cli.py`) and every entry point read it via
`rtd.experiments.load_config()`, which returns `(list[Experiment], defaults)`.
Add or change an experiment by editing this file — no code changes needed.

The file has two top-level keys:

```yaml
defaults:            # global settings + the default background buffer
  ...
experiments:         # a list of experiment blocks
  - name: C1
    ...
```

Unknown fields are rejected: `load_config` raises `ValueError` if an experiment
block contains a key outside the allowed set, so a typo fails loudly rather than
being silently ignored.

---

## `defaults`

Applied to every experiment / figure unless overridden by a CLI flag or a
per-experiment field. All keys are optional.

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `style` | `paper` \| `overlay` | `paper` | Plot layout. `paper` stacks UV / flow / conductivity in separate panels (as in the paper); `overlay` draws them on shared axes. |
| `focus` | bool | `false` | Tight x-axis auto-crop around the signal. Overridden per-run by `--focus` / `--no-focus`. |
| `dpi` | int | `130` | Resolution of the multi-panel **figure** grids (`figure` command). |
| `single_dpi` | int | `300` | Resolution of **per-experiment** high-res plots (`plot` command). |
| `single_size` | `[w, h]` inches | `[9.0, 6.0]` | Figure size for per-experiment plots. |
| `n_time` | int | `1400` | Number of time samples per simulation (grid density in time). |
| `background` | species map \| `null` | — | Equilibration buffer added to **single-tracer** experiments (see below). |

### `defaults.background` — the equilibration buffer

In the real experiments the system is pre-equilibrated with a conductive but
UV-transparent buffer (ÄKTA pump A); NaNO₃ is then stepped/pulsed in against it.
Setting a default background reproduces this without per-experiment boilerplate:

```yaml
defaults:
  background:
    name: buffer
    baseline: 1.0
```

It is prepended to the species list of every experiment that uses the
single-tracer shorthand (`kind` / `c_tracer`). Effect: conductivity of a step
dips into the paper's **"U"** shape (buffer displaced by less-conductive NaNO₃)
while UV rises; for a pulse it is a conductivity pedestal. Set to `null` to model
tracer-on-water. Experiments that declare their own `species:` list ignore it.

---

## An experiment block

Each item under `experiments:` maps to a `rtd.experiments.Experiment`. Only
`name` is strictly required; everything else has a default.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `name` | str | — (**required**) | Short id, e.g. `C1`, `V2-3`. Used for CLI lookup (case-insensitive) and plot titles. |
| `figure` | `3` \| `4` \| `null` | `null` | Which grid the panel belongs to (`figure --which 3\|4\|both`). `null` = demo/standalone, not in any grid. |
| `connection` | `bypass` \| `connector` \| `filter` | `bypass` | Equipment train. `bypass` = ÄKTA plumbing only; `connector` = a connector in the column position; `filter` = a virus filter (needs `surface`). |
| `surface` | `3` \| `10` \| `100` \| `null` | `null` | Filter surface area (cm²). Required when `connection: filter`; `null` otherwise. |
| `flow` | number \| flow-spec | `1.0` | Set-point flow rate (mL/min), or a variable-flow spec (see **Flow** below). |
| `kind` | `pulse` \| `step` | `pulse` | Single-tracer shorthand: `pulse` injects a bolus via the sample loop; `step` switches concentration on then off. Ignored if `species:` is given. |
| `c_tracer` | number | `0.5` | NaNO₃ concentration (mol/L) for the single-tracer shorthand. |
| `loop_uL` | number | `260` | Sample-loop volume (µL); sets the injected pulse width. |
| `inject_at` | str \| `null` | `null` | Injection unit label. `null` → `"Loop"` for a pulse, `"5"` (sample pump) for a step. |
| `species` | list | `null` | Explicit multi-species list; overrides `kind` / `c_tracer` (see **Species**). |
| `background` | species map \| `null` | inherits `defaults.background` | Per-experiment override of the equilibration buffer. |
| `timing` | map | `null` | Explicit protocol times; `null` → auto-size from residence time (see **Timing**). |
| `flow_dips` | bool | `true` | Add flow-interruption dips at transitions (only applied at high flow, ≥10 mL/min). `false` = flat plateau. |
| `step_flow_dip` | number | `0.0` | Flow (mL/min) at the bottom of the step-off / valve-switch dip. |
| `pulse_flow_dip` | number | `0.0` | Flow (mL/min) at the bottom of the pulse dip. |
| `description` | str | `""` | Free text shown in plot titles and `list`. |
| `xmax` | number \| `null` | `null` | Explicit x-axis limit (s). `null` = auto from the simulated window. |

---

## Flow

`flow` is either a **scalar** (constant set-point, mL/min) or a **spec map** with
a `type`. A scalar uses the default profile: a ramp to the set-point over the
~6 s pump lag, held constant, with interruption dips at transition events (dips
apply only at high flow — see `flow_dips`).

| `type` | Extra keys (defaults) | Profile |
|--------|-----------------------|---------|
| `constant` | `setpoint` | Flat flow at `setpoint`. |
| `ramp` | `setpoint`, `lag` (6) | Linear ramp to `setpoint` over `lag` s, then hold. |
| `sawtooth` | `base` (0), `peak` (setpoint), `period` (15) | Repeating linear ramp `base → peak` every `period` s, with a sharp reset — the measured "saw-tooth at high flow". |
| `from_data` | `csv` | Replay a measured flow trace from a Unicorn CSV (highest fidelity). |
| `ramp_dips` | `setpoint`, `lag` (6), `dip_to` (0), `dip_width` (6) | Explicit form of the default: ramp + hold + triangular dips toward `dip_to`. |

Example (C2-3, matching the paper's saw-tooth in Fig. 3d):

```yaml
flow:
  type: sawtooth
  base: 0.0
  peak: 15.0
  period: 26.0
```

### Flow dips

For a stepwise run at high flow the instrument briefly interrupts flow at each
transition (valve switch / shut-down / pulse introduction). The model adds a
triangular dip at those event times, tied to the protocol so flow and signal
stay in sync. Control the depth per event:

```yaml
flow_dips: true        # master toggle (default); false = no dips, flat plateau
step_flow_dip: 2.5     # step-off / valve-switch dip bottoms at 2.5 mL/min (not 0)
pulse_flow_dip: 2.5    # pulse dip bottoms at 2.5 mL/min (not 0)
```

The dip times come from `timing` (`t_off` for the step, `t_pulse` for the
pulse), so they always land on the transitions. See `docs/TIMING_AND_FLOW.md`.

---

## Species (multi-component runs)

Instead of the `kind` / `c_tracer` shorthand, list explicit species. Each has an
inlet built from up to three components (any may be zero):

| Field | Meaning |
|-------|---------|
| `name` | Must match a registered species in `detectors.SPECIES_UV` / `SPECIES_COND_FN`: currently `NaNO3`, `buffer`, `antibody`. |
| `baseline` | Concentration present *outside* the step "on" window (e.g. a buffer that is later displaced). The train starts pre-equilibrated at this level. |
| `step` | Concentration during the step "on" window (`t_on` → `t_off`). |
| `pulse` | Concentration of a bolus injected during steady state (via the sample loop, at `t_pulse`). |

Each species is propagated independently through the same train and the detector
signals are per-species weighted sums (Beer's / Kohlrausch's law). This is exact
because the units are linear (filter `film_resistance = False`, the default).
See `docs/MULTICOMPONENT.md` for the physics and the detector coefficients.

Example — a true transition (buffer displaced by NaNO₃, opposite-sign detectors):

```yaml
- name: TR1
  connection: connector
  flow: 1.0
  species:
    - name: buffer
      baseline: 1.0        # conductive, UV-transparent
    - name: NaNO3
      step: 0.05           # UV-absorbing, less conductive than the buffer
      pulse: 0.5
```

`has_step` (and therefore whether a step window exists) is derived from whether
any species has a non-zero `step` — a baseline-only buffer does **not** create a
step window.

---

## Timing

`timing` pins the protocol times explicitly; omit it to auto-size from the
residence time. All values are seconds.

| Key | Applies to | Auto default |
|-----|-----------|--------------|
| `t_on` | step | `0.5 × mean_res` |
| `t_off` | step | `t_on + 3 × mean_res` |
| `t_pulse` | pulse / combo | pulse-only: `0`; step+pulse: mid-plateau |
| `t_end` | all | step: `t_off + 3 × mean_res`; pulse: `t_pulse + pulse_width + 2 × mean_res` |

`mean_res` = train hold-up ÷ volumetric flow, so auto-sizing scales with the
experiment. Provide a `timing` block when you need the window to match a specific
paper panel (e.g. V1's 0–300 s, or C2-3's 0–40 s). You can set just the keys you
care about; the rest fall back to the auto values.

```yaml
timing:
  t_on: 10
  t_off: 255
  t_end: 300
```

---

## Worked examples

Single-tracer pulse (uses `defaults.background`):

```yaml
- name: C1
  figure: 3
  kind: pulse
  connection: bypass
  flow: 1.0
  surface: null
  c_tracer: 0.5
  description: "bypass, 1 mL/min"
```

Stepwise run with an explicit window and a partial pulse dip:

```yaml
- name: V5
  figure: 4
  connection: bypass
  flow: 10.0
  surface: null
  description: "0.05 M step + 0.5 M pulse, bypass, 10 mL/min"
  pulse_flow_dip: 2.5
  step_flow_dip: 2.5
  timing:
    t_on: 10
    t_off: 255
    t_end: 480
    t_pulse: 257
  species:
    - name: buffer
      baseline: 1.0
    - name: NaNO3
      step: 0.05
      pulse: 0.5
```

---

## Style notes

* Use **block style** (indented maps), not inline `{}` — it is easier to read and
  diff. Put a `#` comment before each experiment describing it.
* `null` is the explicit "none" value (e.g. `surface: null`, `figure: null`).
* Numbers may be written plainly (`10`, `0.05`); units are per the tables above.

## Related docs

* `docs/TIMING_AND_FLOW.md` — protocol timing and event-driven flow in depth.
* `docs/MULTICOMPONENT.md` — multi-species physics, detectors, the buffer "U".
* `docs/PARAMETERS.md` — every model constant, its units, and provenance.
