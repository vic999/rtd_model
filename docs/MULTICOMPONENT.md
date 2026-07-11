# Multi-component tracer support (improvement #2)

Real virus-filtration runs often involve **more than one chemical species** —
for example an equilibration buffer (tris-acetate) that is displaced by NaNO₃.
Because the two species affect UV and conductivity *differently*, the two
detector traces then have **different shapes** (in the uploaded transition data,
conductivity *drops* across the transition while UV *rises*). A single-tracer
model cannot reproduce that; this document describes the multi-species support
that can.

## Model

Each experiment declares one or more **species**. Each species has an inlet made
of up to three components (any may be zero):

| field | meaning |
|-------|---------|
| `baseline` | concentration present *outside* the step window (e.g. a buffer that is later displaced); the train is started pre-equilibrated at this level |
| `step` | concentration during the step "on" window (start-up plateau → wash-out) |
| `pulse` | a bolus of this concentration injected during steady state (volume-based, via the sample loop) |

Because the equipment units are **linear** and the default filter uses a
constant through-flow fraction, the species do not interact, so each species'
inlet is propagated **independently** through the same train and the detector
signals are formed as **per-species weighted sums** (superposition):

```
UV(t)   = Σ_i  ε_i     · c_i(t)          (Beer's law,       Eq. 11)
κ(t)    = Σ_i  Λ_i     · c_i(t)          (Kohlrausch's law, Eq. 12)
```

The per-species coefficients live in `rtd/detectors.py`
(`SPECIES_UV`, `SPECIES_COND`) and are combined by `uv_from_species` /
`cond_from_species`.

### Initial condition

A background species (e.g. the buffer) must already fill the system at *t = 0*,
otherwise the detector would start at zero and only rise as the buffer reaches
it. `run_train(..., c0=baseline)` initialises every unit to that species'
baseline, so a buffer run starts at its true plateau (≈11.9 mS/cm) immediately.

## Configuration

In `experiments.yaml`, give an experiment a `species:` list instead of the
single-tracer `kind`/`c_tracer` shorthand. Example — the transition demo `TR1`:

```yaml
- name: TR1
  connection: connector
  flow: 1.0
  species:
    - {name: buffer, baseline: 1.0}          # tris-acetate: high cond, ~0 UV
    - {name: NaNO3,  step: 0.05, pulse: 0.5} # NaNO3: UV-absorbing, less conductive
```

`name` must match a species registered in `detectors.SPECIES_UV/COND`
(currently `NaNO3`, `buffer`, `antibody`). The single-tracer experiments (C*,
V1–V4) omit `species:` and are unchanged.

### Bundled multi-species experiments

| name | species | shows |
|------|---------|-------|
| `V5`   | NaNO3 (0.05 M step + 0.5 M pulse) | combined step+pulse, *same* species → UV and cond share shape |
| `V6-2` | NaNO3 (step + pulse), 10 cm² filter | same, through a filter |
| `V7`   | antibody (0.5 + 2 g/L)             | antibody detector response |
| `TR1`  | buffer + NaNO3                     | **true multi-component**: UV rises while conductivity drops across the transition, then both spike on the pulse |

Render or export any of them, e.g.:

```bash
python3 rtd_cli.py plot --experiment TR1 V5 --dpi 300
python3 rtd_cli.py csv  --experiment TR1
```

## Result

For `TR1` the model reproduces the opposite-sign behaviour of the real data:

- conductivity starts high (buffer ≈11.9 mS/cm), **drops** to ≈6 as NaNO₃
  displaces the buffer, then **spikes** to ≈54 on the 0.5 M pulse;
- UV starts at ≈0 (buffer is transparent), **rises** to the NaNO₃ plateau, then
  spikes with the pulse.

## Caveats

- **Superposition validity.** Independent per-species propagation is exact only
  while the filter's `film_resistance = False` (constant ε, the default). With
  the concentration-dependent ε (Eq. 8) the species couple through the driving
  force and a single coupled solve would be required.
- **Coefficient provenance.** Only the NaNO₃ conductivity constant is
  physically grounded; the buffer and antibody constants are **illustrative**
  (chosen to reproduce the observed magnitudes/signs). See `docs/PARAMETERS.md`.
  For quantitative work they should be measured or fit.
