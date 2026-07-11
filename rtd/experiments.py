"""
Experiment definitions, YAML loader and simulator (improvement #14).

An ``Experiment`` is one panel of the paper's Figure 3 (pulse, "C" series) or
Figure 4 (stepwise, "V" series).  Experiments are configured in a YAML file
(``experiments.yaml``) so new ones can be added without touching code; the CLI
(`rtd_cli.py`) discovers them automatically.

``simulate(exp)`` runs the model for one experiment and returns the time trace
plus the two detector signals (UV in mAU, conductivity in mS/cm) and the flow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml

from .equipment import build_train, FILTERS
from .flow import DelayedStep, Sawtooth, as_flow_fn
from .injection import pulse_inlet, step_inlet
from .detectors import uv_from_species, cond_from_species
from .simulate import run_train

# --------------------------------------------------------------------------
# Flow-pattern helper (the paper's two flow behaviours; see run_figures notes)
# --------------------------------------------------------------------------
PUMP_LAG_S = 6.0            # pump reaches set-point with ~6 s delay (Fig 3a,b)
HIGH_FLOW_ML_MIN = 10.0     # >= this shows the saw-tooth (Fig 3g, 4a-c,f,i,l)
SAWTOOTH_PERIOD_S = 15.0    # illustrative
COND_BASELINE = 0.0         # conductivity buffer baseline added in the figures


def experiment_flow(setpoint):
    """Flow profile the paper describes for a given set-point (mL/min)."""
    if setpoint >= HIGH_FLOW_ML_MIN:
        return Sawtooth(v_base=0.5 * setpoint, v_peak=setpoint,
                        period=SAWTOOTH_PERIOD_S, t_start=0.0)
    return DelayedStep(setpoint, lag=PUMP_LAG_S, t_start=0.0)


# --------------------------------------------------------------------------
# Experiment model
# --------------------------------------------------------------------------
@dataclass
class Species:
    """
    One chemical species in an experiment (improvement #2).

    Its inlet is the sum of up to three components (any may be zero):
      * ``baseline`` -- concentration present *outside* the step "on" window
                        (e.g. an equilibration buffer that is displaced);
      * ``step``     -- concentration during the step "on" window;
      * ``pulse``    -- a bolus of this concentration injected during steady state.
    ``name`` must match a key in ``detectors.SPECIES_UV`` / ``SPECIES_COND``.
    """
    name: str
    baseline: float = 0.0
    step: float = 0.0
    pulse: float = 0.0


@dataclass
class Experiment:
    name: str
    kind: str = "pulse"             # "pulse" | "step" (single-species shorthand)
    connection: str = "bypass"      # "bypass" | "connector" | "filter"
    flow: float = 1.0               # set-point mL/min
    figure: Optional[int] = None    # 3, 4, or None
    surface: Optional[int] = None   # filter cm^2 (3/10/100) or None
    c_tracer: float = 0.5           # mol/L (single-species shorthand)
    loop_uL: float = 260.0          # sample-loop volume (pulse)
    inject_at: Optional[str] = None # unit label; default by kind
    description: str = ""
    xmax: Optional[float] = None    # explicit x-limit (s)
    species: Optional[list] = None  # multi-species; None -> single NaNO3 tracer
    background: Optional[dict] = None  # equilibration buffer added to single-tracer runs

    @property
    def title(self) -> str:
        return f"{self.name}  ({self.description})" if self.description else self.name

    def species_list(self):
        """Resolve to a list[Species].

        If a `species:` list is declared it is used as-is.  Otherwise a single
        NaNO3 species is synthesised from ``kind`` / ``c_tracer``, and -- if a
        ``background`` (equilibration buffer) is configured -- that buffer is
        prepended.  The buffer is present at baseline and, for a step, is
        displaced by the NaNO3 (so conductivity dips: the paper's "U" shape)."""
        if self.species:
            return [s if isinstance(s, Species) else Species(**s)
                    for s in self.species]
        out = []
        if self.background:
            out.append(self.background if isinstance(self.background, Species)
                       else Species(**self.background))
        nano3 = (Species("NaNO3", pulse=self.c_tracer) if self.kind == "pulse"
                 else Species("NaNO3", step=self.c_tracer))
        out.append(nano3)
        return out

    @property
    def has_step(self) -> bool:
        # only a real STEP component defines a step window (a baseline buffer
        # that is present throughout does not).
        return any(s.step for s in self.species_list())

    @property
    def injection_node(self) -> str:
        if self.inject_at:
            return self.inject_at
        # loop pulse traverses the sample loop; sample-pump step enters after it
        return "5" if self.has_step else "Loop"


def _train_holdup_uL(seq):
    total = 0.0
    for u in seq:
        if hasattr(u, "volume_uL"):
            total += u.volume_uL
        else:                                   # Filter
            f = FILTERS[u.surface_cm2]
            total += f["V_I"] + f["V_wall"] + f["V_O"]
    return total


def _rep_flow(flow):
    """Representative (set-point) flow mL/min, probed past any start-up ramp."""
    fn = as_flow_fn(flow)
    v = np.atleast_1d(fn(np.linspace(0.0, 600.0, 600)))
    pos = v[v > 1e-9]
    return float(pos.max()) if pos.size else float(np.max(v))


def _species_inlet(t, sp: Species, has_step, t_on, t_off, t_pulse,
                   loop_uL, flow_profile):
    """Build one species' inlet trace = baseline + step + pulse components."""
    c = np.zeros_like(t)
    if has_step:
        on = (t >= t_on) & (t < t_off)
        if sp.step:
            c[on] += sp.step
        if sp.baseline:                       # present when NOT in the on-window
            c[~on] += sp.baseline
    elif sp.baseline:                         # no step window -> baseline everywhere
        c += sp.baseline
    if sp.pulse:
        c = c + pulse_inlet(t, loop_uL, flow_profile, sp.pulse, t_start=t_pulse)
    return c


def simulate(exp: Experiment, n_time: int = 1400,
             pulse_span: float = 2.0, step_span: float = 3.0):
    """
    Run the model for one experiment (single- or multi-species, improvement #2).

    Each species is propagated independently through the same train
    (superposition — valid because the units are linear and the default filter
    uses a constant through-flow fraction), then the two detector signals are
    formed as per-species weighted sums (Beer's / Kohlrausch's law).

    Returns a dict of equal-length arrays:
        t          time (s)
        uv_mAU     UV 280 signal (mAU)
        cond_mScm  conductivity signal (mS/cm)
        flow_mLmin flow rate driving the model
        conc_uv    total tracer concentration at the UV monitor (mol/L)
        conc_cond  total tracer concentration at the conductivity monitor (mol/L)
        species    {name: concentration-at-UV-monitor} per species (mol/L)
    plus scalar ``xmax``.
    """
    species = exp.species_list()
    has_step = exp.has_step
    flow_profile = experiment_flow(exp.flow)
    seq, names, uv_i, cond_i = build_train(
        exp.connection, surface_cm2=exp.surface, inject_at=exp.injection_node)

    holdup = _train_holdup_uL(seq)
    Vdot = _rep_flow(flow_profile) * 1000.0 / 60.0        # uL/s
    mean_res = holdup / Vdot
    pulse_w = exp.loop_uL / Vdot

    if has_step:
        t_on = 0.5 * mean_res
        t_off = t_on + step_span * mean_res
        t_end = t_off + step_span * mean_res
        t_pulse = t_on + 0.5 * (t_off - t_on)             # pulse fires mid-plateau
    else:
        t_on = t_off = None
        t_pulse = 0.0
        t_end = pulse_w + pulse_span * mean_res
    t = np.linspace(0.0, t_end, n_time)

    conc_uv_by, conc_cond_by = {}, {}

    # --- pass 1: species with a step and/or pulse component (need a solve) ---
    pure_step = None                                      # (step_conc, cu, cc)
    for sp in species:
        if not (sp.step or sp.pulse):
            continue
        c_in = _species_inlet(t, sp, has_step, t_on, t_off, t_pulse,
                              exp.loop_uL, flow_profile)
        signals, _ = run_train(seq, t, c_in, flow_profile,
                               read_indices=[uv_i, cond_i], c0=sp.baseline)
        cu, cc = signals[uv_i], signals[cond_i]
        conc_uv_by[sp.name] = conc_uv_by.get(sp.name, 0.0) + cu
        conc_cond_by[sp.name] = conc_cond_by.get(sp.name, 0.0) + cc
        if sp.step and not sp.pulse and not sp.baseline:
            pure_step = (sp.step, cu, cc)

    # --- pass 2: baseline-only species (equilibration buffer) ---------------
    # No extra ODE solve needed in the common cases:
    #   * displaced by a single pure step  -> analytic complement (linearity),
    #   * present throughout (pulse runs)   -> constant baseline.
    for sp in species:
        if sp.step or sp.pulse or not sp.baseline:
            continue
        if has_step and pure_step is not None:
            step_c, cu0, cc0 = pure_step
            cu = sp.baseline * (1.0 - cu0 / step_c)       # buffer = complement
            cc = sp.baseline * (1.0 - cc0 / step_c)
        elif not has_step:
            cu = np.full_like(t, sp.baseline)             # constant buffer
            cc = np.full_like(t, sp.baseline)
        else:                                             # general fallback
            c_in = _species_inlet(t, sp, has_step, t_on, t_off, t_pulse,
                                  exp.loop_uL, flow_profile)
            signals, _ = run_train(seq, t, c_in, flow_profile,
                                   read_indices=[uv_i, cond_i], c0=sp.baseline)
            cu, cc = signals[uv_i], signals[cond_i]
        conc_uv_by[sp.name] = conc_uv_by.get(sp.name, 0.0) + cu
        conc_cond_by[sp.name] = conc_cond_by.get(sp.name, 0.0) + cc

    flow_trace = np.atleast_1d(as_flow_fn(flow_profile)(t)) * np.ones_like(t)
    total_uv = sum(conc_uv_by.values())
    total_cond = sum(conc_cond_by.values())
    return dict(
        t=t,
        uv_mAU=uv_from_species(conc_uv_by),
        cond_mScm=cond_from_species(conc_cond_by, baseline=COND_BASELINE),
        flow_mLmin=flow_trace,
        conc_uv=total_uv,
        conc_cond=total_cond,
        species=conc_uv_by,
        xmax=exp.xmax,
    )


# --------------------------------------------------------------------------
# YAML config
# --------------------------------------------------------------------------
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments.yaml")

_ALLOWED = {"name", "kind", "connection", "flow", "figure", "surface",
            "c_tracer", "loop_uL", "inject_at", "description", "xmax",
            "species", "background"}


def load_config(path: Optional[str] = None):
    """
    Load ``experiments.yaml``.  Returns (experiments, defaults) where
    experiments is a list[Experiment] and defaults is a dict.
    """
    path = path or DEFAULT_CONFIG
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    defaults = cfg.get("defaults", {}) or {}
    background = defaults.get("background")               # equilibration buffer
    experiments = []
    for i, raw in enumerate(cfg.get("experiments", []) or []):
        unknown = set(raw) - _ALLOWED
        if unknown:
            raise ValueError(f"experiment #{i} ({raw.get('name','?')}) has "
                             f"unknown field(s): {sorted(unknown)}")
        exp = Experiment(**raw)
        # apply the default background buffer unless the experiment overrode it
        # or defined its own species list
        if exp.background is None:
            exp.background = background
        experiments.append(exp)
    return experiments, defaults


def find_experiment(experiments, name):
    """Case-insensitive lookup by name."""
    for e in experiments:
        if e.name.lower() == name.lower():
            return e
    raise KeyError(f"experiment {name!r} not found "
                   f"(available: {[e.name for e in experiments]})")
