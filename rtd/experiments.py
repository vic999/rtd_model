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
from .detectors import beer_uv, kohlrausch_cond
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
class Experiment:
    name: str
    kind: str                       # "pulse" | "step"
    connection: str                 # "bypass" | "connector" | "filter"
    flow: float                     # set-point mL/min
    figure: Optional[int] = None    # 3, 4, or None
    surface: Optional[int] = None   # filter cm^2 (3/10/100) or None
    c_tracer: float = 0.5           # mol/L
    loop_uL: float = 260.0          # sample-loop volume (pulse)
    inject_at: Optional[str] = None # unit label; default by kind
    description: str = ""
    xmax: Optional[float] = None    # explicit x-limit (s)

    @property
    def title(self) -> str:
        return f"{self.name}  ({self.description})" if self.description else self.name

    @property
    def injection_node(self) -> str:
        if self.inject_at:
            return self.inject_at
        # loop pulse traverses the sample loop; sample-pump step enters after it
        return "Loop" if self.kind == "pulse" else "5"


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


def simulate(exp: Experiment, n_time: int = 1400,
             pulse_span: float = 2.0, step_span: float = 3.0):
    """
    Run the model for one experiment.

    Returns a dict of equal-length arrays:
        t          time (s)
        uv_mAU     UV 280 signal (Beer's law)
        cond_mScm  conductivity signal (Kohlrausch's law)
        flow_mLmin flow rate driving the model
        conc_uv    raw tracer concentration at the UV monitor (mol/L)
        conc_cond  raw tracer concentration at the conductivity monitor (mol/L)
    plus scalar ``xmax`` (explicit x-limit or None).
    """
    flow_profile = experiment_flow(exp.flow)
    seq, names, uv_i, cond_i = build_train(
        exp.connection, surface_cm2=exp.surface, inject_at=exp.injection_node)

    holdup = _train_holdup_uL(seq)
    Vdot = _rep_flow(flow_profile) * 1000.0 / 60.0        # uL/s
    mean_res = holdup / Vdot

    if exp.kind == "pulse":
        pulse_w = exp.loop_uL / Vdot
        t_end = pulse_w + pulse_span * mean_res
        t = np.linspace(0.0, t_end, n_time)
        c_in = pulse_inlet(t, exp.loop_uL, flow_profile, exp.c_tracer, t_start=0.0)
    elif exp.kind == "step":
        t_on = 0.5 * mean_res
        t_off = t_on + step_span * mean_res
        t_end = t_off + step_span * mean_res
        t = np.linspace(0.0, t_end, n_time)
        c_in = step_inlet(t, exp.c_tracer, t_on=t_on, t_off=t_off)
    else:
        raise ValueError(f"Unknown experiment kind {exp.kind!r} (use pulse/step)")

    signals, _ = run_train(seq, t, c_in, flow_profile, read_indices=[uv_i, cond_i])
    flow_trace = np.atleast_1d(as_flow_fn(flow_profile)(t)) * np.ones_like(t)
    return dict(
        t=t,
        uv_mAU=beer_uv(signals[uv_i]),
        cond_mScm=kohlrausch_cond(signals[cond_i], baseline=COND_BASELINE),
        flow_mLmin=flow_trace,
        conc_uv=signals[uv_i],
        conc_cond=signals[cond_i],
        xmax=exp.xmax,
    )


# --------------------------------------------------------------------------
# YAML config
# --------------------------------------------------------------------------
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments.yaml")

_ALLOWED = {"name", "kind", "connection", "flow", "figure", "surface",
            "c_tracer", "loop_uL", "inject_at", "description", "xmax"}


def load_config(path: Optional[str] = None):
    """
    Load ``experiments.yaml``.  Returns (experiments, defaults) where
    experiments is a list[Experiment] and defaults is a dict.
    """
    path = path or DEFAULT_CONFIG
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    defaults = cfg.get("defaults", {}) or {}
    experiments = []
    for i, raw in enumerate(cfg.get("experiments", []) or []):
        unknown = set(raw) - _ALLOWED
        if unknown:
            raise ValueError(f"experiment #{i} ({raw.get('name','?')}) has "
                             f"unknown field(s): {sorted(unknown)}")
        experiments.append(Experiment(**raw))
    return experiments, defaults


def find_experiment(experiments, name):
    """Case-insensitive lookup by name."""
    for e in experiments:
        if e.name.lower() == name.lower():
            return e
    raise KeyError(f"experiment {name!r} not found "
                   f"(available: {[e.name for e in experiments]})")
