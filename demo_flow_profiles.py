#!/usr/bin/env python3
"""
Demonstration of variable flow-rate profiles.

Runs the SAME 260 µL pulse through the SAME connector train under four flow
profiles, showing how a time-varying flow reshapes the RTD response.  Outputs
demo_flow_profiles.png.

    python3 demo_flow_profiles.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rtd import (build_train, run_train, beer_uv, kohlrausch_cond,
                 Constant, Ramp, DelayedStep, Sawtooth, Piecewise, as_flow_fn)
from rtd.injection import pulse_inlet

HERE = os.path.dirname(os.path.abspath(__file__))

PROFILES = [
    ("Constant 1 mL/min",            Constant(1.0)),
    ("DelayedStep to 1 (6 s lag)",   DelayedStep(1.0, lag=6.0)),
    ("Ramp 0.2->2 over 40 s",        Ramp(2.0, t_start=0, t_ramp=40, v_start=0.2)),
    ("Sawtooth 0.5<->1.0, period 30 s",  Sawtooth(0.50, 1.0, period=30.0)),
    ("Piecewise (Ramp 0->1 24 s / Constant 10 from 30s / ramp down at 120 to 0",  Piecewise([
        (0.0,   Ramp(0.5, t_ramp=12.0)),   # ramp up
      ##  (12.0,   Ramp(1.0, t_ramp=36.0)),   # ramp up
        (48.0, Constant(1.0)),           # steady
        (90.0, Ramp(0.0, t_start=90.0, t_ramp=24.0, v_start=1.0)),  # ramp down
    ])),
]


def run(profile):
    seq, names, uv_i, cond_i = build_train("connector")
    t = np.linspace(0, 300, 2000)
    c_in = pulse_inlet(t, 260.0, profile, c_tracer=0.5, t_start=5.0)
    signals, _ = run_train(seq, t, c_in, profile, read_indices=[uv_i, cond_i])
    uv = beer_uv(signals[uv_i])
    flow = np.atleast_1d(as_flow_fn(profile)(t)) * np.ones_like(t)
    return t, uv, flow


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (label, prof) in zip(axes.ravel(), PROFILES):
        t, uv, flow = run(prof)
        ln1, = ax.plot(t, uv, color="#1f6fb2", lw=2, label="UV (mAU)")
        ax2 = ax.twinx()
        ln2, = ax2.plot(t, flow, color="#e69500", lw=1.5, ls=":", label="Flow (mL/min)")
        ax2.set_ylim(0, max(flow) * 1.5)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Time / s"); ax.set_ylabel("UV 280 (mAU)", color="#1f6fb2")
        ax2.set_ylabel("Flow (mL/min)", color="#e69500")
        ax.legend([ln1, ln2], [l.get_label() for l in (ln1, ln2)], loc="upper right",
                  fontsize=8)
    fig.suptitle("Variable flow: same 260 µL pulse through the connector train\n"
                 "flow reshapes and re-times the RTD response", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(HERE, "demo_flow_profiles.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
