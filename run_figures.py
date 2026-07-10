#!/usr/bin/env python3
"""
Reproduce the model-predicted RTD curves of Chen et al. (2024), Figures 3 & 4.

Because the raw experimental tracer time series are not available, this script
performs *forward simulations* using the calibrated model parameters reported
in the paper:

    l = 3,  eta = 0.13,  alpha = 1.14,  dcmax = 2.17e-7

and reproduces the shapes of the model curves (solid lines in the paper's
figures) for the calibration pulse experiments (Figure 3, C-series) and the
stepwise validation experiments (Figure 4, V-series).

Outputs:  figure3_calibration.png, figure4_validation.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rtd import build_train, run_train
from rtd.injection import pulse_inlet, step_inlet

HERE = os.path.dirname(os.path.abspath(__file__))


def _train_holdup_uL(seq):
    total = 0.0
    for u in seq:
        if hasattr(u, "volume_uL"):
            total += u.volume_uL
        else:  # Filter
            from rtd.equipment import FILTERS
            f = FILTERS[u.surface_cm2]
            total += f["V_I"] + f["V_wall"] + f["V_O"]
    return total


def simulate_pulse(connection, flow, surface=None, loop_uL=260.0,
                   c_tracer=0.5, t_start=5.0, n_time=1200, t_span_factor=5.0):
    seq, names, uv_i, cond_i = build_train(connection, surface_cm2=surface)
    holdup = _train_holdup_uL(seq)
    Vdot = flow * 1000.0 / 60.0
    mean_res = holdup / Vdot
    pulse_w = loop_uL / Vdot
    t_end = t_start + pulse_w + t_span_factor * mean_res
    t = np.linspace(0, t_end, n_time)
    c_in = pulse_inlet(t, loop_uL, flow, c_tracer, t_start=t_start)
    signals, _ = run_train(seq, t, c_in, flow, read_indices=[uv_i, cond_i])
    return t, signals[uv_i], signals[cond_i]


def simulate_step(connection, flow, surface=None, c_tracer=0.05,
                  n_time=1400, t_span_factor=4.0):
    seq, names, uv_i, cond_i = build_train(connection, surface_cm2=surface)
    holdup = _train_holdup_uL(seq)
    Vdot = flow * 1000.0 / 60.0
    mean_res = holdup / Vdot
    t_on = 0.5 * mean_res
    t_off = t_on + t_span_factor * mean_res      # plateau then wash-out
    t_end = t_off + t_span_factor * mean_res
    t = np.linspace(0, t_end, n_time)
    c_in = step_inlet(t, c_tracer, t_on=t_on, t_off=t_off)
    signals, _ = run_train(seq, t, c_in, flow, read_indices=[uv_i, cond_i])
    return t, signals[uv_i], signals[cond_i]


# --------------------------------------------------------------------------
# Figure 3 -- pulse-injection calibration experiments (260 uL, 0.5 M NaNO3).
# --------------------------------------------------------------------------
FIG3 = [
    ("C1  (bypass, 1 mL/min)",      dict(connection="bypass",    flow=1.0)),
    ("C2-1 (connector, 0.35)",      dict(connection="connector", flow=0.35)),
    ("C2-2 (connector, 1)",         dict(connection="connector", flow=1.0)),
    ("C2-3 (connector, 10)",        dict(connection="connector", flow=10.0)),
    ("C3-1 (3 cm2 filter, 0.35)",   dict(connection="filter", flow=0.35, surface=3)),
    ("C3-2 (10 cm2 filter, 1)",     dict(connection="filter", flow=1.0,  surface=10)),
    ("C3-3 (100 cm2 filter, 10)",   dict(connection="filter", flow=10.0, surface=100)),
]

# --------------------------------------------------------------------------
# Figure 4 -- stepwise validation experiments.
# --------------------------------------------------------------------------
FIG4 = [
    ("V1  (bypass, 0.05 M, 10)",    dict(connection="bypass", flow=10.0, c_tracer=0.05)),
    ("V3  (bypass, 0.1 M, 10)",     dict(connection="bypass", flow=10.0, c_tracer=0.1)),
    ("V2-1 (3 cm2, 0.05 M, 0.35)",  dict(connection="filter", flow=0.35, surface=3,   c_tracer=0.05)),
    ("V2-2 (10 cm2, 0.05 M, 1)",    dict(connection="filter", flow=1.0,  surface=10,  c_tracer=0.05)),
    ("V2-3 (100 cm2, 0.05 M, 10)",  dict(connection="filter", flow=10.0, surface=100, c_tracer=0.05)),
    ("V4-1 (3 cm2, 0.1 M, 0.35)",   dict(connection="filter", flow=0.35, surface=3,   c_tracer=0.1)),
    ("V4-2 (10 cm2, 0.1 M, 1)",     dict(connection="filter", flow=1.0,  surface=10,  c_tracer=0.1)),
    ("V4-3 (100 cm2, 0.1 M, 10)",   dict(connection="filter", flow=10.0, surface=100, c_tracer=0.1)),
]


def make_figure3():
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    axes = axes.ravel()
    for ax, (title, kw) in zip(axes, FIG3):
        t, uv, cond = simulate_pulse(**kw)
        ax.plot(t, uv, color="#1f6fb2", lw=2, label="UV (V_I..)")
        ax2 = ax.twinx()
        ax2.plot(t, cond, color="#2ca02c", lw=1.5, ls="--", label="Cond.")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time / s", fontsize=8)
        ax.set_ylabel("norm. tracer", fontsize=8, color="#1f6fb2")
        ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    for ax in axes[len(FIG3):]:
        ax.axis("off")
    fig.suptitle("Figure 3 (reproduced): pulse-injection calibration curves\n"
                 "forward simulation with l=3, eta=0.13, alpha=1.14, dcmax=2.17e-7",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, "figure3_calibration.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


def make_figure4():
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    axes = axes.ravel()
    for ax, (title, kw) in zip(axes, FIG4):
        t, uv, cond = simulate_step(**kw)
        ax.plot(t, uv, color="#1f6fb2", lw=2)
        ax2 = ax.twinx()
        ax2.plot(t, cond, color="#2ca02c", lw=1.5, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Time / s", fontsize=8)
        ax.set_ylabel("norm. tracer", fontsize=8, color="#1f6fb2")
        ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    for ax in axes[len(FIG4):]:
        ax.axis("off")
    fig.suptitle("Figure 4 (reproduced): stepwise validation curves\n"
                 "start-up plateau + wash-out, forward simulation",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, "figure4_validation.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    make_figure3()
    make_figure4()
