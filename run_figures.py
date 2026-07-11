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

from rtd import build_train, run_train, DelayedStep, Sawtooth
from rtd.injection import pulse_inlet, step_inlet
from rtd.detectors import beer_uv, kohlrausch_cond
from rtd.flow import as_flow_fn

HERE = os.path.dirname(os.path.abspath(__file__))

# Conductivity buffer baseline (mS/cm) so the tracer signal sits on a
# background, as in the instrument.  Illustrative; see rtd/detectors.py.
COND_BASELINE = 0.0

# Pump start-up delay (s) — the paper reports the pump reaches its preset with a
# ~6 s delay ("gradient pattern", pronounced at low flow; Fig 3a,b).
PUMP_LAG_S = 6.0
# Flow at/above this (mL/min) shows the sawtooth flow-interruption pattern the
# paper reports at high flow (Fig 3g, 4a-c,f,i,l).
HIGH_FLOW_ML_MIN = 10.0
# Sawtooth period (s) for the high-flow interruption pattern. Illustrative;
# a longer period keeps the model tractable (fewer solver restarts) while still
# showing the characteristic teeth.
SAWTOOTH_PERIOD_S = 15.0


def experiment_flow(setpoint):
    """
    Return the flow profile the paper describes for a given set-point.

    * setpoint < 10 mL/min  -> DelayedStep: pump ramps to the preset with a ~6 s
      delay (the "gradient pattern", Fig 3a,b).
    * setpoint >= 10 mL/min -> Sawtooth: at high flow the introduction of a
      tracer / valve switch interrupts the flow, which then recovers, giving a
      sawtooth (Fig 3g, 4a-c,f,i,l).  The first rising tooth also serves as the
      start-up ramp.  Amplitude/period are illustrative (the paper used the
      measured flow trace; exact values are not tabulated).
    """
    if setpoint >= HIGH_FLOW_ML_MIN:
        return Sawtooth(v_base=0.5 * setpoint, v_peak=setpoint,
                        period=SAWTOOTH_PERIOD_S, t_start=0.0)
    return DelayedStep(setpoint, lag=PUMP_LAG_S, t_start=0.0)


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


def _rep_flow(flow, t):
    """Representative (set-point) flow in mL/min for sizing time windows."""
    fn = as_flow_fn(flow)
    v = np.atleast_1d(fn(t))
    pos = v[v > 1e-9]
    return float(pos.max()) if pos.size else float(np.max(v))


def simulate_pulse(connection, flow, surface=None, loop_uL=260.0,
                   c_tracer=0.5, t_start=0.0, n_time=1200, t_span_factor=2.0):
    """
    Returns t, UV(mAU), Cond(mS/cm), flow(mL/min).  ``flow`` may be a scalar or
    a FlowProfile.
    """
    seq, names, uv_i, cond_i = build_train(connection, surface_cm2=surface)
    holdup = _train_holdup_uL(seq)
    t_probe = np.linspace(0, 1, 50)             # for representative flow
    Vdot = _rep_flow(flow, t_probe) * 1000.0 / 60.0
    mean_res = holdup / Vdot
    pulse_w = loop_uL / Vdot
    t_end = t_start + pulse_w + t_span_factor * mean_res
    t = np.linspace(0, t_end, n_time)
    c_in = pulse_inlet(t, loop_uL, flow, c_tracer, t_start=t_start)
    signals, _ = run_train(seq, t, c_in, flow, read_indices=[uv_i, cond_i])
    uv = beer_uv(signals[uv_i])
    cond = kohlrausch_cond(signals[cond_i], baseline=COND_BASELINE)
    flow_trace = np.atleast_1d(as_flow_fn(flow)(t)) * np.ones_like(t)
    return t, uv, cond, flow_trace


def simulate_step(connection, flow, surface=None, c_tracer=0.05,
                  n_time=1400, t_span_factor=3.0):
    """Returns t, UV(mAU), Cond(mS/cm), flow(mL/min)."""
    seq, names, uv_i, cond_i = build_train(connection, surface_cm2=surface)
    holdup = _train_holdup_uL(seq)
    t_probe = np.linspace(0, 1, 50)
    Vdot = _rep_flow(flow, t_probe) * 1000.0 / 60.0
    mean_res = holdup / Vdot
    t_on = 0.5 * mean_res
    t_off = t_on + t_span_factor * mean_res      # plateau then wash-out
    t_end = t_off + t_span_factor * mean_res
    t = np.linspace(0, t_end, n_time)
    c_in = step_inlet(t, c_tracer, t_on=t_on, t_off=t_off)
    signals, _ = run_train(seq, t, c_in, flow, read_indices=[uv_i, cond_i])
    uv = beer_uv(signals[uv_i])
    cond = kohlrausch_cond(signals[cond_i], baseline=COND_BASELINE)
    flow_trace = np.atleast_1d(as_flow_fn(flow)(t)) * np.ones_like(t)
    return t, uv, cond, flow_trace


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


# --------------------------------------------------------------------------
# Plot style
# --------------------------------------------------------------------------
# Two panel layouts are available; switch with this flag (or the command-line
# argument, e.g.  `python3 run_figures.py overlay`).
#
#   "overlay" : UV, conductivity and flow share ONE set of axes (three y-axes
#               overlaid), one panel per experiment.  Compact.
#   "paper"   : the layout used in Chen et al. (2024) -- three stacked sub-panels
#               per experiment sharing the time axis: UV (mAU, blue) on top,
#               flow rate (mL/min, orange) in the middle, conductivity (mS/cm,
#               green) at the bottom.  Matches the paper's Figure 3/4 style.
#
# Default: "paper" (the paper-faithful layout).
PLOT_STYLE = "paper"

# --------------------------------------------------------------------------
# X-axis auto-focus
# --------------------------------------------------------------------------
# The full window is always simulated (so tails and mass balance are intact),
# but the plot is cropped to where the signal actually lives -- like the paper's
# tight x-axes -- instead of showing a long flat tail.
#
# The crop is the last time UV or conductivity is still above FOCUS_FRAC of its
# own peak, plus a FOCUS_MARGIN fraction of head-room.  Set FOCUS_ENABLED=False
# to show the whole simulated window, or give an experiment an explicit
# "xmax" in its FIG dict to override the auto-crop for that panel.
FOCUS_ENABLED = True
FOCUS_FRAC = 0.02       # signal considered "back to baseline" below 2 % of peak
FOCUS_MARGIN = 0.15     # add 15 % head-room to the right of the active region


def _focus_xmax(t, *signals, frac=FOCUS_FRAC, margin=FOCUS_MARGIN):
    """
    Upper x-limit that focuses on the active region of the given signals.

    For each signal, find the last time it is still above ``frac`` of its peak
    amplitude (relative to its baseline at t=0); take the latest such time
    across signals and add a ``margin`` of head-room.  Returns t[-1] if nothing
    stands out (e.g. a flat signal).
    """
    t = np.asarray(t, float)
    tmax = 0.0
    for y in signals:
        y = np.asarray(y, float)
        base = y[0]
        amp = np.max(np.abs(y - base))
        if amp <= 0:
            continue
        active = np.flatnonzero(np.abs(y - base) > frac * amp)
        if active.size:
            tmax = max(tmax, t[active[-1]])
    if tmax <= 0:
        return float(t[-1])
    return float(min(t[-1], t[0] + (tmax - t[0]) * (1.0 + margin)))


def _plot_panel_overlay(ax, t, uv, cond, flow, title, xmax=None):
    """Type 1 -- UV (mAU), conductivity (mS/cm) and flow (mL/min) overlaid on
    three separate y-axes of a single panel, with a combined legend."""
    ln1, = ax.plot(t, uv, color="#1f6fb2", lw=2, label="UV (mAU)")
    ax.set_ylabel("UV 280 (mAU)", fontsize=8, color="#1f6fb2")

    ax2 = ax.twinx()
    ln2, = ax2.plot(t, cond, color="#2ca02c", lw=1.5, ls="--", label="Cond (mS/cm)")
    ax2.set_ylabel("Cond (mS/cm)", fontsize=8, color="#2ca02c")

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 34))
    ln3, = ax3.plot(t, flow, color="#e69500", lw=1.2, ls=":", label="Flow (mL/min)")
    ax3.set_ylabel("Flow (mL/min)", fontsize=8, color="#e69500")
    ax3.set_ylim(0, max(1e-6, np.max(flow) * 1.6))

    lns = [ln1, ln2, ln3]
    ax.legend(lns, [l.get_label() for l in lns], loc="upper right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time / s", fontsize=8)
    if xmax is not None:
        ax.set_xlim(t[0], xmax)
    ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7); ax3.tick_params(labelsize=7)


def _plot_panel_paper(fig, subspec, t, uv, cond, flow, title, xmax=None):
    """Type 2 -- paper layout: three stacked sub-panels sharing the x-axis,
    UV (top, blue) / flow (middle, orange) / conductivity (bottom, green)."""
    inner = subspec.subgridspec(3, 1, hspace=0.08, height_ratios=[1.0, 0.55, 1.0])
    ax_uv = fig.add_subplot(inner[0])
    ax_fl = fig.add_subplot(inner[1], sharex=ax_uv)
    ax_cn = fig.add_subplot(inner[2], sharex=ax_uv)

    ax_uv.plot(t, uv, color="#1f6fb2", lw=1.8)
    ax_uv.set_ylabel("UV/mAU", fontsize=8, color="#1f6fb2")
    ax_uv.set_title(title, fontsize=9)

    ax_fl.plot(t, flow, color="#e69500", lw=1.5)
    ax_fl.set_ylabel("Flow/\n(mL/min)", fontsize=7, color="#e69500")
    ax_fl.set_ylim(0, max(1e-6, np.max(flow) * 1.4))
    # Move the ticks and the text label to the right side
    ax_fl.yaxis.tick_right()
    ax_fl.yaxis.set_label_position("right")

    ax_cn.plot(t, cond, color="#2ca02c", lw=1.8)
    ax_cn.set_ylabel("Cond./\n(mS/cm)", fontsize=8, color="#2ca02c")
    ax_cn.set_xlabel("Time / s", fontsize=8)

    if xmax is not None:                      # shared x-axis -> set once
        ax_uv.set_xlim(t[0], xmax)

    for a in (ax_uv, ax_fl):
        plt.setp(a.get_xticklabels(), visible=False)
    for a in (ax_uv, ax_fl, ax_cn):
        a.tick_params(labelsize=6)


def _make_figure(experiments, simulate_fn, suptitle, outfile, style=None):
    """Build a full figure in either layout (see PLOT_STYLE)."""
    style = style or PLOT_STYLE
    n = len(experiments)
    ncols = 3
    nrows = -(-n // ncols)                       # ceil division

    def _prep(kw):
        """Pull an optional per-experiment 'xmax' override out of the kwargs,
        set the paper flow profile, run the simulation, and decide the x-limit."""
        kw = dict(kw)
        xmax = kw.pop("xmax", None)                     # manual override (optional)
        kw["flow"] = experiment_flow(kw["flow"])
        t, uv, cond, flow = simulate_fn(**kw)
        if xmax is None and FOCUS_ENABLED:
            xmax = _focus_xmax(t, uv, cond)             # auto-focus
        return t, uv, cond, flow, xmax

    if style == "overlay":
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (title, kw) in zip(axes, experiments):
            t, uv, cond, flow, xmax = _prep(kw)
            _plot_panel_overlay(ax, t, uv, cond, flow, title, xmax=xmax)
        for ax in axes[n:]:
            ax.axis("off")
    elif style == "paper":
        fig = plt.figure(figsize=(14, 3.6 * nrows))
        outer = fig.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.45)
        for idx, (title, kw) in enumerate(experiments):
            t, uv, cond, flow, xmax = _prep(kw)
            r, c = divmod(idx, ncols)
            _plot_panel_paper(fig, outer[r, c], t, uv, cond, flow, title, xmax=xmax)
    else:
        raise ValueError(f"Unknown PLOT_STYLE {style!r} (use 'overlay' or 'paper')")

    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outfile, dpi=130)
    print("wrote", outfile)


def make_figure3(style=None):
    _make_figure(
        FIG3, simulate_pulse,
        "Figure 3 (reproduced): pulse-injection calibration curves\n"
        "UV via Beer's law, conductivity via Kohlrausch's law; "
        "l=3, eta=0.13, alpha=1.14, dcmax=2.17e-7",
        os.path.join(HERE, "figure3.png"), style=style)


def make_figure4(style=None):
    _make_figure(
        FIG4, simulate_step,
        "Figure 4 (reproduced): stepwise validation curves\n"
        "UV via Beer's law, conductivity via Kohlrausch's law; "
        "start-up plateau + wash-out",
        os.path.join(HERE, "figure4.png"), style=style)


if __name__ == "__main__":
    import sys
    # Optional CLI overrides (any order):
    #   style:  'paper' | 'overlay'
    #   focus:  'focus' | 'nofocus'   (toggles the x-axis auto-focus)
    # e.g.  python3 run_figures.py overlay nofocus
    args = [a.lower() for a in sys.argv[1:]]
    style = next((a for a in args if a in ("paper", "overlay")), PLOT_STYLE)
    if "nofocus" in args:
        FOCUS_ENABLED = False
    elif "focus" in args:
        FOCUS_ENABLED = True
    make_figure3(style=style)
    make_figure4(style=style)
