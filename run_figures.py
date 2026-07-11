#!/usr/bin/env python3
"""
Reproduce the model-predicted RTD curves of Chen et al. (2024), Figures 3 & 4.

This is now a thin wrapper around the reusable machinery:
  * experiments are configured in ``experiments.yaml`` (see improvement #14);
  * ``rtd.experiments`` simulates them, ``rtd.plots`` draws them.

For a full command-line interface (list experiments, per-experiment
high-resolution plots, CSV export, resolution flags) use ``rtd_cli.py``.

    python3 run_figures.py                 # both figures, default style
    python3 run_figures.py overlay focus   # overlay layout, tight x-axis

Outputs: figure3.png, figure4.png
"""

from __future__ import annotations

import os

from rtd.experiments import Experiment, simulate, load_config
from rtd.plots import plot_grid

HERE = os.path.dirname(os.path.abspath(__file__))

# Default plot style / focus (overridable on the command line).
PLOT_STYLE = "paper"          # "paper" | "overlay"
FOCUS_ENABLED = False         # tight x-axis auto-crop; default full window


def _make_figure(figure_no, suptitle, outfile, style=None, focus=None):
    style = style or PLOT_STYLE
    focus = FOCUS_ENABLED if focus is None else focus
    experiments, _defaults = load_config()
    exps = [e for e in experiments if e.figure == figure_no]
    pairs = [(e, simulate(e)) for e in exps]
    out = os.path.join(HERE, outfile)
    plot_grid(pairs, suptitle, out, style=style, focus=focus)
    print("wrote", out)


def make_figure3(style=None, focus=None):
    _make_figure(
        3,
        "Figure 3 (reproduced): pulse-injection calibration curves\n"
        "UV via Beer's law, conductivity via Kohlrausch's law; "
        "l=3, eta=0.13, alpha=1.14, dcmax=2.17e-7",
        "figure3.png", style=style, focus=focus)


def make_figure4(style=None, focus=None):
    _make_figure(
        4,
        "Figure 4 (reproduced): stepwise validation curves\n"
        "UV via Beer's law, conductivity via Kohlrausch's law; "
        "start-up plateau + wash-out",
        "figure4.png", style=style, focus=focus)


# --------------------------------------------------------------------------
# Backward-compatible simulation shims (used by verify.py and older callers).
# --------------------------------------------------------------------------
def simulate_pulse(connection, flow, surface=None, loop_uL=260.0,
                   c_tracer=0.5, n_time=1200, **_ignored):
    exp = Experiment(name="_", kind="pulse", connection=connection, flow=flow,
                     surface=surface, loop_uL=loop_uL, c_tracer=c_tracer)
    r = simulate(exp, n_time=n_time)
    return r["t"], r["uv_mAU"], r["cond_mScm"], r["flow_mLmin"]


def simulate_step(connection, flow, surface=None, c_tracer=0.05,
                  n_time=1400, **_ignored):
    exp = Experiment(name="_", kind="step", connection=connection, flow=flow,
                     surface=surface, c_tracer=c_tracer)
    r = simulate(exp, n_time=n_time)
    return r["t"], r["uv_mAU"], r["cond_mScm"], r["flow_mLmin"]


if __name__ == "__main__":
    import sys
    # Optional CLI overrides (any order):  style 'paper'|'overlay'
    # and 'focus'|'nofocus' (x-axis auto-focus; default is nofocus).
    args = [a.lower() for a in sys.argv[1:]]
    style = next((a for a in args if a in ("paper", "overlay")), PLOT_STYLE)
    focus = FOCUS_ENABLED
    if "nofocus" in args:
        focus = False
    elif "focus" in args:
        focus = True
    make_figure3(style=style, focus=focus)
    make_figure4(style=style, focus=focus)
