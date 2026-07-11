"""
Plotting and CSV export for experiments (improvement #14).

Two panel layouts (``style``):
  * "paper"   -- three stacked sub-panels sharing the time axis (UV / flow /
                 conductivity), the Chen et al. (2024) layout.
  * "overlay" -- UV, conductivity and flow overlaid on three y-axes.

Functions:
  * ``plot_grid``   -- a multi-panel figure (e.g. Figure 3 or 4).
  * ``plot_single`` -- one experiment as a high-resolution standalone figure.
  * ``export_csv``  -- the simulated data for one experiment as CSV.

The ``simulate`` results (dicts) come from ``rtd.experiments``.
"""

from __future__ import annotations

import csv
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# default look
FOCUS_FRAC = 0.02
FOCUS_MARGIN = 0.15
UV_COLOR = "#1f6fb2"
COND_COLOR = "#2ca02c"
FLOW_COLOR = "#e69500"


def focus_xmax(t, *signals, frac=FOCUS_FRAC, margin=FOCUS_MARGIN):
    """Upper x-limit focusing on the active region of the given signals."""
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


def _resolve_xmax(sim, focus):
    """Explicit xmax wins; else auto-focus if requested; else None (full)."""
    if sim.get("xmax") is not None:
        return sim["xmax"]
    if focus:
        return focus_xmax(sim["t"], sim["uv_mAU"], sim["cond_mScm"])
    return None


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------
def _panel_overlay(ax, sim, title, xmax=None, legend=True):
    t = sim["t"]
    ln1, = ax.plot(t, sim["uv_mAU"], color=UV_COLOR, lw=2, label="UV (mAU)")
    ax.set_ylabel("UV 280 (mAU)", fontsize=8, color=UV_COLOR)
    ax2 = ax.twinx()
    ln2, = ax2.plot(t, sim["cond_mScm"], color=COND_COLOR, lw=1.5, ls="--",
                    label="Cond (mS/cm)")
    ax2.set_ylabel("Cond (mS/cm)", fontsize=8, color=COND_COLOR)
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 34))
    ln3, = ax3.plot(t, sim["flow_mLmin"], color=FLOW_COLOR, lw=1.2, ls=":",
                    label="Flow (mL/min)")
    ax3.set_ylabel("Flow (mL/min)", fontsize=8, color=FLOW_COLOR)
    ax3.set_ylim(0, max(1e-6, np.max(sim["flow_mLmin"]) * 1.6))
    if legend:
        lns = [ln1, ln2, ln3]
        ax.legend(lns, [l.get_label() for l in lns], loc="upper right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time / s", fontsize=8)
    if xmax is not None:
        ax.set_xlim(t[0], xmax)
    ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7); ax3.tick_params(labelsize=7)


def _panel_paper(fig, subspec, sim, title, xmax=None):
    t = sim["t"]
    inner = subspec.subgridspec(3, 1, hspace=0.08, height_ratios=[1.0, 0.55, 1.0])
    ax_uv = fig.add_subplot(inner[0])
    ax_fl = fig.add_subplot(inner[1], sharex=ax_uv)
    ax_cn = fig.add_subplot(inner[2], sharex=ax_uv)

    ax_uv.plot(t, sim["uv_mAU"], color=UV_COLOR, lw=1.8)
    ax_uv.set_ylabel("UV/mAU", fontsize=8, color=UV_COLOR)
    ax_uv.set_title(title, fontsize=9)

    ax_fl.plot(t, sim["flow_mLmin"], color=FLOW_COLOR, lw=1.5)
    ax_fl.set_ylabel("Flow/\n(mL/min)", fontsize=7, color=FLOW_COLOR)
    ax_fl.set_ylim(0, max(1e-6, np.max(sim["flow_mLmin"]) * 1.4))
    ax_fl.yaxis.tick_right(); ax_fl.yaxis.set_label_position("right")

    ax_cn.plot(t, sim["cond_mScm"], color=COND_COLOR, lw=1.8)
    ax_cn.set_ylabel("Cond./\n(mS/cm)", fontsize=8, color=COND_COLOR)
    ax_cn.set_xlabel("Time / s", fontsize=8)

    if xmax is not None:
        ax_uv.set_xlim(t[0], xmax)
    for a in (ax_uv, ax_fl):
        plt.setp(a.get_xticklabels(), visible=False)
    for a in (ax_uv, ax_fl, ax_cn):
        a.tick_params(labelsize=6)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_grid(pairs, suptitle, outfile, style="paper", focus=False, dpi=130):
    """
    Multi-panel grid figure.  ``pairs`` is a list of (Experiment, sim-dict).
    """
    n = len(pairs)
    ncols = 3
    nrows = -(-n // ncols)

    if style == "overlay":
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, (exp, sim) in zip(axes, pairs):
            _panel_overlay(ax, sim, exp.title, xmax=_resolve_xmax(sim, focus))
        for ax in axes[n:]:
            ax.axis("off")
    elif style == "paper":
        fig = plt.figure(figsize=(14, 3.6 * nrows))
        outer = fig.add_gridspec(nrows, ncols, hspace=0.5, wspace=0.45)
        for idx, (exp, sim) in enumerate(pairs):
            r, c = divmod(idx, ncols)
            _panel_paper(fig, outer[r, c], sim, exp.title, xmax=_resolve_xmax(sim, focus))
    else:
        raise ValueError(f"Unknown style {style!r} (use 'paper' or 'overlay')")

    fig.suptitle(suptitle, fontsize=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # tight_layout + subgridspec
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    return outfile


def plot_single(exp, sim, outfile, style="paper", focus=False,
                dpi=300, figsize=(9.0, 6.0)):
    """One experiment as a standalone high-resolution figure."""
    xmax = _resolve_xmax(sim, focus)
    if style == "overlay":
        fig, ax = plt.subplots(figsize=figsize)
        _panel_overlay(ax, sim, exp.title, xmax=xmax)
    else:  # paper
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 1)
        _panel_paper(fig, gs[0, 0], sim, exp.title, xmax=xmax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    return outfile


def export_csv(exp, sim, outfile):
    """Write the full simulated data for one experiment to CSV."""
    cols = ["time_s", "UV_mAU", "Cond_mS_cm", "Flow_mL_min",
            "conc_UV_molL", "conc_Cond_molL"]
    keys = ["t", "uv_mAU", "cond_mScm", "flow_mLmin", "conc_uv", "conc_cond"]
    rows = np.column_stack([sim[k] for k in keys])
    with open(outfile, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# experiment {exp.name}: {exp.description}"])
        w.writerow(cols)
        w.writerows(rows.tolist())
    return outfile
