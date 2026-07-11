"""
Loader for ÄKTA UNICORN CSV exports and event detection.

A UNICORN export interleaves several curves, each with its OWN time column
(in minutes):

    UV 1_280 (min, mAU) | Cond (min, mS/cm) | System flow (min, ml/min) | ...

This module parses those pairs and resamples them onto a single uniform time
grid (in seconds), and provides simple detectors for the step-transition and
pulse events used to reconstruct the injection programme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .flow import FromData


def load_unicorn_csv(path):
    """
    Parse a UNICORN export.

    Returns a dict with raw (time, value) pairs in seconds:
        uv_t, uv, cond_t, cond, flow_t, flow
    """
    df = pd.read_csv(path, skiprows=3, header=None)

    def col(i):
        return pd.to_numeric(df[i], errors="coerce").to_numpy()

    out = {
        "uv_t":   col(0) * 60.0, "uv":   col(1),   # min -> s
        "cond_t": col(2) * 60.0, "cond": col(3),
        "flow_t": col(4) * 60.0, "flow": col(5),
    }
    # drop NaN pairs
    for a, b in [("uv_t", "uv"), ("cond_t", "cond"), ("flow_t", "flow")]:
        m = np.isfinite(out[a]) & np.isfinite(out[b])
        out[a], out[b] = out[a][m], out[b][m]
    return out


def resample(t_src, y_src, t_grid):
    """Linear resample of (t_src, y_src) onto t_grid (both in seconds)."""
    return np.interp(t_grid, t_src, y_src)


def common_grid(data, n=2000, t0=None, t1=None):
    """Build a uniform time grid (s) spanning the UV record (or [t0,t1])."""
    lo = data["uv_t"].min() if t0 is None else t0
    hi = data["uv_t"].max() if t1 is None else t1
    return np.linspace(lo, hi, n)


def nominal_flow(data):
    """Most common non-zero system-flow value (mL/min)."""
    f = data["flow"]
    f = f[f > 1e-6]
    if len(f) == 0:
        return None
    vals, counts = np.unique(np.round(f, 2), return_counts=True)
    return float(vals[counts.argmax()])


def detect_events(t, cond, uv):
    """
    Locate the buffer transitions (from conductivity) and the pulse (from UV).

    The buffer "transition" is a programmed gradient, so it has a finite width:
    we return the ramp edges (10 %/90 % crossings), not just a midpoint.

    Returns dict:
        on_start,  on_end   -- down-gradient (enter low-cond plateau) edges [s]
        off_start, off_end  -- up-gradient   (leave low-cond plateau) edges [s]
        t_pulse             -- UV spike location on the plateau           [s]
    """
    t = np.asarray(t, float)
    c = np.asarray(cond, float)
    hi, lo = np.percentile(c, 95), np.percentile(c, 5)
    rng = hi - lo
    hi_lvl = hi - 0.1 * rng                 # "still high" threshold
    lo_lvl = lo + 0.1 * rng                 # "reached low" threshold

    low_plateau = np.flatnonzero(c < lo_lvl)   # samples firmly on the low plateau
    i0, i1 = low_plateau[0], low_plateau[-1]

    def last_cross_before(i, level, above):
        seg = np.arange(0, i)
        cond_ = (c[seg] > level) if above else (c[seg] < level)
        hit = seg[cond_]
        return hit[-1] if len(hit) else 0

    def first_cross_after(i, level, above):
        seg = np.arange(i, len(c))
        cond_ = (c[seg] > level) if above else (c[seg] < level)
        hit = seg[cond_]
        return hit[0] if len(hit) else len(c) - 1

    on_start = t[last_cross_before(i0, hi_lvl, above=True)]   # leaves high plateau
    on_end = t[i0]                                            # reaches low plateau
    off_start = t[i1]                                         # leaves low plateau
    off_end = t[first_cross_after(i1, hi_lvl, above=True)]    # back to high plateau

    # Pulse = UV maximum within the plateau TIME window [on_end, off_start].
    # (Do NOT mask by low conductivity: the pulse itself spikes the conductivity
    #  above the low threshold, which would exclude the very peak we want.)
    plateau_win = (t >= on_end) & (t <= off_start)
    uv_masked = np.where(plateau_win, uv, -np.inf)
    t_pulse = t[int(np.argmax(uv_masked))]

    return {"on_start": float(on_start), "on_end": float(on_end),
            "off_start": float(off_start), "off_end": float(off_end),
            "t_pulse": float(t_pulse)}


def detect_run_parameters(data, n=1600, t=None):
    """
    Automatically detect the run parameters from a raw ÄKTA export.

    Generalises the analysis so any CSV can be characterised without the caller
    knowing the experiment in advance.  Returns a dict with:

        t, uv, cond        -- signals resampled on a common grid
        flow_profile       -- a FlowProfile (FromData) built from the measured
                              System-flow column (drives the model directly)
        flow_setpoint      -- steady flow (mL/min, max of the positive samples)
        flow_min           -- min positive flow (mL/min)
        flow_is_varying    -- True if the flow changes appreciably during the run
        events             -- transition edges + pulse time (see detect_events)
        has_pulse          -- True if a distinct UV spike sits on the plateau
        has_transition     -- True if a conductivity plateau/transition exists
        pulse_prominence   -- UV spike height above the plateau (mAU)
        duration_s         -- record length (s)
    """
    if t is None:
        t = common_grid(data, n=n)
    uv = resample(data["uv_t"], data["uv"], t)
    cond = resample(data["cond_t"], data["cond"], t)

    # --- flow: build a FromData profile straight from the measurement --------
    flow_profile = FromData(data["flow_t"], data["flow"])
    fvals = np.asarray(flow_profile(t), float)
    pos = fvals[fvals > 1e-6]
    flow_setpoint = float(pos.max()) if pos.size else 0.0
    flow_min = float(pos.min()) if pos.size else 0.0
    flow_is_varying = bool(pos.size and (np.ptp(pos) > 0.05 * max(flow_setpoint, 1e-9)))

    # --- transition + pulse events ------------------------------------------
    rng = np.ptp(cond)
    has_transition = bool(rng > 0.2 * (abs(np.median(cond)) + 1e-9))
    try:
        events = detect_events(t, cond, uv)
    except Exception:
        events = None
        has_transition = False

    # --- pulse detection: UV spike prominence above the local plateau --------
    has_pulse, prominence = False, 0.0
    if events is not None:
        on_end, off_start = events["on_end"], events["off_start"]
        plateau_mask = (t > on_end) & (t < off_start)
        if plateau_mask.sum() > 5:
            plateau_uv = np.median(uv[plateau_mask])
            i_pulse = int(np.argmin(np.abs(t - events["t_pulse"])))
            prominence = float(uv[i_pulse] - plateau_uv)
            has_pulse = prominence > 0.15 * (np.ptp(uv) + 1e-9)

    return dict(
        t=t, uv=uv, cond=cond,
        flow_profile=flow_profile, flow_setpoint=flow_setpoint,
        flow_min=flow_min, flow_is_varying=flow_is_varying,
        events=events, has_pulse=has_pulse, has_transition=has_transition,
        pulse_prominence=prominence, duration_s=float(t[-1] - t[0]),
    )
