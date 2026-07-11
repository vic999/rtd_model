"""
Injection system and inlet-signal generators (Chen et al., 2024, Sec. 2.2.3).

* Pulse injection via the sample loop  -> the loop is pre-filled with tracer
  at concentration ``c_tracer``; the finite loop volume enters the stream as a
  near-rectangular bolus whose width equals loop_volume / flow.

* Stepwise injection via a pump  -> a step change from the equilibration
  concentration to ``c_tracer`` (start-up), optionally returning to baseline
  (shut-down) after ``step_off_s``.

Concentrations are in molar units; the detector conversions (Beer /
Kohlrausch) are applied later, but for RTD *shape* the linear signal is used
directly.
"""

from __future__ import annotations

import numpy as np

from .flow import as_flow_fn


def pulse_inlet(t_grid, loop_volume_uL, flow_mL_min, c_tracer,
                c_base=0.0, t_start=5.0):
    """
    Rectangular pulse delivering a fixed loop VOLUME starting at t_start.

    A sample loop pushes out a fixed volume, not a fixed duration.  The pulse
    therefore ends when the cumulative delivered volume reaches
    ``loop_volume_uL``:

        int_{t_start}^{t_end} Vdot(t) dt = loop_volume_uL

    For a constant flow this reduces to the usual  width = loop_volume / Vdot.
    ``flow_mL_min`` may be a scalar or a FlowProfile.
    """
    t_grid = np.asarray(t_grid, float)
    fn = as_flow_fn(flow_mL_min)
    vdot = np.asarray(fn(t_grid), float) * 1000.0 / 60.0   # uL/s on the grid
    # cumulative volume delivered from t_start
    delivered = np.zeros_like(t_grid)
    dt = np.diff(t_grid)
    inc = 0.5 * (vdot[1:] + vdot[:-1]) * dt                # trapezoid
    started = t_grid[1:] > t_start
    delivered[1:] = np.cumsum(np.where(started, inc, 0.0))
    c = np.full_like(t_grid, c_base, dtype=float)
    mask = (t_grid >= t_start) & (delivered < loop_volume_uL)
    c[mask] = c_tracer
    return c


def step_inlet(t_grid, c_tracer, c_base=0.0, t_on=5.0, t_off=None):
    """
    Step from c_base to c_tracer at t_on; back to c_base at t_off (if given).
    """
    c = np.full_like(t_grid, c_base, dtype=float)
    c[t_grid >= t_on] = c_tracer
    if t_off is not None:
        c[t_grid >= t_off] = c_base
    return c


def combined_inlet(t_grid, c_step, pulse_volume_uL, flow_mL_min, c_pulse,
                   t_on=5.0, t_pulse=None, t_off=None):
    """
    Stepwise background (c_step) with a superimposed pulse during steady state
    (paper Sec. 2.3.4, experiments V5-V7).
    """
    c = step_inlet(t_grid, c_step, c_base=0.0, t_on=t_on, t_off=t_off)
    if t_pulse is not None:
        Vdot = flow_mL_min * 1000.0 / 60.0
        width = pulse_volume_uL / Vdot
        mask = (t_grid >= t_pulse) & (t_grid < t_pulse + width)
        c[mask] = c_pulse
    return c
