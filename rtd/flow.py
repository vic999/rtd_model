"""
Flow-rate profiles for the RTD model.

Every solver in this package can take its flow rate either as a plain number
(mL/min, constant -- the original behaviour) or as a **FlowProfile**: a callable
that returns the volumetric flow rate (mL/min) as a function of time (s).

    Vdot(t) [mL/min] = profile(t [s])

This lets the model reproduce the time-varying flow the paper describes -- the
delayed pump ramp on start-up (the controller reaches its set-point ~6 s late)
and the saw-tooth pattern at high flow -- as well as replay a *measured* flow
trace from an ÄKTA CSV.

All profiles accept a scalar or a NumPy array for ``t`` and return the same
shape, so they can be used both inside the ODE right-hand sides (scalar t) and
for plotting (array t).

Design
------
* ``Constant(v)``            -- flat flow (identical to passing the number v).
* ``Ramp(...)``              -- linear ramp from v_start to v_final.
* ``DelayedStep(...)``       -- flat v after a short linear lead-in (the ~6 s
                                pump lag); a Ramp with a physical default.
* ``Sawtooth(...)``          -- periodic saw-tooth between two levels.
* ``FromData(t, v)``         -- interpolate a measured flow trace.
* ``Piecewise([...])``       -- concatenate profiles over time windows.

Use ``as_flow_fn(x)`` to turn either a number or a FlowProfile into a callable,
and ``representative_flow(fn, t_grid)`` to get a single scalar (the set-point)
for sizing time windows and injection widths.
"""

from __future__ import annotations

import numpy as np


class FlowProfile:
    """Base class: a callable t[s] -> flow[mL/min]."""

    def __call__(self, t):
        raise NotImplementedError

    # convenience: sample on a grid (always returns an array)
    def sample(self, t_grid):
        return np.asarray(self(np.asarray(t_grid, float)), float)


class Constant(FlowProfile):
    """Constant flow rate ``v`` (mL/min)."""

    def __init__(self, v):
        self.v = float(v)

    def __call__(self, t):
        return np.full_like(np.asarray(t, float), self.v) if np.ndim(t) else self.v


class Ramp(FlowProfile):
    """
    Linear ramp from ``v_start`` to ``v_final`` (mL/min).

    Flow is ``v_start`` for t < t_start, ramps linearly over ``t_ramp`` seconds,
    then holds ``v_final``.
    """

    def __init__(self, v_final, t_start=0.0, t_ramp=30.0, v_start=0.0):
        self.v_final = float(v_final)
        self.v_start = float(v_start)
        self.t_start = float(t_start)
        self.t_ramp = float(max(t_ramp, 1e-9))

    def __call__(self, t):
        t = np.asarray(t, float)
        frac = np.clip((t - self.t_start) / self.t_ramp, 0.0, 1.0)
        v = self.v_start + (self.v_final - self.v_start) * frac
        return v if t.ndim else float(v)


class DelayedStep(FlowProfile):
    """
    Flat flow ``v`` reached after a short linear lead-in of ``lag`` seconds.

    Models the paper's observation that the pump reaches its preset flow with a
    delay of roughly 6 s.  Equivalent to ``Ramp(v, t_start, lag, v_start)``.
    """

    def __init__(self, v, lag=6.0, t_start=0.0, v_start=0.0):
        self._ramp = Ramp(v, t_start=t_start, t_ramp=lag, v_start=v_start)

    def __call__(self, t):
        return self._ramp(t)


class Sawtooth(FlowProfile):
    """
    Periodic saw-tooth flow between ``v_base`` and ``v_peak`` with ``period`` s.

    Reproduces the flow interruption / recovery pattern seen at high flow when a
    second tracer is introduced (paper Fig. 3g, 4).  Each cycle rises linearly
    from v_base to v_peak, then resets.
    """

    def __init__(self, v_base, v_peak, period, t_start=0.0):
        self.v_base = float(v_base)
        self.v_peak = float(v_peak)
        self.period = float(max(period, 1e-9))
        self.t_start = float(t_start)

    def __call__(self, t):
        t = np.asarray(t, float)
        phase = np.clip((t - self.t_start), 0.0, None) % self.period
        v = np.where(t < self.t_start, self.v_base,
                     self.v_base + (self.v_peak - self.v_base) * (phase / self.period))
        return v if t.ndim else float(v)


class FromData(FlowProfile):
    """Interpolate a measured flow trace (t in s, v in mL/min)."""

    def __init__(self, t, v):
        self.t = np.asarray(t, float)
        self.v = np.asarray(v, float)

    def __call__(self, t):
        out = np.interp(np.asarray(t, float), self.t, self.v)
        return out if np.ndim(t) else float(out)


class Piecewise(FlowProfile):
    """
    Concatenate profiles over time windows.

    ``segments`` is a list of (t_from, profile); the last profile with
    t_from <= t applies.  Useful for start-up ramp -> hold -> shut-down.
    """

    def __init__(self, segments):
        self.segments = sorted(segments, key=lambda s: s[0])

    def __call__(self, t):
        t = np.asarray(t, float)
        starts = np.array([s[0] for s in self.segments])
        if t.ndim == 0:
            idx = np.searchsorted(starts, t, side="right") - 1
            idx = max(idx, 0)
            return float(self.segments[idx][1](float(t)))
        out = np.empty_like(t)
        for i, tt in enumerate(t):
            idx = max(np.searchsorted(starts, tt, side="right") - 1, 0)
            out[i] = self.segments[idx][1](float(tt))
        return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def as_flow_fn(flow):
    """
    Normalise ``flow`` into a callable t[s] -> mL/min.

    Accepts a number (constant flow) or any FlowProfile / callable.
    """
    if isinstance(flow, FlowProfile) or callable(flow):
        return flow
    return Constant(flow)


def representative_flow(flow, t_grid):
    """
    A single scalar flow (mL/min) used for sizing time windows / pulse widths.

    Returns the set-point (max of the sampled profile), which is the steady
    operating flow for ramps and delayed steps.
    """
    fn = as_flow_fn(flow)
    vals = fn.sample(t_grid) if isinstance(fn, FlowProfile) else np.asarray(
        [fn(t) for t in t_grid], float)
    vals = np.asarray(vals, float)
    pos = vals[vals > 1e-9]
    return float(pos.max()) if pos.size else float(vals.max())


def flow_feature_scale(flow, t_grid):
    """
    Shortest timescale over which the flow changes (s), or inf if constant.
    Used to cap the solver ``max_step`` so a ramp/saw-tooth is resolved.
    """
    fn = as_flow_fn(flow)
    v = fn.sample(t_grid) if isinstance(fn, FlowProfile) else np.asarray(
        [fn(t) for t in t_grid], float)
    dv = np.abs(np.diff(v))
    changing = dv > (np.ptp(v) * 1e-3 if np.ptp(v) > 0 else np.inf)
    if not changing.any():
        return np.inf
    idx = np.flatnonzero(changing)
    # width of the changing region
    return max(t_grid[idx[-1]] - t_grid[idx[0]], t_grid[1] - t_grid[0]) / 4.0
