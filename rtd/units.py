"""
Core RTD sub-models for continuous virus filtration.

Reproduces the mathematical models of:

    Chen, Recanati, De Mathia, Lin & Jungbauer (2024),
    "Residence time distribution in continuous virus filtration",
    Biotechnology and Bioengineering 121:1876-1888.
    DOI: 10.1002/bit.28696

Two building blocks are defined here:

* ``dpf_outlet``  -- Dispersed Plug Flow model (paper Eq. 1) for tubing,
                     sample loops, nozzles and the hollow-space compartment
                     V_I of the filter.
* ``cst_outlet``  -- Continuously Stirred Tank model (paper Eq. 2) for
                     valves, mixers, monitors and the fibre-wall compartment
                     V_wall.

Both are solved as *input/output* operators: given the inlet concentration
signal c_in(t) sampled on a common time grid, they return the outlet
concentration signal.  Because the equipment train has no downstream
feedback, an entire series of units can be simulated by chaining these
operators (the outlet of unit i is the inlet of unit i+1).

All volumes are handled in microlitres (uL), lengths in mm, so that flow
rate can be supplied in mL/min and converted internally.  Concentrations are
carried in arbitrary (molar) units -- the models are linear in concentration
so absolute scaling is irrelevant for RTD shape.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .flow import as_flow_fn, flow_feature_scale


# --------------------------------------------------------------------------
# Unit helpers
# --------------------------------------------------------------------------
def _flow_uL_per_s(flow_mL_min: float) -> float:
    """Convert a volumetric flow rate mL/min -> uL/s (accepts scalar only)."""
    return flow_mL_min * 1000.0 / 60.0


def _vdot_fn(flow):
    """Return a callable t[s] -> Vdot[uL/s] from a scalar or FlowProfile."""
    fn = as_flow_fn(flow)
    return lambda t: fn(t) * 1000.0 / 60.0


def _make_inlet(t_grid: np.ndarray, c_in: np.ndarray):
    """Return a callable c_in(t) with flat extrapolation, from sampled data."""
    return interp1d(
        t_grid, c_in, kind="linear",
        bounds_error=False, fill_value=(c_in[0], c_in[-1]),
    )


def compute_max_step(t_grid, c_in, flow=None):
    """
    Choose a solver ``max_step`` that (a) resolves the narrowest forcing feature
    (e.g. an injection pulse) so the adaptive integrator cannot step over it,
    (b) resolves any flow-rate change (ramp / saw-tooth), and (c) stays coarse
    enough elsewhere for speed.
    """
    t_grid = np.asarray(t_grid, float)
    c_in = np.asarray(c_in, float)
    span = t_grid[-1] - t_grid[0]
    cap = span / 500.0                         # >= ~500 steps across the run
    active = np.abs(c_in) > 1e-12              # "active" (nonzero) input samples
    if active.any():
        idx = np.flatnonzero(active)
        splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        widths = [t_grid[r[-1]] - t_grid[r[0]] for r in splits if len(r) > 1]
        if widths:
            cap = min(cap, min(widths) / 4.0)
    if flow is not None:
        cap = min(cap, flow_feature_scale(flow, t_grid))
    return max(cap, span / 20000.0)


# --------------------------------------------------------------------------
# CST model -- paper Eq. (2):   dc/dt = (Vdot / V) * (c_in - c)
# --------------------------------------------------------------------------
def cst_outlet(t_grid, c_in, volume_uL, flow_mL_min, c0=0.0, max_step=None):
    """
    Ideal continuously stirred tank.

    Parameters
    ----------
    t_grid : (N,) array of times [s]
    c_in   : (N,) inlet concentration sampled on t_grid
    volume_uL : tank hold-up volume [uL]
    flow_mL_min : volumetric flow rate [mL/min]
    c0 : initial tank concentration

    Returns
    -------
    (N,) outlet concentration = tank concentration.
    """
    vdot = _vdot_fn(flow_mL_min)                # callable t -> uL/s
    cin = _make_inlet(t_grid, c_in)

    def rhs(t, y):
        return np.array([vdot(t) / volume_uL * (cin(t) - y[0])])

    # Cap the step so the adaptive solver cannot step over narrow forcing
    # features (e.g. a short injection pulse) while staying fast elsewhere.
    if max_step is None:
        max_step = compute_max_step(t_grid, c_in, flow=flow_mL_min)
    sol = solve_ivp(
        rhs, (t_grid[0], t_grid[-1]), [c0],
        t_eval=t_grid, method="BDF", rtol=1e-6, atol=1e-9, max_step=max_step,
    )
    return sol.y[0]


# --------------------------------------------------------------------------
# DPF model -- paper Eq. (1):
#     dc/dt = -u dc/dz + Dax d2c/dz2
#     u   = Vdot / A,   A = pi d^2 / 4
#     Dax = u d / Pe,   Pe = 0.5   (Peclet based on tube diameter)
#     Danckwerts inlet BC:  Dax dc/dz|_0 = u (c(0) - c_in)
#     Neumann outlet BC:    dc/dz|_L = 0
# --------------------------------------------------------------------------
def dpf_outlet(t_grid, c_in, volume_uL, length_mm, diameter_mm, flow_mL_min,
               n_cells=40, Pe=0.5, c0=0.0, max_step=None):
    """
    Dispersed plug flow through a straight conduit, solved by a finite-volume
    method of lines (upwind advection + central diffusion) with Danckwerts
    boundary conditions.

    ``volume_uL`` overrides the geometric volume when a unit's tabulated
    hold-up differs from pi/4 d^2 L (as several do in Table 1); the
    cross-section A is then taken as volume/length so that residence time is
    exactly V/Vdot while the diameter still sets the dispersion length scale.
    """
    vdot = _vdot_fn(flow_mL_min)                           # callable t -> uL/s
    L = length_mm                                          # mm

    # Cross-sectional area consistent with the tabulated hold-up volume:
    #   A = V / L   (uL/mm = mm^2)     -> preserves mean residence time V/Vdot
    A = volume_uL / L                                      # mm^2

    dz = L / n_cells
    zc = (np.arange(n_cells) + 0.5) * dz                   # cell centres
    cin = _make_inlet(t_grid, c_in)

    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / dz ** 2

    def rhs(t, c):
        cf = cin(t)
        u = vdot(t) / A                                    # mm/s (time-dependent)
        Dax = u * diameter_mm / Pe                         # mm^2/s
        dcdt = np.empty_like(c)

        # --- advection (first-order upwind, u > 0) -----------------------
        # flux at faces; inlet face uses c_in, interior faces use upwind cell
        adv = np.empty(n_cells)
        adv[0] = u * (cf - c[0]) * inv_dz
        adv[1:] = u * (c[:-1] - c[1:]) * inv_dz

        # --- diffusion (central) with Danckwerts + Neumann BCs -----------
        dif = np.empty(n_cells)
        # inlet ghost from Danckwerts flux: Dax dc/dz|0 = u (c0 - c_in)
        # -> face gradient approximated so inlet diffusive flux balances
        left_flux0 = u * (c[0] - cf)                       # = Dax dc/dz at inlet
        dif[0] = (Dax * (c[1] - c[0]) * inv_dz - left_flux0) * inv_dz
        dif[1:-1] = Dax * (c[2:] - 2 * c[1:-1] + c[:-2]) * inv_dz2
        # outlet Neumann: dc/dz|L = 0 -> no diffusive flux through last face
        dif[-1] = Dax * (c[-2] - c[-1]) * inv_dz2

        dcdt[:] = adv + dif
        return dcdt

    c_init = np.full(n_cells, c0, dtype=float)
    if max_step is None:
        max_step = compute_max_step(t_grid, c_in, flow=flow_mL_min)
    sol = solve_ivp(
        rhs, (t_grid[0], t_grid[-1]), c_init,
        t_eval=t_grid, method="BDF", rtol=1e-6, atol=1e-9, max_step=max_step,
    )
    # Outlet = concentration in the last cell.
    return sol.y[-1]
