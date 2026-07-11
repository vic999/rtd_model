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
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .flow import as_flow_fn, flow_feature_scale


# --------------------------------------------------------------------------
# Optional Numba acceleration (improvement #10, part 4)
# --------------------------------------------------------------------------
# Numba JIT-compiles the DPF spatial stencil to machine code, avoiding the
# Python interpreter on the hottest inner loop.  It is a *soft* dependency:
# if Numba is not installed we fall back to the vectorised NumPy path, which
# gives identical results.  Set USE_NUMBA=False to force the NumPy path.
try:                                              # pragma: no cover
    from numba import njit as _njit
    HAVE_NUMBA = True
except Exception:                                 # Numba not installed
    HAVE_NUMBA = False

    def _njit(*args, **kwargs):
        """No-op stand-in for numba.njit supporting @_njit and @_njit(...)."""
        if args and callable(args[0]):
            return args[0]

        def wrap(fn):
            return fn
        return wrap

USE_NUMBA = HAVE_NUMBA


@_njit(cache=True)
def _dpf_rates_core(c, cf, u, Dax, inv_dz, inv_dz2):
    """
    DPF finite-volume rates (upwind advection + central diffusion) as an
    explicit loop -- fast under Numba.  Must stay identical to the NumPy path
    in ``dpf_outlet``.
    """
    n = c.shape[0]
    dcdt = np.empty(n)
    # inlet cell (Danckwerts): advective inflow + diffusive balance
    adv0 = u * (cf - c[0]) * inv_dz
    left_flux0 = u * (c[0] - cf)
    dcdt[0] = adv0 + (Dax * (c[1] - c[0]) * inv_dz - left_flux0) * inv_dz
    for i in range(1, n - 1):
        adv = u * (c[i - 1] - c[i]) * inv_dz
        dif = Dax * (c[i + 1] - 2.0 * c[i] + c[i - 1]) * inv_dz2
        dcdt[i] = adv + dif
    # outlet cell (Neumann): no diffusive flux through the last face
    dcdt[n - 1] = u * (c[n - 2] - c[n - 1]) * inv_dz \
        + Dax * (c[n - 2] - c[n - 1]) * inv_dz2
    return dcdt


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

    # Analytic Jacobian (improvement #10): d(dc/dt)/dc = -Vdot/V.  Supplying it
    # spares BDF the finite-difference probing it would otherwise do each step.
    def jac(t, y):
        return np.array([[-vdot(t) / volume_uL]])

    # Cap the step so the adaptive solver cannot step over narrow forcing
    # features (e.g. a short injection pulse) while staying fast elsewhere.
    if max_step is None:
        max_step = compute_max_step(t_grid, c_in, flow=flow_mL_min)
    sol = solve_ivp(
        rhs, (t_grid[0], t_grid[-1]), [c0], jac=jac,
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

    def _uD(t):
        u = vdot(t) / A                                    # mm/s (time-dependent)
        return u, u * diameter_mm / Pe                     # (u, Dax)

    if USE_NUMBA:
        # JIT-compiled explicit-loop core (improvement #10, part 4).
        def rhs(t, c):
            u, Dax = _uD(t)
            return _dpf_rates_core(c, cin(t), u, Dax, inv_dz, inv_dz2)
    else:
        # Vectorised NumPy path (identical result); used when Numba is absent.
        def rhs(t, c):
            cf = cin(t)
            u, Dax = _uD(t)
            adv = np.empty(n_cells)
            adv[0] = u * (cf - c[0]) * inv_dz
            adv[1:] = u * (c[:-1] - c[1:]) * inv_dz
            dif = np.empty(n_cells)
            left_flux0 = u * (c[0] - cf)                   # = Dax dc/dz at inlet
            dif[0] = (Dax * (c[1] - c[0]) * inv_dz - left_flux0) * inv_dz
            dif[1:-1] = Dax * (c[2:] - 2 * c[1:-1] + c[:-2]) * inv_dz2
            dif[-1] = Dax * (c[-2] - c[-1]) * inv_dz2
            return adv + dif

    # Analytic sparse Jacobian (improvement #10, part 1).  The finite-volume
    # stencil couples each cell only to its neighbours, so J is TRIDIAGONAL and
    # (for constant flow) independent of c; it depends on t only through u, Dax.
    #   row 0      : diag = -2u/dz - Dax/dz^2,           super = Dax/dz^2
    #   interior i : sub = u/dz + Dax/dz^2, diag = -u/dz - 2Dax/dz^2, super = Dax/dz^2
    #   row n-1    : sub = u/dz + Dax/dz^2, diag = -u/dz - Dax/dz^2
    n = n_cells

    def jac(t, c):
        u, Dax = _uD(t)
        a = u * inv_dz
        d = Dax * inv_dz2
        lower = np.full(n - 1, a + d)                      # J[i, i-1]
        upper = np.full(n - 1, d)                          # J[i, i+1]
        main = np.empty(n)
        main[0] = -2.0 * a - d
        main[1:-1] = -a - 2.0 * d
        main[-1] = -a - d
        return sparse.diags([lower, main, upper], [-1, 0, 1], format="csc")

    c_init = np.full(n_cells, c0, dtype=float)
    if max_step is None:
        max_step = compute_max_step(t_grid, c_in, flow=flow_mL_min)
    sol = solve_ivp(
        rhs, (t_grid[0], t_grid[-1]), c_init, jac=jac,
        t_eval=t_grid, method="BDF", rtol=1e-6, atol=1e-9, max_step=max_step,
    )
    # Outlet = concentration in the last cell.
    return sol.y[-1]
