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

# Default DPF spatial scheme (improvement #6):
#   "vanleer" : higher-order flux-limited (MUSCL) advection -- low numerical
#               diffusion, so the modelled dispersion matches the physical
#               D_ax (Pe = 0.5) and results are grid-independent at far fewer
#               cells.  Default.
#   "upwind"  : first-order upwind -- simplest and gives an exact constant
#               tridiagonal Jacobian, but adds numerical diffusion ~ u*dz/2.
# See docs/DISCRETIZATION.md.
DPF_SCHEME = "vanleer"


@_njit(cache=True)
def _dpf_rates_core(c, cf, u, Dax, inv_dz, inv_dz2):
    """
    DPF finite-volume rates, FIRST-ORDER UPWIND advection + central diffusion,
    as an explicit loop (fast under Numba).  Simple and robust but adds
    numerical diffusion ~ u*dz/2.  Must match the NumPy path in ``dpf_outlet``.
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


@_njit(cache=True)
def _dpf_vanleer_core(c, cf, u, Dax, inv_dz, inv_dz2):
    """
    DPF finite-volume rates, HIGHER-ORDER van Leer (MUSCL) advection + central
    diffusion, in conservative flux form.  The face concentration is a
    slope-limited linear reconstruction from the upwind cell, which is ~2nd
    order in smooth regions but limits toward first order near sharp fronts, so
    it avoids the large numerical diffusion of plain upwind WITHOUT introducing
    spurious oscillations.  Flow is forward (u > 0).

    Face fluxes F[j] (j = 0..n):  F[0] inlet (advective inflow u*cf),
    F[n] outlet (upwind u*c[n-1]); interior F[j] uses a limited left state
    reconstructed from cell j-1.  The inlet ghost value is taken as cf.
    """
    n = c.shape[0]
    # TOTAL face flux Ftot[j] = advective - diffusive (conservative form).
    Ftot = np.empty(n + 1)
    Ftot[0] = u * cf                                  # Danckwerts inlet total flux
    for j in range(1, n):                             # interior faces
        cU = c[j - 1]                                 # upwind cell
        if j - 2 < 0:                                 # one further upwind (ghost=cf)
            cUm = cf
        else:
            cUm = c[j - 2]
        den = cU - cUm
        num = c[j] - cU
        if den == 0.0:
            phi = 0.0
        else:
            r = num / den
            phi = (r + abs(r)) / (1.0 + abs(r))       # van Leer limiter
        cL = cU + 0.5 * phi * (cU - cUm)              # limited face value
        Ftot[j] = u * cL - Dax * (c[j] - c[j - 1]) * inv_dz
    Ftot[n] = u * c[n - 1]                            # Neumann outlet (no diffusion)

    dcdt = np.empty(n)
    for i in range(n):
        dcdt[i] = -(Ftot[i + 1] - Ftot[i]) * inv_dz   # exact conservation
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
def _vanleer_rates_np(c, cf, u, Dax, inv_dz, inv_dz2):
    """NumPy fallback for the van Leer core (used when Numba is absent).
    Conservative total-flux form -- must match ``_dpf_vanleer_core``."""
    n = c.shape[0]
    Ftot = np.empty(n + 1)
    Ftot[0] = u * cf                                     # Danckwerts inlet
    cU = c[:-1]                                          # upwind cells, faces 1..n-1
    cUm = np.empty(n - 1)
    cUm[0] = cf
    cUm[1:] = c[:-2]
    den = cU - cUm
    num = c[1:] - cU
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den != 0.0, num / np.where(den != 0.0, den, 1.0), 0.0)
    phi = (r + np.abs(r)) / (1.0 + np.abs(r))            # van Leer limiter
    cL = cU + 0.5 * phi * (cU - cUm)
    Ftot[1:n] = u * cL - Dax * (c[1:] - c[:-1]) * inv_dz
    Ftot[n] = u * c[-1]                                  # Neumann outlet
    return -(Ftot[1:] - Ftot[:-1]) * inv_dz             # exact conservation


def dpf_outlet(t_grid, c_in, volume_uL, length_mm, diameter_mm, flow_mL_min,
               n_cells=40, Pe=0.5, c0=0.0, max_step=None, scheme=None):
    """
    Dispersed plug flow through a straight conduit, solved by a finite-volume
    method of lines with Danckwerts boundary conditions.

    ``scheme`` selects the advection discretization (default ``DPF_SCHEME``):
      * "vanleer" -- higher-order flux-limited (MUSCL); low numerical diffusion.
      * "upwind"  -- first-order upwind; exact constant tridiagonal Jacobian.
    See docs/DISCRETIZATION.md.

    ``volume_uL`` overrides the geometric volume when a unit's tabulated
    hold-up differs from pi/4 d^2 L (as several do in Table 1); the
    cross-section A is then taken as volume/length so that residence time is
    exactly V/Vdot while the diameter still sets the dispersion length scale.
    """
    scheme = (scheme or DPF_SCHEME).lower()
    vdot = _vdot_fn(flow_mL_min)                           # callable t -> uL/s
    L = length_mm                                          # mm

    # Cross-sectional area consistent with the tabulated hold-up volume:
    #   A = V / L   (uL/mm = mm^2)     -> preserves mean residence time V/Vdot
    A = volume_uL / L                                      # mm^2

    dz = L / n_cells
    cin = _make_inlet(t_grid, c_in)

    inv_dz = 1.0 / dz
    inv_dz2 = 1.0 / dz ** 2
    n = n_cells

    def _uD(t):
        u = vdot(t) / A                                    # mm/s (time-dependent)
        return u, u * diameter_mm / Pe                     # (u, Dax)

    if scheme == "vanleer":
        core = _dpf_vanleer_core if USE_NUMBA else _vanleer_rates_np

        def rhs(t, c):
            u, Dax = _uD(t)
            return core(c, float(cin(t)), u, Dax, inv_dz, inv_dz2)

        # The MUSCL stencil for cell i reaches cells i-2..i+1, so the Jacobian
        # is banded (bands -2,-1,0,+1) but state-dependent (the limiter is
        # nonlinear).  Give BDF the sparsity pattern for a cheap grouped
        # finite-difference Jacobian instead of an analytic one.
        jac = None
        bands = [np.ones(n - 2), np.ones(n - 1), np.ones(n), np.ones(n - 1)]
        jac_sparsity = sparse.diags(bands, [-2, -1, 0, 1], format="csc")

    elif scheme == "upwind":
        if USE_NUMBA:
            def rhs(t, c):
                u, Dax = _uD(t)
                return _dpf_rates_core(c, cin(t), u, Dax, inv_dz, inv_dz2)
        else:
            def rhs(t, c):
                cf = cin(t)
                u, Dax = _uD(t)
                adv = np.empty(n)
                adv[0] = u * (cf - c[0]) * inv_dz
                adv[1:] = u * (c[:-1] - c[1:]) * inv_dz
                dif = np.empty(n)
                left_flux0 = u * (c[0] - cf)
                dif[0] = (Dax * (c[1] - c[0]) * inv_dz - left_flux0) * inv_dz
                dif[1:-1] = Dax * (c[2:] - 2 * c[1:-1] + c[:-2]) * inv_dz2
                dif[-1] = Dax * (c[-2] - c[-1]) * inv_dz2
                return adv + dif

        # First-order upwind: exact constant TRIDIAGONAL Jacobian (see #10).
        def jac(t, c):
            u, Dax = _uD(t)
            a = u * inv_dz
            d = Dax * inv_dz2
            lower = np.full(n - 1, a + d)
            upper = np.full(n - 1, d)
            main = np.empty(n)
            main[0] = -2.0 * a - d
            main[1:-1] = -a - 2.0 * d
            main[-1] = -a - d
            return sparse.diags([lower, main, upper], [-1, 0, 1], format="csc")

        jac_sparsity = None
    else:
        raise ValueError(f"Unknown DPF scheme {scheme!r} (use 'vanleer' or 'upwind')")

    c_init = np.full(n_cells, c0, dtype=float)
    if max_step is None:
        max_step = compute_max_step(t_grid, c_in, flow=flow_mL_min)
    sol = solve_ivp(
        rhs, (t_grid[0], t_grid[-1]), c_init,
        jac=(jac if scheme == "upwind" else None),
        jac_sparsity=jac_sparsity,
        t_eval=t_grid, method="BDF", rtol=1e-6, atol=1e-9, max_step=max_step,
    )
    # Outlet = concentration in the last cell.
    return sol.y[-1]
