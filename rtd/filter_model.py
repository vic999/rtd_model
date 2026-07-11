"""
Three-compartment virus-filter model (Chen et al., 2024, Figure 2).

The dead-end hollow-fibre filter hold-up is split into:

    V_I    hollow spaces & headers   -> DPF          (paper Eq. 1)
    V_wall within the fibre walls    -> CST          (paper Eq. 2)
    V_O    permeate space            -> radial TIS of `l` tanks, each an
                                        axial pair of interconnected CSTs
                                        with film resistance (Eqs. 3-9)

Flow path used for RTD:   inlet -> V_I -> V_wall -> V_O -> outlet.

Permeate-space model (paper Eqs. 3-9)
-------------------------------------
Radial non-ideal mixing is a tanks-in-series (TIS) cascade of `l` stages.
Within every stage the axial non-ideal mixing is two interconnected
well-mixed CSTs (Panchoi et al., 2022):

    eps * (V_O/l) dc_k1/dt = Vdot (c_k1_prev - c_k1) + eta*Vdot (c_k2 - c_k1)   (Eq.3)
  (1-eps)*(V_O/l) dc_k2/dt =                            eta*Vdot (c_k1 - c_k2)   (Eq.4)

* c_k1 is the through-flow tank (has in/outflow),
* c_k2 is the exchange tank (no net flow, exchanges at rate eta*Vdot),
* eps in [0,1] is the through-flow volume fraction, made time dependent by
  film resistance.

Film resistance (Eqs. 5-9): the Graetz-Leveque correlation gives a mass
transfer coefficient proportional to u^(1/3).  With
    u = Vdot / (pi * d_bar * L)         (side-area velocity of the equiv. cyl.)
the through-flow fraction follows (Eq. 8, clamped by Eq. 9):

    eps(t) = clip( 1 - u^(1/3) * (A3/Aj) *
                   (dceq_k1 + alpha*dc_k1) / dcmax ,  0, 1 )

Here `dc_k1 = c_k1_prev - c_k1` is the tracer driving force at the stage
inlet, and `dceq_k1` the complementary driving force of the displaced
equilibration buffer.  For a binary tracer/buffer system the two are mirror
images; we use dceq_k1 = -dc_k1 * (1/alpha_ratio) only through the fitted
composite parameter dcmax, so in practice `alpha`, `dcmax` (and the surface
ratio A3/Aj) are the knobs that were calibrated in the paper.

Set ``film_resistance=False`` to freeze eps at ``eps_const`` (a plain
two-CST-with-dead-zone cascade); this is useful for checking the influence of
the film-resistance term and for fast sanity runs.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .units import (dpf_outlet, cst_outlet, _flow_uL_per_s, _vdot_fn,
                    _make_inlet, compute_max_step)


# Reference smallest surface area (3 cm^2) used in the A3/Aj ratio (Eq. 7-8).
A3_REF_CM2 = 3.0
# Equivalent-cylinder length for the permeate space compartments [mm]
# (paper: L = 10.8 cm).
L_EQUIV_MM = 108.0


def _permeate_diameter_mm(volume_uL: float, length_mm: float = L_EQUIV_MM) -> float:
    """Equivalent-cylinder diameter d_bar = 2 sqrt(V/(pi L))  (paper caption)."""
    # volume in mm^3 (= uL), length in mm  ->  diameter in mm
    return 2.0 * np.sqrt(volume_uL / (np.pi * length_mm))


def permeate_space_outlet(
    t_grid, c_in, V_O_uL, flow_mL_min, surface_cm2,
    l=3, eta=0.13, alpha=1.14, dcmax=2.17e-7,
    film_resistance=True, eps_const=0.5, c0=0.0, max_step=None,
):
    """
    Permeate-space compartment V_O:  radial TIS of `l` axial two-CST stages.

    Parameters mirror the calibrated values reported by Chen et al. (2024):
        l=3, eta=0.13, alpha=1.14, dcmax=2.17e-7.

    Returns the outlet (last through-flow tank, k1 of stage l).
    """
    vdot = _vdot_fn(flow_mL_min)                          # callable t -> uL/s
    Vstage = V_O_uL / l                                   # uL per radial stage
    cin = _make_inlet(t_grid, c_in)

    d_bar = _permeate_diameter_mm(V_O_uL)                 # mm
    area_ratio = A3_REF_CM2 / surface_cm2                 # A3 / Aj

    eps_floor = 0.05

    def eps_of(dc_k1, u13):
        """Through-flow fraction eps.  Works on a scalar or a NumPy array of
        stage driving forces (vectorised, improvement #10, part 2)."""
        if not film_resistance:
            return np.clip(np.full_like(np.atleast_1d(dc_k1), eps_const,
                                        dtype=float), eps_floor, 0.999)
        # Eq. 8 with the binary-buffer term.  In a binary tracer/buffer system
        # the displaced-buffer driving force mirrors the tracer one, so
        #   dceq_k1 + alpha*dc_k1  ~  (alpha - 1) * dc_k1.
        # NOTE: quantitative use of this term needs the km,eq scale factor,
        # which is not tabulated in the paper; dcmax alone cannot fix the
        # absolute scale, so we saturate the argument to keep eps in (0,1)
        # and the ODE non-stiff.
        arg = u13 * area_ratio * ((alpha - 1.0) * np.asarray(dc_k1)) / dcmax
        val = 1.0 - (1.0 - eps_floor) * np.tanh(np.abs(arg))
        return np.clip(val, eps_floor, 0.999)

    # State vector: [c_k1_1, c_k2_1, c_k1_2, c_k2_2, ..., c_k1_l, c_k2_l]
    # Vectorised right-hand side (no Python stage loop).
    def rhs(t, y):
        cf = cin(t)
        Vdot = vdot(t)                                    # uL/s (time-dependent)
        u_side = Vdot / (np.pi * d_bar * L_EQUIV_MM)      # mm/s (film resistance)
        u13 = abs(u_side) ** (1.0 / 3.0)

        ck1 = y[0::2]
        ck2 = y[1::2]
        prev = np.empty(l)
        prev[0] = cf
        prev[1:] = ck1[:-1]
        dc_k1 = prev - ck1
        eps = np.broadcast_to(eps_of(dc_k1, u13), (l,))
        Vk1 = eps * Vstage
        Vk2 = (1.0 - eps) * Vstage

        dydt = np.empty_like(y)
        dydt[0::2] = (Vdot * dc_k1 + eta * Vdot * (ck2 - ck1)) / Vk1   # Eq. 3
        dydt[1::2] = (eta * Vdot * (ck1 - ck2)) / Vk2                  # Eq. 4
        return dydt

    # --- Jacobian info (improvement #10, part 1) ---------------------------
    # Sparsity pattern: dck1_j depends on ck1_j, ck2_j and ck1_{j-1};
    #                   dck2_j depends on ck1_j, ck2_j.
    S = sparse.lil_matrix((2 * l, 2 * l))
    for j in range(l):
        S[2 * j, 2 * j] = 1; S[2 * j, 2 * j + 1] = 1
        if j > 0:
            S[2 * j, 2 * (j - 1)] = 1
        S[2 * j + 1, 2 * j] = 1; S[2 * j + 1, 2 * j + 1] = 1
    S = S.tocsc()

    jac = None
    jac_sparsity = S
    if not film_resistance:
        # Constant eps -> the system is linear, so an exact analytic Jacobian is
        # available (depends on t only through Vdot).  Faster than finite diff.
        eps_c = float(np.clip(eps_const, eps_floor, 0.999))
        Vk1_c = eps_c * Vstage
        Vk2_c = (1.0 - eps_c) * Vstage

        def jac(t, y):
            Vdot = vdot(t)
            J = sparse.lil_matrix((2 * l, 2 * l))
            for j in range(l):
                J[2 * j, 2 * j] = (-Vdot - eta * Vdot) / Vk1_c
                J[2 * j, 2 * j + 1] = eta * Vdot / Vk1_c
                if j > 0:
                    J[2 * j, 2 * (j - 1)] = Vdot / Vk1_c
                J[2 * j + 1, 2 * j] = eta * Vdot / Vk2_c
                J[2 * j + 1, 2 * j + 1] = -eta * Vdot / Vk2_c
            return J.tocsc()
        jac_sparsity = None                               # jac supersedes sparsity

    y0 = np.full(2 * l, c0, dtype=float)
    if max_step is None:
        max_step = compute_max_step(t_grid, c_in, flow=flow_mL_min)
    sol = solve_ivp(
        rhs, (t_grid[0], t_grid[-1]), y0, jac=jac, jac_sparsity=jac_sparsity,
        t_eval=t_grid, method="BDF", rtol=1e-6, atol=1e-9, max_step=max_step,
    )
    return sol.y[2 * (l - 1)]                              # k1 of last stage


def filter_outlet(
    t_grid, c_in, flow_mL_min, surface_cm2,
    V_I_uL, len_I_mm, dia_I_mm, V_wall_uL, V_O_uL,
    l=3, eta=0.13, alpha=1.14, dcmax=2.17e-7,
    film_resistance=False, eps_const=0.85, c0=0.0, max_step=None,
):
    """
    Full three-compartment filter: V_I (DPF) -> V_wall (CST) -> V_O (permeate).
    """
    if max_step is None:
        max_step = compute_max_step(t_grid, c_in, flow=flow_mL_min)
    # V_I : hollow spaces & headers -> DPF
    c_I = dpf_outlet(
        t_grid, c_in, volume_uL=V_I_uL, length_mm=len_I_mm,
        diameter_mm=dia_I_mm, flow_mL_min=flow_mL_min, c0=c0, max_step=max_step,
    )
    # V_wall : within the fibre walls -> CST
    c_W = cst_outlet(t_grid, c_I, volume_uL=V_wall_uL,
                     flow_mL_min=flow_mL_min, c0=c0, max_step=max_step)
    # V_O : permeate space -> radial TIS of axial two-CST stages
    c_O = permeate_space_outlet(
        t_grid, c_W, V_O_uL=V_O_uL, flow_mL_min=flow_mL_min,
        surface_cm2=surface_cm2, l=l, eta=eta, alpha=alpha, dcmax=dcmax,
        film_resistance=film_resistance, eps_const=eps_const,
        c0=c0, max_step=max_step,
    )
    return c_O
