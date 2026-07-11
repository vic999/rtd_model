"""
Test suite for the RTD model (improvement #11).

Covers the physics invariants, the DPF grid-convergence behaviour, the
multi-component superposition and opposite-sign behaviour, injection-location
routing, and the experiment/CLI configuration. Run with:

    pytest -q
"""

import os
import sys

import numpy as np
import pytest
from scipy.integrate import trapezoid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rtd.units import cst_outlet, dpf_outlet
from rtd.flow import Constant, Ramp
from rtd import UV_NANO3, COND_NANO3


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------
def _impulse(T=120.0, n=4000):
    t = np.linspace(0, T, n)
    c = np.zeros_like(t)
    c[(t >= 1) & (t < 2)] = 1.0
    return t, c


def _moments(t, y):
    m0 = trapezoid(y, t)
    tbar = trapezoid(y * t, t) / m0
    var = trapezoid(y * (t - tbar) ** 2, t) / m0
    return m0, tbar, var


# --------------------------------------------------------------------------
# conservation & residence time
# --------------------------------------------------------------------------
def test_cst_mass_and_mrt():
    t, c = _impulse(T=600, n=6000)
    out = cst_outlet(t, c, volume_uL=1000.0, flow_mL_min=1.0)   # V/Vdot = 60 s
    m0, mrt, _ = _moments(t, out)
    assert abs(m0 / trapezoid(c, t) - 1.0) < 0.02
    assert abs(mrt - 61.5) < 2.0


@pytest.mark.parametrize("scheme", ["upwind", "vanleer"])
def test_dpf_mass_and_mrt(scheme):
    t, c = _impulse(T=300, n=6000)
    out = dpf_outlet(t, c, volume_uL=1000.0, length_mm=200, diameter_mm=0.75,
                     flow_mL_min=1.0, scheme=scheme)
    m0, mrt, _ = _moments(t, out)
    assert abs(m0 / trapezoid(c, t) - 1.0) < 0.02, "mass not conserved"
    assert abs(mrt - 61.5) < 3.0, "mean residence time off"


def test_constant_flow_regression():
    t, c = _impulse()
    a = cst_outlet(t, c, 1000.0, 1.0)
    b = cst_outlet(t, c, 1000.0, Constant(1.0))
    assert np.max(np.abs(a - b)) < 1e-12


def test_mass_conserved_varying_flow():
    t, c = _impulse(T=300, n=6000)
    prof = Ramp(2.0, t_start=0, t_ramp=60, v_start=0.2)
    out = cst_outlet(t, c, 1000.0, prof)
    vdot = prof(t) * 1000.0 / 60.0
    ratio = trapezoid(out * vdot, t) / trapezoid(c * vdot, t)
    assert abs(ratio - 1.0) < 0.02


# --------------------------------------------------------------------------
# DPF grid convergence (improvement #6)
# --------------------------------------------------------------------------
def test_vanleer_converges_faster_than_upwind():
    # long thin conduit -> high cell Peclet -> upwind badly over-diffuses
    t, c = _impulse(T=120, n=4000)
    geom = dict(volume_uL=260.0, length_mm=1324.2, diameter_mm=0.5, flow_mL_min=1.0)

    def var(scheme, n):
        y = dpf_outlet(t, c, n_cells=n, scheme=scheme, **geom)
        return _moments(t, y)[2]

    # van Leer is essentially grid-converged between 160 and 320 cells
    v160, v320 = var("vanleer", 160), var("vanleer", 320)
    assert abs(v160 - v320) / v320 < 0.15
    # upwind at 320 is still far more diffuse (larger spread) than van Leer
    assert var("upwind", 320) > 1.3 * v320


# --------------------------------------------------------------------------
# experiments / multi-component (improvements #2, #14)
# --------------------------------------------------------------------------
def test_config_loads_and_has_series():
    from rtd.experiments import load_config
    exps, defaults = load_config()
    names = {e.name for e in exps}
    assert {"C1", "C3-3", "V1", "V4-3"} <= names
    assert any(e.figure == 3 for e in exps)
    assert any(e.figure == 4 for e in exps)


def test_single_species_backward_compatible():
    from rtd.experiments import Experiment, simulate
    exp = Experiment(name="_", kind="pulse", connection="bypass", flow=1.0)
    r = simulate(exp, n_time=1200)
    t, uv, cond = r["t"], r["uv_mAU"], r["cond_mScm"]
    # single NaNO3 tracer: the two detectors see the same conserved tracer, so
    # the ratio of the areas equals the ratio of the molar coefficients
    # (independent of the small U9-D -> C9 transport delay).
    ratio = trapezoid(cond, t) / trapezoid(uv, t)
    assert abs(ratio - COND_NANO3 / UV_NANO3) < 0.02


def test_multicomponent_opposite_sign():
    from rtd.experiments import load_config, simulate, find_experiment
    exps, _ = load_config()
    r = simulate(find_experiment(exps, "TR1"), n_time=1000)
    uv, cond = r["uv_mAU"], r["cond_mScm"]
    # buffer present at t=0 -> conductivity starts high, UV starts ~0
    assert uv[10] < 5.0
    assert cond[10] > 8.0
    # across the transition conductivity DROPS below its start while UV RISES
    assert cond.min() < 0.7 * cond[10]
    assert uv.max() > 10 * (uv[10] + 1e-9)


def test_injection_location_routing():
    from rtd.equipment import build_train
    _, names_loop, _, _ = build_train("filter", surface_cm2=10, inject_at="Loop")
    _, names_pump, _, _ = build_train("filter", surface_cm2=10, inject_at="5")
    assert names_loop[0] == "Loop"
    assert names_pump[0] == "5"
    assert "Loop" not in names_pump           # loop hold-up skipped for the pump


# --------------------------------------------------------------------------
# CLI / config plumbing (improvement #14)
# --------------------------------------------------------------------------
def test_cli_parser_and_lookup():
    import rtd_cli
    from rtd.experiments import load_config, find_experiment
    parser = rtd_cli.build_parser()
    args = parser.parse_args(["plot", "--experiment", "C1", "--dpi", "150"])
    assert args.command == "plot" and args.dpi == 150
    exps, _ = load_config()
    assert find_experiment(exps, "c1").name == "C1"      # case-insensitive
    with pytest.raises(KeyError):
        find_experiment(exps, "does-not-exist")
