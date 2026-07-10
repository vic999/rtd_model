#!/usr/bin/env python3
"""
Numerical verification of the RTD sub-models.

These are physics checks that do NOT require the (unavailable) experimental
data.  They confirm that the reproduced code obeys the conservation and
timing properties every RTD model must satisfy:

1. Mass conservation -- the area under an outlet pulse equals the area of the
   inlet pulse (a tracer is neither created nor destroyed).
2. Mean residence time -- for a single CST or DPF the first moment of the
   impulse response equals V / Vdot.
3. Steady-state gain -- a stepwise input eventually reaches the same plateau
   concentration at the outlet (unit gain).
4. Monotone ordering -- a larger filter (more hold-up) has a later peak than a
   smaller one at the same flow.

Run:  python3 verify.py
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from rtd.units import cst_outlet, dpf_outlet
from run_figures import simulate_pulse, simulate_step


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


def main():
    all_ok = True

    # --- 1 & 2: single CST impulse response --------------------------------
    t = np.linspace(0, 800, 8000)
    cin = np.zeros_like(t); cin[(t >= 1) & (t < 2)] = 1.0
    V, flow = 1000.0, 1.0                     # V/Vdot = 60 s
    out = cst_outlet(t, cin, V, flow)
    area_ratio = trapezoid(out, t) / trapezoid(cin, t)
    mrt = trapezoid(out * t, t) / trapezoid(out, t)
    all_ok &= check("CST mass conservation", abs(area_ratio - 1) < 0.02,
                    f"area_out/area_in = {area_ratio:.3f}")
    all_ok &= check("CST mean residence time", abs(mrt - 61.5) < 2.0,
                    f"MRT = {mrt:.1f} s (expected ~61.5)")

    # --- 2: single DPF impulse response ------------------------------------
    out = dpf_outlet(t, cin, volume_uL=V, length_mm=200, diameter_mm=0.75,
                     flow_mL_min=flow)
    area_ratio = trapezoid(out, t) / trapezoid(cin, t)
    mrt = trapezoid(out * t, t) / trapezoid(out, t)
    all_ok &= check("DPF mass conservation", abs(area_ratio - 1) < 0.02,
                    f"area_out/area_in = {area_ratio:.3f}")
    all_ok &= check("DPF mean residence time", abs(mrt - 61.5) < 3.0,
                    f"MRT = {mrt:.1f} s (expected ~61.5)")

    # --- 3: step response reaches unit gain (filter train) -----------------
    t, uv, cond = simulate_step(connection="filter", flow=1.0, surface=10,
                                c_tracer=0.05)
    all_ok &= check("Filter step plateau = feed conc.",
                    abs(uv.max() - 0.05) < 2e-3,
                    f"plateau = {uv.max():.4f} (feed 0.05)")

    # --- 4: larger filter -> later pulse peak at same flow -----------------
    _, uv3, _ = simulate_pulse(connection="filter", flow=1.0, surface=3)
    t10, uv10, _ = simulate_pulse(connection="filter", flow=1.0, surface=10)
    tpk3 = simulate_pulse(connection="filter", flow=1.0, surface=3)[0][uv3.argmax()]
    tpk10 = t10[uv10.argmax()]
    all_ok &= check("Peak time increases with filter size",
                    tpk10 > tpk3, f"3cm2 peak {tpk3:.0f}s < 10cm2 peak {tpk10:.0f}s")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
