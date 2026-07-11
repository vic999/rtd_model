#!/usr/bin/env python3
"""
Grid-convergence study for the DPF discretization (improvement #6).

Demonstrates that the higher-order van Leer (MUSCL) scheme reaches a
grid-independent answer at far fewer cells than first-order upwind, because it
does not add the large numerical diffusion (~ u*dz/2) that upwind does.

For an impulse fed through the sample loop (the worst case: long and thin, so
the cell Peclet number is high), it sweeps the cell count for both schemes and
reports the peak height, mean residence time (first moment) and variance
(spread) of the outlet response.  A converged scheme shows these stop changing.

Outputs: convergence_study.png  and a printed table.

    python3 convergence_study.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

from rtd.units import dpf_outlet

HERE = os.path.dirname(os.path.abspath(__file__))

# Sample loop geometry (Table 1): d = 0.5 mm, L = 1324.2 mm, V = 260 uL.
GEOM = dict(volume_uL=260.0, length_mm=1324.2, diameter_mm=0.5, flow_mL_min=1.0)
N_CELLS = [20, 40, 80, 160, 320, 640]
SCHEMES = ["upwind", "vanleer"]


def moments(t, y):
    m0 = trapezoid(y, t)
    tbar = trapezoid(y * t, t) / m0
    var = trapezoid(y * (t - tbar) ** 2, t) / m0
    return m0, tbar, var, y.max()


def main():
    t = np.linspace(0, 120, 6000)
    cin = np.zeros_like(t)
    cin[(t >= 1) & (t < 2)] = 1.0
    area_in = trapezoid(cin, t)

    results = {s: [] for s in SCHEMES}
    print(f"{'scheme':8s} {'n':>4s} {'area/in':>8s} {'MRT':>7s} {'var':>7s} {'peak':>8s}")
    for s in SCHEMES:
        for n in N_CELLS:
            y = dpf_outlet(t, cin, n_cells=n, scheme=s, **GEOM)
            m0, tbar, var, pk = moments(t, y)
            results[s].append((n, m0 / area_in, tbar, var, pk))
            print(f"{s:8s} {n:4d} {m0/area_in:8.4f} {tbar:7.2f} {var:7.2f} {pk:8.4f}")

    # plot: peak and variance vs cell count for both schemes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for s, color in [("upwind", "#d62728"), ("vanleer", "#1f6fb2")]:
        n = [r[0] for r in results[s]]
        peak = [r[4] for r in results[s]]
        var = [r[3] for r in results[s]]
        ax1.plot(n, peak, "o-", color=color, label=s)
        ax2.plot(n, var, "o-", color=color, label=s)
    ax1.set_xscale("log", base=2); ax1.set_xlabel("cells"); ax1.set_ylabel("peak height")
    ax1.set_title("Outlet peak vs mesh (higher = less over-diffused)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xscale("log", base=2); ax2.set_xlabel("cells"); ax2.set_ylabel("variance (spread)")
    ax2.set_title("Outlet spread vs mesh (converges to the physical value)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("DPF grid convergence — sample loop impulse response\n"
                 "van Leer reaches the grid-independent answer at far fewer cells "
                 "than first-order upwind", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(HERE, "convergence_study.png")
    fig.savefig(out, dpi=130)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
