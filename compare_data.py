#!/usr/bin/env python3
"""
Compare the RTD model against a real ÄKTA UNICORN run, auto-detect the run
parameters, and infer the flow-path configuration.

The script is *general*: point it at any UNICORN CSV and it will

  * detect the flow profile from the measured System-flow column and drive the
    model with it (rtd.flow.FromData) -- no hard-coded flow rate,
  * detect the buffer-transition edges and whether a pulse is present,
  * reconstruct the corresponding input programmes,
  * propagate them through each candidate train, fit the two detectors
    (Beer's / Kohlrausch's law -> the linear a,b,c set units, not shape),
  * and report which configuration reproduces the data best.

For the bundled file this is a combined stepwise + pulse experiment (260 µL /
0.5 M NaNO3 pulse on a 50 mM NaNO3 transition).  Conductivity steps DOWN during
the NaNO3 plateau while the pulse makes both UV and conductivity spike -- the
step (buffer swap) and the pulse (added NaNO3) are two independent components.

Run:  python3 compare_data.py "<path to csv>"
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rtd import build_train, run_train, r2_score, as_flow_fn
from rtd.injection import pulse_inlet
from rtd.data import (load_unicorn_csv, resample, detect_run_parameters)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(
    HERE, "20230208_260Microliters0.5MPulseson50mMNaNO3Transition_FDM 002.csv")

CONFIGS = [
    ("ÄKTA bypass",       dict(connection="bypass")),
    ("Connector",         dict(connection="connector")),
    ("10 cm² filter",     dict(connection="filter", surface_cm2=10)),
    ("3 cm² filter",      dict(connection="filter", surface_cm2=3)),
    ("100 cm² filter",    dict(connection="filter", surface_cm2=100)),
]


def _ramp_input(t, ev):
    """
    Buffer-B fraction input: a linear up-gradient over [on_start, on_end],
    a plateau at 1, then a linear down-gradient over [off_start, off_end].
    This mirrors the *programmed* ÄKTA gradient (the transition is not a step).
    """
    f = np.zeros_like(t)
    up = (t > ev["on_start"]) & (t < ev["on_end"])
    f[up] = (t[up] - ev["on_start"]) / (ev["on_end"] - ev["on_start"])
    f[(t >= ev["on_end"]) & (t <= ev["off_start"])] = 1.0
    dn = (t > ev["off_start"]) & (t < ev["off_end"])
    f[dn] = 1.0 - (t[dn] - ev["off_start"]) / (ev["off_end"] - ev["off_start"])
    return f


def basis_responses(cfg_kw, t, ev, flow, loop_uL=260.0):
    """
    Propagate the transition (ramp) and pulse inputs through a train; return the
    UV- and cond-monitor responses to each (4 arrays).

    ``flow`` may be a scalar or a FlowProfile (e.g. FromData of the measured
    System-flow column).  The pulse is delivered by VOLUME via pulse_inlet, so
    it is correct even when the flow varies.
    """
    seq, names, uv_i, cond_i = build_train(**cfg_kw)

    trans_in = _ramp_input(t, ev)
    pulse_in = pulse_inlet(t, loop_uL, flow, c_tracer=1.0, t_start=ev["t_pulse"])

    s_sig, _ = run_train(seq, t, trans_in, flow, read_indices=[uv_i, cond_i])
    p_sig, _ = run_train(seq, t, pulse_in, flow, read_indices=[uv_i, cond_i])
    return (s_sig[uv_i], p_sig[uv_i], s_sig[cond_i], p_sig[cond_i])


def _shift(y, k):
    """Shift array y by integer k samples (positive = later), zero-fill."""
    out = np.zeros_like(y)
    if k == 0:
        out[:] = y
    elif k > 0:
        out[k:] = y[:-k]
    else:
        out[:k] = y[-k:]
    return out


def fit_detector(bases, y_meas):
    """Least-squares fit  y ~ a*b1 + b*b2 + c ; return (coef, yhat, R²)."""
    B = np.column_stack([bases[0], bases[1], np.ones_like(y_meas)])
    coef, *_ = np.linalg.lstsq(B, y_meas, rcond=None)
    yhat = B @ coef
    return coef, yhat, r2_score(y_meas, yhat)


def _pulse_r2(t, ev, pulse_b, y_meas):
    """
    Isolated RTD test on the 260 µL bolus.  In a window around the pulse,
    subtract the local (pre-pulse) baseline and fit  residual ~ A*pulse_shift + B
    over a local lag search.  Returns the best R² and the fitted curve on window.
    """
    t0 = ev["t_pulse"] - 120.0
    t1 = ev["t_pulse"] + 0.6 * (ev["off_start"] - ev["t_pulse"])
    w = (t >= t0) & (t <= t1)
    yw = y_meas[w]
    pre = (t >= t0) & (t < ev["t_pulse"] - 20.0)
    base = np.median(y_meas[pre]) if pre.any() else yw[0]
    resid = yw - base
    dt = t[1] - t[0]
    best_r, best_fit, best_lag = -np.inf, None, 0
    for lag in range(0, int(400 / dt) + 1):         # up to ~400 s input->detector
        pk = _shift(pulse_b, lag)[w]
        B = np.column_stack([pk, np.ones_like(pk)])
        coef, *_ = np.linalg.lstsq(B, resid, rcond=None)
        fit = B @ coef
        r = r2_score(resid, fit)
        if r > best_r:
            best_r, best_fit, best_lag = r, fit + base, lag
    return best_r, w, best_fit


def evaluate(cfg_kw, t, ev, flow, uv_meas, cond_meas, dt):
    """Full-run fit (context) + isolated pulse R² (the RTD discriminator)."""
    su, pu, sc, pc = basis_responses(cfg_kw, t, ev, flow)

    # full-run overlay fit with a single global input->detector shift
    best = None
    for k in range(0, int(len(t) * 0.5), max(1, len(t) // 300)):
        suk, puk = _shift(su, k), _shift(pu, k)
        sck, pck = _shift(sc, k), _shift(pc, k)
        _, uvhat, r_uv = fit_detector((suk, puk), uv_meas)
        _, condhat, r_cond = fit_detector((sck, pck), cond_meas)
        score = 0.5 * (r_uv + r_cond)
        if best is None or score > best["score"]:
            best = dict(score=score, r_uv=r_uv, r_cond=r_cond, k=k,
                        uvhat=uvhat, condhat=condhat)

    # isolated pulse test (independent local alignment)
    r_uv_p, w, uv_pulse_fit = _pulse_r2(t, ev, pu, uv_meas)
    r_cond_p, _, cond_pulse_fit = _pulse_r2(t, ev, pc, cond_meas)
    best.update(r_uv_pulse=r_uv_p, r_cond_pulse=r_cond_p,
                pulse_w=w, uv_pulse_fit=uv_pulse_fit, cond_pulse_fit=cond_pulse_fit)
    return best


def _report_parameters(p):
    """Print the auto-detected run parameters."""
    ev = p["events"]
    fv = "varying" if p["flow_is_varying"] else "constant"
    print("=" * 64)
    print("AUTO-DETECTED RUN PARAMETERS")
    print("-" * 64)
    print(f"  duration            : {p['duration_s']:.0f} s")
    print(f"  flow (from data)    : {fv}, set-point {p['flow_setpoint']:.2f} "
          f"mL/min (min {p['flow_min']:.2f}) -> driven by FromData profile")
    print(f"  buffer transition   : {'yes' if p['has_transition'] else 'no'}")
    if ev is not None:
        print(f"    gradient edges (s): on {ev['on_start']:.0f}->{ev['on_end']:.0f}, "
              f"off {ev['off_start']:.0f}->{ev['off_end']:.0f}")
    print(f"  pulse present       : {'yes' if p['has_pulse'] else 'no'} "
          f"(prominence {p['pulse_prominence']:.1f} mAU)")
    if ev is not None:
        print(f"    pulse time (s)    : {ev['t_pulse']:.0f}")
    print("=" * 64 + "\n")


def main(csv_path):
    data = load_unicorn_csv(csv_path)
    params = detect_run_parameters(data, n=1600)
    _report_parameters(params)

    t = params["t"]
    dt = t[1] - t[0]
    uv_meas, cond_meas = params["uv"], params["cond"]
    ev = params["events"]
    flow = params["flow_profile"]          # FromData profile from the measurement
    if ev is None:
        print("No buffer transition detected -- cannot reconstruct inputs; abort.")
        return

    print(f"{'configuration':16s}  {'R2_UV':>7s} {'R2_Cond':>7s} {'R2_pulse(UV/Cond)':>18s}")
    results = []
    for label, kw in CONFIGS:
        best = evaluate(kw, t, ev, flow, uv_meas, cond_meas, dt)
        results.append((label, kw, best))
        print(f"  {label:16s}  {best['r_uv']:6.3f} {best['r_cond']:7.3f}   "
              f"{best['r_uv_pulse']:6.3f} /{best['r_cond_pulse']:6.3f}")

    # Rank by the full-run mean R²: it cleanly separates no-filter from filter
    # configurations (a filter's large dead volume distorts the whole trace).
    # The pulse R² is reported as a secondary RTD diagnostic -- it is a weaker
    # discriminator here because the real pulse tails more than the ideal model.
    label, kw, best = max(results, key=lambda r: r[2]["score"])
    print(f"\nBest configuration (by full-run mean R²): {label}  "
          f"full-run mean R² = {best['score']:.3f}  "
          f"(pulse R² diagnostic = {0.5*(best['r_uv_pulse']+best['r_cond_pulse']):.3f})")

    # overlay plot for the winning configuration: full run + pulse zoom
    fig = plt.figure(figsize=(13, 8))
    a1 = fig.add_subplot(2, 2, 1)
    a2 = fig.add_subplot(2, 2, 3, sharex=a1)
    z1 = fig.add_subplot(2, 2, 2)
    z2 = fig.add_subplot(2, 2, 4, sharex=z1)

    a1.plot(t, uv_meas, color="#1f6fb2", lw=1.2, label="experiment")
    a1.plot(t, best["uvhat"], "k--", lw=1.0, label="model")
    a1.set_ylabel("UV 280 (mAU)"); a1.legend(loc="upper left")
    a1.set_title(f"Full run — {label}  (R²_UV={best['r_uv']:.3f}, "
                 f"R²_Cond={best['r_cond']:.3f})")
    # measured flow (drives the model) on a third axis
    aflow = a1.twinx()
    flow_vals = np.atleast_1d(as_flow_fn(flow)(t)) * np.ones_like(t)
    aflow.plot(t, flow_vals, color="#e69500", lw=1.0, ls=":")
    aflow.set_ylabel("Flow (mL/min)", color="#e69500")
    aflow.set_ylim(0, max(1e-6, np.max(flow_vals) * 1.6))
    a2.plot(t, cond_meas, color="#2ca02c", lw=1.2)
    a2.plot(t, best["condhat"], "k--", lw=1.0)
    a2.set_ylabel("Cond (mS/cm)"); a2.set_xlabel("time (s)")

    # pulse zoom window with the isolated pulse fit
    w = best["pulse_w"]
    zt0 = ev["t_pulse"] - 120.0
    zt1 = ev["t_pulse"] + 0.6 * (ev["off_start"] - ev["t_pulse"])
    z1.plot(t, uv_meas, color="#1f6fb2", lw=1.2, label="experiment")
    z1.plot(t[w], best["uv_pulse_fit"], "k--", lw=1.2, label="model (pulse)")
    z1.set_xlim(zt0, zt1); z1.set_ylabel("UV 280 (mAU)"); z1.legend(loc="upper right")
    z1.set_title(f"Pulse zoom — RTD test  (R²_UV={best['r_uv_pulse']:.3f}, "
                 f"R²_Cond={best['r_cond_pulse']:.3f})")
    z2.plot(t, cond_meas, color="#2ca02c", lw=1.2)
    z2.plot(t[w], best["cond_pulse_fit"], "k--", lw=1.2)
    z2.set_xlim(zt0, zt1); z2.set_ylabel("Cond (mS/cm)"); z2.set_xlabel("time (s)")
    fig.tight_layout()
    out = os.path.join(HERE, "data_comparison.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # ranking bar summary
    fig2, ax = plt.subplots(figsize=(8, 4))
    labels = [r[0] for r in results]
    ax.bar(labels, [r[2]["r_uv"] for r in results], width=0.4, align="edge",
           label="R² UV", color="#1f6fb2")
    ax.bar(labels, [r[2]["r_cond"] for r in results], width=-0.4, align="edge",
           label="R² Cond", color="#2ca02c")
    ax.set_ylabel("R²"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Configuration inference by R²")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig2.tight_layout()
    out2 = os.path.join(HERE, "config_ranking.png")
    fig2.savefig(out2, dpi=130)
    print("wrote", out2)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv)
