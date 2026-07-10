"""
Simulation orchestration: chain an equipment train and read out signals.

Because the equipment train is a pure series with no downstream feedback, the
train is solved unit-by-unit: the outlet of unit i is the inlet of unit i+1.
The UV signal is recorded at the U9-D monitor and the conductivity signal at
the C9 monitor.
"""

from __future__ import annotations

import numpy as np

from .units import compute_max_step


def run_train(seq, t_grid, c_in, flow_mL_min, read_indices=None):
    """
    Propagate ``c_in`` through the ordered list ``seq`` of units.

    Returns
    -------
    signals : dict[int, ndarray]
        Concentration at each requested unit index (outlet of that unit).
    c_out : ndarray
        Concentration at the very end of the train.
    """
    read_indices = set(read_indices or [])
    # One max_step chosen from the (sharpest) original input; reused for every
    # unit so narrow injection features are resolved throughout the train.
    max_step = compute_max_step(t_grid, c_in)
    signals = {}
    c = c_in
    for i, unit in enumerate(seq):
        c = unit.propagate(t_grid, c, flow_mL_min, max_step=max_step)
        if i in read_indices:
            signals[i] = c.copy()
    return signals, c


def r2_score(y_true, y_pred):
    """Coefficient of determination (same definition as sklearn.r2_score)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
