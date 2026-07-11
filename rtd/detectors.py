"""
Detector models: concentration -> measured signal.

The RTD solvers produce a tracer **concentration** c(t) (mol/L).  The ÄKTA does
not measure concentration directly; it measures two transducer signals that are
*linear functionals* of the composition:

* UV absorbance (mAU) via **Beer's law**      (paper Eq. 11)
* conductivity (mS/cm) via **Kohlrausch's law** (paper Eq. 12)

    A_UV(t)  = baseline_UV  + sum_i  eps_i  * c_i(t)          [mAU]
    kappa(t) = baseline_Cond+ sum_i  Lambda_i * c_i(t)        [mS/cm]

For a single tracer both signals are proportional to the same c(t) -- so their
*shapes* coincide -- but they sit on different magnitudes and baselines, which
is why they must be plotted on separate axes.  For a multi-component system
(e.g. tris-acetate + NaNO3) the per-species coefficients differ, so UV and
conductivity then also differ in *shape*.

Coefficient provenance
----------------------
* ``COND_NANO3`` (mS/cm per mol/L): from the limiting molar conductivity of
  NaNO3, Lambda_m ~= 121.5 S.cm^2/mol.  With c in mol/L,
      kappa[S/cm]  = Lambda_m[S.cm^2/mol] * c[mol/L] / 1000
      kappa[mS/cm] = 121.5 * c[mol/L]
  i.e. 0.05 M NaNO3 -> ~6.1 mS/cm, consistent with the measured baseline.
* ``UV_NANO3`` (mAU per mol/L): nitrate absorbs weakly at 280 nm.  The exact
  molar absorptivity at this wavelength is uncertain, so this constant is an
  *illustrative calibration* chosen to give paper-like UV magnitudes (tens to a
  few hundred mAU).  It scales magnitude only, not shape, and must be
  re-calibrated (or fit, see the inverse-calibration plan) for quantitative UV.

All coefficients are exposed so they can be overridden or fit.
"""

from __future__ import annotations

import numpy as np

# --- default single-tracer (NaNO3) coefficients ---------------------------
COND_NANO3 = 121.5     # mS/cm per (mol/L)   -- from limiting molar conductivity
UV_NANO3 = 430.0       # mAU  per (mol/L)    -- illustrative calibration (see above)


def _combine(concentrations, coeffs):
    """
    Sum_i coeff_i * c_i.

    ``concentrations`` may be a single array (uses ``coeffs`` as a scalar) or a
    dict {species: array} (uses ``coeffs`` as a matching dict).
    """
    if isinstance(concentrations, dict):
        total = None
        for name, c in concentrations.items():
            k = coeffs[name] if isinstance(coeffs, dict) else coeffs
            term = k * np.asarray(c, float)
            total = term if total is None else total + term
        return total
    k = coeffs if np.isscalar(coeffs) else float(coeffs)
    return k * np.asarray(concentrations, float)


def beer_uv(concentrations, eps=UV_NANO3, baseline=0.0):
    """
    UV absorbance (mAU) via Beer's law.

    concentrations : ndarray (single tracer) or {species: ndarray}
    eps            : scalar mAU per mol/L, or {species: value}
    baseline       : baseline absorbance (mAU)
    """
    return baseline + _combine(concentrations, eps)


def kohlrausch_cond(concentrations, Lambda=COND_NANO3, baseline=0.0):
    """
    Conductivity (mS/cm) via Kohlrausch's law.

    concentrations : ndarray (single tracer) or {species: ndarray}
    Lambda         : scalar mS/cm per mol/L, or {species: value}
    baseline       : buffer background conductivity (mS/cm)
    """
    return baseline + _combine(concentrations, Lambda)
