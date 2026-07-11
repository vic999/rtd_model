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

# --- multi-species detector coefficients (improvement #2) -----------------
# Per-species responses so a run with several chemicals (e.g. a tris-acetate
# buffer displaced by NaNO3) produces UV and conductivity traces of DIFFERENT
# shape.  Values other than NaNO3 are ILLUSTRATIVE (chosen to reproduce the
# magnitudes/signs seen in the real transition data); see docs/PARAMETERS.md
# and docs/MULTICOMPONENT.md.
UV_BUFFER = 0.0        # tris-acetate absorbs ~nothing at 280 nm
COND_BUFFER = 11.9     # mS/cm per (unit buffer) -- gives the measured ~11.9 baseline
UV_ANTIBODY = 90.0     # mAU per (g/L) IgG at 280 nm  (illustrative)
COND_ANTIBODY = 0.5    # mS/cm per (g/L)              (illustrative, small)

SPECIES_UV = {"NaNO3": UV_NANO3, "buffer": UV_BUFFER, "antibody": UV_ANTIBODY}
SPECIES_COND = {"NaNO3": COND_NANO3, "buffer": COND_BUFFER, "antibody": COND_ANTIBODY}


def _species_coeffs(names, table, kind):
    missing = [n for n in names if n not in table]
    if missing:
        raise KeyError(f"no {kind} coefficient for species {missing}; "
                       f"known: {sorted(table)}")
    return {n: table[n] for n in names}


def uv_from_species(conc_by_species, baseline=0.0):
    """UV (mAU) from a {species: concentration} dict, per-species Beer's law."""
    coeffs = _species_coeffs(conc_by_species.keys(), SPECIES_UV, "UV")
    return beer_uv(conc_by_species, eps=coeffs, baseline=baseline)


def cond_from_species(conc_by_species, baseline=0.0):
    """Conductivity (mS/cm) from a {species: concentration} dict (Kohlrausch)."""
    coeffs = _species_coeffs(conc_by_species.keys(), SPECIES_COND, "conductivity")
    return kohlrausch_cond(conc_by_species, Lambda=coeffs, baseline=baseline)


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
