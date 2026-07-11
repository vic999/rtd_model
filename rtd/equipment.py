"""
Equipment definitions (Chen et al., 2024, Table 1) and train assembly.

Each peripheral unit is a DPF or CST element; the virus filter is the
three-compartment model in ``filter_model``.  A *train* is an ordered list of
units connected in series; the outlet signal of one unit is the inlet of the
next.

Volumes are in uL, lengths & diameters in mm, exactly as tabulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .units import dpf_outlet, cst_outlet
from .filter_model import filter_outlet


# --------------------------------------------------------------------------
# Table 1 -- description of filters and peripheral equipment.
# (diameter_mm, length_mm, volume_uL, model)   model in {"DPF","CST"}
# --------------------------------------------------------------------------
PERIPHERAL = {
    "3":     dict(desc="Tubing pressure monitor to mixer", d=0.75, L=400.0, V=176.7, model="DPF"),
    "M9":    dict(desc="Mixer",                            d=None, L=None,  V=1400.0, model="CST"),
    "4":     dict(desc="Tubing mixer to injection valve",  d=0.75, L=200.0, V=88.4,  model="DPF"),
    "Loop":  dict(desc="Sample loop",                      d=0.5,  L=1324.2, V=260.0, model="DPF"),
    "5":     dict(desc="Tubing mixer to injection valve",  d=0.75, L=160.0, V=70.7,  model="DPF"),
    "C-VF":  dict(desc="Tubing column valve to filter",    d=0.75, L=280.0, V=123.7, model="DPF"),
    "VF-In": dict(desc="Inlet nozzle of filter",           d=2.5,  L=10.0,  V=49.1,  model="DPF"),
    "C":     dict(desc="Connector",                        d=None, L=None,  V=130.0, model="CST"),
    "VF-Out":dict(desc="Outlet nozzle of filter",          d=2.5,  L=18.0,  V=88.4,  model="DPF"),
    "VF-C":  dict(desc="Tubing filter to column valve",    d=0.75, L=200.0, V=88.4,  model="DPF"),
    "6":     dict(desc="Tubing column valve to UV monitor",d=0.75, L=160.0, V=70.7,  model="DPF"),
    "U9-D":  dict(desc="UV monitor",                       d=None, L=None,  V=30.0,  model="CST"),
    "7":     dict(desc="Tubing UV to conductivity monitors",d=0.75, L=170.0, V=75.1, model="CST_or_DPF"),
    "C9":    dict(desc="Conductivity monitor",             d=None, L=None,  V=22.0,  model="CST"),
}
# Note: label "7" is listed as DPF in Table 1; kept as DPF below.
PERIPHERAL["7"]["model"] = "DPF"

# Filter compartments per surface area.  V_I (DPF), V_wall (CST), V_O (permeate).
# len_I / dia handled via equivalent cylinder (len 108 mm; d from volume).
FILTERS = {
    3:   dict(V_I=250.9,  len_I=108.0, V_wall=500.0, V_O=2411.6),
    10:  dict(V_I=350.9,  len_I=108.0, V_wall=600.0, V_O=7311.6),
    100: dict(V_I=1550.9, len_I=108.0, V_wall=200.0, V_O=5711.6),
}


def _equiv_diameter_mm(volume_uL, length_mm):
    """d_bar = 2 sqrt(V/(pi L)) for the equivalent cylinder of a compartment."""
    return 2.0 * np.sqrt(volume_uL / (np.pi * length_mm))


@dataclass
class Unit:
    """A single peripheral unit (DPF or CST)."""
    label: str
    model: str
    volume_uL: float
    diameter_mm: Optional[float] = None
    length_mm: Optional[float] = None

    def propagate(self, t_grid, c_in, flow_mL_min, max_step=None, c0=0.0):
        if self.model == "CST":
            return cst_outlet(t_grid, c_in, self.volume_uL, flow_mL_min,
                              max_step=max_step, c0=c0)
        elif self.model == "DPF":
            return dpf_outlet(
                t_grid, c_in, volume_uL=self.volume_uL,
                length_mm=self.length_mm, diameter_mm=self.diameter_mm,
                flow_mL_min=flow_mL_min, max_step=max_step, c0=c0,
            )
        raise ValueError(f"Unknown model {self.model!r}")


@dataclass
class Filter:
    """
    Three-compartment virus filter of a given surface area (cm^2).

    Calibrated parameters from Chen et al. (2024): l=3, eta=0.13, alpha=1.14,
    dcmax=2.17e-7.  ``film_resistance`` defaults to False (constant through-flow
    fraction ``eps_const``): this is numerically stable and reproduces the RTD
    curve shapes.  Set ``film_resistance=True`` to activate the full,
    concentration-dependent eps(t) of Eqs. 5-9 (see note in filter_model.py).
    """
    surface_cm2: int
    l: int = 3
    eta: float = 0.13
    alpha: float = 1.14
    dcmax: float = 2.17e-7
    film_resistance: bool = False
    eps_const: float = 0.85

    def propagate(self, t_grid, c_in, flow_mL_min, max_step=None, c0=0.0):
        f = FILTERS[self.surface_cm2]
        dia_I = _equiv_diameter_mm(f["V_I"], f["len_I"])
        return filter_outlet(
            t_grid, c_in, flow_mL_min=flow_mL_min, surface_cm2=self.surface_cm2,
            V_I_uL=f["V_I"], len_I_mm=f["len_I"], dia_I_mm=dia_I,
            V_wall_uL=f["V_wall"], V_O_uL=f["V_O"],
            l=self.l, eta=self.eta, alpha=self.alpha, dcmax=self.dcmax,
            film_resistance=self.film_resistance, eps_const=self.eps_const,
            max_step=max_step, c0=c0,
        )


def make_unit(label: str) -> Unit:
    spec = PERIPHERAL[label]
    return Unit(
        label=label, model=spec["model"], volume_uL=spec["V"],
        diameter_mm=spec["d"], length_mm=spec["L"],
    )


# --------------------------------------------------------------------------
# Pre-defined equipment trains for the paper's connections (Figure 1).
# The detector chain that is common to every experiment (post-column):
#   ... -> VF-C -> 6 -> U9-D (UV) -> 7 -> C9 (cond.)
# The UV signal is read at U9-D, the conductivity signal at C9.
# --------------------------------------------------------------------------
# Units that carry the tracer before the "column valve" position.  The sample
# loop injects here in pulse experiments.
PRE_COLUMN = ["Loop", "5", "C-VF"]          # loop -> tubing -> tubing to filter
DETECTOR_CHAIN = ["VF-C", "6", "U9-D", "7", "C9"]


def build_train(connection: str, surface_cm2: Optional[int] = None,
                inject_at: Optional[str] = None, **filter_kwargs):
    """
    Assemble an ordered list of (name, propagator) for a given connection.

    connection : one of
        "bypass"     -- (a) AKTA bypass: loop + detector chain, no filter
        "connector"  -- (b) connector in place of the filter
        "filter"     -- (c) batch operation with one filter (needs surface_cm2)

    inject_at : optional unit label at which the tracer is introduced
        (improvement #12).  Units *before* this one are dropped from the tracer
        path, so the two physical injection points are modelled distinctly:
          * loop pulse  -> inject_at="Loop" (default): tracer traverses the
            260 uL sample loop and everything downstream;
          * sample-pump step -> inject_at="5": the pump injects downstream of
            the loop, so the loop's hold-up is NOT traversed.

    Returns a list of objects exposing ``.propagate(t, c_in, flow)`` and a
    parallel list of names, plus the indices of the UV (U9-D) and conductivity
    (C9) read-out units.
    """
    seq = []
    names = []

    for lbl in PRE_COLUMN:
        seq.append(make_unit(lbl)); names.append(lbl)

    if connection == "bypass":
        pass                                   # nothing in the column position
    elif connection == "connector":
        seq.append(make_unit("C")); names.append("C")
    elif connection == "filter":
        if surface_cm2 is None:
            raise ValueError("surface_cm2 required for a filter connection")
        seq.append(make_unit("VF-In")); names.append("VF-In")
        seq.append(Filter(surface_cm2=surface_cm2, **filter_kwargs))
        names.append(f"VF-{surface_cm2}cm2")
        seq.append(make_unit("VF-Out")); names.append("VF-Out")
    else:
        raise ValueError(f"Unknown connection {connection!r}")

    for lbl in DETECTOR_CHAIN:
        seq.append(make_unit(lbl)); names.append(lbl)

    # Drop units upstream of the injection point (the tracer never sees them).
    if inject_at is not None:
        if inject_at not in names:
            raise ValueError(f"inject_at={inject_at!r} not a unit in this train "
                             f"(available: {names})")
        start = names.index(inject_at)
        seq, names = seq[start:], names[start:]

    uv_idx = names.index("U9-D")
    cond_idx = names.index("C9")
    return seq, names, uv_idx, cond_idx
