"""
RTD model for continuous virus filtration (Chen et al., 2024).

Reproduction of the mechanistic residence-time-distribution model described in
Biotechnology and Bioengineering 121:1876-1888 (DOI 10.1002/bit.28696).
"""

from .units import dpf_outlet, cst_outlet
from .filter_model import filter_outlet, permeate_space_outlet
from .equipment import build_train, make_unit, Unit, Filter, PERIPHERAL, FILTERS
from .injection import pulse_inlet, step_inlet, combined_inlet
from .simulate import run_train, r2_score

__all__ = [
    "dpf_outlet", "cst_outlet",
    "filter_outlet", "permeate_space_outlet",
    "build_train", "make_unit", "Unit", "Filter", "PERIPHERAL", "FILTERS",
    "pulse_inlet", "step_inlet", "combined_inlet",
    "run_train", "r2_score",
]
