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
from .flow import (FlowProfile, Constant, Ramp, DelayedStep, Sawtooth,
                   FromData, Piecewise, as_flow_fn, representative_flow)
from .detectors import beer_uv, kohlrausch_cond, UV_NANO3, COND_NANO3

__all__ = [
    "dpf_outlet", "cst_outlet",
    "filter_outlet", "permeate_space_outlet",
    "build_train", "make_unit", "Unit", "Filter", "PERIPHERAL", "FILTERS",
    "pulse_inlet", "step_inlet", "combined_inlet",
    "run_train", "r2_score",
    "FlowProfile", "Constant", "Ramp", "DelayedStep", "Sawtooth",
    "FromData", "Piecewise", "as_flow_fn", "representative_flow",
    "beer_uv", "kohlrausch_cond", "UV_NANO3", "COND_NANO3",
]
