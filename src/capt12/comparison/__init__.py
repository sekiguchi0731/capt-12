"""Privacy-matched prior-art comparison primitives.

The modules in this package deliberately separate method adaptation, formal
calibration, and empirical diagnostics.  No function treats an empirical
privacy estimate as a certificate.
"""

from capt12.comparison.calibration import CoverCalibration, calibrate_common_cover
from capt12.comparison.contracts import MethodContract, default_method_contracts
from capt12.comparison.mass12 import Mass12FiniteChannel, fit_mass12_finite
from capt12.comparison.pbp import PBPOracleSolution, solve_pbp_common_nominal, solve_pbp_oracle

__all__ = [
    "CoverCalibration",
    "Mass12FiniteChannel",
    "MethodContract",
    "PBPOracleSolution",
    "calibrate_common_cover",
    "default_method_contracts",
    "fit_mass12_finite",
    "solve_pbp_common_nominal",
    "solve_pbp_oracle",
]
