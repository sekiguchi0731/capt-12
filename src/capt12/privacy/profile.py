from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass
class ProfilePrivacyDiagnostic:
    mode: str
    epsilon: float
    worst_pair: tuple[str, str] | None
    output: int | None
    synthetic_assignment_required: bool = True
    part_of_main_guarantee: bool = False


def evaluate_profile_privacy(
    channels: Mapping[str, np.ndarray],
    *,
    mode: str,
    common_input_distribution: np.ndarray | None = None,
    observational_input_distributions: Mapping[str, np.ndarray] | None = None,
    include_empty_profile: bool = False,
) -> ProfilePrivacyDiagnostic:
    """Optional S-privacy diagnostic, deliberately separate from A_S privacy."""
    selected = {
        profile: np.asarray(channel, dtype=float)
        for profile, channel in channels.items()
        if include_empty_profile or profile
    }
    if mode not in {"counterfactual", "observational"}:
        raise ValueError("profile privacy mode must be counterfactual or observational")
    outputs = {}
    for profile, channel in selected.items():
        if mode == "counterfactual":
            if common_input_distribution is None:
                raise ValueError("counterfactual mode requires a common input distribution")
            distribution = np.asarray(common_input_distribution, dtype=float)
        else:
            if observational_input_distributions is None or profile not in observational_input_distributions:
                raise ValueError("observational mode requires P(Z|S=s) for every profile")
            distribution = np.asarray(observational_input_distributions[profile], dtype=float)
        outputs[profile] = distribution @ channel
    epsilon = 0.0
    worst_pair = None
    worst_output = None
    for left, right in itertools.permutations(outputs, 2):
        for output, (numerator, denominator) in enumerate(
            zip(outputs[left], outputs[right], strict=True)
        ):
            candidate = (
                math.inf
                if denominator <= 0 < numerator
                else math.log(numerator / denominator)
                if numerator > 0 and denominator > 0
                else 0.0
            )
            if candidate > epsilon:
                epsilon = candidate
                worst_pair = (left, right)
                worst_output = output
    return ProfilePrivacyDiagnostic(mode, epsilon, worst_pair, worst_output)

