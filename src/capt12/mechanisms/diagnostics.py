from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class UtilityInformativeness:
    constant_distortion: float
    free_distortion: float
    information_gap: float
    row_argmins: tuple[int, ...]
    unique_argmin_count: int
    no_privacy_max_row_tv: float
    informative: bool
    failure_reasons: tuple[str, ...]
    tolerance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def utility_informativeness(
    cost: np.ndarray,
    input_weights: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> UtilityInformativeness:
    """Diagnose whether an input-dependent block channel can improve utility."""
    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("cost must be a square matrix")
    weights = np.asarray(input_weights, dtype=float)
    if weights.shape != (len(matrix),) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("input_weights must be nonnegative and have length n")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    weights = weights / weights.sum()

    row_argmins = np.argmin(matrix, axis=1)
    free_distortion = float(weights @ np.min(matrix, axis=1))
    constant_distortion = float(np.min(weights @ matrix))
    information_gap = max(0.0, constant_distortion - free_distortion)
    unique_argmins = len(np.unique(row_argmins))
    # Each no-privacy optimum is a one-hot row at its row-wise argmin.  The
    # maximum pairwise TV is therefore exactly zero when all argmins agree and
    # one otherwise; materializing every pairwise row difference would be O(n^3).
    row_tv = float(unique_argmins > 1)
    reasons: list[str] = []
    if information_gap <= tolerance:
        reasons.append("information_gap_at_or_below_tolerance")
    if unique_argmins == 1:
        reasons.append("all_rows_share_one_argmin")
    if row_tv <= tolerance:
        reasons.append("no_privacy_optimum_is_input_independent")
    return UtilityInformativeness(
        constant_distortion=constant_distortion,
        free_distortion=free_distortion,
        information_gap=information_gap,
        row_argmins=tuple(map(int, row_argmins)),
        unique_argmin_count=unique_argmins,
        no_privacy_max_row_tv=row_tv,
        informative=not reasons,
        failure_reasons=tuple(reasons),
        tolerance=float(tolerance),
    )
