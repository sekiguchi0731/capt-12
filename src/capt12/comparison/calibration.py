from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from capt12.certification.robust import VerificationResult, verify_robust_channel
from capt12.confidence.boxes import ConfidenceBox
from capt12.mechanisms.lp import validate_channel
from capt12.privacy.adjacency import AdjacentPair


@dataclass(frozen=True)
class CoverCalibration:
    """A frozen channel calibrated to a robust epsilon with a common cover."""

    channel: np.ndarray
    cover_distribution: np.ndarray
    mixing_weight: float
    target_epsilon: float
    bisection_tolerance: float
    pre_verification: VerificationResult
    post_verification: VerificationResult
    independent_verification: VerificationResult
    pre_utility: float | None
    post_utility: float | None
    max_row_tv_change: float
    constant_channel: bool

    def provenance(self) -> dict[str, object]:
        return {
            "wrapper": "common-cover robust calibration",
            "mixing_weight": self.mixing_weight,
            "target_epsilon": self.target_epsilon,
            "bisection_tolerance": self.bisection_tolerance,
            "pre_upper_epsilon": self.pre_verification.realized_epsilon,
            "post_upper_epsilon": self.post_verification.realized_epsilon,
            "independent_upper_epsilon": self.independent_verification.realized_epsilon,
            "pre_utility": self.pre_utility,
            "post_utility": self.post_utility,
            "max_row_tv_change": self.max_row_tv_change,
            "constant_channel": self.constant_channel,
            "cover_sha256": hashlib.sha256(self.cover_distribution.tobytes()).hexdigest(),
        }


def design_cover_distribution(
    output_weights: np.ndarray,
    *,
    floor: float = 1e-12,
    source_split: str = "D_design",
) -> np.ndarray:
    """Freeze a full-support cover from D_design output weights only."""
    if source_split != "D_design":
        raise ValueError("the common cover must be frozen from D_design only")
    weights = np.asarray(output_weights, dtype=float)
    if (
        weights.ndim != 1
        or len(weights) == 0
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("cover weights must be a nonempty nonnegative vector")
    if not 0 < floor < 1 / len(weights):
        raise ValueError("cover floor must be in (0, 1/K)")
    empirical = weights / weights.sum()
    return floor + (1 - floor * len(weights)) * empirical


def mix_with_common_cover(
    channel: np.ndarray, cover_distribution: np.ndarray, mixing_weight: float
) -> np.ndarray:
    matrix = np.asarray(channel, dtype=float)
    validate_channel(matrix)
    cover = np.asarray(cover_distribution, dtype=float)
    if (
        cover.shape != (matrix.shape[1],)
        or not np.isfinite(cover).all()
        or np.any(cover <= 0)
    ):
        raise ValueError("cover must have one strictly positive probability per output")
    if not np.isclose(cover.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("cover distribution must sum to one")
    if not 0 <= mixing_weight <= 1:
        raise ValueError("mixing_weight must be in [0,1]")
    mixed = (1 - mixing_weight) * matrix + mixing_weight * cover[None, :]
    validate_channel(mixed)
    return mixed


def _expected_utility(
    channel: np.ndarray,
    input_weights: np.ndarray | None,
    cost: np.ndarray | None,
) -> float | None:
    if input_weights is None and cost is None:
        return None
    if input_weights is None or cost is None:
        raise ValueError("input_weights and cost must be provided together")
    weights = np.asarray(input_weights, dtype=float)
    matrix_cost = np.asarray(cost, dtype=float)
    if weights.shape != (channel.shape[0],) or matrix_cost.shape != channel.shape:
        raise ValueError("utility weights/cost do not match the channel")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("input_weights must be nonnegative with positive mass")
    weights = weights / weights.sum()
    return float(np.sum(weights[:, None] * channel * matrix_cost))


def calibrate_common_cover(
    channel: np.ndarray,
    cover_distribution: np.ndarray,
    boxes: Mapping[str, ConfidenceBox],
    adjacency: Sequence[AdjacentPair],
    *,
    target_epsilon: float = 1.0,
    bisection_tolerance: float = 1e-8,
    verifier_tolerance: float = 1e-10,
    input_weights: np.ndarray | None = None,
    cost: np.ndarray | None = None,
) -> CoverCalibration:
    """Find the smallest common-cover mixture passing the robust verifier.

    The cover is an input-independent full-support channel.  It must already
    have been frozen from D_design; this routine never chooses it from D_cert.
    The returned endpoint is verified once during bisection and again through
    an independent verifier call before it is accepted.
    """
    matrix = np.asarray(channel, dtype=float)
    validate_channel(matrix)
    if target_epsilon < 0 or not math.isfinite(target_epsilon):
        raise ValueError("target_epsilon must be finite and nonnegative")
    if not 0 < bisection_tolerance <= 1e-8:
        raise ValueError("bisection_tolerance must be in (0, 1e-8]")

    pre = verify_robust_channel(matrix, boxes, adjacency, tolerance=verifier_tolerance)

    def accepted(weight: float) -> tuple[bool, VerificationResult, np.ndarray]:
        candidate = mix_with_common_cover(matrix, cover_distribution, weight)
        verification = verify_robust_channel(
            candidate, boxes, adjacency, tolerance=verifier_tolerance
        )
        return verification.realized_epsilon <= target_epsilon, verification, candidate

    if pre.realized_epsilon <= target_epsilon:
        mixing_weight = 0.0
        calibrated = matrix.copy()
        post = pre
    else:
        cover_ok, cover_verification, _ = accepted(1.0)
        if not cover_ok or cover_verification.realized_epsilon > max(target_epsilon, 1e-12):
            raise RuntimeError("input-independent cover did not verify at epsilon 0")
        low, high = 0.0, 1.0
        while high - low > bisection_tolerance:
            midpoint = (low + high) / 2
            midpoint_ok, _, _ = accepted(midpoint)
            if midpoint_ok:
                high = midpoint
            else:
                low = midpoint
        mixing_weight = high
        _, post, calibrated = accepted(mixing_weight)

    independent = verify_robust_channel(
        np.asarray(calibrated, dtype=float).copy(),
        dict(boxes),
        tuple(adjacency),
        tolerance=verifier_tolerance,
    )
    if independent.realized_epsilon > target_epsilon or not math.isfinite(
        independent.realized_epsilon
    ):
        raise RuntimeError("calibrated channel failed independent robust verification")

    row_tv = 0.5 * np.abs(calibrated - matrix).sum(axis=1)
    constant = bool(
        np.allclose(calibrated, calibrated[0][None, :], atol=1e-10, rtol=0)
    )
    return CoverCalibration(
        channel=calibrated,
        cover_distribution=np.asarray(cover_distribution, dtype=float).copy(),
        mixing_weight=float(mixing_weight),
        target_epsilon=float(target_epsilon),
        bisection_tolerance=float(bisection_tolerance),
        pre_verification=pre,
        post_verification=post,
        independent_verification=independent,
        pre_utility=_expected_utility(matrix, input_weights, cost),
        post_utility=_expected_utility(calibrated, input_weights, cost),
        max_row_tv_change=float(row_tv.max(initial=0.0)),
        constant_channel=constant,
    )
