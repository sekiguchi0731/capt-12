"""Information decomposition and certified crossings on a supplied empirical law."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import cache

import numpy as np

from capt12.comparison.optimal_ldp.numeric import DOWN, ONE, UP, decimal_ratio, float_bound
from capt12.comparison.optimal_ldp.retraining import FixedBlocks
from capt12.comparison.optimal_ldp.utility import (
    CTRGrid,
    Joint,
    _discount,
    _spr,
    bayes_loss_bounds,
    joint_probabilities,
)


@dataclass(frozen=True)
class Bounds:
    lower: float
    upper: float

    def __post_init__(self):
        if not (
            math.isfinite(self.lower) and math.isfinite(self.upper) and self.lower <= self.upper
        ):
            raise ValueError("invalid finite interval")

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2

    def minus(self, other: Bounds) -> Bounds:
        return Bounds(
            float_bound(
                DOWN.subtract(Decimal.from_float(self.lower), Decimal.from_float(other.upper)), DOWN
            ),
            float_bound(
                UP.subtract(Decimal.from_float(self.upper), Decimal.from_float(other.lower)), UP
            ),
        )


def weighted_bounds(bounds: Sequence[Bounds], weights: Sequence[Fraction]) -> Bounds:
    if not weights or min(weights) <= 0 or sum(weights) != 1:
        raise ValueError("positive weights must sum to one")
    endpoints = (
        sum(w * Fraction(getattr(b, key)) for b, w in zip(bounds, weights, strict=True))
        for key in ("lower", "upper")
    )
    return Bounds(
        *(
            float_bound(decimal_ratio(x, ctx), ctx)
            for x, ctx in zip(endpoints, (DOWN, UP), strict=True)
        )
    )


def _ctr(row: tuple[Fraction, Fraction]) -> Fraction | None:
    return row[1] / sum(row) if sum(row) else None


def _conditional_mi(source: Joint, target: Joint, transitions: Iterable) -> float:
    # Direct conditional KL, independent of the entropy differences used for
    # I(Y;X). Zero-mass labels and edges contribute exactly zero.
    source_mass = [sum(row) for row in source]
    target_mass = [sum(row) for row in target]
    return math.fsum(
        float(source[i][y] * weight)
        * _log_ratio(source[i][y] * target_mass[j] / (target[j][y] * source_mass[i]))
        for i, j, weight in transitions
        for y in (0, 1)
        if weight and source[i][y]
    )


def _log_ratio(ratio: Fraction) -> float:
    return math.log(float(ratio)) if ratio < Fraction(1, 2) else math.log1p(float(ratio - 1))


@dataclass(frozen=True)
class Information:
    joint: Joint
    entropy_y: Bounds
    entropy_z: Bounds
    entropy_c: Bounds
    entropy_release: Bounds
    block_information_loss: float
    randomized_information_loss: float
    raw_informative: bool
    released_informative: bool
    preserves_raw_information: bool

    @property
    def raw_information(self) -> Bounds:
        return self.entropy_y.minus(self.entropy_z)

    @property
    def released_information(self) -> Bounds:
        return self.entropy_y.minus(self.entropy_release)

    def metrics(self) -> dict:
        h, hz, hc, hr = (
            x.midpoint
            for x in (self.entropy_y, self.entropy_z, self.entropy_c, self.entropy_release)
        )
        mi_z = self.raw_information.midpoint
        return {
            "h_y": h,
            "h_y_given_z": hz,
            "h_y_given_c": hc,
            "h_y_given_release": hr,
            "i_y_z": max(0.0, mi_z),
            "i_y_c": max(0.0, h - hc),
            "i_y_release": max(0.0, self.released_information.midpoint),
            "i_y_z_given_c": self.block_information_loss,
            "i_y_c_given_release": self.randomized_information_loss,
            "information_retention": (
                min(1.0, max(0.0, self.released_information.midpoint / mi_z))
                if mi_z > 0 and self.raw_informative
                else None
            ),
            "d_capt": hr,
            "d_capt_lower": self.entropy_release.lower,
            "d_capt_upper": self.entropy_release.upper,
            "decomposition_residual": math.fsum(
                (
                    hr,
                    -hz,
                    -self.block_information_loss,
                    -self.randomized_information_loss,
                )
            ),
        }


def information_decomposition(counts: np.ndarray, capt: FixedBlocks) -> Information:
    if counts.shape != (len(capt.assignment), 2) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("expected original-token by binary-label integer counts")
    original = joint_probabilities(counts)
    block = capt.exact_input_joint(counts)
    released = capt.release_joint(block)
    marginal = (tuple(map(sum, zip(*original, strict=True))),)
    scores = [_ctr(row) for row in original]
    block_scores = [_ctr(row) for row in block]
    release_scores = [_ctr(row) for row in released]
    release_edges = [
        (c, o, weight)
        for c, row in enumerate(capt.rows)
        for o, weight in enumerate(row)
        if weight and block_scores[c] is not None
    ]
    information = Information(
        original,
        *(Bounds(*bayes_loss_bounds(law)) for law in (marginal, original, block, released)),
        _conditional_mi(original, block, ((z, c, 1) for z, c in enumerate(capt.assignment))),
        _conditional_mi(block, released, release_edges),
        any(score is not None and score != _ctr(marginal[0]) for score in scores),
        any(score is not None and score != _ctr(marginal[0]) for score in release_scores),
        (
            all(
                score is None or score == block_scores[c]
                for score, c in zip(scores, capt.assignment, strict=True)
            )
            and all(block_scores[c] == release_scores[o] for c, o, _ in release_edges)
        ),
    )
    if abs(information.metrics()["decomposition_residual"]) > 1e-12:
        raise ArithmeticError("CAPT information decomposition did not reconcile")
    return information


def empirical_ldp_oracle(
    info: Information,
    grid: Sequence[float],
    exact_max_k: int,
) -> Callable[[float], Bounds]:
    prepared = CTRGrid.from_joint(info.joint, grid) if len(info.joint) > exact_max_k else None

    @cache
    def oracle(epsilon: float) -> Bounds:
        if epsilon == 0:
            return info.entropy_y
        if epsilon == math.inf:
            return info.entropy_z
        result = prepared.bounds(epsilon) if prepared is not None else _spr(info.joint, epsilon)
        return Bounds(
            max(result.lower, info.entropy_z.lower), min(result.upper, info.entropy_y.upper)
        )

    return oracle


def empirical_decision(capt: Bounds, ldp: Bounds, *, privacy_eligible: bool) -> str:
    if privacy_eligible and capt.upper < ldp.lower:
        return "empirical_plugin_win"
    if capt.lower > ldp.upper:
        return "empirical_plugin_loss"
    return "empirical_plugin_inconclusive"


def sdpi_retention_check(info: Information, epsilon: float, epsilon_cert: float) -> dict:
    discount = _discount(epsilon)[0]
    ratio = UP.divide(UP.subtract(ONE, discount), DOWN.add(ONE, discount))
    chi_upper = UP.multiply(ratio, ratio)
    threshold = UP.multiply(chi_upper, Decimal.from_float(max(0.0, info.raw_information.upper)))
    suffices = Decimal.from_float(info.released_information.lower) > threshold
    return {
        "epsilon": epsilon,
        "retention_threshold_upper": float_bound(chi_upper, UP),
        "ldp_information_upper": float_bound(threshold, UP),
        "sufficient_for_strict_utility_win": suffices,
        "status": (
            "sufficient_condition_met" if epsilon >= epsilon_cert else "privacy_not_eligible"
        )
        if suffices
        else "sufficient_condition_inconclusive",
    }


@dataclass(frozen=True)
class Crossing:
    lower: float
    upper: float | None
    status: str
    win_interval_empty: bool | None


def find_crossing(
    oracle: Callable[[float], Bounds],
    capt: Bounds,
    epsilon_cert: float,
    *,
    released_informative: bool,
    preserves_raw_information: bool,
    initial: float = 2,
    cap: float = 16,
    tolerance: float = 1e-3,
) -> Crossing:
    """Find certified inner/outer bounds without substituting a grid midpoint."""
    if not (
        math.isfinite(cap)
        and 0 < initial <= cap
        and math.isfinite(tolerance)
        and tolerance > 0
        and math.isfinite(epsilon_cert)
        and epsilon_cert >= 0
    ):
        raise ValueError("invalid crossing search range or certificate")
    if not released_informative:
        return Crossing(0.0, 0.0, "zero", True)
    if preserves_raw_information:
        return Crossing(math.inf, math.inf, "infinite", False)

    def wins(epsilon: float) -> bool:
        return oracle(epsilon).lower > capt.upper

    def catches(epsilon: float) -> bool:
        return oracle(epsilon).upper <= capt.lower

    def bisect(predicate: Callable[[float], bool], a: float, b: float) -> tuple[float, float]:
        while b - a > tolerance:
            midpoint = a + (b - a) / 2
            if midpoint == a or midpoint == b:
                break
            if predicate(midpoint):
                a = midpoint
            else:
                b = midpoint
        return a, b

    high = min(cap, max(initial, epsilon_cert))
    while high < cap and not catches(high):
        high = min(cap, 2 * high)
    high_wins = wins(high)
    lo = high if high_wins else bisect(wins, 0.0, high)[0]
    if catches(high):
        # Search the distinct predicate for a witness that LDP catches CAPT.
        upper = bisect(lambda epsilon: not catches(epsilon), 0.0, high)[1]
        status = "finite_bracketed"
    else:
        upper = None
        status = "finite_above_search_cap" if high_wins else "finite_unresolved_at_search_cap"
    if upper is not None and upper <= epsilon_cert:
        empty = True
    elif lo > epsilon_cert or wins(epsilon_cert):
        empty = False
    else:
        empty = True if catches(epsilon_cert) else None
    return Crossing(lo, upper, status, empty)
