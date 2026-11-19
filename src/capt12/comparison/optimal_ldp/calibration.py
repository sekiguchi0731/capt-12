"""Uniform original-token LDP calibration over ordered attribute edges."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

import numpy as np

from capt12.comparison.optimal_ldp.numeric import (
    DOWN,
    ONE,
    UP,
    ZERO,
    decimal_ratio,
    exp_bound,
    float_bound,
    log_bound,
    multinomial_tv_radius,
)


@dataclass(frozen=True)
class TokenTVBall:
    center: tuple[Fraction, ...]
    radius: Fraction

    @classmethod
    def from_counts(
        cls, counts: Sequence[int], *, alpha: float | None, shift_tv: float = 0
    ) -> TokenTVBall:
        """alpha=None is reserved for supplied, known population weights."""
        counts = tuple(counts)
        if not counts or any(not isinstance(x, (int, np.integer)) or x < 0 for x in counts):
            raise ValueError("token histograms require nonnegative integer counts")
        if not 0 <= shift_tv <= 1:
            raise ValueError("token shift TV must be in [0,1]")
        n = sum(map(int, counts))
        if n == 0:
            return cls((Fraction(1, len(counts)),) * len(counts), Fraction(1))
        radius = multinomial_tv_radius(n, len(counts), alpha) if alpha is not None else 0
        return cls(
            tuple(Fraction(int(x), n) for x in counts),
            min(Fraction(1), Fraction(radius) + Fraction(shift_tv)),
        )


@dataclass(frozen=True)
class Calibration:
    beta: float
    delta_tv_upper: float
    j_beta_upper: float
    tv_budget: float
    uniform_budget: float
    tv_method: str
    uniform_method: str


def _safe_budget(beta: float, denominator: Decimal) -> float:
    if denominator == ZERO:
        return math.inf
    if beta == 0 or denominator == ONE:
        return beta
    # Use the exp(-beta) form at large budgets. The equal-budget endpoint
    # also stays feasible when beta is too small to resolve exp(beta)-1.
    if beta > 100:
        leakage = UP.multiply(UP.subtract(ONE, denominator), exp_bound(Decimal(-100), UP))
        factor = DOWN.divide(DOWN.subtract(ONE, leakage), denominator)
        budget = DOWN.add(Decimal.from_float(beta), log_bound(factor, DOWN))
    else:
        excess = max(ZERO, DOWN.subtract(exp_bound(Decimal.from_float(beta), DOWN), ONE))
        budget = log_bound(DOWN.add(ONE, DOWN.divide(excess, denominator)), DOWN)
    return max(beta, float_bound(budget, DOWN))


def _positive_difference(p: Sequence[Fraction], q: Sequence[Fraction], gamma: Fraction) -> Fraction:
    return sum((max(Fraction(0), x - gamma * y) for x, y in zip(p, q, strict=True)), Fraction(0))


def _event_maximum(left: TokenTVBall, right: TokenTVBall, gamma: Fraction) -> Fraction:
    # Empty and full events contribute at most zero. On every other event the
    # TV-ball extrema are attained by moving mass between it and its complement.
    masses = [(Fraction(0), Fraction(0))]
    for x, y in zip(left.center, right.center, strict=True):
        masses.extend([(a + x, b + y) for a, b in masses])
    return max(
        Fraction(0),
        max(
            (min(1, p + left.radius) - gamma * max(0, q - right.radius) for p, q in masses[1:-1]),
            default=Fraction(0),
        ),
    )


def calibrate(
    beta: float,
    balls: Mapping[str, TokenTVBall] | None,
    edges: Sequence[tuple[str, str]],
    *,
    alphabet_size: int,
    exact_limit: int = 12,
    restricted_by_block_certificate: bool = False,
) -> Calibration:
    """Use the supplied edges verbatim. A block restriction means intersection.

    With no raw balls, the common sets can be the full token preimages of the
    block sets. The universal bound 1 is then available without raw counts.
    """
    if not math.isfinite(beta) or beta < 0 or alphabet_size < 1 or not 0 <= exact_limit <= 16:
        raise ValueError("invalid beta, alphabet size, or exact subset limit")
    if not edges or alphabet_size == 1:
        return Calibration(beta, 0, 0, math.inf, math.inf, "exact_degenerate", "exact_degenerate")
    if balls is None:
        return Calibration(beta, 1, 1, beta, beta, "universal_upper_bound", "universal_upper_bound")
    if any(
        len(ball.center) != alphabet_size
        or min(ball.center) < 0
        or sum(ball.center) != 1
        or not 0 <= ball.radius <= 1
        for ball in balls.values()
    ):
        raise ValueError("invalid original-token uncertainty balls")
    if any(left not in balls or right not in balls for left, right in edges):
        raise ValueError("each ordered edge must have two uncertainty sets")

    # A lower factor gives an upper hockey-stick divergence. For large beta
    # this remains a valid (possibly loose) relaxation with bounded arithmetic.
    gamma = Fraction(exp_bound(Decimal.from_float(min(beta, 100)), DOWN))
    point = all(balls[key].radius == 0 for edge in edges for key in edge)
    enumerate_events = alphabet_size <= exact_limit
    tv, hockey = Fraction(0), Fraction(0)
    for a, b in edges:
        left, right = balls[a], balls[b]
        if point:
            edge_tv = _positive_difference(left.center, right.center, Fraction(1))
            edge_j = _positive_difference(left.center, right.center, gamma)
        elif enumerate_events:
            edge_tv = _event_maximum(left, right, Fraction(1))
            edge_j = _event_maximum(left, right, gamma)
        else:
            edge_tv = _positive_difference(left.center, right.center, Fraction(1))
            edge_tv += left.radius + right.radius
            edge_j = _positive_difference(left.center, right.center, gamma)
            edge_j += left.radius + gamma * right.radius
        tv = max(tv, min(1, edge_tv))
        hockey = max(hockey, min(1, edge_tv, edge_j))
    tv_decimal = decimal_ratio(tv, UP)
    j_decimal = decimal_ratio(hockey, UP)
    exact_method = "singleton_populations" if point else "exhaustive_ordered_events"
    tv_method = exact_method if point or enumerate_events else "tv_ball_triangle_upper_bound"
    j_method = exact_method if point or enumerate_events else "tv_ball_hockey_stick_upper_bound"
    if restricted_by_block_certificate:
        tv_method += "_outer_bound_for_intersection"
        j_method += "_outer_bound_for_intersection"
    if beta > 100:
        j_method += "_relaxed_exponential"
    return Calibration(
        beta,
        float_bound(tv_decimal, UP),
        float_bound(j_decimal, UP),
        _safe_budget(beta, tv_decimal),
        _safe_budget(beta, j_decimal),
        tv_method,
        j_method,
    )
