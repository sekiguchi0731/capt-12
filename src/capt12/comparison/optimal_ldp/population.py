"""One-sided finite-sample bounds on frozen binary CAPT and the population LDP infimum.

The sampling assumptions and the proof for the unknown Bayes decoder are in
docs/binary_capt_population.md. All losses use natural logs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
    sqrt_upper,
)
from capt12.comparison.optimal_ldp.theory import Bounds


@dataclass(frozen=True)
class LossCell:
    count: int
    loss: Bounds


def binary_loss_cells(
    counts: np.ndarray, high: np.ndarray, release_one: Sequence[float], ctr: Sequence[float]
) -> list[LossCell]:
    """Include all four fixed support values, even for cells absent from the test."""
    counts, high = np.asarray(counts), np.asarray(high)
    if (
        high.ndim != 1
        or counts.shape != (len(high), 2)
        or not np.issubdtype(counts.dtype, np.integer)
        or (counts < 0).any()
        or high.dtype != bool
    ):
        raise ValueError("expected nonnegative K by 2 counts and a binary token partition")
    probs, scores = tuple(map(Fraction, release_one)), tuple(map(Fraction, ctr))
    if len(probs) != 2 or len(scores) != 2 or min(probs) < 0 or max(probs) > 1:
        raise ValueError("invalid binary release")
    if min(scores) <= 0 or max(scores) >= 1:
        raise ValueError("frozen CTR scores must be strictly between zero and one")
    cells = []
    for bit, q in enumerate(probs):
        hist = counts[high == bool(bit)].sum(axis=0)
        for y in (0, 1):
            endpoints = []
            for ctx in (DOWN, UP):
                total = ZERO
                for mass, p in zip((1 - q, q), scores, strict=True):
                    reciprocal = 1 / (p if y else 1 - p)
                    loss = log_bound(decimal_ratio(reciprocal, ctx), ctx)
                    total = ctx.add(total, ctx.multiply(decimal_ratio(mass, ctx), loss))
                endpoints.append(float_bound(total, ctx))
            cells.append(LossCell(int(hist[y]), Bounds(*endpoints)))
    return cells


@dataclass(frozen=True)
class CaptUpper:
    mean: Bounds
    variance_upper: float
    support: Bounds
    correction_upper: float
    upper: float


def _log_inverse(delta: Fraction, factor: int = 1) -> Decimal:
    if not 0 < delta < 1:
        raise ValueError("error probability must lie in (0,1)")
    return log_bound(decimal_ratio(factor / Fraction(delta), UP), UP)


def frozen_capt_ucb(cells: Sequence[LossCell], delta: Fraction) -> CaptUpper:
    n = sum(c.count for c in cells)
    if n < 2 or any(c.count < 0 or c.loss.lower < 0 for c in cells):
        raise ValueError("empirical Bernstein requires at least two nonnegative losses")
    low = sum(c.count * Fraction(c.loss.lower) for c in cells)
    high = sum(c.count * Fraction(c.loss.upper) for c in cells)
    second = sum(c.count * Fraction(c.loss.upper) ** 2 for c in cells)
    variance = decimal_ratio((second - low**2 / n) / (n - 1), UP)
    mean = Bounds(
        float_bound(decimal_ratio(low / n, DOWN), DOWN),
        float_bound(decimal_ratio(high / n, UP), UP),
    )
    support = Bounds(min(c.loss.lower for c in cells), max(c.loss.upper for c in cells))
    width = UP.subtract(Decimal.from_float(support.upper), Decimal.from_float(support.lower))
    log = _log_inverse(delta, 2)
    rad = UP.add(
        sqrt_upper(UP.divide(UP.multiply(2, UP.multiply(variance, log)), n)),
        UP.divide(UP.multiply(7, UP.multiply(width, log)), 3 * (n - 1)),
    )
    upper = min(Decimal.from_float(support.upper), UP.add(decimal_ratio(high / n, UP), rad))
    return CaptUpper(
        mean, float_bound(variance, UP), support, float_bound(rad, UP), float_bound(upper, UP)
    )


def binary_oracle_correction(n: int, delta: Fraction) -> float:
    """Distribution-free upper deviation of an unknown binary Bayes log loss."""
    if n <= 0:
        raise ValueError("positive sample size required")
    log = _log_inverse(delta)
    v = Decimal("0.64")
    t = min(Decimal("0.01"), sqrt_upper(UP.divide(UP.multiply(2, log), UP.multiply(n, v))))
    penalty = UP.add(UP.divide(log, DOWN.multiply(n, t)), UP.divide(UP.multiply(v, t), 2))
    return float_bound(penalty, UP)


def ldp_population_lower(grid_lower: float, n: int, delta: Fraction) -> tuple[float, float]:
    if not math.isfinite(grid_lower) or not 0 <= grid_lower <= math.log(2) + 1e-15:
        raise ValueError("expected a certified empirical lower endpoint in nats")
    correction = binary_oracle_correction(n, delta)
    lower = max(ZERO, DOWN.subtract(Decimal.from_float(grid_lower), Decimal.from_float(correction)))
    return float_bound(lower, DOWN), correction


def binomial_upper_tail_bound(n: int, k: int, p: Fraction) -> Decimal:
    """Rigorous P(Bin(n,p) >= k) bound via first mass and decreasing ratios."""
    if not 0 <= k <= n or not 0 <= p <= 1:
        raise ValueError("invalid binomial tail")
    if k == 0 or p == 1:
        return ONE
    if p == 0:
        return ZERO
    ratio = (n - k) * p / ((k + 1) * (1 - p))
    if ratio >= 1:
        return ONE
    log_mass = log_bound(Decimal(math.comb(n, k)), UP)
    for count, prob in ((k, p), (n - k, 1 - p)):
        log_mass = UP.add(log_mass, UP.multiply(count, log_bound(decimal_ratio(prob, UP), UP)))
    return min(ONE, UP.divide(exp_bound(log_mass, UP), decimal_ratio(1 - ratio, DOWN)))


def check_binary_cp_interval(n: int, k: int, low: Fraction, high: Fraction, tail: Fraction):
    """Check that the selected interval contains the exact equal-tailed CP interval."""
    if n <= 0 or not 0 <= k <= n or not 0 <= low <= high <= 1 or not 0 < tail < Fraction(1, 2):
        raise ValueError("invalid binary confidence interval")
    lower_ok = low == 0 if k == 0 else binomial_upper_tail_bound(n, k, low) <= tail
    upper_ok = high == 1 if k == n else binomial_upper_tail_bound(n, n - k, 1 - high) <= tail
    if not (lower_ok and upper_ok):
        raise ValueError("saved two-cell box did not enclose the allocated exact CP interval")
