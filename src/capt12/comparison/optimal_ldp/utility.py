"""Bayes log loss for original-token LDP via SPR and CTR grids."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Context, Decimal
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.special import entr

from capt12.comparison.optimal_ldp.numeric import (
    DOWN,
    ONE,
    UP,
    ZERO,
    decimal_ratio,
    entropy_mass,
    exp_bound,
    float_bound,
)

Joint = tuple[tuple[Fraction, Fraction], ...]


def joint_probabilities(weights: np.ndarray) -> Joint:
    """Normalize the supplied K-by-2 masses exactly, including integer counts."""
    values = np.asarray(weights)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("expected finite K-by-2 joint weights")
    if (values < 0).any():
        raise ValueError("joint weights must be nonnegative")
    rows = tuple((Fraction(a.item()), Fraction(b.item())) for a, b in values)
    total = sum(a + b for a, b in rows)
    if not total:
        raise ValueError("joint weights have no mass")
    return tuple((a / total, b / total) for a, b in rows)


@dataclass(frozen=True)
class RandomizedResponse:
    """A partition followed by ideal RR. Output symbols can be any distinct token IDs."""

    assignment: np.ndarray
    outputs: int
    epsilon: float

    def output_counts(self, counts: np.ndarray) -> np.ndarray:
        bins = np.column_stack(
            [
                np.bincount(self.assignment, weights=label, minlength=self.outputs)
                for label in counts.T
            ]
        )
        rate = math.exp(-self.epsilon)
        denominator = 1 + (self.outputs - 1) * rate
        return (-math.expm1(-self.epsilon) * bins + rate * bins.sum(axis=0)) / denominator


@dataclass(frozen=True)
class SPRResult:
    lower: float
    upper: float
    channel: RandomizedResponse

    @property
    def estimate(self) -> float:
        return (self.lower + self.upper) / 2


def _discount(epsilon: float) -> tuple[Decimal, Decimal]:
    if epsilon < 0 or math.isnan(epsilon):
        raise ValueError("epsilon must be nonnegative")
    if math.isinf(epsilon):
        return ZERO, ZERO
    if epsilon > 100:
        return ZERO, exp_bound(Decimal(-100), UP)
    exponent = Decimal.from_float(epsilon).copy_negate()
    return exp_bound(exponent, DOWN), exp_bound(exponent, UP)


def _costs(rows: Joint, t: Decimal, ctx: Context) -> list[list[Decimal]]:
    first, second = [Fraction(0)], [Fraction(0)]
    for a, b in rows:
        first.append(first[-1] + a)
        second.append(second[-1] + b)

    def mass(prefix: list[Fraction], i: int, j: int) -> Decimal:
        inside = prefix[j] - prefix[i]
        # a_j times the common denominator is mass_inside + t * mass_outside.
        return ctx.add(
            decimal_ratio(inside, ctx),
            ctx.multiply(t, decimal_ratio(prefix[-1] - inside, ctx)),
        )

    return [
        [ZERO] * (i + 1)
        + [
            entropy_mass(mass(first, i, j), mass(second, i, j), ctx)
            for j in range(i + 1, len(rows) + 1)
        ]
        for i in range(len(rows))
    ]


def _partition_dp(
    costs: list[list[Decimal]], denominator_discount: Decimal, ctx: Context
) -> tuple[Decimal, list[list[int]], int]:
    n = len(costs)
    previous = [ZERO] * (n + 1)
    parents = [[0] * (n + 1)]
    best = Decimal("Infinity")
    best_count = 0
    denominator_ctx = UP if ctx is DOWN else DOWN
    for count in range(1, n + 1):
        current = [ZERO] * (n + 1)
        parent = [0] * (n + 1)
        for end in range(count, n + 1):
            if count == 1:
                current[end] = costs[0][end]
                continue
            current[end], parent[end] = min(
                (ctx.add(previous[start], costs[start][end]), start)
                for start in range(count - 1, end)
            )
        denominator = denominator_ctx.add(
            ONE, denominator_ctx.multiply(Decimal(count - 1), denominator_discount)
        )
        objective = ctx.divide(current[n], denominator)
        if objective < best:
            best, best_count = objective, count
        previous = current
        parents.append(parent)
    return best, parents, best_count


def _spr(joint: Joint, epsilon: float) -> SPRResult:
    low_t, high_t = _discount(epsilon)
    active = sorted(
        (i for i, (a, b) in enumerate(joint) if a + b),
        key=lambda i: joint[i][1] / sum(joint[i]),
    )
    rows = tuple(joint[i] for i in active)
    assignment = np.zeros(len(joint), dtype=int)
    if epsilon == 0 or len(rows) == 1:
        a = sum(x for x, _ in rows)
        b = sum(y for _, y in rows)
        lower, upper = (
            entropy_mass(decimal_ratio(a, ctx), decimal_ratio(b, ctx), ctx) for ctx in (DOWN, UP)
        )
        count = 1
    elif math.isinf(epsilon):
        lower, upper = (_sum_entropy(rows, ctx) for ctx in (DOWN, UP))
        assignment[active] = np.arange(len(rows))
        count = len(rows)
    else:
        lower, _, _ = _partition_dp(_costs(rows, low_t, DOWN), high_t, DOWN)
        upper, parents, count = _partition_dp(_costs(rows, high_t, UP), low_t, UP)
        end = len(rows)
        for group in range(count, 0, -1):
            start = parents[group][end]
            assignment[active[start:end]] = group - 1
            end = start
    return SPRResult(
        float_bound(lower, DOWN),
        float_bound(upper, UP),
        RandomizedResponse(assignment, count, epsilon),
    )


def _sum_entropy(joint: Joint, ctx: Context) -> Decimal:
    total = ZERO
    for a, b in joint:
        total = ctx.add(total, entropy_mass(decimal_ratio(a, ctx), decimal_ratio(b, ctx), ctx))
    return total


def solve_spr(weights: np.ndarray, epsilon: float) -> SPRResult:
    """O(K^3) time, O(K^2) memory, with directed bounds for the given law."""
    return _spr(joint_probabilities(weights), epsilon)


def staircase_lp(weights: np.ndarray, epsilon: float) -> float:
    """Independent exponential-size LP for numerical cross-checks on small K."""
    joint = np.asarray(joint_probabilities(weights), dtype=float)
    k = len(joint)
    if k > 12 or not 0 <= epsilon <= 30:
        raise ValueError("staircase cross-check supports K <= 12 and epsilon in [0,30]")
    masks = (np.arange(2**k)[None, :] >> np.arange(k)[:, None]) & 1
    discount = math.exp(-epsilon)
    columns = np.where(masks, 1.0, discount)
    masses = joint.T @ columns
    cost = entr(masses).sum(axis=0) - entr(masses.sum(axis=0))
    result = linprog(cost, A_eq=columns, b_eq=np.ones(k), method="highs")
    if not result.success:
        raise RuntimeError(f"staircase LP failed: {result.message}")
    return float(result.fun)


@dataclass(frozen=True)
class GridResult:
    lower: float
    upper: float
    channel: RandomizedResponse


def _grid_laws(joint: Joint, grid: Sequence[float]) -> tuple[Joint, Joint, np.ndarray]:
    cut = [Fraction(float(x)) for x in grid]
    if (
        len(cut) < 2
        or cut[0] != 0
        or cut[-1] != 1
        or any(a >= b for a, b in zip(cut[:-1], cut[1:], strict=True))
    ):
        raise ValueError("CTR grid must be strictly increasing from 0 to 1")
    bins = [[Fraction(0), Fraction(0)] for _ in cut[1:]]
    grid_mass = [Fraction(0) for _ in cut]
    assignment = np.zeros(len(joint), dtype=int)
    for i, (a, b) in enumerate(joint):
        mass = a + b
        if not mass:
            continue
        ctr = b / mass
        index = min(len(bins) - 1, bisect_right(cut, ctr) - 1)
        assignment[i] = index
        bins[index][0] += a
        bins[index][1] += b
        lo, hi = cut[index : index + 2]
        upper = (b - lo * mass) / (hi - lo)
        grid_mass[index] += mass - upper
        grid_mass[index + 1] += upper
    upgraded = tuple(
        ((1 - ctr) * mass, ctr * mass) for ctr, mass in zip(cut, grid_mass, strict=True)
    )
    return upgraded, tuple((a, b) for a, b in bins), assignment


@dataclass(frozen=True)
class CTRGrid:
    """A fixed pair of comparison laws, reusable across privacy budgets."""

    upgraded: Joint
    bins: Joint
    assignment: np.ndarray

    @classmethod
    def from_joint(cls, joint: Joint, grid: Sequence[float]) -> CTRGrid:
        return cls(*_grid_laws(joint, grid))

    def bounds(self, epsilon: float) -> GridResult:
        lower = _spr(self.upgraded, epsilon).lower
        upper = _spr(self.bins, epsilon)
        return GridResult(
            lower,
            upper.upper,
            RandomizedResponse(
                upper.channel.assignment[self.assignment], upper.channel.outputs, epsilon
            ),
        )


def grid_bounds(weights: np.ndarray, epsilon: float, grid: Sequence[float]) -> GridResult:
    """Upgraded grid provides the lower bound, binned input the upper bound."""
    return CTRGrid.from_joint(joint_probabilities(weights), grid).bounds(epsilon)


def bayes_loss_bounds(joint: Joint) -> tuple[float, float]:
    return tuple(float_bound(_sum_entropy(joint, ctx), ctx) for ctx in (DOWN, UP))
