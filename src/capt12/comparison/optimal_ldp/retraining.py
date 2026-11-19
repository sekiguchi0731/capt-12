"""Fit fixed-channel CTR predictors and compare simultaneous population limits."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

import numpy as np
from scipy.special import xlogy

from capt12.comparison.optimal_ldp.numeric import (
    DOWN,
    ONE,
    UP,
    ZERO,
    decimal_ratio,
    entropy_mass,
    float_bound,
    log_bound,
    multinomial_tv_radius,
    sqrt_upper,
)
from capt12.comparison.optimal_ldp.utility import (
    Joint,
    RandomizedResponse,
    _discount,
    _spr,
    bayes_loss_bounds,
    grid_bounds,
    joint_probabilities,
    solve_spr,
)


def _by_input(counts: np.ndarray, assignment: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros((size, 2), dtype=np.int64)
    np.add.at(result, assignment, counts)
    return result


@dataclass(frozen=True)
class FixedBlocks:
    assignment: np.ndarray
    rows: tuple[tuple[Fraction, ...], ...]

    @classmethod
    def from_arrays(cls, assignment: np.ndarray, channel: np.ndarray) -> FixedBlocks:
        if (
            channel.ndim != 2
            or channel.shape[0] != channel.shape[1]
            or not len(channel)
            or assignment.ndim != 1
            or not len(assignment)
            or not np.issubdtype(assignment.dtype, np.integer)
            or (assignment < 0).any()
            or (assignment >= len(channel)).any()
            or not np.isfinite(channel).all()
        ):
            raise ValueError("expected a square block channel and valid original-token block IDs")
        rows = tuple(tuple(Fraction(float(x)) for x in row) for row in channel)
        if any(min(row) < 0 or abs(sum(row) - 1) > Fraction(2e-15) for row in rows):
            raise ValueError("fixed channel must be stochastic within serialization tolerance")
        # Interpret the serialized weights as categorical probabilities.
        return cls(assignment, tuple(tuple(x / sum(row) for x in row) for row in rows))

    def output_counts(self, counts: np.ndarray) -> np.ndarray:
        return np.asarray(self.rows, dtype=float).T @ _by_input(
            counts, self.assignment, len(self.rows)
        )

    def exact_output_joint(self, counts: np.ndarray) -> Joint:
        bins = joint_probabilities(_by_input(counts, self.assignment, len(self.rows)))
        return tuple(
            tuple(sum(p[y] * w for p, w in zip(bins, column, strict=True)) for y in (0, 1))
            for column in zip(*self.rows, strict=True)
        )


Channel = FixedBlocks | RandomizedResponse


def _validate_counts(counts: np.ndarray, k: int) -> None:
    if (
        counts.shape != (k, 2)
        or not np.issubdtype(counts.dtype, np.integer)
        or (counts < 0).any()
        or not counts.sum()
    ):
        raise ValueError("expected nonempty K-by-2 tables of nonnegative integer counts")


def _expected_loss(output_counts: np.ndarray, ctr: np.ndarray) -> float:
    return float(
        -np.sum(xlogy(output_counts[:, 0], 1 - ctr) + xlogy(output_counts[:, 1], ctr))
        / output_counts.sum()
    )


def _clip(eta: float) -> tuple[float, float]:
    # Ensure the represented upper endpoint is at most the real 1-eta.
    hi = float_bound(DOWN.subtract(ONE, Decimal.from_float(eta)), DOWN)
    return eta, hi


@dataclass(frozen=True)
class Predictor:
    channel: Channel
    ctr: np.ndarray
    pseudocount: float
    validation_log_loss: float

    def loss(self, counts: np.ndarray) -> float:
        return _expected_loss(self.channel.output_counts(counts), self.ctr)


def fit_ctr(
    channel: Channel,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    pseudocounts: Sequence[float],
    eta: float,
) -> Predictor:
    if (
        not 0 < eta < 0.5
        or not pseudocounts
        or any(not math.isfinite(a) or a <= 0 for a in pseudocounts)
    ):
        raise ValueError("positive pseudocounts and eta in (0, 0.5) are required")
    train_out = channel.output_counts(train)
    val_out = channel.output_counts(validation)
    prior = train[:, 1].sum() / train.sum()
    lo, hi = _clip(eta)
    candidates = []
    for weight in pseudocounts:
        # Written as a convex combination to avoid numerator overflow.
        share = weight / (train_out.sum(axis=1) + weight)
        observed = np.divide(
            train_out[:, 1],
            train_out.sum(axis=1),
            out=np.full(len(train_out), prior),
            where=train_out.sum(axis=1) > 0,
        )
        ctr = np.clip((1 - share) * observed + share * prior, lo, hi)
        candidates.append(Predictor(channel, ctr, weight, _expected_loss(val_out, ctr)))
    return min(candidates, key=lambda item: item.validation_log_loss)


def _mean_loss_upper(model: Predictor, counts: np.ndarray) -> Decimal:
    channel = model.channel
    if isinstance(channel, FixedBlocks):
        joint = [
            tuple(decimal_ratio(x, UP) for x in row) for row in channel.exact_output_joint(counts)
        ]
    else:
        low_t, high_t = _discount(channel.epsilon)
        denominator = DOWN.add(ONE, DOWN.multiply(Decimal(channel.outputs - 1), low_t))
        bins = joint_probabilities(_by_input(counts, channel.assignment, channel.outputs))
        totals = tuple(sum(row[y] for row in bins) for y in (0, 1))
        joint = [
            tuple(
                UP.divide(
                    UP.add(
                        decimal_ratio(mass, UP),
                        UP.multiply(high_t, decimal_ratio(totals[y] - mass, UP)),
                    ),
                    denominator,
                )
                for y, mass in enumerate(row)
            )
            for row in bins
        ]
    total = ZERO
    for (a, b), score in zip(joint, model.ctr, strict=True):
        p = Decimal.from_float(float(score))
        loss_zero = log_bound(DOWN.subtract(ONE, p), DOWN).copy_negate()
        loss_one = log_bound(p, DOWN).copy_negate()
        total = UP.add(total, UP.add(UP.multiply(a, loss_zero), UP.multiply(b, loss_one)))
    return total


def _hoeffding(n: int, alpha: float, *, two_sided: bool = False) -> Decimal:
    log_alpha = log_bound(Decimal.from_float(alpha), DOWN)
    numerator = log_alpha.copy_negate()
    if two_sided:
        numerator = UP.add(numerator, log_bound(Decimal(2), UP))
    return sqrt_upper(UP.divide(numerator, Decimal(2 * n)))


def _test_ucb(model: Predictor, counts: np.ndarray, eta: float, alpha: float) -> float:
    noise = _hoeffding(int(counts.sum()), alpha)
    clip = Decimal.from_float(eta)
    width = log_bound(UP.divide(UP.subtract(ONE, clip), clip), UP)
    return float_bound(UP.add(_mean_loss_upper(model, counts), UP.multiply(width, noise)), UP)


def _entropy_radius(tv: float) -> Decimal:
    log2 = log_bound(Decimal(2), UP)
    if tv >= 0.5:
        return log2
    r = Decimal.from_float(tv)
    return min(log2, UP.add(entropy_mass(r, UP.subtract(ONE, r), UP), UP.multiply(r, log2)))


def _two_point_lcb(validation: np.ndarray, epsilon: float, alpha: float) -> float:
    n = int(validation.sum())
    q = Fraction(int(validation[:, 1].sum()), n)
    radius = Fraction(_hoeffding(n, alpha, two_sided=True))
    low, high = max(0, q - radius), min(1, q + radius)
    # Concavity in the label prior places the minimum at an endpoint.
    return min(_spr(((1 - q, Fraction(0)), (Fraction(0), q)), epsilon).lower for q in (low, high))


def _continuity_limits(lower: float, upper: float, radius: Decimal) -> tuple[float, float]:
    return (
        float_bound(max(ZERO, DOWN.subtract(Decimal.from_float(lower), radius)), DOWN),
        float_bound(min(log_bound(Decimal(2), UP), UP.add(Decimal.from_float(upper), radius)), UP),
    )


@dataclass(frozen=True)
class FittedComparison:
    capt: Predictor
    ldp: dict[str, Predictor]
    plugin_ldp: dict[str, dict]
    validation_capt_bayes: tuple[float, float]
    validation: np.ndarray
    eta: float


def prepare_comparison(
    capt: FixedBlocks,
    budgets: dict[str, float],
    train: np.ndarray,
    validation: np.ndarray,
    *,
    grid: Sequence[float],
    exact_max_k: int = 64,
    pseudocounts: Sequence[float] = (1, 10, 100),
    eta: float = 0.01,
) -> FittedComparison:
    for counts in (train, validation):
        _validate_counts(counts, len(capt.assignment))
    capt_model = fit_ctr(capt, train, validation, pseudocounts=pseudocounts, eta=eta)
    ldp, plugin = {}, {}
    for name, budget in budgets.items():
        validation_grid = grid_bounds(validation, budget, grid)
        detail = {
            "grid_lower": validation_grid.lower,
            "grid_upper": validation_grid.upper,
            "lower": validation_grid.lower,
            "upper": validation_grid.upper,
            "spr_estimate": None,
        }
        if len(train) <= exact_max_k or budget in (0, math.inf):
            designed = solve_spr(train, budget)
            optimal = solve_spr(validation, budget)
            detail.update(lower=optimal.lower, upper=optimal.upper, spr_estimate=optimal.estimate)
        else:
            designed = grid_bounds(train, budget, grid)
        ldp[name] = fit_ctr(designed.channel, train, validation, pseudocounts=pseudocounts, eta=eta)
        plugin[name] = detail
    return FittedComparison(
        capt_model,
        ldp,
        plugin,
        bayes_loss_bounds(capt.exact_output_joint(validation)),
        validation,
        eta,
    )


def classify_gap(capt_lcb: float, capt_ucb: float, ldp_lcb: float, ldp_ucb: float) -> str:
    if capt_ucb < ldp_lcb:
        return "certified_win"
    if capt_lcb > ldp_ucb:
        return "certified_loss"
    return "inconclusive"


def evaluate_comparison(
    fitted: FittedComparison,
    test: np.ndarray,
    *,
    iid_target: bool,
    alpha_test: float,
    alpha_utility: float,
) -> dict:
    _validate_counts(test, len(fitted.validation))
    if not (0 < alpha_test < 1 and 0 < alpha_utility < 1):
        raise ValueError("utility error probabilities must be in (0,1)")
    validation = fitted.validation
    n_val = int(validation.sum())
    eta = fitted.eta
    result = {
        "capt_test_log_loss": fitted.capt.loss(test),
        "capt_pseudocount": fitted.capt.pseudocount,
        "capt_validation_log_loss": fitted.capt.validation_log_loss,
        "capt_validation_plugin_bayes_bounds": fitted.validation_capt_bayes,
        "capt_population_bayes_lcb": None,
        "capt_test_loss_ucb": None,
        "validation_joint_tv_radius": None,
        "alpha_test_family": alpha_test if iid_target else None,
        "alpha_utility_family": alpha_utility if iid_target else None,
        "comparisons": {},
    }
    if iid_target:
        alpha_per_test = math.nextafter(alpha_test / (1 + len(fitted.ldp)), 0)
        alpha_per_utility = alpha_utility / 2
        tv = multinomial_tv_radius(n_val, 2 * len(validation), alpha_per_utility)
        radius = _entropy_radius(tv)
        capt_lcb, _ = _continuity_limits(*fitted.validation_capt_bayes, radius)
        capt_ucb = _test_ucb(fitted.capt, test, eta, alpha_per_test)
        result.update(
            capt_population_bayes_lcb=capt_lcb,
            capt_test_loss_ucb=capt_ucb,
            validation_joint_tv_radius=tv,
        )
    for name, ldp in fitted.ldp.items():
        plugin = fitted.plugin_ldp[name]
        comparison = {
            "lambda": ldp.channel.epsilon,
            "ldp_test_log_loss": ldp.loss(test),
            "ldp_pseudocount": ldp.pseudocount,
            "ldp_validation_log_loss": ldp.validation_log_loss,
            "ldp_output_count": ldp.channel.outputs,
            "validation_plugin_v_z": plugin,
            "validation_plugin_gap_v_z_minus_capt_bayes": [
                float_bound(
                    DOWN.subtract(
                        Decimal.from_float(plugin["lower"]),
                        Decimal.from_float(fitted.validation_capt_bayes[1]),
                    ),
                    DOWN,
                ),
                float_bound(
                    UP.subtract(
                        Decimal.from_float(plugin["upper"]),
                        Decimal.from_float(fitted.validation_capt_bayes[0]),
                    ),
                    UP,
                ),
            ],
            "population_v_z_bounds": None,
            "population_gap_v_z_minus_capt_bayes_bounds": None,
            "population_decision": "inconclusive" if iid_target else "assumptions_not_enabled",
        }
        if iid_target:
            lcb, ucb = _continuity_limits(plugin["lower"], plugin["upper"], radius)
            two_point = _two_point_lcb(validation, ldp.channel.epsilon, alpha_per_utility)
            lcb = max(lcb, two_point)
            ucb = min(ucb, _test_ucb(ldp, test, eta, alpha_per_test))
            comparison.update(
                population_v_z_bounds=[lcb, ucb],
                population_gap_v_z_minus_capt_bayes_bounds=[
                    float_bound(
                        DOWN.subtract(Decimal.from_float(lcb), Decimal.from_float(capt_ucb)), DOWN
                    ),
                    float_bound(
                        UP.subtract(Decimal.from_float(ucb), Decimal.from_float(capt_lcb)), UP
                    ),
                ],
                ldp_two_point_lcb=two_point,
                population_decision=classify_gap(capt_lcb, capt_ucb, lcb, ucb),
            )
        result["comparisons"][name] = comparison
    return result


def fixed_reference_loss(
    capt: FixedBlocks, decoder: np.ndarray, scores: np.ndarray, test: np.ndarray
) -> float:
    """Preserve reference utility over decoded tokens, aggregating both releases."""
    decoded_counts = decoder.T @ capt.output_counts(test)
    return _expected_loss(decoded_counts, np.clip(scores, 1e-6, 1 - 1e-6))
