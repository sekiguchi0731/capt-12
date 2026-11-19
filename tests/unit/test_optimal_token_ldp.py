from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from capt12.comparison.optimal_ldp.calibration import TokenTVBall, calibrate
from capt12.comparison.optimal_ldp.retraining import (
    FixedBlocks,
    evaluate_comparison,
    fit_ctr,
    prepare_comparison,
)
from capt12.comparison.optimal_ldp.utility import (
    bayes_loss_bounds,
    grid_bounds,
    solve_spr,
    staircase_lp,
)
from capt12.experiments.fixed_capt_ctr_comparison import ComparisonSettings


def point(*counts):
    return TokenTVBall.from_counts(counts, alpha=None)


def test_calibration_tightness_all_binary_staircase_events():
    # Astra's analytic example Z=U, beta=log(14/11).
    p = np.array([0.6, 0.4])
    q = p[::-1]
    beta = math.log(14 / 11)
    balls = {"p": point(3, 2), "q": point(2, 3)}
    edges = [("p", "q"), ("q", "p")]
    result = calibrate(beta, balls, edges, alphabet_size=2)
    assert result.tv_budget == pytest.approx(math.log1p(math.expm1(beta) / 0.2))
    assert result.uniform_budget == pytest.approx(math.log(4))
    assert beta < result.tv_budget < result.uniform_budget
    for budget in (0, beta, result.tv_budget, result.uniform_budget, result.uniform_budget + 0.01):
        largest = 0
        for mask in product((0, 1), repeat=2):
            column = 1 + math.expm1(budget) * np.asarray(mask)
            largest = max(
                largest,
                math.log((p @ column) / (q @ column)),
                math.log((q @ column) / (p @ column)),
            )
        assert (largest <= beta + 1e-13) == (budget <= result.uniform_budget)


def test_calibration_direction_zero_missing_and_fallback():
    balls = {"p": point(1, 1), "q": point(2, 0), "missing": point(0, 0)}
    beta = 0.8
    one_way = calibrate(beta, balls, [("q", "p")], alphabet_size=2)
    both = calibrate(beta, balls, [("q", "p"), ("p", "q")], alphabet_size=2)
    assert one_way.j_beta_upper == 0
    assert math.isinf(one_way.uniform_budget)
    assert both.j_beta_upper == 0.5
    assert calibrate(0, balls, [("p", "q")], alphabet_size=2).uniform_budget == 0
    assert math.isinf(calibrate(0, balls, [("p", "p")], alphabet_size=2).uniform_budget)
    assert math.isinf(calibrate(0, None, [], alphabet_size=2).tv_budget)
    missing = calibrate(0, balls, [("missing", "q")], alphabet_size=2)
    assert missing.j_beta_upper == 1
    assert missing.tv_budget == 0
    fallback = calibrate(beta, None, [("p", "q")], alphabet_size=2)
    assert fallback.uniform_budget == beta
    assert fallback.uniform_method == "universal_upper_bound"


def test_robust_ball_bounds_and_exact_subset_formula():
    balls = {
        "p": TokenTVBall.from_counts((70_000, 20_000, 10_000), alpha=0.005, shift_tv=0.03),
        "q": TokenTVBall.from_counts((20_000, 70_000, 10_000), alpha=0.005, shift_tv=0.02),
    }
    edges = [("p", "q"), ("q", "p")]
    exact = calibrate(0.4, balls, edges, alphabet_size=3)
    upper = calibrate(0.4, balls, edges, alphabet_size=3, exact_limit=0)
    assert exact.uniform_method == "exhaustive_ordered_events"
    assert upper.j_beta_upper >= exact.j_beta_upper
    assert upper.uniform_budget <= exact.uniform_budget
    beta = exact.beta
    # Every extremal RR event satisfies the guarantee simultaneously over the balls.
    for left, right in edges:
        for size in (1, 2):
            from itertools import combinations

            for event in combinations(range(3), size):
                p = min(1, sum(balls[left].center[i] for i in event) + balls[left].radius)
                q = max(0, sum(balls[right].center[i] for i in event) - balls[right].radius)
                assert (1 + math.expm1(exact.uniform_budget) * float(p)) <= (
                    math.exp(beta) * (1 + math.expm1(exact.uniform_budget) * float(q)) + 1e-14
                )


@pytest.mark.parametrize("epsilon", [0, 0.03, 0.7, 2.4, 5.0])
def test_spr_matches_staircase_and_grid_sandwich(epsilon):
    rng = np.random.default_rng(1618)
    for k in (2, 3, 5, 8):
        for _ in range(3):
            weights = rng.integers(0, 12, size=(k, 2))
            exact = solve_spr(weights, epsilon)
            assert exact.estimate == pytest.approx(staircase_lp(weights, epsilon), abs=2e-10)
            assert exact.upper - exact.lower < 2e-14
            grid = grid_bounds(weights, epsilon, [0, 0.1, 0.35, 0.6, 1])
            assert grid.lower <= exact.upper
            assert grid.upper >= exact.lower
            np.testing.assert_allclose(
                exact.channel.output_counts(weights).sum(axis=0), weights.sum(axis=0)
            )


def test_spr_edge_cases_and_weighted_predictor():
    weights = np.array([[0, 0], [9, 0], [0, 1]])
    assert solve_spr(weights, math.inf).upper == 0
    assert solve_spr(np.array([[10, 0], [5, 0]]), 10).upper == 0
    capt = FixedBlocks.from_arrays(np.array([0, 0, 1]), np.array([[0.8, 0.2], [0.2, 0.8]]))
    fitted = fit_ctr(capt, weights, weights, pseudocounts=[1], eta=0.01)
    output = capt.output_counts(weights)
    np.testing.assert_allclose(fitted.ctr, (0.1 + output[:, 1]) / (1 + output.sum(axis=1)))
    assert bayes_loss_bounds(capt.exact_output_joint(weights))[1] < fitted.loss(weights)


@pytest.mark.parametrize(
    ("matrix", "budget", "expected"),
    [
        (np.eye(2), 0.0, "certified_win"),
        (np.full((2, 2), 0.5), 2.0, "certified_loss"),
        (np.full((2, 2), 0.5), 0.0, "inconclusive"),
    ],
)
def test_statistical_decisions(matrix, budget, expected):
    # Three equal count tables represent the same law at distinct split roles.
    data = np.array([[1_000_000, 0], [0, 1_000_000]])
    fitted = prepare_comparison(
        FixedBlocks.from_arrays(np.arange(2), matrix),
        {"ldp": budget},
        data.copy(),
        data.copy(),
        grid=[0, 0.5, 1],
    )
    result = evaluate_comparison(
        fitted, data.copy(), iid_target=True, alpha_test=0.01, alpha_utility=0.01
    )
    comparison = result["comparisons"]["ldp"]
    assert comparison["population_decision"] == expected
    lo, hi = comparison["population_v_z_bounds"]
    assert lo <= solve_spr(data, budget).estimate <= hi
    empirical = evaluate_comparison(
        fitted, data, iid_target=False, alpha_test=0.01, alpha_utility=0.01
    )
    assert empirical["comparisons"]["ldp"]["population_v_z_bounds"] is None


def test_predictor_split_must_avoid_test_cert_and_overlap():
    splits = {"D_attack_train": [19, 20, 21], "D_cert": [13], "D_test": [25]}
    ComparisonSettings((19,), (20,)).validate(splits)
    for train, val in [((19,), (19,)), ((13,), (20,)), ((19,), (25,))]:
        with pytest.raises(ValueError):
            ComparisonSettings(train, val).validate(splits)
