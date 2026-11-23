from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb, exp, log

import numpy as np
import pytest
from scipy.special import xlogy
from scipy.stats import binom

from capt12.comparison.optimal_ldp.population import (
    binary_loss_cells,
    binary_oracle_correction,
    binomial_upper_tail_bound,
    check_binary_cp_interval,
    frozen_capt_ucb,
    ldp_population_lower,
)
from capt12.comparison.optimal_ldp.utility import grid_bounds, solve_spr


def test_frozen_moments_integrate_channel_and_include_unseen_support():
    counts = np.array([[5, 10], [0, 0]])
    cells = binary_loss_cells(counts, np.array([False, True]), [0.25, 1.0], [0.5, 0.125])
    with localcontext() as ctx:
        ctx.prec = 80
        values = [
            -(3 * Decimal(".5").ln() + p.ln()) / 4 for p in (Decimal(".875"), Decimal(".125"))
        ]
        mean = (5 * values[0] + 10 * values[1]) / 15
        s2 = (5 * (values[0] - mean) ** 2 + 10 * (values[1] - mean) ** 2) / 14
        bound = frozen_capt_ucb(cells, Fraction(1, 20))
        assert bound.mean.lower <= mean <= bound.mean.upper
        assert bound.variance_upper >= s2
        assert bound.support.upper >= -Decimal(".125").ln()
        assert bound.mean.upper <= bound.upper <= bound.support.upper
    assert [c.count for c in cells] == [5, 10, 0, 0]


def test_inadmissible_confidence_and_scores():
    with pytest.raises(ValueError):
        binary_loss_cells(
            np.ones((2, 2), dtype=int), np.array([False, True]), [0.2, 0.8], [0.0, 1.0]
        )
    with pytest.raises(ValueError):
        frozen_capt_ucb([], Fraction(1, 2))
    for n, delta in [(0, Fraction(1, 10)), (10, Fraction(1)), (10, Fraction(0))]:
        with pytest.raises(ValueError):
            binary_oracle_correction(n, delta)
    assert ldp_population_lower(0.1, 5, Fraction(1, 20))[0] == 0


def test_suprisal_mgf_bound_including_rare_and_impossible_labels():
    # This mixture permits arbitrary posteriors across (public context, output).
    post = np.array([0.0, 1e-100, 0.007, 0.1, 0.5, 0.97, 1.0])
    p = np.column_stack((post, 1 - post))
    mean = -xlogy(p, p).sum() / len(post)
    for t in (0.001, 0.005, 0.01):
        assert log(np.sum(p ** (1 - t)) / len(post)) - t * mean < 0.64 * t * t / 2
    assert 2 * max(0.56**2, 4 * exp(-2 - 0.99 * 0.56) / 0.99**2) < 0.64


def test_empirical_infimum_is_below_risk_of_unobserved_population_oracle():
    # Opposite population token/label association in the two public contexts.
    P = [np.array([[70, 3], [2, 25]]), np.array([[10, 40], [49, 1]])]
    sample = [np.array([[15, 1], [0, 0]]), np.array([[0, 2], [1, 2]])]
    total = sum(s.sum() for s in sample)
    grid, oracle_risk = 0.0, 0.0
    for pop, obs in zip(P, sample, strict=True):
        opt = solve_spr(pop, 1.0)
        channel = opt.channel
        # Recover the probabilities of each output from the SPR release.
        m = channel.outputs
        q = np.array(
            [[exp(1.0) if j == g else 1 for j in range(m)] for g in channel.assignment]
        ) / (exp(1.0) + m - 1)
        joint = q.T @ pop
        decoder = joint / joint.sum(axis=1, keepdims=True)
        oracle_risk -= ((q.T @ obs) * np.log(decoder)).sum() / total
        grid += obs.sum() / total * grid_bounds(obs, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]).lower
    assert grid < oracle_risk


def test_binomial_tail_enclosures_and_cp_gate():
    for n in (1, 10, 32):
        for p in (Fraction(0), Fraction(1, 8), Fraction(1, 2), Fraction(1)):
            for k in range(n + 1):
                exact = sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))
                assert binomial_upper_tail_bound(n, k, p) >= exact
    check_binary_cp_interval(10, 0, Fraction(0), Fraction(1, 2), Fraction(1, 40))
    check_binary_cp_interval(10, 10, Fraction(1, 2), Fraction(1), Fraction(1, 40))
    with pytest.raises(ValueError, match="exact CP"):
        check_binary_cp_interval(10, 5, Fraction(1, 2), Fraction(1, 2), Fraction(1, 40))


def test_binomial_oracle_lcb_coverage_on_known_population():
    # Constant Z makes V exactly the binary entropy. Find the exact binomial rejection region.
    n, p, delta = 1_993_209, 0.16, Fraction(3, 2000)
    rad = binary_oracle_correction(n, delta)
    h = lambda x: -xlogy(x, x) - xlogy(1 - x, 1 - x)  # noqa: E731
    lo, hi = int(p * n), n // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if h(mid / n) > h(p) + rad:
            hi = mid
        else:
            lo = mid
    assert h(lo / n) <= h(p) + rad < h(hi / n)
    assert binom.sf(hi - 1, n, p) - binom.sf(n - hi, n, p) <= float(delta)
    assert rad == pytest.approx(0.00204343928)
