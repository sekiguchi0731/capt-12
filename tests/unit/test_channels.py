from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from capt12.bounds.theorem4 import fractional_knapsack_envelope, theorem4_envelope
from capt12.certification.robust import solve_robust_block_lp, verify_robust_channel
from capt12.confidence.boxes import ConfidenceBox, cp_box, dp_aware_box, full_simplex_box
from capt12.confidence.support import support
from capt12.data.synthetic import theorem4_counterexample
from capt12.decoders.registry import build_decoder
from capt12.mechanisms.baselines import common_cover, k_ary_rr
from capt12.mechanisms.lp import (
    lift_block_channel,
    solve_block_lp,
    solve_full_lp,
    solve_ldp_block_lp,
    validate_channel,
)
from capt12.mechanisms.online import ContextualSanitizer, Sanitizer
from capt12.privacy.adjacency import AdjacentPair


def test_channel_nonnegative_and_rows_sum_to_one() -> None:
    for channel in [k_ary_rr(8, 0.5), common_cover(np.arange(1, 9))]:
        validate_channel(channel)
        assert np.min(channel) >= 0
        np.testing.assert_allclose(channel.sum(axis=1), 1)


def test_block_channel_lift() -> None:
    assignment = np.array([0, 0, 1, 1])
    decoder = build_decoder("uniform_within_block", assignment, np.ones(4))
    block = np.array([[0.8, 0.2], [0.3, 0.7]])
    lifted = lift_block_channel(block, assignment, decoder)
    np.testing.assert_allclose(lifted[0], [0.4, 0.4, 0.1, 0.1])
    np.testing.assert_allclose(lifted[3], [0.15, 0.15, 0.35, 0.35])


def test_common_decoder_block_privacy_implies_token_privacy() -> None:
    token_groups = {"g0": np.array([0.4, 0.2, 0.1, 0.3]), "g1": np.array([0.1, 0.5, 0.2, 0.2])}
    assignment = np.array([0, 0, 1, 1])
    decoder = build_decoder("design_frequency", assignment, np.array([0.2, 0.3, 0.1, 0.4]))
    block = common_cover(np.array([0.6, 0.4]))
    lifted = lift_block_channel(block, assignment, decoder)
    out0 = token_groups["g0"] @ lifted
    out1 = token_groups["g1"] @ lifted
    np.testing.assert_allclose(out0, out1)


def test_singleton_block_lp_matches_full_lp_objective_and_channel() -> None:
    rng = np.random.default_rng(4)
    k = 4
    cost = rng.random((k, k))
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    groups = {"g0": np.array([0.4, 0.3, 0.2, 0.1]), "g1": np.array([0.1, 0.2, 0.3, 0.4])}
    adjacency = [AdjacentPair("g0", "g1", 0.4), AdjacentPair("g1", "g0", 0.4)]
    block = solve_block_lp(cost, weights, groups, adjacency)
    full = solve_full_lp(cost, weights, groups, adjacency)
    assert block.solver.status == full.solver.status == "optimal"
    np.testing.assert_allclose(block.solver.objective, full.solver.objective, atol=1e-9)
    np.testing.assert_allclose(block.channel, full.channel, atol=1e-9)


def test_theorem4_full_is_below_envelope_and_strict_counterexample() -> None:
    groups, adjacency = theorem4_counterexample()
    weights = np.ones(3) / 3
    full = solve_full_lp(1 - np.eye(3), weights, groups, adjacency)
    envelope = theorem4_envelope(groups, adjacency, weights)
    u_full = 1 - float(full.solver.objective)
    assert np.isclose(u_full, 2 / 3)
    assert np.isclose(envelope.utility, 11 / 12)
    assert u_full < envelope.utility
    np.testing.assert_allclose(envelope.retention, [1, 0.75, 1])


def test_single_constraint_fractional_knapsack_matches_lp() -> None:
    envelope = np.array([0.3, 0.8, 0.1])
    weights = np.ones(3) / 3
    closed = fractional_knapsack_envelope(envelope, weights)
    direct = linprog(-weights, A_ub=envelope[None, :], b_ub=[1], bounds=(0, 1), method="highs")
    assert direct.success
    assert np.isclose(closed.utility, -direct.fun)


def test_cover_is_feasible_at_epsilon_zero() -> None:
    groups = {"g0": np.array([0.9, 0.1]), "g1": np.array([0.2, 0.8])}
    boxes = {key: ConfidenceBox(p, p, p, "point", 1.0) for key, p in groups.items()}
    adjacency = [AdjacentPair("g0", "g1", 0), AdjacentPair("g1", "g0", 0)]
    result = verify_robust_channel(common_cover(np.array([0.7, 0.3])), boxes, adjacency)
    assert result.valid
    assert result.realized_epsilon <= 1e-12


def test_independent_verification_rejects_broken_channel() -> None:
    p = np.array([0.5, 0.5])
    boxes = {"g0": ConfidenceBox(p, p, p, "point", 1), "g1": ConfidenceBox(p, p, p, "point", 1)}
    broken = np.array([[1.1, -0.1], [0.0, 1.0]])
    result = verify_robust_channel(broken, boxes, [AdjacentPair("g0", "g1", 0)])
    assert not result.valid


def test_sparse_solver_matches_dense_reference() -> None:
    cost = np.array([[0.0, 2.0], [1.0, 0.0]])
    weights = np.array([0.3, 0.7])
    groups = {"g0": np.array([0.8, 0.2]), "g1": np.array([0.2, 0.8])}
    adjacency = [AdjacentPair("g0", "g1", 0), AdjacentPair("g1", "g0", 0)]
    sparse_result = solve_full_lp(cost, weights, groups, adjacency)
    c = (weights[:, None] * cost).ravel()
    a_eq = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    a_ub = []
    for pair in adjacency:
        coefficient = groups[pair.left] - groups[pair.right]
        for output in range(2):
            row = np.zeros(4)
            row[np.arange(2) * 2 + output] = coefficient
            a_ub.append(row)
    dense = linprog(c, A_ub=np.asarray(a_ub), b_ub=np.zeros(4), A_eq=a_eq, b_eq=np.ones(2), bounds=(0, 1), method="highs")
    assert dense.success
    assert np.isclose(sparse_result.solver.objective, dense.fun)


def test_robust_cutting_plane_and_experimental_dp_guard() -> None:
    counts = {"g0": np.array([80, 20]), "g1": np.array([20, 80])}
    boxes = {
        key: cp_box(value, alpha=0.1, group_count=2, comparisons=2)
        for key, value in counts.items()
    }
    adjacency = [AdjacentPair("g0", "g1", 0), AdjacentPair("g1", "g0", 0)]
    solution, verification = solve_robust_block_lp(
        1 - np.eye(2), np.ones(2) / 2, boxes, adjacency
    )
    assert solution.solver.status == "optimal"
    assert verification.valid
    experimental = {
        "g0": dp_aware_box(np.array([80.0, 20.0]), noise_scale=1.0),
        "g1": dp_aware_box(np.array([20.0, 80.0]), noise_scale=1.0),
    }
    with np.testing.assert_raises(ValueError):
        solve_robust_block_lp(1 - np.eye(2), np.ones(2) / 2, experimental, adjacency)


def test_shared_support_bounds_match_paired_witness_objective() -> None:
    counts = {"g0": np.array([80, 20]), "g1": np.array([20, 80])}
    boxes = {
        key: cp_box(value, alpha=0.1, group_count=2, comparisons=2)
        for key, value in counts.items()
    }
    adjacency = [AdjacentPair("g0", "g1", 0.2), AdjacentPair("g1", "g0", 0.2)]
    cost = 1 - np.eye(2)
    weights = np.ones(2) / 2
    paired, paired_verification = solve_robust_block_lp(
        cost,
        weights,
        boxes,
        adjacency,
        cut_formulation="paired_witness",
    )
    shared, shared_verification = solve_robust_block_lp(
        cost,
        weights,
        boxes,
        adjacency,
        cut_formulation="shared_support_bounds",
    )
    assert paired_verification.valid and shared_verification.valid
    assert paired.solver.status == shared.solver.status == "optimal"
    np.testing.assert_allclose(shared.solver.objective, paired.solver.objective, atol=1e-10)
    np.testing.assert_allclose(shared.channel, paired.channel, atol=1e-9)


def test_shared_support_checkpoint_resumes_after_explicit_iteration_limit(tmp_path) -> None:
    counts = {"g0": np.array([80, 20]), "g1": np.array([20, 80])}
    boxes = {
        key: cp_box(value, alpha=0.1, group_count=2, comparisons=2)
        for key, value in counts.items()
    }
    adjacency = [AdjacentPair("g0", "g1", 0.2), AdjacentPair("g1", "g0", 0.2)]
    checkpoint = tmp_path / "support-cuts.npz"
    limited, limited_verification = solve_robust_block_lp(
        1 - np.eye(2),
        np.ones(2) / 2,
        boxes,
        adjacency,
        cut_formulation="shared_support_bounds",
        max_iterations=1,
        checkpoint_path=checkpoint,
        resume_checkpoint=True,
    )
    assert limited.channel is not None
    assert limited.solver.status == "cutting_plane_limit"
    assert not limited_verification.valid
    assert checkpoint.exists()

    resumed, resumed_verification = solve_robust_block_lp(
        1 - np.eye(2),
        np.ones(2) / 2,
        boxes,
        adjacency,
        cut_formulation="shared_support_bounds",
        max_iterations=10,
        checkpoint_path=checkpoint,
        resume_checkpoint=True,
    )
    assert resumed.solver.status == "optimal"
    assert resumed_verification.valid
    with np.testing.assert_raises_regex(ValueError, "checkpoint does not match"):
        solve_robust_block_lp(
            2 * (1 - np.eye(2)),
            np.ones(2) / 2,
            boxes,
            adjacency,
            cut_formulation="shared_support_bounds",
            checkpoint_path=checkpoint,
            resume_checkpoint=True,
        )


def test_tv_shift_expands_sampling_confidence_set() -> None:
    coefficients = np.array([1.0, 0.0])
    no_shift = cp_box(np.array([60, 40]), alpha=0.05, tv_radius=0.0)
    shifted = cp_box(np.array([60, 40]), alpha=0.05, tv_radius=0.01)
    base_max, _ = support(coefficients, no_shift)
    shifted_max, witness = support(coefficients, shifted)
    base_min, _ = support(coefficients, no_shift, maximize=False)
    shifted_min, _ = support(coefficients, shifted, maximize=False)
    assert shifted_max >= base_max
    assert shifted_min <= base_min
    assert np.isclose(shifted_max, min(1.0, base_max + 0.01), atol=1e-9)
    assert witness[0] > no_shift.nominal[0]


def test_full_simplex_support_is_exact_coordinate_extremum() -> None:
    coefficients = np.array([-2.0, 0.5, 3.0])
    box = full_simplex_box(np.zeros(3, dtype=int))
    maximum, max_witness = support(coefficients, box, maximize=True)
    minimum, min_witness = support(coefficients, box, maximize=False)
    assert maximum == 3.0
    assert minimum == -2.0
    np.testing.assert_array_equal(max_witness, [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(min_witness, [1.0, 0.0, 0.0])


def test_optimal_ldp_dominates_rr_and_cover_in_same_block_class() -> None:
    cost = np.array([[0.0, 2.0, 1.0], [1.0, 0.0, 3.0], [2.0, 1.0, 0.0]])
    weights = np.array([0.2, 0.3, 0.5])
    epsilon = 0.7
    solution = solve_ldp_block_lp(cost, weights, epsilon)
    assert solution.solver.status == "optimal"
    assert solution.channel is not None
    factor = np.exp(epsilon)
    assert np.max(
        solution.channel[:, None, :] - factor * solution.channel[None, :, :]
    ) <= 1e-8

    def objective(channel: np.ndarray) -> float:
        return float(np.sum(weights[:, None] * channel * cost))

    assert objective(solution.channel) <= objective(k_ary_rr(3, epsilon)) + 1e-9
    assert objective(solution.channel) <= objective(common_cover(weights)) + 1e-9


def test_lp_rescales_tiny_utility_coefficients_before_optimization() -> None:
    cost = 1e-12 * (1 - np.eye(3))
    weights = np.array([0.2, 0.3, 0.5])
    solution = solve_ldp_block_lp(cost, weights, 1.0)
    assert solution.channel is not None
    constant = common_cover(np.array([0.0, 0.0, 1.0]))

    def distortion(channel: np.ndarray) -> float:
        return float(np.sum(weights[:, None] * channel * cost))

    assert distortion(solution.channel) < distortion(constant) - 1e-14
    row_difference = np.max(
        np.abs(solution.channel[:, None, :] - solution.channel[None, :, :])
    )
    assert row_difference > 0.1
    for output in range(solution.channel.shape[1]):
        column = solution.channel[:, output]
        assert np.all(column == 0) or np.all(column > 0)


def test_full_simplex_pair_is_compiled_to_exact_ldp_constraints() -> None:
    boxes = {
        "g0": full_simplex_box(np.zeros(2, dtype=int)),
        "g1": full_simplex_box(np.zeros(2, dtype=int)),
    }
    adjacency = [AdjacentPair("g0", "g1", 0.7)]
    cost = 1 - np.eye(2)
    weights = np.array([0.4, 0.6])
    robust, verification = solve_robust_block_lp(cost, weights, boxes, adjacency)
    ldp = solve_ldp_block_lp(cost, weights, 0.7)
    assert robust.channel is not None and ldp.channel is not None
    assert verification.valid
    assert len(robust.cuts) == 1
    assert {cut["source"] for cut in robust.cuts} == {"full_simplex_ldp_seed"}
    assert robust.cuts[0]["constraint_count"] == 4
    assert robust.solver.constraint_count == 6
    np.testing.assert_allclose(robust.solver.objective, ldp.solver.objective, atol=1e-10)


def test_full_simplex_seed_keeps_smaller_non_simplex_edge() -> None:
    boxes = {
        "simplex-left": full_simplex_box(np.zeros(2, dtype=int)),
        "simplex-right": full_simplex_box(np.zeros(2, dtype=int)),
        "point-left": ConfidenceBox(
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
            "point",
            1.0,
        ),
        "point-right": ConfidenceBox(
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            "point",
            1.0,
        ),
    }
    adjacency = [
        AdjacentPair("simplex-left", "simplex-right", 1.0),
        AdjacentPair("point-left", "point-right", 0.0),
        AdjacentPair("point-right", "point-left", 0.0),
    ]
    robust, verification = solve_robust_block_lp(
        1 - np.eye(2),
        np.array([0.4, 0.6]),
        boxes,
        adjacency,
    )
    assert robust.channel is not None
    assert verification.valid
    assert any(
        cut.get("source") == "full_simplex_ldp_seed"
        and cut.get("pure_ldp_replacement") is False
        for cut in robust.cuts
    )
    np.testing.assert_allclose(robust.channel[0], robust.channel[1], atol=1e-8)


def test_online_sanitizer_memoization_and_fallback() -> None:
    assignment = np.array([0, 0, 1, 1])
    decoder = build_decoder("uniform_within_block", assignment, np.ones(4))
    sanitizer = Sanitizer(
        channels={"proxy": np.eye(2)},
        token_to_block=assignment,
        decoder=decoder,
        fallback_distribution=np.ones(4) / 4,
    )
    first = sanitizer.sanitize(0, "proxy", np.random.default_rng(3), "user-day")
    second = sanitizer.sanitize(0, "proxy", np.random.default_rng(999), "user-day")
    assert first == second
    sanitizer.authenticated = False
    fallback = sanitizer.sanitize(1, "proxy", np.random.default_rng(2), "different-user-day")
    assert 0 <= fallback < 4


def test_contextual_sanitizer_selects_channel_without_sensitive_value() -> None:
    assignment = np.array([0, 1])
    decoder = np.eye(2)
    sanitizer = ContextualSanitizer(
        channels={
            ("secret", "morning"): np.eye(2),
            ("secret", "evening"): np.array([[0.0, 1.0], [1.0, 0.0]]),
        },
        token_to_block=assignment,
        decoder=decoder,
    )
    morning = sanitizer.sanitize(
        0, "secret", "morning", np.random.default_rng(1), "u-day"
    )
    evening = sanitizer.sanitize(
        0, "secret", "evening", np.random.default_rng(1), "u-day"
    )
    assert morning == 0
    assert evening == 1
