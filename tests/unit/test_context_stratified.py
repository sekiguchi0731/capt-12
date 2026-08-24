from __future__ import annotations

import numpy as np

from capt12.experiments.context_stratified import (
    ContextObjective,
    _context_objectives,
    _solve_design,
    aggregate_context_objective,
)
from capt12.experiments.simplex_completion import _objective
from capt12.experiments.utility_design import UtilityDesign, _build_designs
from capt12.mechanisms.diagnostics import utility_informativeness
from capt12.privacy.adjacency import Group
from capt12.utils.progress import ProgressLogger


def test_aggregate_context_objective_preserves_decomposed_objective() -> None:
    first = ContextObjective(
        context="b0",
        design_mass=0.25,
        token_probabilities=np.array([0.1, 0.2]),
        block_cost=np.array([[0.0, 2.0], [1.0, 0.0]]),
        block_weights=np.array([0.8, 0.2]),
    )
    second = ContextObjective(
        context="b1",
        design_mass=0.75,
        token_probabilities=np.array([0.3, 0.4]),
        block_cost=np.array([[0.0, 1.0], [3.0, 0.0]]),
        block_weights=np.array([0.1, 0.9]),
    )
    channel = np.array([[0.7, 0.3], [0.2, 0.8]])
    cost, weights = aggregate_context_objective([first, second])
    decomposed = sum(
        item.design_mass * _objective(channel, item.block_cost, item.block_weights)
        for item in [first, second]
    )
    assert np.isclose(_objective(channel, cost, weights), decomposed)


def test_context_objective_switches_to_empirical_design_log_loss() -> None:
    class Reference:
        def predict(self, frame):
            return np.where(frame["__token__"].to_numpy() == 0, 0.2, 0.8)

    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    weights = np.array([0.5, 0.5])
    design = UtilityDesign(
        name="identity",
        label="Identity",
        partition_method="singleton",
        decoder_method="identity",
        assignment=np.array([0, 1]),
        decoder=np.eye(2),
        block_cost=cost,
        block_weights=weights,
        diagnostic=utility_informativeness(cost, weights),
    )
    frozen = {
        "reference": Reference(),
        "cartesian_support": {("a", "morning")},
        "assignment": np.array([0, 1]),
        "context_levels": ["morning"],
        "token_context_weights": np.array([[10.0], [10.0]]),
        "token_context_label_count": np.array([[10.0], [10.0]]),
        "token_context_label_sum": np.array([[0.0], [10.0]]),
        "frequencies": weights,
        "design_only_probability": {("a", "morning"): 1.0},
    }
    empirical = _context_objectives(
        frozen,
        design,
        {
            "context_cols": ["context"],
            "context_utility_objective": "empirical_logloss",
            "distortion_clip": 1e-6,
        },
    )[0]
    expected = np.array(
        [
            [-np.log(0.8), -np.log(0.2)],
            [-np.log(0.2), -np.log(0.8)],
        ]
    )
    np.testing.assert_allclose(empirical.block_cost, expected)
    assert empirical.empirical_label_count == 20
    assert empirical.utility_objective == "empirical_logloss"

    teacher = _context_objectives(
        frozen,
        design,
        {
            "context_cols": ["context"],
            "context_utility_objective": "teacher_kl",
            "distortion_clip": 1e-6,
        },
    )[0]
    assert np.allclose(np.diag(teacher.block_cost), 0)
    assert not np.allclose(teacher.block_cost, empirical.block_cost)


def test_objective_aligned_mode_changes_joint_partition_or_decoder() -> None:
    token_count = 32
    positions = np.arange(token_count, dtype=float)
    teacher_cost = np.abs(positions[:, None] - positions[None, :])
    permutation = np.ravel(np.column_stack((np.arange(16), np.arange(16, 32))))
    aligned_positions = np.empty(token_count, dtype=float)
    aligned_positions[permutation] = positions
    aligned_cost = np.abs(aligned_positions[:, None] - aligned_positions[None, :])
    frozen = {
        "frequencies": np.full(token_count, 1 / token_count),
        "objective_weights": np.full(token_count, 1 / token_count),
        "token_scores": np.linspace(0.01, 0.99, token_count),
        "token_cost": teacher_cost,
        "representation_token_cost": aligned_cost,
        "assignment": np.repeat(np.arange(16), 2),
        "decoder": np.repeat(np.eye(16), 2, axis=1) / 2,
    }
    fixed = _build_designs(
        frozen,
        {"context_representation_mode": "teacher_kl_fixed"},
    )
    aligned = _build_designs(
        frozen,
        {
            "context_representation_mode": "objective_aligned",
            "context_utility_objective": "empirical_logloss",
        },
    )
    fixed_joint = next(item for item in fixed if item.name == "joint_kmedoids_cost_medoid_L16")
    aligned_joint = next(item for item in aligned if item.name == "joint_kmedoids_cost_medoid_L16")
    for designs in (fixed, aligned):
        joint_dimensions = {
            len(item.block_weights)
            for item in designs
            if item.name.startswith("joint_kmedoids_cost_medoid_L")
        }
        assert joint_dimensions == {8, 16, 32}
    assert aligned_joint.representation_objective == "empirical_logloss"
    assert aligned_joint.representation_token_cost_hash != (
        fixed_joint.representation_token_cost_hash
    )
    assert not (
        np.array_equal(aligned_joint.assignment, fixed_joint.assignment)
        and np.array_equal(aligned_joint.decoder, fixed_joint.decoder)
    )


def test_context_solver_emits_context_and_shared_progress(tmp_path) -> None:
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    weights = np.array([0.5, 0.5])
    design = UtilityDesign(
        name="tiny",
        label="Tiny",
        partition_method="singleton",
        decoder_method="identity",
        assignment=np.array([0, 1]),
        decoder=np.eye(2),
        block_cost=cost,
        block_weights=weights,
        diagnostic=utility_informativeness(cost, weights),
    )
    objectives = [
        ContextObjective(context, 0.5, weights, cost, weights) for context in ("morning", "evening")
    ]
    groups = [
        Group("secret", (value,), context)
        for context in ("morning", "evening")
        for value in ("a", "b")
    ]
    counts = {
        group.key(): np.array([80, 20]) if group.values == ("a",) else np.array([20, 80])
        for group in groups
    }
    progress = ProgressLogger(tmp_path, name="tiny-context")

    cells, shared_ldp, shared_capt, verification = _solve_design(
        design,
        objectives,
        groups,
        counts,
        {
            "epsilon": 1.0,
            "alpha_cert": 0.05,
            "confidence": "cp_box",
            "missing_group_policy": "full_simplex",
            "min_group_count": 20,
            "solver_tolerance": 1e-8,
            "max_cutting_plane_iterations": 10,
            "solver_heartbeat_seconds": 1,
            "robust_cut_formulation": "shared_support_bounds",
            "resume_cutting_plane": True,
        },
        progress,
        tmp_path / "checkpoints",
    )

    assert len(cells) == 2
    assert shared_ldp.channel is not None
    assert shared_capt.channel is not None
    assert verification.valid
    assert all(np.isfinite(cell.capt_verification.realized_epsilon) for cell in cells)
    assert all(cell.capt_repair.conservative_verification.valid for cell in cells)
    assert all(cell.capt_repair.mixing_weight > 0 for cell in cells)
    log = (tmp_path / "progress.log").read_text()
    assert log.count("[context_started]") == 2
    assert "[cutting_plane_iteration_started]" in log
    assert "[support_checkpoint_written]" in log
    assert "[shared_capt_finished]" in log
    assert len(list((tmp_path / "checkpoints" / "tiny").glob("*.npz"))) == 3
