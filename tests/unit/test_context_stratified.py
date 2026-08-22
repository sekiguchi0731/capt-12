from __future__ import annotations

import numpy as np

from capt12.experiments.context_stratified import (
    ContextObjective,
    _solve_design,
    aggregate_context_objective,
)
from capt12.experiments.simplex_completion import _objective
from capt12.experiments.utility_design import UtilityDesign
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
        ContextObjective(context, 0.5, weights, cost, weights)
        for context in ("morning", "evening")
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
    log = (tmp_path / "progress.log").read_text()
    assert log.count("[context_started]") == 2
    assert "[cutting_plane_iteration_started]" in log
    assert "[support_checkpoint_written]" in log
    assert "[shared_capt_finished]" in log
    assert len(list((tmp_path / "checkpoints" / "tiny").glob("*.npz"))) == 3
