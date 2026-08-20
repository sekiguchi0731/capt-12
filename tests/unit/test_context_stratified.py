from __future__ import annotations

import numpy as np

from capt12.experiments.context_stratified import (
    ContextObjective,
    aggregate_context_objective,
)
from capt12.experiments.simplex_completion import _objective


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
