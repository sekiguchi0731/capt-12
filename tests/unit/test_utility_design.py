from __future__ import annotations

import numpy as np

from capt12.decoders.registry import cost_medoid
from capt12.mechanisms.diagnostics import utility_informativeness
from capt12.partitions.registry import weighted_cost_kmedoids


def test_utility_gate_rejects_one_pointwise_best_column() -> None:
    cost = np.array([[2.0, 0.0], [1.0, 0.0]])
    result = utility_informativeness(cost, np.array([0.4, 0.6]))
    assert not result.informative
    assert result.information_gap == 0
    assert result.unique_argmin_count == 1
    assert result.no_privacy_max_row_tv == 0


def test_utility_gate_accepts_row_dependent_optima() -> None:
    cost = 1 - np.eye(3)
    result = utility_informativeness(cost, np.array([0.2, 0.3, 0.5]))
    assert result.informative
    assert np.isclose(result.free_distortion, 0)
    assert np.isclose(result.constant_distortion, 0.5)
    assert np.isclose(result.information_gap, 0.5)
    assert result.no_privacy_max_row_tv == 1


def test_cost_medoid_minimizes_weighted_within_block_distortion() -> None:
    assignment = np.array([0, 0, 0, 1])
    weights = np.array([0.7, 0.2, 0.1, 1.0])
    cost = np.array(
        [
            [0.0, 1.0, 4.0, 8.0],
            [3.0, 0.0, 2.0, 8.0],
            [5.0, 1.0, 0.0, 8.0],
            [8.0, 8.0, 8.0, 0.0],
        ]
    )
    decoder = cost_medoid(
        assignment,
        weights,
        token_cost=cost,
        token_weights=weights,
    )
    assert np.argmax(decoder[0]) == 1
    assert np.argmax(decoder[1]) == 3


def test_weighted_cost_kmedoids_is_deterministic_and_nonempty() -> None:
    points = np.array([0.0, 0.1, 0.2, 2.0, 2.1, 5.0])
    cost = np.abs(points[:, None] - points[None, :])
    weights = np.array([4.0, 2.0, 1.0, 3.0, 2.0, 1.0])
    first = weighted_cost_kmedoids(weights, 3, token_cost=cost, token_weights=weights)
    second = weighted_cost_kmedoids(weights, 3, token_cost=cost, token_weights=weights)
    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) == {0, 1, 2}
