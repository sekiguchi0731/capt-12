from __future__ import annotations

import numpy as np

from capt12.experiments.simplex_completion import (
    _best_input_independent_channel,
    _objective,
)


def test_best_input_independent_channel_selects_lowest_weighted_destination() -> None:
    cost = np.array(
        [
            [0.0, 3.0, 2.0],
            [4.0, 0.0, 1.0],
            [2.0, 2.0, 0.0],
        ]
    )
    weights = np.array([0.2, 0.3, 0.5])
    channel, destination = _best_input_independent_channel(cost, weights)
    expected_costs = weights @ cost
    assert destination == int(np.argmin(expected_costs))
    np.testing.assert_allclose(channel, np.tile(channel[0], (len(channel), 1)))
    assert np.isclose(_objective(channel, cost, weights), expected_costs.min())
