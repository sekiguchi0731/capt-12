import numpy as np

from capt12.comparison.binary_capt import fit_binary_capt
from capt12.confidence.boxes import cp_box
from capt12.experiments.fixed_capt_ctr_comparison import _certified_block_beta
from capt12.privacy.adjacency import AdjacentPair


def test_simultaneous_thresholds_missing_group_and_directional_limits():
    assignment = np.repeat(np.arange(4), 2)
    design = np.array([[10, 90], [95, 5], [75, 25], [20, 80]])
    hist = {"left": [130, 100, 110, 90], "right": [180, 120, 130, 125], "missing": [0] * 4}
    edges = [
        AdjacentPair("left", "right", 0.3),
        AdjacentPair("right", "left", 1.0),
        AdjacentPair("missing", "right", 0.7),
    ]
    result = fit_binary_capt(assignment, design, hist, edges, 0.01)
    assert set(result.high_blocks) == {0, 3}
    assert result.boxes["missing"].method == "full_simplex"
    np.testing.assert_array_equal(
        result.channel.assignment, np.isin(assignment, result.high_blocks)
    )
    expected = cp_box(np.array([210, 220]), alpha=0.01, group_count=3, comparisons=3)
    np.testing.assert_array_equal(result.boxes["left"].lower, expected.lower)
    assert result.probabilities[1] > result.probabilities[0]
    for edge in edges:
        assert _certified_block_beta(result.channel, result.boxes, [edge]) <= edge.epsilon


def test_zero_budget_and_empty_design_are_deterministic():
    groups = {"a": [1, 9, 3], "b": [4, 5, 3]}
    edges = [AdjacentPair("a", "b", 0)]
    result = fit_binary_capt(np.arange(3), np.zeros((3, 2), dtype=int), groups, edges, 0.05)
    assert result.high_blocks == [1, 2]
    assert result.probabilities == (0.5, 0.5)
    assert _certified_block_beta(result.channel, result.boxes, edges) == 0
