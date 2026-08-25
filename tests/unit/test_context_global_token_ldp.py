from __future__ import annotations

import json

import numpy as np
import pandas as pd

from capt12.experiments.context_global_token_ldp import (
    _comparison_tables,
    _json_default,
    _ldp_max_violation,
    _repair_ldp_uniform,
)


def test_json_default_normalizes_numpy_scalars() -> None:
    encoded = json.dumps(
        {"seed": np.int64(4), "epsilon": np.float64(1), "valid": np.bool_(True)},
        default=_json_default,
    )
    assert json.loads(encoded) == {"seed": 4, "epsilon": 1.0, "valid": True}


def test_uniform_repair_makes_released_channel_strictly_ldp() -> None:
    channel = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    repaired, mixing, before, after = _repair_ldp_uniform(channel, 1.0, margin=1e-12)
    assert before > 0
    assert 0 < mixing < 1
    assert after < 0
    assert _ldp_max_violation(repaired, 1.0) < 0
    assert np.allclose(repaired.sum(axis=1), 1.0, atol=1e-14, rtol=0)
    assert np.min(repaired) > 0


def test_global_comparison_uses_same_seed_baseline_for_each_block_size() -> None:
    frontier = pd.DataFrame(
        {
            "frozen_design_seed": [0, 1],
            "epsilon": [1.0, 1.0],
            "test_context_capt_expected_randomized_log_loss": [0.4, 0.5],
        }
    )
    blocks = pd.DataFrame(
        {
            "frozen_design_seed": [0, 0, 1, 1],
            "L": [8, 16, 8, 16],
            "test_context_capt_expected_randomized_log_loss": [0.45, 0.4, 0.55, 0.5],
        }
    )
    baselines = pd.DataFrame(
        {
            "frozen_design_seed": [0, 1],
            "epsilon": [1.0, 1.0],
            "expected_randomized_log_loss": [0.6, 0.7],
        }
    )
    frontier_result, block_result = _comparison_tables(frontier, blocks, baselines)
    assert np.allclose(frontier_result["global_token_ldp_gain_micro"], [200_000, 200_000])
    assert np.allclose(
        block_result.sort_values(["frozen_design_seed", "L"])[
            "global_token_ldp_gain_micro"
        ],
        [150_000, 200_000, 150_000, 200_000],
    )
