from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from capt12.experiments.context_cost_global_ldp_figure import _load_and_join


def test_global_cost_figure_uses_paired_seed_and_encoder(tmp_path: Path) -> None:
    cost_dir = tmp_path / "cost"
    global_dir = tmp_path / "global"
    (cost_dir / "tables").mkdir(parents=True)
    (global_dir / "tables").mkdir(parents=True)
    objectives = ["teacher_kl", "empirical_logloss", "hybrid_logloss_kl"]
    rows = []
    for objective in objectives:
        for seed in (0, 1):
            rows.append(
                {
                    "utility_objective": objective,
                    "frozen_design_seed": seed,
                    "epsilon": 1.0,
                    "encoder_sha256": f"encoder-{seed}",
                    "test_context_capt_expected_randomized_log_loss": 0.4 + 0.01 * seed,
                    "test_context_capt_ROC_AUC": 0.6,
                    "test_context_capt_PR_AUC": 0.5,
                    "test_context_capt_ECE": 0.02,
                }
            )
    pd.DataFrame(rows).to_csv(
        cost_dir / "tables" / "objective_seed_results.csv", index=False
    )
    (cost_dir / "context_cost_comparison_metadata.json").write_text(
        json.dumps({"epsilon": 1.0, "L": 16, "frozen_design_seeds": [0, 1]}),
        encoding="utf-8",
    )
    (cost_dir / "sol_context_cost_comparison_bundle.zip").write_bytes(b"cost")
    pd.DataFrame(
        {
            "frozen_design_seed": [0, 1],
            "epsilon": [1.0, 1.0],
            "encoder_sha256": ["encoder-0", "encoder-1"],
            "expected_randomized_log_loss": [0.5, 0.51],
            "ROC_AUC": [0.59, 0.59],
            "PR_AUC": [0.49, 0.49],
            "ECE": [0.03, 0.03],
            "realized_epsilon": [0.99, 0.99],
            "max_additive_violation": [-1e-12, -1e-12],
            "channel_sha256": ["channel-0", "channel-1"],
        }
    ).to_csv(global_dir / "tables" / "global_token_ldp_results.csv", index=False)
    (global_dir / "global_token_ldp_metadata.json").write_text(
        json.dumps({"all_global_ldp_checks_valid": True}), encoding="utf-8"
    )
    (global_dir / "sol_global_token_ldp_comparison_bundle.zip").write_bytes(b"global")

    joined, info = _load_and_join(cost_dir, global_dir)

    assert len(joined) == 6
    assert np.allclose(joined["global_ldp_minus_capt_logloss_micro"], 100_000)
    assert np.allclose(joined["capt_minus_global_ldp_roc_auc_milli"], 10)
    assert np.allclose(joined["capt_minus_global_ldp_pr_auc_milli"], 10)
    assert np.allclose(joined["global_ldp_minus_capt_ece_milli"], 10)
    assert info["L"] == 16
