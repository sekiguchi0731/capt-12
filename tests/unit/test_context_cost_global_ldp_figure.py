from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capt12.experiments.context_cost_global_ldp_figure import _load_and_join


def _write_comparison_inputs(tmp_path: Path) -> tuple[Path, Path]:
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
                    "source_git_sha": "a" * 40,
                    "encoder_sha256": f"encoder-{seed}",
                    "reference_model_sha256": f"reference-{seed}",
                    "reference_feature_schema": "categorical_token_v1",
                    "representation_token_cost_hash": f"representation-{seed}",
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
        json.dumps(
            {
                "epsilon": 1.0,
                "L": 16,
                "frozen_design_seeds": [0, 1],
                "experiment_source_git_sha": "a" * 40,
                "reference_feature_schema": "categorical_token_v1",
            }
        ),
        encoding="utf-8",
    )
    (cost_dir / "sol_context_cost_comparison_bundle.zip").write_bytes(b"cost")
    pd.DataFrame(
        {
            "frozen_design_seed": [0, 1],
            "epsilon": [1.0, 1.0],
            "encoder_sha256": ["encoder-0", "encoder-1"],
            "reference_model_sha256": ["reference-0", "reference-1"],
            "reference_feature_schema": [
                "categorical_token_v1",
                "categorical_token_v1",
            ],
            "representation_token_cost_hash": [
                "representation-0",
                "representation-1",
            ],
            "source_git_sha": ["a" * 40, "a" * 40],
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
        json.dumps(
            {
                "all_global_ldp_checks_valid": True,
                "frontier_source_git_sha": "a" * 40,
                "block_source_git_sha": "a" * 40,
                "reference_feature_schema": "categorical_token_v1",
            }
        ),
        encoding="utf-8",
    )
    (global_dir / "sol_global_token_ldp_comparison_bundle.zip").write_bytes(b"global")
    return cost_dir, global_dir


def test_global_cost_figure_uses_paired_seed_and_encoder(tmp_path: Path) -> None:
    cost_dir, global_dir = _write_comparison_inputs(tmp_path)

    joined, info = _load_and_join(cost_dir, global_dir)

    assert len(joined) == 6
    assert np.allclose(joined["global_ldp_minus_capt_logloss_micro"], 100_000)
    assert np.allclose(joined["capt_minus_global_ldp_roc_auc_milli"], 10)
    assert np.allclose(joined["capt_minus_global_ldp_pr_auc_milli"], 10)
    assert np.allclose(joined["global_ldp_minus_capt_ece_milli"], 10)
    assert info["L"] == 16


def test_global_cost_figure_rejects_source_sha_mismatch(tmp_path: Path) -> None:
    cost_dir, global_dir = _write_comparison_inputs(tmp_path)
    metadata_path = global_dir / "global_token_ldp_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["block_source_git_sha"] = "b" * 40
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="share one source Git SHA"):
        _load_and_join(cost_dir, global_dir)


def test_global_cost_figure_rejects_reference_model_mismatch(tmp_path: Path) -> None:
    cost_dir, global_dir = _write_comparison_inputs(tmp_path)
    table_path = global_dir / "tables" / "global_token_ldp_results.csv"
    frame = pd.read_csv(table_path)
    frame.loc[frame["frozen_design_seed"] == 0, "reference_model_sha256"] = "different"
    frame.to_csv(table_path, index=False)

    with pytest.raises(ValueError, match="reference hashes"):
        _load_and_join(cost_dir, global_dir)


def test_global_cost_figure_rejects_capt_row_source_mismatch(tmp_path: Path) -> None:
    cost_dir, global_dir = _write_comparison_inputs(tmp_path)
    table_path = cost_dir / "tables" / "objective_seed_results.csv"
    frame = pd.read_csv(table_path)
    frame["source_git_sha"] = "b" * 40
    frame.to_csv(table_path, index=False)

    with pytest.raises(ValueError, match="CAPT rows"):
        _load_and_join(cost_dir, global_dir)
