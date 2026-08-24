from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from capt12.experiments.context_cost_comparison import _load_inputs


def _summary(root: Path, objective: str) -> Path:
    path = root / objective
    (path / "tables").mkdir(parents=True)
    metadata = {
        "source_git_sha": "a" * 40,
        "L": 8,
        "frozen_design_seeds": [0, 1],
        "context_utility_objective": objective,
        "context_representation_mode": "objective_aligned",
        "all_certificates_valid": True,
    }
    (path / "context_seed_stability_metadata.json").write_text(json.dumps(metadata))
    config = {
        "base_config": {
            "source_git_sha": "a" * 40,
            "epsilon": 1.0,
            "context_designs": ["joint_kmedoids_cost_medoid_L8"],
            "context_representation_mode": "objective_aligned",
            "context_utility_objective": objective,
        }
    }
    (path / "summary_config.json").write_text(json.dumps(config))
    rows = []
    for seed in (0, 1):
        rows.append(
            {
                "frozen_design_seed": seed,
                "source_git_sha": "a" * 40,
                "encoder_sha256": f"encoder-{seed}",
                "L": 8,
                "epsilon": 1.0,
                "utility_objective": objective,
                "representation_mode": "objective_aligned",
                "hybrid_empirical_weight": 0.5,
                "all_certificates_valid": True,
                "relative_excess_capt_reduction_vs_context_ldp": 0.1,
                "test_capt_minus_ldp_expected_randomized_log_loss": -1e-6,
                "test_capt_minus_ldp_ROC_AUC": 1e-3,
                "test_capt_minus_ldp_PR_AUC": 2e-3,
                "test_capt_minus_ldp_ECE": -1e-4,
            }
        )
    pd.DataFrame(rows).to_csv(path / "tables" / "seed_results.csv", index=False)
    with zipfile.ZipFile(path / "sol_seed_stability_review_bundle.zip", "w") as archive:
        archive.writestr("summary.txt", objective)
    return path


def test_cost_comparison_requires_matched_objective_aligned_inputs(tmp_path: Path) -> None:
    paths = [
        _summary(tmp_path, objective)
        for objective in ("teacher_kl", "empirical_logloss", "hybrid_logloss_kl")
    ]
    frame, info = _load_inputs(paths)
    assert len(frame) == 6
    assert info["L"] == 8
    assert info["frozen_design_seeds"] == [0, 1]
    assert set(frame["utility_objective"]) == {
        "teacher_kl",
        "empirical_logloss",
        "hybrid_logloss_kl",
    }
