from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from capt12.experiments import context_cost_comparison as comparison_module
from capt12.experiments.context_cost_comparison import (
    _load_inputs,
    run_context_cost_stability,
)


def _summary(root: Path, objective: str, *, metadata_epsilon: bool = True) -> Path:
    path = root / objective
    (path / "tables").mkdir(parents=True)
    metadata = {
        "source_git_sha": "a" * 40,
        "L": 8,
        "frozen_design_seeds": [0, 1],
        "context_utility_objective": objective,
        "context_representation_mode": "objective_aligned",
        "reference_feature_schema": "categorical_token_v1",
        "reference_model_sha256_by_seed": {
            "0": "reference-0",
            "1": "reference-1",
        },
        "all_certificates_valid": True,
    }
    if metadata_epsilon:
        metadata["epsilon"] = 0.5
    (path / "context_seed_stability_metadata.json").write_text(json.dumps(metadata))
    config = {
        "base_config": {
            "source_git_sha": "a" * 40,
            "epsilon": 0.5,
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
                "reference_model_sha256": f"reference-{seed}",
                "reference_feature_schema": "categorical_token_v1",
                "representation_token_cost_hash": f"representation-{objective}-{seed}",
                "L": 8,
                "epsilon": 0.5,
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
    assert info["epsilon"] == 0.5
    assert info["frozen_design_seeds"] == [0, 1]
    assert set(frame["utility_objective"]) == {
        "teacher_kl",
        "empirical_logloss",
        "hybrid_logloss_kl",
    }


def test_cost_comparison_accepts_legacy_summary_epsilon_from_seed_rows(tmp_path: Path) -> None:
    paths = [
        _summary(tmp_path, objective, metadata_epsilon=False)
        for objective in ("teacher_kl", "empirical_logloss", "hybrid_logloss_kl")
    ]
    frame, info = _load_inputs(paths)
    assert info["epsilon"] == 0.5
    assert set(frame["epsilon"]) == {0.5}


def test_cost_comparison_rejects_reference_model_drift(tmp_path: Path) -> None:
    paths = [
        _summary(tmp_path, objective)
        for objective in ("teacher_kl", "empirical_logloss", "hybrid_logloss_kl")
    ]
    frame = pd.read_csv(paths[1] / "tables" / "seed_results.csv")
    frame.loc[frame["frozen_design_seed"] == 0, "reference_model_sha256"] = "different"
    frame.to_csv(paths[1] / "tables" / "seed_results.csv", index=False)
    with pytest.raises(ValueError, match="reference-model hashes"):
        _load_inputs(paths)


def test_cost_stability_runs_all_objectives_before_integrating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[int], str, float]] = []

    def fake_stability(config, seeds):
        objective = config["context_utility_objective"]
        calls.append(
            (
                objective,
                seeds,
                config["context_representation_mode"],
                config["hybrid_empirical_weight"],
            )
        )
        return tmp_path / objective

    def fake_comparison(paths, output_root):
        assert paths == [tmp_path / objective for objective in comparison_module._OBJECTIVES]
        assert output_root == tmp_path / "combined"
        return output_root / "result"

    monkeypatch.setattr(comparison_module, "run_context_seed_stability", fake_stability)
    monkeypatch.setattr(comparison_module, "run_context_cost_comparison", fake_comparison)
    result = run_context_cost_stability(
        {"context_representation_mode": "teacher_kl_fixed"},
        [0, 1],
        hybrid_empirical_weight=0.25,
        output_root=tmp_path / "combined",
    )
    assert result == tmp_path / "combined" / "result"
    assert calls == [
        (objective, [0, 1], "objective_aligned", 0.25)
        for objective in comparison_module._OBJECTIVES
    ]
