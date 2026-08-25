from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from capt12.experiments.context_epsilon_grid import (
    _load_epsilon_summaries,
    _plot_grid,
    _summary_table,
)


def _epsilon_summary(root: Path, epsilon: float) -> Path:
    path = root / f"epsilon-{epsilon:g}"
    (path / "tables").mkdir(parents=True)
    metadata = {
        "source_git_sha": "a" * 40,
        "L": 16,
        "epsilon": epsilon,
        "frozen_design_seeds": [0, 1],
        "context_utility_objective": "empirical_logloss",
        "context_representation_mode": "objective_aligned",
        "all_certificates_valid": True,
    }
    (path / "context_seed_stability_metadata.json").write_text(json.dumps(metadata))
    config = {
        "base_config": {
            "source_git_sha": "a" * 40,
            "epsilon": epsilon,
            "context_designs": ["joint_kmedoids_cost_medoid_L16"],
            "context_utility_objective": "empirical_logloss",
            "context_representation_mode": "objective_aligned",
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
                "assignment_hash": f"assignment-{seed}",
                "decoder_hash": f"decoder-{seed}",
                "L": 16,
                "epsilon": epsilon,
                "utility_objective": "empirical_logloss",
                "representation_mode": "objective_aligned",
                "all_certificates_valid": True,
                "relative_excess_capt_reduction_vs_context_ldp": 0.1 + epsilon / 100,
                "test_capt_minus_ldp_expected_randomized_log_loss": -1e-6,
                "test_capt_minus_ldp_ROC_AUC": 1e-3,
                "test_capt_minus_ldp_PR_AUC": 2e-3,
                "test_capt_minus_ldp_ECE": -1e-4,
                "mass_weighted_capt_row_tv": 0.2 + epsilon / 10,
                "max_capt_row_tv": 0.3 + epsilon / 10,
                "strict_advantage_context_mass": 0.8,
                "ldp_degraded_context_mass": 0.01,
                "conservative_max_realized_epsilon": epsilon - 1e-10,
                "conservative_max_additive_violation": -1e-10,
                "certificate_count": 33,
                "certificate_checked_constraints": 1000,
            }
        )
    pd.DataFrame(rows).to_csv(path / "tables" / "seed_results.csv", index=False)
    with zipfile.ZipFile(path / "sol_seed_stability_review_bundle.zip", "w") as archive:
        archive.writestr("summary.txt", f"epsilon={epsilon}")
    (path / "sol_seed_stability_review_bundle_manifest.json").write_text("{}\n")
    return path


def test_epsilon_grid_validates_paired_design_and_plots(tmp_path: Path) -> None:
    paths = [_epsilon_summary(tmp_path, epsilon) for epsilon in (0.5, 1.0, 2.0)]
    frame, info = _load_epsilon_summaries(paths)
    assert info["epsilon_values"] == [0.5, 1.0, 2.0]
    assert info["frozen_design_seeds"] == [0, 1]
    assert len(frame) == 6
    assert (frame["conservative_max_realized_epsilon"] <= frame["epsilon"]).all()
    assert set(_summary_table(frame)["epsilon"]) == {0.5, 1.0, 2.0}

    (tmp_path / "figures").mkdir(exist_ok=True)
    _plot_grid(frame, info, tmp_path)
    assert (tmp_path / "figures" / "context_epsilon_grid.pdf").stat().st_size > 0
    assert (tmp_path / "figures" / "context_epsilon_grid.png").stat().st_size > 0


def test_epsilon_grid_rejects_design_drift(tmp_path: Path) -> None:
    paths = [_epsilon_summary(tmp_path, epsilon) for epsilon in (0.5, 1.0)]
    frame = pd.read_csv(paths[1] / "tables" / "seed_results.csv")
    frame.loc[frame["frozen_design_seed"] == 0, "decoder_hash"] = "different"
    frame.to_csv(paths[1] / "tables" / "seed_results.csv", index=False)
    with pytest.raises(ValueError, match="encoder, partition, or decoder"):
        _load_epsilon_summaries(paths)
