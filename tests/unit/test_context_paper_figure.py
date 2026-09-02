from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from capt12.experiments.context_paper_figure import run_context_paper_figure


def _write_epsilon_bundle(path: Path, *, source_sha: str = "a" * 40) -> Path:
    rows = []
    for epsilon in (0.5, 1.0, 2.0):
        for seed in (0, 1):
            rows.append(
                {
                    "frozen_design_seed": seed,
                    "source_git_sha": source_sha,
                    "encoder_sha256": f"encoder-{seed}",
                    "reference_model_sha256": f"reference-{seed}",
                    "reference_feature_schema": "categorical_token_v1",
                    "representation_token_cost_hash": f"representation-{seed}",
                    "L": 16,
                    "epsilon": epsilon,
                    "utility_objective": "empirical_logloss",
                    "representation_mode": "objective_aligned",
                    "all_certificates_valid": True,
                    "conservative_max_realized_epsilon": epsilon - 1e-10,
                    "test_capt_minus_ldp_expected_randomized_log_loss": -(seed + 1) * 1e-6,
                }
            )
    metadata = {
        "L": 16,
        "context_utility_objective": "empirical_logloss",
        "context_representation_mode": "objective_aligned",
        "source_git_sha": source_sha,
        "reference_feature_schema": "categorical_token_v1",
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("grid/context_epsilon_grid_metadata.json", json.dumps(metadata))
        archive.writestr(
            "grid/tables/epsilon_seed_results.csv", pd.DataFrame(rows).to_csv(index=False)
        )
    return path


def _write_block_bundle(
    path: Path,
    block_count: int,
    *,
    valid: bool = True,
    source_sha: str = "a" * 40,
    row_source_sha: str | None = None,
    reference_prefix: str = "reference",
) -> Path:
    rows = []
    for objective in ("teacher_kl", "empirical_logloss", "hybrid_logloss_kl"):
        for seed in (0, 1):
            rows.append(
                {
                    "frozen_design_seed": seed,
                    "source_git_sha": row_source_sha or source_sha,
                    "encoder_sha256": f"encoder-{seed}",
                    "reference_model_sha256": f"{reference_prefix}-{seed}",
                    "reference_feature_schema": "categorical_token_v1",
                    "representation_token_cost_hash": f"representation-{seed}",
                    "L": block_count,
                    "epsilon": 1.0,
                    "utility_objective": objective,
                    "representation_mode": "objective_aligned",
                    "all_certificates_valid": valid,
                    "test_capt_minus_ldp_expected_randomized_log_loss": -(seed + 1) * 1e-6,
                }
            )
    metadata = {
        "L": block_count,
        "epsilon": 1.0,
        "experiment_source_git_sha": source_sha,
        "reference_feature_schema": "categorical_token_v1",
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "comparison/context_cost_comparison_metadata.json", json.dumps(metadata)
        )
        archive.writestr(
            "comparison/tables/objective_seed_results.csv",
            pd.DataFrame(rows).to_csv(index=False),
        )
    return path


def test_context_paper_figure_uses_certified_frontier_and_available_blocks(
    tmp_path: Path,
) -> None:
    epsilon = _write_epsilon_bundle(tmp_path / "epsilon.zip")
    blocks = [
        _write_block_bundle(tmp_path / "l8.zip", 8),
        _write_block_bundle(tmp_path / "l16.zip", 16),
    ]
    output = run_context_paper_figure(epsilon, blocks, tmp_path / "output")
    assert (output / "figures" / "context_capt_main.pdf").stat().st_size > 0
    assert (output / "figures" / "context_capt_main.png").stat().st_size > 0
    assert (output / "context_capt_main_figure_bundle.zip").stat().st_size > 0
    latex_caption = (output / "caption.tex").read_text()
    assert "Privacy--utility trade-off" in latex_caption
    assert "no seed-level confidence interval is implied" in latex_caption
    metadata = json.loads((output / "context_capt_main_figure_metadata.json").read_text())
    assert metadata["block_counts"] == [8, 16]
    assert metadata["missing_block_counts"] == [32]
    assert metadata["mass_method_included"] is False
    frontier = pd.read_csv(output / "tables" / "frontier_seed_points.csv")
    assert set(frontier["robust_certified_upper_epsilon"]) == {
        0.5 - 1e-10,
        1.0 - 1e-10,
        2.0 - 1e-10,
    }


def test_context_paper_figure_rejects_invalid_certificate(tmp_path: Path) -> None:
    epsilon = _write_epsilon_bundle(tmp_path / "epsilon.zip")
    blocks = [
        _write_block_bundle(tmp_path / "l8.zip", 8, valid=False),
        _write_block_bundle(tmp_path / "l16.zip", 16),
    ]
    with pytest.raises(ValueError, match="invalid certificate"):
        run_context_paper_figure(epsilon, blocks, tmp_path / "output")


def test_context_paper_figure_rejects_source_sha_mismatch(tmp_path: Path) -> None:
    epsilon = _write_epsilon_bundle(tmp_path / "epsilon.zip")
    blocks = [
        _write_block_bundle(tmp_path / "l8.zip", 8, source_sha="b" * 40),
        _write_block_bundle(tmp_path / "l16.zip", 16, source_sha="b" * 40),
    ]
    with pytest.raises(ValueError, match="share one source Git SHA"):
        run_context_paper_figure(epsilon, blocks, tmp_path / "output")


def test_context_paper_figure_rejects_block_row_source_mismatch(tmp_path: Path) -> None:
    epsilon = _write_epsilon_bundle(tmp_path / "epsilon.zip")
    blocks = [
        _write_block_bundle(tmp_path / "l8.zip", 8, row_source_sha="b" * 40),
        _write_block_bundle(tmp_path / "l16.zip", 16),
    ]
    with pytest.raises(ValueError, match="rows do not match their source Git SHA"):
        run_context_paper_figure(epsilon, blocks, tmp_path / "output")


def test_context_paper_figure_rejects_reference_model_mismatch(tmp_path: Path) -> None:
    epsilon = _write_epsilon_bundle(tmp_path / "epsilon.zip")
    blocks = [
        _write_block_bundle(tmp_path / "l8.zip", 8, reference_prefix="different"),
        _write_block_bundle(tmp_path / "l16.zip", 16, reference_prefix="different"),
    ]
    with pytest.raises(ValueError, match="categorical f_ref design"):
        run_context_paper_figure(epsilon, blocks, tmp_path / "output")
