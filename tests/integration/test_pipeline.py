from __future__ import annotations

import pandas as pd
import pytest

from capt12.pipeline import run_synthetic, run_theorem4_grid
from capt12.plots.paper import plot_paper_suite, theorem4_gap_table


@pytest.fixture(autouse=True)
def _clean_committed_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "capt12.pipeline.require_clean_worktree", lambda: "test-source-sha"
    )
    monkeypatch.setattr(
        "capt12.certification.artifact.require_clean_worktree",
        lambda: "test-source-sha",
    )


def test_synthetic_pipeline_and_all_paper_figures(tmp_path) -> None:
    config = {
        "dataset": "synthetic",
        "output_dir": str(tmp_path / "runs"),
        "seed": 2,
        "K": 4,
        "L": 2,
        "n_rows": 500,
        "epsilon": 0.5,
        "distortion": "retention",
        "partition": "frequency_balanced",
        "decoder": "design_frequency",
        "mechanisms": ["common_cover", "capt_block", "capt_full"],
    }
    path, metrics = run_synthetic(config)
    assert (path / "certificate.json").exists()
    assert len(metrics) >= 3
    generated = plot_paper_suite(tmp_path / "runs")
    assert len(generated) == (8 + 6) * 3
    assert all(item.exists() for item in generated)
    assert len(list((tmp_path / "runs" / "paper_figures").glob("*_source.csv"))) == 14
    assert isinstance(pd.read_parquet(path / "metrics.parquet"), pd.DataFrame)


def test_theorem4_grid_enumerates_decoder_and_marks_verified_full(tmp_path) -> None:
    config = {
        "dataset": "synthetic_theorem4",
        "output_dir": str(tmp_path / "runs"),
        "seeds": [0],
        "epsilon_list": [0.0],
        "K": 2,
        "L_list": [1, 2],
        "partition_list": ["frequency_balanced"],
        "decoder_list": ["design_frequency", "uniform_within_block"],
        "n_rows": 200,
        "confidence": "point",
        "solver_tolerance": 1e-9,
    }
    result = run_theorem4_grid(config)
    capt = result[
        (result["mechanism"] == "capt_block") & (result["case"] == "standard")
    ]
    assert len(capt) == 4
    assert set(capt["decoder"]) == {"design_frequency", "uniform_within_block"}
    full = result[
        (result["mechanism"] == "capt_full") & (result["case"] == "standard")
    ]
    assert full["full_oracle_optimized"].all()
    assert full["full_verification_valid"].all()
    assert full["full_comparison_eligible"].all()
    gap = theorem4_gap_table(result)
    assert gap["U_full"].notna().all()
    ineligible = result.copy()
    ineligible.loc[
        ineligible["mechanism"] == "capt_full", "full_oracle_optimized"
    ] = False
    rejected_gap = theorem4_gap_table(ineligible)
    assert rejected_gap["U_full"].isna().all()
