from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from capt12.experiments.context_seed_stability import (
    _joint_design,
    _plot_stability,
    _stability_table,
    _write_review_bundle,
)
from capt12.utils.artifacts import sha256_file


def test_joint_seed_stability_design_accepts_reported_dimensions() -> None:
    for block_count in (8, 16, 32):
        name = f"joint_kmedoids_cost_medoid_L{block_count}"
        assert _joint_design({"context_designs": [name]}) == (name, block_count)


def _seed_frame() -> pd.DataFrame:
    rows = []
    for seed, shift in [(0, 0.0), (1, 1e-10), (2, -1e-10)]:
        row = {
            "frozen_design_seed": seed,
            "L": 8,
            "context_capt_distortion": 3.2e-8 + shift,
            "context_ldp_distortion": 3.7e-8 + shift,
            "capt_advantage_over_context_ldp": 5e-9,
            "relative_capt_reduction_vs_context_ldp": 5e-9 / (3.7e-8 + shift),
            "objective_constant_floor": 0.64,
            "context_ldp_excess_objective": 3.7e-8 + shift,
            "context_capt_excess_objective": 3.2e-8 + shift,
            "relative_excess_capt_reduction_vs_context_ldp": 5e-9 / (3.7e-8 + shift),
            "capt_advantage_micro_objective_units": 0.005,
            "strict_advantage_context_count": 20 + seed,
            "strict_advantage_context_mass": 0.99 - seed * 1e-4,
            "ldp_degraded_context_count": 5,
            "ldp_degraded_context_mass": 2e-4,
            "tie_or_other_context_mass": 1 - (0.99 - seed * 1e-4) - 2e-4,
            "mass_weighted_capt_row_tv": 0.51,
            "max_capt_row_tv": 0.6,
            "max_repair_lambda": 1e-9,
            "mass_weighted_repair_lambda": 9e-10,
            "conservative_max_realized_epsilon": 0.9999999998,
            "conservative_max_additive_violation": -1e-10,
            "wall_seconds": 200.0 + seed,
            "process_peak_rss_bytes": 2_000_000_000 + seed,
            "utility_objective": "teacher_kl",
            "representation_mode": "objective_aligned",
            "representation_objective": "teacher_kl",
            "lower_audit_context_capt_epsilon": 0.1 + seed * 0.01,
            "lower_audit_context_ldp_epsilon": 0.08 + seed * 0.01,
        }
        for method, method_shift in [("capt", 0.0), ("ldp", 1e-6)]:
            row[f"test_context_{method}_expected_randomized_log_loss"] = (
                0.647 + method_shift + seed * 1e-7
            )
            row[f"test_context_{method}_ROC_AUC"] = 0.568 - method_shift
            row[f"test_context_{method}_PR_AUC"] = 0.438 - method_shift
            row[f"test_context_{method}_ECE"] = 0.011 + method_shift
        for metric in ["expected_randomized_log_loss", "ROC_AUC", "PR_AUC"]:
            row[f"test_capt_minus_ldp_{metric}"] = (
                row[f"test_context_capt_{metric}"] - row[f"test_context_ldp_{metric}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_stability_table_and_figure_are_complete(tmp_path: Path) -> None:
    frame = _seed_frame()
    summary = _stability_table(frame)
    advantage = summary.set_index("metric").loc["capt_advantage_over_context_ldp"]
    assert advantage["seed_count"] == 3
    assert advantage["range"] == 0

    (tmp_path / "figures").mkdir()
    _plot_stability(frame, tmp_path)
    assert (tmp_path / "figures" / "context_seed_stability.pdf").stat().st_size > 0
    assert (tmp_path / "figures" / "context_seed_stability.png").stat().st_size > 0


def test_seed_review_bundle_is_deterministic_and_expands_runs(tmp_path: Path) -> None:
    output = tmp_path / "summary"
    output.mkdir()
    (output / "context_seed_stability_report.md").write_text("stable\n")
    seed_paths = {}
    for seed in [0, 1]:
        run = tmp_path / f"run-{seed}"
        run.mkdir()
        with zipfile.ZipFile(run / "sol_review_bundle.zip", "w") as archive:
            archive.writestr(f"run-{seed}/certificate.json", f"seed={seed}")
        seed_paths[seed] = run

    bundle = _write_review_bundle(output, seed_paths)
    first_hash = sha256_file(bundle)
    bundle = _write_review_bundle(output, seed_paths)
    assert sha256_file(bundle) == first_hash
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "summary/context_seed_stability_report.md" in names
    assert "runs/seed-0/run-0/certificate.json" in names
    assert "runs/seed-1/run-1/certificate.json" in names
    assert "sol_seed_stability_review_bundle_manifest.json" in names
