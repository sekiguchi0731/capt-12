from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import capt12.experiments.context_fixed_test_seeds as fixed_test_seeds
from capt12.experiments.context_fixed_test_seeds import (
    _metric_summary,
    _paired_differences,
    _sample_context_outputs,
    _sample_token_outputs,
    _sampled_prediction_scores,
    normalize_test_baselines,
    render_fixed_test_seed_figure,
    run_fixed_mechanism_test_seeds,
)
from capt12.utils.artifacts import sha256_file


def test_sample_context_outputs_uses_fixed_context_channels() -> None:
    tokens = np.asarray([0, 1, 0, 1])
    context_values = np.asarray(["a", "a", "b", "b"])
    contexts = np.asarray(["a", "b"])
    block_channels = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    assignment = np.asarray([0, 1])
    decoder = np.eye(2)
    outputs = _sample_context_outputs(
        tokens,
        context_values,
        contexts,
        block_channels,
        assignment,
        decoder,
        np.asarray([0.1, 0.9, 0.3, 0.7]),
    )
    assert outputs.tolist() == [0, 1, 1, 0]


def test_sampled_prediction_scores_respect_public_context() -> None:
    scores = _sampled_prediction_scores(
        np.asarray([0, 1, 1, 0]),
        np.asarray(["a", "a", "b", "b"]),
        np.asarray(["a", "b"]),
        np.asarray([[0.1, 0.2], [0.8, 0.9]]),
    )
    assert np.allclose(scores, [0.1, 0.2, 0.9, 0.8])


def test_sample_token_outputs_applies_one_context_independent_rr_channel() -> None:
    outputs = _sample_token_outputs(
        np.asarray([0, 1, 1, 0]),
        np.asarray([[0.75, 0.25], [0.25, 0.75]]),
        np.asarray([0.1, 0.1, 0.9, 0.9]),
    )
    assert outputs.tolist() == [0, 0, 1, 1]


def test_test_baseline_names_accept_aliases_and_use_stable_order() -> None:
    assert normalize_test_baselines(["RR", "context_ldp", "rr"]) == [
        "block-ldp",
        "rr",
    ]


def test_test_seed_tables_use_paired_capt_minus_ldp_differences() -> None:
    rows = []
    for seed in (0, 1):
        for method, offset in (("context_capt", 0.0), ("context_ldp", 0.1)):
            rows.append(
                {
                    "mechanism_run_id": "run",
                    "fixed_mechanism_seed": 7,
                    "test_seed": seed,
                    "method": method,
                    "sampled_log_loss": 0.5 + offset + seed * 0.01,
                    "LLHCompVN": 0.2 - offset,
                    "ROC_AUC": 0.7 - offset,
                    "PR_AUC": 0.6 - offset,
                    "ECE": 0.02 + offset,
                    "calibration_ratio": 1.0 + offset,
                }
            )
    metrics = pd.DataFrame(rows)
    paired = _paired_differences(metrics)
    assert np.allclose(paired["capt_minus_ldp_sampled_log_loss"], -0.1)
    assert np.allclose(paired["capt_minus_ldp_ROC_AUC"], 0.1)
    summary = _metric_summary(metrics, paired)
    delta = summary.loc[
        (summary["series"] == "context_capt_minus_context_ldp")
        & (summary["metric"] == "sampled_log_loss")
    ].iloc[0]
    assert delta["seed_count"] == 2
    assert np.isclose(delta["mean"], -0.1)


def test_fixed_mechanism_test_seeds_write_and_reuse_evaluation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mechanism_run = tmp_path / "mechanism-run"
    (mechanism_run / "tables").mkdir(parents=True)
    (mechanism_run / "context_stratified_metadata.json").write_text(
        json.dumps({"source_git_sha": "abc"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "method": method,
                "expected_randomized_log_loss": 0.5,
                "ROC_AUC": 0.6,
                "PR_AUC": 0.5,
                "ECE": 0.1,
            }
            for method in ("context_capt", "context_ldp")
        ]
    ).to_csv(mechanism_run / "tables" / "test_metrics.csv", index=False)
    channel_path = mechanism_run / "channels.npz"
    channel_path.write_bytes(b"fixed-channel")

    class IdentityMapper:
        def transform(self, frame):
            return frame.copy()

    class TokenEncoder:
        def transform(self, frame):
            return frame["src"].to_numpy(dtype=int)

    class Reference:
        def predict(self, frame):
            return np.where(frame["__token__"].to_numpy(dtype=int) == 0, 0.2, 0.8)

    config = {
        "frozen_design_seed": 0,
        "context_cols": ["ctx"],
        "profiles": ["s"],
        "label_col": "y",
        "id_col": "id",
        "user_col": "user",
        "phi_source_cols": ["src"],
        "data_root": "unused",
        "splits": {"D_test": [1]},
        "epsilon": 1.0,
        "context_r_pooling_weight": 0.25,
    }
    arrays = {
        "contexts": np.asarray(["a"]),
        "channels": np.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
        "ldp_channels": np.asarray([[[0.5, 0.5], [0.5, 0.5]]]),
        "assignment": np.asarray([0, 1]),
        "decoder": np.eye(2),
    }
    monkeypatch.setattr(
        fixed_test_seeds,
        "_load_fixed_artifacts",
        lambda path: {
            "resolved": config,
            "design": "design-L2",
            "arrays": arrays,
            "mapper": IdentityMapper(),
            "encoder": TokenEncoder(),
            "reference": Reference(),
            "channel_manifest": {"designs": {"design-L2": {"L": 2}}},
            "channel_path": channel_path,
        },
    )
    test_frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "user": [10, 11, 12, 13],
            "y": [0, 1, 0, 1],
            "s": ["x"] * 4,
            "ctx": ["a"] * 4,
            "src": [0, 1, 0, 1],
        }
    )
    monkeypatch.setattr(
        fixed_test_seeds,
        "load_parquet_sample",
        lambda **kwargs: test_frame.copy(),
    )
    output = run_fixed_mechanism_test_seeds(
        mechanism_run,
        [1, 0, 1],
        output_root=tmp_path / "evaluations",
    )
    metrics = pd.read_csv(output / "tables" / "test_seed_metrics.csv")
    assert len(metrics) == 4
    assert metrics["test_seed"].tolist() == [0, 0, 1, 1]
    assert set(metrics["method"]) == {"context_capt", "context_ldp"}
    assert (output / "context_fixed_test_seed_report.md").is_file()
    assert (output / "figures" / "fixed_capt_ldp_baseline_comparison.png").is_file()
    metadata = json.loads((output / "context_fixed_test_seed_metadata.json").read_text())
    assert metadata["context_r_pooling_weight"] == 0.25

    monkeypatch.setattr(
        fixed_test_seeds,
        "load_parquet_sample",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("evaluation should be reused")),
    )
    assert (
        run_fixed_mechanism_test_seeds(
            mechanism_run,
            [0, 1],
            output_root=tmp_path / "evaluations",
        )
        == output
    )


def test_fixed_mechanism_test_seeds_can_evaluate_rr_and_block_ldp_together(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mechanism_run = tmp_path / "mechanism-run"
    (mechanism_run / "tables").mkdir(parents=True)
    (mechanism_run / "context_stratified_metadata.json").write_text(
        json.dumps({"source_git_sha": "abc"}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "method": method,
                "expected_randomized_log_loss": 0.5,
                "ROC_AUC": 0.6,
                "PR_AUC": 0.5,
                "ECE": 0.1,
            }
            for method in ("context_capt", "context_ldp")
        ]
    ).to_csv(mechanism_run / "tables" / "test_metrics.csv", index=False)
    channel_path = mechanism_run / "channels.npz"
    channel_path.write_bytes(b"fixed-channel")

    class IdentityMapper:
        def transform(self, frame):
            return frame.copy()

    class TokenEncoder:
        def transform(self, frame):
            return frame["src"].to_numpy(dtype=int)

    class Reference:
        def predict(self, frame):
            return np.where(frame["__token__"].to_numpy(dtype=int) == 0, 0.2, 0.8)

    config = {
        "frozen_design_seed": 0,
        "context_cols": ["ctx"],
        "profiles": ["s"],
        "label_col": "y",
        "id_col": "id",
        "user_col": "user",
        "phi_source_cols": ["src"],
        "data_root": "unused",
        "splits": {"D_test": [1]},
        "epsilon": 1.0,
    }
    arrays = {
        "contexts": np.asarray(["a"]),
        "channels": np.asarray([[[1.0, 0.0], [0.0, 1.0]]]),
        "ldp_channels": np.asarray([[[0.5, 0.5], [0.5, 0.5]]]),
        "assignment": np.asarray([0, 1]),
        "decoder": np.eye(2),
    }
    monkeypatch.setattr(
        fixed_test_seeds,
        "_load_fixed_artifacts",
        lambda path: {
            "resolved": config,
            "design": "design-L2",
            "arrays": arrays,
            "mapper": IdentityMapper(),
            "encoder": TokenEncoder(),
            "reference": Reference(),
            "channel_manifest": {"designs": {"design-L2": {"L": 2}}},
            "channel_path": channel_path,
        },
    )
    test_frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "user": [10, 11, 12, 13],
            "y": [0, 1, 0, 1],
            "s": ["x"] * 4,
            "ctx": ["a"] * 4,
            "src": [0, 1, 0, 1],
        }
    )
    monkeypatch.setattr(fixed_test_seeds, "load_parquet_sample", lambda **kwargs: test_frame.copy())

    output = run_fixed_mechanism_test_seeds(
        mechanism_run,
        [0, 1],
        baselines=["rr", "block-ldp"],
        output_root=tmp_path / "evaluations",
    )

    metrics = pd.read_csv(output / "tables" / "test_seed_metrics.csv")
    assert set(metrics["method"]) == {"context_capt", "context_ldp", "kary_rr"}
    assert len(metrics) == 6
    paired = pd.read_csv(output / "tables" / "test_seed_paired_differences.csv")
    assert "capt_minus_ldp_sampled_log_loss" in paired
    assert "capt_minus_rr_sampled_log_loss" in paired
    metadata = json.loads((output / "context_fixed_test_seed_metadata.json").read_text())
    assert metadata["selected_baselines"] == ["block-ldp", "rr"]
    assert np.isclose(metadata["rr_keep_probability"], np.e / (np.e + 1))

    png = output / "figures" / "fixed_capt_ldp_baseline_comparison.png"
    png.write_bytes(b"stale figure")
    original_plot = Axes.plot
    series_x: list[np.ndarray] = []

    def tracked_plot(axis, x, y, *args, **kwargs):
        if kwargs.get("label"):
            series_x.append(np.asarray(x, dtype=float))
        return original_plot(axis, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", tracked_plot)
    rendered_png, rendered_pdf = render_fixed_test_seed_figure(output)
    assert rendered_png == png
    assert rendered_pdf.is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["files"][str(png.relative_to(output))] == sha256_file(png)
    assert series_x
    assert all(np.array_equal(values, [0.0, 1.0]) for values in series_x)
