from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from capt12.certification.artifact import hash_array
from capt12.config import run_id
from capt12.data.loader import load_parquet_sample
from capt12.evaluation.metrics import prediction_metrics
from capt12.mechanisms.lp import lift_block_channel
from capt12.models.reference import ReferenceModel
from capt12.utils.artifacts import sha256_file

_EVALUATION_VERSION = 1
_METHOD_ARRAYS = {
    "context_capt": "channels",
    "context_ldp": "ldp_channels",
}
_SAMPLED_METRICS = [
    "sampled_log_loss",
    "ROC_AUC",
    "PR_AUC",
    "ECE",
    "calibration_ratio",
]


def _context_positions(
    context_values: np.ndarray,
    contexts: np.ndarray,
) -> list[np.ndarray]:
    positions = [np.flatnonzero(context_values == str(context)) for context in contexts]
    covered = np.zeros(len(context_values), dtype=bool)
    for item in positions:
        covered[item] = True
    if not covered.all():
        missing = sorted(set(context_values[~covered].astype(str)))
        raise RuntimeError(f"D_test contexts are missing from the fixed mechanism: {missing}")
    return positions


def _token_channel_cdfs(
    block_channels: np.ndarray,
    assignment: np.ndarray,
    decoder: np.ndarray,
) -> list[np.ndarray]:
    cumulative_channels = []
    for block_channel in block_channels:
        cumulative = np.cumsum(
            lift_block_channel(block_channel, assignment, decoder),
            axis=1,
        )
        cumulative[:, -1] = 1.0
        cumulative_channels.append(cumulative)
    return cumulative_channels


def _completed_evaluation(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = [
        path / "tables" / "test_seed_metrics.csv",
        path / "tables" / "test_seed_paired_differences.csv",
        path / "tables" / "test_seed_summary.csv",
        path / "context_fixed_test_seed_metadata.json",
        path / "context_fixed_test_seed_report.md",
    ]
    return manifest.get("status") == "complete" and all(item.is_file() for item in required)


def _sample_context_outputs(
    tokens: np.ndarray,
    context_values: np.ndarray,
    contexts: np.ndarray,
    block_channels: np.ndarray,
    assignment: np.ndarray,
    decoder: np.ndarray,
    uniforms: np.ndarray,
    *,
    positions_by_context: list[np.ndarray] | None = None,
    cumulative_channels: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Sample released tokens from serialized context channels."""
    if len(tokens) != len(context_values) or len(tokens) != len(uniforms):
        raise ValueError("tokens, contexts, and uniforms must have equal length")
    if len(contexts) != len(block_channels):
        raise ValueError("serialized contexts and channels must have equal length")
    positions_by_context = positions_by_context or _context_positions(context_values, contexts)
    cumulative_channels = cumulative_channels or _token_channel_cdfs(
        block_channels,
        assignment,
        decoder,
    )
    if len(positions_by_context) != len(contexts) or len(cumulative_channels) != len(contexts):
        raise ValueError("precomputed context positions and CDFs must match serialized contexts")
    outputs = np.empty(len(tokens), dtype=np.int64)
    for context_index, positions in enumerate(positions_by_context):
        if not len(positions):
            continue
        cumulative = cumulative_channels[context_index]
        context_tokens = tokens[positions]
        for token in np.unique(context_tokens):
            token_positions = positions[context_tokens == token]
            outputs[token_positions] = np.searchsorted(
                cumulative[int(token)],
                uniforms[token_positions],
                side="left",
            )
    return outputs


def _sampled_prediction_scores(
    outputs: np.ndarray,
    context_values: np.ndarray,
    contexts: np.ndarray,
    probability_grid: np.ndarray,
    *,
    positions_by_context: list[np.ndarray] | None = None,
) -> np.ndarray:
    positions_by_context = positions_by_context or _context_positions(context_values, contexts)
    scores = np.empty(len(outputs), dtype=float)
    for context_index, positions in enumerate(positions_by_context):
        if not len(positions):
            continue
        scores[positions] = probability_grid[context_index, outputs[positions]]
    return scores


def _paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    capt = metrics.loc[metrics["method"] == "context_capt"].set_index("test_seed")
    ldp = metrics.loc[metrics["method"] == "context_ldp"].set_index("test_seed")
    if set(capt.index) != set(ldp.index):
        raise RuntimeError("CAPT and LDP test seeds do not match")
    rows = []
    for seed in sorted(capt.index.astype(int)):
        row: dict[str, Any] = {
            "test_seed": seed,
            "mechanism_run_id": str(capt.loc[seed, "mechanism_run_id"]),
            "fixed_mechanism_seed": int(capt.loc[seed, "fixed_mechanism_seed"]),
        }
        for metric in _SAMPLED_METRICS:
            row[f"capt_minus_ldp_{metric}"] = float(
                capt.loc[seed, metric] - ldp.loc[seed, metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_summary(metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in metrics.groupby("method", sort=True):
        for metric in _SAMPLED_METRICS:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "series": method,
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    for metric in _SAMPLED_METRICS:
        values = paired[f"capt_minus_ldp_{metric}"].to_numpy(dtype=float)
        rows.append(
            {
                "series": "context_capt_minus_context_ldp",
                "metric": metric,
                "seed_count": len(values),
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _load_fixed_artifacts(mechanism_run: Path) -> dict[str, Any]:
    run_manifest = json.loads((mechanism_run / "manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != "complete":
        raise RuntimeError(f"fixed mechanism run is not complete: {mechanism_run}")
    resolved = yaml.safe_load(
        (mechanism_run / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (mechanism_run / "context_stratified_metadata.json").read_text(encoding="utf-8")
    )
    channel_manifest = json.loads(
        (mechanism_run / "mechanism" / "context_channel_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    designs = list(channel_manifest.get("designs", {}))
    if len(designs) != 1:
        raise RuntimeError("fixed test-seed evaluation requires exactly one serialized design")
    design = designs[0]
    channel_path = mechanism_run / "mechanism" / f"context_channels-{design}.npz"
    with np.load(channel_path) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    contexts = arrays["contexts"].astype(str)
    design_manifest = channel_manifest["designs"][design]
    if hash_array(arrays["assignment"]) != design_manifest["assignment_hash"]:
        raise RuntimeError("serialized assignment hash does not match the mechanism manifest")
    if hash_array(arrays["decoder"]) != design_manifest["decoder_hash"]:
        raise RuntimeError("serialized decoder hash does not match the mechanism manifest")
    for index, context in enumerate(contexts):
        if hash_array(arrays["channels"][index]) != design_manifest["contexts"][context]:
            raise RuntimeError(f"CAPT channel hash mismatch for context {context}")
        if hash_array(arrays["ldp_channels"][index]) != design_manifest["ldp_contexts"][context]:
            raise RuntimeError(f"LDP channel hash mismatch for context {context}")

    mapper_path = mechanism_run / "models" / "category_mapper.joblib"
    encoder_path = mechanism_run / "models" / "encoder.joblib"
    reference_path = mechanism_run / "models" / "reference.joblib"
    if sha256_file(mapper_path) != channel_manifest["runtime_mapper_hash"]:
        raise RuntimeError("runtime mapper hash does not match the mechanism manifest")
    if sha256_file(reference_path) != channel_manifest["reference_model_hash"]:
        raise RuntimeError("reference model hash does not match the mechanism manifest")
    return {
        "resolved": resolved,
        "metadata": metadata,
        "channel_manifest": channel_manifest,
        "design": design,
        "arrays": arrays,
        "mapper": joblib.load(mapper_path),
        "encoder": joblib.load(encoder_path),
        "reference": ReferenceModel.load(reference_path),
        "channel_path": channel_path,
    }


def run_fixed_mechanism_test_seeds(
    mechanism_run: str | Path,
    test_seeds: list[int],
    *,
    output_root: str | Path = "outputs/context_fixed_test_seed_evaluations",
) -> Path:
    """Evaluate sampled D_test releases without rebuilding the fixed mechanism."""
    mechanism_run = Path(mechanism_run).resolve()
    seeds = sorted(set(map(int, test_seeds)))
    if len(seeds) < 2:
        raise ValueError("fixed mechanism test evaluation requires at least two test seeds")
    if any(seed < 0 for seed in seeds):
        raise ValueError("test seeds must be nonnegative")
    run_metadata = json.loads(
        (mechanism_run / "context_stratified_metadata.json").read_text(encoding="utf-8")
    )
    signature = {
        "experiment": "context_fixed_mechanism_test_seeds",
        "version": _EVALUATION_VERSION,
        "mechanism_run_id": mechanism_run.name,
        "mechanism_source_git_sha": run_metadata["source_git_sha"],
        "test_seeds": seeds,
    }
    output_dir = Path(output_root) / run_id(signature)
    if _completed_evaluation(output_dir):
        return output_dir
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    artifacts = _load_fixed_artifacts(mechanism_run)
    config = artifacts["resolved"]
    arrays = artifacts["arrays"]
    contexts = arrays["contexts"].astype(str)
    context_column = str(config["context_cols"][0])
    profile = str(config["profiles"][0])
    label_column = str(config.get("label_col", "is_clicked"))
    columns = list(
        dict.fromkeys(
            [
                str(config.get("id_col", "id")),
                str(config.get("user_col", "user_id")),
                label_column,
                *profile.split("+"),
                context_column,
                *map(str, config.get("phi_source_cols", [])),
            ]
        )
    )
    test_frame = load_parquet_sample(
        data_root=config["data_root"],
        columns=columns,
        days=config["splits"]["D_test"],
    )
    test_frame = artifacts["mapper"].transform(test_frame)
    test_tokens = artifacts["encoder"].transform(test_frame)
    labels = test_frame[label_column].to_numpy(dtype=int)
    context_values = test_frame[context_column].astype(str).to_numpy()
    positions_by_context = _context_positions(context_values, contexts)
    token_count = len(arrays["assignment"])
    probability_grid = np.stack(
        [
            artifacts["reference"].predict(
                pd.DataFrame(
                    {
                        "__token__": np.arange(token_count, dtype=int),
                        context_column: context,
                    }
                )
            )
            for context in contexts
        ]
    )
    cumulative_by_method = {
        method: _token_channel_cdfs(
            arrays[array_name],
            arrays["assignment"],
            arrays["decoder"],
        )
        for method, array_name in _METHOD_ARRAYS.items()
    }
    analytic = pd.read_csv(mechanism_run / "tables" / "test_metrics.csv").set_index("method")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        uniforms = np.random.default_rng(seed).random(len(test_frame))
        for method, array_name in _METHOD_ARRAYS.items():
            outputs = _sample_context_outputs(
                test_tokens,
                context_values,
                contexts,
                arrays[array_name],
                arrays["assignment"],
                arrays["decoder"],
                uniforms,
                positions_by_context=positions_by_context,
                cumulative_channels=cumulative_by_method[method],
            )
            scores = _sampled_prediction_scores(
                outputs,
                context_values,
                contexts,
                probability_grid,
                positions_by_context=positions_by_context,
            )
            sampled = prediction_metrics(labels, scores)
            exact = analytic.loc[method]
            rows.append(
                {
                    "mechanism_run_id": mechanism_run.name,
                    "fixed_mechanism_seed": int(config["frozen_design_seed"]),
                    "test_seed": seed,
                    "method": method,
                    "test_rows": len(test_frame),
                    "sampled_log_loss": sampled["unweighted_log_loss"],
                    "ROC_AUC": sampled["ROC_AUC"],
                    "PR_AUC": sampled["PR_AUC"],
                    "ECE": sampled["ECE"],
                    "calibration_ratio": sampled["calibration_ratio"],
                    "analytic_expected_randomized_log_loss": exact[
                        "expected_randomized_log_loss"
                    ],
                    "analytic_expected_score_ROC_AUC": exact["ROC_AUC"],
                    "analytic_expected_score_PR_AUC": exact["PR_AUC"],
                    "analytic_expected_score_ECE": exact["ECE"],
                }
            )
    metrics = pd.DataFrame(rows).sort_values(["test_seed", "method"]).reset_index(drop=True)
    paired = _paired_differences(metrics)
    summary = _metric_summary(metrics, paired)
    metrics.to_csv(output_dir / "tables" / "test_seed_metrics.csv", index=False)
    metrics.to_parquet(output_dir / "tables" / "test_seed_metrics.parquet", index=False)
    paired.to_csv(output_dir / "tables" / "test_seed_paired_differences.csv", index=False)
    summary.to_csv(output_dir / "tables" / "test_seed_summary.csv", index=False)

    metadata = {
        **signature,
        "mechanism_run_path": str(mechanism_run),
        "fixed_mechanism_seed": int(config["frozen_design_seed"]),
        "design": artifacts["design"],
        "L": int(artifacts["channel_manifest"]["designs"][artifacts["design"]]["L"]),
        "epsilon": float(config["epsilon"]),
        "context_r_pooling_weight": float(config.get("context_r_pooling_weight", 0.0)),
        "test_split": config["splits"]["D_test"],
        "test_rows": len(test_frame),
        "common_random_numbers_across_methods": True,
        "test_seed_semantics": "Monte Carlo draws from the fixed released channel",
        "analytic_primary_metrics_are_seed_invariant": True,
        "mechanism_channel_file": str(artifacts["channel_path"]),
        "mechanism_channel_sha256": sha256_file(artifacts["channel_path"]),
    }
    metadata_path = output_dir / "context_fixed_test_seed_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# Fixed-mechanism seeded D_test evaluation",
        "",
        f"- Mechanism run: `{mechanism_run.name}`; frozen-design seed: {config['frozen_design_seed']}.",
        f"- Context-R convex pooling weight rho: {float(config.get('context_r_pooling_weight', 0.0)):.6g}.",
        f"- Test release seeds: {', '.join(map(str, seeds))}; D_test rows: {len(test_frame):,}.",
        "- The encoder, reference model, partition, decoder, and every context-specific R matrix are fixed. Only Monte Carlo draws from the released channel change.",
        "- Primary expected randomized log loss and expected-score AUC/PR-AUC/ECE remain analytic and seed-invariant; these seeded rows quantify finite-release Monte Carlo variation.",
        "",
        "## Paired CAPT minus LDP means",
        "",
    ]
    paired_summary = summary.loc[summary["series"] == "context_capt_minus_context_ldp"]
    for row in paired_summary.itertuples(index=False):
        report_lines.append(
            f"- `{row.metric}`: mean {row.mean:.9g}; sample SD {row.sample_std:.9g}; "
            f"range [{row.min:.9g}, {row.max:.9g}]."
        )
    (output_dir / "context_fixed_test_seed_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    output_files = [
        output_dir / "tables" / "test_seed_metrics.csv",
        output_dir / "tables" / "test_seed_metrics.parquet",
        output_dir / "tables" / "test_seed_paired_differences.csv",
        output_dir / "tables" / "test_seed_summary.csv",
        metadata_path,
        output_dir / "context_fixed_test_seed_report.md",
    ]
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "signature": signature,
                "files": {
                    str(path.relative_to(output_dir)): sha256_file(path) for path in output_files
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir
