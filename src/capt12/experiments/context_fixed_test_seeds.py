from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from capt12.certification.artifact import hash_array
from capt12.config import run_id
from capt12.data.loader import load_parquet_sample
from capt12.evaluation.metrics import prediction_metrics
from capt12.experiments.context_global_token_ldp import (
    _realized_ldp_epsilon,
    _repair_ldp_uniform,
)
from capt12.mechanisms.baselines import k_ary_rr
from capt12.mechanisms.lp import lift_block_channel, validate_channel
from capt12.models.reference import ReferenceModel
from capt12.utils.artifacts import git_sha, sha256_file

_EVALUATION_VERSION = 3
_BASELINE_METHODS = {
    "block-ldp": "context_ldp",
    "rr": "kary_rr",
}
_METHODS = {
    "capt": "context_capt",
    "block-ldp": "context_ldp",
    "rr": "kary_rr",
    "context-token-ldp": "context_token_ldp",
    "nonprivate-k64": "nonprivate_k64",
    "nonprivate-l32": "nonprivate_l32",
    "constant": "constant",
}
_BASELINE_ALIASES = {
    "block-ldp": "block-ldp",
    "block_ldp": "block-ldp",
    "context-ldp": "block-ldp",
    "context_ldp": "block-ldp",
    "rr": "rr",
    "krr": "rr",
    "k-ary-rr": "rr",
    "kary-rr": "rr",
}
_METHOD_ALIASES = {
    "capt": "capt",
    "context-capt": "capt",
    "context_capt": "capt",
    **_BASELINE_ALIASES,
    "context-token-ldp": "context-token-ldp",
    "context_token_ldp": "context-token-ldp",
    "token-ldp": "context-token-ldp",
    "nonprivate-k64": "nonprivate-k64",
    "nonprivate_k64": "nonprivate-k64",
    "nonprivate-token": "nonprivate-k64",
    "nonprivate_token": "nonprivate-k64",
    "nonprivate-l32": "nonprivate-l32",
    "nonprivate_l32": "nonprivate-l32",
    "nonprivate-block": "nonprivate-l32",
    "nonprivate_block": "nonprivate-l32",
    "constant": "constant",
    "null": "constant",
}
_METHOD_LABELS = {
    "context_capt": "CAPT (S-only)",
    "context_ldp": "Context-optimal block LDP",
    "kary_rr": "K-ary RR",
    "context_token_ldp": "Context-optimal token LDP",
    "nonprivate_k64": "No privacy (token identity)",
    "nonprivate_l32": "No privacy (block compression only)",
    "constant": "D_test constant reference",
}
_METHOD_COLORS = {
    "context_capt": "#2563A6",
    "context_ldp": "#D97706",
    "kary_rr": "#4B5563",
    "context_token_ldp": "#7C3AED",
    "nonprivate_k64": "#059669",
    "nonprivate_l32": "#0D9488",
    "constant": "#9CA3AF",
}
_METHOD_MARKERS = {
    "context_capt": "o",
    "context_ldp": "^",
    "kary_rr": "s",
    "context_token_ldp": "D",
    "nonprivate_k64": "P",
    "nonprivate_l32": "X",
    "constant": "v",
}
_METHOD_LINESTYLES = {
    "context_capt": "-",
    "context_ldp": "--",
    "kary_rr": ":",
    "context_token_ldp": "-.",
    "nonprivate_k64": "-",
    "nonprivate_l32": "--",
    "constant": ":",
}
_SAMPLED_METRICS = [
    "sampled_log_loss",
    "LLHCompVN",
    "ROC_AUC",
    "PR_AUC",
    "ECE",
    "calibration_ratio",
]


def normalize_test_baselines(baselines: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return canonical fixed-test baseline names in stable display order."""
    requested = ["block-ldp"] if baselines is None else list(baselines)
    canonical: set[str] = set()
    for value in requested:
        key = str(value).strip().lower()
        if not key:
            continue
        try:
            canonical.add(_BASELINE_ALIASES[key])
        except KeyError as error:
            raise ValueError(
                f"unknown test baseline {value!r}; choose from block-ldp,rr"
            ) from error
    if not canonical:
        raise ValueError("at least one test baseline is required")
    return [name for name in _BASELINE_METHODS if name in canonical]


def normalize_test_methods(
    methods: list[str] | tuple[str, ...] | None,
    *,
    default: tuple[str, ...] = ("capt", "block-ldp", "rr"),
) -> list[str]:
    """Return canonical method selectors in stable display order."""
    requested = list(default if methods is None else methods)
    canonical: set[str] = set()
    for value in requested:
        key = str(value).strip().lower()
        if not key:
            continue
        if key == "all":
            canonical.update(_METHODS)
            continue
        try:
            canonical.add(_METHOD_ALIASES[key])
        except KeyError as error:
            raise ValueError(
                "unknown test method "
                f"{value!r}; choose from {','.join(_METHODS)} or all"
            ) from error
    if not canonical:
        raise ValueError("at least one test method is required")
    return [name for name in _METHODS if name in canonical]


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


def _lift_context_block_channels(
    block_channels: np.ndarray,
    assignment: np.ndarray,
    decoder: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            lift_block_channel(channel, assignment, decoder)
            for channel in np.asarray(block_channels, dtype=float)
        ]
    )


def _sample_context_token_outputs(
    tokens: np.ndarray,
    context_values: np.ndarray,
    contexts: np.ndarray,
    token_channels: np.ndarray,
    uniforms: np.ndarray,
    *,
    positions_by_context: list[np.ndarray] | None = None,
    cumulative_channels: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Sample one KxK token channel selected by public context."""
    if len(tokens) != len(context_values) or len(tokens) != len(uniforms):
        raise ValueError("tokens, contexts, and uniforms must have equal length")
    channels = np.asarray(token_channels, dtype=float)
    if channels.ndim != 3 or channels.shape[0] != len(contexts):
        raise ValueError("context token channels must have shape CxKxK")
    if channels.shape[1] != channels.shape[2]:
        raise ValueError("context token channels must be square")
    positions_by_context = positions_by_context or _context_positions(
        context_values, contexts
    )
    if cumulative_channels is None:
        cumulative_channels = [np.cumsum(channel, axis=1) for channel in channels]
        for cumulative in cumulative_channels:
            cumulative[:, -1] = 1.0
    if len(positions_by_context) != len(contexts) or len(cumulative_channels) != len(
        contexts
    ):
        raise ValueError("precomputed context positions and CDFs must match contexts")
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
        path / "figures" / "fixed_capt_ldp_baseline_comparison.png",
        path / "figures" / "fixed_capt_ldp_baseline_comparison.pdf",
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


def _sample_token_outputs(
    tokens: np.ndarray,
    channel: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Sample one context-independent token channel without a block decoder."""
    if len(tokens) != len(uniforms):
        raise ValueError("tokens and uniforms must have equal length")
    matrix = np.asarray(channel, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("token channel must be square")
    cumulative = np.cumsum(matrix, axis=1)
    cumulative[:, -1] = 1.0
    outputs = np.empty(len(tokens), dtype=np.int64)
    for token in np.unique(tokens):
        positions = np.flatnonzero(tokens == token)
        outputs[positions] = np.searchsorted(
            cumulative[int(token)], uniforms[positions], side="left"
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


def _analytic_context_token_channel_metrics(
    channels: np.ndarray,
    tokens: np.ndarray,
    labels: np.ndarray,
    positions_by_context: list[np.ndarray],
    probability_grid: np.ndarray,
) -> dict[str, float]:
    """Evaluate expected randomized loss and metrics of expected scores."""
    matrices = np.asarray(channels, dtype=float)
    if matrices.ndim == 2:
        matrices = np.repeat(matrices[None, :, :], len(positions_by_context), axis=0)
    if matrices.ndim != 3 or matrices.shape[0] != len(positions_by_context):
        raise ValueError("analytic channels must have shape CxKxK or KxK")
    expected_scores = np.empty(len(tokens), dtype=float)
    expected_loss_sum = 0.0
    for context_index, positions in enumerate(positions_by_context):
        if not len(positions):
            continue
        channel = matrices[context_index]
        probabilities = probability_grid[context_index]
        token_scores = channel @ probabilities
        loss_one = channel @ -np.log(np.clip(probabilities, 1e-6, 1))
        loss_zero = channel @ -np.log(np.clip(1 - probabilities, 1e-6, 1))
        inputs = tokens[positions]
        context_labels = labels[positions]
        expected_scores[positions] = token_scores[inputs]
        expected_loss_sum += float(
            np.where(context_labels == 1, loss_one[inputs], loss_zero[inputs]).sum()
        )
    expected_score_metrics = prediction_metrics(labels, expected_scores)
    return {
        "analytic_expected_randomized_log_loss": expected_loss_sum / len(labels),
        "analytic_expected_score_ROC_AUC": expected_score_metrics["ROC_AUC"],
        "analytic_expected_score_PR_AUC": expected_score_metrics["PR_AUC"],
        "analytic_expected_score_ECE": expected_score_metrics["ECE"],
    }


def _analytic_token_channel_metrics(
    channel: np.ndarray,
    tokens: np.ndarray,
    labels: np.ndarray,
    positions_by_context: list[np.ndarray],
    probability_grid: np.ndarray,
) -> dict[str, float]:
    """Backward-compatible wrapper for one context-independent token channel."""
    return _analytic_context_token_channel_metrics(
        channel,
        tokens,
        labels,
        positions_by_context,
        probability_grid,
    )


def _plot_comparison(
    metrics: pd.DataFrame,
    output_dir: Path,
    *,
    epsilon: float,
    block_count: int,
    token_count: int,
) -> tuple[Path, Path]:
    methods = [
        method
        for method in _METHODS.values()
        if method in set(metrics["method"])
    ]
    if not methods:
        raise ValueError("fixed-test metrics contain no recognized comparison methods")
    figure, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    for axis, (metric, title) in zip(
        axes,
        (
            ("sampled_log_loss", "Log-loss (lower is better)"),
            ("ROC_AUC", "ROC-AUC (higher is better)"),
        ),
        strict=True,
    ):
        pivot = metrics.pivot(index="test_seed", columns="method", values=metric).sort_index()
        for method in methods:
            values = pivot[method]
            label = _METHOD_LABELS[method]
            if method in {"context_capt", "context_ldp", "nonprivate_l32"}:
                label += f" (L={block_count})"
            elif method in {
                "kary_rr",
                "context_token_ldp",
                "nonprivate_k64",
            }:
                label += f" (K={token_count})"
            axis.plot(
                values.index.to_numpy(dtype=float),
                values.to_numpy(dtype=float),
                color=_METHOD_COLORS[method],
                marker=_METHOD_MARKERS[method],
                linestyle=_METHOD_LINESTYLES[method],
                linewidth=1.7,
                markersize=6,
                label=f"{label}; mean={values.mean():.6f}",
                zorder=2,
            )
        all_values = pivot[methods].to_numpy(dtype=float)
        spread = float(all_values.max() - all_values.min())
        padding = max(spread * 0.16, 2e-5)
        axis.set_ylim(float(all_values.min() - padding), float(all_values.max() + padding))
        axis.set_xticks(pivot.index)
        axis.set_xlabel("Test release seed")
        axis.set_ylabel(metric.replace("sampled_", "").replace("_", " "))
        axis.set_title(title)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7.5, loc="best")
    figure.suptitle(
        f"Private and non-private mechanisms on D_test (epsilon={epsilon:g})",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.01,
        "Absolute values; focused y-axes. Each seed uses common random numbers across methods.",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / "fixed_capt_ldp_baseline_comparison.png"
    pdf = figure_dir / "fixed_capt_ldp_baseline_comparison.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def render_fixed_test_seed_figure(
    evaluation_dir: str | Path,
) -> tuple[Path, Path]:
    """Regenerate only the fixed-test PNG/PDF from serialized evaluation tables."""
    output_dir = Path(evaluation_dir).resolve()
    metadata_path = output_dir / "context_fixed_test_seed_metadata.json"
    metrics_path = output_dir / "tables" / "test_seed_metrics.csv"
    manifest_path = output_dir / "manifest.json"
    for required in (metadata_path, metrics_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"fixed-test figure input is missing: {required}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    required_columns = {"test_seed", "method", "sampled_log_loss", "ROC_AUC"}
    missing_columns = required_columns - set(metrics)
    if missing_columns:
        raise ValueError(
            f"fixed-test metric table is missing columns: {sorted(missing_columns)}"
        )
    png, pdf = _plot_comparison(
        metrics,
        output_dir,
        epsilon=float(metadata["epsilon"]),
        block_count=int(metadata["L"]),
        token_count=int(metadata.get("K", metadata.get("rr_K", 64))),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.setdefault("files", {})
    for figure in (png, pdf):
        files[str(figure.relative_to(output_dir))] = sha256_file(figure)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return png, pdf


def _paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    identifier_columns = [
        "test_seed",
        "mechanism_run_id",
        "fixed_mechanism_seed",
    ]
    if "context_capt" not in set(metrics["method"]):
        return (
            metrics[identifier_columns]
            .drop_duplicates()
            .sort_values("test_seed")
            .reset_index(drop=True)
        )
    capt = metrics.loc[metrics["method"] == "context_capt"].set_index("test_seed")
    baselines = {
        method: metrics.loc[metrics["method"] == method].set_index("test_seed")
        for method in _METHODS.values()
        if method != "context_capt" and method in set(metrics["method"])
    }
    for method, baseline in baselines.items():
        if set(capt.index) != set(baseline.index):
            raise RuntimeError(f"CAPT and {method} test seeds do not match")
    rows = []
    for seed in sorted(capt.index.astype(int)):
        row: dict[str, Any] = {
            "test_seed": seed,
            "mechanism_run_id": str(capt.loc[seed, "mechanism_run_id"]),
            "fixed_mechanism_seed": int(capt.loc[seed, "fixed_mechanism_seed"]),
        }
        for method, baseline in baselines.items():
            short = {
                "context_ldp": "ldp",
                "kary_rr": "rr",
                "context_token_ldp": "token_ldp",
                "nonprivate_k64": "nonprivate_k64",
                "nonprivate_l32": "nonprivate_l32",
                "constant": "constant",
            }[method]
            for metric in _SAMPLED_METRICS:
                row[f"capt_minus_{short}_{metric}"] = float(
                    capt.loc[seed, metric] - baseline.loc[seed, metric]
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
    for method, short in (
        ("context_ldp", "ldp"),
        ("kary_rr", "rr"),
        ("context_token_ldp", "token_ldp"),
        ("nonprivate_k64", "nonprivate_k64"),
        ("nonprivate_l32", "nonprivate_l32"),
        ("constant", "constant"),
    ):
        series = f"context_capt_minus_{method}"
        for metric in _SAMPLED_METRICS:
            column = f"capt_minus_{short}_{metric}"
            if column not in paired:
                continue
            values = paired[column].to_numpy(dtype=float)
            rows.append(
                {
                    "series": series,
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _load_fixed_artifacts(
    mechanism_run: Path,
    *,
    load_runtime_models: bool = True,
) -> dict[str, Any]:
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
    result = {
        "resolved": resolved,
        "metadata": metadata,
        "channel_manifest": channel_manifest,
        "design": design,
        "arrays": arrays,
        "reference": ReferenceModel.load(reference_path),
        "channel_path": channel_path,
    }
    if load_runtime_models:
        result["mapper"] = joblib.load(mapper_path)
        result["encoder"] = joblib.load(encoder_path)
    return result


def run_fixed_mechanism_test_seeds(
    mechanism_run: str | Path,
    test_seeds: list[int],
    *,
    methods: list[str] | tuple[str, ...] | None = None,
    baselines: list[str] | tuple[str, ...] | None = None,
    output_root: str | Path = "outputs/context_fixed_test_seed_evaluations",
    context_token_ldp_output_root: str | Path = "outputs/context_token_ldp_channels",
    solver_time_limit: float = 1800,
) -> Path:
    """Evaluate selected private/non-private mechanisms on one fixed D_test."""
    mechanism_run = Path(mechanism_run).resolve()
    seeds = sorted(set(map(int, test_seeds)))
    if methods is not None and baselines is not None:
        raise ValueError("methods and legacy baselines cannot be specified together")
    if methods is not None:
        selected_methods = normalize_test_methods(methods)
        selected_baselines: list[str] | None = None
    else:
        # Preserve the historical direct-Python default (CAPT plus block-LDP).
        selected_baselines = normalize_test_baselines(baselines)
        selected_methods = [
            "capt",
            *[name for name in selected_baselines],
        ]
    if len(seeds) < 2:
        raise ValueError("fixed mechanism test evaluation requires at least two test seeds")
    if any(seed < 0 for seed in seeds):
        raise ValueError("test seeds must be nonnegative")
    if solver_time_limit <= 0:
        raise ValueError("solver time limit must be positive")
    run_metadata = json.loads(
        (mechanism_run / "context_stratified_metadata.json").read_text(encoding="utf-8")
    )
    signature = {
        "experiment": "context_fixed_mechanism_test_seeds",
        "version": _EVALUATION_VERSION,
        "mechanism_run_id": mechanism_run.name,
        "mechanism_source_git_sha": run_metadata["source_git_sha"],
        "evaluation_source_git_sha": git_sha(),
        "test_seeds": seeds,
        "methods": selected_methods,
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
    epsilon = float(config["epsilon"])
    context_count = len(contexts)
    token_channels: dict[str, np.ndarray] = {}
    method_details: dict[str, Any] = {}
    block_repair_records: list[dict[str, Any]] = []
    context_token_artifact: Path | None = None

    if "capt" in selected_methods:
        token_channels["context_capt"] = _lift_context_block_channels(
            arrays["channels"], arrays["assignment"], arrays["decoder"]
        )
        method_details["context_capt"] = {
            "selector": "capt",
            "definition": (
                "The serialized context-specific CAPT channel protecting S only, "
                "lifted through the fixed L-block decoder."
            ),
        }

    if "block-ldp" in selected_methods:
        repair_margin = max(
            float(config.get("certificate_repair_margin", 1e-10)), 1e-12
        )
        repaired_blocks = np.empty_like(arrays["ldp_channels"], dtype=float)
        from capt12.experiments.context_token_ldp import verify_pure_ldp_decimal

        for context_index, (context, channel) in enumerate(
            zip(contexts, arrays["ldp_channels"], strict=True)
        ):
            repaired, mixing, before, after = _repair_ldp_uniform(
                channel, epsilon, margin=repair_margin
            )
            exact = verify_pure_ldp_decimal(repaired, epsilon)
            if not exact["valid"]:
                raise RuntimeError(
                    f"repaired block-LDP channel failed Decimal verification: {context}"
                )
            repaired_blocks[context_index] = repaired
            block_repair_records.append(
                {
                    "context": str(context),
                    "repair_lambda": mixing,
                    "pre_repair_max_additive_violation": before,
                    "post_repair_max_additive_violation": after,
                    "realized_epsilon": _realized_ldp_epsilon(repaired),
                    "decimal_verification": exact,
                }
            )
        token_channels["context_ldp"] = _lift_context_block_channels(
            repaired_blocks, arrays["assignment"], arrays["decoder"]
        )
        method_details["context_ldp"] = {
            "selector": "block-ldp",
            "definition": (
                "A context-specific utility-optimal epsilon-LDP LxL block channel, "
                "strictly repaired and lifted through the fixed decoder."
            ),
            "strict_uniform_repair_margin": repair_margin,
            "repair_records": block_repair_records,
        }

    if "rr" in selected_methods:
        rr_channel = k_ary_rr(token_count, epsilon)
        token_channels["kary_rr"] = np.repeat(
            rr_channel[None, :, :], context_count, axis=0
        )
        rr_denominator = math.exp(epsilon) + token_count - 1
        method_details["kary_rr"] = {
            "selector": "rr",
            "definition": (
                "Classical symmetric K-ary randomized response on the K-token "
                "alphabet; independent of public context."
            ),
            "K": token_count,
            "keep_probability": math.exp(epsilon) / rr_denominator,
            "other_probability": 1.0 / rr_denominator,
        }

    if "context-token-ldp" in selected_methods:
        from capt12.experiments.context_token_ldp import (
            load_context_token_ldp_channels,
            solve_context_token_ldp_channels,
        )

        context_token_artifact = solve_context_token_ldp_channels(
            mechanism_run,
            output_root=context_token_ldp_output_root,
            solver_time_limit=solver_time_limit,
        )
        solved_contexts, solved_channels, solved_metadata = (
            load_context_token_ldp_channels(context_token_artifact)
        )
        if not np.array_equal(solved_contexts.astype(str), contexts.astype(str)):
            raise RuntimeError("context-token-LDP contexts do not match the mechanism")
        if solved_channels.shape != (context_count, token_count, token_count):
            raise RuntimeError("context-token-LDP channels have an incompatible shape")
        token_channels["context_token_ldp"] = solved_channels
        method_details["context_token_ldp"] = {
            "selector": "context-token-ldp",
            "definition": (
                "A separately optimized unrestricted KxK pure epsilon-LDP channel "
                "for each public context, using the frozen D_design objective."
            ),
            "artifact": str(context_token_artifact),
            "artifact_channel_sha256": solved_metadata["channel_sha256"],
        }

    if "nonprivate-k64" in selected_methods:
        identity = np.eye(token_count, dtype=float)
        token_channels["nonprivate_k64"] = np.repeat(
            identity[None, :, :], context_count, axis=0
        )
        method_details["nonprivate_k64"] = {
            "selector": "nonprivate-k64",
            "definition": (
                "No privacy and no block compression: identity release on the K-token "
                "alphabet. This is the non-private f_ref ceiling for the fixed encoder."
            ),
        }

    if "nonprivate-l32" in selected_methods:
        block_count = int(
            artifacts["channel_manifest"]["designs"][artifacts["design"]]["L"]
        )
        compression_channel = lift_block_channel(
            np.eye(block_count, dtype=float),
            arrays["assignment"],
            arrays["decoder"],
        )
        token_channels["nonprivate_l32"] = np.repeat(
            compression_channel[None, :, :], context_count, axis=0
        )
        method_details["nonprivate_l32"] = {
            "selector": "nonprivate-l32",
            "definition": (
                "No privacy randomization, but retain the fixed L-block compression "
                "and decoder. This isolates representation loss."
            ),
        }

    if "constant" in selected_methods:
        method_details["constant"] = {
            "selector": "constant",
            "definition": (
                "Evaluation-only D_test null reference: predict the empirical D_test "
                "positive prevalence for every row. It is not a deployable mechanism."
            ),
            "D_test_positive_prevalence": float(labels.mean()),
        }

    expected_internal_methods = [_METHODS[name] for name in selected_methods]
    if set(token_channels) | ({"constant"} if "constant" in selected_methods else set()) != set(
        expected_internal_methods
    ):
        raise RuntimeError("selected mechanism construction is incomplete")
    for method, channels in token_channels.items():
        if channels.shape != (context_count, token_count, token_count):
            raise RuntimeError(f"{method} channel tensor has an incompatible shape")
        for channel in channels:
            validate_channel(channel, tolerance=1e-10)
    cumulative_by_method = {
        method: [np.cumsum(channel, axis=1) for channel in channels]
        for method, channels in token_channels.items()
    }
    for cumulative_channels in cumulative_by_method.values():
        for cumulative in cumulative_channels:
            cumulative[:, -1] = 1.0
    analytic_by_method = {
        method: _analytic_context_token_channel_metrics(
            channels,
            test_tokens,
            labels,
            positions_by_context,
            probability_grid,
        )
        for method, channels in token_channels.items()
    }
    if "constant" in selected_methods:
        constant_scores = np.full(len(labels), float(labels.mean()), dtype=float)
        constant_metrics = prediction_metrics(labels, constant_scores)
        analytic_by_method["constant"] = {
            "analytic_expected_randomized_log_loss": constant_metrics[
                "unweighted_log_loss"
            ],
            "analytic_expected_score_ROC_AUC": constant_metrics["ROC_AUC"],
            "analytic_expected_score_PR_AUC": constant_metrics["PR_AUC"],
            "analytic_expected_score_ECE": constant_metrics["ECE"],
        }

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        uniforms = np.random.default_rng(seed).random(len(test_frame))
        for method in expected_internal_methods:
            if method == "constant":
                scores = np.full(len(labels), float(labels.mean()), dtype=float)
            else:
                outputs = _sample_context_token_outputs(
                    test_tokens,
                    context_values,
                    contexts,
                    token_channels[method],
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
            rows.append(
                {
                    "mechanism_run_id": mechanism_run.name,
                    "fixed_mechanism_seed": int(config["frozen_design_seed"]),
                    "test_seed": seed,
                    "method": method,
                    "test_rows": len(test_frame),
                    "sampled_log_loss": sampled["unweighted_log_loss"],
                    "LLHCompVN": sampled["LLHCompVN"],
                    "ROC_AUC": sampled["ROC_AUC"],
                    "PR_AUC": sampled["PR_AUC"],
                    "ECE": sampled["ECE"],
                    "calibration_ratio": sampled["calibration_ratio"],
                    **analytic_by_method[method],
                }
            )
    metrics = pd.DataFrame(rows).sort_values(["test_seed", "method"]).reset_index(drop=True)
    paired = _paired_differences(metrics)
    summary = _metric_summary(metrics, paired)
    metrics.to_csv(output_dir / "tables" / "test_seed_metrics.csv", index=False)
    metrics.to_parquet(output_dir / "tables" / "test_seed_metrics.parquet", index=False)
    paired.to_csv(output_dir / "tables" / "test_seed_paired_differences.csv", index=False)
    summary.to_csv(output_dir / "tables" / "test_seed_summary.csv", index=False)

    block_count = int(artifacts["channel_manifest"]["designs"][artifacts["design"]]["L"])
    figure_png, figure_pdf = _plot_comparison(
        metrics,
        output_dir,
        epsilon=float(config["epsilon"]),
        block_count=block_count,
        token_count=token_count,
    )

    metadata = {
        **signature,
        "mechanism_run_path": str(mechanism_run),
        "fixed_mechanism_seed": int(config["frozen_design_seed"]),
        "design": artifacts["design"],
        "L": block_count,
        "K": token_count,
        "epsilon": float(config["epsilon"]),
        "selected_methods": selected_methods,
        "comparison_methods": expected_internal_methods,
        "method_details": method_details,
        "context_r_pooling_weight": float(config.get("context_r_pooling_weight", 0.0)),
        "test_split": config["splits"]["D_test"],
        "test_rows": len(test_frame),
        "common_random_numbers_across_methods": True,
        "test_seed_semantics": "Monte Carlo draws from the fixed released channel",
        "analytic_primary_metrics_are_seed_invariant": True,
        "mechanism_channel_file": str(artifacts["channel_path"]),
        "mechanism_channel_sha256": sha256_file(artifacts["channel_path"]),
    }
    if selected_baselines is not None:
        metadata["selected_baselines"] = selected_baselines
    if "context_ldp" in method_details:
        metadata["block_ldp_definition"] = method_details["context_ldp"]["definition"]
    if "kary_rr" in method_details:
        metadata["rr_definition"] = method_details["kary_rr"]["definition"]
        metadata["rr_K"] = token_count
        metadata["rr_keep_probability"] = method_details["kary_rr"]["keep_probability"]
        metadata["rr_other_probability"] = method_details["kary_rr"]["other_probability"]
    if context_token_artifact is not None:
        metadata["context_token_ldp_artifact"] = str(context_token_artifact)
    metadata_path = output_dir / "context_fixed_test_seed_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    report_lines = [
        "# Fixed-mechanism seeded D_test evaluation",
        "",
        f"- Mechanism run: `{mechanism_run.name}`; frozen-design seed: {config['frozen_design_seed']}.",
        f"- Context-R convex pooling weight rho: {float(config.get('context_r_pooling_weight', 0.0)):.6g}.",
        f"- Test release seeds: {', '.join(map(str, seeds))}; D_test rows: {len(test_frame):,}.",
        "- The encoder, reference model, partition, decoder, and every context-specific R matrix are fixed. Only Monte Carlo draws from the released channel change.",
        f"- Selected methods: {', '.join(selected_methods)}.",
        f"- Comparison methods: {', '.join(expected_internal_methods)}.",
        "- Primary expected randomized log loss and expected-score AUC/PR-AUC/ECE remain analytic and seed-invariant; these seeded rows quantify finite-release Monte Carlo variation.",
        "",
    ]
    for method in expected_internal_methods:
        if method == "context_capt":
            continue
        series = f"context_capt_minus_{method}"
        paired_summary = summary.loc[summary["series"] == series]
        if paired_summary.empty:
            continue
        report_lines.extend(
            [f"## Paired CAPT minus {_METHOD_LABELS[method]} means", ""]
        )
        for row in paired_summary.itertuples(index=False):
            report_lines.append(
                f"- `{row.metric}`: mean {row.mean:.9g}; sample SD {row.sample_std:.9g}; "
                f"range [{row.min:.9g}, {row.max:.9g}]."
            )
        report_lines.append("")
    (output_dir / "context_fixed_test_seed_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    output_files = [
        output_dir / "tables" / "test_seed_metrics.csv",
        output_dir / "tables" / "test_seed_metrics.parquet",
        output_dir / "tables" / "test_seed_paired_differences.csv",
        output_dir / "tables" / "test_seed_summary.csv",
        figure_png,
        figure_pdf,
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
