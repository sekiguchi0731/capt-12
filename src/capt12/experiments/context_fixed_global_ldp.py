from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capt12.certification.artifact import hash_array
from capt12.data.loader import load_parquet_sample
from capt12.evaluation.metrics import prediction_metrics
from capt12.experiments.context_fixed_test_seeds import _load_fixed_artifacts
from capt12.experiments.context_global_token_ldp import (
    _ldp_max_violation,
    _progress,
    _realized_ldp_epsilon,
    _repair_ldp_uniform,
)
from capt12.mechanisms.lp import solve_ldp_block_lp
from capt12.models.reference import TOKEN_REFERENCE_FEATURE_SCHEMA
from capt12.utils.artifacts import sha256_file

_VERSION = 1
_BLUE = "#2563A6"
_GREY = "#4B5563"
_METRICS = ("sampled_log_loss", "ROC_AUC", "PR_AUC", "ECE", "calibration_ratio")


def _emit(event: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[fixed_global_token_ldp] [{event}] {details}".rstrip(), flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_frozen_design(mechanism_run: Path) -> dict[str, np.ndarray]:
    path = mechanism_run / "mechanism" / "frozen_design.npz"
    with np.load(path) as stored:
        arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    required = {
        "objective_weights",
        "representation_token_cost",
        "representation_token_cost_hash",
        "representation_probability_grid",
        "representation_mode",
        "representation_objective",
        "reference_feature_schema",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"frozen design is missing global-LDP inputs: {sorted(missing)}")
    cost = np.asarray(arrays["representation_token_cost"], dtype=float)
    if cost.shape != (64, 64):
        raise ValueError("global token-LDP requires a 64 by 64 representation cost")
    weights = np.asarray(arrays["objective_weights"], dtype=float)
    if weights.shape != (64,) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("global token-LDP requires 64 finite positive-mass objective weights")
    stored_hash = str(arrays["representation_token_cost_hash"].item())
    if stored_hash != hash_array(cost):
        raise ValueError("frozen representation-token cost hash does not match its array")
    if str(arrays["reference_feature_schema"].item()) != TOKEN_REFERENCE_FEATURE_SCHEMA:
        raise ValueError("global token-LDP requires the categorical-token reference model")
    if str(arrays["representation_mode"].item()) != "objective_aligned":
        raise ValueError("global token-LDP requires objective_aligned representation")
    if str(arrays["representation_objective"].item()) != "empirical_logloss":
        raise ValueError("global token-LDP requires empirical_logloss representation")
    return arrays


def _test_columns(config: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            [
                str(config.get("label_col", "is_clicked")),
                *str(config["profiles"][0]).split("+"),
                *map(str, config.get("context_cols", [])),
                *map(str, config.get("phi_source_cols", [])),
            ]
        )
    )


def _sample_global_outputs(
    tokens: np.ndarray,
    channel: np.ndarray,
    uniforms: np.ndarray,
) -> np.ndarray:
    if len(tokens) != len(uniforms):
        raise ValueError("tokens and uniforms must have equal length")
    cumulative = np.cumsum(np.asarray(channel, dtype=float), axis=1)
    cumulative[:, -1] = 1.0
    outputs = np.empty(len(tokens), dtype=np.int64)
    for token in np.unique(tokens):
        positions = np.flatnonzero(tokens == token)
        outputs[positions] = np.searchsorted(
            cumulative[int(token)], uniforms[positions], side="left"
        )
    return outputs


def _probability_grid(
    reference: Any,
    context_column: str,
    contexts: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            reference.predict(
                pd.DataFrame(
                    {
                        "__token__": np.arange(64, dtype=int),
                        context_column: context,
                    }
                )
            )
            for context in contexts
        ]
    )


def _scores_for_outputs(
    outputs: np.ndarray,
    context_values: np.ndarray,
    contexts: np.ndarray,
    probability_grid: np.ndarray,
) -> np.ndarray:
    scores = np.empty(len(outputs), dtype=float)
    covered = np.zeros(len(outputs), dtype=bool)
    for index, context in enumerate(contexts):
        positions = np.flatnonzero(context_values == context)
        scores[positions] = probability_grid[index, outputs[positions]]
        covered[positions] = True
    if not covered.all():
        raise RuntimeError("D_test contains a context absent from the probability grid")
    return scores


def _analytic_global_metrics(
    channel: np.ndarray,
    tokens: np.ndarray,
    labels: np.ndarray,
    context_values: np.ndarray,
    contexts: np.ndarray,
    probability_grid: np.ndarray,
) -> dict[str, float]:
    expected_scores = np.empty(len(tokens), dtype=float)
    expected_loss = 0.0
    for index, context in enumerate(contexts):
        positions = np.flatnonzero(context_values == context)
        probabilities = probability_grid[index]
        token_scores = channel @ probabilities
        loss_one = channel @ -np.log(np.clip(probabilities, 1e-6, 1))
        loss_zero = channel @ -np.log(np.clip(1 - probabilities, 1e-6, 1))
        inputs = tokens[positions]
        context_labels = labels[positions]
        expected_scores[positions] = token_scores[inputs]
        expected_loss += float(
            np.where(context_labels == 1, loss_one[inputs], loss_zero[inputs]).sum()
        )
    metrics = prediction_metrics(labels, expected_scores)
    return {
        "expected_randomized_log_loss": expected_loss / len(labels),
        "expected_score_ROC_AUC": metrics["ROC_AUC"],
        "expected_score_PR_AUC": metrics["PR_AUC"],
        "expected_score_ECE": metrics["ECE"],
    }


def _plot(metrics: pd.DataFrame, output_dir: Path) -> None:
    labels = {"context_capt": "CAPT (L=32, rho=0)", "global_ldp": "Global LDP (K=64)"}
    colors = {"context_capt": _BLUE, "global_ldp": _GREY}
    markers = {"context_capt": "o", "global_ldp": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    panels = (
        ("sampled_log_loss", "Log-loss (lower is better)"),
        ("ROC_AUC", "ROC-AUC (higher is better)"),
    )
    for axis, (metric, title) in zip(axes, panels, strict=True):
        pivot = metrics.pivot(index="test_seed", columns="method", values=metric).sort_index()
        for seed, row in pivot.iterrows():
            axis.plot(
                [seed - 0.06, seed + 0.06],
                [row["context_capt"], row["global_ldp"]],
                color="#D1D5DB",
                linewidth=1.0,
                zorder=1,
            )
        for method, offset in (("context_capt", -0.06), ("global_ldp", 0.06)):
            values = pivot[method]
            axis.plot(
                values.index.to_numpy(dtype=float) + offset,
                values.to_numpy(dtype=float),
                color=colors[method],
                marker=markers[method],
                linewidth=1.7,
                markersize=6,
                label=f"{labels[method]}  mean={values.mean():.6f}",
                zorder=2,
            )
        all_values = pivot.to_numpy(dtype=float)
        spread = float(all_values.max() - all_values.min())
        padding = max(spread * 0.18, 2e-5)
        axis.set_ylim(float(all_values.min() - padding), float(all_values.max() + padding))
        axis.set_xticks(pivot.index)
        axis.set_xlabel("Test release seed")
        axis.set_ylabel(metric.replace("sampled_", "").replace("_", " "))
        axis.set_title(title)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8.5, loc="best")
    fig.suptitle("Fixed CAPT vs unrestricted global LDP on D_test", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Absolute metric values; focused y-axes. The same uniform draws are used within each seed.",
        ha="center",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "fixed_capt_vs_global_ldp_absolute.png", dpi=220)
    fig.savefig(figure_dir / "fixed_capt_vs_global_ldp_absolute.pdf")
    plt.close(fig)


def _completed(output_dir: Path, expected: dict[str, Any]) -> bool:
    metadata_path = output_dir / "capt_vs_global_ldp_absolute_metadata.json"
    required = (
        output_dir / "tables" / "test_seed_capt_vs_global_ldp_absolute.csv",
        output_dir / "tables" / "test_seed_capt_vs_global_ldp_absolute_summary.csv",
        output_dir / "figures" / "fixed_capt_vs_global_ldp_absolute.png",
        output_dir / "figures" / "fixed_capt_vs_global_ldp_absolute.pdf",
    )
    if not metadata_path.is_file() or not all(path.is_file() for path in required):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(metadata.get(key) == value for key, value in expected.items())


def run_fixed_capt_global_ldp_comparison(
    mechanism_run: str | Path,
    fixed_evaluation: str | Path,
    *,
    solver_time_limit: float = 1800,
) -> Path:
    """Solve one matched global K=64 LDP channel and compare five test draws."""
    started = time.perf_counter()
    mechanism_run = Path(mechanism_run).resolve()
    output_dir = Path(fixed_evaluation).resolve()
    fixed_metadata = json.loads(
        (output_dir / "context_fixed_test_seed_metadata.json").read_text(encoding="utf-8")
    )
    if fixed_metadata.get("mechanism_run_id") != mechanism_run.name:
        raise ValueError("fixed evaluation does not belong to the supplied mechanism run")
    capt_metrics = pd.read_csv(output_dir / "tables" / "test_seed_metrics.csv")
    capt_metrics = capt_metrics.loc[capt_metrics["method"] == "context_capt"].copy()
    if capt_metrics.empty or capt_metrics["test_seed"].duplicated().any():
        raise ValueError("fixed evaluation must contain one context_capt row per test seed")
    test_seeds = sorted(capt_metrics["test_seed"].astype(int).tolist())

    artifacts = _load_fixed_artifacts(mechanism_run)
    config = artifacts["resolved"]
    frozen = _load_frozen_design(mechanism_run)
    epsilon = float(config["epsilon"])
    if epsilon <= 0:
        raise ValueError("global token-LDP requires epsilon > 0")
    encoder_path = mechanism_run / "models" / "encoder.joblib"
    reference_path = mechanism_run / "models" / "reference.joblib"
    expected = {
        "version": _VERSION,
        "fixed_capt_run": mechanism_run.name,
        "fixed_evaluation": output_dir.name,
        "epsilon": epsilon,
        "global_ldp_K": 64,
        "encoder_sha256": sha256_file(encoder_path),
        "reference_model_sha256": sha256_file(reference_path),
        "representation_token_cost_hash": str(frozen["representation_token_cost_hash"].item()),
        "test_seeds": test_seeds,
    }
    if _completed(output_dir, expected):
        return output_dir

    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "channels").mkdir(parents=True, exist_ok=True)
    _emit("solve_started", epsilon=epsilon, K=64, time_limit=solver_time_limit)
    solution = solve_ldp_block_lp(
        np.asarray(frozen["representation_token_cost"], dtype=float),
        np.asarray(frozen["objective_weights"], dtype=float),
        epsilon,
        tolerance=1e-10,
        time_limit=solver_time_limit,
        progress=_progress,
        progress_label="fixed-global-token-ldp",
        heartbeat_seconds=30,
    )
    if solution.channel is None:
        raise RuntimeError(f"global token-LDP solve failed: {solution.solver.message}")
    channel, repair_lambda, pre_repair, post_repair = _repair_ldp_uniform(
        solution.channel, epsilon, margin=1e-12
    )
    channel_path = output_dir / "channels" / f"global_ldp_epsilon-{epsilon:g}.npz"
    np.savez_compressed(channel_path, channel=channel, epsilon=np.asarray(epsilon))
    _emit(
        "solve_finished",
        solver_seconds=f"{solution.solver.runtime_seconds:.1f}",
        repair_lambda=f"{repair_lambda:.3g}",
        max_violation=f"{post_repair:.3g}",
    )

    _emit("test_data_load_started", days=config["splits"]["D_test"])
    test_frame = load_parquet_sample(
        data_root=config["data_root"],
        columns=_test_columns(config),
        days=config["splits"]["D_test"],
    )
    test_frame = artifacts["mapper"].transform(test_frame)
    tokens = artifacts["encoder"].transform(test_frame)
    label_column = str(config.get("label_col", "is_clicked"))
    context_column = str(config["context_cols"][0])
    labels = test_frame[label_column].to_numpy(dtype=int)
    context_values = test_frame[context_column].astype(str).to_numpy()
    contexts = np.asarray(sorted(set(context_values)), dtype=str)
    probability_grid = _probability_grid(artifacts["reference"], context_column, contexts)
    analytic = _analytic_global_metrics(
        channel,
        tokens,
        labels,
        context_values,
        contexts,
        probability_grid,
    )
    _emit("test_data_load_finished", rows=len(test_frame), contexts=len(contexts))

    global_rows: list[dict[str, Any]] = []
    for seed in test_seeds:
        uniforms = np.random.default_rng(seed).random(len(test_frame))
        outputs = _sample_global_outputs(tokens, channel, uniforms)
        scores = _scores_for_outputs(outputs, context_values, contexts, probability_grid)
        sampled = prediction_metrics(labels, scores)
        global_rows.append(
            {
                "test_seed": seed,
                "method": "global_ldp",
                "test_rows": len(test_frame),
                "sampled_log_loss": sampled["unweighted_log_loss"],
                "ROC_AUC": sampled["ROC_AUC"],
                "PR_AUC": sampled["PR_AUC"],
                "ECE": sampled["ECE"],
                "calibration_ratio": sampled["calibration_ratio"],
            }
        )
        _emit(
            "test_seed_finished",
            seed=seed,
            log_loss=f"{sampled['unweighted_log_loss']:.9g}",
            roc_auc=f"{sampled['ROC_AUC']:.9g}",
        )

    keep = ["test_seed", "method", "test_rows", *_METRICS]
    comparison = pd.concat(
        [capt_metrics[keep], pd.DataFrame(global_rows)[keep]], ignore_index=True
    ).sort_values(["test_seed", "method"])
    summary = comparison.groupby("method", sort=True)[list(_METRICS)].agg(
        ["mean", "std", "min", "max"]
    )
    table_path = output_dir / "tables" / "test_seed_capt_vs_global_ldp_absolute.csv"
    summary_path = output_dir / "tables" / "test_seed_capt_vs_global_ldp_absolute_summary.csv"
    comparison.to_csv(table_path, index=False)
    summary.to_csv(summary_path)
    _plot(comparison, output_dir)

    weights = np.asarray(frozen["objective_weights"], dtype=float)
    weights /= weights.sum()
    objective = float(
        np.sum(
            weights[:, None]
            * channel
            * np.asarray(frozen["representation_token_cost"], dtype=float)
        )
    )
    metadata = {
        **expected,
        "fixed_mechanism_seed": int(config["frozen_design_seed"]),
        "capt_L": int(fixed_metadata["L"]),
        "common_random_numbers_across_methods": True,
        "global_ldp_definition": (
            "One unrestricted 64x64 epsilon-LDP token channel; singleton partition, "
            "identity decoder, and no public-context channel selector."
        ),
        "global_ldp_channel": str(channel_path.relative_to(output_dir)),
        "global_ldp_channel_sha256": sha256_file(channel_path),
        "objective_value": objective,
        "solver": asdict(solution.solver),
        "repair_lambda": repair_lambda,
        "pre_repair_max_additive_violation": pre_repair,
        "max_additive_violation": post_repair,
        "independent_max_additive_violation": _ldp_max_violation(channel, epsilon),
        "realized_epsilon": _realized_ldp_epsilon(channel),
        "row_sum_max_error": float(np.max(np.abs(channel.sum(axis=1) - 1))),
        "minimum_entry": float(np.min(channel)),
        "analytic_global_ldp": analytic,
        "test_rows": len(test_frame),
        "wall_seconds": time.perf_counter() - started,
    }
    (output_dir / "capt_vs_global_ldp_absolute_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _emit(
        "finished",
        output=output_dir,
        figure=output_dir / "figures" / "fixed_capt_vs_global_ldp_absolute.png",
        wall_seconds=f"{metadata['wall_seconds']:.1f}",
    )
    return output_dir
