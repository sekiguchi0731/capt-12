from __future__ import annotations

import gc
import io
import json
import math
import time
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

from capt12.config import run_id
from capt12.data.loader import load_parquet_sample
from capt12.evaluation.metrics import prediction_metrics
from capt12.experiments.context_paper_figure import (
    _load_block_sensitivity,
    _load_frontier,
)
from capt12.mechanisms.lp import solve_ldp_block_lp, validate_channel
from capt12.utils.artifacts import git_sha, sha256_file

_BLUE = "#2563A6"
_GREY = "#6B7280"
_LIGHT_BLUE = "#93B7D5"
_VERSION = 2


def _emit(event: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[global_token_ldp] [{event}] {details}".rstrip(), flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_result_checkpoint(output_dir: Path, result_rows: list[dict[str, Any]]) -> None:
    """Persist every completed cell so a later finalization error loses no LP work."""
    baselines = pd.DataFrame(result_rows)
    baselines.drop(columns="solver").to_csv(
        output_dir / "tables" / "global_token_ldp_results.csv",
        index=False,
    )
    (output_dir / "tables" / "global_token_ldp_solver_results.json").write_text(
        json.dumps(result_rows, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _ldp_max_violation(channel: np.ndarray, epsilon: float) -> float:
    matrix = np.asarray(channel, dtype=float)
    return float(np.max(matrix[:, None, :] - math.exp(epsilon) * matrix[None, :, :]))


def _repair_ldp_uniform(
    channel: np.ndarray,
    epsilon: float,
    *,
    margin: float = 1e-12,
) -> tuple[np.ndarray, float, float, float]:
    """Mix the minimum uniform mass needed for strict pure-epsilon LDP.

    Every input row of U is uniform, so each LDP difference after mixing is
    (1-lambda)d + lambda(1-exp(epsilon))/K.  The latter term is strictly
    negative for positive epsilon.  Choosing lambda from the worst additive
    violation therefore repairs the released channel itself; no denominator
    floor or diagnostic-only tolerance is used.
    """
    matrix = np.asarray(channel, dtype=float).copy()
    validate_channel(matrix, tolerance=1e-7)
    if epsilon <= 0:
        raise ValueError("global token-LDP repair requires epsilon > 0")
    if margin <= 0:
        raise ValueError("repair margin must be positive")
    # HiGHS may return entries a few ulps below zero. Canonicalize the exact
    # released row-stochastic array before deriving the certificate repair.
    if float(np.min(matrix)) < -1e-7:
        raise RuntimeError("global token-LDP solver returned a negative channel entry")
    matrix = np.maximum(matrix, 0.0)
    matrix /= matrix.sum(axis=1, keepdims=True)
    dimension = matrix.shape[0]
    before = _ldp_max_violation(matrix, epsilon)
    strict_uniform = (1 - math.exp(epsilon)) / dimension
    uniform = np.full_like(matrix, 1 / dimension)
    repaired = matrix
    mixing = 0.0
    # The first update is the analytical minimum needed for strict slack.
    # Rechecking the resulting float array and composing another minimum
    # update if needed avoids relying on cancellation in the closed form.
    for _ in range(4):
        violation = _ldp_max_violation(repaired, epsilon)
        if violation <= -margin:
            break
        additional = min(
            1.0,
            max(0.0, (violation + 2 * margin) / (violation - strict_uniform)),
        )
        repaired = (1 - additional) * repaired + additional * uniform
        repaired /= repaired.sum(axis=1, keepdims=True)
        mixing = 1 - (1 - mixing) * (1 - additional)
    validate_channel(repaired, tolerance=1e-12)
    after = _ldp_max_violation(repaired, epsilon)
    if after > -margin:
        raise RuntimeError(
            f"uniform LDP repair did not create strict slack: epsilon={epsilon:g}, {after=}"
        )
    return repaired, float(mixing), before, after


def _realized_ldp_epsilon(channel: np.ndarray) -> float:
    matrix = np.asarray(channel, dtype=float)
    realized = 0.0
    for output in range(matrix.shape[1]):
        column = matrix[:, output]
        maximum = float(np.max(column))
        minimum = float(np.min(column))
        if maximum == 0:
            continue
        if minimum <= 0:
            return math.inf
        realized = max(realized, math.log(maximum / minimum))
    return realized


def _nested_seed_bundle(archive: zipfile.ZipFile, epsilon: float) -> zipfile.ZipFile:
    wanted = f"epsilon/{epsilon:g}/sol_seed_stability_review_bundle.zip"
    try:
        payload = archive.read(wanted)
    except KeyError as error:
        raise ValueError(f"epsilon-grid bundle is missing {wanted}") from error
    return zipfile.ZipFile(io.BytesIO(payload))


def _seed_prefix(archive: zipfile.ZipFile, seed: int, run_id_value: str) -> str:
    prefix = f"runs/seed-{seed}/{run_id_value}/"
    if not any(name.startswith(prefix) for name in archive.namelist()):
        raise ValueError(f"seed bundle is missing {prefix}")
    return prefix


def _load_seed_artifact(
    epsilon_grid_bundle: Path,
    epsilon: float,
    seed: int,
    run_id_value: str,
) -> dict[str, Any]:
    with zipfile.ZipFile(epsilon_grid_bundle) as outer:
        with _nested_seed_bundle(outer, epsilon) as seed_bundle:
            prefix = _seed_prefix(seed_bundle, seed, run_id_value)
            config = yaml.safe_load(seed_bundle.read(prefix + "resolved_config.yaml"))
            mapper = joblib.load(
                io.BytesIO(seed_bundle.read(prefix + "models/category_mapper.joblib"))
            )
            encoder = joblib.load(io.BytesIO(seed_bundle.read(prefix + "models/encoder.joblib")))
            with np.load(
                io.BytesIO(seed_bundle.read(prefix + "mechanism/frozen_design.npz"))
            ) as stored:
                arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    if int(config.get("K", -1)) != 64:
        raise ValueError("global token-LDP baseline requires K=64")
    if str(arrays["representation_mode"].item()) != "objective_aligned":
        raise ValueError("global token-LDP baseline requires objective-aligned representation")
    if str(arrays["representation_objective"].item()) != "empirical_logloss":
        raise ValueError("global token-LDP baseline requires empirical_logloss")
    if arrays["representation_token_cost"].shape != (64, 64):
        raise ValueError("global token cost must be 64 by 64")
    if arrays["representation_probability_grid"].shape[0] != 64:
        raise ValueError("global probability grid must use the 64-token alphabet")
    return {"config": config, "mapper": mapper, "encoder": encoder, "arrays": arrays}


def _test_columns(config: dict[str, Any]) -> list[str]:
    profile_columns = str(config["profiles"][0]).split("+")
    return list(
        dict.fromkeys(
            [
                str(config.get("label_col", "is_clicked")),
                *profile_columns,
                *map(str, config.get("context_cols", [])),
                *map(str, config.get("phi_source_cols", [])),
            ]
        )
    )


def _evaluate_global_channel(
    channel: np.ndarray,
    mapped_test: pd.DataFrame,
    test_tokens: np.ndarray,
    labels: np.ndarray,
    arrays: dict[str, np.ndarray],
    *,
    context_column: str,
) -> dict[str, float]:
    levels = [str(value) for value in arrays["context_levels"].tolist()]
    level_index = {value: index for index, value in enumerate(levels)}
    probability_grid = np.asarray(arrays["representation_probability_grid"], dtype=float)
    contexts = mapped_test[context_column].astype(str).to_numpy()
    unknown = sorted(set(contexts) - set(level_index))
    if unknown:
        raise RuntimeError(f"D_test contains contexts absent from the frozen grid: {unknown}")
    scores = np.empty(len(mapped_test), dtype=float)
    expected_loss_sum = 0.0
    for context, context_index in level_index.items():
        mask = contexts == context
        if not mask.any():
            continue
        probabilities = probability_grid[:, context_index]
        token_scores = channel @ probabilities
        loss_one = channel @ -np.log(np.clip(probabilities, 1e-6, 1))
        loss_zero = channel @ -np.log(np.clip(1 - probabilities, 1e-6, 1))
        inputs = test_tokens[mask]
        context_labels = labels[mask]
        scores[mask] = token_scores[inputs]
        expected_loss_sum += float(
            np.where(context_labels == 1, loss_one[inputs], loss_zero[inputs]).sum()
        )
    metrics = prediction_metrics(labels, scores)
    return {
        "expected_randomized_log_loss": expected_loss_sum / len(mapped_test),
        "mixture_mean_log_loss": float(metrics["unweighted_log_loss"]),
        "ROC_AUC": float(metrics["ROC_AUC"]),
        "PR_AUC": float(metrics["PR_AUC"]),
        "ECE": float(metrics["ECE"]),
        "calibration_ratio": float(metrics["calibration_ratio"]),
    }


def _progress(event: str, fields: dict[str, Any]) -> None:
    if event in {
        "lp_problem_build_finished",
        "lp_solver_started",
        "lp_solver_heartbeat",
        "lp_solver_retry_started",
        "lp_solver_finished",
    }:
        _emit(event, **fields)


def _comparison_tables(
    frontier: pd.DataFrame,
    blocks: pd.DataFrame,
    baselines: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baselines[
        ["frozen_design_seed", "epsilon", "expected_randomized_log_loss"]
    ].rename(columns={"expected_randomized_log_loss": "global_token_ldp_log_loss"})
    frontier_comparison = frontier.merge(
        baseline,
        on=["frozen_design_seed", "epsilon"],
        how="inner",
        validate="one_to_one",
    )
    if len(frontier_comparison) != len(frontier):
        raise ValueError("global baseline is incomplete for the epsilon frontier")
    frontier_comparison["global_token_ldp_gain_micro"] = 1e6 * (
        frontier_comparison["global_token_ldp_log_loss"]
        - frontier_comparison["test_context_capt_expected_randomized_log_loss"]
    )
    epsilon_one = baseline.loc[baseline["epsilon"] == 1].drop(columns="epsilon")
    block_comparison = blocks.merge(
        epsilon_one,
        on="frozen_design_seed",
        how="inner",
        validate="many_to_one",
    )
    if len(block_comparison) != len(blocks):
        raise ValueError("global epsilon-one baseline is incomplete for block sensitivity")
    block_comparison["global_token_ldp_gain_micro"] = 1e6 * (
        block_comparison["global_token_ldp_log_loss"]
        - block_comparison["test_context_capt_expected_randomized_log_loss"]
    )
    return frontier_comparison, block_comparison


def _plot_global_comparison(
    frontier: pd.DataFrame,
    blocks: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(7.15, 3.75), constrained_layout=True, sharey=True
    )
    axis = axes[0]
    for _, group in frontier.groupby("frozen_design_seed", sort=True):
        group = group.sort_values("epsilon")
        axis.plot(
            group["robust_certified_upper_epsilon"],
            group["global_token_ldp_gain_micro"],
            color=_LIGHT_BLUE,
            lw=0.75,
            alpha=0.65,
        )
        axis.scatter(
            group["robust_certified_upper_epsilon"],
            group["global_token_ldp_gain_micro"],
            s=24,
            facecolor="white",
            edgecolor=_BLUE,
            linewidth=0.9,
            zorder=2,
        )
    means = (
        frontier.groupby("epsilon", as_index=False)
        .agg(
            robust_certified_upper_epsilon=("robust_certified_upper_epsilon", "mean"),
            global_token_ldp_gain_micro=("global_token_ldp_gain_micro", "mean"),
        )
        .sort_values("robust_certified_upper_epsilon")
    )
    axis.plot(
        means["robust_certified_upper_epsilon"],
        means["global_token_ldp_gain_micro"],
        color=_BLUE,
        marker="o",
        markersize=4.8,
        lw=2.1,
        zorder=3,
    )
    axis.axhline(0, color=_GREY, lw=1, ls="--")
    axis.set_title(
        "(a) Advantage over global\n" r"token-level $\epsilon$-LDP ($L=16$)",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel(r"Robust certified upper $\bar{\epsilon}$")
    axis.set_ylabel(
        "Improvement over global token-level LDP\n"
        r"($\mu$nats/display; higher is better)"
    )
    axis.grid(axis="y", color="#E5E7EB", lw=0.6)

    axis = axes[1]
    seeds = sorted(blocks["frozen_design_seed"].astype(int).unique())
    offsets = dict(zip(seeds, np.linspace(-0.28, 0.28, len(seeds)), strict=True))
    for seed in seeds:
        group = blocks.loc[blocks["frozen_design_seed"] == seed].sort_values("L")
        x = group["L"].to_numpy(float) + offsets[seed]
        y = group["global_token_ldp_gain_micro"].to_numpy(float)
        axis.plot(x, y, color=_LIGHT_BLUE, lw=0.75, alpha=0.65)
        axis.scatter(
            x,
            y,
            s=24,
            facecolor="white",
            edgecolor=_BLUE,
            linewidth=0.9,
            zorder=2,
        )
    block_counts = sorted(blocks["L"].astype(int).unique())
    block_means = blocks.groupby("L")["global_token_ldp_gain_micro"].mean()
    for block_count in block_counts:
        axis.hlines(
            block_means.loc[block_count],
            block_count - 0.52,
            block_count + 0.52,
            color=_BLUE,
            lw=2.4,
            zorder=4,
        )
    axis.axhline(0, color=_GREY, lw=1, ls="--")
    axis.set_title(
        "(b) Effect of the number of\n" r"CAPT blocks ($\epsilon=1$)",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("Number of CAPT blocks L")
    axis.set_xticks(block_counts, [str(value) for value in block_counts])
    axis.grid(axis="y", color="#E5E7EB", lw=0.6)
    handles = [
        Line2D(
            [0],
            [0],
            color=_LIGHT_BLUE,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=_BLUE,
            lw=0.75,
            label="Paired frozen-design seed",
        ),
        Line2D([0], [0], color=_GREY, lw=1, ls="--", label="Parity with global LDP"),
        Line2D([0], [0], color=_BLUE, lw=2.4, label="Seed mean"),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.4,
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figure_dir / "context_global_token_ldp_main.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "capt12 context-global-token-ldp",
            "CreationDate": datetime(2000, 1, 1, tzinfo=UTC),
        },
    )
    fig.savefig(
        figure_dir / "context_global_token_ldp_main.png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "capt12 context-global-token-ldp"},
    )
    plt.close(fig)


def _write_bundle(output_dir: Path) -> Path:
    bundle = output_dir / "sol_global_token_ldp_comparison_bundle.zip"
    temporary = output_dir / ".sol_global_token_ldp_comparison_bundle.zip.tmp"
    external_manifest = output_dir / "sol_global_token_ldp_comparison_bundle_manifest.json"
    members = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path not in {bundle, temporary, external_manifest}
    )
    fixed_timestamp = (2000, 1, 1, 0, 0, 0)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in members:
                info = zipfile.ZipInfo(path.relative_to(output_dir).as_posix(), fixed_timestamp)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        temporary.replace(bundle)
    finally:
        temporary.unlink(missing_ok=True)
    return bundle


def run_global_token_ldp_comparison(
    epsilon_grid_bundle: Path,
    block_comparison_bundles: list[Path],
    *,
    data_root: Path | None = None,
    solver_time_limit: float = 300,
    output_root: Path = Path("outputs/global_token_ldp_comparisons"),
) -> Path:
    started = time.perf_counter()
    frontier, frontier_metadata = _load_frontier(epsilon_grid_bundle)
    blocks, block_metadata = _load_block_sensitivity(block_comparison_bundles)
    epsilons = sorted(frontier["epsilon"].astype(float).unique())
    seeds = sorted(frontier["frozen_design_seed"].astype(int).unique())
    if tuple(seeds) != tuple(block_metadata["frozen_design_seeds"]):
        raise ValueError("epsilon and block inputs must share the paired seed family")
    signature = {
        "experiment": "global_token_ldp_comparison",
        "version": _VERSION,
        "K": 64,
        "partition": "singleton",
        "decoder": "identity",
        "objective": "empirical_logloss",
        "epsilon_grid_bundle_sha256": sha256_file(epsilon_grid_bundle),
        "block_comparison_bundle_sha256": [
            sha256_file(path) for path in block_comparison_bundles
        ],
    }
    output_dir = output_root / run_id(signature)
    for directory in (output_dir, output_dir / "tables", output_dir / "channels"):
        directory.mkdir(parents=True, exist_ok=True)

    first = frontier.sort_values(["epsilon", "frozen_design_seed"]).iloc[0]
    first_artifact = _load_seed_artifact(
        epsilon_grid_bundle,
        float(first["epsilon"]),
        int(first["frozen_design_seed"]),
        str(first["run_id"]),
    )
    reference_config = first_artifact["config"]
    resolved_data_root = data_root or Path(reference_config["data_root"])
    _emit(
        "test_data_load_started",
        data_root=resolved_data_root,
        days=reference_config["splits"]["D_test"],
    )
    raw_test = load_parquet_sample(
        data_root=resolved_data_root,
        columns=_test_columns(reference_config),
        days=reference_config["splits"]["D_test"],
    )
    _emit("test_data_load_finished", rows=len(raw_test))
    result_rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds, start=1):
        seed_rows = frontier.loc[frontier["frozen_design_seed"] == seed].set_index("epsilon")
        token_cache: tuple[pd.DataFrame, np.ndarray, np.ndarray] | None = None
        expected_encoder_hash = str(seed_rows.iloc[0]["encoder_sha256"])
        for epsilon in epsilons:
            row = seed_rows.loc[epsilon]
            if str(row["encoder_sha256"]) != expected_encoder_hash:
                raise ValueError(f"seed {seed} encoder changed across epsilon")
            artifact = _load_seed_artifact(
                epsilon_grid_bundle,
                epsilon,
                seed,
                str(row["run_id"]),
            )
            config = artifact["config"]
            if config["splits"]["D_test"] != reference_config["splits"]["D_test"]:
                raise ValueError("D_test split changed across global baseline cells")
            if token_cache is None:
                mapped = artifact["mapper"].transform(raw_test)
                tokens = artifact["encoder"].transform(mapped)
                labels = mapped[str(config.get("label_col", "is_clicked"))].to_numpy(dtype=int)
                token_cache = (mapped, tokens, labels)
            mapped, tokens, labels = token_cache
            arrays = artifact["arrays"]
            _emit(
                "cell_started",
                seed=seed,
                seed_index=f"{seed_index}/{len(seeds)}",
                epsilon=epsilon,
                variables=4096,
                ldp_constraints=64 * 63 * 64,
            )
            cell_started = time.perf_counter()
            solution = solve_ldp_block_lp(
                arrays["representation_token_cost"],
                arrays["objective_weights"],
                epsilon,
                tolerance=1e-10,
                time_limit=solver_time_limit,
                progress=_progress,
                progress_label=f"global-token-ldp/seed={seed}/epsilon={epsilon:g}",
                heartbeat_seconds=30,
            )
            if solution.channel is None:
                raise RuntimeError(
                    f"global token-LDP failed for seed={seed}, epsilon={epsilon:g}: "
                    f"{solution.solver.message}"
                )
            repaired, mixing, before, after = _repair_ldp_uniform(
                solution.channel,
                epsilon,
                margin=1e-12,
            )
            metrics = _evaluate_global_channel(
                repaired,
                mapped,
                tokens,
                labels,
                arrays,
                context_column=str(config["context_cols"][0]),
            )
            objective_weights = np.asarray(arrays["objective_weights"], dtype=float)
            objective_weights /= objective_weights.sum()
            objective = float(
                np.sum(
                    objective_weights[:, None]
                    * repaired
                    * np.asarray(arrays["representation_token_cost"], dtype=float)
                )
            )
            realized = _realized_ldp_epsilon(repaired)
            if not math.isfinite(realized) or realized > epsilon + 1e-10:
                raise RuntimeError(
                    f"global token-LDP realized epsilon failed: {seed=}, {epsilon=}, {realized=}"
                )
            channel_path = output_dir / "channels" / f"seed-{seed}-epsilon-{epsilon:g}.npz"
            np.savez_compressed(
                channel_path,
                channel=repaired,
                epsilon=np.asarray(epsilon),
                frozen_design_seed=np.asarray(seed),
            )
            result_rows.append(
                {
                    "frozen_design_seed": seed,
                    "epsilon": epsilon,
                    "K": 64,
                    "partition": "singleton",
                    "decoder": "identity",
                    "channel_selector": "global_Z_only",
                    "utility_objective": "empirical_logloss",
                    "encoder_sha256": expected_encoder_hash,
                    "source_run_id": str(row["run_id"]),
                    "objective_value": objective,
                    "solver": asdict(solution.solver),
                    "solver_runtime_seconds": solution.solver.runtime_seconds,
                    "repair_lambda": mixing,
                    "pre_repair_max_additive_violation": before,
                    "max_additive_violation": after,
                    "realized_epsilon": realized,
                    "row_sum_max_error": float(np.max(np.abs(repaired.sum(axis=1) - 1))),
                    "minimum_entry": float(np.min(repaired)),
                    "channel_sha256": sha256_file(channel_path),
                    **metrics,
                }
            )
            _emit(
                "cell_finished",
                seed=seed,
                epsilon=epsilon,
                wall_seconds=f"{time.perf_counter() - cell_started:.1f}",
                solver_seconds=f"{solution.solver.runtime_seconds:.1f}",
                expected_log_loss=f"{metrics['expected_randomized_log_loss']:.12g}",
                max_additive_violation=f"{after:.3g}",
            )
            _write_result_checkpoint(output_dir, result_rows)
        del token_cache
        gc.collect()
    baselines = pd.DataFrame(result_rows)
    _write_result_checkpoint(output_dir, result_rows)
    frontier_comparison, block_comparison = _comparison_tables(frontier, blocks, baselines)
    frontier_comparison.to_csv(
        output_dir / "tables" / "global_frontier_seed_points.csv", index=False
    )
    block_comparison.to_csv(
        output_dir / "tables" / "global_block_sensitivity_seed_points.csv", index=False
    )
    _plot_global_comparison(frontier_comparison, block_comparison, output_dir)
    summary = {
        "frontier": (
            frontier_comparison.groupby("epsilon")["global_token_ldp_gain_micro"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .to_dict(orient="records")
        ),
        "blocks": (
            block_comparison.groupby("L")["global_token_ldp_gain_micro"]
            .agg(["mean", "min", "max"])
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    metadata = {
        **signature,
        "analysis_git_sha": git_sha(),
        "baseline_definition": (
            "One global unrestricted 64x64 epsilon-LDP token channel with singleton "
            "partition and identity decoder; the public context is not a channel selector."
        ),
        "epsilon_values": epsilons,
        "frozen_design_seeds": seeds,
        "all_global_ldp_checks_valid": bool(
            np.isfinite(baselines["realized_epsilon"]).all()
            and (baselines["realized_epsilon"] <= baselines["epsilon"] + 1e-10).all()
            and (baselines["max_additive_violation"] <= 0).all()
        ),
        "wall_seconds": time.perf_counter() - started,
        "input_paths": {
            "epsilon_grid_bundle": str(epsilon_grid_bundle.resolve()),
            "block_comparison_bundles": [
                str(path.resolve()) for path in block_comparison_bundles
            ],
            "data_root": str(resolved_data_root.resolve()),
        },
        "summary": summary,
        "frontier_source_git_sha": frontier_metadata["source_git_sha"],
        "block_source_git_sha": block_metadata["source_git_sha"],
    }
    (output_dir / "global_token_ldp_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Global token-level optimal LDP comparison",
        "",
        metadata["baseline_definition"],
        "",
        "The baseline is stronger than every matched block-LDP class used in the CAPT runs. "
        "Positive plotted values mean held-out CAPT log loss is lower than this global baseline.",
        "",
        f"All 15 direct LDP checks valid: **{metadata['all_global_ldp_checks_valid']}**.",
        f"Total wall time: **{metadata['wall_seconds']:.1f} seconds**.",
        "",
        "Seed variation is a mechanism-design sensitivity analysis, not an independent-sample "
        "confidence interval.",
    ]
    (output_dir / "global_token_ldp_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    bundle = _write_bundle(output_dir)
    manifest = {
        "bundle": bundle.name,
        "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": sha256_file(bundle),
    }
    (output_dir / "sol_global_token_ldp_comparison_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _emit("finished", output=output_dir, bundle=bundle, wall_seconds=f"{metadata['wall_seconds']:.1f}")
    return output_dir
