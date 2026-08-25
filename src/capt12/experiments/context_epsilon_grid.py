from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capt12.config import canonical_json, run_id, validate_config
from capt12.experiments.context_seed_stability import run_context_seed_stability
from capt12.pipeline import record_source_provenance
from capt12.utils.artifacts import sha256_file

_GRID_VERSION = 1


def _boolean_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"invalid boolean values in {series.name}")
    return normalized == "true"


def _load_epsilon_summaries(summary_dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(summary_dirs) < 2:
        raise ValueError("epsilon grid requires at least two seed-summary directories")
    frames: dict[float, pd.DataFrame] = {}
    metadata: dict[float, dict[str, Any]] = {}
    normalized_configs: dict[float, str] = {}
    for raw_path in summary_dirs:
        path = raw_path.resolve()
        metadata_path = path / "context_seed_stability_metadata.json"
        results_path = path / "tables" / "seed_results.csv"
        config_path = path / "summary_config.json"
        bundle_path = path / "sol_seed_stability_review_bundle.zip"
        for required in (metadata_path, results_path, config_path, bundle_path):
            if not required.is_file():
                raise ValueError(f"incomplete epsilon-grid input: missing {required}")
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        epsilon = float(item["epsilon"])
        if epsilon <= 0 or epsilon in frames:
            raise ValueError(f"duplicate or invalid epsilon summary: {epsilon}")
        if not bool(item.get("all_certificates_valid")):
            raise ValueError(f"epsilon={epsilon:g} contains an invalid certificate")
        frame = pd.read_csv(results_path)
        if set(frame["epsilon"].astype(float)) != {epsilon}:
            raise ValueError(f"epsilon={epsilon:g} seed table has inconsistent budgets")
        if not _boolean_values(frame["all_certificates_valid"]).all():
            raise ValueError(f"epsilon={epsilon:g} seed table contains invalid certificates")
        realized = frame["conservative_max_realized_epsilon"].to_numpy(float)
        if not np.isfinite(realized).all() or float(realized.max()) > epsilon:
            raise ValueError(f"epsilon={epsilon:g} fails conservative pure-epsilon verification")
        config = json.loads(config_path.read_text(encoding="utf-8"))["base_config"]
        config = dict(config)
        config.pop("epsilon", None)
        normalized_configs[epsilon] = canonical_json(config)
        frame.insert(0, "epsilon_summary_path", str(path))
        frames[epsilon] = frame
        metadata[epsilon] = item

    if len(set(normalized_configs.values())) != 1:
        raise ValueError("epsilon summaries differ in a resolved setting other than epsilon")
    source_shas = {str(item["source_git_sha"]) for item in metadata.values()}
    dimensions = {int(item["L"]) for item in metadata.values()}
    seed_sets = {tuple(map(int, item["frozen_design_seeds"])) for item in metadata.values()}
    objectives = {str(item["context_utility_objective"]) for item in metadata.values()}
    representations = {str(item["context_representation_mode"]) for item in metadata.values()}
    if any(len(values) != 1 for values in (source_shas, dimensions, seed_sets, objectives, representations)):
        raise ValueError("source SHA, L, seeds, objective, and representation must match across epsilon")

    source_sha = next(iter(source_shas))
    block_count = next(iter(dimensions))
    seeds = next(iter(seed_sets))
    objective = next(iter(objectives))
    representation = next(iter(representations))
    reference_design: dict[int, tuple[str, str, str]] | None = None
    for epsilon, frame in frames.items():
        if tuple(sorted(frame["frozen_design_seed"].astype(int))) != seeds:
            raise ValueError(f"epsilon={epsilon:g} changed the frozen seed family")
        if set(frame["source_git_sha"].astype(str)) != {source_sha}:
            raise ValueError(f"epsilon={epsilon:g} changed source SHA")
        if set(frame["L"].astype(int)) != {block_count}:
            raise ValueError(f"epsilon={epsilon:g} changed L")
        if set(frame["utility_objective"].astype(str)) != {objective} or set(
            frame["representation_mode"].astype(str)
        ) != {representation}:
            raise ValueError(f"epsilon={epsilon:g} changed the utility design")
        design = {
            int(row.frozen_design_seed): (
                str(row.encoder_sha256),
                str(row.assignment_hash),
                str(row.decoder_hash),
            )
            for row in frame.itertuples()
        }
        if reference_design is None:
            reference_design = design
        elif design != reference_design:
            raise ValueError("encoder, partition, or decoder hashes differ across epsilon")

    combined = pd.concat([frames[value] for value in sorted(frames)], ignore_index=True)
    combined["design_excess_reduction_percent"] = (
        100 * combined["relative_excess_capt_reduction_vs_context_ldp"]
    )
    combined["test_logloss_improvement_micro"] = (
        -1e6 * combined["test_capt_minus_ldp_expected_randomized_log_loss"]
    )
    combined["test_roc_auc_improvement_milli"] = 1e3 * combined["test_capt_minus_ldp_ROC_AUC"]
    combined["test_pr_auc_improvement_milli"] = 1e3 * combined["test_capt_minus_ldp_PR_AUC"]
    combined["test_ece_improvement_milli"] = -1e3 * combined["test_capt_minus_ldp_ECE"]
    combined["ldp_row_tv_ceiling"] = np.tanh(combined["epsilon"].to_numpy(float) / 2)
    combined["max_row_tv_above_ldp_ceiling"] = (
        combined["max_capt_row_tv"] - combined["ldp_row_tv_ceiling"]
    )
    info = {
        "experiment": "criteo_context_epsilon_grid",
        "version": _GRID_VERSION,
        "source_git_sha": source_sha,
        "L": block_count,
        "epsilon_values": sorted(frames),
        "frozen_design_seeds": list(seeds),
        "context_utility_objective": objective,
        "context_representation_mode": representation,
        "summary_directories": {
            format(epsilon, ".12g"): str(Path(frames[epsilon]["epsilon_summary_path"].iloc[0]))
            for epsilon in sorted(frames)
        },
    }
    return combined, info


def _seed_lines(
    axis,
    frame: pd.DataFrame,
    metric: str,
    *,
    zero_line: bool = False,
    color: str = "#2563A6",
    label: str = "seed mean",
) -> None:
    seeds = sorted(frame["frozen_design_seed"].astype(int).unique())
    epsilon_values = sorted(frame["epsilon"].astype(float).unique())
    for seed in seeds:
        group = frame.loc[frame["frozen_design_seed"] == seed].sort_values("epsilon")
        axis.plot(group["epsilon"], group[metric], color=color, lw=0.8, alpha=0.24)
        axis.scatter(group["epsilon"], group[metric], color=color, s=18, alpha=0.45)
    means = frame.groupby("epsilon", sort=True)[metric].mean().reindex(epsilon_values)
    axis.plot(epsilon_values, means, color=color, marker="o", lw=2.2, label=label)
    if zero_line:
        axis.axhline(0, color="#6B7280", lw=1, ls="--")
    axis.set_xticks(epsilon_values, [f"{value:g}" for value in epsilon_values])
    axis.set_xlabel("Privacy budget ε")
    axis.grid(alpha=0.2)


def _plot_grid(frame: pd.DataFrame, info: dict[str, Any], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.8))
    _seed_lines(
        axes[0, 0],
        frame,
        "design_excess_reduction_percent",
        zero_line=True,
        color="#111827",
    )
    axes[0, 0].set_ylabel("LDP − CAPT excess reduction (%)")
    axes[0, 0].set_title("A. D_design objective advantage", loc="left")

    _seed_lines(
        axes[0, 1], frame, "test_logloss_improvement_micro", zero_line=True, color="#111827"
    )
    axes[0, 1].set_ylabel(r"$(LDP-CAPT)\times10^6$ nats/display")
    axes[0, 1].set_title("B. D_test expected log-loss improvement", loc="left")

    _seed_lines(axes[0, 2], frame, "max_capt_row_tv", label="CAPT seed mean")
    ceilings = np.tanh(np.asarray(info["epsilon_values"], dtype=float) / 2)
    axes[0, 2].plot(
        info["epsilon_values"], ceilings, color="#D97706", ls="--", lw=1.8, label="ε-LDP ceiling"
    )
    axes[0, 2].set_ylabel("Maximum pairwise row TV")
    axes[0, 2].set_title("C. Non-LDP channel region", loc="left")
    axes[0, 2].legend(frameon=False, fontsize=8)

    _seed_lines(
        axes[1, 0],
        frame,
        "strict_advantage_context_mass",
        label="strict advantage",
    )
    _seed_lines(
        axes[1, 0],
        frame,
        "ldp_degraded_context_mass",
        color="#D97706",
        label="LDP-degraded",
    )
    axes[1, 0].set_ylabel("D_design context mass")
    axes[1, 0].set_title("D. Advantage and LDP-degraded mass", loc="left")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="best")

    _seed_lines(
        axes[1, 1],
        frame,
        "test_roc_auc_improvement_milli",
        zero_line=True,
        label="ROC-AUC mean",
    )
    _seed_lines(
        axes[1, 1],
        frame,
        "test_pr_auc_improvement_milli",
        color="#D97706",
        label="PR-AUC mean",
    )
    axes[1, 1].set_ylabel(r"$(CAPT-LDP)\times10^3$")
    axes[1, 1].set_title("E. D_test ranking metrics", loc="left")
    axes[1, 1].legend(frameon=False, fontsize=8)

    frame = frame.copy()
    frame["certificate_epsilon_slack"] = (
        frame["epsilon"] - frame["conservative_max_realized_epsilon"]
    )
    _seed_lines(
        axes[1, 2],
        frame,
        "certificate_epsilon_slack",
        zero_line=True,
        color="#111827",
        label="target − realized",
    )
    axes[1, 2].set_ylabel("Certificate ε slack")
    axes[1, 2].set_title("F. Conservative certificate check", loc="left")
    axes[1, 2].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Criteo public-context CAPT epsilon grid; "
        f"L={info['L']}; objective={info['context_utility_objective']}; "
        f"representation={info['context_representation_mode']}",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Thin paths are paired frozen-design seeds; black paths are seed means. No seed-level confidence interval is implied.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.91, wspace=0.30, hspace=0.34)
    fixed_time = datetime(2000, 1, 1, tzinfo=UTC)
    fig.savefig(
        output_dir / "figures" / "context_epsilon_grid.pdf",
        metadata={
            "Creator": "CAPT-12",
            "Producer": "CAPT-12",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        output_dir / "figures" / "context_epsilon_grid.png",
        dpi=220,
        metadata={"Software": "CAPT-12"},
    )
    plt.close(fig)


def _summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "design_excess_reduction_percent",
        "test_logloss_improvement_micro",
        "test_roc_auc_improvement_milli",
        "test_pr_auc_improvement_milli",
        "test_ece_improvement_milli",
        "mass_weighted_capt_row_tv",
        "max_capt_row_tv",
        "max_row_tv_above_ldp_ceiling",
        "strict_advantage_context_mass",
        "ldp_degraded_context_mass",
        "conservative_max_realized_epsilon",
        "conservative_max_additive_violation",
    ]
    rows: list[dict[str, Any]] = []
    for epsilon, group in frame.groupby("epsilon", sort=True):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _write_report(frame: pd.DataFrame, info: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Public-context CAPT epsilon grid",
        "",
        "## Fixed scope",
        "",
        f"- Source SHA: `{info['source_git_sha']}`.",
        f"- Epsilon values: {', '.join(format(value, '.12g') for value in info['epsilon_values'])}; L={info['L']}; seeds: {', '.join(map(str, info['frozen_design_seeds']))}.",
        f"- Objective: `{info['context_utility_objective']}`; representation: `{info['context_representation_mode']}`.",
        "- Temporal splits, encoder for each paired seed, partition, decoder, support, adjacency, confidence construction, and certificate policy are identical across epsilon.",
        "",
        "## Paired results",
        "",
        "| epsilon | excess improvement mean [min, max] | log-loss wins | log-loss improvement mean (×1e6) | max-TV above LDP ceiling (max) | strict mass mean | LDP-degraded mass mean | certificates | constraints |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epsilon, group in frame.groupby("epsilon", sort=True):
        reduction = group["design_excess_reduction_percent"]
        logloss = group["test_logloss_improvement_micro"]
        lines.append(
            f"| {epsilon:g} | {reduction.mean():.6g}% [{reduction.min():.6g}, {reduction.max():.6g}] | "
            f"{int((logloss > 0).sum())}/{len(group)} | {logloss.mean():.6g} | "
            f"{group['max_row_tv_above_ldp_ceiling'].max():.6g} | "
            f"{group['strict_advantage_context_mass'].mean():.6g} | "
            f"{group['ldp_degraded_context_mass'].mean():.6g} | "
            f"{int(group['certificate_count'].sum())} | "
            f"{int(group['certificate_checked_constraints'].sum()):,} |"
        )
    lines.extend(
        [
            "",
            "## Verification and interpretation",
            "",
            f"- All {int(frame['certificate_count'].sum())} certificates are valid; {int(frame['certificate_checked_constraints'].sum()):,} constraints were rechecked.",
            f"- Maximum conservative additive violation: {frame['conservative_max_additive_violation'].max():.9g}.",
            "- A positive maximum-row-TV margin above `tanh(epsilon/2)` proves that at least one released context channel lies outside the corresponding epsilon-LDP feasible set; it does not by itself prove held-out CTR improvement.",
            "- D_test metric differences are paired diagnostics. Frozen-design seeds are design perturbations, not independent test samples, so the figure reports points and ranges rather than a seed-level confidence interval.",
            "- Each epsilon/seed certificate is its own simultaneous context family. The grid is not claimed as one joint 95% family across all epsilon values and seeds.",
            "- Epsilon must be positive: the deterministic full-support repair gets strict numerical slack from `(exp(epsilon)-1)/L`; epsilon=0 requires a separate exact-equality mechanism and is intentionally rejected.",
        ]
    )
    (output_dir / "context_epsilon_grid_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_bundle(output_dir: Path, info: dict[str, Any]) -> Path:
    bundle_path = output_dir / "sol_context_epsilon_grid_bundle.zip"
    temporary_path = output_dir / ".sol_context_epsilon_grid_bundle.zip.tmp"
    external_manifest_path = output_dir / "sol_context_epsilon_grid_bundle_manifest.json"
    members: list[tuple[str, bytes]] = []
    for candidate in sorted(output_dir.rglob("*")):
        if candidate.is_file() and candidate not in {
            bundle_path,
            temporary_path,
            external_manifest_path,
        }:
            members.append(
                (f"grid/{candidate.relative_to(output_dir).as_posix()}", candidate.read_bytes())
            )
    for epsilon_text, summary_text in sorted(info["summary_directories"].items(), key=lambda item: float(item[0])):
        summary = Path(summary_text)
        members.append(
            (
                f"epsilon/{epsilon_text}/sol_seed_stability_review_bundle.zip",
                (summary / "sol_seed_stability_review_bundle.zip").read_bytes(),
            )
        )
        members.append(
            (
                f"epsilon/{epsilon_text}/sol_seed_stability_review_bundle_manifest.json",
                (summary / "sol_seed_stability_review_bundle_manifest.json").read_bytes(),
            )
        )
    manifest = {
        **info,
        "files": [
            {"path": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in members
        ],
    }
    members.append(
        (
            "sol_context_epsilon_grid_bundle_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
    )
    fixed_timestamp = (2000, 1, 1, 0, 0, 0)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, payload in sorted(members):
                item = zipfile.ZipInfo(name, fixed_timestamp)
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = 0o100644 << 16
                archive.writestr(item, payload)
        temporary_path.replace(bundle_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return bundle_path


def run_context_epsilon_grid(
    config: dict[str, Any],
    seeds: list[int],
    epsilon_values: list[float],
    *,
    output_root: Path = Path("outputs/context_epsilon_grids"),
) -> Path:
    """Run/reuse matched seed families over positive epsilon and aggregate them."""
    started = time.perf_counter()
    epsilons = sorted(set(map(float, epsilon_values)))
    if len(epsilons) < 2:
        raise ValueError("epsilon grid requires at least two distinct budgets")
    if any(not math.isfinite(value) or value <= 0 for value in epsilons):
        raise ValueError("epsilon grid values must be finite and strictly positive")
    seeds = sorted(set(map(int, seeds)))
    if len(seeds) < 2 or any(seed < 0 for seed in seeds):
        raise ValueError("epsilon grid requires at least two nonnegative frozen design seeds")
    base = record_source_provenance(validate_config(config))
    summaries: list[Path] = []
    for index, epsilon in enumerate(epsilons, start=1):
        print(
            "[context_epsilon_grid] "
            f"epsilon_started epsilon={epsilon:g} index={index}/{len(epsilons)} "
            f"seeds={','.join(map(str, seeds))}",
            flush=True,
        )
        summary = run_context_seed_stability({**base, "epsilon": epsilon}, seeds)
        summaries.append(summary)
        print(
            f"[context_epsilon_grid] epsilon_finished epsilon={epsilon:g} summary={summary}",
            flush=True,
        )
    frame, info = _load_epsilon_summaries(summaries)
    signature = {
        "experiment": "criteo_context_epsilon_grid",
        "version": _GRID_VERSION,
        "source_git_sha": info["source_git_sha"],
        "epsilon_values": info["epsilon_values"],
        "L": info["L"],
        "seeds": info["frozen_design_seeds"],
        "objective": info["context_utility_objective"],
        "representation": info["context_representation_mode"],
        "summary_ids": [path.name for path in summaries],
    }
    output_dir = output_root / run_id(signature)
    for directory in (output_dir, output_dir / "tables", output_dir / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "tables" / "epsilon_seed_results.csv", index=False)
    _summary_table(frame).to_csv(output_dir / "tables" / "epsilon_summary.csv", index=False)
    info["wall_seconds_including_new_runs"] = time.perf_counter() - started
    (output_dir / "context_epsilon_grid_metadata.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot_grid(frame, info, output_dir)
    _write_report(frame, info, output_dir)
    bundle = _write_bundle(output_dir, info)
    (output_dir / "sol_context_epsilon_grid_bundle_manifest.json").write_text(
        json.dumps(
            {
                **info,
                "bundle": bundle.name,
                "bundle_size_bytes": bundle.stat().st_size,
                "bundle_sha256": sha256_file(bundle),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "[context_epsilon_grid] "
        f"finished output={output_dir} bundle={bundle} seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return output_dir
