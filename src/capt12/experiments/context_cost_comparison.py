from __future__ import annotations

import hashlib
import json
import math
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capt12.config import canonical_json
from capt12.experiments.context_seed_stability import run_context_seed_stability
from capt12.utils.artifacts import git_sha, sha256_file

_OBJECTIVES = ("teacher_kl", "empirical_logloss", "hybrid_logloss_kl")
_LABELS = {
    "teacher_kl": "Teacher KL",
    "empirical_logloss": "Empirical\nlog loss",
    "hybrid_logloss_kl": "Hybrid\nlog loss + KL",
}
_VERSION = 2


def _boolean_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"invalid boolean values in {series.name}")
    return normalized == "true"


def _load_inputs(summary_dirs: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(summary_dirs) != len(_OBJECTIVES):
        raise ValueError("cost comparison requires exactly three seed-summary directories")
    frames: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    normalized_configs: dict[str, str] = {}
    for raw_path in summary_dirs:
        path = raw_path.resolve()
        metadata_path = path / "context_seed_stability_metadata.json"
        results_path = path / "tables" / "seed_results.csv"
        config_path = path / "summary_config.json"
        bundle_path = path / "sol_seed_stability_review_bundle.zip"
        for required in (metadata_path, results_path, config_path, bundle_path):
            if not required.is_file():
                raise ValueError(f"incomplete seed summary: missing {required}")
        item = json.loads(metadata_path.read_text(encoding="utf-8"))
        objective = str(item.get("context_utility_objective"))
        if objective not in _OBJECTIVES or objective in frames:
            raise ValueError(f"duplicate or unsupported utility objective: {objective}")
        if item.get("context_representation_mode") != "objective_aligned":
            raise ValueError(f"{objective} is not an objective-aligned representation")
        if not bool(item.get("all_certificates_valid")):
            raise ValueError(f"{objective} contains an invalid certificate")
        frame = pd.read_csv(results_path)
        if set(frame["utility_objective"].astype(str)) != {objective}:
            raise ValueError(f"{objective} seed table has inconsistent objective labels")
        frame_epsilons = set(frame["epsilon"].astype(float))
        if len(frame_epsilons) != 1:
            raise ValueError(f"{objective} seed table mixes epsilon values")
        frame_epsilon = next(iter(frame_epsilons))
        if "epsilon" in item and float(item["epsilon"]) != frame_epsilon:
            raise ValueError(f"{objective} metadata and seed table disagree on epsilon")
        # Summary version 4 (before epsilon-grid orchestration) recorded epsilon
        # in every seed row but not in the summary metadata.  Recovering that
        # single validated value keeps already-certified L8/L16/L32 runs usable
        # without weakening any matching check.
        item["epsilon"] = frame_epsilon
        if set(frame["representation_mode"].astype(str)) != {"objective_aligned"}:
            raise ValueError(f"{objective} seed table is not objective-aligned")
        if not _boolean_values(frame["all_certificates_valid"]).all():
            raise ValueError(f"{objective} seed table contains invalid certificates")
        config = json.loads(config_path.read_text(encoding="utf-8"))["base_config"]
        config = dict(config)
        config.pop("context_utility_objective", None)
        normalized_configs[objective] = canonical_json(config)
        frame.insert(0, "summary_path", str(path))
        frames[objective] = frame
        metadata[objective] = item

    if set(frames) != set(_OBJECTIVES):
        missing = sorted(set(_OBJECTIVES) - set(frames))
        raise ValueError(f"missing utility objectives: {missing}")
    if len(set(normalized_configs.values())) != 1:
        raise ValueError("seed-summary configs differ beyond utility objective")

    source_shas = {str(item["source_git_sha"]) for item in metadata.values()}
    dimensions = {int(item["L"]) for item in metadata.values()}
    epsilons = {float(item["epsilon"]) for item in metadata.values()}
    seed_sets = {tuple(map(int, item["frozen_design_seeds"])) for item in metadata.values()}
    if (
        len(source_shas) != 1
        or len(dimensions) != 1
        or len(epsilons) != 1
        or len(seed_sets) != 1
    ):
        raise ValueError("source SHA, L, epsilon, and frozen seed set must match across objectives")
    source_sha = next(iter(source_shas))
    block_count = next(iter(dimensions))
    epsilon = next(iter(epsilons))
    seeds = next(iter(seed_sets))
    hybrid_weights = set(
        frames["hybrid_logloss_kl"]["hybrid_empirical_weight"].astype(float)
    )
    if len(hybrid_weights) != 1:
        raise ValueError("hybrid seed rows do not use one fixed empirical weight")
    hybrid_weight = next(iter(hybrid_weights))

    reference_encoders: dict[int, str] | None = None
    for objective in _OBJECTIVES:
        frame = frames[objective]
        if set(frame["source_git_sha"].astype(str)) != {source_sha}:
            raise ValueError(f"{objective} seed rows have inconsistent source SHA")
        if set(frame["L"].astype(int)) != {block_count} or set(
            frame["epsilon"].astype(float)
        ) != {epsilon}:
            raise ValueError(f"{objective} changed L or epsilon")
        if tuple(sorted(frame["frozen_design_seed"].astype(int))) != seeds:
            raise ValueError(f"{objective} changed the frozen seed family")
        encoders = {
            int(row.frozen_design_seed): str(row.encoder_sha256)
            for row in frame.itertuples()
        }
        if reference_encoders is None:
            reference_encoders = encoders
        elif encoders != reference_encoders:
            raise ValueError("encoder hashes differ across utility objectives")

    combined = pd.concat([frames[objective] for objective in _OBJECTIVES], ignore_index=True)
    combined["objective_label"] = combined["utility_objective"].map(_LABELS)
    combined["design_excess_reduction_percent"] = (
        100 * combined["relative_excess_capt_reduction_vs_context_ldp"]
    )
    combined["test_logloss_improvement_micro"] = (
        -1e6 * combined["test_capt_minus_ldp_expected_randomized_log_loss"]
    )
    combined["test_roc_auc_improvement_milli"] = 1e3 * combined["test_capt_minus_ldp_ROC_AUC"]
    combined["test_pr_auc_improvement_milli"] = 1e3 * combined["test_capt_minus_ldp_PR_AUC"]
    combined["test_ece_improvement_milli"] = -1e3 * combined["test_capt_minus_ldp_ECE"]
    info = {
        "version": _VERSION,
        "experiment_source_git_sha": source_sha,
        "analysis_git_sha": git_sha(),
        "L": block_count,
        "epsilon": epsilon,
        "frozen_design_seeds": list(seeds),
        "objectives": list(_OBJECTIVES),
        "hybrid_empirical_weight": hybrid_weight,
        "summary_directories": {
            objective: str(Path(frames[objective]["summary_path"].iloc[0]))
            for objective in _OBJECTIVES
        },
    }
    return combined, info


def _summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "design_excess_reduction_percent",
        "test_logloss_improvement_micro",
        "test_roc_auc_improvement_milli",
        "test_pr_auc_improvement_milli",
        "test_ece_improvement_milli",
        "mass_weighted_capt_row_tv",
        "max_capt_row_tv",
        "strict_advantage_context_mass",
        "ldp_degraded_context_mass",
        "lower_audit_context_capt_epsilon",
        "lower_audit_context_ldp_epsilon",
    ]
    rows = []
    for objective in _OBJECTIVES:
        group = frame.loc[frame["utility_objective"] == objective]
        for metric in metrics:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "utility_objective": objective,
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _seed_positions(seed_count: int) -> np.ndarray:
    return np.linspace(-0.18, 0.18, seed_count) if seed_count > 1 else np.zeros(1)


def _strip_panel(
    axis,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    *,
    zero_line: bool = False,
) -> None:
    blue = "#2563A6"
    for index, objective in enumerate(_OBJECTIVES):
        group = frame.loc[frame["utility_objective"] == objective].sort_values(
            "frozen_design_seed"
        )
        values = group[metric].to_numpy(float)
        positions = index + _seed_positions(len(values))
        axis.scatter(
            positions,
            values,
            s=38,
            color=blue,
            edgecolor="#1F2937",
            linewidth=0.7,
            zorder=3,
        )
        axis.hlines(values.mean(), index - 0.28, index + 0.28, color="#111827", lw=2.2)
    if zero_line:
        axis.axhline(0, color="#6B7280", lw=1, ls="--")
    axis.set_xticks(range(3), [_LABELS[item] for item in _OBJECTIVES])
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontsize=11)
    axis.grid(axis="y", alpha=0.2)


def _plot(frame: pd.DataFrame, info: dict[str, Any], output_dir: Path) -> None:
    blue = "#2563A6"
    orange = "#D97706"
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.8))
    _strip_panel(
        axes[0, 0],
        frame,
        "design_excess_reduction_percent",
        "A. D_design excess-objective improvement",
        "LDP − CAPT reduction (%)",
        zero_line=True,
    )
    _strip_panel(
        axes[0, 1],
        frame,
        "test_logloss_improvement_micro",
        "B. D_test expected log-loss improvement",
        r"$(LDP-CAPT)\times10^6$ nats/display",
        zero_line=True,
    )

    axis = axes[0, 2]
    for index, objective in enumerate(_OBJECTIVES):
        group = frame.loc[frame["utility_objective"] == objective].sort_values(
            "frozen_design_seed"
        )
        offsets = _seed_positions(len(group))
        for metric, shift, color, marker, label in [
            ("test_roc_auc_improvement_milli", -0.12, blue, "o", "ROC-AUC"),
            ("test_pr_auc_improvement_milli", 0.12, orange, "^", "PR-AUC"),
        ]:
            values = group[metric].to_numpy(float)
            axis.scatter(
                index + shift + 0.45 * offsets,
                values,
                s=34,
                color=color,
                marker=marker,
                edgecolor="#1F2937",
                linewidth=0.6,
                label=label if index == 0 else None,
                zorder=3,
            )
            axis.hlines(values.mean(), index + shift - 0.1, index + shift + 0.1, color="#111827")
    axis.axhline(0, color="#6B7280", lw=1, ls="--")
    axis.set_xticks(range(3), [_LABELS[item] for item in _OBJECTIVES])
    axis.set_ylabel(r"$(CAPT-LDP)\times10^3$")
    axis.set_title("C. D_test ranking-metric improvement", loc="left", fontsize=11)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 0]
    for index, objective in enumerate(_OBJECTIVES):
        group = frame.loc[frame["utility_objective"] == objective].sort_values(
            "frozen_design_seed"
        )
        offsets = _seed_positions(len(group))
        for metric, marker, face, label in [
            ("mass_weighted_capt_row_tv", "o", blue, "mass-weighted"),
            ("max_capt_row_tv", "^", "white", "maximum context"),
        ]:
            values = group[metric].to_numpy(float)
            axis.scatter(
                index + offsets,
                values,
                s=38,
                marker=marker,
                facecolor=face,
                edgecolor=blue,
                linewidth=1,
                label=label if index == 0 else None,
                zorder=3,
            )
    axis.axhline(
        math.tanh(float(info["epsilon"]) / 2),
        color=orange,
        lw=1.6,
        ls="--",
        label="ε-LDP ceiling",
    )
    axis.set_xticks(range(3), [_LABELS[item] for item in _OBJECTIVES])
    axis.set_ylabel("Maximum pairwise row TV")
    axis.set_title("D. CAPT channel row variation", loc="left", fontsize=11)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    _strip_panel(
        axes[1, 1],
        frame,
        "strict_advantage_context_mass",
        "E. Context mass with strict CAPT advantage",
        "D_design context mass",
    )

    axis = axes[1, 2]
    for index, objective in enumerate(_OBJECTIVES):
        group = frame.loc[frame["utility_objective"] == objective].sort_values(
            "frozen_design_seed"
        )
        offsets = _seed_positions(len(group))
        for metric, shift, color, marker, label in [
            ("lower_audit_context_capt_epsilon", -0.12, blue, "o", "CAPT lower"),
            ("lower_audit_context_ldp_epsilon", 0.12, orange, "^", "LDP lower"),
        ]:
            values = group[metric].to_numpy(float)
            axis.scatter(
                index + shift + 0.45 * offsets,
                values,
                s=34,
                color=color,
                marker=marker,
                edgecolor="#1F2937",
                linewidth=0.6,
                label=label if index == 0 else None,
            )
    axis.set_xticks(range(3), [_LABELS[item] for item in _OBJECTIVES])
    axis.set_ylabel("privacy lower ε")
    axis.set_title("F. D_test privacy lower audit", loc="left", fontsize=11)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Criteo public-context CAPT: objective comparison (ε={info['epsilon']:g}, L={info['L']})",
        fontsize=14,
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "Dots are the same frozen-design seeds; black ticks are means. Positive panels A–C favor CAPT. Raw objectives are not compared across costs.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.92, wspace=0.30, hspace=0.34)
    fixed_time = datetime(2000, 1, 1, tzinfo=UTC)
    fig.savefig(
        output_dir / "figures" / "context_cost_comparison.pdf",
        metadata={
            "Creator": "CAPT-12",
            "Producer": "CAPT-12",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        output_dir / "figures" / "context_cost_comparison.png",
        dpi=220,
        metadata={"Software": "CAPT-12"},
    )
    plt.close(fig)


def _write_report(frame: pd.DataFrame, info: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Objective-aligned cost comparison",
        "",
        "## Validated scope",
        "",
        f"- Experiment source SHA: `{info['experiment_source_git_sha']}`.",
        f"- Comparison-code SHA: `{info['analysis_git_sha']}`.",
        f"- epsilon={info['epsilon']:g}, L={info['L']}, frozen-design seeds: {', '.join(map(str, info['frozen_design_seeds']))}.",
        f"- Teacher KL, empirical log loss, and hybrid (empirical weight {info['hybrid_empirical_weight']:.6g}) each use objective-aligned partition, decoder, and R cost.",
        "- All non-objective config fields, seed-specific encoders, temporal splits, support, adjacency, and privacy definitions match.",
        "",
        "## Paired results",
        "",
        "| objective | excess improvement mean [min, max] | log-loss wins | log-loss improvement mean (×1e6) | ROC-AUC Δ mean (×1e3) | PR-AUC Δ mean (×1e3) | strict mass mean | maximum row TV | privacy lower max (CAPT/LDP) | certificates |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for objective in _OBJECTIVES:
        group = frame.loc[frame["utility_objective"] == objective]
        reduction = group["design_excess_reduction_percent"]
        logloss = group["test_logloss_improvement_micro"]
        certificate_count = int(group["certificate_count"].sum())
        lines.append(
            f"| `{objective}` | {reduction.mean():.4g}% [{reduction.min():.4g}, {reduction.max():.4g}] | "
            f"{int((logloss > 0).sum())}/{len(group)} | {logloss.mean():.6g} | "
            f"{group['test_roc_auc_improvement_milli'].mean():.6g} | "
            f"{group['test_pr_auc_improvement_milli'].mean():.6g} | "
            f"{group['strict_advantage_context_mass'].mean():.6g} | "
            f"{group['max_capt_row_tv'].max():.6g} | "
            f"{group['lower_audit_context_capt_epsilon'].max():.6g} / "
            f"{group['lower_audit_context_ldp_epsilon'].max():.6g} | {certificate_count} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Raw objective values are not compared across costs. Panel A subtracts each objective's channel-invariant floor before within-cost CAPT-versus-LDP normalization.",
            "- D_test log loss, ROC-AUC, PR-AUC, and ECE are evaluated on the same held-out rows. The frozen-design seeds change the encoder/design realization; they are not independent test samples and no seed-level confidence interval is claimed.",
            "- Each seed's lower audit is a separate simultaneous family. Without a D_cert-to-D_test population bridge it is not combined with the certificate upper into a sandwich.",
            "- The comparison validates mechanism-level and held-out utility differences; choosing a final objective after inspecting D_test is exploratory rather than confirmatory.",
        ]
    )
    (output_dir / "context_cost_comparison_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_bundle(output_dir: Path, info: dict[str, Any]) -> Path:
    bundle_path = output_dir / "sol_context_cost_comparison_bundle.zip"
    temporary_path = output_dir / ".sol_context_cost_comparison_bundle.zip.tmp"
    external_manifest_path = output_dir / "sol_context_cost_comparison_bundle_manifest.json"
    members: list[tuple[str, bytes]] = []
    for candidate in sorted(output_dir.rglob("*")):
        if candidate.is_file() and candidate not in {
            bundle_path,
            temporary_path,
            external_manifest_path,
        }:
            members.append((f"comparison/{candidate.relative_to(output_dir).as_posix()}", candidate.read_bytes()))
    for objective in _OBJECTIVES:
        summary_dir = Path(info["summary_directories"][objective])
        with zipfile.ZipFile(summary_dir / "sol_seed_stability_review_bundle.zip") as source:
            for name in sorted(source.namelist()):
                if not name.endswith("/"):
                    members.append((f"objectives/{objective}/{name}", source.read(name)))
    manifest = {
        **info,
        "files": [
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in members
        ],
    }
    members.append(
        (
            "sol_context_cost_comparison_bundle_manifest.json",
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


def run_context_cost_comparison(
    summary_dirs: list[Path],
    output_root: Path = Path("outputs/context_cost_comparisons"),
) -> Path:
    frame, info = _load_inputs(summary_dirs)
    signature = json.dumps(
        {
            "version": _VERSION,
            "analysis_git_sha": info["analysis_git_sha"],
            "experiment_source_git_sha": info["experiment_source_git_sha"],
            "L": info["L"],
            "epsilon": info["epsilon"],
            "seeds": info["frozen_design_seeds"],
            "summary_ids": {
                objective: Path(path).name
                for objective, path in info["summary_directories"].items()
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    output_dir = output_root / hashlib.sha256(signature).hexdigest()[:16]
    for directory in (output_dir, output_dir / "tables", output_dir / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "tables" / "objective_seed_results.csv", index=False)
    _summary_table(frame).to_csv(output_dir / "tables" / "objective_summary.csv", index=False)
    (output_dir / "context_cost_comparison_metadata.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(frame, info, output_dir)
    _write_report(frame, info, output_dir)
    bundle = _write_bundle(output_dir, info)
    (output_dir / "sol_context_cost_comparison_bundle_manifest.json").write_text(
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
    return output_dir


def run_context_cost_stability(
    config: dict[str, Any],
    seeds: list[int],
    *,
    hybrid_empirical_weight: float = 0.5,
    output_root: Path = Path("outputs/context_cost_comparisons"),
) -> Path:
    """Run/reuse all aligned objectives and always finish with one bundle.

    Each objective remains an independently certified seed-stability family.
    The comparison is attempted only after all three families complete, so a
    partial experiment can never be mislabeled as a three-cost review bundle.
    """
    if not 0 <= hybrid_empirical_weight <= 1:
        raise ValueError("hybrid empirical weight must be in [0, 1]")
    summaries: list[Path] = []
    for objective in _OBJECTIVES:
        objective_config = {
            **config,
            "context_utility_objective": objective,
            "context_representation_mode": "objective_aligned",
            "hybrid_empirical_weight": hybrid_empirical_weight,
        }
        print(
            "[context_cost_stability] "
            f"objective_started objective={objective} seeds={','.join(map(str, seeds))}",
            flush=True,
        )
        summary = run_context_seed_stability(objective_config, seeds)
        summaries.append(summary)
        print(
            "[context_cost_stability] "
            f"objective_finished objective={objective} summary={summary}",
            flush=True,
        )
    comparison = run_context_cost_comparison(summaries, output_root)
    print(
        "[context_cost_stability] "
        f"comparison_finished comparison={comparison} "
        f"bundle={comparison / 'sol_context_cost_comparison_bundle.zip'}",
        flush=True,
    )
    return comparison
