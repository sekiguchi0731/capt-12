from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capt12.experiments.context_cost_comparison import (
    _LABELS,
    _OBJECTIVES,
    _seed_positions,
)
from capt12.utils.artifacts import git_sha, sha256_file

_BLUE = "#2563A6"
_GREY = "#6B7280"
_VERSION = 1

_METRICS = (
    (
        "global_ldp_minus_capt_logloss_micro",
        "A. Held-out expected log loss",
        r"$(\mathrm{Global\ LDP}-\mathrm{CAPT})\times10^6$ nats/display",
    ),
    (
        "capt_minus_global_ldp_roc_auc_milli",
        "B. Held-out ROC-AUC",
        r"$(\mathrm{CAPT}-\mathrm{Global\ LDP})\times10^3$",
    ),
    (
        "capt_minus_global_ldp_pr_auc_milli",
        "C. Held-out PR-AUC",
        r"$(\mathrm{CAPT}-\mathrm{Global\ LDP})\times10^3$",
    ),
    (
        "global_ldp_minus_capt_ece_milli",
        "D. Held-out calibration error",
        r"$(\mathrm{Global\ LDP}-\mathrm{CAPT})\times10^3$",
    ),
)


def _load_and_join(
    cost_comparison_dir: Path,
    global_comparison_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cost_table = cost_comparison_dir / "tables" / "objective_seed_results.csv"
    cost_metadata_path = cost_comparison_dir / "context_cost_comparison_metadata.json"
    cost_bundle = cost_comparison_dir / "sol_context_cost_comparison_bundle.zip"
    global_table = global_comparison_dir / "tables" / "global_token_ldp_results.csv"
    global_metadata_path = global_comparison_dir / "global_token_ldp_metadata.json"
    global_bundle = global_comparison_dir / "sol_global_token_ldp_comparison_bundle.zip"
    for required in (
        cost_table,
        cost_metadata_path,
        cost_bundle,
        global_table,
        global_metadata_path,
        global_bundle,
    ):
        if not required.is_file():
            raise ValueError(f"missing comparison input: {required}")

    cost_metadata = json.loads(cost_metadata_path.read_text(encoding="utf-8"))
    global_metadata = json.loads(global_metadata_path.read_text(encoding="utf-8"))
    if not bool(global_metadata.get("all_global_ldp_checks_valid")):
        raise ValueError("global token-LDP input did not pass its direct checks")
    cost = pd.read_csv(cost_table)
    global_ldp = pd.read_csv(global_table)
    epsilon = float(cost_metadata["epsilon"])
    global_ldp = global_ldp.loc[np.isclose(global_ldp["epsilon"], epsilon)].copy()
    if len(global_ldp) != len(cost_metadata["frozen_design_seeds"]):
        raise ValueError("global token-LDP table is incomplete for the paired seeds")
    if not np.isfinite(global_ldp["realized_epsilon"]).all():
        raise ValueError("global token-LDP contains a non-finite realized epsilon")
    if not (global_ldp["realized_epsilon"] <= global_ldp["epsilon"]).all():
        raise ValueError("global token-LDP exceeds its target epsilon")
    if not (global_ldp["max_additive_violation"] <= 0).all():
        raise ValueError("global token-LDP has a positive additive violation")

    baseline_columns = {
        "expected_randomized_log_loss": "global_ldp_expected_randomized_log_loss",
        "ROC_AUC": "global_ldp_ROC_AUC",
        "PR_AUC": "global_ldp_PR_AUC",
        "ECE": "global_ldp_ECE",
        "realized_epsilon": "global_ldp_realized_epsilon",
        "max_additive_violation": "global_ldp_max_additive_violation",
        "channel_sha256": "global_ldp_channel_sha256",
        "encoder_sha256": "global_ldp_encoder_sha256",
    }
    baseline = global_ldp[
        ["frozen_design_seed", "epsilon", *baseline_columns]
    ].rename(columns=baseline_columns)
    joined = cost.merge(
        baseline,
        on=["frozen_design_seed", "epsilon"],
        how="inner",
        validate="many_to_one",
    )
    if len(joined) != len(cost):
        raise ValueError("global token-LDP baseline is incomplete for the cost comparison")
    if not (
        joined["encoder_sha256"].astype(str)
        == joined["global_ldp_encoder_sha256"].astype(str)
    ).all():
        raise ValueError("CAPT and global token-LDP encoder hashes do not match by seed")
    if set(joined["utility_objective"].astype(str)) != set(_OBJECTIVES):
        raise ValueError("cost comparison does not contain all three aligned objectives")

    joined["global_ldp_minus_capt_logloss_micro"] = 1e6 * (
        joined["global_ldp_expected_randomized_log_loss"]
        - joined["test_context_capt_expected_randomized_log_loss"]
    )
    joined["capt_minus_global_ldp_roc_auc_milli"] = 1e3 * (
        joined["test_context_capt_ROC_AUC"] - joined["global_ldp_ROC_AUC"]
    )
    joined["capt_minus_global_ldp_pr_auc_milli"] = 1e3 * (
        joined["test_context_capt_PR_AUC"] - joined["global_ldp_PR_AUC"]
    )
    joined["global_ldp_minus_capt_ece_milli"] = 1e3 * (
        joined["global_ldp_ECE"] - joined["test_context_capt_ECE"]
    )
    info = {
        "version": _VERSION,
        "analysis_git_sha": git_sha(),
        "L": int(cost_metadata["L"]),
        "epsilon": epsilon,
        "frozen_design_seeds": list(map(int, cost_metadata["frozen_design_seeds"])),
        "objectives": list(_OBJECTIVES),
        "comparison_definition": (
            "Held-out utility of each objective-aligned context CAPT channel versus the "
            "same-seed unrestricted global K=64 singleton/identity optimal-LDP channel."
        ),
        "global_ldp_optimization_objective": "empirical_logloss",
        "cost_comparison_dir": str(cost_comparison_dir.resolve()),
        "global_comparison_dir": str(global_comparison_dir.resolve()),
        "cost_bundle_sha256": sha256_file(cost_bundle),
        "global_bundle_sha256": sha256_file(global_bundle),
    }
    return joined, info


def _plot(frame: pd.DataFrame, info: dict[str, Any], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.3), constrained_layout=False)
    for axis, (metric, title, ylabel) in zip(axes.flat, _METRICS, strict=True):
        for index, objective in enumerate(_OBJECTIVES):
            group = frame.loc[frame["utility_objective"] == objective].sort_values(
                "frozen_design_seed"
            )
            values = group[metric].to_numpy(float)
            positions = index + _seed_positions(len(values))
            axis.scatter(
                positions,
                values,
                s=42,
                facecolor="white",
                edgecolor=_BLUE,
                linewidth=1.1,
                zorder=3,
            )
            axis.hlines(
                values.mean(),
                index - 0.28,
                index + 0.28,
                color=_BLUE,
                lw=2.4,
                zorder=4,
            )
        axis.axhline(0, color=_GREY, lw=1.1, ls="--")
        axis.set_xticks(range(3), [_LABELS[item] for item in _OBJECTIVES])
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E5E7EB", lw=0.65)
    fig.suptitle(
        "Criteo CAPT versus global token-level optimal LDP\n"
        rf"($\epsilon={info['epsilon']:g}$, $L={info['L']}$; paired frozen-design seeds)",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "Positive values favor CAPT. Open circles are paired seeds; blue ticks are seed means. "
        "The global K=64 LDP channel is optimized for empirical log loss.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.88, wspace=0.27, hspace=0.38)
    figure_dir = output_dir / "figures"
    fixed_time = datetime(2000, 1, 1, tzinfo=UTC)
    fig.savefig(
        figure_dir / "context_cost_comparison_vs_global_token_ldp.pdf",
        metadata={
            "Creator": "CAPT-12",
            "Producer": "CAPT-12",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        figure_dir / "context_cost_comparison_vs_global_token_ldp.png",
        dpi=220,
        metadata={"Software": "CAPT-12"},
    )
    plt.close(fig)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for objective in _OBJECTIVES:
        group = frame.loc[frame["utility_objective"] == objective]
        for metric, _, _ in _METRICS:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "utility_objective": objective,
                    "metric": metric,
                    "seed_count": len(values),
                    "wins": int((values > 0).sum()),
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def _write_report(frame: pd.DataFrame, info: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Cost comparison versus global token-level optimal LDP",
        "",
        info["comparison_definition"],
        "",
        f"- epsilon={info['epsilon']:g}, CAPT L={info['L']}, global LDP K=64.",
        "- Positive reported differences favor CAPT.",
        "- The global channel is optimized for empirical log loss; teacher and hybrid rows are held-out utility comparisons, not matched design-objective comparisons.",
        "- The original matched block/context-LDP figure and bundle are preserved unchanged.",
        "",
        "| objective | log-loss mean [min, max] (micro-nats) | wins | ROC-AUC mean (milli) | PR-AUC mean (milli) | ECE improvement mean (milli) |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for objective in _OBJECTIVES:
        group = frame.loc[frame["utility_objective"] == objective]
        logloss = group["global_ldp_minus_capt_logloss_micro"]
        lines.append(
            f"| `{objective}` | {logloss.mean():.6g} [{logloss.min():.6g}, {logloss.max():.6g}] | "
            f"{int((logloss > 0).sum())}/{len(group)} | "
            f"{group['capt_minus_global_ldp_roc_auc_milli'].mean():.6g} | "
            f"{group['capt_minus_global_ldp_pr_auc_milli'].mean():.6g} | "
            f"{group['global_ldp_minus_capt_ece_milli'].mean():.6g} |"
        )
    (output_dir / "context_cost_comparison_vs_global_token_ldp_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_bundle(output_dir: Path, global_comparison_dir: Path, info: dict[str, Any]) -> Path:
    bundle = output_dir / "sol_context_cost_global_token_ldp_comparison_bundle.zip"
    temporary = output_dir / ".sol_context_cost_global_token_ldp_comparison_bundle.zip.tmp"
    external_manifest = (
        output_dir / "sol_context_cost_global_token_ldp_comparison_bundle_manifest.json"
    )
    generated = [
        output_dir / "figures" / "context_cost_comparison_vs_global_token_ldp.pdf",
        output_dir / "figures" / "context_cost_comparison_vs_global_token_ldp.png",
        output_dir / "tables" / "objective_seed_results_vs_global_token_ldp.csv",
        output_dir / "tables" / "objective_summary_vs_global_token_ldp.csv",
        output_dir / "context_cost_comparison_vs_global_token_ldp_metadata.json",
        output_dir / "context_cost_comparison_vs_global_token_ldp_report.md",
    ]
    payloads = [
        (path.relative_to(output_dir).as_posix(), path.read_bytes()) for path in generated
    ]
    payloads.extend(
        [
            (
                "inputs/sol_context_cost_comparison_bundle.zip",
                (output_dir / "sol_context_cost_comparison_bundle.zip").read_bytes(),
            ),
            (
                "inputs/sol_global_token_ldp_comparison_bundle.zip",
                (
                    global_comparison_dir / "sol_global_token_ldp_comparison_bundle.zip"
                ).read_bytes(),
            ),
        ]
    )
    manifest = {
        **info,
        "files": [
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in payloads
        ],
    }
    payloads.append(
        (
            "sol_context_cost_global_token_ldp_comparison_bundle_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
        )
    )
    fixed_timestamp = (2000, 1, 1, 0, 0, 0)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, payload in sorted(payloads):
                item = zipfile.ZipInfo(name, fixed_timestamp)
                item.compress_type = zipfile.ZIP_DEFLATED
                item.external_attr = 0o100644 << 16
                archive.writestr(item, payload)
        temporary.replace(bundle)
    finally:
        temporary.unlink(missing_ok=True)
    external_manifest.write_text(
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
    return bundle


def add_global_ldp_figures(
    cost_comparison_dirs: list[Path],
    global_comparison_dir: Path,
) -> list[Path]:
    if not cost_comparison_dirs:
        raise ValueError("at least one cost-comparison directory is required")
    bundles: list[Path] = []
    for output_dir in cost_comparison_dirs:
        frame, info = _load_and_join(output_dir, global_comparison_dir)
        frame.to_csv(
            output_dir / "tables" / "objective_seed_results_vs_global_token_ldp.csv",
            index=False,
        )
        _summary(frame).to_csv(
            output_dir / "tables" / "objective_summary_vs_global_token_ldp.csv",
            index=False,
        )
        (output_dir / "context_cost_comparison_vs_global_token_ldp_metadata.json").write_text(
            json.dumps(info, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _plot(frame, info, output_dir)
        _write_report(frame, info, output_dir)
        bundles.append(_write_bundle(output_dir, global_comparison_dir, info))
    return bundles
