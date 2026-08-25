from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capt12.comparison.contracts import MethodContract
from capt12.utils.artifacts import sha256_file

RESULT_COLUMNS = (
    "method",
    "display_name",
    "cost",
    "seed",
    "target_epsilon",
    "L",
    "mass_m",
    "mass_n",
    "mass_privacy_weight",
    "mass_utility_weight",
    "mass_temperature",
    "raw_upper_epsilon",
    "certified_upper_epsilon",
    "certificate_valid",
    "cover_lambda",
    "cover_row_tv",
    "constant_channel",
    "violating_constraint_count",
    "worst_witness",
    "empirical_attack_cmi_raw_nats",
    "empirical_attack_cmi_zero_clipped_nats",
    "empirical_attack_cmi_ci95_low_nats",
    "empirical_attack_cmi_ci95_high_nats",
    "lower_audit_epsilon",
    "empirical_log_loss",
    "excess_ctr_log_loss",
    "excess_ctr_log_loss_ci95_low",
    "excess_ctr_log_loss_ci95_high",
    "teacher_kl",
    "hybrid_objective",
    "roc_auc",
    "pr_auc",
    "distortion_objective",
    "paired_excess_log_loss_difference_vs_capt",
    "paired_difference_ci95_low",
    "paired_difference_ci95_high",
    "formal_comparable",
    "deployable_under_capt",
    "uses_sensitive_value_online",
    "oracle",
    "exclusion_reason",
    "channel_sha256",
    "certificate_path",
    "lower_audit_path",
)

_METHOD_COLORS = {
    "capt": "#2563A6",
    "optimal_ldp": "#D97706",
    "pbp_common_nominal_calibrated": "#7C3AED",
    "mass12_calibrated": "#059669",
    "mass12_raw": "#10B981",
    "pbp_oracle": "#9CA3AF",
    "best_constant_cover": "#4B5563",
}


def validate_results(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(RESULT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"prior-art results are missing columns: {missing}")
    result = frame.loc[:, RESULT_COLUMNS].copy()
    if len(result) == 0:
        raise ValueError("results.csv has no completed comparison rows")
    booleans = (
        "certificate_valid",
        "constant_channel",
        "formal_comparable",
        "deployable_under_capt",
        "uses_sensitive_value_online",
        "oracle",
    )
    for name in booleans:
        if not pd.api.types.is_bool_dtype(result[name]):
            normalized = result[name].astype(str).str.lower()
            if not normalized.isin({"true", "false"}).all():
                raise ValueError(f"{name} contains non-boolean values")
            result[name] = normalized == "true"
    invalid_formal = result["formal_comparable"] & ~result["certificate_valid"]
    if invalid_formal.any():
        raise ValueError("formal-comparable rows require valid certificates")
    leaked_oracle = result["uses_sensitive_value_online"] & result["deployable_under_capt"]
    if leaked_oracle.any():
        raise ValueError("rows using A_S online cannot be deployable under CAPT")
    return result.sort_values(["cost", "method", "seed", "target_epsilon"], kind="stable")


def summarize_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return median, IQR, and all seed values without hiding failed points."""
    results = validate_results(frame)
    metrics = (
        "excess_ctr_log_loss",
        "teacher_kl",
        "hybrid_objective",
        "roc_auc",
        "pr_auc",
        "distortion_objective",
        "certified_upper_epsilon",
    )
    rows = []
    keys = ["method", "display_name", "cost", "target_epsilon"]
    for group_key, group in results.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, group_key, strict=True))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "seed_count": len(values),
                    "median": float(np.median(values)) if len(values) else math.nan,
                    "q1": float(np.quantile(values, 0.25)) if len(values) else math.nan,
                    "q3": float(np.quantile(values, 0.75)) if len(values) else math.nan,
                    "all_seed_values": json.dumps(values.tolist(), separators=(",", ":")),
                }
            )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    fixed_time = datetime(2000, 1, 1, tzinfo=UTC)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(
        pdf,
        metadata={
            "Creator": "CAPT-12",
            "Producer": "CAPT-12",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(png, dpi=220, metadata={"Software": "CAPT-12"})
    plt.close(fig)
    return pdf, png


def _errorbar(axis: plt.Axes, rows: pd.DataFrame, x: str, y: str, method: str) -> None:
    values_x = rows[x].to_numpy(float)
    values_y = rows[y].to_numpy(float)
    low_name = f"{y}_ci95_low" if f"{y}_ci95_low" in rows else None
    high_name = f"{y}_ci95_high" if f"{y}_ci95_high" in rows else None
    y_error = None
    if low_name and high_name:
        low = rows[low_name].to_numpy(float)
        high = rows[high_name].to_numpy(float)
        y_error = np.vstack([values_y - low, high - values_y])
    axis.errorbar(
        values_x,
        values_y,
        yerr=y_error,
        marker="o",
        linestyle="-",
        linewidth=1.2,
        capsize=2,
        color=_METHOD_COLORS.get(method, "#111827"),
        label=str(rows["display_name"].iloc[0]),
    )


def _figure_a(frame: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    eligible = frame.loc[frame["certificate_valid"] & frame["formal_comparable"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    costs = ("teacher_kl", "empirical_logloss", "hybrid_logloss_kl")
    for axis, cost in zip(axes, costs, strict=True):
        subset = eligible.loc[eligible["cost"] == cost]
        for method, rows in subset.groupby("method", sort=True):
            finite = rows.loc[np.isfinite(rows["certified_upper_epsilon"].astype(float))]
            if len(finite):
                _errorbar(axis, finite, "certified_upper_epsilon", "excess_ctr_log_loss", method)
        axis.axvline(1.0, color="#6B7280", linestyle="--", linewidth=1)
        axis.set_title(cost.replace("_", " "))
        axis.set_xlabel("robust certified upper epsilon")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("excess CTR log loss (nats/display)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False)
    fig.suptitle("A. Certified privacy–utility frontier")
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.07, right=0.99, wspace=0.10)
    return _save_figure(fig, output / "figure_a_certified_frontier")


def _figure_b(frame: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    formal = frame["certificate_valid"] & (
        frame["certified_upper_epsilon"].astype(float) <= 1.0 + 1e-12
    )
    eligible = frame.loc[formal | frame["oracle"]].copy()
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    eligible = eligible.sort_values(["cost", "method", "seed"], kind="stable")
    labels = []
    for position, row in enumerate(eligible.itertuples()):
        labels.append(f"{row.cost}\n{row.display_name}\ns{row.seed}")
        marker = "s" if row.oracle else "o"
        axis.scatter(
            position,
            row.paired_excess_log_loss_difference_vs_capt,
            marker=marker,
            color=_METHOD_COLORS.get(row.method, "#111827"),
            edgecolor="#111827",
            linewidth=0.5,
        )
        axis.vlines(
            position,
            row.paired_difference_ci95_low,
            row.paired_difference_ci95_high,
            color=_METHOD_COLORS.get(row.method, "#111827"),
        )
    axis.axhline(0, color="#6B7280", linestyle="--", linewidth=1)
    axis.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    axis.set_ylabel("baseline − CAPT excess log loss")
    axis.set_title(
        "B. Utility at robust epsilon ≤ 1; squares are non-deployable oracle references"
    )
    axis.grid(axis="y", alpha=0.2)
    fig.subplots_adjust(bottom=0.36, left=0.10, right=0.99, top=0.90)
    return _save_figure(fig, output / "figure_b_utility_at_epsilon_1")


def _figure_c(frame: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    mass = frame.loc[frame["method"].isin({"mass12_raw", "mass12_calibrated"})].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for label, metric, marker in (
        ("raw", "raw_upper_epsilon", "o"),
        ("calibrated", "certified_upper_epsilon", "s"),
    ):
        rows = mass.loc[np.isfinite(mass[metric].astype(float))]
        axes[0].scatter(rows["mass_m"], rows[metric], marker=marker, label=label, alpha=0.8)
    axes[0].axhline(1.0, color="#6B7280", linestyle="--", linewidth=1)
    axes[0].set_xlabel("MaSS loss-m (nats; control, not achieved privacy)")
    axes[0].set_ylabel("robust upper epsilon")
    axes[0].legend(frameon=False)
    axes[1].scatter(
        mass["mass_m"],
        mass["cover_lambda"],
        c=mass["constant_channel"].map({True: "#DC2626", False: "#059669"}),
    )
    axes[1].set_xlabel("MaSS loss-m (nats)")
    axes[1].set_ylabel("certified cover lambda*")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("C. MaSS-12 robust privacy calibration")
    fig.subplots_adjust(top=0.85, bottom=0.17, left=0.09, right=0.98, wspace=0.27)
    return _save_figure(fig, output / "figure_c_mass_calibration")


def _figure_d(frame: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    rows = frame.loc[
        np.isfinite(frame["empirical_attack_cmi_raw_nats"].astype(float))
        & np.isfinite(frame["excess_ctr_log_loss"].astype(float))
    ]
    rows = empirical_overlap_pareto_frontiers(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for (cost, method), group in rows.groupby(["cost", "method"], sort=True):
        group = group.sort_values("empirical_attack_cmi_zero_clipped_nats", kind="stable")
        axes[0].plot(
            group["empirical_attack_cmi_zero_clipped_nats"],
            group["excess_ctr_log_loss"],
            marker="o",
            color=_METHOD_COLORS.get(method, "#111827"),
            label=f"{group['display_name'].iloc[0]} / {cost}",
        )
        axes[1].scatter(
            group["empirical_attack_cmi_zero_clipped_nats"],
            group["lower_audit_epsilon"],
            color=_METHOD_COLORS.get(method, "#111827"),
        )
    axes[0].set_ylabel("excess CTR log loss")
    axes[1].set_ylabel("lower-audit epsilon (violation witness only)")
    for axis in axes:
        axis.set_xlabel("cross-fitted attack-CMI proxy (zero-clipped nats)")
        axis.grid(alpha=0.2)
    if len(rows):
        axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("D. Empirical diagnostic; not a certificate")
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.08, right=0.98, wspace=0.26)
    return _save_figure(fig, output / "figure_d_empirical_frontier")


def empirical_overlap_pareto_frontiers(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep nondominated observed points inside the shared leakage range.

    Both axes are losses, so a point is dominated when another point from the
    same method has no larger leakage and no larger excess log loss.  The
    function never interpolates or extrapolates.
    """
    if len(frame) == 0:
        return frame.copy()
    leakage = "empirical_attack_cmi_zero_clipped_nats"
    frontiers = []
    for _cost, cost_frame in frame.groupby("cost", sort=True):
        methods = list(cost_frame.groupby("method", sort=True))
        overlap_low = max(float(group[leakage].min()) for _, group in methods)
        overlap_high = min(float(group[leakage].max()) for _, group in methods)
        if overlap_low > overlap_high:
            continue
        for _method, group in methods:
            ordered = group.sort_values([leakage, "excess_ctr_log_loss"], kind="stable")
            best_utility = math.inf
            keep = []
            for index, row in ordered.iterrows():
                utility = float(row["excess_ctr_log_loss"])
                if utility < best_utility:
                    keep.append(index)
                    best_utility = utility
            frontier = ordered.loc[keep]
            frontier = frontier.loc[
                frontier[leakage].between(overlap_low, overlap_high, inclusive="both")
            ]
            frontiers.append(frontier)
    return pd.concat(frontiers, ignore_index=True) if frontiers else frame.iloc[0:0].copy()


def _figure_e(
    frame: pd.DataFrame, contracts: list[MethodContract], output: Path
) -> tuple[Path, Path]:
    fig, axis = plt.subplots(figsize=(13.5, max(4.8, 0.42 * len(contracts) + 1.5)))
    axis.axis("off")
    by_method = frame.groupby("method", sort=False).first()
    rows = []
    for contract in contracts:
        observed = by_method.loc[contract.method] if contract.method in by_method.index else None
        rows.append(
            [
                contract.display_name,
                ",".join(contract.online_inputs) or "—",
                "yes" if contract.uses_sensitive_value_online else "no",
                str(contract.output_bits) if contract.output_bits else "—",
                contract.privacy_definition,
                "yes" if contract.formal_comparable else "no",
                "yes" if contract.deployable_under_capt else "no",
                "not run" if observed is None else f"{float(observed['excess_ctr_log_loss']):.4g}",
                contract.exclusion_reason or "—",
            ]
        )
    table = axis.table(
        cellText=rows,
        colLabels=[
            "method",
            "online inputs",
            "A_S online",
            "bits",
            "privacy definition",
            "formal",
            "deployable",
            "utility",
            "exclusion/qualification",
        ],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.35)
    axis.set_title("E. Mechanism comparability", pad=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
    return _save_figure(fig, output / "figure_e_mechanism_comparability")


def render_prior_art_figures(
    results: pd.DataFrame,
    contracts: list[MethodContract],
    output_dir: Path,
) -> list[Path]:
    frame = validate_results(results)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for renderer in (_figure_a, _figure_b, _figure_c, _figure_d):
        paths.extend(renderer(frame, output))
    paths.extend(_figure_e(frame, contracts, output))
    return paths


def write_method_contracts(contracts: list[MethodContract], path: Path) -> None:
    payload = {"schema_version": 1, "methods": [asdict(contract) for contract in contracts]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_method_contracts(path: Path) -> list[MethodContract]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported method-contract schema")
    contracts = []
    for raw in payload["methods"]:
        item = dict(raw)
        item["online_inputs"] = tuple(item["online_inputs"])
        contracts.append(MethodContract(**item))
    return contracts


def write_review_packet(output_dir: Path, metadata: dict[str, Any]) -> Path:
    output = Path(output_dir)
    packet = output / "prior_art_comparison_review_packet.zip"
    manifest_path = output / "review_packet_manifest.json"
    members = [
        candidate
        for candidate in sorted(output.rglob("*"))
        if candidate.is_file() and candidate not in {packet, manifest_path}
    ]
    manifest = {
        **metadata,
        "files": [
            {
                "path": candidate.relative_to(output).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
            for candidate in members
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    members.append(manifest_path)
    with zipfile.ZipFile(packet, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for candidate in members:
            info = zipfile.ZipInfo(candidate.relative_to(output).as_posix())
            info.date_time = (2000, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, candidate.read_bytes())
    return packet


def figure_hash_manifest(paths: list[Path]) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
