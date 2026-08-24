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
import yaml

from capt12.config import canonical_json, run_id, validate_config
from capt12.experiments.context_stratified import run_context_stratified_diagnostic
from capt12.pipeline import record_source_provenance
from capt12.utils.artifacts import sha256_file

_DESIGN = "joint_kmedoids_cost_medoid_L16"
_SUMMARY_VERSION = 2


def _emit(event: str, **fields: Any) -> None:
    timestamp = datetime.now(UTC).isoformat()
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[{timestamp}] [context_seed_stability] [{event}] {details}".rstrip(), flush=True)


def _completed_run(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = [
        path / "tables" / "aggregate_results.csv",
        path / "tables" / "context_results.csv",
        path / "tables" / "test_metrics.csv",
        path / "tables" / "certificate_summary.csv",
        path / "tables" / "lower_audit.csv",
        path / "context_stratified_metadata.json",
        path / "sol_review_bundle.zip",
    ]
    return manifest.get("status") == "complete" and all(
        candidate.is_file() for candidate in required
    )


def _method_row(frame: pd.DataFrame, method: str) -> pd.Series:
    rows = frame.loc[frame["method"] == method]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one {method!r} row, found {len(rows)}")
    return rows.iloc[0]


def _boolean_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise RuntimeError(f"invalid boolean values in {series.name}")
    return normalized == "true"


def _read_seed_run(
    seed: int,
    path: Path,
    source_git_sha: str,
    expected_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "context_stratified_metadata.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((path / "resolved_config.yaml").read_text(encoding="utf-8"))
    mechanism = json.loads(
        (path / "mechanism" / "context_channel_manifest.json").read_text(encoding="utf-8")
    )
    tables = {
        "aggregate": pd.read_csv(path / "tables" / "aggregate_results.csv"),
        "contexts": pd.read_csv(
            path / "tables" / "context_results.csv", dtype={"public_context": str}
        ),
        "tests": pd.read_csv(path / "tables" / "test_metrics.csv"),
        "certificates": pd.read_csv(
            path / "tables" / "certificate_summary.csv", dtype={"public_context": str}
        ),
        "audits": pd.read_csv(path / "tables" / "lower_audit.csv"),
    }
    aggregate = tables["aggregate"]
    contexts = tables["contexts"]
    tests = tables["tests"]
    certificates = tables["certificates"]
    audits = tables["audits"]

    if manifest.get("status") != "complete":
        raise RuntimeError(f"seed {seed} run is not complete: {path}")
    if (
        manifest.get("source_git_sha") != source_git_sha
        or metadata.get("source_git_sha") != source_git_sha
    ):
        raise RuntimeError(f"seed {seed} source Git SHA differs from the stability run")
    if int(resolved.get("frozen_design_seed", -1)) != seed:
        raise RuntimeError(f"seed {seed} resolved config does not record the requested seed")
    if canonical_json(resolved) != canonical_json(expected_config):
        raise RuntimeError(f"seed {seed} changed a resolved setting other than frozen_design_seed")
    if run_id(resolved) != path.name:
        raise RuntimeError(f"seed {seed} run directory is not bound to its resolved config")
    if float(resolved.get("epsilon", -1)) != 1.0:
        raise RuntimeError("seed stability experiment is fixed at epsilon=1")
    if resolved.get("context_designs") != [_DESIGN]:
        raise RuntimeError("seed stability experiment is fixed at joint k-medoids L=16")
    if set(aggregate["L"].astype(int)) != {16} or set(contexts["L"].astype(int)) != {16}:
        raise RuntimeError("seed result contains a channel dimension other than L=16")
    if len(certificates) != int(metadata["context_count"]):
        raise RuntimeError("one certificate per public context is required")
    if not _boolean_values(certificates["certificate_valid"]).all():
        raise RuntimeError(f"seed {seed} has an invalid ordinary certificate")
    if not _boolean_values(certificates["certificate_conservative_valid"]).all():
        raise RuntimeError(f"seed {seed} has an invalid conservative certificate")
    realized = certificates["certificate_conservative_realized_epsilon"].to_numpy(float)
    if not np.isfinite(realized).all() or float(realized.max()) > 1.0:
        raise RuntimeError(f"seed {seed} does not satisfy finite pure epsilon=1 verification")
    if int(metadata["post_repair_zero_denominator_positive_numerator_count"]) != 0:
        raise RuntimeError(f"seed {seed} retains a positive-over-zero privacy constraint")

    capt = _method_row(aggregate, "context_capt")
    capt_pre = _method_row(aggregate, "context_capt_pre_repair")
    ldp = _method_row(aggregate, "context_ldp")
    constant = _method_row(aggregate, "context_constant")
    capt_test = _method_row(tests, "context_capt")
    ldp_test = _method_row(tests, "context_ldp")
    degraded = _boolean_values(contexts["ldp_degraded"])
    strict_advantage = (contexts["capt_advantage_over_context_ldp"] > 1e-12) & ~degraded
    masses = contexts["design_mass"].to_numpy(float)
    capt_audit = _method_row(audits, "context_capt")
    ldp_audit = _method_row(audits, "context_ldp")
    relative_reduction = float(
        (ldp["aggregate_distortion"] - capt["aggregate_distortion"])
        / ldp["aggregate_distortion"]
    )
    strict_mass = float(contexts.loc[strict_advantage, "design_mass"].sum())
    degraded_mass = float(contexts.loc[degraded, "design_mass"].sum())

    design_manifest = mechanism["designs"][_DESIGN]
    row = {
        "frozen_design_seed": seed,
        "run_id": path.name,
        "run_path": str(path),
        "source_git_sha": source_git_sha,
        "encoder_sha256": sha256_file(path / "models" / "encoder.joblib"),
        "assignment_hash": design_manifest["assignment_hash"],
        "decoder_hash": design_manifest["decoder_hash"],
        "context_count": int(metadata["context_count"]),
        "L": 16,
        "epsilon": 1.0,
        "utility_objective": str(resolved["context_utility_objective"]),
        "hybrid_empirical_weight": float(resolved["hybrid_empirical_weight"]),
        "context_constant_distortion": float(constant["aggregate_distortion"]),
        "context_ldp_distortion": float(ldp["aggregate_distortion"]),
        "context_capt_pre_repair_distortion": float(capt_pre["aggregate_distortion"]),
        "context_capt_distortion": float(capt["aggregate_distortion"]),
        "capt_advantage_over_context_ldp": float(
            ldp["aggregate_distortion"] - capt["aggregate_distortion"]
        ),
        "relative_capt_reduction_vs_context_ldp": relative_reduction,
        "repair_distortion_cost": float(
            capt["aggregate_distortion"] - capt_pre["aggregate_distortion"]
        ),
        "strict_advantage_context_count": int(strict_advantage.sum()),
        "strict_advantage_context_mass": strict_mass,
        "ldp_degraded_context_count": int(degraded.sum()),
        "ldp_degraded_context_mass": degraded_mass,
        "tie_or_other_context_mass": max(0.0, 1.0 - strict_mass - degraded_mass),
        "mass_weighted_capt_row_tv": float(
            np.sum(masses * contexts["context_capt_row_tv"].to_numpy(float))
        ),
        "max_capt_row_tv": float(contexts["context_capt_row_tv"].max()),
        "pre_repair_infinite_context_count": int(metadata["pre_repair_infinite_context_count"]),
        "pre_repair_infinite_constraint_count": int(
            metadata["pre_repair_infinite_constraint_count"]
        ),
        "max_repair_lambda": float(metadata["max_repair_lambda"]),
        "mass_weighted_repair_lambda": float(metadata["mass_weighted_repair_lambda"]),
        "conservative_max_realized_epsilon": float(metadata["conservative_max_realized_epsilon"]),
        "conservative_max_additive_violation": float(
            metadata["conservative_max_additive_violation"]
        ),
        "certificate_count": int(metadata["certificate_count"]),
        "certificate_checked_constraints": int(metadata["certificate_checked_constraints"]),
        "all_certificates_valid": True,
        "wall_seconds": float(metadata["wall_seconds"]),
        "process_peak_rss_bytes": int(metadata["process_peak_rss_bytes"]),
        "test_rows": int(capt_test["test_rows"]),
        "lower_audit_context_capt_epsilon": float(capt_audit["lower_epsilon"]),
        "lower_audit_context_ldp_epsilon": float(ldp_audit["lower_epsilon"]),
        "lower_audit_context_capt_tests": int(capt_audit["events_tested"]),
        "lower_audit_context_ldp_tests": int(ldp_audit["events_tested"]),
        "lower_audit_total_bounds": int(
            capt_audit["bounds_tested"] + ldp_audit["bounds_tested"]
        ),
        "lower_audit_familywise_alpha": float(capt_audit["audit_familywise_alpha"]),
        "lower_audit_any_comparable": bool(
            _boolean_values(
                pd.Series(
                    [
                        capt_audit["certificate_comparable"],
                        ldp_audit["certificate_comparable"],
                    ]
                )
            ).any()
        ),
    }
    for metric in [
        "expected_randomized_log_loss",
        "mixture_mean_log_loss",
        "ROC_AUC",
        "PR_AUC",
        "ECE",
        "calibration_ratio",
    ]:
        capt_value = float(capt_test[metric])
        ldp_value = float(ldp_test[metric])
        row[f"test_context_capt_{metric}"] = capt_value
        row[f"test_context_ldp_{metric}"] = ldp_value
        row[f"test_capt_minus_ldp_{metric}"] = capt_value - ldp_value
    return row, tables


def _stability_table(seed_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "context_capt_distortion",
        "context_ldp_distortion",
        "capt_advantage_over_context_ldp",
        "relative_capt_reduction_vs_context_ldp",
        "strict_advantage_context_count",
        "strict_advantage_context_mass",
        "ldp_degraded_context_count",
        "ldp_degraded_context_mass",
        "tie_or_other_context_mass",
        "mass_weighted_capt_row_tv",
        "max_capt_row_tv",
        "max_repair_lambda",
        "mass_weighted_repair_lambda",
        "conservative_max_realized_epsilon",
        "conservative_max_additive_violation",
        "test_context_capt_expected_randomized_log_loss",
        "test_context_ldp_expected_randomized_log_loss",
        "test_capt_minus_ldp_expected_randomized_log_loss",
        "test_context_capt_ROC_AUC",
        "test_context_ldp_ROC_AUC",
        "test_capt_minus_ldp_ROC_AUC",
        "test_context_capt_PR_AUC",
        "test_context_ldp_PR_AUC",
        "test_capt_minus_ldp_PR_AUC",
        "test_context_capt_ECE",
        "test_context_ldp_ECE",
        "lower_audit_context_capt_epsilon",
        "lower_audit_context_ldp_epsilon",
        "wall_seconds",
        "process_peak_rss_bytes",
    ]
    rows = []
    for metric in metrics:
        values = seed_results[metric].to_numpy(float)
        mean = float(values.mean())
        minimum = float(values.min())
        maximum = float(values.max())
        value_range = maximum - minimum
        rows.append(
            {
                "metric": metric,
                "seed_count": len(values),
                "mean": mean,
                "sample_std": float(values.std(ddof=1)),
                "min": minimum,
                "max": maximum,
                "range": value_range,
                "relative_range_to_abs_mean": (value_range / abs(mean) if mean != 0 else math.nan),
            }
        )
    return pd.DataFrame(rows)


def _plot_stability(seed_results: pd.DataFrame, output_dir: Path) -> None:
    frame = seed_results.sort_values("frozen_design_seed").reset_index(drop=True)
    labels = [str(value) for value in frame["frozen_design_seed"]]
    x = np.arange(len(frame))
    blue = "#2563A6"
    orange = "#D97706"
    pale = "#D9E6F2"
    relative = 100 * (
        frame["context_ldp_distortion"].to_numpy(float)
        - frame["context_capt_distortion"].to_numpy(float)
    ) / frame["context_ldp_distortion"].to_numpy(float)
    strict = frame["strict_advantage_context_mass"].to_numpy(float)
    degraded = frame["ldp_degraded_context_mass"].to_numpy(float)
    tie = np.maximum(0.0, 1.0 - strict - degraded)
    ldp_ceiling = math.tanh(0.5)

    fig = plt.figure(figsize=(12.4, 8.4))
    outer = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
    reduction_axis = fig.add_subplot(outer[0, 0])
    tv_axis = fig.add_subplot(outer[0, 1])
    mass_axis = fig.add_subplot(outer[1, 0])
    metric_grid = outer[1, 1].subgridspec(3, 1, hspace=0.15)
    metric_axes = [fig.add_subplot(metric_grid[index, 0]) for index in range(3)]

    bars = reduction_axis.bar(x, relative, color=blue, edgecolor="#1F2937")
    reduction_axis.axhline(0, color="#111827", lw=0.8)
    reduction_axis.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    reduction_axis.set_xticks(x, labels)
    reduction_axis.set_xlabel("Frozen design seed")
    reduction_axis.set_ylabel("Reduction vs context LDP (%)")
    reduction_axis.set_title("A. Paired D_design objective reduction")
    reduction_axis.grid(axis="y", alpha=0.22)
    reduction_axis.margins(y=0.16)

    tv_axis.plot(
        x,
        frame["mass_weighted_capt_row_tv"],
        color=blue,
        marker="o",
        lw=1.8,
        label="mass-weighted CAPT row TV",
    )
    tv_axis.scatter(
        x,
        frame["max_capt_row_tv"],
        color=blue,
        facecolors="white",
        marker="^",
        s=48,
        label="maximum context row TV",
        zorder=3,
    )
    tv_axis.axhline(
        ldp_ceiling,
        color=orange,
        ls="--",
        lw=1.6,
        label=r"epsilon-LDP ceiling $\tanh(1/2)$",
    )
    tv_axis.set_xticks(x, labels)
    tv_axis.set_xlabel("Frozen design seed")
    tv_axis.set_ylabel("Maximum pairwise row TV")
    tv_axis.set_title("B. CAPT uses a non-LDP feasible region")
    tv_axis.grid(axis="y", alpha=0.22)
    tv_axis.legend(fontsize=8, loc="center right")

    mass_axis.bar(x, strict, color=blue, edgecolor="#1F2937", label="strict CAPT advantage")
    mass_axis.bar(x, tie, bottom=strict, color=pale, edgecolor="#1F2937", label="tie / other")
    mass_axis.bar(
        x,
        degraded,
        bottom=strict + tie,
        color=orange,
        edgecolor="#1F2937",
        hatch="//",
        label="LDP-degraded",
    )
    for index, (strict_value, degraded_value) in enumerate(zip(strict, degraded, strict=True)):
        mass_axis.text(
            index,
            strict_value / 2,
            f"{100 * strict_value:.2f}% strict",
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            rotation=90 if len(frame) > 6 else 0,
        )
        mass_axis.text(
            index,
            0.985,
            f"LDP {100 * degraded_value:.4f}%",
            ha="center",
            va="top",
            color=orange,
            fontsize=6.5,
            rotation=90,
        )
    mass_axis.set_xticks(x, labels)
    mass_axis.set_ylim(0, 1.02)
    mass_axis.set_xlabel("Frozen design seed")
    mass_axis.set_ylabel("D_design context mass")
    mass_axis.set_title("C. Mutually exclusive context-mass states")
    mass_axis.legend(fontsize=8, loc="lower left")
    mass_axis.grid(axis="y", alpha=0.18)

    metric_specs = [
        (
            "Expected log loss",
            1e6
            * (
                frame["test_context_ldp_expected_randomized_log_loss"].to_numpy(float)
                - frame["test_context_capt_expected_randomized_log_loss"].to_numpy(float)
            ),
            r"$(LDP-CAPT)\times10^6$",
        ),
        (
            "ROC-AUC",
            1e3 * frame["test_capt_minus_ldp_ROC_AUC"].to_numpy(float),
            r"$(CAPT-LDP)\times10^3$",
        ),
        (
            "PR-AUC",
            1e3 * frame["test_capt_minus_ldp_PR_AUC"].to_numpy(float),
            r"$(CAPT-LDP)\times10^3$",
        ),
    ]
    for index, (axis, (metric, values, ylabel)) in enumerate(
        zip(metric_axes, metric_specs, strict=True)
    ):
        colors = np.where(values >= 0, blue, orange)
        axis.axhline(0, color="#111827", lw=0.8)
        axis.bar(x, values, color=colors, width=0.68)
        limit = max(float(np.max(np.abs(values))) * 1.3, 1e-9)
        axis.set_ylim(-limit, limit)
        axis.set_ylabel(ylabel, fontsize=7)
        axis.text(0.01, 0.83, metric, transform=axis.transAxes, fontsize=8)
        axis.grid(axis="y", alpha=0.18)
        if index < 2:
            axis.set_xticks(x, [])
        else:
            axis.set_xticks(x, labels)
            axis.set_xlabel("Frozen design seed")
    metric_axes[0].set_title("D. Paired D_test improvement (positive favors CAPT)")

    objective = str(frame.get("utility_objective", pd.Series(["teacher_kl"])).iloc[0])
    fig.suptitle(
        f"Criteo public-context CAPT stability; epsilon=1, L=16; objective={objective}",
        y=0.985,
    )
    fig.text(
        0.5,
        0.008,
        "Only frozen_design_seed changes. Bars/points are paired by seed; no seed-level confidence interval is implied.",
        ha="center",
        fontsize=8,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.92)
    fixed_time = datetime(2000, 1, 1, tzinfo=UTC)
    fig.savefig(
        output_dir / "figures" / "context_seed_stability.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "CAPT-12",
            "Producer": "CAPT-12",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        output_dir / "figures" / "context_seed_stability.png",
        bbox_inches="tight",
        dpi=220,
        metadata={"Software": "CAPT-12"},
    )
    plt.close(fig)


def _write_report(seed_results: pd.DataFrame, stability: pd.DataFrame, output_dir: Path) -> None:
    ordered = seed_results.sort_values("frozen_design_seed")
    all_advantage = bool((ordered["capt_advantage_over_context_ldp"] > 0).all())
    all_certified = bool(ordered["all_certificates_valid"].all())
    all_finite = bool(np.isfinite(ordered["conservative_max_realized_epsilon"]).all())
    total_constraints = int(ordered["certificate_checked_constraints"].sum())
    objective = str(ordered["utility_objective"].iloc[0])
    conclusion = (
        "The CAPT-over-context-LDP design utility advantage is positive for every seed."
        if all_advantage
        else "The CAPT-over-context-LDP design utility advantage is not positive for every seed."
    )
    lines = [
        "# Frozen-design seed stability: public-context CAPT",
        "",
        "## Answer",
        "",
        conclusion,
        f"All certificates valid: **{all_certified}**; all conservative realized epsilons finite and at most 1: **{all_finite and bool((ordered['conservative_max_realized_epsilon'] <= 1).all())}**.",
        "",
        "## Fixed scope",
        "",
        f"- Frozen design seeds: {', '.join(map(str, ordered['frozen_design_seed']))}.",
        "- Criteo `features_kv_bits_constrained_2`; public-context channels; unified `__UNKNOWN__`; epsilon=1; joint weighted k-medoids; L=16.",
        f"- LP utility objective: `{objective}`; hybrid empirical weight: {ordered['hybrid_empirical_weight'].iloc[0]:.6g}.",
        "- Temporal splits, support/adjacency/privacy definition, utility objective, D_cert, and D_test are fixed. Only `frozen_design_seed` changes the frozen encoder/design realization.",
        "- Runs are sequential to bound local peak memory. A completed run with the exact source SHA and resolved seed config is reused.",
        "",
        "## Per-seed primary results",
        "",
        "| seed | run | relative objective reduction | mass-weighted CAPT row TV | strict-advantage mass | LDP-degraded mass | D_test CAPT-LDP log loss | privacy lower (CAPT/LDP) | certificates |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {int(row['frozen_design_seed'])} | `{row['run_id']}` | "
            f"{100 * row['relative_capt_reduction_vs_context_ldp']:.3f}% | "
            f"{row['mass_weighted_capt_row_tv']:.6g} | "
            f"{row['strict_advantage_context_mass']:.6g} | {row['ldp_degraded_context_mass']:.6g} | "
            f"{row['test_capt_minus_ldp_expected_randomized_log_loss']:.9g} | "
            f"{row['lower_audit_context_capt_epsilon']:.6g} / {row['lower_audit_context_ldp_epsilon']:.6g} | valid |"
        )
    lines.extend(
        [
            "",
            "## Stability ranges",
            "",
            "| metric | mean | sample SD | min | max | range |",
            "|:---|---:|---:|---:|---:|---:|",
        ]
    )
    display_metrics = [
        "capt_advantage_over_context_ldp",
        "relative_capt_reduction_vs_context_ldp",
        "strict_advantage_context_mass",
        "ldp_degraded_context_mass",
        "test_capt_minus_ldp_expected_randomized_log_loss",
        "test_capt_minus_ldp_ROC_AUC",
        "test_capt_minus_ldp_PR_AUC",
        "max_repair_lambda",
        "conservative_max_realized_epsilon",
        "lower_audit_context_capt_epsilon",
        "lower_audit_context_ldp_epsilon",
    ]
    indexed = stability.set_index("metric")
    for metric in display_metrics:
        row = indexed.loc[metric]
        lines.append(
            f"| `{metric}` | {row['mean']:.9g} | {row['sample_std']:.9g} | "
            f"{row['min']:.9g} | {row['max']:.9g} | {row['range']:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Certificate and repair checks",
            "",
            f"- {int(ordered['certificate_count'].sum())} context certificates rechecked {total_constraints:,} constraints across all seeds.",
            f"- Maximum conservative realized epsilon: {ordered['conservative_max_realized_epsilon'].max():.12g}.",
            f"- Maximum conservative additive violation: {ordered['conservative_max_additive_violation'].max():.9g}.",
            f"- Maximum repair lambda: {ordered['max_repair_lambda'].max():.9g}.",
            "- Every released channel has zero positive-numerator/zero-denominator constraints; this is checked inside each constituent run before aggregation.",
            "",
            "## Privacy lower audit",
            "",
            f"Each seed reports D_attack_train-fixed lower audits for context CAPT and matched context LDP. Alpha={ordered['lower_audit_familywise_alpha'].iloc[0]:.6g} is split across the two mechanism families before each audit allocates over all tests and both CP bounds. Across seeds, the maximum CAPT lower is {ordered['lower_audit_context_capt_epsilon'].max():.9g}, and the maximum LDP lower is {ordered['lower_audit_context_ldp_epsilon'].max():.9g}.",
            "",
            "The five seed audits are five separate within-seed simultaneous families; they are not one joint 95% statement over all seeds. Because no D_cert-to-D_test population bridge is asserted, lower witnesses and certificate path uppers are not treated as a comparable sandwich or subtracted into a gap.",
            "",
            "## Interpretation limits",
            "",
            "This experiment measures sensitivity to the frozen design seed, not sampling uncertainty: the data rows and temporal splits do not change. With only a few seeds, ranges and individual points are more informative than asymptotic confidence intervals. The Sol bundle includes exact utility, lower-audit, certificate, and context tables, figures, resolved configs, mechanisms, and all per-context certificates for every seed.",
        ]
    )
    (output_dir / "context_seed_stability_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_review_bundle(output_dir: Path, seed_paths: dict[int, Path]) -> Path:
    bundle_path = output_dir / "sol_seed_stability_review_bundle.zip"
    temporary_path = output_dir / ".sol_seed_stability_review_bundle.zip.tmp"
    summary_files = sorted(
        candidate
        for candidate in output_dir.rglob("*")
        if candidate.is_file()
        and candidate not in {bundle_path, temporary_path}
        and candidate.name != "sol_seed_stability_review_bundle_manifest.json"
    )
    members: list[tuple[str, bytes]] = []
    for candidate in summary_files:
        archive_path = f"summary/{candidate.relative_to(output_dir).as_posix()}"
        members.append((archive_path, candidate.read_bytes()))
    for seed, run_path in sorted(seed_paths.items()):
        with zipfile.ZipFile(run_path / "sol_review_bundle.zip") as source:
            for name in sorted(source.namelist()):
                if name.endswith("/"):
                    continue
                members.append((f"runs/seed-{seed}/{name}", source.read(name)))
    manifest = {
        "version": 1,
        "purpose": "One-file ChatGPT Sol review of multi-seed CAPT stability",
        "seeds": sorted(seed_paths),
        "files": [
            {
                "path": name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for name, payload in members
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    members.append(("sol_seed_stability_review_bundle_manifest.json", manifest_bytes))
    fixed_timestamp = (2000, 1, 1, 0, 0, 0)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, payload in sorted(members):
                info = zipfile.ZipInfo(name, fixed_timestamp)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload)
        temporary_path.replace(bundle_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    external_manifest = {
        **manifest,
        "bundle": bundle_path.name,
        "bundle_size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256_file(bundle_path),
    }
    (output_dir / "sol_seed_stability_review_bundle_manifest.json").write_text(
        json.dumps(external_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle_path


def run_context_seed_stability(config: dict[str, Any], seeds: list[int]) -> Path:
    """Run/reuse prescribed L16 context CAPT seeds, then make one review packet."""
    started = time.perf_counter()
    seeds = sorted(set(map(int, seeds)))
    if len(seeds) < 2:
        raise ValueError("context seed stability requires at least two distinct seeds")
    if any(seed < 0 for seed in seeds):
        raise ValueError("frozen design seeds must be nonnegative")
    base = validate_config(config)
    if float(base.get("epsilon", -1)) != 1.0 or base.get("context_designs") != [_DESIGN]:
        raise ValueError("context seed stability is fixed at epsilon=1 and joint k-medoids L=16")
    provenance = record_source_provenance(base)
    source_git_sha = str(provenance["source_git_sha"])
    signature_config = dict(provenance)
    signature_config.pop("frozen_design_seed", None)
    summary_signature = {
        "experiment": "criteo_context_seed_stability",
        "version": _SUMMARY_VERSION,
        "source_git_sha": source_git_sha,
        "seeds": seeds,
        "base_config": json.loads(canonical_json(signature_config)),
    }
    output_dir = Path("outputs/context_stratified_seed_summaries") / run_id(summary_signature)
    for directory in [output_dir, output_dir / "tables", output_dir / "figures"]:
        directory.mkdir(parents=True, exist_ok=True)
    _emit(
        "started",
        seeds=",".join(map(str, seeds)),
        source_git_sha=source_git_sha,
        output_path=output_dir.resolve(),
        execution="sequential",
    )

    seed_paths: dict[int, Path] = {}
    expected_configs: dict[int, dict[str, Any]] = {}
    for index, seed in enumerate(seeds, start=1):
        seed_config = dict(base)
        seed_config["frozen_design_seed"] = seed
        expected_config = record_source_provenance(seed_config)
        expected_configs[seed] = expected_config
        expected_path = Path(expected_config.get("output_dir", "outputs/runs")) / run_id(
            expected_config
        )
        if _completed_run(expected_path):
            _emit(
                "seed_reused",
                seed=seed,
                index=f"{index}/{len(seeds)}",
                run_id=expected_path.name,
                path=expected_path.resolve(),
            )
            seed_path = expected_path
        else:
            _emit(
                "seed_run_started",
                seed=seed,
                index=f"{index}/{len(seeds)}",
                expected_run_id=expected_path.name,
            )
            seed_started = time.perf_counter()
            seed_path = run_context_stratified_diagnostic(seed_config)
            _emit(
                "seed_run_finished",
                seed=seed,
                run_id=seed_path.name,
                seconds=f"{time.perf_counter() - seed_started:.1f}",
            )
        if seed_path.resolve() != expected_path.resolve() or not _completed_run(seed_path):
            raise RuntimeError(f"seed {seed} did not produce the expected complete run")
        seed_paths[seed] = seed_path

    result_rows: list[dict[str, Any]] = []
    combined: dict[str, list[pd.DataFrame]] = {
        "aggregate": [],
        "contexts": [],
        "tests": [],
        "certificates": [],
        "audits": [],
    }
    for seed, seed_path in sorted(seed_paths.items()):
        row, tables = _read_seed_run(
            seed,
            seed_path,
            source_git_sha,
            expected_configs[seed],
        )
        result_rows.append(row)
        for name, table in tables.items():
            table = table.copy()
            table.insert(0, "run_id", seed_path.name)
            table.insert(0, "frozen_design_seed", seed)
            combined[name].append(table)
        _emit(
            "seed_validated",
            seed=seed,
            capt_advantage=f"{row['capt_advantage_over_context_ldp']:.9g}",
            strict_advantage_mass=f"{row['strict_advantage_context_mass']:.9g}",
            certificates=row["certificate_count"],
            constraints=row["certificate_checked_constraints"],
        )

    seed_results = pd.DataFrame(result_rows).sort_values("frozen_design_seed")
    stability = _stability_table(seed_results)
    seed_results.to_csv(output_dir / "tables" / "seed_results.csv", index=False)
    stability.to_csv(output_dir / "tables" / "seed_stability_summary.csv", index=False)
    for name, tables in combined.items():
        pd.concat(tables, ignore_index=True).to_csv(
            output_dir / "tables" / f"seed_{name}_results.csv", index=False
        )
    _plot_stability(seed_results, output_dir)
    _write_report(seed_results, stability, output_dir)
    metadata = {
        "experiment": "criteo_context_seed_stability",
        "version": _SUMMARY_VERSION,
        "source_git_sha": source_git_sha,
        "frozen_design_seeds": seeds,
        "context_utility_objective": base["context_utility_objective"],
        "hybrid_empirical_weight": base["hybrid_empirical_weight"],
        "seed_count": len(seeds),
        "run_ids": {str(seed): path.name for seed, path in sorted(seed_paths.items())},
        "all_certificates_valid": bool(seed_results["all_certificates_valid"].all()),
        "all_seed_advantages_positive": bool(
            (seed_results["capt_advantage_over_context_ldp"] > 0).all()
        ),
        "total_certificate_count": int(seed_results["certificate_count"].sum()),
        "total_checked_constraints": int(seed_results["certificate_checked_constraints"].sum()),
        "lower_audit_familywise_alpha_per_seed": float(
            seed_results["lower_audit_familywise_alpha"].iloc[0]
        ),
        "max_context_capt_lower_epsilon": float(
            seed_results["lower_audit_context_capt_epsilon"].max()
        ),
        "wall_seconds_including_new_seed_runs": time.perf_counter() - started,
    }
    (output_dir / "context_seed_stability_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary_config.json").write_text(
        json.dumps(summary_signature, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    bundle = _write_review_bundle(output_dir, seed_paths)
    _emit(
        "finished",
        output_path=output_dir.resolve(),
        bundle=bundle.resolve(),
        bundle_size_bytes=bundle.stat().st_size,
        bundle_sha256=sha256_file(bundle),
        seconds=f"{time.perf_counter() - started:.1f}",
    )
    return output_dir
