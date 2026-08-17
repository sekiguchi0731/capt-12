from __future__ import annotations

import gc
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capt12.certification.artifact import make_certificate, verify_certificate
from capt12.certification.robust import (
    VerificationResult,
    solve_robust_block_lp,
    verify_robust_channel,
)
from capt12.confidence.boxes import ConfidenceBox, confidence_box_from_counts
from capt12.config import validate_config
from capt12.data.loader import assert_disjoint_splits, assert_no_row_overlap, load_parquet_sample
from capt12.data.preprocessing import build_group_histograms
from capt12.experiments.fixed_support import (
    _expected_frame_from_cartesian,
    _fixed_design,
    _peak_rss_bytes,
)
from capt12.experiments.sampling import select_one_display_per_user_day
from capt12.mechanisms.baselines import common_cover, k_ary_rr
from capt12.mechanisms.lp import SolverInfo, solve_ldp_block_lp
from capt12.pipeline import record_source_provenance
from capt12.privacy.adjacency import build_adjacency, disconnected_hybrid_components
from capt12.utils.artifacts import environment, finish_run, prepare_run


def _objective(channel: np.ndarray, cost: np.ndarray, weights: np.ndarray) -> float:
    normalized = np.asarray(weights, dtype=float) / np.asarray(weights, dtype=float).sum()
    return float(np.sum(normalized[:, None] * np.asarray(channel) * np.asarray(cost)))


def _best_input_independent_channel(
    cost: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, int]:
    """Return the utility-optimal constant-row channel in the fixed block class."""
    normalized = np.asarray(weights, dtype=float) / np.asarray(weights, dtype=float).sum()
    destination_cost = normalized @ np.asarray(cost, dtype=float)
    destination = int(np.argmin(destination_cost))
    distribution = np.zeros(len(destination_cost), dtype=float)
    distribution[destination] = 1.0
    return common_cover(distribution), destination


def _max_row_tv(channel: np.ndarray) -> float:
    channel = np.asarray(channel, dtype=float)
    differences = np.abs(channel[:, None, :] - channel[None, :, :]).sum(axis=2) / 2
    return float(np.max(differences))


def _ldp_max_violation(channel: np.ndarray, epsilon: float) -> float:
    channel = np.asarray(channel, dtype=float)
    violations = channel[:, None, :] - np.exp(epsilon) * channel[None, :, :]
    return float(np.max(violations))


def _analytic_solver(
    status: str,
    channel: np.ndarray,
    cost: np.ndarray,
    weights: np.ndarray,
) -> SolverInfo:
    return SolverInfo(
        status=status,
        objective=_objective(channel, cost, weights),
        runtime_seconds=0.0,
        iterations=0,
        primal_gap=0.0,
        dual_gap=0.0,
        message="closed-form baseline",
        variable_count=int(channel.size),
        constraint_count=0,
        estimated_memory_bytes=int(channel.nbytes),
    )


def _build_boxes(
    counts: dict[str, np.ndarray],
    *,
    adjacency_count: int,
    config: dict[str, Any],
) -> dict[str, ConfidenceBox]:
    return {
        key: confidence_box_from_counts(
            value,
            confidence=config.get("confidence", "cp_box"),
            missing_group_policy=config.get("missing_group_policy", "force_cover"),
            alpha=float(config.get("alpha_cert", 0.05)),
            group_count=len(counts),
            comparisons=max(1, adjacency_count),
            tv_radius=float(config.get("shift_tv", 0.0)),
        )
        for key, value in counts.items()
    }


def _plot_comparison(results: pd.DataFrame, path: Path) -> None:
    labels = ["Frozen mu cover", "k-ary RR", "Optimal LDP", "Simplex-CAPT"]
    order = ["common_cover", "k_ary_rr", "optimal_ldp", "simplex_capt"]
    indexed = results.set_index("method").loc[order]
    colors = ["#777777", "#D97706", "#D97706", "#2563A6"]
    hatches = ["", "//", "", "xx"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, column, title, ylabel in (
        (
            axes[0],
            "utility_gain_over_common_cover",
            "Utility gain over frozen mu cover",
            "utility difference (higher is better)",
        ),
        (
            axes[1],
            "max_pairwise_row_tv",
            "Input dependence of each block channel",
            "maximum pairwise row TV",
        ),
    ):
        bars = axis.bar(
            labels,
            indexed[column].to_numpy(),
            color=colors,
            edgecolor="#222222",
            linewidth=0.8,
        )
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.22, color="#777777")
        for bar, value in zip(bars, indexed[column], strict=True):
            axis.annotate(
                f"{value:.3g}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Criteo constrained_2 full D_cert; fixed decoder, L=16, epsilon=1")
    fig.tight_layout()
    fixed_time = datetime(2000, 1, 1, tzinfo=UTC)
    fig.savefig(
        path / "figures" / "simplex_completion_comparison.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "CAPT-12",
            "Producer": "CAPT-12",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        path / "figures" / "simplex_completion_comparison.png",
        bbox_inches="tight",
        dpi=220,
        metadata={"Software": "CAPT-12"},
    )
    plt.close(fig)


def _write_report(results: pd.DataFrame, diagnostics: dict[str, Any], path: Path) -> None:
    capt = results.set_index("method").loc["simplex_capt"]
    ldp = results.set_index("method").loc["optimal_ldp"]
    cover = results.set_index("method").loc["common_cover"]
    escaped = bool(capt["nontrivial_channel"])
    dominance = bool(
        capt["utility"] + 1e-9 >= ldp["utility"] and ldp["utility"] + 1e-9 >= cover["utility"]
    )
    matches_best_cover = bool(capt["matches_best_input_independent"])
    report = f"""# Criteo full-simplex completion: one-condition comparison

## Fixed condition

- Profile: `features_kv_bits_constrained_2`
- D_cert: full split with {diagnostics["cert_user_days"]:,} one-display-per-user-day contributions
- K/L: {diagnostics["K"]}/{diagnostics["L"]}; epsilon: {diagnostics["epsilon"]}
- Frozen mapper, encoder, reference model, partition, common decoder, objective, cost, expected Cartesian support, and adjacency family
- Observed groups: CP/TV confidence box even when count is below 20
- Unobserved expected groups: exact full-simplex uncertainty set

## Support and solver diagnostics

- Expected/observed/missing groups: {diagnostics["expected_group_count"]} / {diagnostics["observed_group_count"]} / {diagnostics["missing_group_count"]}
- Observed groups below count 20: {diagnostics["rare_group_count"]}; rare mass: {diagnostics["rare_group_mass"]:.6g}
- Minimum positive group count: {diagnostics["minimum_positive_group_count"]}
- Ordered adjacency edges: {diagnostics["adjacency_count"]}; hybrid connectivity gaps: {diagnostics["hybrid_connectivity_gaps"]}
- Simplex-CAPT solver status: `{capt["solver_status"]}`; independent certificate: `{bool(capt["certificate_valid"])}`

## Result

Simplex-CAPT {"produced a non-input-independent channel" if escaped else "remained input-independent"} (`max_pairwise_row_tv={capt["max_pairwise_row_tv"]:.6g}`). Its utility gain over the frozen design-weight `mu` cover is {capt["utility_gain_over_common_cover"]:.9g}, but its gain over the best input-independent channel in the same fixed decoder class is {capt["utility_gain_over_best_input_independent"]:.9g}. It {"exactly matches" if matches_best_cover else "does not match"} that best constant-row channel within 1e-10. The same-class ordering `U_simplex-CAPT >= U_optimal-LDP >= U_frozen-mu-cover` is {"satisfied" if dominance else "not satisfied"} within a 1e-9 numerical tolerance.

This is not a nontrivial CAPT privacy-utility improvement. The LP was executed and optimized, but selected an input-independent deterministic destination-block cover. Therefore this gate does not trigger the nested D_cert/epsilon/L expansion.

All four saved channels were checked against the same mixed CP/full-simplex robust constraints. This establishes robust feasibility for the saved channels; it does not establish that a different partition, decoder, support definition, or objective would behave the same way.

## Interpretation caveat

Information monotonicity is a set-inclusion theorem for frozen confidence sets and a frozen objective. Ordinary CP intervals recomputed at larger samples need not be nested realization by realization, so a later sample-size grid should not describe every empirical step as theorem-guaranteed monotone unless nested confidence sets are explicitly constructed.
"""
    (path / "simplex_completion_report.md").write_text(report)


def run_simplex_completion(config: dict[str, Any]) -> Path:
    """Run the full-D_cert, one-condition full-simplex CAPT comparison."""
    config = validate_config(config)
    required_splits = {"D_model", "D_design", "D_cert", "D_attack_train", "D_test"}
    if set(config.get("splits", {})) != required_splits:
        raise ValueError("simplex completion requires the five fixed temporal splits")
    assert_disjoint_splits(config["splits"])
    if config.get("profiles") != ["features_kv_bits_constrained_2"]:
        raise ValueError("simplex completion currently requires constrained_2 only")
    if int(config.get("L", 0)) != 16 or float(config.get("epsilon", -1)) != 1.0:
        raise ValueError("the first simplex-completion experiment requires L=16 and epsilon=1")
    if config.get("missing_group_policy") != "full_simplex":
        raise ValueError("simplex completion requires missing_group_policy=full_simplex")
    if config.get("rare_group_policy") != "confidence_box":
        raise ValueError("simplex completion requires rare_group_policy=confidence_box")

    config = record_source_provenance(config)
    path = prepare_run(config)
    started = time.perf_counter()
    frozen, support_table = _fixed_design(config, path)
    support_table.to_csv(path / "tables" / "frozen_support.csv", index=False)

    profile = config["profiles"][0]
    contexts = list(config.get("context_cols", []))
    source = list(config.get("phi_source_cols", []))
    user_col = config.get("user_col", "user_id")
    id_col = config.get("id_col", "id")
    columns = list(dict.fromkeys([id_col, user_col, *profile.split("+"), *contexts, *source]))
    cert_source = load_parquet_sample(
        data_root=config["data_root"],
        columns=columns,
        days=config["splits"]["D_cert"],
    )
    assert_no_row_overlap({"D_cert": cert_source}, id_col)
    cert_source = frozen["mapper"].transform(cert_source)
    cert_frame = select_one_display_per_user_day(
        cert_source,
        user_col=user_col,
        day_col="day_int",
        id_col=id_col,
    )
    cert_tokens = frozen["encoder"].transform(cert_frame)
    expected_frame = _expected_frame_from_cartesian(
        frozen["cartesian_support"],
        profile=profile,
        context=contexts[0],
    )
    hist = build_group_histograms(
        cert_frame,
        cert_tokens,
        profile=profile,
        context_columns=contexts,
        alphabet_size=int(config["L"]),
        token_to_block=frozen["assignment"],
        min_group_count=int(config.get("min_group_count", 20)),
        rare_group_policy="confidence_box",
        missing_group_policy="full_simplex",
        expected_frame=expected_frame,
        include_fallback_levels=False,
    )
    adjacency = build_adjacency(
        hist.groups,
        config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
        float(config["epsilon"]),
        config.get("epsilon_by_attr", {}),
    )
    connectivity_gaps = disconnected_hybrid_components(hist.groups, adjacency)
    if hist.force_cover or not adjacency or connectivity_gaps:
        raise RuntimeError(
            "full Cartesian simplex completion must produce a connected, non-fallback problem"
        )
    boxes = _build_boxes(hist.counts, adjacency_count=len(adjacency), config=config)

    cost = frozen["block_cost"]
    weights = frozen["block_weights"]
    epsilon = float(config["epsilon"])
    cover_channel = common_cover(weights)
    best_cover_channel, best_cover_destination = _best_input_independent_channel(cost, weights)
    rr_channel = k_ary_rr(int(config["L"]), epsilon)
    ldp_solution = solve_ldp_block_lp(
        cost,
        weights,
        epsilon,
        tolerance=float(config.get("solver_tolerance", 1e-8)),
        time_limit=config.get("time_limit"),
    )
    if ldp_solution.channel is None:
        raise RuntimeError(f"optimal LDP LP failed: {ldp_solution.solver.message}")
    capt_solution, capt_verification = solve_robust_block_lp(
        cost,
        weights,
        boxes,
        adjacency,
        tolerance=float(config.get("solver_tolerance", 1e-8)),
        max_iterations=int(config.get("max_cutting_plane_iterations", 100)),
        time_limit=config.get("time_limit"),
    )
    if capt_solution.channel is None:
        raise RuntimeError(f"simplex-CAPT LP failed: {capt_solution.solver.message}")

    channels = {
        "common_cover": cover_channel,
        "k_ary_rr": rr_channel,
        "optimal_ldp": ldp_solution.channel,
        "simplex_capt": capt_solution.channel,
    }
    solvers: dict[str, SolverInfo] = {
        "common_cover": _analytic_solver("analytic_common_cover", cover_channel, cost, weights),
        "k_ary_rr": _analytic_solver("analytic_k_ary_rr", rr_channel, cost, weights),
        "optimal_ldp": ldp_solution.solver,
        "simplex_capt": capt_solution.solver,
    }
    verifications: dict[str, VerificationResult] = {
        name: (
            capt_verification
            if name == "simplex_capt"
            else verify_robust_channel(
                channel,
                boxes,
                adjacency,
                tolerance=float(config.get("solver_tolerance", 1e-8)),
            )
        )
        for name, channel in channels.items()
    }
    if not all(result.valid for result in verifications.values()):
        invalid = [name for name, result in verifications.items() if not result.valid]
        raise RuntimeError(f"robust verification failed for: {invalid}")

    missing_keys = set(hist.missing_groups)
    positive_counts = [int(value.sum()) for value in hist.counts.values() if value.sum() > 0]
    coverage = {
        "expected_group_count": hist.expected_group_count,
        "observed_group_count": hist.expected_group_count - hist.missing_group_count,
        "missing_group_count": hist.missing_group_count,
        "missing_groups": list(hist.missing_groups),
        "rare_group_count": hist.rare_group_count,
        "hybrid_connectivity_gaps": connectivity_gaps,
        "requires_universal_cover": False,
        "policy": {"rare": "confidence_box", "missing": "full_simplex"},
        "profile": profile,
        "bundle_profiles": [profile],
        "frozen_support_hash": frozen["support_hash"],
        "full_simplex_group_count": len(missing_keys),
    }
    rows: list[dict[str, Any]] = []
    certificate_paths: list[str] = []
    cover_utility = -_objective(cover_channel, cost, weights)
    best_cover_utility = -_objective(best_cover_channel, cost, weights)
    for name, channel in channels.items():
        certificate = make_certificate(
            config=config,
            channel=channel,
            boxes=boxes,
            adjacency=adjacency,
            verification=verifications[name],
            solver=solvers[name],
            component_hashes={
                "encoder": frozen["encoder_hash"],
                "model": frozen["model_hash"],
            },
            split_identifiers=config["splits"],
            dp_parameters={
                "epsilon": config.get("dp_hist_epsilon"),
                "delta": config.get("dp_hist_delta"),
                "contribution_policy": config.get(
                    "contribution_policy", "one-display-per-uuid-day"
                ),
            },
            histogram_counts=hist.counts,
            assignment=frozen["assignment"],
            decoder=frozen["decoder"],
            coverage=coverage,
            groups=hist.groups,
        )
        certificate_path = path / f"certificate-{name}.json"
        certificate.write(certificate_path)
        independent = verify_certificate(certificate_path)
        if not independent.valid:
            raise RuntimeError(f"independent certificate verification failed: {name}")
        certificate_paths.append(certificate_path.name)
        utility = -_objective(channel, cost, weights)
        row_tv = _max_row_tv(channel)
        rows.append(
            {
                "method": name,
                "utility": utility,
                "utility_gain_over_common_cover": utility - cover_utility,
                "utility_gain_over_best_input_independent": utility - best_cover_utility,
                "matches_best_input_independent": bool(
                    np.allclose(channel, best_cover_channel, atol=1e-10, rtol=0)
                ),
                "max_pairwise_row_tv": row_tv,
                "nontrivial_channel": row_tv > 1e-10,
                "is_universal_channel": row_tv <= 1e-10,
                "solver_status": solvers[name].status,
                "solver_objective": solvers[name].objective,
                "solver_runtime_seconds": solvers[name].runtime_seconds,
                "solver_iterations": solvers[name].iterations,
                "solver_primal_gap": solvers[name].primal_gap,
                "solver_constraint_count": solvers[name].constraint_count,
                "solver_cut_count": len(capt_solution.cuts) if name == "simplex_capt" else 0,
                "lp_executed": name in {"optimal_ldp", "simplex_capt"},
                "pre_lp_fallback": False,
                "mechanism_fallback_share": 0.0,
                "universal_cover_activated": False,
                "ldp_max_violation": _ldp_max_violation(channel, epsilon),
                "robust_valid": verifications[name].valid,
                "robust_checked_constraints": verifications[name].checked_constraints,
                "robust_max_violation": verifications[name].max_violation,
                "certificate_valid": independent.valid,
                "certificate_checked_constraints": independent.checked_constraints,
                "certificate_max_violation": independent.max_violation,
                "certificate_path": certificate_path.name,
                "process_peak_rss_bytes_after": _peak_rss_bytes(),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(path / "tables" / "simplex_completion_comparison.csv", index=False)
    results.to_parquet(path / "simplex_completion_comparison.parquet", index=False)
    np.savez_compressed(
        path / "mechanism" / "comparison_channels.npz",
        **channels,
        best_input_independent_cover=best_cover_channel,
    )

    group_rows = []
    for group in hist.groups:
        counts = hist.counts[group.key()]
        group_rows.append(
            {
                "group": group.key(),
                "profile": group.profile,
                "context": group.context,
                "values": "|".join(map(str, group.values)),
                "count": int(counts.sum()),
                "uncertainty_method": boxes[group.key()].method,
                "missing": group.key() in missing_keys,
                "rare_observed": 0 < counts.sum() < int(config.get("min_group_count", 20)),
            }
        )
    pd.DataFrame(group_rows).to_csv(path / "tables" / "simplex_completion_groups.csv", index=False)
    diagnostics = {
        "profile": profile,
        "K": int(config["K"]),
        "L": int(config["L"]),
        "epsilon": epsilon,
        "cert_source_rows": len(cert_source),
        "cert_user_days": len(cert_frame),
        "expected_group_count": hist.expected_group_count,
        "observed_group_count": hist.expected_group_count - hist.missing_group_count,
        "missing_group_count": hist.missing_group_count,
        "rare_group_count": hist.rare_group_count,
        "rare_group_mass": hist.rare_group_mass,
        "minimum_positive_group_count": min(positive_counts),
        "adjacency_count": len(adjacency),
        "hybrid_connectivity_gaps": connectivity_gaps,
        "cp_box_count": sum(box.method == "cp_box" for box in boxes.values()),
        "full_simplex_box_count": sum(box.method == "full_simplex" for box in boxes.values()),
        "frozen_mu_cover_utility": cover_utility,
        "best_input_independent_utility": best_cover_utility,
        "best_input_independent_destination_block": best_cover_destination,
        "simplex_matches_best_input_independent": bool(
            np.allclose(
                channels["simplex_capt"], best_cover_channel, atol=1e-10, rtol=0
            )
        ),
        "source_git_sha": config["source_git_sha"],
        "wall_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "environment": environment(),
    }
    (path / "simplex_completion_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )
    _plot_comparison(results, path)
    _write_report(results, diagnostics, path)
    del cert_source, cert_frame, cert_tokens
    gc.collect()
    finish_run(
        path,
        {
            "dataset": "criteo",
            "experiment": "full_simplex_completion_one_condition",
            "profile": profile,
            "certificates": certificate_paths,
            "source_git_sha": config["source_git_sha"],
            "result_rows": len(results),
        },
    )
    return path
