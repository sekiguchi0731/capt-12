from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capt12.certification.artifact import hash_array, make_certificate, verify_certificate
from capt12.certification.robust import solve_robust_block_lp, verify_robust_channel
from capt12.config import validate_config
from capt12.data.loader import assert_disjoint_splits, assert_no_row_overlap, load_parquet_sample
from capt12.data.preprocessing import build_group_histograms
from capt12.decoders.registry import build_decoder
from capt12.distortions.registry import block_cost_matrix
from capt12.experiments.fixed_support import (
    _expected_frame_from_cartesian,
    _fixed_design,
    _peak_rss_bytes,
)
from capt12.experiments.sampling import select_one_display_per_user_day
from capt12.experiments.simplex_completion import (
    _best_input_independent_channel,
    _build_boxes,
    _ldp_max_violation,
    _max_row_tv,
    _objective,
)
from capt12.mechanisms.diagnostics import UtilityInformativeness, utility_informativeness
from capt12.mechanisms.lp import solve_ldp_block_lp
from capt12.partitions.registry import build_partition
from capt12.pipeline import record_source_provenance
from capt12.privacy.adjacency import build_adjacency, disconnected_hybrid_components
from capt12.utils.artifacts import environment, finish_run, prepare_run


@dataclass(frozen=True)
class UtilityDesign:
    name: str
    label: str
    partition_method: str
    decoder_method: str
    assignment: np.ndarray
    decoder: np.ndarray
    block_cost: np.ndarray
    block_weights: np.ndarray
    diagnostic: UtilityInformativeness
    representation_mode: str = "teacher_kl_fixed"
    representation_objective: str = "teacher_kl"
    representation_token_cost_hash: str = ""


def _build_designs(
    frozen_arrays: Any,
    config: dict[str, Any],
) -> list[UtilityDesign]:
    frequencies = np.asarray(frozen_arrays["frequencies"], dtype=float)
    objective_weights = np.asarray(frozen_arrays["objective_weights"], dtype=float)
    token_scores = np.asarray(frozen_arrays["token_scores"], dtype=float)
    representation_mode = str(config.get("context_representation_mode", "teacher_kl_fixed"))
    representation_objective = (
        str(config.get("context_utility_objective", "teacher_kl"))
        if representation_mode == "objective_aligned"
        else "teacher_kl"
    )
    if representation_mode == "objective_aligned":
        try:
            token_cost = np.asarray(frozen_arrays["representation_token_cost"], dtype=float)
        except KeyError as error:
            raise ValueError(
                "objective-aligned representation requires representation_token_cost"
            ) from error
    else:
        token_cost = np.asarray(frozen_arrays["token_cost"], dtype=float)
    representation_token_cost_hash = hash_array(token_cost)
    current_assignment = np.asarray(frozen_arrays["assignment"], dtype=int)
    current_decoder = np.asarray(frozen_arrays["decoder"], dtype=float)
    cost_weights = (
        np.ones(len(frequencies)) / len(frequencies)
        if config.get("cost_aggregation", "empirical") == "uniform"
        else frequencies
    )
    tolerance = float(config.get("utility_gate_tolerance", 1e-12))

    raw: list[tuple[str, str, str, str, np.ndarray, np.ndarray]] = [
        (
            "current_design_frequency_L16",
            "Current decoder (L=16)",
            "current_frozen",
            "design_frequency",
            current_assignment,
            current_decoder,
        ),
        (
            "current_utility_medoid_L16",
            "Utility medoid (L=16)",
            "current_frozen",
            "utility_medoid",
            current_assignment,
            build_decoder(
                "utility_medoid",
                current_assignment,
                frequencies,
                scores=token_scores,
            ),
        ),
        (
            "current_cost_medoid_L16",
            "Cost medoid (L=16)",
            "current_frozen",
            "cost_medoid",
            current_assignment,
            build_decoder(
                "cost_medoid",
                current_assignment,
                frequencies,
                token_cost=token_cost,
                token_weights=cost_weights,
            ),
        ),
    ]
    joint_assignment = build_partition(
        "weighted_cost_kmedoids",
        frequencies,
        16,
        token_cost=token_cost,
        token_weights=objective_weights,
    )
    raw.append(
        (
            "joint_kmedoids_cost_medoid_L16",
            "Joint k-medoids (L=16)",
            "weighted_cost_kmedoids",
            "cost_medoid",
            joint_assignment,
            build_decoder(
                "cost_medoid",
                joint_assignment,
                frequencies,
                token_cost=token_cost,
                token_weights=cost_weights,
            ),
        )
    )
    singleton_assignment = np.arange(len(frequencies), dtype=int)
    raw.append(
        (
            "singleton_identity_L64",
            "Singleton identity (L=64)",
            "singleton",
            "identity",
            singleton_assignment,
            np.eye(len(frequencies)),
        )
    )

    designs: list[UtilityDesign] = []
    for name, label, partition_method, decoder_method, assignment, decoder in raw:
        l_count = int(decoder.shape[0])
        cost = block_cost_matrix(token_cost, assignment, decoder, cost_weights)
        weights = np.bincount(
            assignment,
            weights=objective_weights,
            minlength=l_count,
        )
        designs.append(
            UtilityDesign(
                name=name,
                label=label,
                partition_method=partition_method,
                decoder_method=decoder_method,
                assignment=assignment,
                decoder=decoder,
                block_cost=cost,
                block_weights=weights,
                diagnostic=utility_informativeness(
                    cost,
                    weights,
                    tolerance=tolerance,
                ),
                representation_mode=representation_mode,
                representation_objective=representation_objective,
                representation_token_cost_hash=representation_token_cost_hash,
            )
        )
    return designs


def _screen_frame(designs: list[UtilityDesign]) -> pd.DataFrame:
    rows = []
    for design in designs:
        diagnostic = design.diagnostic
        rows.append(
            {
                "design": design.name,
                "label": design.label,
                "L": len(design.block_weights),
                "partition_method": design.partition_method,
                "decoder_method": design.decoder_method,
                "representation_mode": design.representation_mode,
                "representation_objective": design.representation_objective,
                "representation_token_cost_hash": design.representation_token_cost_hash,
                "D_constant": diagnostic.constant_distortion,
                "D_free": diagnostic.free_distortion,
                "G_info": diagnostic.information_gap,
                "unique_row_argmins": diagnostic.unique_argmin_count,
                "row_argmins": "|".join(map(str, diagnostic.row_argmins)),
                "no_privacy_max_row_tv": diagnostic.no_privacy_max_row_tv,
                "utility_gate_passed": diagnostic.informative,
                "utility_gate_failure_reasons": "|".join(diagnostic.failure_reasons),
                "assignment_hash": hash_array(design.assignment),
                "decoder_hash": hash_array(design.decoder),
                "block_cost_hash": hash_array(design.block_cost),
            }
        )
    return pd.DataFrame(rows)


def _plot_results(screen: pd.DataFrame, privacy: pd.DataFrame, path: Path) -> None:
    order = screen["design"].tolist()
    labels = screen.set_index("design").loc[order, "label"].tolist()
    merged = (
        screen.set_index("design")
        .join(
            privacy.set_index("design")[["capt_gain_over_constant", "capt_max_row_tv"]],
            how="left",
        )
        .loc[order]
    )
    y = np.arange(len(order))
    height = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))

    axes[0].barh(
        y - height / 2,
        merged["G_info"],
        height,
        color="#777777",
        edgecolor="#222222",
        label="No-privacy information gap",
    )
    axes[0].barh(
        y + height / 2,
        merged["capt_gain_over_constant"].fillna(0),
        height,
        color="#2563A6",
        edgecolor="#222222",
        hatch="//",
        label="Certified CAPT gain over best constant",
    )
    axes[0].set_title("Utility available before and after privacy")
    axes[0].set_xlabel("distortion reduction (higher is better)")
    axes[0].legend(fontsize=8)

    axes[1].barh(
        y - height / 2,
        merged["no_privacy_max_row_tv"],
        height,
        color="#777777",
        edgecolor="#222222",
        label="No-privacy optimum",
    )
    axes[1].barh(
        y + height / 2,
        merged["capt_max_row_tv"].fillna(0),
        height,
        color="#D97706",
        edgecolor="#222222",
        hatch="xx",
        label="Certified CAPT",
    )
    axes[1].set_title("Input dependence of optimized channels")
    axes[1].set_xlabel("maximum pairwise row TV")
    axes[1].set_xlim(0, 1.05)
    axes[1].legend(fontsize=8)

    for axis in axes:
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.22, color="#777777")
    fig.suptitle("Criteo constrained_2 utility-design diagnostics; full D_cert, epsilon=1")
    fig.text(
        0.5,
        0.01,
        "Current L=16 decoder fails the utility gate and is not re-solved; "
        "each solved design has 374 ordered full-simplex/full-simplex edges.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fixed_time = datetime(2000, 1, 1, tzinfo=UTC)
    fig.savefig(
        path / "figures" / "utility_design_diagnostic.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "CAPT-12",
            "Producer": "CAPT-12",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        path / "figures" / "utility_design_diagnostic.png",
        bbox_inches="tight",
        dpi=220,
        metadata={"Software": "CAPT-12"},
    )
    plt.close(fig)


def _write_report(
    screen: pd.DataFrame,
    privacy: pd.DataFrame,
    metadata: dict[str, Any],
    path: Path,
) -> None:
    baseline = screen.iloc[0]
    informative = screen.loc[screen["utility_gate_passed"]]
    advantage_count = int((privacy["capt_utility_advantage_over_ldp"] > 1e-12).sum())
    nontrivial_count = int(privacy["capt_nontrivial_channel"].sum())
    nontrivial_names = privacy.loc[privacy["capt_nontrivial_channel"], "label"].tolist()
    report = f"""# Criteo utility-aware design and simplex-CAPT diagnostic

## Scope

- Frozen source: full D_model and D_design; certificate population: {metadata["cert_user_days"]:,} one-display-per-user-day D_cert contributions.
- Profile: `features_kv_bits_constrained_2`; epsilon=1; expected Cartesian support fixed at {metadata["expected_group_count"]} groups.
- Compared five fixed channel classes: current decoder, utility medoid, cost medoid, joint weighted k-medoids/cost-medoid, and L=K=64 singleton identity.

## P0-A utility gate

The current L=16 design fails before privacy optimization: `D_constant={baseline["D_constant"]:.9g}`, `D_free={baseline["D_free"]:.9g}`, and `G_info=0`; all source-block rows select destination block 6. It is recorded but not re-solved.

The other {len(informative)} designs pass the gate. Their no-privacy information gaps range from {informative["G_info"].min():.9g} to {informative["G_info"].max():.9g}, confirming that utility-aware decoders/partitions restore a reason to use input-dependent channels.

## Privacy optimization

Of the {len(privacy)} informative designs, {nontrivial_count}/{len(privacy)} produce CAPT channels with positive row TV: {", ".join(nontrivial_names)}. Every CAPT certificate passes independent verification. However, {advantage_count}/{len(privacy)} have utility strictly above optimal LDP at tolerance 1e-12.

Every solved design contains 374 ordered adjacent pairs whose two uncertainty sets are full simplexes. One such pair compiles to the global row-wise epsilon-LDP constraints for the shared channel. Accordingly, each simplex-CAPT optimum has the same objective as its optimal-LDP comparator. Cost medoid, joint k-medoids, and singleton identity repair the second degeneration (constant optimum); utility medoid remains constant after privacy. None repairs the first degeneration (CAPT feasible region equals LDP).

## Decision

Do not start the epsilon/L/sample-size grid yet. The next experiment must change the support/completion design without silently dropping protected groups—for example, a predeclared coarsening or another auditable structural assumption—and must include all supported/unsupported boundary constraints in the certificate. Avazu should remain support-only until that rule is defined.
"""
    (path / "utility_design_report.md").write_text(report)


def run_utility_design_diagnostic(config: dict[str, Any]) -> Path:
    """Screen utility designs, then compare informative CAPT classes with LDP."""
    config = validate_config(config)
    required_splits = {"D_model", "D_design", "D_cert", "D_attack_train", "D_test"}
    if set(config.get("splits", {})) != required_splits:
        raise ValueError("utility design diagnostic requires the five fixed temporal splits")
    assert_disjoint_splits(config["splits"])
    if config.get("profiles") != ["features_kv_bits_constrained_2"]:
        raise ValueError("utility design diagnostic currently requires constrained_2 only")
    if float(config.get("epsilon", -1)) != 1.0:
        raise ValueError("the first utility design diagnostic requires epsilon=1")
    if config.get("missing_group_policy") != "full_simplex":
        raise ValueError("utility design diagnostic requires full-simplex completion")
    if config.get("rare_group_policy") != "confidence_box":
        raise ValueError("observed rare groups must retain their confidence boxes")

    config = record_source_provenance(config)
    path = prepare_run(config)
    started = time.perf_counter()
    frozen, support_table = _fixed_design(config, path)
    support_table.to_csv(path / "tables" / "frozen_support.csv", index=False)
    frozen_arrays = np.load(path / "mechanism" / "frozen_design.npz")
    designs = _build_designs(frozen_arrays, config)
    screen = _screen_frame(designs)
    screen.to_csv(path / "tables" / "utility_design_screen.csv", index=False)

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

    results: list[dict[str, Any]] = []
    certificate_paths: list[str] = []
    expected_group_count = 0
    for design in designs:
        if not design.diagnostic.informative:
            continue
        l_count = len(design.block_weights)
        hist = build_group_histograms(
            cert_frame,
            cert_tokens,
            profile=profile,
            context_columns=contexts,
            alphabet_size=l_count,
            token_to_block=design.assignment,
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
            raise RuntimeError(f"invalid completed support for utility design: {design.name}")
        boxes = _build_boxes(hist.counts, adjacency_count=len(adjacency), config=config)
        full_simplex_keys = {key for key, box in boxes.items() if box.method == "full_simplex"}
        full_full_edges = sum(
            pair.left in full_simplex_keys and pair.right in full_simplex_keys for pair in adjacency
        )

        ldp_solution = solve_ldp_block_lp(
            design.block_cost,
            design.block_weights,
            float(config["epsilon"]),
            tolerance=float(config.get("solver_tolerance", 1e-8)),
            time_limit=config.get("time_limit"),
        )
        if ldp_solution.channel is None:
            raise RuntimeError(f"optimal LDP failed for {design.name}")
        capt_solution, capt_verification = solve_robust_block_lp(
            design.block_cost,
            design.block_weights,
            boxes,
            adjacency,
            tolerance=float(config.get("solver_tolerance", 1e-8)),
            max_iterations=int(config.get("max_cutting_plane_iterations", 100)),
            time_limit=config.get("time_limit"),
        )
        if capt_solution.channel is None or not capt_verification.valid:
            raise RuntimeError(f"simplex-CAPT failed for {design.name}")
        ldp_verification = verify_robust_channel(
            ldp_solution.channel,
            boxes,
            adjacency,
            tolerance=float(config.get("solver_tolerance", 1e-8)),
        )
        if not ldp_verification.valid:
            raise RuntimeError(f"optimal LDP robust verification failed for {design.name}")

        best_constant_channel, best_constant_block = _best_input_independent_channel(
            design.block_cost,
            design.block_weights,
        )
        d_constant = _objective(
            best_constant_channel,
            design.block_cost,
            design.block_weights,
        )
        d_ldp = _objective(
            ldp_solution.channel,
            design.block_cost,
            design.block_weights,
        )
        d_capt = _objective(
            capt_solution.channel,
            design.block_cost,
            design.block_weights,
        )
        cell_config = dict(config)
        cell_config["utility_design_diagnostic"] = {
            "design": design.name,
            "partition_method": design.partition_method,
            "decoder_method": design.decoder_method,
            "L": l_count,
            "utility_gate": design.diagnostic.to_dict(),
        }
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
            "full_simplex_group_count": len(full_simplex_keys),
            "full_simplex_ordered_adjacency_count": full_full_edges,
        }
        certificate = make_certificate(
            config=cell_config,
            channel=capt_solution.channel,
            boxes=boxes,
            adjacency=adjacency,
            verification=capt_verification,
            solver=capt_solution.solver,
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
            assignment=design.assignment,
            decoder=design.decoder,
            coverage=coverage,
            groups=hist.groups,
        )
        certificate_path = path / f"certificate-{design.name}.json"
        certificate.write(certificate_path)
        independent = verify_certificate(certificate_path)
        if not independent.valid:
            raise RuntimeError(f"independent certificate verification failed: {design.name}")
        certificate_paths.append(certificate_path.name)
        np.savez_compressed(
            path / "mechanism" / f"{design.name}.npz",
            assignment=design.assignment,
            decoder=design.decoder,
            block_cost=design.block_cost,
            block_weights=design.block_weights,
            optimal_ldp=ldp_solution.channel,
            simplex_capt=capt_solution.channel,
            best_constant=best_constant_channel,
        )

        capt_row_tv = _max_row_tv(capt_solution.channel)
        ldp_row_tv = _max_row_tv(ldp_solution.channel)
        utility_advantage = d_ldp - d_capt
        tolerance = design.diagnostic.tolerance
        seed_cut_count = sum(
            int(cut.get("constraint_count", 0))
            for cut in capt_solution.cuts
            if cut.get("source") == "full_simplex_ldp_seed"
        )
        results.append(
            {
                "design": design.name,
                "label": design.label,
                "L": l_count,
                "partition_method": design.partition_method,
                "decoder_method": design.decoder_method,
                "D_constant": d_constant,
                "D_free": design.diagnostic.free_distortion,
                "G_info": design.diagnostic.information_gap,
                "best_constant_block": best_constant_block,
                "D_optimal_ldp": d_ldp,
                "D_simplex_capt": d_capt,
                "optimal_ldp_gain_over_constant": d_constant - d_ldp,
                "capt_gain_over_constant": d_constant - d_capt,
                "capt_utility_advantage_over_ldp": utility_advantage,
                "ldp_max_row_tv": ldp_row_tv,
                "capt_max_row_tv": capt_row_tv,
                "capt_nontrivial_channel": capt_row_tv > 1e-10,
                "capt_equals_ldp_objective": abs(utility_advantage) <= tolerance,
                "capt_ldp_channel_max_abs_difference": float(
                    np.max(np.abs(capt_solution.channel - ldp_solution.channel))
                ),
                "full_simplex_group_count": len(full_simplex_keys),
                "full_simplex_ordered_adjacency_count": full_full_edges,
                "hybrid_connectivity_gaps": connectivity_gaps,
                "solver_status": capt_solution.solver.status,
                "capt_solver_runtime_seconds": capt_solution.solver.runtime_seconds,
                "ldp_solver_runtime_seconds": ldp_solution.solver.runtime_seconds,
                "capt_solver_constraint_count": capt_solution.solver.constraint_count,
                "compiled_ldp_constraint_count": seed_cut_count,
                "support_oracle_cut_count": sum(
                    cut.get("source") == "support_oracle" for cut in capt_solution.cuts
                ),
                "ldp_direct_max_violation": _ldp_max_violation(
                    capt_solution.channel, float(config["epsilon"])
                ),
                "robust_valid": capt_verification.valid,
                "robust_checked_constraints": capt_verification.checked_constraints,
                "robust_max_violation": capt_verification.max_violation,
                "certificate_valid": independent.valid,
                "certificate_checked_constraints": independent.checked_constraints,
                "certificate_max_violation": independent.max_violation,
                "mechanism_fallback_share": 0.0,
                "pre_lp_fallback": False,
                "success_row_tv": capt_row_tv > 1e-10,
                "success_capt_above_ldp": utility_advantage > tolerance,
                "success_certificate": independent.valid,
                "success_not_constant_only": d_constant - d_capt > tolerance,
                "success_all": bool(
                    capt_row_tv > 1e-10
                    and utility_advantage > tolerance
                    and independent.valid
                    and d_constant - d_capt > tolerance
                ),
                "certificate_path": certificate_path.name,
                "process_peak_rss_bytes_after": _peak_rss_bytes(),
            }
        )
        expected_group_count = hist.expected_group_count

    privacy = pd.DataFrame(results)
    privacy.to_csv(path / "tables" / "utility_design_privacy.csv", index=False)
    privacy.to_parquet(path / "utility_design_privacy.parquet", index=False)
    _plot_results(screen, privacy, path)
    total_seconds = time.perf_counter() - started
    metadata = {
        "experiment": "criteo_utility_design_full_simplex_diagnostic",
        "source_git_sha": config["source_git_sha"],
        "profile": profile,
        "cert_source_rows": len(cert_source),
        "cert_user_days": len(cert_frame),
        "expected_group_count": expected_group_count,
        "screened_design_count": len(screen),
        "utility_gate_pass_count": int(screen["utility_gate_passed"].sum()),
        "privacy_solved_design_count": len(privacy),
        "successful_design_count": int(privacy["success_all"].sum()),
        "wall_seconds": total_seconds,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "environment": environment(),
    }
    (path / "utility_design_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    _write_report(screen, privacy, metadata, path)
    del cert_source, cert_frame, cert_tokens
    gc.collect()
    finish_run(
        path,
        {
            "dataset": "criteo",
            "experiment": metadata["experiment"],
            "profile": profile,
            "certificates": certificate_paths,
            "source_git_sha": config["source_git_sha"],
            "screened_design_count": len(screen),
            "privacy_solved_design_count": len(privacy),
            "successful_design_count": metadata["successful_design_count"],
        },
    )
    return path
