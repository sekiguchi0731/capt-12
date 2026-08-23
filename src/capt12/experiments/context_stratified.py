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

from capt12.certification.artifact import (
    hash_array,
    make_certificate,
    verify_certificate,
)
from capt12.certification.robust import solve_robust_block_lp, verify_robust_channel
from capt12.confidence.boxes import ConfidenceBox
from capt12.config import validate_config
from capt12.data.loader import assert_disjoint_splits, assert_no_row_overlap, load_parquet_sample
from capt12.data.preprocessing import build_group_histograms
from capt12.distortions.registry import DISTORTION_REGISTRY, block_cost_matrix
from capt12.evaluation.metrics import prediction_metrics
from capt12.experiments.fixed_support import (
    _expected_frame_from_cartesian,
    _fixed_design,
    _peak_rss_bytes,
)
from capt12.experiments.sampling import select_one_display_per_user_day
from capt12.experiments.simplex_completion import (
    _best_input_independent_channel,
    _build_boxes,
    _max_row_tv,
    _objective,
)
from capt12.experiments.utility_design import UtilityDesign, _build_designs
from capt12.mechanisms.lp import ChannelSolution, lift_block_channel, solve_ldp_block_lp
from capt12.pipeline import record_source_provenance
from capt12.privacy.adjacency import AdjacentPair, Group, build_adjacency
from capt12.utils.artifacts import (
    environment,
    finish_run,
    prepare_run,
    sha256_file,
    write_sol_review_bundle,
)
from capt12.utils.progress import ProgressLogger, process_memory_bytes


@dataclass(frozen=True)
class ContextObjective:
    context: str
    design_mass: float
    token_probabilities: np.ndarray
    block_cost: np.ndarray
    block_weights: np.ndarray


@dataclass
class ContextCell:
    design: UtilityDesign
    objective: ContextObjective
    groups: list[Group]
    counts: dict[str, np.ndarray]
    boxes: dict[str, ConfidenceBox]
    adjacency: list[AdjacentPair]
    constant_channel: np.ndarray
    ldp_solution: ChannelSolution
    capt_solution: ChannelSolution
    capt_verification: Any
    full_simplex_keys: set[str]
    full_full_edges: int
    rare_group_count: int


def aggregate_context_objective(
    objectives: list[ContextObjective],
) -> tuple[np.ndarray, np.ndarray]:
    """Return one cost/weight pair representing the mass-weighted objective."""
    if not objectives:
        raise ValueError("at least one context objective is required")
    dimension = len(objectives[0].block_weights)
    weighted_cost = np.zeros((dimension, dimension), dtype=float)
    weights = np.zeros(dimension, dtype=float)
    for objective in objectives:
        if objective.block_cost.shape != (dimension, dimension):
            raise ValueError("context objectives must share one channel dimension")
        conditional = np.asarray(objective.block_weights, dtype=float)
        conditional = conditional / conditional.sum()
        mass = float(objective.design_mass)
        weights += mass * conditional
        weighted_cost += mass * conditional[:, None] * objective.block_cost
    if weights.sum() <= 0:
        raise ValueError("context design masses must have positive total mass")
    cost = np.divide(
        weighted_cost,
        weights[:, None],
        out=np.zeros_like(weighted_cost),
        where=weights[:, None] > 0,
    )
    return cost, weights


def _context_probability_grid(
    reference: Any,
    contexts: list[str],
    *,
    context_column: str,
    k: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for context in contexts:
        frame = pd.DataFrame(
            {
                "__token__": np.arange(k, dtype=int),
                context_column: context,
            }
        )
        result[context] = reference.predict(frame)
    return result


def _context_objectives(
    frozen: dict[str, Any],
    design: UtilityDesign,
    config: dict[str, Any],
) -> list[ContextObjective]:
    context_column = config["context_cols"][0]
    contexts = sorted({str(value[-1]) for value in frozen["cartesian_support"]})
    k = len(frozen["assignment"])
    probability_grid = _context_probability_grid(
        frozen["reference"],
        contexts,
        context_column=context_column,
        k=k,
    )
    stored_contexts = list(map(str, frozen["context_levels"]))
    stored_index = {value: index for index, value in enumerate(stored_contexts)}
    stored_weights = np.asarray(frozen["token_context_weights"], dtype=float)
    global_weights = np.asarray(frozen["frequencies"], dtype=float)
    context_mass = {
        context: sum(
            probability
            for group, probability in frozen["design_only_probability"].items()
            if str(group[-1]) == context
        )
        for context in contexts
    }
    distortion = config.get("distortion", "bernoulli_kl")
    function = DISTORTION_REGISTRY[distortion]
    eta = float(config.get("distortion_clip", 1e-6))
    objectives: list[ContextObjective] = []
    for context in contexts:
        probabilities = probability_grid[context]
        if distortion == "retention":
            token_cost = 1 - np.eye(k)
        else:
            token_cost = function(
                probabilities[:, None],
                probabilities[None, :],
                eta,
            )
        if context in stored_index and context_mass[context] > 0:
            token_weights = stored_weights[:, stored_index[context]].copy()
            token_weights /= token_weights.sum()
        else:
            token_weights = global_weights.copy()
            token_weights /= token_weights.sum()
        block_cost = block_cost_matrix(
            token_cost,
            design.assignment,
            design.decoder,
            token_weights,
        )
        block_weights = np.bincount(
            design.assignment,
            weights=token_weights,
            minlength=len(design.block_weights),
        )
        objectives.append(
            ContextObjective(
                context=context,
                design_mass=float(context_mass[context]),
                token_probabilities=probabilities,
                block_cost=block_cost,
                block_weights=block_weights,
            )
        )
    total_mass = sum(value.design_mass for value in objectives)
    if not np.isclose(total_mass, 1.0, atol=1e-10):
        raise RuntimeError(f"context design masses do not sum to one: {total_mass}")
    return objectives


def _split_histogram_by_context(
    groups: list[Group],
    counts: dict[str, np.ndarray],
) -> dict[str, tuple[list[Group], dict[str, np.ndarray]]]:
    contexts = sorted({str(group.context) for group in groups})
    return {
        context: (
            [group for group in groups if str(group.context) == context],
            {group.key(): counts[group.key()] for group in groups if str(group.context) == context},
        )
        for context in contexts
    }


def _solve_design(
    design: UtilityDesign,
    objectives: list[ContextObjective],
    groups: list[Group],
    counts: dict[str, np.ndarray],
    config: dict[str, Any],
    progress: ProgressLogger,
    checkpoint_dir: Path | None = None,
) -> tuple[list[ContextCell], ChannelSolution, ChannelSolution, Any]:
    by_context = _split_histogram_by_context(groups, counts)
    objective_by_context = {value.context: value for value in objectives}
    context_count = len(by_context)
    tolerance = float(config.get("solver_tolerance", 1e-8))
    solver_verbose = bool(config.get("solver_verbose", False))
    heartbeat_seconds = float(config.get("solver_heartbeat_seconds", 60.0))
    cut_formulation = str(config.get("robust_cut_formulation", "paired_witness"))
    resume_checkpoint = bool(config.get("resume_cutting_plane", False))
    checkpoint_every = int(config.get("cutting_plane_checkpoint_every", 1))
    cells: list[ContextCell] = []
    progress.emit(
        "design_context_solve_started",
        design=design.name,
        dimension=len(design.block_weights),
        context_count=context_count,
        solver_verbose=solver_verbose,
        heartbeat_seconds=heartbeat_seconds,
        tolerance=tolerance,
        time_limit_seconds=config.get("time_limit"),
        max_cutting_plane_iterations=int(config.get("max_cutting_plane_iterations", 100)),
        robust_cut_formulation=cut_formulation,
        resume_cutting_plane=resume_checkpoint,
        cutting_plane_checkpoint_every=checkpoint_every,
        **process_memory_bytes(),
    )
    for context_index, (context, (context_groups, context_counts)) in enumerate(
        by_context.items(), start=1
    ):
        context_started = time.perf_counter()
        adjacency = build_adjacency(
            context_groups,
            config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
            float(config["epsilon"]),
            config.get("epsilon_by_attr", {}),
        )
        cell_config = dict(config)
        cell_config["alpha_cert"] = float(config.get("alpha_cert", 0.05)) / context_count
        boxes = _build_boxes(
            context_counts,
            adjacency_count=len(adjacency),
            config=cell_config,
        )
        full_simplex = {key for key, box in boxes.items() if box.method == "full_simplex"}
        full_edges = sum(
            pair.left in full_simplex and pair.right in full_simplex for pair in adjacency
        )
        objective = objective_by_context[context]
        group_totals = np.asarray(
            [int(value.sum()) for value in context_counts.values()], dtype=int
        )
        positive_totals = group_totals[group_totals > 0]
        progress.emit(
            "context_started",
            design=design.name,
            context=context,
            context_index=context_index,
            context_count=context_count,
            context_percent_complete=100.0 * (context_index - 1) / context_count,
            design_mass=objective.design_mass,
            dimension=len(design.block_weights),
            variable_count=len(design.block_weights) ** 2,
            group_count=len(context_groups),
            observed_group_count=int(np.sum(group_totals > 0)),
            missing_group_count=int(np.sum(group_totals == 0)),
            rare_group_count=int(
                np.sum((group_totals > 0) & (group_totals < int(config.get("min_group_count", 20))))
            ),
            min_positive_group_count=(
                int(np.min(positive_totals)) if len(positive_totals) else None
            ),
            median_positive_group_count=(
                float(np.median(positive_totals)) if len(positive_totals) else None
            ),
            max_group_count=int(np.max(group_totals)),
            adjacency_count=len(adjacency),
            full_simplex_box_count=len(full_simplex),
            full_simplex_ordered_edge_count=int(full_edges),
            nominal_robust_constraint_count=(len(adjacency) * len(design.block_weights)),
            alpha_cert=cell_config["alpha_cert"],
            **process_memory_bytes(),
        )
        constant, _ = _best_input_independent_channel(
            objective.block_cost,
            objective.block_weights,
        )
        progress.emit(
            "context_ldp_started",
            design=design.name,
            context=context,
            context_index=context_index,
            context_count=context_count,
            dimension=len(design.block_weights),
            expected_ldp_constraint_count=(
                len(design.block_weights)
                * (len(design.block_weights) - 1)
                * len(design.block_weights)
            ),
        )
        ldp = solve_ldp_block_lp(
            objective.block_cost,
            objective.block_weights,
            float(config["epsilon"]),
            tolerance=tolerance,
            time_limit=config.get("time_limit"),
            progress=progress,
            progress_label=f"{design.name}/context={context}/ldp",
            solver_verbose=solver_verbose,
            heartbeat_seconds=heartbeat_seconds,
        )
        progress.emit(
            "context_ldp_finished",
            design=design.name,
            context=context,
            context_index=context_index,
            status=ldp.solver.status,
            runtime_seconds=ldp.solver.runtime_seconds,
            iterations=ldp.solver.iterations,
            objective=ldp.solver.objective,
            variable_count=ldp.solver.variable_count,
            constraint_count=ldp.solver.constraint_count,
            primal_gap=ldp.solver.primal_gap,
            dual_gap=ldp.solver.dual_gap,
            **process_memory_bytes(),
        )
        progress.emit(
            "context_capt_started",
            design=design.name,
            context=context,
            context_index=context_index,
            context_count=context_count,
            adjacency_count=len(adjacency),
            full_simplex_ordered_edge_count=int(full_edges),
            pure_ldp_shortcut_expected=bool(full_edges),
        )
        capt, verification = solve_robust_block_lp(
            objective.block_cost,
            objective.block_weights,
            boxes,
            adjacency,
            tolerance=tolerance,
            max_iterations=int(config.get("max_cutting_plane_iterations", 100)),
            time_limit=config.get("time_limit"),
            progress=progress,
            progress_label=f"{design.name}/context={context}/capt",
            solver_verbose=solver_verbose,
            heartbeat_seconds=heartbeat_seconds,
            cut_formulation=cut_formulation,
            checkpoint_path=(
                checkpoint_dir / design.name / f"context-{context_index:02d}.npz"
                if checkpoint_dir is not None
                else None
            ),
            resume_checkpoint=resume_checkpoint,
            checkpoint_every=checkpoint_every,
        )
        progress.emit(
            "context_capt_finished",
            design=design.name,
            context=context,
            context_index=context_index,
            status=capt.solver.status,
            runtime_seconds=capt.solver.runtime_seconds,
            iterations=capt.solver.iterations,
            objective=capt.solver.objective,
            variable_count=capt.solver.variable_count,
            constraint_count=capt.solver.constraint_count,
            support_cut_count=len(capt.cuts),
            robust_valid=verification.valid,
            robust_checked_constraints=verification.checked_constraints,
            robust_max_violation=verification.max_violation,
            **process_memory_bytes(),
        )
        if ldp.channel is None or capt.channel is None or not verification.valid:
            raise RuntimeError(f"context optimization failed: {design.name}/{context}")
        if not verify_robust_channel(
            ldp.channel,
            boxes,
            adjacency,
            tolerance=tolerance,
        ).valid:
            raise RuntimeError(f"context LDP verification failed: {design.name}/{context}")
        cells.append(
            ContextCell(
                design=design,
                objective=objective,
                groups=context_groups,
                counts=context_counts,
                boxes=boxes,
                adjacency=adjacency,
                constant_channel=constant,
                ldp_solution=ldp,
                capt_solution=capt,
                capt_verification=verification,
                full_simplex_keys=full_simplex,
                full_full_edges=full_edges,
                rare_group_count=sum(
                    0 < int(value.sum()) < int(config.get("min_group_count", 20))
                    for value in context_counts.values()
                ),
            )
        )
        progress.emit(
            "context_finished",
            design=design.name,
            context=context,
            context_index=context_index,
            context_count=context_count,
            context_seconds=time.perf_counter() - context_started,
            ldp_objective=ldp.solver.objective,
            capt_objective=capt.solver.objective,
            capt_advantage_over_ldp=(
                float(ldp.solver.objective - capt.solver.objective)
                if ldp.solver.objective is not None and capt.solver.objective is not None
                else None
            ),
            capt_row_tv=_max_row_tv(capt.channel),
            **process_memory_bytes(),
        )

    progress.emit(
        "shared_objective_build_started",
        design=design.name,
        context_count=context_count,
    )
    aggregate_cost, aggregate_weights = aggregate_context_objective(objectives)
    all_boxes = {key: value for cell in cells for key, value in cell.boxes.items()}
    all_adjacency = [pair for cell in cells for pair in cell.adjacency]
    progress.emit(
        "shared_baseline_started",
        design=design.name,
        dimension=len(design.block_weights),
        box_count=len(all_boxes),
        adjacency_count=len(all_adjacency),
        nominal_robust_constraint_count=(len(all_adjacency) * len(design.block_weights)),
        **process_memory_bytes(),
    )
    shared_ldp = solve_ldp_block_lp(
        aggregate_cost,
        aggregate_weights,
        float(config["epsilon"]),
        tolerance=tolerance,
        time_limit=config.get("time_limit"),
        progress=progress,
        progress_label=f"{design.name}/shared/ldp",
        solver_verbose=solver_verbose,
        heartbeat_seconds=heartbeat_seconds,
    )
    progress.emit(
        "shared_ldp_finished",
        design=design.name,
        status=shared_ldp.solver.status,
        runtime_seconds=shared_ldp.solver.runtime_seconds,
        iterations=shared_ldp.solver.iterations,
        objective=shared_ldp.solver.objective,
        constraint_count=shared_ldp.solver.constraint_count,
        **process_memory_bytes(),
    )
    shared_capt, shared_verification = solve_robust_block_lp(
        aggregate_cost,
        aggregate_weights,
        all_boxes,
        all_adjacency,
        tolerance=tolerance,
        max_iterations=int(config.get("max_cutting_plane_iterations", 100)),
        time_limit=config.get("time_limit"),
        progress=progress,
        progress_label=f"{design.name}/shared/capt",
        solver_verbose=solver_verbose,
        heartbeat_seconds=heartbeat_seconds,
        cut_formulation=cut_formulation,
        checkpoint_path=(
            checkpoint_dir / design.name / "shared.npz"
            if checkpoint_dir is not None
            else None
        ),
        resume_checkpoint=resume_checkpoint,
        checkpoint_every=checkpoint_every,
    )
    progress.emit(
        "shared_capt_finished",
        design=design.name,
        status=shared_capt.solver.status,
        runtime_seconds=shared_capt.solver.runtime_seconds,
        iterations=shared_capt.solver.iterations,
        objective=shared_capt.solver.objective,
        constraint_count=shared_capt.solver.constraint_count,
        robust_valid=shared_verification.valid,
        robust_checked_constraints=shared_verification.checked_constraints,
        robust_max_violation=shared_verification.max_violation,
        **process_memory_bytes(),
    )
    if shared_ldp.channel is None or shared_capt.channel is None or not shared_verification.valid:
        raise RuntimeError(f"shared baseline failed: {design.name}")
    return cells, shared_ldp, shared_capt, shared_verification


def _aggregate_rows(
    design: UtilityDesign,
    cells: list[ContextCell],
    shared_ldp: ChannelSolution,
    shared_capt: ChannelSolution,
) -> list[dict[str, Any]]:
    methods = {
        "context_constant": lambda cell: cell.constant_channel,
        "context_ldp": lambda cell: cell.ldp_solution.channel,
        "context_capt": lambda cell: cell.capt_solution.channel,
        "shared_ldp": lambda cell: shared_ldp.channel,
        "shared_capt": lambda cell: shared_capt.channel,
    }
    rows = []
    for method, channel_for in methods.items():
        distortion = sum(
            cell.objective.design_mass
            * _objective(
                channel_for(cell),
                cell.objective.block_cost,
                cell.objective.block_weights,
            )
            for cell in cells
        )
        rows.append(
            {
                "design": design.name,
                "label": design.label,
                "L": len(design.block_weights),
                "method": method,
                "aggregate_distortion": distortion,
            }
        )
    constant = next(
        row["aggregate_distortion"] for row in rows if row["method"] == "context_constant"
    )
    for row in rows:
        row["gain_over_context_constant"] = constant - row["aggregate_distortion"]
    return rows


def _context_rows(cells: list[ContextCell]) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        constant = _objective(
            cell.constant_channel,
            cell.objective.block_cost,
            cell.objective.block_weights,
        )
        ldp = _objective(
            cell.ldp_solution.channel,
            cell.objective.block_cost,
            cell.objective.block_weights,
        )
        capt = _objective(
            cell.capt_solution.channel,
            cell.objective.block_cost,
            cell.objective.block_weights,
        )
        rows.append(
            {
                "design": cell.design.name,
                "label": cell.design.label,
                "L": len(cell.design.block_weights),
                "public_context": cell.objective.context,
                "design_mass": cell.objective.design_mass,
                "expected_group_count": len(cell.groups),
                "missing_group_count": len(cell.full_simplex_keys),
                "rare_group_count": cell.rare_group_count,
                "full_simplex_ordered_edge_count": cell.full_full_edges,
                "ldp_degraded": cell.full_full_edges > 0,
                "D_context_constant": constant,
                "D_context_ldp": ldp,
                "D_context_capt": capt,
                "capt_gain_over_constant": constant - capt,
                "capt_advantage_over_context_ldp": ldp - capt,
                "context_ldp_row_tv": _max_row_tv(cell.ldp_solution.channel),
                "context_capt_row_tv": _max_row_tv(cell.capt_solution.channel),
                "capt_equals_context_ldp_channel": bool(
                    np.allclose(
                        cell.capt_solution.channel,
                        cell.ldp_solution.channel,
                        atol=1e-10,
                        rtol=0,
                    )
                ),
                "robust_valid": cell.capt_verification.valid,
                "robust_checked_constraints": cell.capt_verification.checked_constraints,
                "robust_max_violation": cell.capt_verification.max_violation,
                "capt_solver_runtime_seconds": cell.capt_solution.solver.runtime_seconds,
                "ldp_solver_runtime_seconds": cell.ldp_solution.solver.runtime_seconds,
            }
        )
    return rows


def _support_origin_rows(
    cells: list[ContextCell],
    design_probability: dict[tuple[str, ...], float],
) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        edge_counts: dict[str, int] = {key: 0 for key in cell.full_simplex_keys}
        for pair in cell.adjacency:
            if pair.left in cell.full_simplex_keys and pair.right in cell.full_simplex_keys:
                edge_counts[pair.left] += 1
                edge_counts[pair.right] += 1
        for group in cell.groups:
            sensitive = "|".join(map(str, group.values))
            context = str(group.context)
            rows.append(
                {
                    "design": cell.design.name,
                    "public_context": context,
                    "sensitive_value": sensitive,
                    "sensitive_origin": (
                        "unified_unknown" if "__UNKNOWN__" in group.values else "known"
                    ),
                    "context_origin": (
                        "sentinel" if context in {"__MISSING__", "__OTHER__"} else "known"
                    ),
                    "design_mass": design_probability.get(
                        (*tuple(map(str, group.values)), context),
                        0.0,
                    ),
                    "certificate_count": int(cell.counts[group.key()].sum()),
                    "full_simplex": group.key() in cell.full_simplex_keys,
                    "full_full_edge_incidence": edge_counts.get(group.key(), 0),
                }
            )
    return rows


def _channel_manifest(
    path: Path,
    profile: str,
    context_column: str,
    mapper_hash: str,
    designs: list[UtilityDesign],
    cells_by_design: dict[str, list[ContextCell]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": 1,
        "profile": profile,
        "public_context_column": context_column,
        "channel_selector_inputs": ["Z", "profile", context_column],
        "protected_value_used_online": False,
        "runtime_mapper_hash": mapper_hash,
        "sensitive_coarsening": {
            "missing": "__UNKNOWN__",
            "unseen": "__UNKNOWN__",
        },
        "designs": {},
    }
    for design in designs:
        cells = cells_by_design[design.name]
        ordered = sorted(cells, key=lambda cell: cell.objective.context)
        channels = np.stack([cell.capt_solution.channel for cell in ordered])
        np.savez_compressed(
            path / "mechanism" / f"context_channels-{design.name}.npz",
            contexts=np.asarray([cell.objective.context for cell in ordered], dtype=str),
            channels=channels,
            assignment=design.assignment,
            decoder=design.decoder,
        )
        payload["designs"][design.name] = {
            "L": len(design.block_weights),
            "assignment_hash": hash_array(design.assignment),
            "decoder_hash": hash_array(design.decoder),
            "table_entries": int(len(ordered) * len(design.block_weights) ** 2),
            "contexts": {
                cell.objective.context: hash_array(cell.capt_solution.channel) for cell in ordered
            },
        }
    manifest_path = path / "mechanism" / "context_channel_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _write_certificates(
    path: Path,
    config: dict[str, Any],
    frozen: dict[str, Any],
    designs: list[UtilityDesign],
    cells_by_design: dict[str, list[ContextCell]],
    progress: ProgressLogger,
) -> tuple[list[str], pd.DataFrame]:
    manifest_hash = sha256_file(path / "mechanism" / "context_channel_manifest.json")
    certificate_paths = []
    rows = []
    total_certificates = sum(len(cells_by_design[design.name]) for design in designs)
    completed = 0
    progress.emit(
        "certificate_batch_started",
        certificate_count=total_certificates,
        manifest_hash=manifest_hash,
        **process_memory_bytes(),
    )
    for design in designs:
        cells = sorted(cells_by_design[design.name], key=lambda cell: cell.objective.context)
        context_count = len(cells)
        for index, cell in enumerate(cells):
            certificate_started = time.perf_counter()
            progress.emit(
                "certificate_started",
                certificate_index=completed + 1,
                certificate_count=total_certificates,
                design=design.name,
                context=cell.objective.context,
                robust_constraint_count=len(cell.adjacency) * len(design.block_weights),
            )
            missing = sorted(cell.full_simplex_keys)
            cell_config = dict(config)
            cell_config.update(
                {
                    "alpha_cert": float(config.get("alpha_cert", 0.05)) / context_count,
                    "public_context_value": cell.objective.context,
                    "context_design": design.name,
                }
            )
            coverage = {
                "expected_group_count": len(cell.groups),
                "observed_group_count": len(cell.groups) - len(missing),
                "missing_group_count": len(missing),
                "missing_groups": missing,
                "rare_group_count": cell.rare_group_count,
                "hybrid_connectivity_gaps": 0,
                "requires_universal_cover": False,
                "policy": {"rare": "confidence_box", "missing": "full_simplex"},
                "profile": config["profiles"][0],
                "bundle_profiles": [config["profiles"][0]],
                "public_context_value": cell.objective.context,
                "public_context_count": context_count,
                "global_alpha_cert": float(config.get("alpha_cert", 0.05)),
                "context_alpha_cert": cell_config["alpha_cert"],
                "simultaneous_context_allocation": "bonferroni",
                "sensitive_secret_domain": "coarsened_A_with___UNKNOWN__",
                "full_simplex_group_count": len(missing),
                "full_simplex_ordered_adjacency_count": cell.full_full_edges,
                "design_mass": cell.objective.design_mass,
            }
            certificate = make_certificate(
                config=cell_config,
                channel=cell.capt_solution.channel,
                boxes=cell.boxes,
                adjacency=cell.adjacency,
                verification=cell.capt_verification,
                solver=cell.capt_solution.solver,
                component_hashes={
                    "encoder": frozen["encoder_hash"],
                    "model": frozen["model_hash"],
                    "mapper": frozen["mapper_hash"],
                    "context_channel_manifest": manifest_hash,
                },
                split_identifiers=config["splits"],
                dp_parameters={
                    "epsilon": config.get("dp_hist_epsilon"),
                    "delta": config.get("dp_hist_delta"),
                    "contribution_policy": config.get(
                        "contribution_policy", "one-display-per-uuid-day"
                    ),
                },
                histogram_counts=cell.counts,
                assignment=design.assignment,
                decoder=design.decoder,
                coverage=coverage,
                groups=cell.groups,
            )
            filename = f"certificate-{design.name}-context-{index:02d}.json"
            certificate_path = path / filename
            certificate.write(certificate_path)
            verification = verify_certificate(certificate_path)
            if not verification.valid:
                raise RuntimeError(f"independent certificate verification failed: {filename}")
            certificate_paths.append(filename)
            rows.append(
                {
                    "design": design.name,
                    "public_context": cell.objective.context,
                    "certificate_path": filename,
                    "certificate_valid": verification.valid,
                    "certificate_checked_constraints": verification.checked_constraints,
                    "certificate_max_violation": verification.max_violation,
                    "certificate_realized_epsilon": verification.realized_epsilon,
                }
            )
            completed += 1
            progress.emit(
                "certificate_finished",
                certificate_index=completed,
                certificate_count=total_certificates,
                design=design.name,
                context=cell.objective.context,
                path=str(certificate_path),
                certificate_seconds=time.perf_counter() - certificate_started,
                valid=verification.valid,
                checked_constraints=verification.checked_constraints,
                max_violation=verification.max_violation,
                realized_epsilon=verification.realized_epsilon,
                **process_memory_bytes(),
            )
    progress.emit(
        "certificate_batch_finished",
        certificate_count=completed,
        **process_memory_bytes(),
    )
    return certificate_paths, pd.DataFrame(rows)


def _evaluate_test(
    test_frame: pd.DataFrame,
    test_tokens: np.ndarray,
    labels: np.ndarray,
    designs: list[UtilityDesign],
    cells_by_design: dict[str, list[ContextCell]],
    shared_by_design: dict[str, tuple[ChannelSolution, ChannelSolution]],
    *,
    context_column: str,
) -> pd.DataFrame:
    context_values = test_frame[context_column].astype(str).to_numpy()
    rows = []
    for design in designs:
        cells = cells_by_design[design.name]
        cell_by_context = {cell.objective.context: cell for cell in cells}
        shared_ldp, shared_capt = shared_by_design[design.name]
        methods: dict[str, dict[str, np.ndarray]] = {
            "context_constant": {
                context: cell.constant_channel for context, cell in cell_by_context.items()
            },
            "context_ldp": {
                context: cell.ldp_solution.channel for context, cell in cell_by_context.items()
            },
            "context_capt": {
                context: cell.capt_solution.channel for context, cell in cell_by_context.items()
            },
            "shared_ldp": {context: shared_ldp.channel for context in cell_by_context},
            "shared_capt": {context: shared_capt.channel for context in cell_by_context},
        }
        for method, channels in methods.items():
            scores = np.empty(len(test_frame), dtype=float)
            expected_loss_sum = 0.0
            covered = 0
            for context, cell in cell_by_context.items():
                mask = context_values == context
                if not mask.any():
                    continue
                token_channel = lift_block_channel(
                    channels[context],
                    design.assignment,
                    design.decoder,
                )
                probabilities = cell.objective.token_probabilities
                token_scores = token_channel @ probabilities
                scores[mask] = token_scores[test_tokens[mask]]
                loss_one = token_channel @ -np.log(np.clip(probabilities, 1e-6, 1))
                loss_zero = token_channel @ -np.log(np.clip(1 - probabilities, 1e-6, 1))
                context_labels = labels[mask]
                context_tokens = test_tokens[mask]
                expected_loss_sum += float(
                    np.where(
                        context_labels == 1,
                        loss_one[context_tokens],
                        loss_zero[context_tokens],
                    ).sum()
                )
                covered += int(mask.sum())
            if covered != len(test_frame):
                raise RuntimeError(
                    f"test contexts are missing from channel table: {len(test_frame) - covered}"
                )
            metrics = prediction_metrics(labels, scores)
            rows.append(
                {
                    "design": design.name,
                    "label": design.label,
                    "L": len(design.block_weights),
                    "method": method,
                    "test_rows": len(test_frame),
                    "expected_randomized_log_loss": expected_loss_sum / len(test_frame),
                    "mixture_mean_log_loss": metrics["unweighted_log_loss"],
                    "ROC_AUC": metrics["ROC_AUC"],
                    "PR_AUC": metrics["PR_AUC"],
                    "ECE": metrics["ECE"],
                    "calibration_ratio": metrics["calibration_ratio"],
                }
            )
    return pd.DataFrame(rows)


def _plot_results(
    aggregate: pd.DataFrame,
    contexts: pd.DataFrame,
    path: Path,
) -> None:
    method_order = ["context_capt", "context_ldp", "shared_capt", "shared_ldp"]
    labels = {
        "context_capt": "Context CAPT",
        "context_ldp": "Context LDP",
        "shared_capt": "Shared CAPT",
        "shared_ldp": "Shared LDP",
    }
    colors = {
        "context_capt": "#2563A6",
        "context_ldp": "#D97706",
        "shared_capt": "#8A8A8A",
        "shared_ldp": "#C7C7C7",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.2))
    designs = aggregate["design"].drop_duplicates().tolist()
    x = np.arange(len(designs))
    width = 0.18
    for offset, method in enumerate(method_order):
        values = (
            aggregate.loc[aggregate["method"] == method]
            .set_index("design")
            .loc[designs, "gain_over_context_constant"]
        )
        axes[0].bar(
            x + (offset - 1.5) * width,
            values,
            width,
            label=labels[method],
            color=colors[method],
            edgecolor="#222222",
            hatch="//" if "capt" in method else None,
        )
    design_labels = aggregate.drop_duplicates("design").set_index("design").loc[designs, "label"]
    axes[0].set_xticks(x, design_labels)
    axes[0].set_ylabel("distortion reduction from context constant")
    axes[0].set_title("Mass-weighted design objective")
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].legend(fontsize=8)

    tolerance = 1e-12
    state_rows = []
    for design, subset in contexts.groupby("design", sort=False):
        states = np.select(
            [
                subset["ldp_degraded"],
                subset["capt_advantage_over_context_ldp"] > tolerance,
            ],
            ["LDP-degraded", "CAPT > context LDP"],
            default="No advantage, no full edge",
        )
        values = (
            pd.DataFrame({"state": states, "mass": subset["design_mass"].to_numpy()})
            .groupby("state")["mass"]
            .sum()
        )
        for state in ["CAPT > context LDP", "No advantage, no full edge", "LDP-degraded"]:
            state_rows.append(
                {"design": design, "state": state, "mass": float(values.get(state, 0.0))}
            )
    state_frame = pd.DataFrame(state_rows)
    state_colors = {
        "CAPT > context LDP": "#2563A6",
        "No advantage, no full edge": "#D97706",
        "LDP-degraded": "#8A8A8A",
    }
    bottom = np.zeros(len(designs))
    for state in state_colors:
        values = (
            state_frame.loc[state_frame["state"] == state]
            .set_index("design")
            .loc[designs, "mass"]
            .to_numpy()
        )
        axes[1].bar(
            x,
            values,
            bottom=bottom,
            label=state,
            color=state_colors[state],
            edgecolor="#222222",
        )
        bottom += values
    axes[1].set_xticks(x, design_labels)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("D_design context mass")
    axes[1].set_title("Where context CAPT improves or degrades")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2, color="#777777")
    fig.suptitle("Criteo constrained_2 public-context CAPT; unified sensitive unknown, epsilon=1")
    fig.text(
        0.5,
        0.01,
        "Channels use Z, profile, and public B only; the protected value is never an online selector.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fixed_time = datetime(2000, 1, 1, tzinfo=UTC)
    fig.savefig(
        path / "figures" / "context_stratified_diagnostic.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "CAPT-12",
            "Producer": "CAPT-12",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        path / "figures" / "context_stratified_diagnostic.png",
        bbox_inches="tight",
        dpi=220,
        metadata={"Software": "CAPT-12"},
    )
    plt.close(fig)


def _write_report(
    aggregate: pd.DataFrame,
    contexts: pd.DataFrame,
    test_metrics: pd.DataFrame,
    certificate_summary: pd.DataFrame,
    metadata: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Public-context stratified CAPT with unified sensitive fallback",
        "",
        "## Scope",
        "",
        f"- Profile: `features_kv_bits_constrained_2`; epsilon=1; {metadata['context_count']} frozen public-context values.",
        f"- D_cert: {metadata['cert_user_days']:,} one-display-per-user-day contributions; D_test: {metadata['test_rows']:,} displays.",
        "- Online selector: Z, profile, and public context B only. The protected value A is not an online input.",
        "- `__MISSING__` and unseen sensitive values are coarsened to one `__UNKNOWN__` secret.",
        "- Each design allocates alpha/B to its context certificates, giving a Bonferroni simultaneous level of at least 95% across contexts.",
        "",
        "## Results",
        "",
    ]
    for design in aggregate["design"].drop_duplicates():
        aggregate_design = aggregate.loc[aggregate["design"] == design].set_index("method")
        context_design = contexts.loc[contexts["design"] == design]
        advantage = context_design["capt_advantage_over_context_ldp"] > 1e-12
        degraded = context_design["ldp_degraded"]
        lines.extend(
            [
                f"### {aggregate_design.iloc[0]['label']}",
                "",
                f"- Context CAPT distortion: {aggregate_design.loc['context_capt', 'aggregate_distortion']:.9g}",
                f"- Context LDP distortion: {aggregate_design.loc['context_ldp', 'aggregate_distortion']:.9g}",
                f"- CAPT utility advantage over context LDP: {aggregate_design.loc['context_ldp', 'aggregate_distortion'] - aggregate_design.loc['context_capt', 'aggregate_distortion']:.9g}",
                f"- Contexts with strict CAPT advantage: {int(advantage.sum())}/{len(context_design)}, design mass {context_design.loc[advantage, 'design_mass'].sum():.9g}",
                f"- Contexts with a full-simplex edge and exact LDP degradation: {int(degraded.sum())}/{len(context_design)}, design mass {context_design.loc[degraded, 'design_mass'].sum():.9g}",
                f"- Channel table entries: {metadata['table_entries'][design]:,}",
                "",
            ]
        )
    lines.extend(
        [
            "## Verification and interpretation",
            "",
            f"All {len(certificate_summary)} context certificates are valid; they independently recheck {int(certificate_summary['certificate_checked_constraints'].sum()):,} robust constraints. The maximum violation is {certificate_summary['certificate_max_violation'].max():.3g}.",
            "",
            "The comparison that determines CAPT advantage is context CAPT versus context-specific optimal LDP under the same partition, decoder, context objective, and public-context table. Shared-channel methods are supplementary baselines.",
            "",
            "D_test AUC and calibration use the expected prediction under mechanism randomness. `expected_randomized_log_loss` instead averages the loss after drawing a sanitized output and is the primary randomized log-loss metric. These are frozen-reference diagnostics; no downstream bidder or CTR model is retrained.",
            "",
        ]
    )
    if not test_metrics.empty:
        best = test_metrics.sort_values("expected_randomized_log_loss").iloc[0]
        lines.append(
            f"The lowest expected randomized D_test log loss is {best['expected_randomized_log_loss']:.9g} for `{best['design']}/{best['method']}`."
        )
    (path / "context_stratified_report.md").write_text("\n".join(lines) + "\n")


def run_context_stratified_diagnostic(config: dict[str, Any]) -> Path:
    """Run one isolated public-context CAPT design."""
    config = validate_config(config)
    config = record_source_provenance(config)
    config = dict(config)
    required_splits = {"D_model", "D_design", "D_cert", "D_attack_train", "D_test"}
    if set(config.get("splits", {})) != required_splits:
        raise ValueError("context diagnostic requires the five fixed temporal splits")
    assert_disjoint_splits(config["splits"])
    if config.get("profiles") != ["features_kv_bits_constrained_2"]:
        raise ValueError("context diagnostic currently requires constrained_2 only")
    if config.get("public_context_policy") != "stratified":
        raise ValueError("context diagnostic requires public_context_policy: stratified")
    if config.get("sensitive_fallback_policy") != "unified_unknown":
        raise ValueError("context diagnostic requires unified sensitive fallback")
    if config.get("channel_selector_inputs") != [
        "Z",
        "profile",
        *config.get("context_cols", []),
    ]:
        raise ValueError("channel selector must use only Z, profile, and public context")
    if float(config.get("epsilon", -1)) != 1.0:
        raise ValueError("the first context diagnostic requires epsilon=1")
    if config.get("missing_group_policy") != "full_simplex":
        raise ValueError("context diagnostic requires full-simplex completion")

    path = prepare_run(config)
    progress = ProgressLogger(path, name=path.name)
    started = time.perf_counter()
    progress.emit(
        "run_started",
        output_path=str(path.resolve()),
        source_git_sha=config.get("source_git_sha"),
        profile=config["profiles"][0],
        public_context_columns=config.get("context_cols"),
        epsilon=config.get("epsilon"),
        confidence=config.get("confidence"),
        alpha_cert=config.get("alpha_cert"),
        missing_group_policy=config.get("missing_group_policy"),
        sensitive_fallback_policy=config.get("sensitive_fallback_policy"),
        solver=config.get("solver", "scipy-highs"),
        solver_verbose=config.get("solver_verbose", False),
        solver_heartbeat_seconds=config.get("solver_heartbeat_seconds", 60.0),
        solver_tolerance=config.get("solver_tolerance"),
        solver_time_limit_seconds=config.get("time_limit"),
        max_cutting_plane_iterations=config.get("max_cutting_plane_iterations"),
        robust_cut_formulation=config.get("robust_cut_formulation", "paired_witness"),
        resume_cutting_plane=config.get("resume_cutting_plane", False),
        context_designs=config.get("context_designs"),
    )
    progress.emit_environment()
    progress.emit("fixed_design_started", **process_memory_bytes())
    fixed_design_started = time.perf_counter()
    frozen, support_table = _fixed_design(config, path)
    progress.emit(
        "fixed_design_finished",
        stage_seconds=time.perf_counter() - fixed_design_started,
        cartesian_group_count=len(frozen["cartesian_support"]),
        design_group_count=len(frozen["design_only_probability"]),
        mapper_hash=frozen["mapper_hash"],
        encoder_hash=frozen["encoder_hash"],
        model_hash=frozen["model_hash"],
        **process_memory_bytes(),
    )
    support_table.to_csv(path / "tables" / "frozen_support.csv", index=False)
    progress.emit(
        "frozen_support_written",
        path=str(path / "tables" / "frozen_support.csv"),
        row_count=len(support_table),
    )
    progress.emit("utility_design_build_started", **process_memory_bytes())
    design_build_started = time.perf_counter()
    frozen_arrays = np.load(path / "mechanism" / "frozen_design.npz")
    all_designs = _build_designs(frozen_arrays, config)
    wanted = set(
        config.get(
            "context_designs",
            ["joint_kmedoids_cost_medoid_L16", "singleton_identity_L64"],
        )
    )
    if len(wanted) != 1:
        raise ValueError("context diagnostic must isolate exactly one utility design")
    designs = [design for design in all_designs if design.name in wanted]
    if {design.name for design in designs} != wanted:
        raise RuntimeError("required utility-aware designs were not built")
    progress.emit(
        "utility_design_build_finished",
        stage_seconds=time.perf_counter() - design_build_started,
        selected_designs=[
            {
                "name": design.name,
                "L": len(design.block_weights),
                "variable_count": len(design.block_weights) ** 2,
                "partition": design.partition_method,
                "decoder": design.decoder_method,
                "information_gap": design.diagnostic.information_gap,
            }
            for design in designs
        ],
        **process_memory_bytes(),
    )

    profile = config["profiles"][0]
    context_column = config["context_cols"][0]
    source = list(config.get("phi_source_cols", []))
    id_col = config.get("id_col", "id")
    user_col = config.get("user_col", "user_id")
    label_col = config.get("label_col", "is_clicked")
    columns = list(
        dict.fromkeys([id_col, user_col, label_col, *profile.split("+"), context_column, *source])
    )
    progress.emit(
        "certificate_data_load_started",
        data_root=config["data_root"],
        days=config["splits"]["D_cert"],
        column_count=len(columns),
        columns=columns,
        **process_memory_bytes(),
    )
    cert_load_started = time.perf_counter()
    cert_source = load_parquet_sample(
        data_root=config["data_root"],
        columns=columns,
        days=config["splits"]["D_cert"],
    )
    progress.emit(
        "certificate_data_load_finished",
        stage_seconds=time.perf_counter() - cert_load_started,
        source_row_count=len(cert_source),
        **process_memory_bytes(),
    )
    assert_no_row_overlap({"D_cert": cert_source}, id_col)
    progress.emit("certificate_mapper_transform_started", row_count=len(cert_source))
    cert_source = frozen["mapper"].transform(cert_source)
    progress.emit(
        "certificate_mapper_transform_finished",
        row_count=len(cert_source),
        **process_memory_bytes(),
    )
    progress.emit(
        "certificate_contribution_filter_started",
        source_row_count=len(cert_source),
        policy="one-display-per-user-day",
    )
    cert_frame = select_one_display_per_user_day(
        cert_source,
        user_col=user_col,
        day_col="day_int",
        id_col=id_col,
    )
    progress.emit(
        "certificate_contribution_filter_finished",
        source_row_count=len(cert_source),
        user_day_count=len(cert_frame),
        retained_fraction=len(cert_frame) / len(cert_source),
        **process_memory_bytes(),
    )
    progress.emit("certificate_encoding_started", row_count=len(cert_frame))
    cert_tokens = frozen["encoder"].transform(cert_frame)
    token_counts = np.bincount(cert_tokens, minlength=int(config["K"]))
    progress.emit(
        "certificate_encoding_finished",
        row_count=len(cert_tokens),
        token_alphabet_size=len(token_counts),
        observed_token_count=int(np.sum(token_counts > 0)),
        min_positive_token_count=int(np.min(token_counts[token_counts > 0])),
        max_token_count=int(np.max(token_counts)),
        **process_memory_bytes(),
    )
    expected_frame = _expected_frame_from_cartesian(
        frozen["cartesian_support"],
        profile=profile,
        context=context_column,
    )
    progress.emit(
        "expected_support_frame_ready",
        row_count=len(expected_frame),
        context_count=int(expected_frame[context_column].nunique()),
        **process_memory_bytes(),
    )

    cells_by_design: dict[str, list[ContextCell]] = {}
    shared_by_design: dict[str, tuple[ChannelSolution, ChannelSolution]] = {}
    context_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for design_index, design in enumerate(designs, start=1):
        design_started = time.perf_counter()
        progress.emit(
            "design_started",
            design=design.name,
            design_index=design_index,
            design_count=len(designs),
            label=design.label,
            dimension=len(design.block_weights),
            variable_count=len(design.block_weights) ** 2,
            ldp_constraint_count=(
                len(design.block_weights)
                * (len(design.block_weights) - 1)
                * len(design.block_weights)
            ),
            partition_method=design.partition_method,
            decoder_method=design.decoder_method,
            **process_memory_bytes(),
        )
        progress.emit("histogram_build_started", design=design.name)
        histogram_started = time.perf_counter()
        hist = build_group_histograms(
            cert_frame,
            cert_tokens,
            profile=profile,
            context_columns=[context_column],
            alphabet_size=len(design.block_weights),
            token_to_block=design.assignment,
            min_group_count=int(config.get("min_group_count", 20)),
            rare_group_policy="confidence_box",
            missing_group_policy="full_simplex",
            expected_frame=expected_frame,
            include_fallback_levels=False,
        )
        histogram_totals = np.asarray(
            [int(value.sum()) for value in hist.counts.values()], dtype=int
        )
        progress.emit(
            "histogram_build_finished",
            design=design.name,
            stage_seconds=time.perf_counter() - histogram_started,
            group_count=len(hist.groups),
            observed_group_count=int(np.sum(histogram_totals > 0)),
            missing_group_count=int(np.sum(histogram_totals == 0)),
            rare_group_count=int(
                np.sum(
                    (histogram_totals > 0)
                    & (histogram_totals < int(config.get("min_group_count", 20)))
                )
            ),
            force_cover=hist.force_cover,
            **process_memory_bytes(),
        )
        if hist.force_cover:
            raise RuntimeError("context diagnostic unexpectedly requested universal cover")
        progress.emit("context_objective_build_started", design=design.name)
        objective_started = time.perf_counter()
        objectives = _context_objectives(frozen, design, config)
        progress.emit(
            "context_objective_build_finished",
            design=design.name,
            stage_seconds=time.perf_counter() - objective_started,
            context_count=len(objectives),
            mass_sum=sum(value.design_mass for value in objectives),
            min_context_mass=min(value.design_mass for value in objectives),
            max_context_mass=max(value.design_mass for value in objectives),
            **process_memory_bytes(),
        )
        cells, shared_ldp, shared_capt, _ = _solve_design(
            design,
            objectives,
            hist.groups,
            hist.counts,
            config,
            progress,
            path / "checkpoints",
        )
        cells_by_design[design.name] = cells
        shared_by_design[design.name] = (shared_ldp, shared_capt)
        context_rows.extend(_context_rows(cells))
        aggregate_rows.extend(_aggregate_rows(design, cells, shared_ldp, shared_capt))
        support_rows.extend(_support_origin_rows(cells, frozen["design_only_probability"]))
        np.savez_compressed(
            path / "mechanism" / f"shared_channels-{design.name}.npz",
            shared_ldp=shared_ldp.channel,
            shared_capt=shared_capt.channel,
            assignment=design.assignment,
            decoder=design.decoder,
        )
        progress.emit(
            "design_checkpoint_written",
            design=design.name,
            path=str(path / "mechanism" / f"shared_channels-{design.name}.npz"),
            completed_context_count=len(cells),
            design_seconds=time.perf_counter() - design_started,
            **process_memory_bytes(),
        )

    progress.emit("context_channel_manifest_started", **process_memory_bytes())
    _channel_manifest(
        path,
        profile,
        context_column,
        frozen["mapper_hash"],
        designs,
        cells_by_design,
    )
    progress.emit(
        "context_channel_manifest_finished",
        path=str(path / "mechanism" / "context_channel_manifest.json"),
        **process_memory_bytes(),
    )
    certificate_paths, certificate_summary = _write_certificates(
        path,
        config,
        frozen,
        designs,
        cells_by_design,
        progress,
    )

    progress.emit(
        "test_data_load_started",
        data_root=config["data_root"],
        days=config["splits"]["D_test"],
        **process_memory_bytes(),
    )
    test_load_started = time.perf_counter()
    test_frame = load_parquet_sample(
        data_root=config["data_root"],
        columns=columns,
        days=config["splits"]["D_test"],
    )
    progress.emit(
        "test_data_load_finished",
        stage_seconds=time.perf_counter() - test_load_started,
        test_row_count=len(test_frame),
        **process_memory_bytes(),
    )
    test_frame = frozen["mapper"].transform(test_frame)
    test_tokens = frozen["encoder"].transform(test_frame)
    labels = test_frame[label_col].to_numpy(dtype=int)
    progress.emit(
        "test_evaluation_started",
        test_row_count=len(test_frame),
        design_count=len(designs),
        method_count=5,
        **process_memory_bytes(),
    )
    evaluation_started = time.perf_counter()
    test_metrics = _evaluate_test(
        test_frame,
        test_tokens,
        labels,
        designs,
        cells_by_design,
        shared_by_design,
        context_column=context_column,
    )
    progress.emit(
        "test_evaluation_finished",
        stage_seconds=time.perf_counter() - evaluation_started,
        metric_row_count=len(test_metrics),
        **process_memory_bytes(),
    )

    progress.emit("result_table_write_started", **process_memory_bytes())
    context_frame = pd.DataFrame(context_rows).merge(
        certificate_summary,
        on=["design", "public_context"],
        how="left",
        validate="one_to_one",
    )
    aggregate_frame = pd.DataFrame(aggregate_rows)
    support_frame = pd.DataFrame(support_rows)
    context_frame.to_csv(path / "tables" / "context_results.csv", index=False)
    aggregate_frame.to_csv(path / "tables" / "aggregate_results.csv", index=False)
    support_frame.to_csv(path / "tables" / "support_origin.csv", index=False)
    certificate_summary.to_csv(path / "tables" / "certificate_summary.csv", index=False)
    test_metrics.to_csv(path / "tables" / "test_metrics.csv", index=False)
    context_frame.to_parquet(path / "context_results.parquet", index=False)
    aggregate_frame.to_parquet(path / "aggregate_results.parquet", index=False)
    test_metrics.to_parquet(path / "test_metrics.parquet", index=False)
    progress.emit(
        "result_table_write_finished",
        context_result_rows=len(context_frame),
        aggregate_result_rows=len(aggregate_frame),
        support_result_rows=len(support_frame),
        certificate_result_rows=len(certificate_summary),
        test_metric_rows=len(test_metrics),
        **process_memory_bytes(),
    )
    progress.emit("figure_render_started", **process_memory_bytes())
    figure_started = time.perf_counter()
    _plot_results(aggregate_frame, context_frame, path)
    progress.emit(
        "figure_render_finished",
        stage_seconds=time.perf_counter() - figure_started,
        pdf_path=str(path / "figures" / "context_stratified_diagnostic.pdf"),
        png_path=str(path / "figures" / "context_stratified_diagnostic.png"),
        **process_memory_bytes(),
    )

    metadata = {
        "experiment": "criteo_public_context_stratified_capt",
        "source_git_sha": config["source_git_sha"],
        "profile": profile,
        "public_context_column": context_column,
        "context_count": int(context_frame["public_context"].nunique()),
        "cert_source_rows": len(cert_source),
        "cert_user_days": len(cert_frame),
        "test_rows": len(test_frame),
        "designs": [design.name for design in designs],
        "table_entries": {
            design.name: int(
                context_frame["public_context"].nunique() * len(design.block_weights) ** 2
            )
            for design in designs
        },
        "certificate_count": len(certificate_summary),
        "certificate_checked_constraints": int(
            certificate_summary["certificate_checked_constraints"].sum()
        ),
        "certificate_max_violation": float(certificate_summary["certificate_max_violation"].max()),
        "wall_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "environment": environment(),
    }
    (path / "context_stratified_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    _write_report(
        aggregate_frame,
        context_frame,
        test_metrics,
        certificate_summary,
        metadata,
        path,
    )
    progress.emit(
        "report_written",
        path=str(path / "context_stratified_report.md"),
        wall_seconds=metadata["wall_seconds"],
        **process_memory_bytes(),
    )
    del cert_source, cert_frame, cert_tokens, test_frame, test_tokens, labels
    gc.collect()
    finish_run(
        path,
        {
            "dataset": "criteo",
            "experiment": metadata["experiment"],
            "profile": profile,
            "certificates": certificate_paths,
            "source_git_sha": config["source_git_sha"],
            "context_count": metadata["context_count"],
            "certificate_count": metadata["certificate_count"],
            "sol_review_bundle": "sol_review_bundle.zip",
        },
    )
    progress.emit("sol_review_bundle_started", **process_memory_bytes())
    bundle_path = write_sol_review_bundle(path)
    progress.emit(
        "sol_review_bundle_finished",
        path=str(bundle_path),
        size_bytes=bundle_path.stat().st_size,
        sha256=sha256_file(bundle_path),
        **process_memory_bytes(),
    )
    progress.emit(
        "run_finished",
        status="complete",
        wall_seconds=time.perf_counter() - started,
        certificate_count=len(certificate_paths),
        output_path=str(path.resolve()),
        **process_memory_bytes(),
    )
    return path
