from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from capt12.confidence.boxes import ConfidenceBox
from capt12.confidence.support import support
from capt12.mechanisms.lp import (
    ChannelSolution,
    ProgressCallback,
    _emit_progress,
    _solve_channel_lp,
    ldp_constraint_matrix,
    solve_ldp_block_lp,
    validate_channel,
)
from capt12.privacy.adjacency import AdjacentPair
from capt12.utils.progress import process_memory_bytes


@dataclass
class VerificationResult:
    valid: bool
    max_violation: float
    realized_epsilon: float
    worst_case: dict | None
    checked_constraints: int


def _full_simplex_ldp_epsilon(
    boxes: Mapping[str, ConfidenceBox],
    adjacency: Sequence[AdjacentPair],
) -> float | None:
    """Return the exact global LDP budget implied by full-simplex pairs."""
    epsilons = [
        float(pair.epsilon)
        for pair in adjacency
        if boxes[pair.left].method == "full_simplex" and boxes[pair.right].method == "full_simplex"
    ]
    return min(epsilons) if epsilons else None


def _cut_key(
    left: np.ndarray, right: np.ndarray, epsilon: float, output: int
) -> tuple[bytes, bytes, float, int]:
    return (
        np.round(np.asarray(left, dtype=float), 14).tobytes(),
        np.round(np.asarray(right, dtype=float), 14).tobytes(),
        round(float(epsilon), 14),
        int(output),
    )


def verify_robust_channel(
    channel: np.ndarray,
    boxes: Mapping[str, ConfidenceBox],
    adjacency: Sequence[AdjacentPair],
    *,
    tolerance: float = 1e-8,
) -> VerificationResult:
    try:
        validate_channel(channel, tolerance)
    except ValueError as error:
        return VerificationResult(False, math.inf, math.inf, {"error": str(error)}, 0)
    max_violation = -math.inf
    realized = -math.inf
    worst = None
    checked = 0
    for pair in adjacency:
        for output in range(channel.shape[1]):
            maximum, p_max = support(channel[:, output], boxes[pair.left], maximize=True)
            minimum, p_min = support(channel[:, output], boxes[pair.right], maximize=False)
            violation = maximum - math.exp(pair.epsilon) * minimum
            epsilon = (
                math.inf
                if minimum <= 0 < maximum
                else (math.log(maximum / minimum) if maximum > 0 and minimum > 0 else -math.inf)
            )
            checked += 1
            if violation > max_violation:
                max_violation = float(violation)
                worst = {
                    "left": pair.left,
                    "right": pair.right,
                    "output_block": output,
                    "target_epsilon": pair.epsilon,
                    "maximum": maximum,
                    "minimum": minimum,
                    "left_witness": p_max.tolist(),
                    "right_witness": p_min.tolist(),
                }
            realized = max(realized, epsilon)
    if checked == 0:
        max_violation = 0.0
        realized = 0.0
    return VerificationResult(max_violation <= tolerance, max_violation, realized, worst, checked)


def solve_robust_block_lp(
    cost: np.ndarray,
    block_weights: np.ndarray,
    boxes: Mapping[str, ConfidenceBox],
    adjacency: Sequence[AdjacentPair],
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    time_limit: float | None = None,
    progress: ProgressCallback | None = None,
    progress_label: str = "robust_block_lp",
    solver_verbose: bool = False,
    heartbeat_seconds: float = 60.0,
) -> tuple[ChannelSolution, VerificationResult]:
    if any(box.experimental for box in boxes.values()):
        raise ValueError("experimental DP-aware boxes cannot produce a certified result")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    nominal = {key: box.nominal for key, box in boxes.items()}
    ldp_epsilon = _full_simplex_ldp_epsilon(boxes, adjacency)
    pure_ldp = ldp_epsilon is not None and ldp_epsilon <= min(
        (float(pair.epsilon) for pair in adjacency),
        default=ldp_epsilon,
    )
    dimension = int(np.asarray(cost).shape[0])
    full_simplex_count = sum(box.method == "full_simplex" for box in boxes.values())
    full_simplex_edge_count = sum(
        boxes[pair.left].method == "full_simplex" and boxes[pair.right].method == "full_simplex"
        for pair in adjacency
    )
    _emit_progress(
        progress,
        "robust_solve_started",
        label=progress_label,
        dimension=dimension,
        variable_count=dimension * dimension,
        box_count=len(boxes),
        adjacency_count=len(adjacency),
        nominal_constraint_count=len(adjacency) * dimension,
        full_simplex_box_count=full_simplex_count,
        full_simplex_edge_count=full_simplex_edge_count,
        implied_ldp_epsilon=ldp_epsilon,
        pure_ldp_replacement=pure_ldp,
        max_cutting_plane_iterations=max_iterations,
        tolerance=tolerance,
        time_limit_seconds=time_limit,
        **process_memory_bytes(),
    )
    if pure_ldp:
        # A single full-simplex/full-simplex edge is exactly global row-wise
        # LDP.  The minimum such epsilon implies every other edge here, so solve
        # that equivalent LP directly and still verify every original robust
        # constraint below.
        solution = solve_ldp_block_lp(
            cost,
            block_weights,
            ldp_epsilon,
            tolerance=tolerance,
            time_limit=time_limit,
            progress=progress,
            progress_label=f"{progress_label}/pure_ldp",
            solver_verbose=solver_verbose,
            heartbeat_seconds=heartbeat_seconds,
        )
        if solution.channel is None:
            return solution, VerificationResult(
                False,
                math.inf,
                math.inf,
                {"error": solution.solver.message},
                0,
            )
        verification_started = time.perf_counter()
        _emit_progress(
            progress,
            "robust_verification_started",
            label=progress_label,
            checked_constraint_target=len(adjacency) * dimension,
        )
        verification = verify_robust_channel(
            solution.channel,
            boxes,
            adjacency,
            tolerance=tolerance,
        )
        _emit_progress(
            progress,
            "robust_verification_finished",
            label=progress_label,
            verification_seconds=time.perf_counter() - verification_started,
            valid=verification.valid,
            checked_constraints=verification.checked_constraints,
            max_violation=verification.max_violation,
            realized_epsilon=verification.realized_epsilon,
            **process_memory_bytes(),
        )
        if not verification.valid:
            solution.solver.status = "verification_failed"
        solution.cuts = [
            {
                "source": "full_simplex_ldp_seed",
                "epsilon": ldp_epsilon,
                "constraint_count": dimension * (dimension - 1) * dimension,
            }
        ]
        return solution, verification

    ldp_seed = None
    if ldp_epsilon is not None:
        _emit_progress(
            progress,
            "robust_ldp_seed_build_started",
            label=progress_label,
            dimension=dimension,
            epsilon=ldp_epsilon,
            expected_constraint_count=dimension * (dimension - 1) * dimension,
        )
        seed_started = time.perf_counter()
        ldp_seed = ldp_constraint_matrix(dimension, ldp_epsilon)
        _emit_progress(
            progress,
            "robust_ldp_seed_build_finished",
            label=progress_label,
            build_seconds=time.perf_counter() - seed_started,
            constraint_count=int(ldp_seed.shape[0]),
            nonzero_count=int(ldp_seed.nnz),
            **process_memory_bytes(),
        )
    cuts: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    cut_keys: set[tuple[bytes, bytes, float, int]] = set()
    solution: ChannelSolution | None = None
    for iteration in range(max_iterations):
        _emit_progress(
            progress,
            "cutting_plane_iteration_started",
            label=progress_label,
            iteration=iteration + 1,
            max_iterations=max_iterations,
            accumulated_support_cut_count=len(cuts),
            nominal_constraint_count=len(adjacency) * dimension,
            ldp_seed_constraint_count=(int(ldp_seed.shape[0]) if ldp_seed is not None else 0),
            **process_memory_bytes(),
        )
        solution = _solve_channel_lp(
            cost,
            block_weights,
            nominal,
            adjacency,
            tolerance=tolerance,
            time_limit=time_limit,
            extra_cuts=cuts,
            precompiled_ub=ldp_seed,
            progress=progress,
            progress_label=f"{progress_label}/iteration_{iteration + 1}",
            solver_verbose=solver_verbose,
            heartbeat_seconds=heartbeat_seconds,
        )
        if solution.channel is None:
            return solution, VerificationResult(
                False, math.inf, math.inf, {"error": solution.solver.message}, 0
            )
        scan_started = time.perf_counter()
        _emit_progress(
            progress,
            "support_oracle_scan_started",
            label=progress_label,
            iteration=iteration + 1,
            adjacency_count=len(adjacency),
            output_count=solution.channel.shape[1],
            support_queries=2 * len(adjacency) * solution.channel.shape[1],
        )
        added = 0
        max_iteration_violation = -math.inf
        for pair in adjacency:
            for output in range(solution.channel.shape[1]):
                maximum, p_max = support(
                    solution.channel[:, output], boxes[pair.left], maximize=True
                )
                minimum, p_min = support(
                    solution.channel[:, output], boxes[pair.right], maximize=False
                )
                violation = maximum - math.exp(pair.epsilon) * minimum
                max_iteration_violation = max(max_iteration_violation, float(violation))
                if violation > tolerance:
                    key = _cut_key(p_max, p_min, pair.epsilon, output)
                    if key in cut_keys:
                        continue
                    cuts.append((p_max, p_min, pair.epsilon, output))
                    cut_keys.add(key)
                    solution.cuts.append(
                        {
                            "iteration": iteration,
                            "left": pair.left,
                            "right": pair.right,
                            "output": output,
                            "violation": float(violation),
                        }
                    )
                    added += 1
        _emit_progress(
            progress,
            "support_oracle_scan_finished",
            label=progress_label,
            iteration=iteration + 1,
            scan_seconds=time.perf_counter() - scan_started,
            added_support_cut_count=added,
            accumulated_support_cut_count=len(cuts),
            max_iteration_violation=max_iteration_violation,
            converged=added == 0,
            **process_memory_bytes(),
        )
        if added == 0:
            break
    assert solution is not None
    verification_started = time.perf_counter()
    _emit_progress(
        progress,
        "robust_verification_started",
        label=progress_label,
        checked_constraint_target=len(adjacency) * dimension,
    )
    verification = verify_robust_channel(solution.channel, boxes, adjacency, tolerance=tolerance)
    _emit_progress(
        progress,
        "robust_verification_finished",
        label=progress_label,
        verification_seconds=time.perf_counter() - verification_started,
        valid=verification.valid,
        checked_constraints=verification.checked_constraints,
        max_violation=verification.max_violation,
        realized_epsilon=verification.realized_epsilon,
        support_cut_count=len(cuts),
        **process_memory_bytes(),
    )
    if not verification.valid:
        solution.solver.status = "verification_failed"
    solution.cuts = (
        [
            {
                "source": "full_simplex_ldp_seed",
                "epsilon": ldp_epsilon,
                "constraint_count": dimension * (dimension - 1) * dimension,
                "pure_ldp_replacement": False,
            }
        ]
        if ldp_seed is not None
        else []
    ) + [
        {
            "source": "support_oracle",
            "left_witness": left.tolist(),
            "right_witness": right.tolist(),
            "epsilon": eps,
            "output": output,
        }
        for left, right, eps, output in cuts
    ]
    return solution, verification
