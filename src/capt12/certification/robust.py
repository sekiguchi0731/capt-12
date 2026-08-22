from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

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


def _support_cut_key(
    group_index: int,
    output: int,
    maximize: bool,
    witness: np.ndarray,
) -> tuple[int, int, bool, bytes]:
    return (
        int(group_index),
        int(output),
        bool(maximize),
        np.round(np.asarray(witness, dtype=float), 14).tobytes(),
    )


def _support_problem_fingerprint(
    cost: np.ndarray,
    block_weights: np.ndarray,
    group_keys: Sequence[str],
    boxes: Mapping[str, ConfidenceBox],
    adjacency: Sequence[AdjacentPair],
    tolerance: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"capt12-shared-support-bounds-v1\0")
    for value in (cost, block_weights):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(list(group_keys), separators=(",", ":")).encode())
    for key in group_keys:
        box = boxes[key]
        digest.update(key.encode())
        digest.update(box.method.encode())
        digest.update(np.float64(box.alpha_familywise).tobytes())
        digest.update(np.float64(box.tv_radius).tobytes())
        digest.update(bytes([int(box.experimental)]))
        for value in (box.lower, box.upper, box.nominal):
            digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    digest.update(
        json.dumps(
            [
                [pair.left, pair.right, float(pair.epsilon), list(pair.changed_attributes)]
                for pair in adjacency
            ],
            separators=(",", ":"),
        ).encode()
    )
    digest.update(np.float64(tolerance).tobytes())
    return digest.hexdigest()


def _write_support_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    completed_iteration: int,
    cuts: Sequence[tuple[int, int, bool, np.ndarray]],
    dimension: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    if cuts:
        group_indices = np.asarray([cut[0] for cut in cuts], dtype=np.int32)
        outputs = np.asarray([cut[1] for cut in cuts], dtype=np.int32)
        maximize = np.asarray([cut[2] for cut in cuts], dtype=np.bool_)
        witnesses = np.vstack([cut[3] for cut in cuts]).astype(np.float64, copy=False)
    else:
        group_indices = np.empty(0, dtype=np.int32)
        outputs = np.empty(0, dtype=np.int32)
        maximize = np.empty(0, dtype=np.bool_)
        witnesses = np.empty((0, dimension), dtype=np.float64)
    np.savez_compressed(
        temporary,
        fingerprint=np.asarray(fingerprint),
        completed_iteration=np.asarray(completed_iteration, dtype=np.int64),
        group_indices=group_indices,
        outputs=outputs,
        maximize=maximize,
        witnesses=witnesses,
    )
    temporary.replace(path)


def _load_support_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    dimension: int,
) -> tuple[int, list[tuple[int, int, bool, np.ndarray]]]:
    with np.load(path, allow_pickle=False) as payload:
        stored_fingerprint = str(payload["fingerprint"].item())
        if stored_fingerprint != fingerprint:
            raise ValueError(f"checkpoint does not match this robust LP: {path}")
        witnesses = np.asarray(payload["witnesses"], dtype=float)
        if witnesses.ndim != 2 or witnesses.shape[1] != dimension:
            raise ValueError(f"checkpoint witness dimension is invalid: {path}")
        group_indices = np.asarray(payload["group_indices"], dtype=int)
        outputs = np.asarray(payload["outputs"], dtype=int)
        maximize = np.asarray(payload["maximize"], dtype=bool)
        if not (len(group_indices) == len(outputs) == len(maximize) == len(witnesses)):
            raise ValueError(f"checkpoint arrays have inconsistent lengths: {path}")
        completed_iteration = int(payload["completed_iteration"].item())
    cuts = [
        (int(group_index), int(output), bool(direction), witness.copy())
        for group_index, output, direction, witness in zip(
            group_indices,
            outputs,
            maximize,
            witnesses,
            strict=True,
        )
    ]
    return completed_iteration, cuts


def _support_bound_matrix(
    dimension: int,
    group_keys: Sequence[str],
    adjacency: Sequence[AdjacentPair],
    cuts: Sequence[tuple[int, int, bool, np.ndarray]],
) -> sparse.csr_matrix:
    """Compile shared upper/lower support bounds and their witness cuts."""
    group_index = {key: index for index, key in enumerate(group_keys)}
    channel_variables = dimension * dimension
    support_variables = len(group_keys) * dimension
    upper_offset = channel_variables
    lower_offset = upper_offset + support_variables
    variable_count = channel_variables + 2 * support_variables
    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    row = 0
    for pair in adjacency:
        left = group_index[pair.left]
        right = group_index[pair.right]
        factor = math.exp(pair.epsilon)
        for output in range(dimension):
            row_indices.extend((row, row))
            column_indices.extend(
                (upper_offset + left * dimension + output, lower_offset + right * dimension + output)
            )
            data.extend((1.0, -factor))
            row += 1
    channel_rows = np.arange(dimension, dtype=int) * dimension
    for group, output, maximize, witness in cuts:
        coefficients = np.asarray(witness, dtype=float) if maximize else -np.asarray(witness)
        row_indices.extend([row] * dimension)
        column_indices.extend((channel_rows + output).tolist())
        data.extend(coefficients.tolist())
        row_indices.append(row)
        column_indices.append(
            (upper_offset if maximize else lower_offset) + group * dimension + output
        )
        data.append(-1.0 if maximize else 1.0)
        row += 1
    return sparse.coo_matrix(
        (data, (row_indices, column_indices)),
        shape=(row, variable_count),
    ).tocsr()


def _solve_shared_support_bounds(
    cost: np.ndarray,
    block_weights: np.ndarray,
    boxes: Mapping[str, ConfidenceBox],
    adjacency: Sequence[AdjacentPair],
    *,
    tolerance: float,
    max_iterations: int,
    time_limit: float | None,
    ldp_seed: sparse.spmatrix | None,
    progress: ProgressCallback | None,
    progress_label: str,
    solver_verbose: bool,
    heartbeat_seconds: float,
    checkpoint_path: Path | None,
    resume_checkpoint: bool,
    checkpoint_every: int,
) -> tuple[ChannelSolution, VerificationResult]:
    dimension = int(np.asarray(cost).shape[0])
    group_keys = sorted(boxes)
    group_count = len(group_keys)
    fingerprint = _support_problem_fingerprint(
        cost,
        block_weights,
        group_keys,
        boxes,
        adjacency,
        tolerance,
    )
    cuts: list[tuple[int, int, bool, np.ndarray]] = []
    # Nominal witnesses make every support variable meaningful in the first
    # master problem. The oracle replaces them with exact extremal witnesses.
    for group_index, key in enumerate(group_keys):
        for output in range(dimension):
            cuts.append((group_index, output, True, boxes[key].nominal.copy()))
            cuts.append((group_index, output, False, boxes[key].nominal.copy()))
    completed_iteration = 0
    if checkpoint_path is not None and resume_checkpoint and checkpoint_path.exists():
        completed_iteration, cuts = _load_support_checkpoint(
            checkpoint_path,
            fingerprint=fingerprint,
            dimension=dimension,
        )
        _emit_progress(
            progress,
            "support_checkpoint_loaded",
            label=progress_label,
            path=str(checkpoint_path),
            completed_iteration=completed_iteration,
            support_cut_count=len(cuts),
        )
    cut_keys = {_support_cut_key(*cut) for cut in cuts}
    solution: ChannelSolution | None = None
    converged = False
    for local_iteration in range(max_iterations):
        iteration = completed_iteration + local_iteration + 1
        _emit_progress(
            progress,
            "cutting_plane_iteration_started",
            label=progress_label,
            formulation="shared_support_bounds",
            iteration=iteration,
            local_iteration=local_iteration + 1,
            max_local_iterations=max_iterations,
            accumulated_support_cut_count=len(cuts),
            group_count=group_count,
            adjacency_bound_constraint_count=len(adjacency) * dimension,
            **process_memory_bytes(),
        )
        matrix = _support_bound_matrix(dimension, group_keys, adjacency, cuts)
        if ldp_seed is not None:
            padded_seed = sparse.hstack(
                [
                    ldp_seed,
                    sparse.csr_matrix((ldp_seed.shape[0], 2 * group_count * dimension)),
                ],
                format="csr",
            )
            matrix = sparse.vstack([padded_seed, matrix], format="csr")
        solution = _solve_channel_lp(
            cost,
            block_weights,
            {},
            [],
            tolerance=tolerance,
            time_limit=time_limit,
            precompiled_ub=matrix,
            auxiliary_variable_count=2 * group_count * dimension,
            progress=progress,
            progress_label=f"{progress_label}/iteration_{iteration}",
            solver_verbose=solver_verbose,
            heartbeat_seconds=heartbeat_seconds,
        )
        if solution.channel is None or solution.auxiliary is None:
            return solution, VerificationResult(
                False, math.inf, math.inf, {"error": solution.solver.message}, 0
            )
        upper = solution.auxiliary[: group_count * dimension].reshape(group_count, dimension)
        lower = solution.auxiliary[group_count * dimension :].reshape(group_count, dimension)
        scan_started = time.perf_counter()
        _emit_progress(
            progress,
            "support_oracle_scan_started",
            label=progress_label,
            formulation="shared_support_bounds",
            iteration=iteration,
            group_count=group_count,
            output_count=dimension,
            support_queries=2 * group_count * dimension,
        )
        added = 0
        max_iteration_violation = -math.inf
        for group_index, key in enumerate(group_keys):
            box = boxes[key]
            for output in range(dimension):
                coefficients = solution.channel[:, output]
                maximum, p_max = support(coefficients, box, maximize=True)
                minimum, p_min = support(coefficients, box, maximize=False)
                max_violation = maximum - upper[group_index, output]
                min_violation = lower[group_index, output] - minimum
                max_iteration_violation = max(
                    max_iteration_violation,
                    float(max_violation),
                    float(min_violation),
                )
                for maximize, witness, violation in (
                    (True, p_max, max_violation),
                    (False, p_min, min_violation),
                ):
                    if violation <= tolerance:
                        continue
                    cut = (group_index, output, maximize, witness)
                    key_value = _support_cut_key(*cut)
                    if key_value not in cut_keys:
                        cuts.append(cut)
                        cut_keys.add(key_value)
                        added += 1
        _emit_progress(
            progress,
            "support_oracle_scan_finished",
            label=progress_label,
            formulation="shared_support_bounds",
            iteration=iteration,
            scan_seconds=time.perf_counter() - scan_started,
            support_queries=2 * group_count * dimension,
            added_support_cut_count=added,
            accumulated_support_cut_count=len(cuts),
            max_iteration_violation=max_iteration_violation,
            converged=added == 0,
            **process_memory_bytes(),
        )
        if checkpoint_path is not None and (
            added == 0 or iteration % checkpoint_every == 0
        ):
            _write_support_checkpoint(
                checkpoint_path,
                fingerprint=fingerprint,
                completed_iteration=iteration,
                cuts=cuts,
                dimension=dimension,
            )
            _emit_progress(
                progress,
                "support_checkpoint_written",
                label=progress_label,
                path=str(checkpoint_path),
                completed_iteration=iteration,
                support_cut_count=len(cuts),
                file_size_bytes=checkpoint_path.stat().st_size,
            )
        if added == 0:
            converged = True
            break
    assert solution is not None
    if not converged:
        _emit_progress(
            progress,
            "cutting_plane_iteration_limit_reached",
            label=progress_label,
            formulation="shared_support_bounds",
            completed_iteration=completed_iteration + max_iterations,
            local_iteration_limit=max_iterations,
            accumulated_support_cut_count=len(cuts),
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
        support_cut_count=len(cuts),
        **process_memory_bytes(),
    )
    if not verification.valid:
        solution.solver.status = (
            "cutting_plane_limit" if not converged else "verification_failed"
        )
    initial_cut_count = 2 * group_count * dimension
    solution.cuts = (
        [
            {
                "source": "full_simplex_ldp_seed",
                "constraint_count": int(ldp_seed.shape[0]),
                "pure_ldp_replacement": False,
            }
        ]
        if ldp_seed is not None
        else []
    ) + [
        {
            "source": "support_oracle",
            "formulation": "shared_support_bounds",
            "group": group_keys[group],
            "output": output,
            "direction": "maximum" if maximize else "minimum",
        }
        for group, output, maximize, _witness in cuts[initial_cut_count:]
    ]
    return solution, verification


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
    cut_formulation: str = "paired_witness",
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: bool = False,
    checkpoint_every: int = 1,
) -> tuple[ChannelSolution, VerificationResult]:
    if any(box.experimental for box in boxes.values()):
        raise ValueError("experimental DP-aware boxes cannot produce a certified result")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if cut_formulation not in {"paired_witness", "shared_support_bounds"}:
        raise ValueError("unknown robust cut formulation")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if cut_formulation == "paired_witness" and (checkpoint is not None or resume_checkpoint):
        raise ValueError("checkpointing requires shared_support_bounds")
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
    channel_variable_count = dimension * dimension
    variable_count = (
        channel_variable_count + 2 * len(boxes) * dimension
        if cut_formulation == "shared_support_bounds" and not pure_ldp
        else channel_variable_count
    )
    _emit_progress(
        progress,
        "robust_solve_started",
        label=progress_label,
        dimension=dimension,
        variable_count=variable_count,
        channel_variable_count=channel_variable_count,
        auxiliary_variable_count=variable_count - channel_variable_count,
        box_count=len(boxes),
        adjacency_count=len(adjacency),
        nominal_constraint_count=len(adjacency) * dimension,
        full_simplex_box_count=full_simplex_count,
        full_simplex_edge_count=full_simplex_edge_count,
        implied_ldp_epsilon=ldp_epsilon,
        pure_ldp_replacement=pure_ldp,
        max_cutting_plane_iterations=max_iterations,
        cut_formulation=cut_formulation,
        checkpoint_path=str(checkpoint) if checkpoint is not None else None,
        resume_checkpoint=resume_checkpoint,
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
    if cut_formulation == "shared_support_bounds":
        return _solve_shared_support_bounds(
            cost,
            block_weights,
            boxes,
            adjacency,
            tolerance=tolerance,
            max_iterations=max_iterations,
            time_limit=time_limit,
            ldp_seed=ldp_seed,
            progress=progress,
            progress_label=progress_label,
            solver_verbose=solver_verbose,
            heartbeat_seconds=heartbeat_seconds,
            checkpoint_path=checkpoint,
            resume_checkpoint=resume_checkpoint,
            checkpoint_every=checkpoint_every,
        )
    cuts: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    cut_keys: set[tuple[bytes, bytes, float, int]] = set()
    solution: ChannelSolution | None = None
    converged = False
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
            converged = True
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
        solution.solver.status = (
            "cutting_plane_limit" if not converged else "verification_failed"
        )
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
