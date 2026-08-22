from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from capt12.privacy.adjacency import AdjacentPair
from capt12.utils.progress import process_memory_bytes

ProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _emit_progress(
    progress: ProgressCallback | None,
    event: str,
    **fields: Any,
) -> None:
    if progress is None:
        return
    try:
        progress(event, fields)
    except Exception as error:  # pragma: no cover - logging must never invalidate a solve
        print(f"[progress_callback_error] event={event} error={error!r}", flush=True)


def _sparse_memory_bytes(matrix: sparse.spmatrix | None) -> int:
    if matrix is None:
        return 0
    csr = matrix.tocsr()
    return int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)


@dataclass
class SolverInfo:
    status: str
    objective: float | None
    runtime_seconds: float
    iterations: int | None
    primal_gap: float | None = None
    dual_gap: float | None = None
    message: str = ""
    variable_count: int = 0
    constraint_count: int = 0
    estimated_memory_bytes: int = 0


@dataclass
class ChannelSolution:
    channel: np.ndarray | None
    solver: SolverInfo
    cuts: list[dict] = field(default_factory=list)
    auxiliary: np.ndarray | None = None


def validate_channel(channel: np.ndarray, tolerance: float = 1e-8) -> None:
    channel = np.asarray(channel, dtype=float)
    if channel.ndim != 2 or channel.shape[0] != channel.shape[1]:
        raise ValueError("channel must be a square matrix")
    if np.min(channel) < -tolerance:
        raise ValueError("channel has a negative entry")
    if not np.allclose(channel.sum(axis=1), 1.0, atol=tolerance, rtol=0):
        raise ValueError("channel rows must sum to one")


def lift_block_channel(
    block_channel: np.ndarray, token_to_block: np.ndarray, decoder: np.ndarray
) -> np.ndarray:
    """Lift R to Q with Q[z,o] = R[c(z),c(o)] nu[c(o),o]."""
    block_channel = np.asarray(block_channel, dtype=float)
    assignment = np.asarray(token_to_block, dtype=int)
    decoder = np.asarray(decoder, dtype=float)
    l_count = block_channel.shape[0]
    if block_channel.shape != (l_count, l_count):
        raise ValueError("block channel must be square")
    if decoder.shape[0] != l_count or decoder.shape[1] != len(assignment):
        raise ValueError("decoder shape does not match assignment")
    if set(assignment.tolist()) != set(range(l_count)):
        raise ValueError("blocks must be non-empty and numbered contiguously")
    q = (
        block_channel[assignment[:, None], assignment[None, :]]
        * decoder[assignment[None, :], np.arange(len(assignment))[None, :]]
    )
    validate_channel(q)
    return q


def privacy_violations(
    channel: np.ndarray,
    group_distributions: Mapping[str, np.ndarray],
    adjacency: Sequence[AdjacentPair],
) -> list[dict]:
    violations = []
    for pair in adjacency:
        left = np.asarray(group_distributions[pair.left]) @ channel
        right = np.asarray(group_distributions[pair.right]) @ channel
        diff = left - math.exp(pair.epsilon) * right
        for output, value in enumerate(diff):
            if value > 0:
                violations.append(
                    {
                        "left": pair.left,
                        "right": pair.right,
                        "output": output,
                        "violation": float(value),
                    }
                )
    return violations


def _solve_channel_lp(
    cost: np.ndarray,
    input_weights: np.ndarray,
    group_distributions: Mapping[str, np.ndarray],
    adjacency: Sequence[AdjacentPair],
    *,
    tolerance: float = 1e-9,
    time_limit: float | None = None,
    extra_cuts: Sequence[tuple[np.ndarray, np.ndarray, float, int]] = (),
    precompiled_ub: sparse.spmatrix | None = None,
    auxiliary_variable_count: int = 0,
    progress: ProgressCallback | None = None,
    progress_label: str = "channel_lp",
    solver_verbose: bool = False,
    heartbeat_seconds: float = 60.0,
) -> ChannelSolution:
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    if cost.shape != (n, n):
        raise ValueError("cost must be square")
    weights = np.asarray(input_weights, dtype=float)
    if weights.shape != (n,) or weights.sum() <= 0:
        raise ValueError("input_weights must be nonnegative and have length n")
    weights = weights / weights.sum()
    if auxiliary_variable_count < 0:
        raise ValueError("auxiliary_variable_count must be nonnegative")
    channel_variable_count = n * n
    variable_count = channel_variable_count + auxiliary_variable_count
    _emit_progress(
        progress,
        "lp_problem_build_started",
        label=progress_label,
        dimension=n,
        variable_count=variable_count,
        channel_variable_count=channel_variable_count,
        auxiliary_variable_count=auxiliary_variable_count,
        nominal_adjacency_count=len(adjacency),
        extra_cut_count=len(extra_cuts),
        precompiled_constraint_count=(
            int(precompiled_ub.shape[0]) if precompiled_ub is not None else 0
        ),
        solver_verbose=solver_verbose,
        heartbeat_seconds=heartbeat_seconds,
        time_limit_seconds=time_limit,
        **process_memory_bytes(),
    )
    build_started = time.perf_counter()
    raw_objective = (weights[:, None] * cost).reshape(-1)
    # HiGHS' feasibility/duality tolerances are absolute.  Criteo distortion
    # coefficients can be around 1e-10, in which case the unscaled objective is
    # numerically indistinguishable from zero and an arbitrary feasible channel
    # can be reported as optimal.  Row-wise centering changes the objective only
    # by a constant because every channel row sums to one; positive scaling also
    # preserves the argmin.
    centered = weights[:, None] * (cost - np.min(cost, axis=1, keepdims=True))
    objective_scale = float(np.max(np.abs(centered)))
    channel_objective = (
        centered.reshape(-1) / objective_scale
        if objective_scale > 0
        else np.zeros_like(raw_objective)
    )
    objective = np.r_[channel_objective, np.zeros(auxiliary_variable_count)]
    a_eq = sparse.lil_matrix((n, variable_count), dtype=float)
    for row in range(n):
        a_eq[row, row * n : (row + 1) * n] = 1.0
    b_eq = np.ones(n)
    rows: list[sparse.csr_matrix] = []
    for pair in adjacency:
        left = np.asarray(group_distributions[pair.left], dtype=float)
        right = np.asarray(group_distributions[pair.right], dtype=float)
        coefficient = left - math.exp(pair.epsilon) * right
        for output in range(n):
            row = sparse.lil_matrix((1, variable_count), dtype=float)
            row[0, np.arange(n) * n + output] = coefficient
            rows.append(row.tocsr())
    for left, right, epsilon, output in extra_cuts:
        coefficient = np.asarray(left) - math.exp(epsilon) * np.asarray(right)
        row = sparse.lil_matrix((1, variable_count), dtype=float)
        row[0, np.arange(n) * n + output] = coefficient
        rows.append(row.tocsr())
    matrices = []
    if precompiled_ub is not None:
        compiled = precompiled_ub.tocsr()
        if compiled.shape[1] == channel_variable_count and auxiliary_variable_count:
            compiled = sparse.hstack(
                [compiled, sparse.csr_matrix((compiled.shape[0], auxiliary_variable_count))],
                format="csr",
            )
        if compiled.shape[1] != variable_count:
            raise ValueError(
                "precompiled_ub column count must equal the channel or total variable count"
            )
        matrices.append(compiled)
    if rows:
        matrices.append(sparse.vstack(rows, format="csr"))
    a_ub = sparse.vstack(matrices, format="csr") if matrices else None
    inequality_count = int(a_ub.shape[0]) if a_ub is not None else 0
    b_ub = np.zeros(inequality_count) if a_ub is not None else None
    options: dict[str, float] = {
        "dual_feasibility_tolerance": tolerance,
        "primal_feasibility_tolerance": tolerance,
    }
    if time_limit is not None:
        options["time_limit"] = time_limit
    if solver_verbose:
        options["disp"] = True
    matrix_memory_bytes = (
        objective.nbytes
        + b_eq.nbytes
        + (b_ub.nbytes if b_ub is not None else 0)
        + _sparse_memory_bytes(a_eq)
        + _sparse_memory_bytes(a_ub)
    )
    _emit_progress(
        progress,
        "lp_problem_build_finished",
        label=progress_label,
        build_seconds=time.perf_counter() - build_started,
        dimension=n,
        variable_count=variable_count,
        channel_variable_count=channel_variable_count,
        auxiliary_variable_count=auxiliary_variable_count,
        equality_constraint_count=n,
        inequality_constraint_count=inequality_count,
        total_constraint_count=n + inequality_count,
        matrix_nonzero_count=(int(a_eq.nnz) + (int(a_ub.nnz) if a_ub is not None else 0)),
        matrix_memory_bytes=matrix_memory_bytes,
        objective_scale=objective_scale,
        raw_objective_min=float(np.min(raw_objective)),
        raw_objective_max=float(np.max(raw_objective)),
        **process_memory_bytes(),
    )
    started = time.perf_counter()
    _emit_progress(
        progress,
        "lp_solver_started",
        label=progress_label,
        method="highs",
        dimension=n,
        variable_count=variable_count,
        total_constraint_count=n + inequality_count,
        time_limit_seconds=time_limit,
        **process_memory_bytes(),
    )
    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    if progress is not None and heartbeat_seconds > 0:

        def heartbeat() -> None:
            sequence = 0
            while not heartbeat_stop.wait(heartbeat_seconds):
                sequence += 1
                _emit_progress(
                    progress,
                    "lp_solver_heartbeat",
                    label=progress_label,
                    heartbeat_sequence=sequence,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    dimension=n,
                    variable_count=variable_count,
                    total_constraint_count=n + inequality_count,
                    **process_memory_bytes(),
                )

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"capt12-heartbeat-{progress_label}",
            daemon=True,
        )
        heartbeat_thread.start()
    try:
        result = linprog(
            objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq.tocsr(),
            b_eq=b_eq,
            bounds=(0.0, 1.0),
            method="highs",
            options=options,
        )
    except BaseException as error:
        _emit_progress(
            progress,
            "lp_solver_raised",
            label=progress_label,
            solver_elapsed_seconds=time.perf_counter() - started,
            error_type=type(error).__name__,
            error=str(error),
            **process_memory_bytes(),
        )
        raise
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
    runtime = time.perf_counter() - started
    _emit_progress(
        progress,
        "lp_solver_finished",
        label=progress_label,
        solver_elapsed_seconds=runtime,
        dimension=n,
        variable_count=variable_count,
        total_constraint_count=n + inequality_count,
        success=bool(result.success),
        scipy_status=int(result.status),
        message=str(result.message),
        iterations=getattr(result, "nit", None),
        crossover_iterations=getattr(result, "crossover_nit", None),
        objective_value=(
            float(raw_objective @ result.x[:channel_variable_count])
            if result.success
            else None
        ),
        **process_memory_bytes(),
    )
    primal_gap = None
    dual_gap = None
    if result.success:
        eq_gap = float(np.max(np.abs(a_eq.tocsr() @ result.x - b_eq)))
        ub_gap = float(max(0.0, np.max(a_ub @ result.x - b_ub))) if a_ub is not None else 0.0
        primal_gap = max(eq_gap, ub_gap, float(max(0.0, -np.min(result.x))))
        # HiGHS reports an optimal primal/dual pair; scipy does not expose a
        # standalone LP duality-gap field, so record zero only on optimal exit.
        dual_gap = 0.0
    info = SolverInfo(
        status="optimal" if result.success else "solver_failure",
        objective=(
            float(raw_objective @ result.x[:channel_variable_count])
            if result.success
            else None
        ),
        runtime_seconds=runtime,
        iterations=getattr(result, "nit", None),
        primal_gap=primal_gap,
        dual_gap=dual_gap,
        message=result.message,
        variable_count=variable_count,
        constraint_count=n + inequality_count,
        estimated_memory_bytes=int(matrix_memory_bytes * 2),
    )
    if not result.success:
        return ChannelSolution(None, info)
    channel = np.clip(result.x[:channel_variable_count].reshape(n, n), 0.0, 1.0)
    # A finite-epsilon LDP solution can only leave an output unused in every
    # row.  HiGHS may return ~1e-13 residue in one row of such a column, which
    # is feasible under the additive solver tolerance but makes a ratio-based
    # realized-epsilon diagnostic spuriously infinite.  Remove only columns
    # that are uniformly below a much smaller cleanup threshold.
    cleanup_threshold = max(tolerance * 1e-3, np.finfo(float).eps * 100)
    channel[:, np.max(channel, axis=0) <= cleanup_threshold] = 0.0
    channel /= channel.sum(axis=1, keepdims=True)
    validate_channel(channel, max(tolerance * 10, 1e-7))
    auxiliary = (
        np.asarray(result.x[channel_variable_count:], dtype=float)
        if auxiliary_variable_count
        else None
    )
    return ChannelSolution(channel, info, auxiliary=auxiliary)


def solve_block_lp(
    cost: np.ndarray,
    block_weights: np.ndarray,
    group_distributions: Mapping[str, np.ndarray],
    adjacency: Sequence[AdjacentPair],
    **kwargs,
) -> ChannelSolution:
    return _solve_channel_lp(cost, block_weights, group_distributions, adjacency, **kwargs)


def solve_ldp_block_lp(
    cost: np.ndarray,
    block_weights: np.ndarray,
    epsilon: float,
    **kwargs,
) -> ChannelSolution:
    """Solve the optimal row-wise epsilon-LDP channel in the same block class."""
    n = int(np.asarray(cost).shape[0])
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    progress = kwargs.get("progress")
    progress_label = str(kwargs.get("progress_label", "ldp_block_lp"))
    _emit_progress(
        progress,
        "ldp_constraint_build_started",
        label=progress_label,
        dimension=n,
        epsilon=epsilon,
        expected_constraint_count=n * (n - 1) * n,
        **process_memory_bytes(),
    )
    started = time.perf_counter()
    matrix = ldp_constraint_matrix(n, epsilon)
    _emit_progress(
        progress,
        "ldp_constraint_build_finished",
        label=progress_label,
        dimension=n,
        epsilon=epsilon,
        constraint_count=int(matrix.shape[0]),
        nonzero_count=int(matrix.nnz),
        matrix_memory_bytes=_sparse_memory_bytes(matrix),
        build_seconds=time.perf_counter() - started,
        **process_memory_bytes(),
    )
    return _solve_channel_lp(
        cost,
        block_weights,
        {},
        [],
        precompiled_ub=matrix,
        **kwargs,
    )


def ldp_constraint_matrix(n: int, epsilon: float) -> sparse.csr_matrix:
    """Build all ordered row-wise epsilon-LDP inequalities."""
    if n < 1:
        raise ValueError("LDP dimension must be positive")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    pairs = [(left, right) for left in range(n) for right in range(n) if left != right]
    pair_left = np.asarray([left for left, _ in pairs], dtype=int)
    pair_right = np.asarray([right for _, right in pairs], dtype=int)
    outputs = np.tile(np.arange(n, dtype=int), len(pairs))
    left_rows = np.repeat(pair_left, n)
    right_rows = np.repeat(pair_right, n)
    row_indices = np.arange(len(outputs), dtype=int)
    matrix = sparse.coo_matrix(
        (
            np.r_[np.ones(len(outputs)), -math.exp(epsilon) * np.ones(len(outputs))],
            (
                np.r_[row_indices, row_indices],
                np.r_[left_rows * n + outputs, right_rows * n + outputs],
            ),
        ),
        shape=(len(outputs), n * n),
    ).tocsr()
    return matrix


def full_problem_size(k: int, adjacency_count: int) -> dict[str, int]:
    variables = k * k
    constraints = k + adjacency_count * k
    # sparse coefficient, index, and solver work estimate; intentionally conservative
    estimated_memory = int((variables * 24) + (constraints * max(k, 1) * 24))
    return {
        "variables": variables,
        "constraints": constraints,
        "estimated_memory_bytes": estimated_memory,
    }


def solve_full_lp(
    cost: np.ndarray,
    token_weights: np.ndarray,
    group_distributions: Mapping[str, np.ndarray],
    adjacency: Sequence[AdjacentPair],
    *,
    full_max_k: int = 256,
    force: bool = False,
    **kwargs,
) -> ChannelSolution:
    k = int(np.asarray(cost).shape[0])
    size = full_problem_size(k, len(adjacency))
    if k > full_max_k and not force:
        return ChannelSolution(
            None,
            SolverInfo(
                status="skipped_safety_limit",
                objective=None,
                runtime_seconds=0.0,
                iterations=0,
                message=f"K={k} exceeds full_max_k={full_max_k}; pass --force-full to attempt",
                variable_count=size["variables"],
                constraint_count=size["constraints"],
                estimated_memory_bytes=size["estimated_memory_bytes"],
            ),
        )
    return _solve_channel_lp(cost, token_weights, group_distributions, adjacency, **kwargs)
