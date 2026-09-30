from __future__ import annotations

import math
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog

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


def _scale_small_inequality_rows(
    matrix: sparse.spmatrix | None,
    bounds: np.ndarray | None,
) -> tuple[sparse.csr_matrix | None, np.ndarray | None, np.ndarray]:
    """Scale small nonzero inequality rows without weakening solver feasibility.

    HiGHS applies an absolute feasibility tolerance.  A privacy row whose
    largest coefficient is much smaller than one can therefore be treated as
    satisfied even when its unscaled violation is scientifically meaningful.
    Multiplying such a row and its bound by a positive factor preserves the
    exact feasible half-space while making the solver's absolute tolerance
    stricter in the original units.  Rows already containing a coefficient of
    magnitude at least one are deliberately left unchanged: scaling those
    down would relax the effective original-space tolerance.
    """
    if matrix is None:
        return None, bounds, np.empty(0, dtype=float)
    csr = matrix.tocsr(copy=True)
    if bounds is None or np.asarray(bounds).shape != (csr.shape[0],):
        raise ValueError("inequality bounds must match the inequality matrix")
    if csr.shape[0] == 0:
        return csr, np.asarray(bounds, dtype=float), np.empty(0, dtype=float)
    row_max = np.asarray(abs(csr).max(axis=1).toarray(), dtype=float).reshape(-1)
    factors = np.ones(csr.shape[0], dtype=float)
    small = (row_max > 0) & (row_max < 1.0)
    factors[small] = 1.0 / row_max[small]
    if np.any(small):
        csr = sparse.diags(factors, format="csr") @ csr
    return csr, np.asarray(bounds, dtype=float) * factors, factors


def _primal_candidate_check(
    result: Any,
    *,
    a_eq: sparse.csr_matrix,
    b_eq: np.ndarray,
    a_ub: sparse.csr_matrix | None,
    b_ub: np.ndarray | None,
    variable_count: int,
    tolerance: float,
) -> dict[str, Any]:
    """Independently recompute primal feasibility in the original LP units."""
    vector = getattr(result, "x", None)
    if vector is None:
        return {
            "accepted": False,
            "reason": "missing_primal_vector",
            "equality_gap": math.inf,
            "inequality_gap": math.inf,
            "lower_bound_gap": math.inf,
            "upper_bound_gap": math.inf,
            "max_primal_gap": math.inf,
        }
    candidate = np.asarray(vector, dtype=float)
    if candidate.shape != (variable_count,) or not np.all(np.isfinite(candidate)):
        return {
            "accepted": False,
            "reason": "invalid_primal_vector",
            "equality_gap": math.inf,
            "inequality_gap": math.inf,
            "lower_bound_gap": math.inf,
            "upper_bound_gap": math.inf,
            "max_primal_gap": math.inf,
        }
    equality_gap = float(np.max(np.abs(a_eq @ candidate - b_eq)))
    inequality_gap = (
        float(max(0.0, np.max(a_ub @ candidate - b_ub)))
        if a_ub is not None and b_ub is not None and a_ub.shape[0]
        else 0.0
    )
    lower_bound_gap = float(max(0.0, -np.min(candidate)))
    upper_bound_gap = float(max(0.0, np.max(candidate - 1.0)))
    max_primal_gap = max(
        equality_gap,
        inequality_gap,
        lower_bound_gap,
        upper_bound_gap,
    )
    solver_success = bool(getattr(result, "success", False))
    return {
        "accepted": solver_success and max_primal_gap <= tolerance,
        "reason": (
            "accepted"
            if solver_success and max_primal_gap <= tolerance
            else ("solver_status" if not solver_success else "independent_primal_check")
        ),
        "equality_gap": equality_gap,
        "inequality_gap": inequality_gap,
        "lower_bound_gap": lower_bound_gap,
        "upper_bound_gap": upper_bound_gap,
        "max_primal_gap": max_primal_gap,
    }


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
    small_matrix_value: float = 1e-12,
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
    if not 1e-12 <= small_matrix_value < 1e-9:
        raise ValueError("small_matrix_value must be in [1e-12, 1e-9)")
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
        small_matrix_value=small_matrix_value,
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
    original_a_ub = sparse.vstack(matrices, format="csr") if matrices else None
    inequality_count = int(original_a_ub.shape[0]) if original_a_ub is not None else 0
    original_b_ub = np.zeros(inequality_count) if original_a_ub is not None else None
    a_ub, b_ub, inequality_row_scale = _scale_small_inequality_rows(
        original_a_ub,
        original_b_ub,
    )
    options: dict[str, float | bool | str | int] = {
        "dual_feasibility_tolerance": tolerance,
        "primal_feasibility_tolerance": tolerance,
        "ipm_optimality_tolerance": max(1e-12, min(tolerance, 1e-8)),
        "small_matrix_value": small_matrix_value,
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
        + _sparse_memory_bytes(original_a_ub)
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
        inequality_row_scaling="upscale_only_max_abs_to_one",
        scaled_inequality_row_count=int(np.sum(inequality_row_scale > 1.0)),
        max_inequality_row_scale=(
            float(np.max(inequality_row_scale)) if inequality_row_scale.size else 1.0
        ),
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
    solver_state: dict[str, Any] = {"method": "highs", "attempt": 1}
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
                    solver_method=solver_state["method"],
                    solver_attempt=solver_state["attempt"],
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
        with warnings.catch_warnings():
            # SciPy forwards this supported HiGHS option but does not list it
            # in linprog's public option schema, so suppress only that wrapper
            # warning. Keeping small probability coefficients is essential for
            # certificate-level feasibility at 1e-8.
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected:.*small_matrix_value",
                category=OptimizeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected:.*run_crossover",
                category=OptimizeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected:.*simplex_strategy",
                category=OptimizeWarning,
            )
            equality_matrix = a_eq.tocsr()

            def solve_once(
                method: str,
                method_options: dict[str, float | bool | str | int],
            ):
                return linprog(
                    objective,
                    A_ub=a_ub,
                    b_ub=b_ub,
                    A_eq=equality_matrix,
                    b_eq=b_eq,
                    bounds=(0.0, 1.0),
                    method=method,
                    options=method_options,
                )

            def verify_candidate(method_label: str, candidate: Any) -> dict[str, Any]:
                raw_check = _primal_candidate_check(
                    candidate,
                    a_eq=equality_matrix,
                    b_eq=b_eq,
                    a_ub=original_a_ub,
                    b_ub=original_b_ub,
                    variable_count=variable_count,
                    tolerance=tolerance,
                )
                check = raw_check
                if raw_check["accepted"]:
                    released_vector = np.asarray(candidate.x, dtype=float).copy()
                    released_channel = np.clip(
                        released_vector[:channel_variable_count].reshape(n, n),
                        0.0,
                        1.0,
                    )
                    cleanup_threshold = max(
                        tolerance * 1e-3,
                        np.finfo(float).eps * 100,
                    )
                    released_channel[
                        :,
                        np.max(released_channel, axis=0) <= cleanup_threshold,
                    ] = 0.0
                    row_sums = released_channel.sum(axis=1, keepdims=True)
                    if np.any(row_sums <= 0):
                        check = {
                            **raw_check,
                            "accepted": False,
                            "reason": "postprocess_zero_row",
                        }
                    else:
                        released_channel /= row_sums
                        released_vector[:channel_variable_count] = released_channel.reshape(-1)
                        check = _primal_candidate_check(
                            SimpleNamespace(x=released_vector, success=True),
                            a_eq=equality_matrix,
                            b_eq=b_eq,
                            a_ub=original_a_ub,
                            b_ub=original_b_ub,
                            variable_count=variable_count,
                            tolerance=tolerance,
                        )
                        if not check["accepted"]:
                            check["reason"] = "postprocess_primal_check"
                check["raw_max_primal_gap"] = raw_check["max_primal_gap"]
                _emit_progress(
                    progress,
                    "lp_candidate_verification_finished",
                    label=progress_label,
                    solver_method=method_label,
                    solver_reported_success=bool(getattr(candidate, "success", False)),
                    accepted=check["accepted"],
                    reason=check["reason"],
                    equality_gap=check["equality_gap"],
                    inequality_gap=check["inequality_gap"],
                    lower_bound_gap=check["lower_bound_gap"],
                    upper_bound_gap=check["upper_bound_gap"],
                    max_primal_gap=check["max_primal_gap"],
                    raw_max_primal_gap=check["raw_max_primal_gap"],
                    verification_tolerance=tolerance,
                    verification_matrix="original_unscaled_constraints",
                    **process_memory_bytes(),
                )
                return check

            result = solve_once("highs", options)
            attempts = [("highs", result)]
            candidate_check = verify_candidate("highs", result)
            # Dual simplex can spend hours pivoting on a degenerate robust
            # master even though the IPM solves the same mathematical LP much
            # faster. With an explicit per-attempt time limit, status 1 is not
            # accepted as a channel: retry the identical LP through IPM and
            # disable crossover so that it cannot fall back into another long
            # simplex phase. Feasibility/optimality tolerances and every
            # privacy constraint remain unchanged.
            if not result.success and int(result.status) == 1:
                no_crossover_options = {
                    **options,
                    "run_crossover": "off",
                }
                solver_state.update(method="highs-ipm", attempt=2)
                _emit_progress(
                    progress,
                    "lp_solver_retry_started",
                    label=progress_label,
                    retry_reason="primary_time_or_iteration_limit",
                    failed_method="highs",
                    failed_scipy_status=int(result.status),
                    failed_message=str(result.message),
                    retry_method="highs-ipm",
                    retry_strategy="without_crossover",
                    retry_attempt=2,
                    time_limit_seconds=time_limit,
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
                result = solve_once("highs-ipm", no_crossover_options)
                attempts.append(("highs-ipm-no-crossover", result))
                candidate_check = verify_candidate("highs-ipm-no-crossover", result)
                _emit_progress(
                    progress,
                    "lp_solver_retry_finished",
                    label=progress_label,
                    retry_method="highs-ipm",
                    retry_strategy="without_crossover",
                    retry_attempt=2,
                    success=candidate_check["accepted"],
                    solver_reported_success=bool(result.success),
                    scipy_status=int(result.status),
                    message=str(result.message),
                    iterations=getattr(result, "nit", None),
                    crossover_iterations=getattr(result, "crossover_nit", None),
                    time_limit_seconds=time_limit,
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
            # HiGHS' dual simplex may find an apparently optimal solution and
            # then downgrade it to status 4/Unknown during its stricter final
            # feasibility check on badly scaled robust masters.  Retrying the
            # identical mathematical LP with the independent HiGHS IPM path is
            # deterministic and does not relax any constraint or tolerance.
            # The returned channel still has to pass the support oracle,
            # independent robust verifier, and pure-epsilon post-solve repair.
            if (
                attempts[-1][0] == "highs"
                and not result.success
                and int(result.status) == 4
            ):
                solver_state.update(method="highs-ipm", attempt=2)
                _emit_progress(
                    progress,
                    "lp_solver_retry_started",
                    label=progress_label,
                    retry_reason="primary_numerical_status",
                    failed_method="highs",
                    failed_scipy_status=int(result.status),
                    failed_message=str(result.message),
                    retry_method="highs-ipm",
                    retry_attempt=2,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
                result = solve_once("highs-ipm", options)
                attempts.append(("highs-ipm", result))
                candidate_check = verify_candidate("highs-ipm", result)
                _emit_progress(
                    progress,
                    "lp_solver_retry_finished",
                    label=progress_label,
                    retry_method="highs-ipm",
                    retry_attempt=2,
                    success=candidate_check["accepted"],
                    solver_reported_success=bool(result.success),
                    scipy_status=int(result.status),
                    message=str(result.message),
                    iterations=getattr(result, "nit", None),
                    crossover_iterations=getattr(result, "crossover_nit", None),
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
            # Some highly degenerate shared-support masters are solved by IPM
            # to the requested primal/dual tolerances and then downgraded to
            # Unknown solely because crossover makes the basic solution less
            # accurate. A final IPM retry disables crossover and returns the
            # already accurate interior solution. Neither feasibility nor
            # optimality tolerance changes. Acceptance still requires the
            # support oracle, independent robust verification, strict channel
            # repair, and Decimal verification downstream.
            if (
                attempts[-1][0] == "highs-ipm"
                and not result.success
                and int(result.status) == 4
            ):
                no_crossover_options = {
                    **options,
                    "run_crossover": "off",
                }
                solver_state.update(method="highs-ipm", attempt=3)
                _emit_progress(
                    progress,
                    "lp_solver_retry_started",
                    label=progress_label,
                    retry_reason="crossover_numerical_status",
                    failed_method="highs-ipm",
                    failed_scipy_status=int(result.status),
                    failed_message=str(result.message),
                    retry_method="highs-ipm",
                    retry_strategy="without_crossover",
                    retry_attempt=3,
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
                result = solve_once("highs-ipm", no_crossover_options)
                attempts.append(("highs-ipm-no-crossover", result))
                candidate_check = verify_candidate("highs-ipm-no-crossover", result)
                _emit_progress(
                    progress,
                    "lp_solver_retry_finished",
                    label=progress_label,
                    retry_method="highs-ipm",
                    retry_strategy="without_crossover",
                    retry_attempt=3,
                    success=candidate_check["accepted"],
                    solver_reported_success=bool(result.success),
                    scipy_status=int(result.status),
                    message=str(result.message),
                    iterations=getattr(result, "nit", None),
                    crossover_iterations=getattr(result, "crossover_nit", None),
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
            # Presolve can itself report Unknown/Infeasible on a numerically
            # difficult but feasible robust master (a constant channel is
            # always feasible here).  This occurs in particular when the
            # primary simplex reaches its time limit and the first IPM run,
            # already without crossover, is downgraded to status 4.  Retry the
            # identical LP once more with presolve disabled.  All variables,
            # constraints, bounds, and feasibility/optimality tolerances stay
            # unchanged; only the HiGHS preprocessing path differs.
            if (
                attempts[-1][0] == "highs-ipm-no-crossover"
                and not result.success
                and int(result.status) == 4
            ):
                no_presolve_options = {
                    **options,
                    "run_crossover": "off",
                    "presolve": False,
                }
                retry_attempt = len(attempts) + 1
                solver_state.update(method="highs-ipm", attempt=retry_attempt)
                _emit_progress(
                    progress,
                    "lp_solver_retry_started",
                    label=progress_label,
                    retry_reason="presolve_numerical_status",
                    failed_method="highs-ipm-no-crossover",
                    failed_scipy_status=int(result.status),
                    failed_message=str(result.message),
                    retry_method="highs-ipm",
                    retry_strategy="without_crossover_or_presolve",
                    retry_attempt=retry_attempt,
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    presolve_changed=True,
                    presolve=False,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
                result = solve_once("highs-ipm", no_presolve_options)
                attempts.append(("highs-ipm-no-crossover-no-presolve", result))
                candidate_check = verify_candidate(
                    "highs-ipm-no-crossover-no-presolve",
                    result,
                )
                _emit_progress(
                    progress,
                    "lp_solver_retry_finished",
                    label=progress_label,
                    retry_method="highs-ipm",
                    retry_strategy="without_crossover_or_presolve",
                    retry_attempt=retry_attempt,
                    success=candidate_check["accepted"],
                    solver_reported_success=bool(result.success),
                    scipy_status=int(result.status),
                    message=str(result.message),
                    iterations=getattr(result, "nit", None),
                    crossover_iterations=getattr(result, "crossover_nit", None),
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    presolve_changed=True,
                    presolve=False,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
            # A no-crossover IPM interior point can be accurate but still fail
            # to produce a valid basis on a degenerate master.  Before changing
            # algorithms, try the same scaled LP with crossover explicitly on
            # and presolve off.  This is a solver-path change only: constraints,
            # bounds, and every feasibility/optimality tolerance are identical.
            if not candidate_check["accepted"]:
                crossover_options = {
                    **options,
                    "run_crossover": "on",
                    "presolve": False,
                }
                retry_attempt = len(attempts) + 1
                failed_method = attempts[-1][0]
                solver_state.update(method="highs-ipm", attempt=retry_attempt)
                _emit_progress(
                    progress,
                    "lp_solver_retry_started",
                    label=progress_label,
                    retry_reason="strict_candidate_unavailable",
                    failed_method=failed_method,
                    failed_scipy_status=int(result.status),
                    failed_message=str(result.message),
                    retry_method="highs-ipm",
                    retry_strategy="with_crossover_without_presolve",
                    retry_attempt=retry_attempt,
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    ipm_optimality_tolerance=float(options["ipm_optimality_tolerance"]),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    presolve=False,
                    run_crossover="on",
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
                result = solve_once("highs-ipm", crossover_options)
                attempts.append(("highs-ipm-crossover-no-presolve", result))
                candidate_check = verify_candidate(
                    "highs-ipm-crossover-no-presolve",
                    result,
                )
                _emit_progress(
                    progress,
                    "lp_solver_retry_finished",
                    label=progress_label,
                    retry_method="highs-ipm",
                    retry_strategy="with_crossover_without_presolve",
                    retry_attempt=retry_attempt,
                    success=candidate_check["accepted"],
                    solver_reported_success=bool(result.success),
                    scipy_status=int(result.status),
                    message=str(result.message),
                    iterations=getattr(result, "nit", None),
                    crossover_iterations=getattr(result, "crossover_nit", None),
                    max_primal_gap=candidate_check["max_primal_gap"],
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    ipm_optimality_tolerance=float(options["ipm_optimality_tolerance"]),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    presolve=False,
                    run_crossover="on",
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
            # HiGHS normally uses dual simplex.  A final explicit primal-simplex
            # solve supplies an algorithmically independent route through the
            # same LP when both IPM/crossover variants are numerically rejected.
            # SciPy forwards simplex_strategy=4 to HiGHS, where it denotes the
            # primal strategy.  The candidate is still accepted only by the
            # original-space check above and the robust verifiers downstream.
            if not candidate_check["accepted"]:
                primal_options = {
                    **options,
                    "presolve": False,
                    "simplex_strategy": 4,
                }
                retry_attempt = len(attempts) + 1
                failed_method = attempts[-1][0]
                solver_state.update(method="highs-primal-simplex", attempt=retry_attempt)
                _emit_progress(
                    progress,
                    "lp_solver_retry_started",
                    label=progress_label,
                    retry_reason="strict_candidate_unavailable",
                    failed_method=failed_method,
                    failed_scipy_status=int(result.status),
                    failed_message=str(result.message),
                    retry_method="highs-ds",
                    retry_strategy="primal_simplex_without_presolve",
                    retry_attempt=retry_attempt,
                    simplex_strategy=4,
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    presolve=False,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
                result = solve_once("highs-ds", primal_options)
                attempts.append(("highs-primal-simplex-no-presolve", result))
                candidate_check = verify_candidate(
                    "highs-primal-simplex-no-presolve",
                    result,
                )
                _emit_progress(
                    progress,
                    "lp_solver_retry_finished",
                    label=progress_label,
                    retry_method="highs-ds",
                    retry_strategy="primal_simplex_without_presolve",
                    retry_attempt=retry_attempt,
                    success=candidate_check["accepted"],
                    solver_reported_success=bool(result.success),
                    scipy_status=int(result.status),
                    message=str(result.message),
                    iterations=getattr(result, "nit", None),
                    max_primal_gap=candidate_check["max_primal_gap"],
                    simplex_strategy=4,
                    primal_feasibility_tolerance=float(tolerance),
                    dual_feasibility_tolerance=float(tolerance),
                    privacy_constraint_tolerance_changed=False,
                    optimality_tolerance_changed=False,
                    presolve=False,
                    solver_elapsed_seconds=time.perf_counter() - started,
                    **process_memory_bytes(),
                )
            selected_method = attempts[-1][0]
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
    channel: np.ndarray | None = None
    auxiliary: np.ndarray | None = None
    postprocess_check: dict[str, Any] | None = None
    if candidate_check["accepted"]:
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
        released_vector = np.asarray(result.x, dtype=float).copy()
        released_vector[:channel_variable_count] = channel.reshape(-1)
        postprocess_check = _primal_candidate_check(
            SimpleNamespace(x=released_vector, success=True),
            a_eq=a_eq.tocsr(),
            b_eq=b_eq,
            a_ub=original_a_ub,
            b_ub=original_b_ub,
            variable_count=variable_count,
            tolerance=tolerance,
        )
        _emit_progress(
            progress,
            "lp_postprocess_verification_finished",
            label=progress_label,
            solver_method=selected_method,
            accepted=postprocess_check["accepted"],
            reason=postprocess_check["reason"],
            equality_gap=postprocess_check["equality_gap"],
            inequality_gap=postprocess_check["inequality_gap"],
            lower_bound_gap=postprocess_check["lower_bound_gap"],
            upper_bound_gap=postprocess_check["upper_bound_gap"],
            max_primal_gap=postprocess_check["max_primal_gap"],
            verification_tolerance=tolerance,
            verification_matrix="original_unscaled_constraints",
            **process_memory_bytes(),
        )
        if not postprocess_check["accepted"]:
            channel = None
            auxiliary = None
    accepted = bool(
        candidate_check["accepted"]
        and postprocess_check is not None
        and postprocess_check["accepted"]
    )
    primal_gap = (
        float(postprocess_check["max_primal_gap"])
        if postprocess_check is not None
        else (
            float(candidate_check["max_primal_gap"])
            if math.isfinite(candidate_check["max_primal_gap"])
            else None
        )
    )
    # HiGHS reports an optimal primal/dual pair; scipy does not expose a
    # standalone LP duality-gap field, so record zero only for a solver-optimal
    # result that also passes both independent original-space primal checks.
    dual_gap = 0.0 if accepted else None
    objective_value = (
        float(raw_objective @ channel.reshape(-1)) if accepted and channel is not None else None
    )
    _emit_progress(
        progress,
        "lp_solver_finished",
        label=progress_label,
        solver_elapsed_seconds=runtime,
        dimension=n,
        variable_count=variable_count,
        total_constraint_count=n + inequality_count,
        success=accepted,
        solver_reported_success=bool(result.success),
        independent_candidate_accepted=candidate_check["accepted"],
        postprocess_accepted=(
            postprocess_check["accepted"] if postprocess_check is not None else False
        ),
        max_primal_gap=primal_gap,
        solver_method=selected_method,
        solver_attempt_count=len(attempts),
        fallback_used=len(attempts) > 1,
        scipy_status=int(result.status),
        message=str(result.message),
        iterations=getattr(result, "nit", None),
        crossover_iterations=getattr(result, "crossover_nit", None),
        objective_value=objective_value,
        **process_memory_bytes(),
    )
    failure_status = (
        "verification_failed" if bool(result.success) and not accepted else "solver_failure"
    )
    info = SolverInfo(
        status="optimal" if accepted else failure_status,
        objective=objective_value,
        runtime_seconds=runtime,
        iterations=getattr(result, "nit", None),
        primal_gap=primal_gap,
        dual_gap=dual_gap,
        message=" | ".join(
            f"{method}(status={int(attempt.status)}): {attempt.message}"
            for method, attempt in attempts
        ),
        variable_count=variable_count,
        constraint_count=n + inequality_count,
        estimated_memory_bytes=int(matrix_memory_bytes * 2),
    )
    if not accepted or channel is None:
        return ChannelSolution(None, info)
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
