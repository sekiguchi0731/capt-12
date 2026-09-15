from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from capt12.mechanisms.lp import ChannelSolution, SolverInfo, solve_full_lp, validate_channel
from capt12.privacy.adjacency import AdjacentPair


@dataclass(frozen=True)
class PBPOracleSolution:
    """Profile-indexed nominal PBP channels.

    Selecting one of these channels requires the realized profile online.  The
    result is therefore an oracle utility bound, not a deployable CAPT method.
    """

    channels: dict[str, np.ndarray] | None
    solver: SolverInfo
    nominal_max_violation: float | None
    uses_sensitive_value_online: bool = True
    deployable_under_capt: bool = False


def _normalized_profiles(
    distributions: Mapping[str, np.ndarray], dimension: int
) -> dict[str, np.ndarray]:
    profiles: dict[str, np.ndarray] = {}
    for key, raw in distributions.items():
        value = np.asarray(raw, dtype=float)
        if (
            value.shape != (dimension,)
            or not np.isfinite(value).all()
            or np.any(value < 0)
            or value.sum() <= 0
        ):
            raise ValueError(f"invalid nominal profile distribution: {key}")
        profiles[str(key)] = value / value.sum()
    if not profiles:
        raise ValueError("at least one nominal profile is required")
    return profiles


def _validate_adjacency_keys(
    profiles: Mapping[str, np.ndarray], adjacency: Sequence[AdjacentPair]
) -> None:
    unknown = sorted(
        {
            endpoint
            for pair in adjacency
            for endpoint in (pair.left, pair.right)
            if endpoint not in profiles
        }
    )
    if unknown:
        raise ValueError(f"adjacency references unknown nominal profiles: {unknown}")


def nominal_profile_privacy_violation(
    channels: Mapping[str, np.ndarray],
    profiles: Mapping[str, np.ndarray],
    adjacency: Sequence[AdjacentPair],
) -> float:
    """Return the largest additive nominal PBP constraint violation."""
    maximum = -math.inf
    for pair in adjacency:
        left = np.asarray(profiles[pair.left]) @ np.asarray(channels[pair.left])
        right = np.asarray(profiles[pair.right]) @ np.asarray(channels[pair.right])
        maximum = max(maximum, float(np.max(left - math.exp(pair.epsilon) * right)))
    return 0.0 if maximum == -math.inf else maximum


def solve_pbp_oracle(
    cost: np.ndarray,
    nominal_profiles: Mapping[str, np.ndarray],
    adjacency: Sequence[AdjacentPair],
    *,
    profile_weights: Mapping[str, float] | None = None,
    source_split: str = "D_design",
    tolerance: float = 1e-9,
    time_limit: float | None = None,
) -> PBPOracleSolution:
    """Solve the profile-indexed LP from the PBP definition.

    ``source_split`` is an executable data-use guard: nominal profiles and
    objective weights may be fitted from D_design only.  D_cert and D_test are
    never accepted as training inputs.
    """
    if source_split != "D_design":
        raise ValueError("nominal PBP must be fitted from D_design only")
    matrix_cost = np.asarray(cost, dtype=float)
    dimension = matrix_cost.shape[0]
    if matrix_cost.shape != (dimension, dimension) or not np.isfinite(matrix_cost).all():
        raise ValueError("cost must be a finite square matrix")
    profiles = _normalized_profiles(nominal_profiles, dimension)
    _validate_adjacency_keys(profiles, adjacency)
    keys = tuple(sorted(profiles))
    key_index = {key: index for index, key in enumerate(keys)}

    if profile_weights is None:
        weights = {key: 1 / len(keys) for key in keys}
    else:
        weights = {key: float(profile_weights.get(key, 0.0)) for key in keys}
        if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("profile_weights must be nonnegative with positive mass")
        total = sum(weights.values())
        weights = {key: value / total for key, value in weights.items()}

    variable_count = len(keys) * dimension * dimension

    def variable(profile: str, token: int, output: int) -> int:
        return (key_index[profile] * dimension + token) * dimension + output

    objective = np.zeros(variable_count, dtype=float)
    for key in keys:
        weighted_profile = weights[key] * profiles[key]
        for token in range(dimension):
            start = variable(key, token, 0)
            objective[start : start + dimension] = weighted_profile[token] * matrix_cost[token]

    eq_rows: list[int] = []
    eq_cols: list[int] = []
    eq_values: list[float] = []
    for profile_index, key in enumerate(keys):
        for token in range(dimension):
            row = profile_index * dimension + token
            for output in range(dimension):
                eq_rows.append(row)
                eq_cols.append(variable(key, token, output))
                eq_values.append(1.0)
    equality = sparse.coo_matrix(
        (eq_values, (eq_rows, eq_cols)),
        shape=(len(keys) * dimension, variable_count),
    ).tocsr()

    ub_rows: list[int] = []
    ub_cols: list[int] = []
    ub_values: list[float] = []
    for pair_index, pair in enumerate(adjacency):
        factor = math.exp(pair.epsilon)
        for output in range(dimension):
            row = pair_index * dimension + output
            for token in range(dimension):
                left_value = profiles[pair.left][token]
                right_value = -factor * profiles[pair.right][token]
                if left_value:
                    ub_rows.append(row)
                    ub_cols.append(variable(pair.left, token, output))
                    ub_values.append(float(left_value))
                if right_value:
                    ub_rows.append(row)
                    ub_cols.append(variable(pair.right, token, output))
                    ub_values.append(float(right_value))
    inequality = sparse.coo_matrix(
        (ub_values, (ub_rows, ub_cols)),
        shape=(len(adjacency) * dimension, variable_count),
    ).tocsr()

    options: dict[str, float] = {
        "primal_feasibility_tolerance": tolerance,
        "dual_feasibility_tolerance": tolerance,
    }
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive")
        options["time_limit"] = float(time_limit)
    started = time.perf_counter()
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(inequality.shape[0]),
        A_eq=equality,
        b_eq=np.ones(equality.shape[0]),
        bounds=(0.0, 1.0),
        method="highs",
        options=options,
    )
    runtime = time.perf_counter() - started
    solver = SolverInfo(
        status="optimal" if result.success else f"failed_{result.status}",
        objective=float(result.fun) if result.success else None,
        runtime_seconds=runtime,
        iterations=int(result.nit) if result.nit is not None else None,
        message=str(result.message),
        variable_count=variable_count,
        constraint_count=equality.shape[0] + inequality.shape[0],
        estimated_memory_bytes=int(
            objective.nbytes
            + equality.data.nbytes
            + equality.indices.nbytes
            + equality.indptr.nbytes
            + inequality.data.nbytes
            + inequality.indices.nbytes
            + inequality.indptr.nbytes
        ),
    )
    if not result.success:
        return PBPOracleSolution(None, solver, None)

    channels: dict[str, np.ndarray] = {}
    solution = np.asarray(result.x, dtype=float).reshape(len(keys), dimension, dimension)
    for index, key in enumerate(keys):
        channel = solution[index]
        channel[channel < 0] = 0
        channel /= channel.sum(axis=1, keepdims=True)
        validate_channel(channel)
        channels[key] = channel
    violation = nominal_profile_privacy_violation(channels, profiles, adjacency)
    if violation > max(tolerance * 10, 1e-8):
        raise RuntimeError("PBP oracle LP failed independent nominal verification")
    return PBPOracleSolution(channels, solver, violation)


def solve_pbp_common_nominal(
    cost: np.ndarray,
    input_weights: np.ndarray,
    nominal_profiles: Mapping[str, np.ndarray],
    adjacency: Sequence[AdjacentPair],
    *,
    source_split: str = "D_design",
    tolerance: float = 1e-9,
    time_limit: float | None = None,
) -> ChannelSolution:
    """Solve common-channel point-estimate profile privacy on D_design."""
    if source_split != "D_design":
        raise ValueError("common nominal PBP must be fitted from D_design only")
    dimension = np.asarray(cost).shape[0]
    profiles = _normalized_profiles(nominal_profiles, dimension)
    _validate_adjacency_keys(profiles, adjacency)
    return solve_full_lp(
        np.asarray(cost, dtype=float),
        np.asarray(input_weights, dtype=float),
        profiles,
        adjacency,
        full_max_k=dimension,
        force=True,
        tolerance=tolerance,
        time_limit=time_limit,
        progress_label="pbp_common_nominal",
    )
