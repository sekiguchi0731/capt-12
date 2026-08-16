from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from capt12.confidence.boxes import ConfidenceBox
from capt12.confidence.support import support
from capt12.mechanisms.lp import ChannelSolution, _solve_channel_lp, validate_channel
from capt12.privacy.adjacency import AdjacentPair


@dataclass
class VerificationResult:
    valid: bool
    max_violation: float
    realized_epsilon: float
    worst_case: dict | None
    checked_constraints: int


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
            epsilon = math.inf if minimum <= 0 < maximum else (math.log(maximum / minimum) if maximum > 0 and minimum > 0 else -math.inf)
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
) -> tuple[ChannelSolution, VerificationResult]:
    if any(box.experimental for box in boxes.values()):
        raise ValueError("experimental DP-aware boxes cannot produce a certified result")
    nominal = {key: box.nominal for key, box in boxes.items()}
    cuts: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    solution: ChannelSolution | None = None
    for iteration in range(max_iterations):
        solution = _solve_channel_lp(
            cost,
            block_weights,
            nominal,
            adjacency,
            tolerance=tolerance,
            time_limit=time_limit,
            extra_cuts=cuts,
        )
        if solution.channel is None:
            return solution, VerificationResult(False, math.inf, math.inf, {"error": solution.solver.message}, 0)
        added = 0
        for pair in adjacency:
            for output in range(solution.channel.shape[1]):
                maximum, p_max = support(solution.channel[:, output], boxes[pair.left], maximize=True)
                minimum, p_min = support(solution.channel[:, output], boxes[pair.right], maximize=False)
                violation = maximum - math.exp(pair.epsilon) * minimum
                if violation > tolerance:
                    cuts.append((p_max, p_min, pair.epsilon, output))
                    solution.cuts.append(
                        {"iteration": iteration, "left": pair.left, "right": pair.right, "output": output, "violation": float(violation)}
                    )
                    added += 1
        if added == 0:
            break
    assert solution is not None
    verification = verify_robust_channel(solution.channel, boxes, adjacency, tolerance=tolerance)
    if not verification.valid:
        solution.solver.status = "verification_failed"
    solution.solver.constraint_count += len(cuts)
    solution.cuts = [
        {"left_witness": left.tolist(), "right_witness": right.tolist(), "epsilon": eps, "output": output}
        for left, right, eps, output in cuts
    ]
    return solution, verification

