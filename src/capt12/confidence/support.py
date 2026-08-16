from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from capt12.confidence.boxes import ConfidenceBox


def _greedy_box_simplex(coefficients: np.ndarray, box: ConfidenceBox, maximize: bool) -> tuple[float, np.ndarray]:
    coefficient = np.asarray(coefficients, dtype=float)
    point = box.lower.copy()
    remaining = 1.0 - point.sum()
    order = np.argsort(coefficient)
    if maximize:
        order = order[::-1]
    for idx in order:
        addition = min(remaining, box.upper[idx] - point[idx])
        point[idx] += addition
        remaining -= addition
        if remaining <= 1e-12:
            break
    if remaining > 1e-8:
        raise ValueError("confidence box is simplex-infeasible")
    return float(coefficient @ point), point


def support(
    coefficients: np.ndarray, box: ConfidenceBox, *, maximize: bool = True
) -> tuple[float, np.ndarray]:
    """Optimize over the box-simplex and its optional TV expansion.

    With a positive radius the optimized distribution ``q`` may be within
    ``tv_radius`` of *any* distribution ``p`` in the sampling confidence set.
    Keeping ``q`` inside the sampling box would intersect the two uncertainty
    sets and incorrectly make a positive shift radius less conservative.
    """
    if box.tv_radius <= 0:
        return _greedy_box_simplex(coefficients, box, maximize)
    a = np.asarray(coefficients, dtype=float)
    n = len(a)
    # Variables are (q, p, t), where p belongs to the finite-sample box and
    # q is the shifted population.  TV(q, p) <= Delta iff an auxiliary
    # t >= |q-p| can satisfy sum(t) <= 2 Delta.
    objective = np.r_[-a if maximize else a, np.zeros(2 * n)]
    a_ub = []
    b_ub = []
    for idx in range(n):
        row = np.zeros(3 * n)
        row[idx] = 1
        row[n + idx] = -1
        row[2 * n + idx] = -1
        a_ub.append(row)
        b_ub.append(0.0)
        row = np.zeros(3 * n)
        row[idx] = -1
        row[n + idx] = 1
        row[2 * n + idx] = -1
        a_ub.append(row)
        b_ub.append(0.0)
    row = np.zeros(3 * n)
    row[2 * n :] = 1
    a_ub.append(row)
    b_ub.append(2 * box.tv_radius)
    a_eq = np.zeros((2, 3 * n))
    a_eq[0, :n] = 1
    a_eq[1, n : 2 * n] = 1
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=[1.0, 1.0],
        bounds=[(0.0, 1.0)] * n
        + list(zip(box.lower, box.upper, strict=True))
        + [(0.0, None)] * n,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"support oracle failed: {result.message}")
    point = result.x[:n]
    return float(a @ point), point
