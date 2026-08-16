from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from capt12.privacy.adjacency import AdjacentPair


@dataclass
class EnvelopeResult:
    retention: np.ndarray
    utility: float
    envelopes: dict[str, np.ndarray]
    status: str


def build_envelopes(
    distributions: Mapping[str, np.ndarray], adjacency: Sequence[AdjacentPair]
) -> dict[str, np.ndarray]:
    result = {key: np.asarray(value, dtype=float).copy() for key, value in distributions.items()}
    for pair in adjacency:
        if pair.right in result:
            result[pair.right] = np.maximum(
                result[pair.right], np.exp(-pair.epsilon) * np.asarray(distributions[pair.left])
            )
    return result


def theorem4_envelope(
    distributions: Mapping[str, np.ndarray],
    adjacency: Sequence[AdjacentPair],
    retention_weights: np.ndarray | None = None,
) -> EnvelopeResult:
    envelopes = build_envelopes(distributions, adjacency)
    k = len(next(iter(envelopes.values())))
    weights = np.ones(k) / k if retention_weights is None else np.asarray(retention_weights, dtype=float)
    weights = weights / weights.sum()
    result = linprog(
        -weights,
        A_ub=np.vstack(list(envelopes.values())),
        b_ub=np.ones(len(envelopes)),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"envelope LP failed: {result.message}")
    return EnvelopeResult(result.x, float(weights @ result.x), envelopes, "optimal")


def fractional_knapsack_envelope(envelope: np.ndarray, weights: np.ndarray) -> EnvelopeResult:
    """Closed form for a single envelope constraint and box 0 <= r <= 1."""
    m = np.asarray(envelope, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    ratio = np.divide(w, m, out=np.full_like(w, np.inf), where=m > 0)
    order = np.argsort(-ratio, kind="stable")
    remaining = 1.0
    retention = np.zeros_like(w)
    for idx in order:
        if m[idx] == 0:
            retention[idx] = 1.0
            continue
        take = min(1.0, remaining / m[idx])
        retention[idx] = take
        remaining -= take * m[idx]
        if remaining <= 1e-15:
            break
    return EnvelopeResult(retention, float(w @ retention), {"single": m}, "closed_form")

