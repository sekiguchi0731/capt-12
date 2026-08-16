from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _clip(value: np.ndarray | float, eta: float) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), eta, 1 - eta)


def bernoulli_kl(p, q, eta: float = 1e-6):
    p = _clip(p, eta)
    q = _clip(q, eta)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def symmetric_kl(p, q, eta: float = 1e-6):
    return 0.5 * (bernoulli_kl(p, q, eta) + bernoulli_kl(q, p, eta))


def jensen_shannon(p, q, eta: float = 1e-6):
    p = _clip(p, eta)
    q = _clip(q, eta)
    midpoint = 0.5 * (p + q)
    return 0.5 * (bernoulli_kl(p, midpoint, eta) + bernoulli_kl(q, midpoint, eta))


def prob_l2(p, q, eta: float = 1e-6):
    del eta
    return (np.asarray(p) - np.asarray(q)) ** 2


def abs_prob(p, q, eta: float = 1e-6):
    del eta
    return np.abs(np.asarray(p) - np.asarray(q))


def logit_l2(p, q, eta: float = 1e-6):
    p = _clip(p, eta)
    q = _clip(q, eta)
    return (np.log(p / (1 - p)) - np.log(q / (1 - q))) ** 2


def retention(p, q, eta: float = 1e-6):
    del eta
    return (np.asarray(p) != np.asarray(q)).astype(float)


DISTORTION_REGISTRY: dict[str, Callable] = {
    "bernoulli_kl": bernoulli_kl,
    "symmetric_kl": symmetric_kl,
    "jensen_shannon": jensen_shannon,
    "brier": prob_l2,
    "prob_l2": prob_l2,
    "abs_prob": abs_prob,
    "logit_l2": logit_l2,
    "retention": retention,
}


def token_cost_matrix(
    token_probabilities_by_context: np.ndarray,
    context_weights_by_token: np.ndarray,
    name: str = "bernoulli_kl",
    eta: float = 1e-6,
) -> np.ndarray:
    """Compute c_S(z,o)=E[d(z,o,B)|Z=z,S] from frozen design inputs."""
    probabilities = np.asarray(token_probabilities_by_context, dtype=float)
    weights = np.asarray(context_weights_by_token, dtype=float)
    if probabilities.ndim != 2 or weights.shape != probabilities.shape:
        raise ValueError("probability and context-weight tables must have shape K x |B|")
    weights = np.divide(weights, weights.sum(axis=1, keepdims=True), out=np.zeros_like(weights), where=weights.sum(axis=1, keepdims=True) > 0)
    function = DISTORTION_REGISTRY[name]
    k, context_count = probabilities.shape
    result = np.empty((k, k), dtype=float)
    for z in range(k):
        for o in range(k):
            if name == "retention":
                result[z, o] = float(z != o)
            else:
                result[z, o] = float(np.sum(weights[z] * function(probabilities[z], probabilities[o], eta)))
    return result


def block_cost_matrix(
    token_cost: np.ndarray,
    token_to_block: np.ndarray,
    decoder: np.ndarray,
    token_weights: np.ndarray,
) -> np.ndarray:
    token_cost = np.asarray(token_cost, dtype=float)
    assignment = np.asarray(token_to_block, dtype=int)
    decoder = np.asarray(decoder, dtype=float)
    weights = np.asarray(token_weights, dtype=float)
    l_count = decoder.shape[0]
    result = np.zeros((l_count, l_count))
    for source_block in range(l_count):
        members = np.flatnonzero(assignment == source_block)
        conditional = weights[members]
        conditional = conditional / conditional.sum() if conditional.sum() else np.ones(len(members)) / len(members)
        for destination_block in range(l_count):
            expected_by_source = token_cost[members] @ decoder[destination_block]
            result[source_block, destination_block] = conditional @ expected_by_source
    return result

