from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.cluster import KMeans


def validate_partition(assignment: np.ndarray, k: int, l_count: int) -> np.ndarray:
    assignment = np.asarray(assignment, dtype=int)
    if assignment.shape != (k,) or np.any(assignment < 0) or np.any(assignment >= l_count):
        raise ValueError("partition must map each token to [L]")
    if set(assignment.tolist()) != set(range(l_count)):
        raise ValueError("partition blocks must all be non-empty")
    return assignment


def _balanced_from_order(order: np.ndarray, l_count: int) -> np.ndarray:
    chunks = np.array_split(np.asarray(order, dtype=int), l_count)
    assignment = np.empty(len(order), dtype=int)
    for block, members in enumerate(chunks):
        assignment[members] = block
    return assignment


def frequency_balanced(frequencies: np.ndarray, l_count: int, **_: object) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=float)
    order = np.argsort(-frequencies, kind="stable")
    block_mass = np.zeros(l_count)
    assignment = np.empty(len(frequencies), dtype=int)
    # seed one token per block, then greedily equalize empirical mass
    for idx, token in enumerate(order):
        block = idx if idx < l_count else int(np.argmin(block_mass))
        assignment[token] = block
        block_mass[block] += frequencies[token]
    return validate_partition(assignment, len(frequencies), l_count)


def score_quantile(frequencies: np.ndarray, l_count: int, *, scores: np.ndarray, **_: object) -> np.ndarray:
    del frequencies
    return validate_partition(_balanced_from_order(np.argsort(scores, kind="stable"), l_count), len(scores), l_count)


def score_kmeans(frequencies: np.ndarray, l_count: int, *, scores: np.ndarray, seed: int = 0, **_: object) -> np.ndarray:
    del frequencies
    labels = KMeans(n_clusters=l_count, random_state=seed, n_init=10).fit_predict(np.asarray(scores)[:, None])
    # reorder labels by score center for stable artifacts
    centers = np.array([np.mean(np.asarray(scores)[labels == label]) for label in range(l_count)])
    remap = np.empty(l_count, dtype=int)
    remap[np.argsort(centers)] = np.arange(l_count)
    return validate_partition(remap[labels], len(scores), l_count)


def risk_utility(
    frequencies: np.ndarray,
    l_count: int,
    *,
    scores: np.ndarray,
    group_distributions: np.ndarray,
    risk_lambda: float = 1.0,
    **_: object,
) -> np.ndarray:
    """Order tokens by normalized utility score + lambda * group-distribution risk."""
    score = np.asarray(scores, dtype=float)
    score_term = (score - score.min()) / max(np.ptp(score), 1e-12)
    groups = np.asarray(group_distributions, dtype=float)
    risk = groups.max(axis=0) - groups.min(axis=0)
    risk_term = (risk - risk.min()) / max(np.ptp(risk), 1e-12)
    order = np.argsort(score_term + risk_lambda * risk_term, kind="stable")
    del frequencies
    return validate_partition(_balanced_from_order(order, l_count), len(score), l_count)


def top_singleton_tail(frequencies: np.ndarray, l_count: int, **_: object) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=float)
    order = np.argsort(-frequencies, kind="stable")
    assignment = np.full(len(frequencies), l_count - 1, dtype=int)
    for block, token in enumerate(order[: l_count - 1]):
        assignment[token] = block
    return validate_partition(assignment, len(frequencies), l_count)


def random_balanced(frequencies: np.ndarray, l_count: int, *, seed: int = 0, **_: object) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(len(frequencies))
    return validate_partition(_balanced_from_order(order, l_count), len(frequencies), l_count)


PARTITION_REGISTRY: dict[str, Callable] = {
    "frequency_balanced": frequency_balanced,
    "score_quantile": score_quantile,
    "score_kmeans": score_kmeans,
    "risk_utility": risk_utility,
    "top_singleton_tail": top_singleton_tail,
    "random_balanced": random_balanced,
}


def build_partition(name: str, frequencies: np.ndarray, l_count: int, **kwargs) -> np.ndarray:
    if not 1 <= l_count <= len(frequencies):
        raise ValueError("partition requires 1 <= L <= K")
    split_id = kwargs.pop("split_id", "D_design")
    if split_id != "D_design":
        raise ValueError("partitions may only be fit on D_design")
    return PARTITION_REGISTRY[name](frequencies, l_count, **kwargs)


def build_nested_partitions(order: np.ndarray, l_values: list[int]) -> dict[int, np.ndarray]:
    """Contiguous refinements for a divisibility chain of L values."""
    values = sorted(set(l_values))
    if any(b % a != 0 for a, b in zip(values, values[1:], strict=False)):
        raise ValueError("nested L values must form a divisibility chain")
    return {l_count: _balanced_from_order(np.asarray(order), l_count) for l_count in values}
