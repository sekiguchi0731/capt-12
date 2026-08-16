from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from capt12.privacy.adjacency import AdjacentPair, Group, build_adjacency


@dataclass
class SyntheticPopulation:
    frame: pd.DataFrame
    distributions: dict[str, np.ndarray]
    groups: list[Group]
    adjacency: list[AdjacentPair]
    token_scores: np.ndarray


def generate_synthetic(
    n: int = 5000,
    k: int = 6,
    seed: int = 0,
    epsilon: float = 0.5,
) -> SyntheticPopulation:
    if k < 2:
        raise ValueError("synthetic generator requires K >= 2")
    rng = np.random.default_rng(seed)
    groups = [Group("proxy_a", (value,), context) for context in (0, 1) for value in (0, 1)]
    distributions: dict[str, np.ndarray] = {}
    base = np.linspace(1.0, 2.0, k)
    for group in groups:
        tilt = np.exp((2 * int(group.values[0]) - 1) * np.linspace(-0.65, 0.65, k))
        context_tilt = np.roll(base, int(group.context))
        probability = tilt * context_tilt
        distributions[group.key()] = probability / probability.sum()
    group_idx = rng.integers(0, len(groups), size=n)
    tokens = np.array([rng.choice(k, p=distributions[groups[idx].key()]) for idx in group_idx])
    contexts = np.array([groups[idx].context for idx in group_idx])
    attrs = np.array([groups[idx].values[0] for idx in group_idx])
    token_scores = np.clip(np.linspace(0.01, 0.2, k), 1e-6, 1 - 1e-6)
    probabilities = np.clip(token_scores[tokens] * (1 + 0.15 * contexts), 1e-6, 1 - 1e-6)
    labels = rng.binomial(1, probabilities)
    days = np.tile(np.arange(1, 31), int(np.ceil(n / 30)))[:n]
    frame = pd.DataFrame(
        {
            "id": [f"row-{idx}" for idx in range(n)],
            "user_id": [f"u-{idx // 2}" for idx in range(n)],
            "day_int": days,
            "proxy_a": attrs,
            "context": contexts,
            "token": tokens,
            "is_clicked": labels,
            "ref_probability": probabilities,
        }
    )
    adjacency = build_adjacency(groups, "tuple_adjacent", epsilon)
    return SyntheticPopulation(frame, distributions, groups, adjacency, token_scores)


def theorem4_counterexample() -> tuple[dict[str, np.ndarray], list[AdjacentPair]]:
    distributions = {"g0": np.array([0.3, 0.6, 0.1]), "g1": np.array([0.1, 0.8, 0.1])}
    adjacency = [AdjacentPair("g0", "g1", 0.0), AdjacentPair("g1", "g0", 0.0)]
    return distributions, adjacency

