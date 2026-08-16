from __future__ import annotations

import numpy as np
import pandas as pd


def apply_contribution_policy(
    frame: pd.DataFrame,
    *,
    user_col: str,
    day_col: str,
    policy: str = "one-display-per-uuid-day",
    cap: int = 1,
    seed: int = 0,
) -> pd.DataFrame:
    """Bound histogram contributions without modifying the source frame."""
    if policy == "per-row":
        return frame.copy()
    if policy not in {"one-display-per-uuid-day", "capped-c"}:
        raise ValueError(f"unknown contribution policy: {policy}")
    limit = 1 if policy == "one-display-per-uuid-day" else int(cap)
    if limit < 1:
        raise ValueError("cap must be at least one")
    shuffled = frame.sample(frac=1.0, random_state=seed)
    selected = shuffled.groupby([user_col, day_col], sort=False, dropna=False).head(limit)
    return selected.sort_index().copy()


def dp_histogram(
    categories: np.ndarray,
    size: int,
    *,
    epsilon: float,
    rng: np.random.Generator,
    sensitivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the one-shot central-DP histogram seen after secure aggregation."""
    if epsilon <= 0 or sensitivity <= 0:
        raise ValueError("epsilon and sensitivity must be positive")
    true_counts = np.bincount(np.asarray(categories, dtype=int), minlength=size).astype(float)
    noisy = true_counts + rng.laplace(0.0, sensitivity / epsilon, size=size)
    return true_counts, noisy

