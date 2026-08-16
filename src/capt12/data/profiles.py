from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd


def assign_simulated_profiles(
    frame: pd.DataFrame,
    *,
    profiles: Sequence[str],
    user_col: str,
    epoch_col: str,
    probabilities: Sequence[float] | None = None,
    seed: int = 0,
) -> pd.Series:
    """Assign one synthetic S per user/privacy epoch, deterministically by seed."""
    if not profiles or any(not profile for profile in profiles):
        raise ValueError("simulated profiles must be non-empty")
    probability = (
        np.ones(len(profiles)) / len(profiles)
        if probabilities is None
        else np.asarray(probabilities, dtype=float)
    )
    if probability.shape != (len(profiles),) or np.any(probability < 0) or probability.sum() <= 0:
        raise ValueError("profile probabilities must be nonnegative and match profiles")
    probability = probability / probability.sum()
    cumulative = np.cumsum(probability)

    def choose(user: object, epoch: object) -> str:
        digest = hashlib.blake2b(f"{seed}|{user}|{epoch}".encode(), digest_size=8).digest()
        uniform = int.from_bytes(digest, "little") / 2**64
        return profiles[min(int(np.searchsorted(cumulative, uniform, side="right")), len(profiles) - 1)]

    return pd.Series(
        [
            choose(user, epoch)
            for user, epoch in zip(frame[user_col], frame[epoch_col], strict=True)
        ],
        index=frame.index,
        name="simulated_profile",
    )

