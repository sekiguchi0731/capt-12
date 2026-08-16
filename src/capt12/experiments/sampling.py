from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

FULL_LABEL = "full"


def stable_hash64(values: Iterable[object], *, seed: int, namespace: str) -> np.ndarray:
    """Return deterministic 64-bit BLAKE2b hashes for stable nested sampling."""
    prefix = f"{namespace}|{seed}|".encode()
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(prefix + str(value).encode(), digest_size=8).digest(),
                "little",
            )
            for value in values
        ),
        dtype=np.uint64,
    )


def select_one_display_per_user_day(
    frame: pd.DataFrame,
    *,
    user_col: str,
    day_col: str,
    id_col: str,
) -> pd.DataFrame:
    """Select one display per user-day using a stable event hash.

    Most user-days contain one display. Hashing and grouping are restricted to
    duplicated epochs to avoid sorting the entire full certificate split.
    """
    epoch = frame[day_col].astype(str) + "|" + frame[user_col].astype(str)
    duplicated = epoch.duplicated(keep=False)
    keep = ~duplicated
    if duplicated.any():
        duplicate_index = np.flatnonzero(duplicated.to_numpy())
        duplicate_events = (
            epoch.iloc[duplicate_index] + "|" + frame.iloc[duplicate_index][id_col].astype(str)
        )
        hashes = stable_hash64(
            duplicate_events,
            seed=0,
            namespace="one-display-per-user-day",
        )
        duplicate_table = pd.DataFrame(
            {
                "position": duplicate_index,
                "epoch": epoch.iloc[duplicate_index].to_numpy(),
                "hash": hashes,
                "id": frame.iloc[duplicate_index][id_col].astype(str).to_numpy(),
            }
        )
        selected = (
            duplicate_table.sort_values(["epoch", "hash", "id"], kind="mergesort")
            .drop_duplicates("epoch", keep="first")["position"]
            .to_numpy(dtype=int)
        )
        keep.iloc[duplicate_index] = False
        keep.iloc[selected] = True
    result = frame.loc[keep].copy().reset_index(drop=True)
    if result.duplicated([user_col, day_col]).any():
        raise RuntimeError("contribution selection retained a duplicate user-day")
    return result


def stable_nested_positions(
    frame: pd.DataFrame,
    *,
    sizes: Sequence[int],
    seed: int,
    user_col: str,
    day_col: str,
) -> dict[str, np.ndarray]:
    """Rank privacy epochs once so requested prefixes are exactly nested."""
    epoch = frame[day_col].astype(str) + "|" + frame[user_col].astype(str)
    if epoch.duplicated().any():
        raise ValueError("stable nested sampling requires one row per user-day")
    hashes = stable_hash64(epoch, seed=seed, namespace="D_cert-user-day-rank")
    order = np.lexsort((epoch.to_numpy(dtype=str), hashes))
    result = {
        str(size): order[: min(int(size), len(order))].copy()
        for size in sorted(set(map(int, sizes)))
    }
    result[FULL_LABEL] = order.copy()
    return result
