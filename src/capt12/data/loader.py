from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from capt12.data.inspect import day_from_path, discover_parquet


def _keep_by_hash(values: pd.Series, fraction: float, seed: int) -> np.ndarray:
    threshold = int(fraction * (2**64 - 1))
    return np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(f"{seed}|{value}".encode(), digest_size=8).digest(), "little")
            <= threshold
            for value in values.astype(str)
        ),
        dtype=bool,
        count=len(values),
    )


def iter_parquet_batches(
    data_root: str | Path,
    *,
    columns: Sequence[str] | None = None,
    days: Sequence[int] | None = None,
    batch_size: int = 8192,
    max_rows: int | None = None,
    sample_frac: float = 1.0,
    seed: int = 0,
    id_col: str | None = None,
) -> Iterator[pd.DataFrame]:
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0,1]")
    selected_days = set(days) if days is not None else None
    emitted = 0
    for path in discover_parquet(data_root):
        day = day_from_path(path)
        if selected_days is not None and day not in selected_days:
            continue
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        requested = list(columns) if columns else list(parquet.schema_arrow.names)
        missing = set(requested) - available
        if missing:
            raise ValueError(f"columns missing from {path.name}: {sorted(missing)}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=requested):
            frame = batch.to_pandas()
            frame["day_int"] = day
            if sample_frac < 1:
                sampling_key = id_col if id_col and id_col in frame else frame.index.to_series().astype(str)
                frame = frame.loc[_keep_by_hash(frame[sampling_key] if isinstance(sampling_key, str) else sampling_key, sample_frac, seed)]
            if max_rows is not None:
                frame = frame.iloc[: max(0, max_rows - emitted)]
            if not frame.empty:
                emitted += len(frame)
                yield frame
            if max_rows is not None and emitted >= max_rows:
                return


def load_parquet_sample(**kwargs) -> pd.DataFrame:
    frames = list(iter_parquet_batches(**kwargs))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_temporal_splits(
    data_root: str | Path,
    splits: dict[str, Sequence[int]],
    *,
    columns: Sequence[str],
    max_rows_per_split: int | None = None,
    sample_frac: float = 1.0,
    seed: int = 0,
    id_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    assert_disjoint_splits(splits)
    return {
        name: load_parquet_sample(
            data_root=data_root,
            columns=columns,
            days=days,
            max_rows=max_rows_per_split,
            sample_frac=sample_frac,
            seed=seed,
            id_col=id_col,
        )
        for name, days in splits.items()
    }


def assert_disjoint_splits(splits: dict[str, Sequence[int]]) -> None:
    seen: dict[int, str] = {}
    for name, days in splits.items():
        for day in days:
            if day in seen:
                raise ValueError(f"day {day} overlaps between {seen[day]} and {name}")
            seen[day] = name


def assert_no_row_overlap(frames: dict[str, pd.DataFrame], id_col: str) -> None:
    seen: set[tuple[int, str]] = set()
    for split, frame in frames.items():
        keys = set(zip(frame["day_int"].astype(int), frame[id_col].astype(str), strict=True))
        overlap = seen.intersection(keys)
        if overlap:
            raise ValueError(f"row/day overlap detected in split {split}")
        seen.update(keys)

