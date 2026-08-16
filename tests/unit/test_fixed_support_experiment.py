from __future__ import annotations

import numpy as np
import pandas as pd

from capt12.experiments.fixed_support import cartesian_support_from_domains
from capt12.experiments.sampling import (
    select_one_display_per_user_day,
    stable_hash64,
    stable_nested_positions,
)


def test_stable_hash_is_deterministic_and_seeded() -> None:
    values = ["a", "b", "c"]
    first = stable_hash64(values, seed=3, namespace="test")
    second = stable_hash64(values, seed=3, namespace="test")
    other = stable_hash64(values, seed=4, namespace="test")
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_one_display_selection_is_stable_per_user_day() -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "user": ["u1", "u1", "u2", "u2"],
            "day": [1, 1, 1, 2],
            "value": [1, 2, 3, 4],
        }
    )
    first = select_one_display_per_user_day(frame, user_col="user", day_col="day", id_col="id")
    second = select_one_display_per_user_day(
        frame.sample(frac=1, random_state=9).reset_index(drop=True),
        user_col="user",
        day_col="day",
        id_col="id",
    )
    assert len(first) == 3
    assert not first.duplicated(["user", "day"]).any()
    assert set(first["id"]) == set(second["id"])


def test_nested_positions_are_exact_prefixes() -> None:
    frame = pd.DataFrame(
        {
            "user": [f"u{value}" for value in range(20)],
            "day": np.repeat([1, 2], 10),
        }
    )
    positions = stable_nested_positions(
        frame,
        sizes=[5, 10],
        seed=7,
        user_col="user",
        day_col="day",
    )
    assert len(positions["5"]) == 5
    assert len(positions["10"]) == 10
    assert len(positions["full"]) == 20
    assert set(positions["5"]).issubset(positions["10"])
    assert set(positions["10"]).issubset(positions["full"])
    other_seed = stable_nested_positions(
        frame,
        sizes=[5],
        seed=8,
        user_col="user",
        day_col="day",
    )
    assert set(positions["5"]) != set(other_seed["5"])


def test_cartesian_support_does_not_add_padding_nan_level() -> None:
    support = cartesian_support_from_domains(
        {
            "a": {"a0", "a1", "a2"},
            "b": {"b0", "b1"},
        },
        profile="a",
        contexts=["b"],
    )
    assert len(support) == 5 * 4
    assert all("nan" not in value for group in support for value in group)
