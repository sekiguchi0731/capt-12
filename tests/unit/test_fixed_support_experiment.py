from __future__ import annotations

import numpy as np
import pandas as pd

from capt12.data.preprocessing import FrozenCategoryMapper
from capt12.experiments.fixed_support import (
    _context_aggregated_representation_cost,
    cartesian_support_from_domains,
    ordered_size_labels,
)
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


def test_unified_sensitive_unknown_is_frozen_and_used_in_support() -> None:
    model = pd.DataFrame({"secret": ["a", "b"], "context": ["x", "y"]})
    mapper = FrozenCategoryMapper(
        3,
        unknown_columns=frozenset({"secret"}),
    ).fit(model, ["secret", "context"])
    transformed = mapper.transform(
        pd.DataFrame(
            {
                "secret": ["a", None, "unseen"],
                "context": ["x", None, "unseen"],
            }
        )
    )
    assert transformed["secret"].tolist() == ["a", "__UNKNOWN__", "__UNKNOWN__"]
    assert transformed["context"].tolist() == ["x", "__OTHER__", "__OTHER__"]
    support = cartesian_support_from_domains(
        {"secret": {"a", "b"}, "context": {"x", "y"}},
        profile="secret",
        contexts=["context"],
        fallback_levels_by_column={
            "secret": ["__UNKNOWN__"],
            "context": ["__OTHER__", "__MISSING__"],
        },
    )
    assert any(group[0] == "__UNKNOWN__" for group in support)
    assert all(group[0] not in {"__OTHER__", "__MISSING__"} for group in support)


def test_plot_size_labels_match_cell_labels() -> None:
    results = pd.DataFrame(
        {
            "cert_target_user_days": [5000.0, 10000.0, np.nan],
            "cert_size_label": ["5000", "10000", "full"],
        }
    )
    labels = ordered_size_labels(results)
    assert labels == ["5000", "10000", "full"]
    assert results["cert_size_label"].isin(labels).all()


def test_objective_aligned_representation_uses_same_context_estimand() -> None:
    probability_grid = np.array([[0.2, 0.3], [0.8, 0.7]])
    counts = np.array([[3.0, 1.0], [1.0, 3.0]])
    successes = np.array([[0.0, 1.0], [1.0, 2.0]])
    cost, floor = _context_aggregated_representation_cost(
        probability_grid,
        counts,
        successes,
        objective="empirical_logloss",
        eta=1e-6,
        hybrid_empirical_weight=0.5,
    )

    rates = successes / counts
    expected_context_costs = []
    expected_context_floors = []
    for context in range(2):
        output = probability_grid[:, context]
        rate = rates[:, context]
        expected_context_costs.append(
            rate[:, None] * -np.log(output[None, :])
            + (1 - rate[:, None]) * -np.log(1 - output[None, :])
        )
        expected_context_floors.append(
            -(rate * np.log(np.where(rate > 0, rate, 1.0)))
            - (1 - rate) * np.log(np.where(rate < 1, 1 - rate, 1.0))
        )
    context_weights = counts / counts.sum(axis=1, keepdims=True)
    expected_cost = sum(
        context_weights[:, context, None] * expected_context_costs[context] for context in range(2)
    )
    expected_floor = sum(
        context_weights[:, context] * expected_context_floors[context] for context in range(2)
    )
    np.testing.assert_allclose(cost, expected_cost)
    np.testing.assert_allclose(floor, expected_floor)
