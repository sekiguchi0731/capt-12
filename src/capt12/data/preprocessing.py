from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from capt12.privacy.adjacency import Group


@dataclass
class FrozenCategoryMapper:
    max_cardinality: int = 32
    mappings: dict[str, set[object]] = field(default_factory=dict)
    fit_split: str | None = None

    def fit(
        self, frame: pd.DataFrame, columns: Sequence[str], *, split_id: str = "D_model"
    ) -> FrozenCategoryMapper:
        if split_id not in {"D_model", "D_design"}:
            raise ValueError("category mappings may only be fit on D_model or D_design")
        self.mappings = {
            column: set(frame[column].value_counts(dropna=False).head(self.max_cardinality - 1).index)
            for column in columns
        }
        self.fit_split = split_id
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column, allowed in self.mappings.items():
            result[column] = result[column].where(result[column].isin(allowed), "__OTHER__")
            result[column] = result[column].fillna("__MISSING__").astype(str)
        return result


@dataclass
class GroupHistograms:
    counts: dict[str, np.ndarray]
    groups: list[Group]
    force_cover: bool
    rare_group_count: int
    rare_group_mass: float
    rare_row_indices: frozenset[object] = frozenset()
    expected_group_count: int = 0
    missing_group_count: int = 0
    missing_groups: tuple[str, ...] = ()


def build_group_histograms(
    frame: pd.DataFrame,
    tokens: np.ndarray,
    *,
    profile: str,
    context_columns: Sequence[str] = (),
    alphabet_size: int,
    token_to_block: np.ndarray | None = None,
    min_group_count: int = 1,
    rare_group_policy: str = "force_cover",
    missing_group_policy: str | None = None,
    expected_frame: pd.DataFrame | None = None,
    include_fallback_levels: bool = False,
) -> GroupHistograms:
    if missing_group_policy is None:
        # Preserve the legacy API in which rare_group_policy governed both
        # observed rare groups and unobserved expected groups.
        missing_group_policy = rare_group_policy
    attributes = profile.split("+") if profile else []
    missing = set(attributes).union(context_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"profile/context columns are missing: {sorted(missing)}")
    values = np.asarray(tokens, dtype=int)
    if token_to_block is not None:
        values = np.asarray(token_to_block, dtype=int)[values]
    if np.any(values < 0) or np.any(values >= alphabet_size):
        raise ValueError("histogram token is outside its alphabet")
    work = frame[list(attributes) + list(context_columns)].copy()
    work["__value__"] = values
    context_key = (
        work[list(context_columns)].astype(str).agg("|".join, axis=1)
        if context_columns
        else pd.Series("all", index=work.index)
    )
    work["__context__"] = context_key
    group_cols = list(attributes) + ["__context__"]
    grouped = work.groupby(group_cols, dropna=False, sort=True)
    rows: list[tuple[Group, np.ndarray, int, frozenset[object]]] = []
    for key, subset in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        attr_values = tuple(key_tuple[: len(attributes)])
        context = key_tuple[-1]
        counts = np.bincount(subset["__value__"].to_numpy(), minlength=alphabet_size)
        rows.append(
            (
                Group(profile, attr_values, context),
                counts,
                len(subset),
                frozenset(subset.index.tolist()),
            )
        )
    observed_tuples = {
        (*group.values, group.context): group for group, _, _, _ in rows
    }
    expected = (
        expected_tuple_grid(
            expected_frame,
            profile,
            context_columns,
            include_fallback_levels=include_fallback_levels,
        )
        if expected_frame is not None
        else set(observed_tuples)
    )
    missing_tuples = sorted(expected - set(observed_tuples), key=lambda item: tuple(map(str, item)))
    missing_groups = tuple(
        Group(profile, tuple(item[: len(attributes)]), item[-1]).key()
        for item in missing_tuples
    )
    if missing_groups and missing_group_policy == "fail":
        raise ValueError(
            f"{len(missing_groups)} expected groups are unobserved in the certificate split"
        )
    if missing_groups and missing_group_policy == "merge_to_other":
        raise ValueError(
            "unobserved expected groups cannot be estimated by merge_to_other; "
            "use fail or force_cover"
        )
    rare = [row for row in rows if row[2] < min_group_count]
    rare_row_indices = frozenset(
        index for _, _, _, indices in rare for index in indices
    )
    if missing_groups and missing_group_policy == "full_simplex":
        rows.extend(
            (
                Group(profile, tuple(item[: len(attributes)]), item[-1]),
                np.zeros(alphabet_size, dtype=int),
                0,
                frozenset(),
            )
            for item in missing_tuples
        )
    if rare and rare_group_policy == "fail":
        raise ValueError(f"{len(rare)} groups have fewer than min_group_count={min_group_count}")
    force_cover = bool(
        (rare and rare_group_policy == "force_cover")
        or (missing_groups and missing_group_policy == "force_cover")
    )
    if rare and rare_group_policy == "merge_to_other":
        common = [row for row in rows if row[2] >= min_group_count]
        merged: dict[str, tuple[Group, np.ndarray, int, frozenset[object]]] = {}
        for group, counts, size, indices in rare:
            other = Group(profile, tuple("__OTHER__" for _ in attributes), group.context)
            if other.key() not in merged:
                merged[other.key()] = (other, counts.copy(), size, indices)
            else:
                old_group, old_counts, old_size, old_indices = merged[other.key()]
                merged[other.key()] = (
                    old_group,
                    old_counts + counts,
                    old_size + size,
                    old_indices.union(indices),
                )
        rows = common + list(merged.values())
    total = max(len(frame), 1)
    return GroupHistograms(
        counts={group.key(): counts for group, counts, _, _ in rows},
        groups=[group for group, _, _, _ in rows],
        force_cover=force_cover,
        rare_group_count=len(rare),
        rare_group_mass=len(rare_row_indices) / total,
        rare_row_indices=rare_row_indices,
        expected_group_count=len(expected.union(observed_tuples)),
        missing_group_count=len(missing_groups),
        missing_groups=missing_groups,
    )


def expected_tuple_grid(
    frame: pd.DataFrame,
    profile: str,
    context_columns: Sequence[str],
    *,
    include_fallback_levels: bool = False,
) -> set[tuple]:
    attributes = profile.split("+") if profile else []
    def domain(column: str) -> list[object]:
        values = frame[column].drop_duplicates().tolist()
        if include_fallback_levels:
            values.extend(
                value for value in ("__OTHER__", "__MISSING__") if value not in values
            )
        return values

    attribute_domains = [domain(column) for column in attributes]
    if context_columns:
        context_domains = [domain(column) for column in context_columns]
        contexts = ["|".join(map(str, values)) for values in product(*context_domains)]
    else:
        contexts = ["all"]
    domains = [*attribute_domains, contexts]
    return set(product(*domains))


def build_marginal_histograms(
    frame: pd.DataFrame,
    tokens: np.ndarray,
    *,
    profile: str,
    context_columns: Sequence[str] = (),
    alphabet_size: int,
    token_to_block: np.ndarray | None = None,
    min_group_count: int = 1,
    rare_group_policy: str = "force_cover",
    missing_group_policy: str | None = None,
    expected_frame: pd.DataFrame | None = None,
    include_fallback_levels: bool = False,
) -> GroupHistograms:
    """Marginal-ablation histograms, one independent attribute at a time."""
    parts = [
        build_group_histograms(
            frame,
            tokens,
            profile=attribute,
            context_columns=context_columns,
            alphabet_size=alphabet_size,
            token_to_block=token_to_block,
            min_group_count=min_group_count,
            rare_group_policy=rare_group_policy,
            missing_group_policy=missing_group_policy,
            expected_frame=expected_frame,
            include_fallback_levels=include_fallback_levels,
        )
        for attribute in profile.split("+")
    ]
    return GroupHistograms(
        counts={key: value for part in parts for key, value in part.counts.items()},
        groups=[group for part in parts for group in part.groups],
        force_cover=any(part.force_cover for part in parts),
        rare_group_count=sum(part.rare_group_count for part in parts),
        rare_group_mass=(
            len(set().union(*(part.rare_row_indices for part in parts)))
            / max(len(frame), 1)
        ),
        rare_row_indices=frozenset(
            set().union(*(part.rare_row_indices for part in parts))
        ),
        expected_group_count=sum(part.expected_group_count for part in parts),
        missing_group_count=sum(part.missing_group_count for part in parts),
        missing_groups=tuple(group for part in parts for group in part.missing_groups),
    )
