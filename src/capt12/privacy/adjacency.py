from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations


@dataclass(frozen=True)
class Group:
    profile: str
    values: tuple[Hashable, ...]
    context: Hashable = "all"

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(self.profile.split("+")) if self.profile else ()

    def key(self) -> str:
        return f"{self.profile}|{self.context}|" + "|".join(map(str, self.values))


@dataclass(frozen=True)
class AdjacentPair:
    left: str
    right: str
    epsilon: float
    changed_attributes: tuple[str, ...] = ()


def hybrid_epsilon(
    left: Sequence[Hashable],
    right: Sequence[Hashable],
    attributes: Sequence[str],
    epsilon_by_attr: Mapping[str, float],
    default_epsilon: float = 0.0,
) -> float:
    if not (len(left) == len(right) == len(attributes)):
        raise ValueError("tuples and attributes must have equal lengths")
    return sum(
        float(epsilon_by_attr.get(attr, default_epsilon))
        for a, b, attr in zip(left, right, attributes, strict=True)
        if a != b
    )


def build_adjacency(
    groups: Sequence[Group],
    mode: str = "tuple_adjacent",
    epsilon: float = 1.0,
    epsilon_by_attr: Mapping[str, float] | None = None,
) -> list[AdjacentPair]:
    epsilon_by_attr = epsilon_by_attr or {}
    pairs: list[AdjacentPair] = []
    for left, right in permutations(groups, 2):
        if left.profile != right.profile or left.context != right.context:
            continue
        attrs = left.attributes
        changed = tuple(
            attr
            for a, b, attr in zip(left.values, right.values, attrs, strict=True)
            if a != b
        )
        if not changed:
            continue
        if mode == "tuple_adjacent" and len(changed) != 1:
            continue
        if mode == "marginal" and len(changed) != 1:
            continue
        if mode not in {"tuple_adjacent", "marginal", "joint_all_pairs"}:
            raise ValueError(f"unknown adjacency mode: {mode}")
        eps = sum(float(epsilon_by_attr.get(attr, epsilon)) for attr in changed)
        pairs.append(AdjacentPair(left.key(), right.key(), eps, changed))
    return pairs


def disconnected_hybrid_components(
    groups: Sequence[Group], adjacency: Sequence[AdjacentPair]
) -> int:
    """Count excess connected components within each profile/context graph."""
    keys_by_scope: dict[tuple[str, Hashable], set[str]] = {}
    for group in groups:
        keys_by_scope.setdefault((group.profile, group.context), set()).add(group.key())
    neighbors: dict[str, set[str]] = {group.key(): set() for group in groups}
    for pair in adjacency:
        neighbors.setdefault(pair.left, set()).add(pair.right)
        neighbors.setdefault(pair.right, set()).add(pair.left)
    excess = 0
    for keys in keys_by_scope.values():
        remaining = set(keys)
        components = 0
        while remaining:
            components += 1
            frontier = [remaining.pop()]
            while frontier:
                current = frontier.pop()
                reached = neighbors.get(current, set()).intersection(remaining)
                remaining.difference_update(reached)
                frontier.extend(reached)
        excess += max(0, components - 1)
    return excess
