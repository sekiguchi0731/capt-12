from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier


def validate_tokens(tokens: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(tokens)
    if not np.issubdtype(values.dtype, np.integer):
        if np.any(values != np.floor(values)):
            raise ValueError("tokens must be integers")
        values = values.astype(int)
    if np.any(values < 0) or np.any(values >= k):
        raise ValueError(f"encoder output must satisfy 0 <= Z < K={k}")
    if k > 4096:
        raise ValueError("K must be at most 4096")
    return values.astype(int)


@dataclass
class Encoder:
    name: str
    k: int
    source_columns: tuple[str, ...] = ()
    fit_split: str | None = None
    sensitive_policy: str = "exclude"

    def _check_fit(
        self,
        source_columns: Sequence[str],
        split_id: str,
        sensitive_columns: Sequence[str],
    ) -> None:
        if split_id != "D_model":
            raise ValueError("Phi_0 may only be fit on D_model")
        if self.k > 4096:
            raise ValueError("K must be at most 4096")
        if self.sensitive_policy == "exclude" and set(source_columns).intersection(sensitive_columns):
            raise ValueError("sensitive proxy columns are excluded from the default encoder")
        if self.sensitive_policy not in {"exclude", "include_stress_test"}:
            raise ValueError("phi_sensitive_policy must be exclude or include_stress_test")


class PrecomputedEncoder(Encoder):
    column: str

    def __init__(self, k: int, column: str):
        super().__init__("precomputed", k, (column,), "external")
        self.column = column

    def fit(self, frame: pd.DataFrame, **_: object) -> PrecomputedEncoder:
        validate_tokens(frame[self.column].to_numpy(), self.k)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return validate_tokens(frame[self.column].to_numpy(), self.k)


class HashEncoder(Encoder):
    def __init__(self, k: int, seed: int = 0, sensitive_policy: str = "exclude"):
        super().__init__("hash", k, sensitive_policy=sensitive_policy)
        self.seed = seed

    def fit(
        self,
        frame: pd.DataFrame,
        source_columns: Sequence[str],
        *,
        split_id: str = "D_model",
        sensitive_columns: Sequence[str] = (),
        **_: object,
    ) -> HashEncoder:
        self._check_fit(source_columns, split_id, sensitive_columns)
        self.source_columns = tuple(source_columns)
        self.fit_split = split_id
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.source_columns:
            raise RuntimeError("hash encoder is not fitted")
        rows = frame[list(self.source_columns)].apply(
            lambda row: "\x1f".join(map(str, row.tolist())), axis=1
        )
        values = np.fromiter(
            (
                int.from_bytes(
                    hashlib.blake2b(f"{self.seed}|{row}".encode(), digest_size=8).digest(),
                    "little",
                )
                % self.k
                for row in rows
            ),
            dtype=int,
            count=len(rows),
        )
        return validate_tokens(values, self.k)


class ScoreEncoder(Encoder):
    def __init__(self, name: str, k: int, sensitive_policy: str = "exclude", seed: int = 0):
        super().__init__(name, k, sensitive_policy=sensitive_policy)
        self.seed = seed
        self.edges: np.ndarray | None = None
        self.centers: np.ndarray | None = None

    def fit_scores(self, scores: np.ndarray, *, split_id: str = "D_model") -> ScoreEncoder:
        if split_id != "D_model":
            raise ValueError("score encoder may only be fit on D_model")
        scores = np.asarray(scores, dtype=float)
        if self.name == "ctr_quantile":
            self.edges = np.unique(np.quantile(scores, np.linspace(0, 1, self.k + 1)))[1:-1]
        elif self.name == "ctr_kmeans":
            clusters = max(1, min(self.k, len(scores), len(np.unique(scores))))
            model = KMeans(n_clusters=clusters, random_state=self.seed, n_init=10).fit(
                scores[:, None]
            )
            self.centers = np.sort(model.cluster_centers_[:, 0])
        else:
            raise ValueError(self.name)
        self.fit_split = split_id
        return self

    def transform_scores(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.edges is not None:
            return validate_tokens(np.searchsorted(self.edges, scores, side="right"), self.k)
        if self.centers is not None:
            return validate_tokens(np.argmin(abs(scores[:, None] - self.centers[None, :]), axis=1), self.k)
        raise RuntimeError("score encoder is not fitted")


class TreeLeafHashEncoder(HashEncoder):
    def __init__(self, k: int, seed: int = 0, sensitive_policy: str = "exclude"):
        super().__init__(k, seed, sensitive_policy)
        self.name = "tree_leaf_hash"
        self.tree: DecisionTreeClassifier | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        source_columns: Sequence[str],
        *,
        labels: np.ndarray,
        split_id: str = "D_model",
        sensitive_columns: Sequence[str] = (),
        **_: object,
    ) -> TreeLeafHashEncoder:
        self._check_fit(source_columns, split_id, sensitive_columns)
        self.source_columns = tuple(source_columns)
        values = frame[list(source_columns)].select_dtypes(include=[np.number]).fillna(0)
        self.source_columns = tuple(values.columns)
        self.tree = DecisionTreeClassifier(max_leaf_nodes=min(self.k * 2, 512), random_state=self.seed)
        self.tree.fit(values, labels)
        self.fit_split = split_id
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("tree encoder is not fitted")
        leaves = self.tree.apply(frame[list(self.source_columns)].fillna(0))
        return np.asarray([int(v) % self.k for v in leaves], dtype=int)


def make_encoder(name: str, k: int, **kwargs) -> Encoder:
    policy = kwargs.get("sensitive_policy", "exclude")
    seed = kwargs.get("seed", 0)
    if name == "precomputed":
        return PrecomputedEncoder(k, kwargs["column"])
    if name == "hash":
        return HashEncoder(k, seed, policy)
    if name in {"ctr_quantile", "ctr_kmeans"}:
        return ScoreEncoder(name, k, policy, seed)
    if name == "tree_leaf_hash":
        return TreeLeafHashEncoder(k, seed, policy)
    raise KeyError(f"unknown encoder: {name}")


ENCODER_REGISTRY = {
    "precomputed": PrecomputedEncoder,
    "hash": HashEncoder,
    "ctr_quantile": ScoreEncoder,
    "ctr_kmeans": ScoreEncoder,
    "tree_leaf_hash": TreeLeafHashEncoder,
}
