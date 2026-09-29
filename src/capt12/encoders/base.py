from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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


class CtrLogisticQuantileEncoder(Encoder):
    """Supervised CTR score followed by a frozen K-bin quantile code.

    CriteoPrivateAd stores hashed categorical values as integer columns and
    monotone-transformed continuous values as floating-point columns.  The
    encoder therefore treats integer/string fields as nominal, standardizes
    floating-point fields, learns a sparse logistic CTR score on ``D_model``,
    and quantizes that score into approximately equiprobable tokens.  The
    downstream ``f_ref`` remains a separately fitted calibration model on the
    categorical token and public context.
    """

    def __init__(
        self,
        k: int,
        seed: int = 0,
        sensitive_policy: str = "exclude",
        *,
        max_categories: int = 256,
        alpha: float = 1e-6,
        epochs: int = 30,
    ):
        super().__init__("ctr_logistic_quantile", k, sensitive_policy=sensitive_policy)
        if max_categories < 2:
            raise ValueError("encoder_max_categories must be at least 2")
        if alpha <= 0:
            raise ValueError("encoder_sgd_alpha must be positive")
        if epochs < 1:
            raise ValueError("encoder_sgd_epochs must be positive")
        self.seed = seed
        self.max_categories = max_categories
        self.alpha = alpha
        self.epochs = epochs
        self.categorical_columns: tuple[str, ...] = ()
        self.numeric_columns: tuple[str, ...] = ()
        self.model: Any | None = None
        self.edges: np.ndarray | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        source_columns: Sequence[str],
        *,
        labels: np.ndarray,
        split_id: str = "D_model",
        sensitive_columns: Sequence[str] = (),
        **_: object,
    ) -> CtrLogisticQuantileEncoder:
        self._check_fit(source_columns, split_id, sensitive_columns)
        requested = list(source_columns)
        missing = set(requested) - set(frame.columns)
        if missing:
            raise ValueError(f"encoder source columns are missing: {sorted(missing)}")
        active = [column for column in requested if frame[column].notna().any()]
        if not active:
            raise ValueError("CTR encoder requires at least one nonempty source column")
        categorical = [
            column
            for column in active
            if pd.api.types.is_integer_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
            or not pd.api.types.is_numeric_dtype(frame[column])
            or column.endswith("_id")
            or column == "display_order"
        ]
        numeric = [column for column in active if column not in categorical]
        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    numeric,
                )
            )
        if categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="infrequent_if_exist",
                                    max_categories=self.max_categories,
                                ),
                            ),
                        ]
                    ),
                    categorical,
                )
            )
        self.model = Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(transformers, sparse_threshold=1.0),
                ),
                (
                    "logistic",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=self.alpha,
                        max_iter=self.epochs,
                        tol=None,
                        average=True,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        y = np.asarray(labels, dtype=int)
        if len(y) != len(frame):
            raise ValueError("encoder labels must have one value per D_model row")
        if len(np.unique(y)) != 2:
            raise ValueError("CTR encoder requires both binary label classes")
        self.model.fit(frame[active], y)
        self.source_columns = tuple(active)
        self.categorical_columns = tuple(categorical)
        self.numeric_columns = tuple(numeric)
        scores = self.predict_scores(frame)
        self.edges = np.unique(np.quantile(scores, np.linspace(0, 1, self.k + 1)))[1:-1]
        self.fit_split = split_id
        return self

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CTR encoder is not fitted")
        return np.asarray(
            self.model.decision_function(frame[list(self.source_columns)]),
            dtype=float,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.edges is None:
            raise RuntimeError("CTR encoder is not fitted")
        values = np.searchsorted(self.edges, self.predict_scores(frame), side="right")
        return validate_tokens(values, self.k)


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
    if name == "ctr_logistic_quantile":
        return CtrLogisticQuantileEncoder(
            k,
            seed,
            policy,
            max_categories=int(kwargs.get("max_categories", 256)),
            alpha=float(kwargs.get("alpha", 1e-6)),
            epochs=int(kwargs.get("epochs", 30)),
        )
    if name == "tree_leaf_hash":
        return TreeLeafHashEncoder(k, seed, policy)
    raise KeyError(f"unknown encoder: {name}")


ENCODER_REGISTRY = {
    "precomputed": PrecomputedEncoder,
    "hash": HashEncoder,
    "ctr_quantile": ScoreEncoder,
    "ctr_kmeans": ScoreEncoder,
    "ctr_logistic_quantile": CtrLogisticQuantileEncoder,
    "tree_leaf_hash": TreeLeafHashEncoder,
}
