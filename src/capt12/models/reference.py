from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LEGACY_REFERENCE_FEATURE_SCHEMA = "legacy_auto_v0"
TABULAR_REFERENCE_FEATURE_SCHEMA = "tabular_auto_v1"
TOKEN_REFERENCE_FEATURE_SCHEMA = "categorical_token_v1"


@dataclass
class ReferenceModel:
    name: str
    eta: float = 1e-6
    model: Any | None = None
    feature_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    feature_schema: str = LEGACY_REFERENCE_FEATURE_SCHEMA
    fit_split: str | None = None
    prediction_column: str | None = None
    input_manifest: dict[str, Any] | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        label_col: str,
        feature_columns: Sequence[str],
        *,
        split_id: str = "D_model",
        sensitive_columns: Sequence[str] = (),
        include_sensitive: bool = False,
        categorical_columns: Sequence[str] = (),
    ) -> ReferenceModel:
        if split_id != "D_model":
            raise ValueError("f_ref may only be fit on D_model")
        columns = list(feature_columns)
        if not include_sensitive:
            forbidden = set(columns).intersection(sensitive_columns)
            if forbidden:
                raise ValueError(f"protected proxy columns cannot enter default f_ref: {sorted(forbidden)}")
        forced_categorical_set = set(categorical_columns)
        unknown_categorical = forced_categorical_set - set(columns)
        if unknown_categorical:
            raise ValueError(
                "categorical reference columns are outside feature_columns: "
                f"{sorted(unknown_categorical)}"
            )
        forced_categorical = [c for c in columns if c in forced_categorical_set]
        inferred_categorical = [
            c
            for c in columns
            if c not in forced_categorical_set
            and not pd.api.types.is_numeric_dtype(frame[c])
        ]
        numeric = [
            c for c in columns if c not in forced_categorical_set and c not in inferred_categorical
        ]
        transformers = []
        if numeric:
            transformers.append(
                ("numeric", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), numeric)
            )
        if forced_categorical:
            transformers.append(
                (
                    "forced_categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    forced_categorical,
                )
            )
        if inferred_categorical:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=128)),
                        ]
                    ),
                    inferred_categorical,
                )
            )
        preprocess = ColumnTransformer(transformers)
        self.model = Pipeline(
            [
                ("preprocess", preprocess),
                ("logistic", LogisticRegression(max_iter=300, random_state=0)),
            ]
        )
        self.model.fit(frame[columns], frame[label_col].astype(int))
        self.feature_columns = tuple(columns)
        self.categorical_columns = tuple(forced_categorical)
        self.feature_schema = (
            TOKEN_REFERENCE_FEATURE_SCHEMA
            if "__token__" in forced_categorical
            else TABULAR_REFERENCE_FEATURE_SCHEMA
        )
        self.fit_split = split_id
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.prediction_column is not None:
            values = frame[self.prediction_column].to_numpy(dtype=float)
        elif self.model is not None:
            values = self.model.predict_proba(frame[list(self.feature_columns)])[:, 1]
        else:
            raise RuntimeError("reference model is not fitted")
        return np.clip(values, self.eta, 1 - self.eta)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ReferenceModel:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("artifact is not a ReferenceModel")
        return loaded


def validate_external_reference_inputs(
    model: ReferenceModel,
    *,
    allowed_feature_columns: Sequence[str],
    sensitive_columns: Sequence[str],
    prediction_manifest_path: str | Path | None = None,
) -> ReferenceModel:
    """Validate the declared inputs of an externally supplied reference.

    A serialized ``ReferenceModel`` carries its feature list.  A precomputed
    prediction column has no inspectable model, so a manifest declaring the
    prediction column and its upstream feature columns is mandatory.
    """
    allowed = set(allowed_feature_columns)
    sensitive = set(sensitive_columns)
    if model.prediction_column is not None:
        if prediction_manifest_path is None:
            raise ValueError(
                "precomputed f_ref requires prediction_manifest_path with declared feature_columns"
            )
        manifest = json.loads(Path(prediction_manifest_path).read_text())
        if manifest.get("prediction_column") != model.prediction_column:
            raise ValueError("prediction manifest does not match prediction_column")
        features = set(manifest.get("feature_columns", []))
        if not features:
            raise ValueError("prediction manifest must declare nonempty feature_columns")
        model.input_manifest = manifest
    else:
        features = set(model.feature_columns)
        if not features:
            raise ValueError("serialized f_ref must declare nonempty feature_columns")
    forbidden = features.intersection(sensitive)
    if forbidden:
        raise ValueError(
            f"external f_ref declares protected input columns: {sorted(forbidden)}"
        )
    unexpected = features - allowed
    if unexpected:
        raise ValueError(
            f"external f_ref inputs are outside phi_source_cols: {sorted(unexpected)}"
        )
    return model


def named_reference_model(name: str, *, eta: float = 1e-6, prediction_column: str | None = None) -> ReferenceModel:
    if name in {"ctr_model", "logistic_regression"}:
        return ReferenceModel(name="logistic_regression", eta=eta)
    if name == "precomputed":
        if not prediction_column:
            raise ValueError("precomputed f_ref requires prediction_column")
        return ReferenceModel(name=name, eta=eta, prediction_column=prediction_column, fit_split="external")
    if name == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except ImportError as error:
            raise ImportError("lightgbm requires the optional 'lightgbm' extra") from error
        raise NotImplementedError("LightGBM adapter is optional; use a serialized artifact in this release")
    raise KeyError(f"unknown fixed model configuration: {name}")


MODEL_REGISTRY = {
    "ctr_model": "logistic regression on configured Z/B source features",
    "logistic_regression": "scikit-learn logistic regression",
    "precomputed": "configured prediction column",
    "serialized_artifact": "ReferenceModel.load(path)",
    "lightgbm": "optional extra / serialized artifact",
}
