from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capt12.models.reference import TOKEN_REFERENCE_FEATURE_SCHEMA, ReferenceModel


def test_numeric_token_can_be_forced_to_categorical_reference_feature(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "__token__": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "context": ["a", "b", "c"] * 3,
            "label": [0, 0, 0, 1, 1, 1, 0, 0, 0],
        }
    )
    reference = ReferenceModel("logistic_regression").fit(
        frame,
        "label",
        ["__token__", "context"],
        categorical_columns=["__token__"],
    )

    preprocess = reference.model.named_steps["preprocess"]
    transformers = {name: columns for name, _, columns in preprocess.transformers_}
    assert "numeric" not in transformers
    assert transformers["forced_categorical"] == ["__token__"]
    assert transformers["categorical"] == ["context"]
    assert reference.categorical_columns == ("__token__",)
    assert reference.feature_schema == TOKEN_REFERENCE_FEATURE_SCHEMA
    predictions = reference.predict(pd.DataFrame({"__token__": [0, 1], "context": ["a", "a"]}))
    assert predictions[0] < predictions[1]

    artifact = tmp_path / "reference.joblib"
    reference.save(artifact)
    loaded = ReferenceModel.load(artifact)
    assert loaded.categorical_columns == ("__token__",)
    assert loaded.feature_schema == TOKEN_REFERENCE_FEATURE_SCHEMA
    np.testing.assert_allclose(loaded.predict(frame), reference.predict(frame))


def test_forced_categorical_token_does_not_merge_more_than_128_ids() -> None:
    token_count = 129
    frame = pd.DataFrame(
        {
            "__token__": np.arange(token_count),
            "context": ["a"] * token_count,
            "label": np.arange(token_count) % 2,
        }
    )
    reference = ReferenceModel("logistic_regression").fit(
        frame,
        "label",
        ["__token__", "context"],
        categorical_columns=["__token__"],
    )

    onehot = reference.model.named_steps["preprocess"].named_transformers_[
        "forced_categorical"
    ].named_steps["onehot"]
    assert len(onehot.categories_[0]) == token_count
    assert onehot.transform(frame[["__token__"]].to_numpy()).shape == (
        token_count,
        token_count,
    )


def test_categorical_token_predictions_are_equivariant_to_id_relabeling() -> None:
    frame = pd.DataFrame(
        {
            "__token__": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "context": ["a", "b", "c"] * 3,
            "label": [0, 0, 0, 1, 1, 1, 0, 0, 0],
        }
    )
    relabeling = {0: 17, 1: 3, 2: 41}
    relabeled = frame.assign(__token__=frame["__token__"].map(relabeling))

    original_reference = ReferenceModel("logistic_regression").fit(
        frame,
        "label",
        ["__token__", "context"],
        categorical_columns=["__token__"],
    )
    relabeled_reference = ReferenceModel("logistic_regression").fit(
        relabeled,
        "label",
        ["__token__", "context"],
        categorical_columns=["__token__"],
    )

    evaluation = pd.DataFrame({"__token__": [0, 1, 2], "context": ["a", "a", "a"]})
    relabeled_evaluation = evaluation.assign(__token__=evaluation["__token__"].map(relabeling))
    np.testing.assert_allclose(
        original_reference.predict(evaluation),
        relabeled_reference.predict(relabeled_evaluation),
        rtol=0.0,
        atol=1e-12,
    )


def test_categorical_reference_columns_must_be_model_inputs() -> None:
    frame = pd.DataFrame({"x": [0, 1], "label": [0, 1]})
    with pytest.raises(ValueError, match="outside feature_columns"):
        ReferenceModel("logistic_regression").fit(
            frame,
            "label",
            ["x"],
            categorical_columns=["missing"],
        )
