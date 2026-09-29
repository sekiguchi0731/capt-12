from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capt12.encoders.base import CtrLogisticQuantileEncoder, make_encoder


def test_ctr_logistic_quantile_uses_labels_and_freezes_quantile_tokens() -> None:
    frame = pd.DataFrame(
        {
            "category": np.repeat(np.arange(8), 20),
            "publisher_id": np.repeat(np.arange(8), 20).astype(float),
            "continuous": np.tile(np.linspace(-1.0, 1.0, 20), 8),
            "all_missing": np.nan,
        }
    )
    labels = ((frame["category"] >= 4) | (frame["continuous"] > 0.8)).astype(int)
    encoder = CtrLogisticQuantileEncoder(8, seed=3, epochs=10).fit(
        frame,
        ["category", "publisher_id", "continuous", "all_missing"],
        labels=labels.to_numpy(),
        sensitive_columns=[],
    )

    encoded = encoder.transform(frame)
    assert encoded.min() >= 0
    assert encoded.max() < 8
    assert len(np.unique(encoded)) == 8
    assert encoder.source_columns == ("category", "publisher_id", "continuous")
    assert encoder.categorical_columns == ("category", "publisher_id")
    assert encoder.numeric_columns == ("continuous",)
    np.testing.assert_array_equal(encoded, encoder.transform(frame.copy()))


def test_ctr_logistic_quantile_rejects_protected_source() -> None:
    frame = pd.DataFrame({"s": [0, 0, 1, 1], "x": [0.0, 1.0, 0.0, 1.0]})
    encoder = make_encoder("ctr_logistic_quantile", 2)
    with pytest.raises(ValueError, match="sensitive proxy"):
        encoder.fit(
            frame,
            ["s", "x"],
            labels=np.array([0, 1, 0, 1]),
            sensitive_columns=["s"],
        )


def test_ctr_logistic_quantile_validates_hyperparameters() -> None:
    with pytest.raises(ValueError, match="max_categories"):
        CtrLogisticQuantileEncoder(2, max_categories=1)
    with pytest.raises(ValueError, match="alpha"):
        CtrLogisticQuantileEncoder(2, alpha=0)
    with pytest.raises(ValueError, match="epochs"):
        CtrLogisticQuantileEncoder(2, epochs=0)
