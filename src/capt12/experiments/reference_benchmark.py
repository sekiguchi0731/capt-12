from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from capt12.config import validate_config
from capt12.data.loader import load_parquet_sample
from capt12.data.preprocessing import FrozenCategoryMapper
from capt12.encoders.base import make_encoder
from capt12.models.reference import named_reference_model


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "logloss": float(log_loss(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, predictions)),
        "average_precision": float(average_precision_score(labels, predictions)),
        "mean_prediction": float(np.mean(predictions)),
        "click_rate": float(np.mean(labels)),
    }


def _load_days(
    *,
    data_root: str | Path,
    columns: list[str],
    days: list[int],
    max_rows_per_day: int | None,
) -> pd.DataFrame:
    if max_rows_per_day is None:
        return load_parquet_sample(data_root=data_root, columns=columns, days=days)
    frames = [
        load_parquet_sample(
            data_root=data_root,
            columns=columns,
            days=[day],
            max_rows=max_rows_per_day,
        )
        for day in days
    ]
    return pd.concat(frames, ignore_index=True)


def benchmark_nonprivate_reference(
    config: dict[str, Any],
    *,
    max_rows_per_day: int | None = None,
) -> dict[str, Any]:
    """Fit on D_model and evaluate the unsanitized K-token f_ref on D_test."""
    config = validate_config(config)
    profile_attributes = list(config["profiles"][0].split("+"))
    contexts = list(config.get("context_cols", []))
    sensitive = list(config.get("sensitive_cols", []))
    source = list(config.get("phi_source_cols", []))
    label = str(config.get("label_col", "is_clicked"))
    columns = list(dict.fromkeys([label, *profile_attributes, *contexts, *source]))
    started = time.perf_counter()

    model_frame = _load_days(
        data_root=config["data_root"],
        columns=columns,
        days=list(map(int, config["splits"]["D_model"])),
        max_rows_per_day=max_rows_per_day,
    )
    unified_unknown = config.get("sensitive_fallback_policy") == "unified_unknown"
    mapper = FrozenCategoryMapper(
        int(config.get("max_context_cardinality", 32)),
        unknown_columns=(frozenset(profile_attributes) if unified_unknown else frozenset()),
        unknown_value=str(config.get("sensitive_unknown_value", "__UNKNOWN__")),
    ).fit(model_frame, [*profile_attributes, *contexts], split_id="D_model")
    model_frame = mapper.transform(model_frame)
    encoder = make_encoder(
        config.get("phi", "hash"),
        int(config.get("K", 64)),
        seed=int(config.get("frozen_design_seed", config.get("seed", 0))),
        sensitive_policy=config.get("phi_sensitive_policy", "exclude"),
        column=config.get("precomputed_token_col", "token"),
        max_categories=int(config.get("encoder_max_categories", 256)),
        alpha=float(config.get("encoder_sgd_alpha", 1e-6)),
        epochs=int(config.get("encoder_sgd_epochs", 30)),
    )
    encoder.fit(
        model_frame,
        source,
        split_id="D_model",
        sensitive_columns=sensitive,
        labels=model_frame[label].to_numpy(),
    )
    model_frame["__token__"] = encoder.transform(model_frame)
    reference = named_reference_model(
        config.get("fixed_model", "ctr_model"),
        eta=float(config.get("distortion_clip", 1e-6)),
    ).fit(
        model_frame,
        label,
        ["__token__", *contexts],
        split_id="D_model",
        sensitive_columns=sensitive,
        categorical_columns=["__token__"],
    )
    training_predictions = reference.predict(model_frame)
    training_labels = model_frame[label].to_numpy(dtype=int)
    training_metrics = _metrics(training_labels, training_predictions)
    training_metrics["rows"] = int(len(model_frame))
    training_metrics["token_count"] = int(model_frame["__token__"].nunique())
    del model_frame, training_predictions, training_labels
    gc.collect()

    label_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    per_day: list[dict[str, float | int]] = []
    for day in config["splits"]["D_test"]:
        test_frame = load_parquet_sample(
            data_root=config["data_root"],
            columns=columns,
            days=[day],
            max_rows=max_rows_per_day,
        )
        test_frame = mapper.transform(test_frame)
        test_frame["__token__"] = encoder.transform(test_frame)
        labels = test_frame[label].to_numpy(dtype=int)
        predictions = reference.predict(test_frame)
        day_metrics: dict[str, float | int] = {
            "day": int(day),
            "rows": int(len(test_frame)),
            **_metrics(labels, predictions),
        }
        per_day.append(day_metrics)
        label_parts.append(labels)
        prediction_parts.append(predictions)
        del test_frame
        gc.collect()

    labels = np.concatenate(label_parts)
    predictions = np.concatenate(prediction_parts)
    prevalence = float(np.mean(labels))
    test_metrics = _metrics(labels, predictions)
    test_metrics.update(
        {
            "rows": int(len(labels)),
            "constant_logloss": float(
                log_loss(labels, np.full(len(labels), prevalence, dtype=float))
            ),
        }
    )
    return {
        "status": "ok",
        "phi": encoder.name,
        "K": int(config.get("K", 64)),
        "D_model_days": list(map(int, config["splits"]["D_model"])),
        "D_test_days": list(map(int, config["splits"]["D_test"])),
        "declared_source_columns": source,
        "active_source_columns": list(encoder.source_columns),
        "categorical_source_columns": list(
            getattr(encoder, "categorical_columns", ())
        ),
        "numeric_source_columns": list(getattr(encoder, "numeric_columns", ())),
        "encoder_hyperparameters": {
            "max_categories": int(config.get("encoder_max_categories", 256)),
            "sgd_alpha": float(config.get("encoder_sgd_alpha", 1e-6)),
            "sgd_epochs": int(config.get("encoder_sgd_epochs", 30)),
            "seed": int(config.get("frozen_design_seed", config.get("seed", 0))),
        },
        "training": training_metrics,
        "test": test_metrics,
        "per_test_day": per_day,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def write_reference_benchmark(result: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return path
