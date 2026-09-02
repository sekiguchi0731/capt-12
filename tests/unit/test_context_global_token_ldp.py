from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from capt12.certification.artifact import hash_array
from capt12.config import run_id
from capt12.experiments.context_global_token_ldp import (
    _comparison_tables,
    _json_default,
    _ldp_max_violation,
    _load_seed_artifact,
    _repair_ldp_uniform,
)
from capt12.models.reference import TOKEN_REFERENCE_FEATURE_SCHEMA, ReferenceModel


def _joblib_payload(value: object) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(value, buffer)
    return buffer.getvalue()


def _seed_artifact_bundle(
    tmp_path: Path,
    *,
    resolved_source_git_sha: str = "a" * 40,
    tamper_representation_cost: bool = False,
) -> tuple[Path, dict[str, str]]:
    mapper_payload = _joblib_payload({"mapper": "fixture"})
    encoder_payload = _joblib_payload({"encoder": "fixture"})
    reference = ReferenceModel(
        name="logistic_regression",
        feature_columns=("__token__", "context"),
        categorical_columns=("__token__",),
        feature_schema=TOKEN_REFERENCE_FEATURE_SCHEMA,
    )
    reference_payload = _joblib_payload(reference)
    mapper_hash = hashlib.sha256(mapper_payload).hexdigest()
    encoder_hash = hashlib.sha256(encoder_payload).hexdigest()
    reference_hash = hashlib.sha256(reference_payload).hexdigest()

    original_cost = np.arange(64 * 64, dtype=float).reshape(64, 64)
    representation_cost_hash = hash_array(original_cost)
    stored_cost = original_cost.copy()
    if tamper_representation_cost:
        stored_cost[0, 0] += 1
    frozen_design = io.BytesIO()
    np.savez(
        frozen_design,
        reference_feature_schema=np.asarray(TOKEN_REFERENCE_FEATURE_SCHEMA),
        reference_feature_columns=np.asarray(["__token__", "context"]),
        reference_categorical_columns=np.asarray(["__token__"]),
        representation_token_cost_hash=np.asarray(representation_cost_hash),
        representation_token_cost=stored_cost,
        representation_probability_grid=np.zeros((64, 1)),
        representation_mode=np.asarray("objective_aligned"),
        representation_objective=np.asarray("empirical_logloss"),
    )
    config = {
        "source_git_sha": resolved_source_git_sha,
        "context_cols": ["context"],
        "K": 64,
    }
    run_id_value = run_id(config)
    prefix = f"runs/seed-0/{run_id_value}/"
    channel_manifest = {
        "version": 4,
        "runtime_mapper_hash": mapper_hash,
        "reference_model_hash": reference_hash,
        "reference_feature_schema": TOKEN_REFERENCE_FEATURE_SCHEMA,
        "reference_feature_columns": ["__token__", "context"],
        "reference_categorical_columns": ["__token__"],
    }

    nested_payload = io.BytesIO()
    with zipfile.ZipFile(nested_payload, "w") as archive:
        archive.writestr(prefix + "resolved_config.yaml", json.dumps(config))
        archive.writestr(
            prefix + "mechanism/context_channel_manifest.json",
            json.dumps(channel_manifest),
        )
        archive.writestr(prefix + "models/category_mapper.joblib", mapper_payload)
        archive.writestr(prefix + "models/encoder.joblib", encoder_payload)
        archive.writestr(prefix + "models/reference.joblib", reference_payload)
        archive.writestr(prefix + "mechanism/frozen_design.npz", frozen_design.getvalue())

    bundle = tmp_path / "epsilon-grid.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "epsilon/1/sol_seed_stability_review_bundle.zip",
            nested_payload.getvalue(),
        )
    return bundle, {
        "source_git_sha": "a" * 40,
        "run_id": run_id_value,
        "encoder_hash": encoder_hash,
        "reference_hash": reference_hash,
        "representation_cost_hash": representation_cost_hash,
    }


def _load_fixture(bundle: Path, expected: dict[str, str]) -> dict[str, object]:
    return _load_seed_artifact(
        bundle,
        1.0,
        0,
        expected["run_id"],
        expected["source_git_sha"],
        expected["encoder_hash"],
        expected["reference_hash"],
        expected["representation_cost_hash"],
    )


def test_json_default_normalizes_numpy_scalars() -> None:
    encoded = json.dumps(
        {"seed": np.int64(4), "epsilon": np.float64(1), "valid": np.bool_(True)},
        default=_json_default,
    )
    assert json.loads(encoded) == {"seed": 4, "epsilon": 1.0, "valid": True}


def test_load_seed_artifact_accepts_matching_nested_bundle(tmp_path: Path) -> None:
    bundle, expected = _seed_artifact_bundle(tmp_path)

    artifact = _load_fixture(bundle, expected)

    assert artifact["encoder_hash"] == expected["encoder_hash"]
    assert artifact["reference_hash"] == expected["reference_hash"]
    assert artifact["arrays"]["representation_token_cost"].shape == (64, 64)


def test_load_seed_artifact_rejects_encoder_bytes_hash_mismatch(tmp_path: Path) -> None:
    bundle, expected = _seed_artifact_bundle(tmp_path)
    expected["encoder_hash"] = "0" * 64

    with pytest.raises(ValueError, match="encoder hash"):
        _load_fixture(bundle, expected)


def test_load_seed_artifact_rejects_tampered_representation_cost(tmp_path: Path) -> None:
    bundle, expected = _seed_artifact_bundle(
        tmp_path,
        tamper_representation_cost=True,
    )

    with pytest.raises(ValueError, match="representation-cost hash"):
        _load_fixture(bundle, expected)


def test_load_seed_artifact_rejects_resolved_source_sha_mismatch(tmp_path: Path) -> None:
    bundle, expected = _seed_artifact_bundle(
        tmp_path,
        resolved_source_git_sha="b" * 40,
    )

    with pytest.raises(ValueError, match="resolved config"):
        _load_fixture(bundle, expected)


def test_uniform_repair_makes_released_channel_strictly_ldp() -> None:
    channel = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    repaired, mixing, before, after = _repair_ldp_uniform(channel, 1.0, margin=1e-12)
    assert before > 0
    assert 0 < mixing < 1
    assert after < 0
    assert _ldp_max_violation(repaired, 1.0) < 0
    assert np.allclose(repaired.sum(axis=1), 1.0, atol=1e-14, rtol=0)
    assert np.min(repaired) > 0


def test_global_comparison_uses_same_seed_baseline_for_each_block_size() -> None:
    frontier = pd.DataFrame(
        {
            "frozen_design_seed": [0, 1],
            "epsilon": [1.0, 1.0],
            "test_context_capt_expected_randomized_log_loss": [0.4, 0.5],
        }
    )
    blocks = pd.DataFrame(
        {
            "frozen_design_seed": [0, 0, 1, 1],
            "L": [8, 16, 8, 16],
            "test_context_capt_expected_randomized_log_loss": [0.45, 0.4, 0.55, 0.5],
        }
    )
    baselines = pd.DataFrame(
        {
            "frozen_design_seed": [0, 1],
            "epsilon": [1.0, 1.0],
            "expected_randomized_log_loss": [0.6, 0.7],
        }
    )
    frontier_result, block_result = _comparison_tables(frontier, blocks, baselines)
    assert np.allclose(frontier_result["global_token_ldp_gain_micro"], [200_000, 200_000])
    assert np.allclose(
        block_result.sort_values(["frozen_design_seed", "L"])[
            "global_token_ldp_gain_micro"
        ],
        [150_000, 200_000, 150_000, 200_000],
    )
