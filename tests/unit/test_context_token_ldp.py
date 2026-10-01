from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import capt12.experiments.context_fixed_test_seeds as fixed_test_module
import capt12.experiments.context_token_ldp as token_ldp_module
from capt12.certification.artifact import hash_array
from capt12.experiments.context_token_ldp import (
    load_context_token_ldp_channels,
    solve_context_token_ldp_channels,
    verify_pure_ldp_decimal,
)
from capt12.mechanisms.lp import ChannelSolution, SolverInfo
from capt12.models.reference import TOKEN_REFERENCE_FEATURE_SCHEMA
from capt12.utils.artifacts import sha256_file


class _Reference:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        context = str(frame["context"].iloc[0])
        values = {
            "a": np.asarray([0.2, 0.8]),
            "__MISSING__": np.asarray([0.3, 0.7]),
        }
        return values[context]


def _solver_info() -> SolverInfo:
    return SolverInfo(
        status="optimal",
        objective=0.5,
        runtime_seconds=0.01,
        iterations=2,
        primal_gap=0.0,
        dual_gap=0.0,
        message="mock optimal",
        variable_count=4,
        constraint_count=6,
        estimated_memory_bytes=128,
    )


def _install_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, list[tuple[np.ndarray, np.ndarray]]]:
    mechanism_run = tmp_path / "mechanism-run"
    mechanism_dir = mechanism_run / "mechanism"
    mechanism_dir.mkdir(parents=True)
    np.savez_compressed(
        mechanism_dir / "frozen_design.npz",
        frequencies=np.asarray([0.5, 0.5]),
        context_levels=np.asarray(["a"]),
        token_context_weights=np.asarray([[30.0], [20.0]]),
        token_context_label_count=np.asarray([[30.0], [20.0]]),
        token_context_label_sum=np.asarray([[5.0], [15.0]]),
        reference_feature_schema=np.asarray(TOKEN_REFERENCE_FEATURE_SCHEMA),
    )
    strict_channel = np.asarray([[0.6, 0.4], [0.4, 0.6]])
    artifacts: dict[str, Any] = {
        "resolved": {
            "context_cols": ["context"],
            "context_utility_objective": "empirical_logloss",
            "distortion_clip": 1e-6,
            "hybrid_empirical_weight": 0.5,
            "epsilon": 1.0,
            "frozen_design_seed": 0,
            "certificate_repair_margin": 1e-10,
        },
        "metadata": {"source_git_sha": "a" * 40},
        "arrays": {
            "contexts": np.asarray(["a", "__MISSING__"]),
            "assignment": np.asarray([0, 1]),
            "decoder": np.eye(2),
            "ldp_channels": np.stack([strict_channel, strict_channel]),
        },
        "reference": _Reference(),
    }

    def load_artifacts(path: Path, *, load_runtime_models: bool = True) -> dict[str, Any]:
        assert Path(path) == mechanism_run.resolve()
        assert load_runtime_models is False
        return artifacts

    monkeypatch.setattr(fixed_test_module, "_load_fixed_artifacts", load_artifacts)
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def solve(cost: np.ndarray, weights: np.ndarray, epsilon: float, **kwargs: Any):
        del epsilon, kwargs
        calls.append((np.asarray(cost).copy(), np.asarray(weights).copy()))
        return ChannelSolution(strict_channel.copy(), _solver_info())

    monkeypatch.setattr(token_ldp_module, "solve_ldp_block_lp", solve)
    output_root = tmp_path / "token-ldp"
    return mechanism_run, output_root, calls


def _remove_completed_files(artifact: Path) -> None:
    for name in (
        "context_token_ldp_channels.npz",
        "context_token_ldp_metadata.json",
        "manifest.json",
    ):
        (artifact / name).unlink()


def test_decimal_verifier_accepts_strict_channel_and_rejects_privacy_failures() -> None:
    valid = verify_pure_ldp_decimal(
        np.asarray([[0.7, 0.3], [0.4, 0.6]]),
        1.0,
    )
    assert valid["valid"] is True
    assert valid["realized_epsilon"] < 1.0
    assert valid["positive_over_zero_output_count"] == 0
    assert valid["equivalent_ordered_constraints"] == 4

    ratio_failure = verify_pure_ldp_decimal(
        np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        1.0,
    )
    assert ratio_failure["valid"] is False
    assert ratio_failure["max_additive_violation"] > 0

    zero_failure = verify_pure_ldp_decimal(np.eye(2), 1.0)
    assert zero_failure["valid"] is False
    assert zero_failure["positive_over_zero_output_count"] == 2

    with pytest.raises(ValueError, match="nonnegative"):
        verify_pure_ldp_decimal(np.asarray([[1.1, -0.1], [0.4, 0.6]]), 1.0)
    with pytest.raises(ValueError, match="rows must sum"):
        verify_pure_ldp_decimal(np.asarray([[0.2, 0.2], [0.2, 0.2]]), 1.0)
    with pytest.raises(ValueError, match="epsilon"):
        verify_pure_ldp_decimal(np.eye(2), float("nan"))


def test_solver_checkpoints_are_reused_and_missing_context_uses_global_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism_run, output_root, calls = _install_fixture(tmp_path, monkeypatch)

    artifact = solve_context_token_ldp_channels(
        mechanism_run,
        output_root=output_root,
        solver_time_limit=5,
    )
    assert len(calls) == 2
    np.testing.assert_allclose(calls[0][1], [0.6, 0.4])
    np.testing.assert_allclose(calls[1][1], [0.5, 0.5])
    contexts, channels, metadata = load_context_token_ldp_channels(artifact)
    assert contexts.tolist() == ["a", "__MISSING__"]
    assert channels.shape == (2, 2, 2)
    assert metadata["signature"]["implementation_sha256"] == sha256_file(
        Path(token_ldp_module.__file__).resolve()
    )
    assert all(
        record["raw_token_minus_block_objective"] <= 1e-8 for record in metadata["context_records"]
    )

    _remove_completed_files(artifact)
    resumed = solve_context_token_ldp_channels(
        mechanism_run,
        output_root=output_root,
        solver_time_limit=5,
    )
    assert resumed == artifact
    assert len(calls) == 2
    load_context_token_ldp_channels(resumed)


def test_corrupt_checkpoint_is_rejected_while_valid_checkpoint_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism_run, output_root, calls = _install_fixture(tmp_path, monkeypatch)
    artifact = solve_context_token_ldp_channels(mechanism_run, output_root=output_root)
    assert len(calls) == 2
    _remove_completed_files(artifact)

    corrupt_path = artifact / "checkpoints" / "context-00.npz"
    np.savez_compressed(corrupt_path, channel=np.eye(2))
    solve_context_token_ldp_channels(mechanism_run, output_root=output_root)

    assert len(calls) == 3
    load_context_token_ldp_channels(artifact)


def test_loader_rejects_metadata_record_corruption_even_with_updated_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism_run, output_root, _ = _install_fixture(tmp_path, monkeypatch)
    artifact = solve_context_token_ldp_channels(mechanism_run, output_root=output_root)
    metadata_path = artifact / "context_token_ldp_metadata.json"
    manifest_path = artifact / "manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["context_records"][0]["context"] = "tampered"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata_sha256"] = sha256_file(metadata_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="record does not match"):
        load_context_token_ldp_channels(artifact)


def test_loader_rejects_privacy_corruption_even_when_hashes_are_rewritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mechanism_run, output_root, _ = _install_fixture(tmp_path, monkeypatch)
    artifact = solve_context_token_ldp_channels(mechanism_run, output_root=output_root)
    channel_path = artifact / "context_token_ldp_channels.npz"
    metadata_path = artifact / "context_token_ldp_metadata.json"
    manifest_path = artifact / "manifest.json"
    contexts, channels, metadata = load_context_token_ldp_channels(artifact)
    channels[0] = np.eye(2)
    np.savez_compressed(channel_path, contexts=contexts, channels=channels)
    channel_sha = sha256_file(channel_path)
    metadata["channel_sha256"] = channel_sha
    metadata["context_records"][0]["channel_hash"] = hash_array(channels[0])
    metadata_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["channel_sha256"] = channel_sha
    manifest["metadata_sha256"] = sha256_file(metadata_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="strict slack"):
        load_context_token_ldp_channels(artifact)
