from __future__ import annotations

import json

import numpy as np
import pytest

from capt12.certification.artifact import (
    hash_array,
    hash_json,
    make_certificate,
    verify_certificate,
)
from capt12.certification.robust import verify_robust_channel
from capt12.confidence.boxes import ConfidenceBox, cp_box, dp_aware_box, full_simplex_box
from capt12.mechanisms.baselines import common_cover, k_ary_rr
from capt12.mechanisms.lp import SolverInfo
from capt12.privacy.adjacency import Group, build_adjacency
from capt12.utils.artifacts import sha256_file


@pytest.fixture(autouse=True)
def _clean_committed_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "capt12.certification.artifact.require_clean_worktree",
        lambda: "test-source-sha",
    )


def test_certificate_roundtrip_verifies_identically(tmp_path) -> None:
    groups = [Group("a", (0,)), Group("a", (1,))]
    p0 = np.array([0.7, 0.3])
    p1 = np.array([0.2, 0.8])
    boxes = {
        groups[0].key(): ConfidenceBox(p0, p0, p0, "point", 1),
        groups[1].key(): ConfidenceBox(p1, p1, p1, "point", 1),
    }
    adjacency = build_adjacency(groups, epsilon=0)
    channel = common_cover(np.array([0.4, 0.6]))
    before = verify_robust_channel(channel, boxes, adjacency)
    solver = SolverInfo("optimal", 0, 0.01, 2, primal_gap=0)
    certificate = make_certificate(
        config={"dataset": "synthetic", "alpha_cert": 0.05, "epsilon": 0},
        channel=channel,
        boxes=boxes,
        adjacency=adjacency,
        verification=before,
        solver=solver,
        assignment=np.arange(2),
        decoder=np.eye(2),
        groups=groups,
    )
    path = tmp_path / "certificate.json"
    certificate.write(path)
    after = verify_certificate(path)
    assert before.valid == after.valid
    assert np.isclose(before.max_violation, after.max_violation)
    assert json.loads(path.read_text())["channel"] == channel.tolist()
    payload = json.loads(path.read_text())
    first_box = next(iter(payload["boxes"].values()))
    first_box["experimental"] = True
    payload["component_hashes"]["boxes"] = hash_json(payload["boxes"])
    path.write_text(json.dumps(payload))
    rejected = verify_certificate(path)
    assert not rejected.valid
    assert "experimental" in rejected.worst_case["error"]
    first_box["experimental"] = False
    payload["resolved_config"]["dataset"] = "criteo"
    payload["resolved_config"]["sampling_assumption"] = "user_day_iid"
    payload["resolved_config"]["contribution_policy"] = (
        "one-display-per-uuid-day"
    )
    payload["dp_parameters"] = {
        "epsilon": None,
        "delta": None,
        "contribution_policy": "one-display-per-uuid-day",
    }
    payload["coverage"] = {
        "expected_group_count": 2,
        "observed_group_count": 2,
        "missing_group_count": 0,
        "missing_groups": [],
        "rare_group_count": 0,
        "hybrid_connectivity_gaps": 0,
        "requires_universal_cover": False,
    }
    payload["component_hashes"]["boxes"] = hash_json(payload["boxes"])
    payload["component_hashes"]["config"] = hash_json(payload["resolved_config"])
    payload["component_hashes"]["dp_parameters"] = hash_json(payload["dp_parameters"])
    payload["component_hashes"]["coverage"] = hash_json(payload["coverage"])
    path.write_text(json.dumps(payload))
    rejected = verify_certificate(path)
    assert not rejected.valid
    assert "synthetic" in rejected.worst_case["error"]


def test_certificate_ignores_embedded_solver_gap_and_checks_hashes(tmp_path) -> None:
    groups = [Group("a", (0,)), Group("a", (1,))]
    counts = {
        groups[0].key(): np.array([60, 40]),
        groups[1].key(): np.array([40, 60]),
    }
    adjacency = build_adjacency(groups, epsilon=0)
    boxes = {
        key: cp_box(value, group_count=2, comparisons=len(adjacency))
        for key, value in counts.items()
    }
    channel = common_cover(np.array([0.5, 0.5]))
    verification = verify_robust_channel(channel, boxes, adjacency)
    certificate = make_certificate(
        config={
            "dataset": "criteo",
            "confidence": "cp_box",
            "alpha_cert": 0.05,
            "epsilon": 0,
            "sampling_assumption": "user_day_iid",
            "contribution_policy": "one-display-per-uuid-day",
        },
        channel=channel,
        boxes=boxes,
        adjacency=adjacency,
        verification=verification,
        solver=SolverInfo("optimal", 0, 0.0, 1, primal_gap=0),
        histogram_counts=counts,
        assignment=np.arange(2),
        decoder=np.eye(2),
        groups=groups,
        dp_parameters={
            "epsilon": None,
            "delta": None,
            "contribution_policy": "one-display-per-uuid-day",
        },
        coverage={
            "expected_group_count": 2,
            "observed_group_count": 2,
            "missing_group_count": 0,
            "missing_groups": [],
            "rare_group_count": 0,
            "hybrid_connectivity_gaps": 0,
            "requires_universal_cover": False,
        },
    )
    path = tmp_path / "certificate.json"
    certificate.write(path)
    payload = json.loads(path.read_text())
    payload["solver"]["primal_gap"] = 1e6
    path.write_text(json.dumps(payload))
    assert verify_certificate(path).valid
    assert not verify_certificate(path, tolerance=1.0).valid

    payload["channel"][0][0] = 0.75
    path.write_text(json.dumps(payload))
    result = verify_certificate(path)
    assert not result.valid
    assert "hash mismatch" in result.worst_case["error"]


def test_certificate_rejects_mismatched_source_provenance(tmp_path) -> None:
    groups = [Group("a", (0,)), Group("a", (1,))]
    values = [np.array([0.6, 0.4]), np.array([0.4, 0.6])]
    boxes = {
        group.key(): ConfidenceBox(value, value, value, "point", 1)
        for group, value in zip(groups, values, strict=True)
    }
    adjacency = build_adjacency(groups, epsilon=0)
    channel = common_cover(np.array([0.5, 0.5]))
    certificate = make_certificate(
        config={"dataset": "synthetic", "epsilon": 0},
        channel=channel,
        boxes=boxes,
        adjacency=adjacency,
        verification=verify_robust_channel(channel, boxes, adjacency),
        solver=SolverInfo("optimal", 0, 0, 1),
        assignment=np.arange(2),
        decoder=np.eye(2),
        groups=groups,
    )
    path = tmp_path / "certificate.json"
    certificate.write(path)
    payload = json.loads(path.read_text())
    payload["resolved_config"]["source_git_sha"] = "different-source-sha"
    payload["component_hashes"]["config"] = hash_json(payload["resolved_config"])
    path.write_text(json.dumps(payload))

    rejected = verify_certificate(path)
    assert not rejected.valid
    assert "source Git SHA" in rejected.worst_case["error"]


def test_experimental_and_real_data_point_certificates_are_rejected() -> None:
    groups = [Group("a", (0,)), Group("a", (1,))]
    adjacency = build_adjacency(groups, epsilon=0)
    channel = common_cover(np.array([0.5, 0.5]))
    experimental = {
        group.key(): dp_aware_box(np.array([50.0, 50.0]), noise_scale=1.0)
        for group in groups
    }
    verification = verify_robust_channel(channel, experimental, adjacency)
    with pytest.raises(ValueError, match="experimental"):
        make_certificate(
            config={"dataset": "criteo"},
            channel=channel,
            boxes=experimental,
            adjacency=adjacency,
            verification=verification,
            solver={},
        )

    point = {
        group.key(): ConfidenceBox(
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            "point",
            1,
        )
        for group in groups
    }
    with pytest.raises(ValueError, match="synthetic"):
        make_certificate(
            config={"dataset": "criteo"},
            channel=channel,
            boxes=point,
            adjacency=adjacency,
            verification=verify_robust_channel(channel, point, adjacency),
            solver={},
        )


def test_merge_to_other_is_rejected_by_writer_and_verifier(tmp_path) -> None:
    groups = [Group("a", (0,)), Group("a", (1,))]
    values = [np.array([0.6, 0.4]), np.array([0.4, 0.6])]
    boxes = {
        group.key(): ConfidenceBox(value, value, value, "point", 1)
        for group, value in zip(groups, values, strict=True)
    }
    adjacency = build_adjacency(groups, epsilon=0)
    channel = common_cover(np.array([0.5, 0.5]))
    verification = verify_robust_channel(channel, boxes, adjacency)
    with pytest.raises(ValueError, match="merge_to_other"):
        make_certificate(
            config={
                "dataset": "synthetic",
                "rare_group_policy": "merge_to_other",
                "epsilon": 0,
            },
            channel=channel,
            boxes=boxes,
            adjacency=adjacency,
            verification=verification,
            solver=SolverInfo("optimal", 0, 0, 1),
            assignment=np.arange(2),
            decoder=np.eye(2),
            groups=groups,
        )

    certificate = make_certificate(
        config={"dataset": "synthetic", "epsilon": 0},
        channel=channel,
        boxes=boxes,
        adjacency=adjacency,
        verification=verification,
        solver=SolverInfo("optimal", 0, 0, 1),
        assignment=np.arange(2),
        decoder=np.eye(2),
        groups=groups,
    )
    path = tmp_path / "certificate.json"
    certificate.write(path)
    payload = json.loads(path.read_text())
    payload["resolved_config"]["rare_group_policy"] = "merge_to_other"
    payload["component_hashes"]["config"] = hash_json(payload["resolved_config"])
    path.write_text(json.dumps(payload))
    rejected = verify_certificate(path)
    assert not rejected.valid
    assert "merge_to_other" in rejected.worst_case["error"]


def test_mixed_cp_and_full_simplex_certificate_reconstructs_zero_count_group(
    tmp_path,
) -> None:
    groups = [Group("a", (0,)), Group("a", (1,))]
    counts = {
        groups[0].key(): np.array([60, 40]),
        groups[1].key(): np.array([0, 0]),
    }
    adjacency = build_adjacency(groups, epsilon=1.0)
    boxes = {
        groups[0].key(): cp_box(
            counts[groups[0].key()], group_count=2, comparisons=len(adjacency)
        ),
        groups[1].key(): full_simplex_box(counts[groups[1].key()]),
    }
    channel = k_ary_rr(2, 1.0)
    verification = verify_robust_channel(channel, boxes, adjacency)
    assert verification.valid
    certificate = make_certificate(
        config={
            "dataset": "criteo",
            "confidence": "cp_box",
            "alpha_cert": 0.05,
            "epsilon": 1.0,
            "sampling_assumption": "user_day_iid",
            "contribution_policy": "one-display-per-uuid-day",
            "rare_group_policy": "confidence_box",
            "missing_group_policy": "full_simplex",
        },
        channel=channel,
        boxes=boxes,
        adjacency=adjacency,
        verification=verification,
        solver=SolverInfo("optimal", 0, 0.0, 1, primal_gap=0),
        histogram_counts=counts,
        assignment=np.arange(2),
        decoder=np.eye(2),
        groups=groups,
        dp_parameters={
            "epsilon": None,
            "delta": None,
            "contribution_policy": "one-display-per-uuid-day",
        },
        coverage={
            "expected_group_count": 2,
            "observed_group_count": 1,
            "missing_group_count": 1,
            "missing_groups": [groups[1].key()],
            "rare_group_count": 0,
            "hybrid_connectivity_gaps": 0,
            "requires_universal_cover": False,
        },
    )
    path = tmp_path / "certificate.json"
    certificate.write(path)
    assert verify_certificate(path).valid

    payload = json.loads(path.read_text())
    payload["boxes"][groups[1].key()]["method"] = "cp_box"
    payload["component_hashes"]["boxes"] = hash_json(payload["boxes"])
    path.write_text(json.dumps(payload))
    rejected = verify_certificate(path)
    assert not rejected.valid
    assert "does not match" in rejected.worst_case["error"]


def test_context_certificate_binds_mapper_and_manifest_semantics(tmp_path) -> None:
    groups = [Group("a", ("known",), "public-b"), Group("a", ("__UNKNOWN__",), "public-b")]
    counts = {
        groups[0].key(): np.array([60, 40]),
        groups[1].key(): np.array([0, 0]),
    }
    adjacency = build_adjacency(groups, epsilon=1.0)
    boxes = {
        groups[0].key(): cp_box(
            counts[groups[0].key()], group_count=2, comparisons=len(adjacency)
        ),
        groups[1].key(): full_simplex_box(counts[groups[1].key()]),
    }
    channel = k_ary_rr(2, 1.0)
    verification = verify_robust_channel(channel, boxes, adjacency)
    (tmp_path / "models").mkdir()
    (tmp_path / "mechanism").mkdir()
    mapper_path = tmp_path / "models" / "category_mapper.joblib"
    mapper_path.write_bytes(b"frozen mapper")
    mapper_hash = sha256_file(mapper_path)
    manifest = {
        "version": 1,
        "profile": "a",
        "public_context_column": "b",
        "channel_selector_inputs": ["Z", "profile", "b"],
        "protected_value_used_online": False,
        "runtime_mapper_hash": mapper_hash,
        "sensitive_coarsening": {
            "missing": "__UNKNOWN__",
            "unseen": "__UNKNOWN__",
        },
        "designs": {
            "test-design": {
                "L": 2,
                "assignment_hash": hash_array(np.arange(2)),
                "decoder_hash": hash_array(np.eye(2)),
                "table_entries": 4,
                "contexts": {
                    "public-b": hash_array(channel)
                },
            }
        },
    }
    manifest_path = tmp_path / "mechanism" / "context_channel_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = {
        "dataset": "criteo",
        "confidence": "cp_box",
        "alpha_cert": 0.05,
        "epsilon": 1.0,
        "sampling_assumption": "user_day_iid",
        "contribution_policy": "one-display-per-uuid-day",
        "rare_group_policy": "confidence_box",
        "missing_group_policy": "full_simplex",
        "sensitive_fallback_policy": "unified_unknown",
        "sensitive_unknown_value": "__UNKNOWN__",
        "public_context_policy": "stratified",
        "profiles": ["a"],
        "context_cols": ["b"],
        "channel_selector_inputs": ["Z", "profile", "b"],
        "public_context_value": "public-b",
        "context_design": "test-design",
        "splits": {},
    }
    certificate = make_certificate(
        config=config,
        channel=channel,
        boxes=boxes,
        adjacency=adjacency,
        verification=verification,
        solver=SolverInfo("optimal", 0, 0, 1, primal_gap=0),
        component_hashes={
            "mapper": mapper_hash,
            "context_channel_manifest": sha256_file(manifest_path),
        },
        split_identifiers={},
        dp_parameters={
            "epsilon": None,
            "delta": None,
            "contribution_policy": "one-display-per-uuid-day",
        },
        histogram_counts=counts,
        assignment=np.arange(2),
        decoder=np.eye(2),
        groups=groups,
        coverage={
            "expected_group_count": 2,
            "observed_group_count": 1,
            "missing_group_count": 1,
            "missing_groups": [groups[1].key()],
            "rare_group_count": 0,
            "hybrid_connectivity_gaps": 0,
            "requires_universal_cover": False,
            "public_context_value": "public-b",
        },
    )
    certificate_path = tmp_path / "certificate.json"
    certificate.write(certificate_path)
    assert verify_certificate(certificate_path).valid

    manifest["designs"]["test-design"]["contexts"]["public-b"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    payload = json.loads(certificate_path.read_text())
    payload["component_hashes"]["context_channel_manifest"] = sha256_file(manifest_path)
    certificate_path.write_text(json.dumps(payload))
    rejected = verify_certificate(certificate_path)
    assert not rejected.valid
    assert "certified context channel" in rejected.worst_case["error"]
