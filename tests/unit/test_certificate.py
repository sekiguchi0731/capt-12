from __future__ import annotations

import json

import numpy as np
import pytest

from capt12.certification.artifact import hash_json, make_certificate, verify_certificate
from capt12.certification.robust import verify_robust_channel
from capt12.confidence.boxes import ConfidenceBox, cp_box, dp_aware_box
from capt12.mechanisms.baselines import common_cover
from capt12.mechanisms.lp import SolverInfo
from capt12.privacy.adjacency import Group, build_adjacency


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
