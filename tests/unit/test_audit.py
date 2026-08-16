from __future__ import annotations

import math

import numpy as np
import pandas as pd

from capt12.audit.lower import LowerAuditResult, lower_audit
from capt12.pipeline import _audit_certificate_comparison
from capt12.privacy.adjacency import Group, build_adjacency


def _audit_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    attack = pd.DataFrame(
        {
            "group": [("a",), ("a",), ("b",), ("b",)],
            "output": [0, 1, 0, 1],
            "context": ["x", "x", "x", "x"],
        }
    )
    test = pd.DataFrame(
        {
            "group": [("a",), ("a",), ("b",), ("b",)],
            "output": [0, 0, 1, 1],
            "context": ["x", "x", "x", "x"],
        }
    )
    return attack, test


def test_lower_audit_allocates_alpha_across_all_two_sided_bounds() -> None:
    attack, test = _audit_frames()
    result = lower_audit(
        attack,
        test,
        group_col="group",
        output_col="output",
        context_cols=["context"],
        alpha=0.05,
    )
    assert result.bounds_tested == 2 * result.events_tested
    assert np.isclose(result.bounds_tested * result.per_bound_alpha, 0.05)


def test_lower_audit_family_does_not_grow_from_test_only_values() -> None:
    attack, test = _audit_frames()
    original = lower_audit(
        attack,
        test,
        group_col="group",
        output_col="output",
        context_cols=["context"],
    )
    augmented = pd.concat(
        [
            test,
            pd.DataFrame(
                {"group": [("test-only",)], "output": [99], "context": ["test-only"]}
            ),
        ],
        ignore_index=True,
    )
    repeated = lower_audit(
        attack,
        augmented,
        group_col="group",
        output_col="output",
        context_cols=["context"],
    )
    assert repeated.events_tested == original.events_tested
    assert repeated.candidate_events == original.candidate_events
    assert repeated.candidate_groups == original.candidate_groups
    assert repeated.candidate_contexts == original.candidate_contexts


def test_witness_comparison_requires_witness_path_and_bridge() -> None:
    groups = [
        Group("a+b", ("0", "0")),
        Group("a+b", ("1", "0")),
        Group("a+b", ("1", "1")),
    ]
    adjacency = build_adjacency(
        groups,
        epsilon_by_attr={"a": 0.2, "b": 0.7},
    )
    no_witness = LowerAuditResult(0.0, None, 1)
    gap, upper, comparable = _audit_certificate_comparison(
        no_witness,
        profile="a+b",
        groups=groups,
        adjacency=adjacency,
        config={},
        universal_upper=None,
    )
    assert gap is None
    assert math.isnan(upper)
    assert not comparable

    witness = LowerAuditResult(
        0.1,
        {
            "left_group_values": ["0", "0"],
            "right_group_values": ["1", "1"],
            "context": "all",
        },
        1,
    )
    gap, upper, comparable = _audit_certificate_comparison(
        witness,
        profile="a+b",
        groups=groups,
        adjacency=adjacency,
        config={
            "audit_population_assumption": "stationary",
            "audit_bridge_evidence": "externally documented stationarity assumption",
        },
        universal_upper=None,
    )
    assert np.isclose(upper, 0.9)
    assert np.isclose(gap, 0.8)
    assert comparable

    disconnected = [groups[0], groups[-1]]
    gap, upper, comparable = _audit_certificate_comparison(
        witness,
        profile="a+b",
        groups=disconnected,
        adjacency=build_adjacency(disconnected),
        config={
            "audit_population_assumption": "stationary",
            "audit_bridge_evidence": "external assumption",
        },
        universal_upper=None,
    )
    assert gap is None
    assert math.isnan(upper)
    assert not comparable

    gap, upper, comparable = _audit_certificate_comparison(
        no_witness,
        profile="a+b",
        groups=[],
        adjacency=[],
        config={},
        universal_upper=0.0,
    )
    assert np.isclose(gap, 0.0)
    assert np.isclose(upper, 0.0)
    assert comparable
