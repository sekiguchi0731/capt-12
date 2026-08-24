from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from capt12.cli import app
from capt12.config import load_config, parse_csv_list, run_id, validate_config
from capt12.data.contributions import apply_contribution_policy
from capt12.data.loader import assert_disjoint_splits, assert_no_row_overlap
from capt12.data.preprocessing import (
    build_group_histograms,
    build_marginal_histograms,
    expected_tuple_grid,
)
from capt12.data.profiles import assign_simulated_profiles
from capt12.decoders.registry import build_decoder
from capt12.encoders.base import HashEncoder
from capt12.models.reference import ReferenceModel, validate_external_reference_inputs
from capt12.partitions.registry import build_partition
from capt12.privacy.adjacency import (
    Group,
    build_adjacency,
    disconnected_hybrid_components,
    hybrid_epsilon,
)
from capt12.privacy.profile import evaluate_profile_privacy


def test_tuple_adjacency_hybrid_sum() -> None:
    groups = [Group("a+b", (0, 0)), Group("a+b", (1, 0)), Group("a+b", (1, 1))]
    edges = build_adjacency(groups, "tuple_adjacent", epsilon_by_attr={"a": 0.2, "b": 0.7})
    assert {edge.epsilon for edge in edges} == {0.2, 0.7}
    assert np.isclose(hybrid_epsilon((0, 0), (1, 1), ("a", "b"), {"a": 0.2, "b": 0.7}), 0.9)
    disconnected = [Group("a+b", (0, 0)), Group("a+b", (1, 1))]
    assert disconnected_hybrid_components(disconnected, build_adjacency(disconnected)) == 1


def test_list_parser_deduplicates_sorts_and_validates() -> None:
    assert parse_csv_list(" 64, 16,64, 32 ", int, minimum=1, maximum=4096) == [16, 32, 64]
    with pytest.raises(ValueError):
        parse_csv_list("0,2", int, minimum=1)


def test_real_data_point_confidence_and_fake_profile_weighting_are_rejected() -> None:
    assert validate_config({"dataset": "synthetic_theorem4", "confidence": "point"})
    with pytest.raises(ValueError, match="point confidence"):
        validate_config({"dataset": "criteo", "confidence": "point"})
    with pytest.raises(ValueError, match="not identifiable"):
        validate_config({"dataset": "criteo", "profile_weighting": "empirical"})
    with pytest.raises(ValueError, match="user_day_iid"):
        validate_config({"dataset": "criteo", "confidence": "cp_box"})
    with pytest.raises(ValueError, match="merge_to_other"):
        validate_config({"rare_group_policy": "merge_to_other"})


def test_cli_accepts_hyphen_and_underscore_aliases(tmp_path) -> None:
    runner = CliRunner()
    output = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "run-grid",
            "--config",
            "configs/smoke.yaml",
            "--output-dir",
            str(output),
            "--K_list",
            "3, 4,3",
            "--L-list",
            "1,2",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        ["run-grid", "--config", "configs/smoke.yaml", "--output-dir", str(output), "--K-list", "3", "--L_list", "1", "--dry-run"],
    )
    assert result.exit_code == 0, result.output


def test_resolve_run_id_cli_reproduces_provenance_config_hash(monkeypatch) -> None:
    full_sha = "a" * 40
    monkeypatch.setattr("capt12.cli._resolve_git_commit", lambda revision: full_sha)
    result = CliRunner().invoke(
        app,
        [
            "resolve-run-id",
            "abc1234",
            "--config",
            "configs/criteo_context_stratified.yaml",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    config = load_config("configs/criteo_context_stratified.yaml")
    config.update(
        {
            "require_clean_worktree": True,
            "source_worktree_clean": True,
            "source_git_sha": full_sha,
        }
    )
    assert payload["source_git_sha"] == full_sha
    assert payload["run_id"] == run_id(config)
    assert payload["output_path"].endswith(payload["run_id"])


def test_context_cli_overrides_frozen_design_seed(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(config):
        captured.update(config)
        return tmp_path / "seed-run"

    monkeypatch.setattr(
        "capt12.experiments.context_stratified.run_context_stratified_diagnostic",
        fake_run,
    )
    result = CliRunner().invoke(
        app,
        [
            "context-stratified",
            "--config",
            "configs/criteo_context_stratified.yaml",
            "--frozen-design-seed",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["frozen_design_seed"] == 7
    assert captured["epsilon"] == 1.0
    assert captured["context_designs"] == ["joint_kmedoids_cost_medoid_L16"]


def test_context_seed_stability_cli_sorts_seeds(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run(config, seeds):
        captured["config"] = config
        captured["seeds"] = seeds
        return tmp_path / "seed-summary"

    monkeypatch.setattr(
        "capt12.experiments.context_seed_stability.run_context_seed_stability",
        fake_run,
    )
    result = CliRunner().invoke(
        app,
        [
            "context-seed-stability",
            "--config",
            "configs/criteo_context_stratified.yaml",
            "--frozen-design-seeds",
            "7,2,7,4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["seeds"] == [2, 4, 7]
    assert captured["config"]["epsilon"] == 1.0
    assert "sol_seed_stability_review_bundle.zip" in result.output


def test_resolve_run_id_changes_with_frozen_design_seed(monkeypatch) -> None:
    monkeypatch.setattr("capt12.cli._resolve_git_commit", lambda revision: "b" * 40)
    runner = CliRunner()
    payloads = []
    for seed in (0, 1):
        result = runner.invoke(
            app,
            [
                "resolve-run-id",
                "HEAD",
                "--config",
                "configs/criteo_context_stratified.yaml",
                "--frozen-design-seed",
                str(seed),
            ],
        )
        assert result.exit_code == 0, result.output
        payloads.append(json.loads(result.output))
    assert payloads[0]["run_id"] != payloads[1]["run_id"]


def test_splits_reject_day_and_row_overlap() -> None:
    with pytest.raises(ValueError):
        assert_disjoint_splits({"D_model": [1, 2], "D_design": [2, 3]})
    frames = {
        "a": pd.DataFrame({"day_int": [1], "id": ["x"]}),
        "b": pd.DataFrame({"day_int": [1], "id": ["x"]}),
    }
    with pytest.raises(ValueError):
        assert_no_row_overlap(frames, "id")


def test_later_splits_cannot_fit_encoder_partition_or_decoder() -> None:
    frame = pd.DataFrame({"x": [1, 2], "s": [0, 1]})
    with pytest.raises(ValueError):
        HashEncoder(2).fit(frame, ["x"], split_id="D_cert")
    with pytest.raises(ValueError):
        build_partition("frequency_balanced", np.ones(4), 2, split_id="D_cert")
    with pytest.raises(ValueError):
        build_decoder("uniform_within_block", np.array([0, 0, 1, 1]), np.ones(4), split_id="D_test")


def test_one_display_per_uuid_day_contribution() -> None:
    frame = pd.DataFrame({"user": ["u", "u", "u", "v"], "day": [1, 1, 2, 1], "x": range(4)})
    result = apply_contribution_policy(frame, user_col="user", day_col="day", seed=7)
    assert len(result) == 3
    assert result.groupby(["user", "day"]).size().max() == 1
    tokens = np.array([10, 11, 12, 13])
    np.testing.assert_array_equal(tokens[result.index.to_numpy()], result.index + 10)


def test_marginal_rare_group_mass_uses_union_of_rows() -> None:
    frame = pd.DataFrame(
        {
            "a": ["common", "common", "rare"],
            "b": ["common", "rare", "common"],
        }
    )
    hist = build_marginal_histograms(
        frame,
        np.array([0, 1, 0]),
        profile="a+b",
        alphabet_size=2,
        min_group_count=2,
        rare_group_policy="force_cover",
    )
    assert hist.rare_row_indices == frozenset({1, 2})
    assert np.isclose(hist.rare_group_mass, 2 / 3)


def test_unobserved_expected_groups_force_cover_or_fail() -> None:
    domain = pd.DataFrame({"a": ["0", "1"], "b": ["0", "1"], "ctx": ["x", "x"]})
    certificate = pd.DataFrame({"a": ["0", "1"], "b": ["0", "1"], "ctx": ["x", "x"]})
    expected = expected_tuple_grid(domain, "a+b", ["ctx"])
    assert len(expected) == 4
    hist = build_group_histograms(
        certificate,
        np.array([0, 1]),
        profile="a+b",
        context_columns=["ctx"],
        alphabet_size=2,
        min_group_count=1,
        rare_group_policy="force_cover",
        expected_frame=domain,
    )
    assert hist.force_cover
    assert hist.missing_group_count == 2
    with pytest.raises(ValueError, match="unobserved"):
        build_group_histograms(
            certificate,
            np.array([0, 1]),
            profile="a+b",
            context_columns=["ctx"],
            alphabet_size=2,
            rare_group_policy="fail",
            expected_frame=domain,
        )


def test_unobserved_groups_can_be_completed_with_zero_count_simplexes() -> None:
    domain = pd.DataFrame({"a": ["0", "1"], "b": ["0", "1"], "ctx": ["x", "x"]})
    certificate = pd.DataFrame({"a": ["0", "1"], "b": ["0", "1"], "ctx": ["x", "x"]})
    hist = build_group_histograms(
        certificate,
        np.array([0, 1]),
        profile="a+b",
        context_columns=["ctx"],
        alphabet_size=2,
        min_group_count=20,
        rare_group_policy="confidence_box",
        missing_group_policy="full_simplex",
        expected_frame=domain,
    )
    assert not hist.force_cover
    assert hist.missing_group_count == 2
    assert len(hist.groups) == 4
    assert sum(int(counts.sum() == 0) for counts in hist.counts.values()) == 2


def test_full_simplex_policy_requires_observed_rare_confidence_boxes() -> None:
    assert validate_config(
        {
            "rare_group_policy": "confidence_box",
            "missing_group_policy": "full_simplex",
        }
    )
    with pytest.raises(ValueError, match="confidence_box"):
        validate_config(
            {
                "rare_group_policy": "force_cover",
                "missing_group_policy": "full_simplex",
            }
        )


def test_solver_progress_config_requires_valid_types_and_interval() -> None:
    assert validate_config(
        {"solver_verbose": True, "solver_heartbeat_seconds": 30}
    )
    with pytest.raises(ValueError, match="solver_heartbeat_seconds"):
        validate_config({"solver_heartbeat_seconds": 0})
    with pytest.raises(ValueError, match="solver_verbose"):
        validate_config({"solver_verbose": "yes"})


def test_resumable_robust_solver_config_is_strict() -> None:
    config = validate_config(
        {
            "robust_cut_formulation": "shared_support_bounds",
            "resume_cutting_plane": True,
            "cutting_plane_checkpoint_every": 2,
            "context_designs": ["singleton_identity_L64"],
        }
    )
    assert config["cutting_plane_checkpoint_every"] == 2
    with pytest.raises(ValueError, match="robust_cut_formulation"):
        validate_config({"robust_cut_formulation": "unknown"})
    with pytest.raises(ValueError, match="exactly one"):
        validate_config(
            {
                "context_designs": [
                    "joint_kmedoids_cost_medoid_L16",
                    "singleton_identity_L64",
                ]
            }
        )


def test_fixed_seed_reproducible_hash_encoder() -> None:
    frame = pd.DataFrame({"x": ["a", "b", "c"]})
    first = HashEncoder(16, seed=8).fit(frame, ["x"]).transform(frame)
    second = HashEncoder(16, seed=8).fit(frame, ["x"]).transform(frame)
    np.testing.assert_array_equal(first, second)


def test_external_reference_declared_inputs_exclude_protected_columns(tmp_path) -> None:
    loaded = ReferenceModel("external", feature_columns=("z", "protected"), model=object())
    with pytest.raises(ValueError, match="protected input"):
        validate_external_reference_inputs(
            loaded,
            allowed_feature_columns=["z"],
            sensitive_columns=["protected"],
        )

    precomputed = ReferenceModel("precomputed", prediction_column="prediction")
    with pytest.raises(ValueError, match="prediction_manifest_path"):
        validate_external_reference_inputs(
            precomputed,
            allowed_feature_columns=["z"],
            sensitive_columns=["protected"],
        )
    manifest = tmp_path / "prediction-manifest.json"
    manifest.write_text(
        '{"prediction_column":"prediction","feature_columns":["z"]}'
    )
    assert (
        validate_external_reference_inputs(
            precomputed,
            allowed_feature_columns=["z"],
            sensitive_columns=["protected"],
            prediction_manifest_path=manifest,
        ).input_manifest
        is not None
    )


def test_simulated_profile_is_stable_within_user_epoch() -> None:
    frame = pd.DataFrame({"user": ["u", "u", "u"], "day": [1, 1, 2]})
    assigned = assign_simulated_profiles(
        frame,
        profiles=["a", "b"],
        user_col="user",
        epoch_col="day",
        probabilities=[0.4, 0.6],
        seed=9,
    )
    assert assigned.iloc[0] == assigned.iloc[1]
    again = assign_simulated_profiles(
        frame,
        profiles=["a", "b"],
        user_col="user",
        epoch_col="day",
        probabilities=[0.4, 0.6],
        seed=9,
    )
    pd.testing.assert_series_equal(assigned, again)


def test_profile_privacy_modes_are_separate_diagnostics() -> None:
    channels = {"a": np.eye(2), "b": np.array([[0, 1], [1, 0]])}
    counterfactual = evaluate_profile_privacy(
        channels,
        mode="counterfactual",
        common_input_distribution=np.array([0.5, 0.5]),
    )
    assert counterfactual.epsilon == 0
    observational = evaluate_profile_privacy(
        channels,
        mode="observational",
        observational_input_distributions={
            "a": np.array([0.9, 0.1]),
            "b": np.array([0.9, 0.1]),
        },
    )
    assert observational.epsilon > 0
    assert not observational.part_of_main_guarantee
