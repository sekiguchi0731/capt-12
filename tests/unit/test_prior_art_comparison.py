from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from capt12.certification.robust import evaluate_robust_constraints, verify_robust_channel
from capt12.cli import app
from capt12.comparison.artifacts import RESULT_COLUMNS, render_prior_art_figures, validate_results
from capt12.comparison.calibration import (
    calibrate_common_cover,
    design_cover_distribution,
    mix_with_common_cover,
)
from capt12.comparison.certificate import (
    verify_factorized_certificate,
    write_factorized_certificate,
)
from capt12.comparison.contracts import default_method_contracts
from capt12.comparison.empirical_privacy import (
    evaluate_attack_cmi_proxy,
    fit_cross_fitted_attack_family,
)
from capt12.comparison.mass12 import check_mass_operational_bounds, fit_mass12_finite
from capt12.comparison.pbp import solve_pbp_common_nominal, solve_pbp_oracle
from capt12.comparison.utility import (
    evaluate_frozen_ctr_utility,
    paired_cluster_bootstrap_difference,
)
from capt12.confidence.boxes import ConfidenceBox
from capt12.config import validate_config
from capt12.privacy.adjacency import AdjacentPair, Group, build_adjacency, hybrid_epsilon


def _point_boxes() -> tuple[dict[str, ConfidenceBox], list[AdjacentPair]]:
    left = np.array([1.0, 0.0, 0.0])
    right = np.array([0.0, 1.0, 0.0])
    boxes = {
        "left": ConfidenceBox(left, left, left, "point", 1.0),
        "right": ConfidenceBox(right, right, right, "point", 1.0),
    }
    adjacency = [
        AdjacentPair("left", "right", 1.0, ("a",)),
        AdjacentPair("right", "left", 1.0, ("a",)),
    ]
    return boxes, adjacency


def test_fixed_channel_upper_epsilon_zero_denominators_and_ordered_edges() -> None:
    boxes, adjacency = _point_boxes()
    channel = np.array(
        [
            [0.8, 0.2, 0.0],
            [0.4, 0.6, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    result = verify_robust_channel(channel, boxes, adjacency)
    assert np.isclose(result.realized_epsilon, math.log(3.0))
    evaluations = evaluate_robust_constraints(channel, boxes, adjacency)
    assert len(evaluations) == 2 * channel.shape[1]
    impossible = [value for value in evaluations if value.output_block == 2]
    assert all(value.maximum == value.minimum == 0 for value in impossible)
    assert all(value.realized_epsilon == 0.0 for value in impossible)

    identity = np.eye(3)
    assert verify_robust_channel(identity, boxes, adjacency).realized_epsilon == math.inf


def test_cover_mixing_endpoints_and_certified_bisection() -> None:
    boxes, adjacency = _point_boxes()
    raw = np.eye(3)
    cover = design_cover_distribution(np.array([4.0, 3.0, 3.0]), floor=1e-6)
    assert np.allclose(mix_with_common_cover(raw, cover, 0), raw)
    endpoint = mix_with_common_cover(raw, cover, 1)
    assert np.allclose(endpoint, cover[None, :])
    assert np.isclose(verify_robust_channel(endpoint, boxes, adjacency).realized_epsilon, 0)

    calibration = calibrate_common_cover(
        raw,
        cover,
        boxes,
        adjacency,
        target_epsilon=1.0,
        bisection_tolerance=1e-8,
        input_weights=np.ones(3),
        cost=1 - np.eye(3),
    )
    assert 0 < calibration.mixing_weight < 1
    assert calibration.independent_verification.realized_epsilon <= 1.0
    assert calibration.pre_verification.realized_epsilon == math.inf
    assert calibration.post_utility is not None
    assert calibration.provenance()["wrapper"] == "common-cover robust calibration"


def test_hybrid_path_budget_is_a_sum() -> None:
    assert np.isclose(
        hybrid_epsilon((0, 0), (1, 1), ("a", "b"), {"a": 0.2, "b": 0.7}),
        0.9,
    )


def test_pbp_oracle_is_non_deployable_and_common_channel_is_stochastic() -> None:
    profiles = {"left": np.array([0.8, 0.2]), "right": np.array([0.2, 0.8])}
    adjacency = [
        AdjacentPair("left", "right", 1.0),
        AdjacentPair("right", "left", 1.0),
    ]
    cost = 1 - np.eye(2)
    oracle = solve_pbp_oracle(cost, profiles, adjacency)
    assert oracle.channels is not None
    assert oracle.uses_sensitive_value_online
    assert not oracle.deployable_under_capt
    assert oracle.nominal_max_violation is not None
    assert oracle.nominal_max_violation <= 1e-8
    for channel in oracle.channels.values():
        assert np.all(channel >= 0)
        assert np.allclose(channel.sum(axis=1), 1)

    common = solve_pbp_common_nominal(cost, np.array([0.5, 0.5]), profiles, adjacency)
    assert common.channel is not None
    assert np.allclose(common.channel.sum(axis=1), 1)
    with pytest.raises(ValueError, match="D_design"):
        solve_pbp_common_nominal(
            cost,
            np.array([0.5, 0.5]),
            profiles,
            adjacency,
            source_split="D_cert",
        )


def test_mass12_channel_is_exact_stochastic_and_has_no_sensitive_inference_input() -> None:
    tokens = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    sensitive = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    decoder = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
    cost = 1 - np.eye(4)
    first = fit_mass12_finite(
        tokens,
        None,
        sensitive,
        decoder,
        cost,
        loss_m=0.1,
        seed=7,
        epochs=3,
    )
    second = fit_mass12_finite(
        tokens,
        None,
        sensitive,
        decoder,
        cost,
        loss_m=0.1,
        seed=7,
        epochs=3,
    )
    assert np.array_equal(first.logits, second.logits)
    assert "sensitive" not in inspect.signature(first.block_probabilities).parameters
    assert "sensitive" not in inspect.signature(first.channel).parameters
    channel = first.channel("all")
    assert np.all(channel >= 0)
    assert np.allclose(channel.sum(axis=1), 1)
    assert np.allclose(first.block_probabilities(0).sum(), 1)
    assert first.provenance()["official_code_reused"] is False
    assert first.display_name == "MaSS-12 (finite-output adaptation)"
    frozen = fit_mass12_finite(
        tokens,
        np.array(["seen"] * len(tokens)),
        sensitive,
        decoder,
        cost,
        loss_m=0.1,
        seed=7,
        epochs=1,
        frozen_context_values=("seen", "unseen"),
    )
    assert frozen.context_values == ("seen", "unseen")
    assert np.allclose(frozen.channel("unseen").sum(axis=1), 1)
    with pytest.raises(ValueError, match="D_design"):
        fit_mass12_finite(
            tokens,
            None,
            sensitive,
            decoder,
            cost,
            source_split="D_test",
            epochs=1,
        )


def test_mass_operational_bounds_use_nats_and_constraint_directions() -> None:
    sensitive = {"s": np.array([0, 0, 1, 1])}
    useful = {"u": np.array([0, 1, 0, 1])}
    diagnostics = check_mass_operational_bounds(sensitive, useful, {"s": 0.0}, {"u": 0.5})
    assert np.isclose(diagnostics["entropy:u"], math.log(2))
    with pytest.raises(ValueError, match=r"exceeds H\(U\)"):
        check_mass_operational_bounds(sensitive, useful, {"s": 0.0}, {"u": 1.0})
    with pytest.raises(ValueError, match="nonnegative"):
        check_mass_operational_bounds(sensitive, useful, {"s": -0.1}, {"u": 0.0})


@pytest.mark.parametrize("epochs", [1, 4])
def test_mass_history_csv_records_both_update_stages_and_the_returned_model(
    tmp_path: Path, epochs: int
) -> None:
    from capt12.experiments.prior_art_comparison import _mass_history

    tokens = np.tile(np.arange(4), 6)
    contexts = np.array(["a"] * 20 + ["b"] * 4)
    sensitive = tokens // 2
    useful = tokens % 2
    cost = 1 - np.eye(4)
    model = fit_mass12_finite(
        tokens,
        contexts,
        sensitive,
        np.eye(4),
        cost,
        useful_labels=useful,
        loss_m=0.01,
        loss_n=0.4,
        privacy_weight=4.0,
        utility_weight=1.7,
        ctr_weight=1.7,
        temperature=0.7,
        seed=1,
        epochs=epochs,
    )
    output_rows = np.stack(
        [model.channel(context)[token] for token, context in zip(tokens, contexts, strict=True)]
    )

    def measured_conditional_mi(labels: np.ndarray) -> float:
        # Independently enumerate the joint label/output table from the final
        # token channel, rather than reusing the training MI implementation.
        information = 0.0
        for context in np.unique(contexts):
            in_context = contexts == context
            joint = np.stack(
                [
                    output_rows[in_context & (labels == label)].sum(axis=0) / len(tokens)
                    for label in np.unique(labels)
                ]
            )
            denominator = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
            positive = joint > 0
            information += float(
                np.sum(
                    joint[positive]
                    * np.log(joint[positive] * in_context.mean() / denominator[positive])
                )
            )
        return information

    sensitive_mi = measured_conditional_mi(sensitive)
    useful_mi = measured_conditional_mi(useful)
    distortion = float(np.sum(output_rows * cost[tokens], axis=1).mean())
    sensitive_violation = max(sensitive_mi - 0.01, 0.0)
    useful_violation = max(0.4 - useful_mi, 0.0)
    expected_loss = (
        1.7 * distortion
        + 4.0 * (sensitive_violation**2 + sensitive_violation)
        + 1.7 * (useful_violation**2 + useful_violation)
    )
    final = model.history[-1]
    assert final.post_update_sensitive_mi_nats["sensitive"] == pytest.approx(sensitive_mi)
    assert final.post_update_useful_mi_nats["useful"] == pytest.approx(useful_mi)
    assert final.post_update_distortion == pytest.approx(distortion)
    assert final.post_update_loss == pytest.approx(expected_loss)
    assert not np.isclose(final.loss, final.post_update_loss)
    for previous, current in zip(model.history[:-1], model.history[1:], strict=True):
        assert previous.post_update_loss == current.loss
        assert previous.post_update_distortion == current.distortion
        assert previous.post_update_sensitive_mi_nats == current.sensitive_mi_nats
        assert previous.post_update_useful_mi_nats == current.useful_mi_nats

    path = tmp_path / "mass-history.csv"
    _mass_history(model, path)
    frame = pd.read_csv(path)
    assert set(frame["mass_ctr_weight"]) == {1.7}
    assert frame["epoch"].tolist() == list(range(1, epochs + 1))
    for field in ("loss", "distortion", "sensitive_mi_nats", "useful_mi_nats"):
        assert frame[field].equals(frame[f"pre_update_{field}"])
        assert f"post_update_{field}" in frame
    assert frame.iloc[-1]["post_update_loss"] == pytest.approx(expected_loss)
    assert frame.iloc[-1]["post_update_distortion"] == pytest.approx(distortion)
    assert json.loads(frame.iloc[-1]["post_update_sensitive_mi_nats"])[
        "sensitive"
    ] == pytest.approx(sensitive_mi)
    assert json.loads(frame.iloc[-1]["post_update_useful_mi_nats"])["useful"] == pytest.approx(
        useful_mi
    )


def test_attack_family_is_frozen_on_attack_train_and_test_is_evaluation_only() -> None:
    groups = np.repeat(np.array(["d0", "d1", "d2", "d3"]), 2)
    labels = np.tile(np.array([0, 1]), 4)
    baseline = np.repeat(np.array(["b0", "b1", "b0", "b1"]), 2)[:, None]
    output = np.c_[baseline[:, 0], labels.astype(str)]
    family = fit_cross_fitted_attack_family(
        baseline,
        output,
        labels,
        groups,
        regularization_grid=(0.1, 1.0),
        fold_count=2,
    )
    assert family.source_split == "D_attack_train"
    proxy = evaluate_attack_cmi_proxy(
        family,
        baseline,
        output,
        labels,
        groups,
        bootstrap_replicates=10,
    )
    assert proxy.metric_name == "cross-fitted attack-CMI proxy"
    assert not proxy.certificate
    assert proxy.zero_clipped_nats >= 0
    with pytest.raises(ValueError, match="D_attack_train"):
        fit_cross_fitted_attack_family(
            baseline, output, labels, groups, fold_count=2, split_name="D_test"
        )
    with pytest.raises(ValueError, match="D_test"):
        evaluate_attack_cmi_proxy(
            family,
            baseline,
            output,
            labels,
            groups,
            bootstrap_replicates=2,
            split_name="D_attack_train",
        )


def test_frozen_ctr_utility_and_paired_cluster_bootstrap_use_d_test_only() -> None:
    tokens = np.array([0, 1, 0, 1])
    labels = np.array([0, 1, 0, 1])
    groups = np.array(["d0", "d0", "d1", "d1"])
    probabilities = np.array([0.1, 0.9])
    identity = evaluate_frozen_ctr_utility(
        np.eye(2),
        tokens,
        None,
        labels,
        probabilities,
        groups,
        bootstrap_replicates=10,
    )
    cover = evaluate_frozen_ctr_utility(
        np.full((2, 2), 0.5),
        tokens,
        None,
        labels,
        probabilities,
        groups,
        bootstrap_replicates=10,
    )
    assert identity.excess_log_loss == pytest.approx(0)
    paired = paired_cluster_bootstrap_difference(
        cover.row_log_loss,
        identity.row_log_loss,
        groups,
        bootstrap_replicates=10,
    )
    assert paired.mean > 0
    with pytest.raises(ValueError, match="D_test"):
        evaluate_frozen_ctr_utility(
            np.eye(2),
            tokens,
            None,
            labels,
            probabilities,
            groups,
            bootstrap_replicates=2,
            split_name="D_design",
        )


def _plot_frame() -> pd.DataFrame:
    defaults: dict[str, object] = {name: np.nan for name in RESULT_COLUMNS}
    defaults.update(
        {
            "worst_witness": "{}",
            "exclusion_reason": "",
            "channel_sha256": "a" * 64,
            "certificate_path": "certificate.json",
            "lower_audit_path": "audit.json",
            "certificate_valid": True,
            "constant_channel": False,
            "formal_comparable": True,
            "deployable_under_capt": True,
            "uses_sensitive_value_online": False,
            "oracle": False,
            "target_epsilon": 1.0,
            "K": 4096,
            "L": 16,
            "seed": 0,
            "raw_upper_epsilon": 1.2,
            "certified_upper_epsilon": 1.0,
            "cover_lambda": 0.2,
            "cover_row_tv": 0.1,
            "violating_constraint_count": 0,
            "empirical_attack_cmi_raw_nats": 0.02,
            "empirical_attack_cmi_zero_clipped_nats": 0.02,
            "empirical_attack_cmi_ci95_low_nats": 0.01,
            "empirical_attack_cmi_ci95_high_nats": 0.03,
            "lower_audit_epsilon": 0.5,
            "empirical_log_loss": 0.4,
            "excess_ctr_log_loss": 0.01,
            "excess_ctr_log_loss_ci95_low": 0.009,
            "excess_ctr_log_loss_ci95_high": 0.011,
            "teacher_kl": 0.02,
            "hybrid_objective": 0.03,
            "roc_auc": 0.7,
            "pr_auc": 0.2,
            "distortion_objective": 0.04,
            "paired_excess_log_loss_difference_vs_capt": 0.001,
            "paired_difference_ci95_low": 0.0005,
            "paired_difference_ci95_high": 0.0015,
            "mass_m": 0.1,
            "mass_n": 0.0,
            "mass_privacy_weight": 1.0,
            "mass_utility_weight": 1.0,
            "mass_ctr_weight": 0.0,
            "mass_temperature": 1.0,
            "mass_infonce_weight": 0.0,
        }
    )
    rows = []
    names = {
        "capt": "CAPT-12",
        "optimal_ldp": "optimal LDP",
        "mass12_calibrated": "MaSS-12 + certified cover calibration",
    }
    for cost in ("teacher_kl", "empirical_logloss", "hybrid_logloss_kl"):
        for method, display in names.items():
            row = dict(defaults)
            row.update({"cost": cost, "method": method, "display_name": display})
            rows.append(row)
    raw = dict(defaults)
    raw.update(
        {
            "cost": "teacher_kl",
            "method": "mass12_raw",
            "display_name": "MaSS-12 (finite-output adaptation)",
            "certificate_valid": False,
            "formal_comparable": False,
            "certified_upper_epsilon": np.nan,
            "certificate_path": "",
        }
    )
    rows.append(raw)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def test_results_claim_gate_and_deterministic_plot_bytes(tmp_path: Path) -> None:
    frame = _plot_frame()
    assert len(validate_results(frame)) == len(frame)
    invalid = frame.copy()
    invalid.loc[0, "certificate_valid"] = False
    with pytest.raises(ValueError, match="formal-comparable"):
        validate_results(invalid)

    contracts = default_method_contracts()
    first = render_prior_art_figures(frame, contracts, tmp_path / "first")
    second = render_prior_art_figures(frame, contracts, tmp_path / "second")
    first_hashes = {path.name: path.read_bytes() for path in first}
    second_hashes = {path.name: path.read_bytes() for path in second}
    assert first_hashes == second_hashes


def test_legacy_results_recover_the_actual_ctr_weight_and_resume_keys_distinguish_it() -> None:
    from capt12.experiments.prior_art_comparison import _result_key

    legacy = _plot_frame().drop(columns="mass_ctr_weight")
    legacy["mass_utility_weight"] = 1.7
    restored = validate_results(legacy)
    assert (restored["mass_ctr_weight"] == 1.7).all()
    off = restored.loc[restored["method"] == "mass12_calibrated"].iloc[0].to_dict()
    on = {**off, "mass_ctr_weight": 0.0}
    assert _result_key(off) != _result_key(on)


def test_comparison_errorbar_preserves_interval_outside_point_estimate() -> None:
    import matplotlib.pyplot as plt

    from capt12.comparison.artifacts import _errorbar

    rows = pd.DataFrame({
        "x": [1.0], "excess_ctr_log_loss": [0.3],
        "excess_ctr_log_loss_ci95_low": [0.1],
        "excess_ctr_log_loss_ci95_high": [0.2], "display_name": ["CAPT"],
    })
    fig, axis = plt.subplots()
    try:
        _errorbar(axis, rows, "x", "excess_ctr_log_loss", "capt")
        np.testing.assert_allclose(axis.collections[0].get_segments()[0], [[1.0, 0.1], [1.0, 0.2]])
        np.testing.assert_allclose(axis.lines[-1].get_ydata(), [0.3])
    finally:
        plt.close(fig)


def test_factorized_comparison_certificate_reconstructs_exact_q(tmp_path: Path) -> None:
    groups = [Group("a", ("left",)), Group("a", ("right",))]
    left = np.array([1.0, 0.0, 0.0])
    right = np.array([0.0, 1.0, 0.0])
    boxes = {
        groups[0].key(): ConfidenceBox(left, left, left, "point", 1.0),
        groups[1].key(): ConfidenceBox(right, right, right, "point", 1.0),
    }
    adjacency = build_adjacency(groups, "tuple_adjacent", 1.0)
    input_to_block = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    decoder = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    path, written = write_factorized_certificate(
        tmp_path / "certificate.json",
        input_to_block={"all": input_to_block},
        decoder=decoder,
        cover_distribution=np.ones(3) / 3,
        mixing_weights={"all": 0.0},
        boxes={"all": boxes},
        adjacency={"all": adjacency},
        groups={"all": groups},
        adjacency_spec={"mode": "tuple_adjacent", "epsilon": 1.0, "epsilon_by_attr": {}},
        metadata={"method": "toy"},
    )
    assert written.valid
    verified = verify_factorized_certificate(path)
    assert verified.valid
    assert verified.realized_epsilon == pytest.approx(0)


def test_external_source_commit_and_license_metadata() -> None:
    path = Path("reports/external_provenance_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    mass = next(
        item
        for item in payload["sources"]
        if item["name"] == "MaSS official implementation"
    )
    assert mass["repository_url"] == "https://github.com/jpmorganchase/MaSS"
    assert mass["commit_sha"] == "6fbe9be1ff155b3fdd2677de7977d801d10000b5"
    assert mass["license"] == "Apache-2.0"


def test_prior_art_config_and_cli_reject_ordinary_grid_and_dispatch_dedicated_runner(
    monkeypatch,
) -> None:
    config = validate_config(
        {
            "dataset": "synthetic",
            "K": 4096,
            "L": 16,
            "prior_art_comparison": True,
            "mass_m_list": "0,0.1",
            "mass_n_list": "0",
            "mass_privacy_weight_list": "0.1,1",
            "mass_utility_weight_list": "1,10",
            "mass_seed_list": "0,1",
            "mass_epochs": 2,
            "mass_output_mode": "finite_block",
            "mass_temperature_list": "0.5,1",
        }
    )
    assert config["mass_m_list"] == [0.0, 0.1]
    assert config["mass_temperature_list"] == [0.5, 1.0]

    ordinary = CliRunner().invoke(
        app,
        [
            "run-grid",
            "--config",
            "configs/smoke.yaml",
            "--mass_m_list",
            "0,0.1",
            "--mass-n-list",
            "0",
            "--mass_privacy_weight_list",
            "0.1,1",
            "--mass-utility-weight-list",
            "1,10",
            "--mass_seed_list",
            "0,1",
            "--mass-epochs",
            "2",
            "--mass_output_mode",
            "finite_block",
            "--mass-temperature-list",
            "0.5,1",
            "--dry-run",
        ],
    )
    assert ordinary.exit_code != 0
    assert "cannot run through run-grid" in ordinary.output

    captured = {}

    def fake_comparison(config, *, phase, resume):
        captured.update({"config": config, "phase": phase, "resume": resume})
        return Path("outputs/fake-prior-art-run")

    monkeypatch.setattr(
        "capt12.experiments.prior_art_comparison.run_prior_art_comparison",
        fake_comparison,
    )
    dedicated = CliRunner().invoke(
        app,
        [
            "prior-art-comparison",
            "--config",
            "configs/criteo_prior_art_privacy_matched.yaml",
            "--phase",
            "pilot",
        ],
    )
    assert dedicated.exit_code == 0, dedicated.output
    assert captured["phase"] == "pilot"
    assert captured["resume"] is True
    assert captured["config"]["prior_art_comparison"] is True


@pytest.mark.parametrize(
    ("arguments", "k", "l_count"),
    [
        (["--K", "32", "--L", "7"], 32, 7),
        (["--input-bits", "5", "--output-bits", "5"], 32, 32),
        (["--K", "3"], 3, 3),
    ],
)
def test_comparison_cli_shares_counts_and_decoder_with_both_methods(
    monkeypatch, arguments: list[str], k: int, l_count: int
) -> None:
    from capt12.experiments.prior_art_comparison import _native_config, _phase_values

    captured = {}

    def fake_comparison(config, *, phase, resume):
        captured.update(config)
        return Path("outputs/fake-prior-art-run")

    monkeypatch.setattr(
        "capt12.experiments.prior_art_comparison.run_prior_art_comparison", fake_comparison
    )
    result = CliRunner().invoke(
        app,
        ["prior-art-comparison", "--config", "configs/criteo_prior_art_privacy_matched.yaml", *arguments],
    )
    assert result.exit_code == 0, result.output
    assert (captured["K"], captured["L"]) == (k, l_count)
    native = _native_config(
        captured, Path("outputs/fake-native"), cost="teacher_kl",
        block_count=l_count, seed=0, epsilon=1.0,
    )
    assert (native["K"], native["L"]) == (k, l_count)
    assert native["context_designs"] == captured["context_designs"]
    assert _phase_values(captured, "pilot")[1] == [l_count]
    assert all(contract.output_bits == (k - 1).bit_length()
               for contract in default_method_contracts(token_count=k) if contract.single_output)


def test_comparison_cli_rejects_conflicting_count_and_bits() -> None:
    result = CliRunner().invoke(
        app,
        ["prior-art-comparison", "--config", "configs/criteo_prior_art_privacy_matched.yaml",
         "--K", "32", "--input-bits", "4"],
    )
    assert result.exit_code != 0
    assert "disagree" in result.output
