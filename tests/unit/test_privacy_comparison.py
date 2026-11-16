from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from typer.testing import CliRunner

from capt12.cli import app
from capt12.comparison.attack_metrics import (
    audit_score_buckets,
    binary_attack_metrics,
    compare_pr_curves,
    fpr_at_recall,
    pr_plot_points,
    precision_at_recall,
    precision_recall,
    rare_score_buckets,
)
from capt12.comparison.attribute_inference import _curve, evaluate_attribute_inference
from capt12.comparison.privacy_comparison import _batches, compare_saved_privacy
from capt12.comparison.privacy_simulation import (
    exact_binary_metrics,
    exact_privacy_profile,
    example_channel,
    expected_ctr_logloss,
    output_profiles,
    run_mi_counterexample,
)
from capt12.config import validate_config
from capt12.privacy.adjacency import Group


def test_pr_and_eau_match_independent_weighted_binary_classification():
    # The largest learned score has no observations. Ties remain grouped.
    scores = np.array([-2, 0, 1, 3.0])
    positive = np.array([0, 1, 2, 2])
    negative = np.array([0, 0, 1, 1, 2, 2])
    curve = _curve(scores, positive, negative)
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    observed_scores = scores[np.r_[positive, negative]]
    for prior in (0.1, 0.5, 0.8):
        metrics = binary_attack_metrics(curve, prior)
        weights = np.r_[
            np.full(len(positive), prior / len(positive)),
            np.full(len(negative), (1 - prior) / len(negative)),
        ]
        assert metrics["average_precision"] == pytest.approx(
            average_precision_score(labels, observed_scores, sample_weight=weights)
        )
        threshold = np.log((1 - prior) / prior)
        expected = weights[(observed_scores >= threshold) == labels.astype(bool)].sum()
        assert metrics["pair_eau"] == pytest.approx(expected)
        pr = precision_recall(curve, prior)
        last = pr.iloc[-1]
        assert last.precision == pytest.approx(prior)
        assert np.isnan(pr.iloc[0].precision)


def test_tied_constant_attacker_prior_eau_is_not_balanced_accuracy_for_unequal_prior():
    curve = _curve(np.zeros(2), np.array([0, 1]), np.array([0, 1]))
    for prior in (0.1, 0.5, 0.8):
        value = binary_attack_metrics(curve, prior)
        assert value["pair_eau"] == pytest.approx(max(prior, 1 - prior))
        assert value["accuracy_uplift"] == 0
        assert value["average_precision"] == pytest.approx(prior)
        np.testing.assert_allclose(
            precision_at_recall(curve, np.array([0.01, 0.5, 1]), prior), prior
        )


def test_plot_pr_mixes_roc_decisions_instead_of_linearly_interpolating_precision():
    curve = pd.DataFrame({"tpr": [0, 0.2, 1, 1], "fpr": [0, 0, 0.8, 1]})
    points = pr_plot_points(curve)
    selected = points.recall.between(0.2, 1)
    expected = points.loc[selected, "recall"] / (2 * points.loc[selected, "recall"] - 0.2)
    np.testing.assert_allclose(points.loc[selected, "precision"], expected)
    assert precision_at_recall(curve, np.array([0.6]), 0.5)[0] == pytest.approx(0.6)


def test_interpolation_retains_zero_recall_edges_and_vertical_discontinuities():
    curve = pd.DataFrame({"tpr": [0, 0, 0.2, 0.2, 1, 1], "fpr": [0, 0.2, 0.2, 0.6, 0.8, 1]})
    np.testing.assert_allclose(
        fpr_at_recall(curve, np.array([0.1, 0.2, 0.6, 1])), [0.2, 0.2, 0.7, 0.8]
    )
    assert fpr_at_recall(curve, np.array([np.nextafter(0.2, 1)]))[0] > 0.59
    equal = compare_pr_curves(curve, curve, 0.5)
    assert equal["capt_pr_no_higher"]
    assert equal["max_capt_minus_other_precision"] == 0


def test_pr_order_detects_crossing_and_non_grid_local_violations():
    weak = pd.DataFrame({"tpr": [0, 1], "fpr": [0, 1]})
    strong = pd.DataFrame({"tpr": [0, 1, 1], "fpr": [0, 0, 1]})
    assert compare_pr_curves(weak, strong, 0.5)["capt_pr_no_higher"]
    assert not compare_pr_curves(strong, weak, 0.5)["capt_pr_no_higher"]
    # Violation confined to a recall interval far below a 1001-point grid.
    local = pd.DataFrame({"tpr": [0, 1e-8, 2e-8, 1], "fpr": [0, 0, 2e-8, 1]})
    assert not compare_pr_curves(local, weak, 0.5)["capt_pr_no_higher"]
    # Crossing rather than a total privacy order.
    crossing = pd.DataFrame({"tpr": [0, 0.2, 0.8, 1], "fpr": [0, 0.1, 0.9, 1]})
    assert not compare_pr_curves(weak, crossing, 0.5)["capt_pr_no_higher"]
    assert not compare_pr_curves(crossing, weak, 0.5)["capt_pr_no_higher"]


def test_zero_test_count_never_proves_infinite_privacy_loss():
    frame = pd.DataFrame(
        {
            "positive_successes": [1, 900, 0],
            "positive_trials": [1, 1000, 100],
            "negative_successes": [0, 0, 0],
            "negative_trials": [1, 1000, 100],
        }
    )
    result = audit_score_buckets(frame)
    assert np.isfinite(result.epsilon_lower).all()
    assert not result.true_infinite_epsilon_proven.any()
    assert result.epsilon_lower.iloc[0] == result.epsilon_lower.iloc[2] == 0
    assert result.epsilon_lower.iloc[1] > 3
    assert result.empirical_log_ratio.isna().all()
    np.testing.assert_allclose(result.tail_alpha, 0.05 / 12)
    assert (result.negative_ci_high > 0).all()


def test_score_event_selection_is_independent_of_test_labels():
    scores = np.arange(30, dtype=float)
    first = _curve(scores, np.array([0, 1, 2]), np.array([2, 3, 4]))
    second = _curve(scores, np.array([20, 21, 22]), np.array([22, 23, 24]))
    a, b = [rare_score_buckets(value, 3, 3, 6) for value in (first, second)]
    np.testing.assert_array_equal(a.score, b.score)
    assert len(a) == 6
    np.testing.assert_array_equal(a.score, [29, 28, 27, 2, 1, 0])


def test_exact_mi_counterexample_and_finite_positive_support_variant():
    q = 0.001
    leaky = example_channel(q)
    bounded = example_channel(q, epsilon=1)
    exact = exact_binary_metrics(output_profiles(leaky))
    assert exact["mutual_information_nats"] == pytest.approx(q * np.log(2), abs=1e-15)
    assert exact["exact_epsilon"] == np.inf
    assert exact["exact_optimal_eau"] == pytest.approx(0.5 + q / 2)
    assert exact_binary_metrics(output_profiles(bounded))["exact_epsilon"] == pytest.approx(1)
    assert expected_ctr_logloss(leaky) == pytest.approx(expected_ctr_logloss(bounded), abs=1e-14)
    np.testing.assert_allclose(
        exact_privacy_profile(output_profiles(leaky), np.array([0, 1, 20])), q
    )
    np.testing.assert_allclose(
        exact_privacy_profile(output_profiles(bounded), np.array([1, 5, 20])), 0, atol=1e-14
    )
    positive = example_channel(q, floor_mix=1e-9)
    positive_ref = example_channel(q, epsilon=1, floor_mix=1e-9)
    assert positive.min() > 0
    finite = exact_binary_metrics(output_profiles(positive))
    assert 10 < finite["exact_epsilon"] < 30
    assert finite["mutual_information_nats"] < 0.001
    assert expected_ctr_logloss(positive) == pytest.approx(expected_ctr_logloss(positive_ref))
    empty = exact_binary_metrics(output_profiles(example_channel(0)))
    assert empty["mutual_information_nats"] == empty["exact_epsilon"] == 0


def _saved_fixture(root: Path) -> Path:
    groups = [Group("attribute", (str(a),), "public") for a in range(3)]
    train_groups = groups * 120
    test_groups = groups[:2] * 120  # third group has no independent test support
    train_ids = np.array([f"train-{i}" for i in range(len(train_groups))])
    test_ids = np.array([f"test-{i}" for i in range(len(test_groups))])
    rows = []
    for method, outputs in (("capt", [0, 0, 0]), ("mass12_raw", [0, 1, 2])):
        attack_id = "attack-" + method
        path = root / "runs" / "original" / "attribute_attacks" / attack_id
        result = evaluate_attribute_inference(
            np.tile(outputs, 120),
            train_groups,
            train_ids,
            np.tile(outputs[:2], 120),
            test_groups,
            test_ids,
            expected_groups=groups,
            k=3,
        )
        result.summary["mechanism_identity"] = {"mechanism": method}
        result.save(path)
        rows.append(
            {
                "method": method,
                "display_name": method,
                "cost": "empirical_logloss",
                "seed": 0,
                "K": 3,
                "L": 2,
                "target_epsilon": 1,
                "mass_m": 0.1,
                "mass_seed": 0,
                "empirical_log_loss": 0.3 if method == "capt" else 0.4,
                "attribute_attack_id": attack_id,
                # Simulate an absolute path copied from a DIFFERENT computer.
                "attribute_attack_path": f"/nonexistent/remote/pilot/runs/original/attribute_attacks/{attack_id}",
                "channel_sha256": method,
                "model_sha256": method,
            }
        )
    path = root / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_saved_comparison_uses_common_pairs_and_never_rewrites_source(tmp_path):
    results = _saved_fixture(tmp_path / "source")
    files = [path for path in results.parent.rglob("*") if path.is_file()]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    output = compare_saved_privacy([results], tmp_path / "analysis")
    summary = pd.read_csv(output / "mechanism_metrics.csv")
    assert summary.common_pairs.eq(2).all()
    assert sorted(summary.mean_pair_eau) == [0.5, 1]
    coverage = pd.read_csv(output / "coverage.csv")
    assert coverage.expected_pairs.eq(6).all()
    assert coverage.eligible_pairs.eq(6).all()
    assert coverage.evaluated_pairs.eq(2).all()
    dominance = pd.read_csv(output / "dominance_summary.csv")
    assert dominance.observed_joint_dominance.all()
    assert not dominance.population_dominance_proven.any()
    assert (summary.mean_pair_eau_ci_low <= summary.mean_pair_eau).all()
    assert (summary.mean_pair_eau_ci_high >= summary.mean_pair_eau).all()
    metadata = json.loads((output / "comparison.json").read_text())
    assert not metadata["mechanisms_retrained"] and not metadata["attacks_retrained"]
    assert (output / "figures/rare_events.png").is_file()
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    with pytest.raises(ValueError, match="no common"):
        compare_saved_privacy([results], tmp_path / "empty", min_test_count=1000)
    with pytest.raises(ValueError, match="separate"):
        compare_saved_privacy([results], results.parent)


def test_comparison_cli_and_simulation_artifacts(tmp_path):
    results = _saved_fixture(tmp_path / "source")
    output = tmp_path / "analysis"
    result = CliRunner().invoke(
        app,
        [
            "compare-attribute-privacy",
            "--results",
            str(results),
            "--output-dir",
            str(output),
            "--prior",
            "0.2",
            "--simulation",
        ],
    )
    assert result.exit_code == 0, result.output
    assert pd.read_csv(output / "mechanism_metrics.csv").positive_prior.eq(0.2).all()
    assert (output / "simulation/exact_metrics.csv").is_file()
    metadata = json.loads((output / "simulation/simulation.json").read_text())
    assert metadata["distribution_known"] and not metadata["trained_mass"]
    exact = pd.read_csv(output / "simulation/exact_metrics.csv")
    assert np.isinf(exact.exact_epsilon).any()
    assert (output / "simulation/simulation_pr.pdf").is_file()
    invalid = CliRunner().invoke(
        app,
        [
            "compare-attribute-privacy",
            "--results",
            str(results),
            "--output-dir",
            str(output),
            "--prior",
            "1",
        ],
    )
    assert invalid.exit_code != 0


def test_corrupted_saved_attack_thresholds_and_counts_are_rejected(tmp_path):
    results = _saved_fixture(tmp_path / "source")
    path = results.parent / "runs/original/attribute_attacks/attack-capt/roc.csv"
    curve = pd.read_csv(path, float_precision="round_trip")
    changed = curve.copy()
    changed.loc[1, "threshold"] += 0.01
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="frozen attacker"):
        compare_saved_privacy([results], tmp_path / "analysis")
    changed = curve.copy()
    changed.loc[1, "tp"] += 1
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="counts/rates"):
        compare_saved_privacy([results], tmp_path / "analysis")


def test_different_capt_budgets_are_compared_to_the_same_mass_family(tmp_path):
    results = _saved_fixture(tmp_path / "source")
    frame = pd.read_csv(results)
    second_capt = frame.loc[frame.method.eq("capt")].assign(target_epsilon=0.5)
    pd.concat([frame, second_capt], ignore_index=True).to_csv(results, index=False)
    output = compare_saved_privacy([results], tmp_path / "analysis")
    metadata = json.loads((output / "comparison.json").read_text())
    assert len(metadata["cells"]) == 1
    summary = pd.read_csv(output / "mechanism_metrics.csv")
    assert len(summary) == 3
    assert len(pd.read_csv(output / "dominance_summary.csv")) == 2


def test_large_grid_pages_keep_references_and_include_every_mass_condition():
    items = [{"method": "capt", "id": "capt"}, {"method": "optimal_ldp", "id": "ldp"}]
    items += [{"method": "mass12_raw", "id": str(i)} for i in range(18)]
    pages = _batches(items, lambda item: item["method"])
    assert all(len(page) <= 8 for page in pages)
    assert all({"capt", "ldp"} <= {item["id"] for item in page} for page in pages)
    assert {item["id"] for page in pages for item in page} == {item["id"] for item in items}
    many_references = [{"method": "capt", "id": f"c{i}"} for i in range(10)]
    pages = _batches(many_references + items[2:], lambda item: item["method"])
    assert all(len(page) <= 8 for page in pages)
    assert {item["id"] for page in pages for item in page} == {
        item["id"] for item in many_references + items[2:]
    }


@pytest.mark.parametrize("prior", [0, 1, float("nan")])
def test_invalid_prior_and_simulation_inputs(prior, tmp_path):
    with pytest.raises(ValueError, match="prior"):
        validate_config({"prior_art_comparison": True, "attribute_attack_prior": prior})
    with pytest.raises(ValueError):
        run_mi_counterexample(tmp_path, positive_floor=0)


def test_b200_launcher_dry_run_creates_separate_paired_configs_without_gpu_access(tmp_path):
    from capt12.config import load_config

    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "grid"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/launch_b200_replicates.py"),
            "--dry-run",
            "--gpus",
            "0,1",
            "--seeds",
            "0",
            "--useful-n-list",
            "0.02,0.05,0.1",
            "--utility-weights",
            "1,10",
            "--output-root",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads((output / "launch.json").read_text())
    assert len(plan["jobs"]) == 6
    assert plan["maximum_parallel_jobs"] <= 2
    assert len({job["output"] for job in plan["jobs"]}) == 6
    for job in plan["jobs"]:
        config = load_config(job["config"])
        assert config["comparison_pilot_seeds"] == config["mass_pilot_seed_list"] == [0]
        assert config["mass_pilot_n_list"] == [job["loss_n"]]
        assert config["mass_pilot_utility_weight_list"] == [job["utility_weight"]]
        assert config["mass_pilot_m_list"] == [0, 0.1, 0.5]
        assert config["mass_ctr_weight"] == 0 and config["mass_batch_size"] == 128
        assert job["status"] == "queued"
