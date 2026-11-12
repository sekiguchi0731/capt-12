from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from capt12.comparison.utility import _cluster_bootstrap_mean, evaluate_frozen_ctr_utility
from capt12.evaluation.released_ranking import ReleasedRankingAccumulator


def _enumerate(channels, scores, tokens, contexts, labels):
    auc, ap = 0.0, 0.0
    for outputs in itertools.product(range(len(next(iter(scores.values())))), repeat=len(tokens)):
        probability = np.prod(
            [channels[c][z, o] for c, z, o in zip(contexts, tokens, outputs, strict=True)]
        )
        released = [scores[c][o] for c, o in zip(contexts, outputs, strict=True)]
        auc += probability * roc_auc_score(labels, released)
        ap += probability * average_precision_score(labels, released)
    return auc, ap


def test_ctr_auc_counterexample_and_logloss_unchanged() -> None:
    q = np.array([[0.5, 0, 0.5], [0, 1, 0], [0, 0, 1]])
    scores, tokens, labels = np.array([0.2, 0.4, 0.8]), np.array([0, 1]), np.array([1, 0])
    value = evaluate_frozen_ctr_utility(
        q,
        tokens,
        None,
        labels,
        scores,
        np.array(["a", "b"]),
        bootstrap_replicates=2,
        ranking_mc_replicates=4096,
    )
    assert value.mean_score_roc_auc == 1
    assert value.roc_auc == 0.5
    assert value.pr_auc == pytest.approx(0.75, abs=0.015)
    expected_loss = (-0.5 * np.log(0.2) - 0.5 * np.log(0.8) - np.log(0.6)) / 2
    assert value.empirical_log_loss == pytest.approx(expected_loss, abs=1e-14)
    assert set(value.released_ranking.ap_mc_values) == {0.5, 1.0}


def test_mean_score_auc_can_also_underestimate_released_auc() -> None:
    q = np.array([[0.1, 0, 0.9], [0, 1, 0], [0, 0, 1]])
    value = evaluate_frozen_ctr_utility(
        q,
        np.array([0, 1]),
        None,
        np.array([1, 0]),
        np.array([0.01, 0.55, 0.6]),
        np.array(["a", "b"]),
        bootstrap_replicates=2,
    )
    assert value.mean_score_roc_auc == 0
    assert value.roc_auc == pytest.approx(0.9)


def test_cross_context_auc_and_finite_ap_match_enumeration_with_ties() -> None:
    channels = {"a": np.array([[0.7, 0.3], [0.2, 0.8]]), "b": np.array([[0.4, 0.6], [0.9, 0.1]])}
    scores = {"a": np.array([0.2, 0.8]), "b": np.array([0.5, 0.8])}
    tokens, contexts, labels = (
        np.array([0, 1, 0, 1]),
        np.array(["a", "a", "b", "b"]),
        np.array([1, 0, 0, 1]),
    )
    exact_auc, exact_ap = _enumerate(channels, scores, tokens, contexts, labels)
    accumulator = ReleasedRankingAccumulator(replicates=12000, seed=31)
    for context in ("a", "b"):
        mask = contexts == context
        accumulator.add_context(channels[context], scores[context], tokens[mask], labels[mask])
    result = accumulator.finish()
    assert result.roc_auc == pytest.approx(exact_auc, abs=1e-14)
    assert result.average_precision == pytest.approx(exact_ap, abs=0.01)
    assert result.ap_mc_se == pytest.approx(result.ap_mc_std / np.sqrt(12000))


def test_finite_sample_ap_is_not_ap_of_fractional_expected_counts() -> None:
    accumulator = ReleasedRankingAccumulator(replicates=12000, seed=8)
    accumulator.add_context(
        np.full((2, 2), 0.5), np.array([0.2, 0.8]), np.array([0, 1]), np.array([1, 0])
    )
    result = accumulator.finish()
    pooled_ap = average_precision_score([1, 1, 0, 0], [0.2, 0.8, 0.2, 0.8], sample_weight=[0.5] * 4)
    assert pooled_ap == 0.5
    assert result.roc_auc == 0.5
    assert result.average_precision == pytest.approx(0.625, abs=0.01)


def test_deterministic_channels_and_constant_score_ties_have_no_sampling_error() -> None:
    tokens, labels = np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])
    for scores, expected_auc, expected_ap in (
        (np.array([0.1, 0.9]), 1, 1),
        (np.full(2, 0.5), 0.5, 0.5),
    ):
        acc = ReleasedRankingAccumulator(replicates=5)
        acc.add_context(np.eye(2), scores, tokens, labels)
        result = acc.finish()
        assert result.roc_auc == expected_auc
        assert result.average_precision == expected_ap
        assert result.ap_mc_se == 0


def test_one_class_splits_and_non_binary_fraction_ties() -> None:
    for label, expected_ap in ((0, 0), (1, 1)):
        acc = ReleasedRankingAccumulator()
        acc.add_context(
            np.full((2, 2), 0.5), np.array([0.1, 0.9]), np.array([0, 1]), np.repeat(label, 2)
        )
        result = acc.finish()
        assert np.isnan(result.roc_auc)
        assert result.average_precision == expected_ap
    acc = ReleasedRankingAccumulator(replicates=5)
    acc.add_context(
        np.ones((1, 1)), np.array([0.4]), np.zeros(5, dtype=int), np.array([1, 1, 0, 0, 0])
    )
    result = acc.finish()
    assert result.average_precision == pytest.approx(0.4)
    assert result.ap_mc_se == 0


def test_streamed_logloss_teacher_and_distortion_match_per_display_sums() -> None:
    rng = np.random.default_rng(41)
    channels = {context: rng.dirichlet(np.ones(4), size=4) for context in ("a", "b")}
    scores = {context: rng.uniform(0.1, 0.9, size=4) for context in channels}
    costs = {context: rng.uniform(0, 2, size=(4, 4)) for context in channels}
    tokens = np.array([0, 1, 2, 3, 0, 2])
    contexts = np.array(["a", "a", "b", "b", "a", "b"])
    labels = np.array([0, 1, 0, 1, 1, 0])
    losses, teachers, distortions = [], [], []
    for token, context, label in zip(tokens, contexts, labels, strict=True):
        q, p = channels[context][token], scores[context]
        losses.append(q @ (-label * np.log(p) - (1 - label) * np.log1p(-p)))
        teacher = p[token] * np.log(p[token] / p) + (1 - p[token]) * np.log(
            (1 - p[token]) / (1 - p)
        )
        teachers.append(q @ teacher)
        distortions.append(q @ costs[context][token])
    actual = evaluate_frozen_ctr_utility(
        channels,
        tokens,
        contexts,
        labels,
        scores,
        np.arange(6),
        distortion_cost=costs,
        bootstrap_replicates=2,
    )
    np.testing.assert_allclose(actual.row_log_loss, losses, atol=1e-14)
    assert actual.teacher_kl == pytest.approx(np.mean(teachers), abs=1e-14)
    assert actual.distortion_objective == pytest.approx(np.mean(distortions), abs=1e-14)


def test_release_histograms_conserve_large_group_counts_and_are_reproducible() -> None:
    q = np.array([[0.8, 0.2], [0.8, 0.2]])

    def evaluate():
        acc = ReleasedRankingAccumulator(replicates=8, seed=19)
        acc.add_context(
            q,
            np.array([0.2, 0.8]),
            np.zeros(5000, dtype=int),
            np.arange(5000) % 2,
            source_groups=np.zeros(2, dtype=int),
        )
        np.testing.assert_array_equal(acc.sampled_positive[0].sum(1), 2500)
        np.testing.assert_array_equal(acc.sampled_negative[0].sum(1), 2500)
        return acc.finish()

    first, second = evaluate(), evaluate()
    np.testing.assert_array_equal(first.ap_mc_values, second.ap_mc_values)
    assert first.roc_auc == 0.5


def test_invalid_channel_or_false_source_equivalence_is_rejected() -> None:
    acc = ReleasedRankingAccumulator()
    with pytest.raises(ValueError, match="stochastic"):
        acc.add_context(
            np.full((2, 2), 0.2), np.array([0.1, 0.9]), np.array([0, 1]), np.array([0, 1])
        )
    with pytest.raises(ValueError, match="different release"):
        acc.add_context(
            np.eye(2),
            np.array([0.1, 0.9]),
            np.array([0, 1]),
            np.array([0, 1]),
            source_groups=np.zeros(2),
        )
    with pytest.raises(ValueError, match="replicates"):
        ReleasedRankingAccumulator(replicates=1)


def test_aggregated_cluster_bootstrap_matches_rowwise_resampling() -> None:
    values = np.array([0.1, 0.3, 0.2, 0.9, 0.5])
    ids = np.array(["a", "a", "b", "c", "c"])
    rng = np.random.default_rng(7)
    means = []
    for _ in range(100):
        sampled = rng.choice(np.unique(ids), size=3, replace=True)
        selected = np.concatenate([np.flatnonzero(ids == value) for value in sampled])
        means.append(values[selected].mean())
    np.testing.assert_allclose(
        _cluster_bootstrap_mean(values, ids, replicates=100, seed=7),
        np.quantile(means, [0.025, 0.975]),
    )


def test_native_capt_and_global_ldp_use_released_rankings() -> None:
    from capt12.experiments.context_global_token_ldp import _evaluate_global_channel
    from capt12.experiments.context_stratified import _evaluate_test

    q = np.array([[0.5, 0, 0.5], [0, 1, 0], [0, 0, 1]])
    p = np.array([0.2, 0.4, 0.8])
    frame = pd.DataFrame({"context": ["c", "c"]})
    objective = SimpleNamespace(
        context="c", token_probabilities=p, utility_objective="empirical_logloss"
    )
    solution = SimpleNamespace(channel=q)
    cell = SimpleNamespace(
        objective=objective,
        constant_channel=q,
        ldp_solution=solution,
        capt_solution=solution,
        raw_capt_channel=q,
        context_r_pooling_weight=0,
    )
    design = SimpleNamespace(
        name="test",
        label="test",
        assignment=np.arange(3),
        decoder=np.eye(3),
        block_weights=np.ones(3) / 3,
        representation_mode="test",
        representation_objective="test",
    )
    native = _evaluate_test(
        frame,
        np.array([0, 1]),
        np.array([1, 0]),
        [design],
        {"test": [cell]},
        {"test": (solution, solution)},
        context_column="context",
    )
    assert native["ROC_AUC"].eq(0.5).all()
    assert native["mean_score_roc_auc"].eq(1).all()
    assert native["ctr_ranking_definition"].eq("independent_released_scores_v1").all()
    global_metrics = _evaluate_global_channel(
        q,
        frame,
        np.array([0, 1]),
        np.array([1, 0]),
        {"context_levels": np.array(["c"]), "representation_probability_grid": p[:, None]},
        context_column="context",
    )
    assert global_metrics["ROC_AUC"] == 0.5
    assert global_metrics["mean_score_roc_auc"] == 1
    assert len(json.loads(global_metrics["ctr_ap_mc_values"])) == 8
