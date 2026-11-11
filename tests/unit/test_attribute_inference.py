from __future__ import annotations

import json
from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from capt12.comparison.attribute_inference import (
    ATTACK_FACTORIES,
    LikelihoodRatioAttack,
    attack_fit_mask,
    evaluate_attribute_inference,
    fit_likelihood_ratio,
)
from capt12.config import validate_config
from capt12.privacy.adjacency import Group


def _data(groups, outputs, repeats=80, prefix="train"):
    return (
        np.tile(np.asarray(outputs, dtype=int), repeats),
        list(groups) * repeats,
        np.array([f"{prefix}-{index}" for index in range(repeats * len(groups))]),
    )


def test_nominal_likelihood_ratio_and_token_permutation() -> None:
    positive, negative = np.array([0, 0, 1]), np.array([0, 1, 1, 2])
    model = fit_likelihood_ratio(positive, negative, 4, 0.5)
    np.testing.assert_allclose(
        model.scores,
        np.log((np.bincount(positive, minlength=4) + 0.5) / 5)
        - np.log((np.bincount(negative, minlength=4) + 0.5) / 6),
    )
    permutation = np.array([3, 1, 0, 2])
    permuted = fit_likelihood_ratio(permutation[positive], permutation[negative], 4, 0.5)
    np.testing.assert_allclose(permuted.scores[permutation], model.scores)


def test_perfect_leakage_roc_matches_sklearn_and_saved_attacker(tmp_path) -> None:
    groups = [Group("a", (str(value),), "public") for value in range(2)]
    train = _data(groups, [0, 1])
    test = _data(groups, [0, 1], prefix="test")
    result = evaluate_attribute_inference(*train, *test, expected_groups=groups, k=3)
    assert result.summary["status"] == "complete"
    assert result.summary["worst_roc_auc"] == 1
    assert result.summary["eligible_pairs"] == result.summary["evaluated_pairs"] == 2
    assert result.pairs["tpr"].min() == 1
    for _, pair in result.pairs.iterrows():
        scores = result.score_tables[pair["pair_id"]]
        labels = np.array(
            [group.values == tuple(json.loads(pair["positive_values"])) for group in test[1]]
        )
        assert pair["roc_auc"] == roc_auc_score(labels, scores[test[0]])
        curve = result.roc.loc[result.roc["pair_id"] == pair["pair_id"]]
        np.testing.assert_allclose(curve.iloc[0][["fpr", "tpr"]].to_numpy(float), 0)
        np.testing.assert_allclose(curve.iloc[-1][["fpr", "tpr"]].to_numpy(float), 1)
    result.save(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["certificate"] is False
    assert (tmp_path / "roc.png").is_file()
    assert len(pd.read_csv(tmp_path / "pairs.csv")) == 2
    with np.load(tmp_path / "attacker_scores.npz") as frozen:
        for key, scores in result.score_tables.items():
            np.testing.assert_array_equal(frozen[key], scores)


def test_public_context_alone_is_not_counted_as_output_leakage() -> None:
    groups = [Group("a", (str(a),), str(c)) for c, a in product(range(2), repeat=2)]
    # Output reveals context exactly, but contains no additional attribute information.
    train = _data(groups, [0, 0, 1, 1], repeats=300)
    test = _data(groups, [0, 0, 1, 1], repeats=300, prefix="test")
    result = evaluate_attribute_inference(
        *train, *test, expected_groups=groups, k=2, target_fpr=0.1
    )
    np.testing.assert_allclose(result.pairs["roc_auc"], 0.5)
    assert result.pairs["tie_probability"].min() == 0.1
    assert result.pairs["tpr"].max() < 0.2
    assert result.pairs["fpr"].max() < 0.2


def test_adjacent_tuple_attack_detects_xor_and_excludes_two_coordinate_pairs() -> None:
    groups = [Group("a+b", (str(a), str(b)), "c") for a, b in product(range(2), repeat=2)]
    train = _data(groups, [0, 1, 1, 0])
    test = _data(groups, [0, 1, 1, 0], prefix="test")
    result = evaluate_attribute_inference(*train, *test, expected_groups=groups, k=2)
    assert len(result.pairs) == 8
    assert all(len(json.loads(value)) == 1 for value in result.pairs["changed_attributes"])
    assert result.summary["mean_roc_auc"] == 1


def test_test_labels_do_not_select_scores_thresholds_or_scope() -> None:
    groups = [Group("a", (str(value),), "c") for value in range(2)]
    train = _data(groups, [0, 1])
    test = _data(groups, [0, 1], prefix="test")
    first = evaluate_attribute_inference(*train, *test, expected_groups=groups, k=2)
    second = evaluate_attribute_inference(
        *train, 1 - test[0], test[1][::-1], test[2], expected_groups=groups, k=2
    )
    for key in first.score_tables:
        np.testing.assert_array_equal(first.score_tables[key], second.score_tables[key])
    pd.testing.assert_frame_equal(
        first.pairs[["pair_id", "threshold", "tie_probability"]],
        second.pairs[["pair_id", "threshold", "tie_probability"]],
    )
    assert first.summary["scope_sha256"] == second.summary["scope_sha256"]


def test_missing_groups_are_not_reported_as_zero_leakage() -> None:
    groups = [Group("a", (str(value),), "c") for value in range(3)]
    train = _data(groups[:2], [0, 1])
    test = _data(groups[:1], [0], prefix="test")
    result = evaluate_attribute_inference(*train, *test, expected_groups=groups, k=2)
    assert result.summary["status"] == "incomplete_test_coverage"
    assert result.summary["worst_roc_auc"] is None
    assert result.summary["expected_pairs"] == 6
    assert result.summary["eligible_pairs"] == 2
    assert result.summary["evaluated_pairs"] == 0
    assert set(result.pairs["status"]) == {"missing_fit_group", "missing_test_group"}


def test_attack_backend_can_be_replaced_without_changing_evaluation(monkeypatch) -> None:
    groups = [Group("a", (str(value),), "c") for value in range(2)]
    train = _data(groups, [0, 1])
    test = _data(groups, [0, 1], prefix="test")
    monkeypatch.setitem(
        ATTACK_FACTORIES, "constant", lambda *args: LikelihoodRatioAttack(np.zeros(2))
    )
    result = evaluate_attribute_inference(
        *train, *test, expected_groups=groups, k=2, method="constant"
    )
    assert result.summary["method"] == "constant"
    assert result.summary["mean_roc_auc"] == 0.5


def test_reproducible_user_day_split_and_invalid_inputs() -> None:
    groups = [Group("a", (str(value),), "c") for value in range(2)]
    train = _data(groups, [0, 1])
    test = _data(groups, [0, 1], prefix="test")
    order = np.arange(len(train[2]))[::-1]
    np.testing.assert_array_equal(
        attack_fit_mask(train[2], seed=5, calibration_fraction=0.5)[order],
        attack_fit_mask(train[2][order], seed=5, calibration_fraction=0.5),
    )
    for key, value, message in (
        ("fit_split", "D_test", "fit only"),
        ("method", "unknown", "unsupported"),
        ("smoothing", 0, "positive"),
        ("target_fpr", 1, "FPR"),
    ):
        with pytest.raises(ValueError, match=message):
            evaluate_attribute_inference(*train, *test, expected_groups=groups, k=2, **{key: value})
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_attribute_inference(*train, *train, expected_groups=groups, k=2)
    with pytest.raises(ValueError, match="one release"):
        evaluate_attribute_inference(
            train[0], train[1], np.repeat("same", len(train[0])), *test, expected_groups=groups, k=2
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"attribute_attack_method": "bad"},
        {"attribute_attack_target_fpr": 0},
        {"attribute_attack_confidence": 1},
        {"attribute_attack_smoothing": float("nan")},
        {"attribute_attack_seed": -1},
    ],
)
def test_invalid_attack_config(overrides) -> None:
    with pytest.raises(ValueError, match="attribute_attack"):
        validate_config({"prior_art_comparison": True, **overrides})
