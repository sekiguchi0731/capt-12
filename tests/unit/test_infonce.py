from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capt12.comparison.infonce import (
    expected_infonce_and_gradient,
    infonce_objective_and_gradient,
    sample_output_batches,
    sample_source_batches,
    source_to_output_infonce_and_gradient,
)
from capt12.comparison.mass12 import fit_mass12_finite
from capt12.config import validate_config


def test_forward_infonce_expected_gradients_match_full_output_enumeration() -> None:
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(1, 2, 2))
    embeddings = rng.normal(size=(2, 3))
    decoder = np.array([[0.8, 0.2], [0.1, 0.9]])
    masses = np.array([[0.4, 0.6]])
    sources = (np.array([0, 1, 0]),)
    temperature = 0.4

    def probabilities(values: np.ndarray) -> np.ndarray:
        exponential = np.exp(values - values.max(axis=-1, keepdims=True))
        return exponential / exponential.sum(axis=-1, keepdims=True)

    def true_expected_loss(values: np.ndarray, features: np.ndarray) -> float:
        q = probabilities(values)[0, sources[0]] @ decoder
        unit = features / np.linalg.norm(features, axis=1, keepdims=True)
        result = 0.0
        for outputs in product(range(2), repeat=3):
            probability = float(np.prod(q[np.arange(3), outputs]))
            scores = unit[sources[0]] @ unit[list(outputs)].T / temperature
            maximum = scores.max(axis=1)
            log_denominator = maximum + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
            result += probability * float((log_denominator - np.diag(scores)).mean())
        return result

    pi = probabilities(logits)
    q = pi[0, sources[0]] @ decoder
    average_loss = 0.0
    average_pi_gradient = np.zeros_like(pi)
    average_embedding_gradient = np.zeros_like(embeddings)
    for outputs in product(range(2), repeat=3):
        probability = float(np.prod(q[np.arange(3), outputs]))
        loss, pi_gradient, embedding_gradient = source_to_output_infonce_and_gradient(
            pi, decoder, masses, embeddings, sources, (np.array(outputs),), temperature
        )
        average_loss += probability * loss
        average_pi_gradient += probability * pi_gradient
        average_embedding_gradient += probability * embedding_gradient
    assert average_loss == pytest.approx(true_expected_loss(logits, embeddings), abs=1e-12)
    logit_gradient = pi * (
        average_pi_gradient - np.sum(pi * average_pi_gradient, axis=-1, keepdims=True)
    )
    h = 1e-5
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += h
        minus[index] -= h
        numerical = (true_expected_loss(plus, embeddings) - true_expected_loss(minus, embeddings)) / (2 * h)
        assert logit_gradient[index] == pytest.approx(numerical, abs=1e-8)
    for index in np.ndindex(embeddings.shape):
        plus, minus = embeddings.copy(), embeddings.copy()
        plus[index] += h
        minus[index] -= h
        numerical = (true_expected_loss(logits, plus) - true_expected_loss(logits, minus)) / (2 * h)
        assert average_embedding_gradient[index] == pytest.approx(numerical, abs=1e-8)


def test_symmetric_infonce_is_mean_of_both_directions() -> None:
    pi = np.array([[[0.8, 0.2], [0.3, 0.7]]])
    decoder = np.eye(2)
    masses = np.array([[0.5, 0.5]])
    embeddings = np.array([[0.8, 0.3], [0.2, 0.9]])
    sources = (np.array([0, 1]),)
    reverse = expected_infonce_and_gradient(pi, decoder, masses, embeddings, sources, 0.4)
    outputs = sample_output_batches(pi, decoder, sources, np.random.default_rng(4))
    forward = source_to_output_infonce_and_gradient(pi, decoder, masses, embeddings, sources, outputs, 0.4)
    symmetric = infonce_objective_and_gradient(
        pi, decoder, masses, embeddings, sources, 0.4, output_seed=4,
    )
    assert symmetric.loss == pytest.approx((reverse[0] + forward[0]) / 2)
    np.testing.assert_allclose(symmetric.probability_gradient, (reverse[1] + forward[1]) / 2)
    np.testing.assert_allclose(symmetric.embedding_gradient, (reverse[2] + forward[2]) / 2)


def test_expected_infonce_matches_enumeration_and_finite_difference_gradients() -> None:
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(3, 3, 2))
    decoder = np.array([[0.7, 0.3, 0.0], [0.0, 0.2, 0.8]])
    masses = np.array([[0.1, 0.2, 0.3], [0.1, 0.0, 0.3], [0.0, 0.0, 0.0]])
    embeddings = rng.normal(size=(3, 3))
    batches = (np.array([0, 1, 1, 2, 0]), np.array([0, 2, 2, 0, 0]), np.array([], dtype=int))
    temperature = 0.4

    def probabilities(values: np.ndarray) -> np.ndarray:
        weights = np.exp(values - values.max(axis=-1, keepdims=True))
        return weights / weights.sum(axis=-1, keepdims=True)

    def objective(values: np.ndarray, features: np.ndarray) -> float:
        return expected_infonce_and_gradient(
            probabilities(values), decoder, masses, features, batches, temperature
        )[0]

    pi = probabilities(logits)
    loss, pi_gradient, embedding_gradient = expected_infonce_and_gradient(
        pi, decoder, masses, embeddings, batches, temperature
    )
    unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    enumerated = 0.0
    for context, sources in enumerate(batches):
        for source in sources:
            for output in range(3):
                scores = np.array([unit[other] @ unit[output] / temperature for other in sources])
                log_sum = scores.max() + math.log(sum(math.exp(score - scores.max()) for score in scores))
                cross_entropy = log_sum - unit[source] @ unit[output] / temperature
                q = sum(pi[context, source, block] * decoder[block, output] for block in range(2))
                enumerated += masses[context].sum() / len(sources) * q * cross_entropy
    assert loss == pytest.approx(enumerated, abs=1e-12)
    logit_gradient = pi * (pi_gradient - np.sum(pi * pi_gradient, axis=-1, keepdims=True))
    h = 1e-5
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += h
        minus[index] -= h
        numerical = (objective(plus, embeddings) - objective(minus, embeddings)) / (2 * h)
        assert logit_gradient[index] == pytest.approx(numerical, abs=1e-8)
    for index in np.ndindex(embeddings.shape):
        plus, minus = embeddings.copy(), embeddings.copy()
        plus[index] += h
        minus[index] -= h
        numerical = (objective(logits, plus) - objective(logits, minus)) / (2 * h)
        assert embedding_gradient[index] == pytest.approx(numerical, abs=1e-8)


def test_infonce_distinguishes_identity_from_constant_output_and_handles_collapse() -> None:
    masses = np.full((1, 3), 1 / 3)
    batches = (np.arange(3),)
    identity = np.eye(3)[None, :, :]
    constant = np.full((1, 3, 3), 1 / 3)
    identity_loss = expected_infonce_and_gradient(identity, np.eye(3), masses, np.eye(3), batches, 0.5)[0]
    constant_loss = expected_infonce_and_gradient(constant, np.eye(3), masses, np.eye(3), batches, 0.5)[0]
    assert identity_loss < constant_loss
    collapsed = expected_infonce_and_gradient(identity, np.eye(3), masses, np.ones((3, 2)), batches, 0.5)[0]
    assert collapsed == pytest.approx(math.log(3))


def test_penalty_coefficients_are_independently_applied() -> None:
    tokens = np.tile(np.arange(4), 10)
    model = fit_mass12_finite(
        tokens, None, tokens // 2, np.eye(4), 1 - np.eye(4),
        useful_labels=tokens % 2, loss_m=0.0, loss_n=0.4,
        privacy_weight=2, utility_weight=3, ctr_weight=0.7,
        privacy_quadratic_weight=4, privacy_linear_weight=0.5,
        useful_quadratic_weight=1.5, useful_linear_weight=0.75,
        epochs=20, seed=2,
    )
    final = model.history[-1]
    sensitive = max(final.post_update_sensitive_mi_nats["sensitive"], 0.0)
    useful = max(0.4 - final.post_update_useful_mi_nats["useful"], 0.0)
    expected = (
        0.7 * final.post_update_distortion
        + 2 * (4 * sensitive**2 + 0.5 * sensitive)
        + 3 * (1.5 * useful**2 + 0.75 * useful)
    )
    assert final.post_update_loss == pytest.approx(expected)
    assert model.provenance()["privacy_quadratic_weight"] == 4
    assert model.provenance()["useful_linear_weight"] == 0.75
    assert model.provenance()["ctr_weight"] == 0.7


def test_ctr_cost_is_off_by_default_without_disabling_mi_or_infonce() -> None:
    tokens = np.tile(np.arange(4), 10)
    controls = dict(
        useful_labels=tokens % 2, loss_m=0.0, loss_n=0.4,
        infonce_weight=1, infonce_temperature=0.4,
        infonce_batch_size=8, infonce_embedding_dim=3, epochs=4, seed=2,
    )
    cost = 1 - np.eye(4)
    first = fit_mass12_finite(tokens, None, tokens // 2, np.eye(4), cost, **controls)
    changed_cost = fit_mass12_finite(
        tokens, None, tokens // 2, np.eye(4), 20 * cost.T + 3, **controls,
    )
    explicit_zero = fit_mass12_finite(
        tokens, None, tokens // 2, np.eye(4), cost, ctr_weight=0, **controls,
    )
    enabled = fit_mass12_finite(
        tokens, None, tokens // 2, np.eye(4), cost, ctr_weight=2, **controls,
    )
    assert first.ctr_weight == 0
    np.testing.assert_array_equal(first.logits, changed_cost.logits)
    np.testing.assert_array_equal(first.infonce_embeddings, changed_cost.infonce_embeddings)
    np.testing.assert_array_equal(first.logits, explicit_zero.logits)
    assert not np.allclose(first.logits, enabled.logits)
    assert first.history[-1].post_update_distortion != changed_cost.history[-1].post_update_distortion
    assert first.history[-1].post_update_loss == changed_cost.history[-1].post_update_loss
    for epoch in first.history:
        sensitive = epoch.sensitive_mi_nats["sensitive"]
        useful = max(0.4 - epoch.useful_mi_nats["useful"], 0)
        assert epoch.infonce_loss > 0
        assert sensitive > 0 and useful > 0
        assert epoch.loss == pytest.approx(
            sensitive**2 + sensitive + useful**2 + useful + epoch.infonce_loss
        )


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_ctr_cost_weight_must_be_finite_and_nonnegative(weight: float) -> None:
    with pytest.raises(ValueError, match="mass_ctr_weight"):
        validate_config({"mass_ctr_weight": weight})
    with pytest.raises(ValueError, match="ctr_weight"):
        fit_mass12_finite(
            np.arange(4), None, np.zeros(4), np.eye(4), np.zeros((4, 4)), ctr_weight=weight,
        )


@pytest.mark.parametrize("direction", ["output_to_source", "source_to_output", "symmetric"])
def test_infonce_training_preserves_source_information_and_records_both_stages(
    tmp_path: Path, direction: str
) -> None:
    from capt12.experiments.prior_art_comparison import _mass_history

    tokens = np.tile(np.arange(4), 10)
    model = fit_mass12_finite(
        tokens, None, np.zeros(len(tokens), dtype=int), np.eye(4), np.zeros((4, 4)),
        privacy_weight=0, utility_weight=1, infonce_weight=1,
        infonce_temperature=0.4, infonce_batch_size=32, infonce_embedding_dim=4,
        infonce_direction=direction,
        epochs=40, seed=1,
    )
    channel = model.channel()
    joint = channel / 4
    denominator = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    information = float(np.sum(joint * np.log(joint / denominator)))
    assert information > 0.05
    assert model.history[-1].post_update_infonce_loss < model.history[0].infonce_loss
    masses = np.full((1, 4), 0.25)
    final = model.history[-1]
    batches = sample_source_batches(masses, 32, np.random.default_rng(final.infonce_batch_seed))
    expected = infonce_objective_and_gradient(
        channel[None, :, :], np.eye(4), masses, model.infonce_embeddings, batches, 0.4,
        direction=direction, output_seed=final.infonce_output_seed,
    ).loss
    assert final.post_update_infonce_loss == pytest.approx(expected)
    assert final.post_update_loss == pytest.approx(expected)
    path = tmp_path / "history.csv"
    _mass_history(model, path)
    frame = pd.read_csv(path)
    assert np.isfinite(frame[["pre_update_infonce_loss", "post_update_infonce_loss"]]).all().all()
    assert frame.iloc[-1]["post_update_infonce_loss"] == pytest.approx(expected)
    assert model.provenance()["infonce_upstream_symmetric"] is (direction == "symmetric")


@pytest.mark.parametrize("overrides", [
    {"mass_infonce_weight": float("nan")}, {"mass_infonce_weight": -1},
    {"mass_infonce_temperature": 0}, {"mass_infonce_batch_size": 1},
    {"mass_infonce_embedding_dim": 0},
])
def test_infonce_config_rejects_invalid_controls(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="mass_infonce"):
        validate_config({"mass_epochs": 1, **overrides})


def test_infonce_cli_controls_are_passed_to_comparison(monkeypatch) -> None:
    from typer.testing import CliRunner

    from capt12.cli import app

    captured = {}

    def fake_comparison(config, *, phase, resume):
        captured.update(config)
        return Path("outputs/fake-prior-art-run")

    monkeypatch.setattr("capt12.experiments.prior_art_comparison.run_prior_art_comparison", fake_comparison)
    result = CliRunner().invoke(app, [
        "prior-art-comparison", "--config", "configs/criteo_prior_art_privacy_matched.yaml",
        "--K", "32", "--L", "32", "--mass-infonce-weight", "0.25",
        "--mass-infonce-temperature", "0.4", "--mass-infonce-batch-size", "8",
        "--mass-infonce-embedding-dim", "3",
        "--mass-infonce-direction", "symmetric",
        "--mass-infonce-pretrain-epochs", "7", "--mass-infonce-pretrain-learning-rate", "0.02",
        "--mass-ctr-weight", "0.7",
        "--mass-m-list", "0.1,0.2", "--mass-n-list", "0.01,0.03",
        "--mass-privacy-weight-list", "2,3", "--mass-utility-weight-list", "4,5",
        "--mass-privacy-quadratic-weight", "2.0", "--mass-privacy-linear-weight", "0.5",
        "--mass-useful-quadratic-weight", "1.5", "--mass-useful-linear-weight", "0.75",
    ])
    assert result.exit_code == 0, result.output
    assert captured["mass_infonce_weight"] == 0.25
    assert captured["mass_infonce_temperature"] == 0.4
    assert captured["mass_infonce_batch_size"] == 8
    assert captured["mass_infonce_embedding_dim"] == 3
    assert captured["mass_infonce_direction"] == "symmetric"
    assert captured["mass_infonce_pretrain_epochs"] == 7
    assert captured["mass_infonce_pretrain_learning_rate"] == 0.02
    assert captured["mass_ctr_weight"] == 0.7
    for name, values in (("m", [0.1, 0.2]), ("n", [0.01, 0.03]),
                         ("privacy_weight", [2.0, 3.0]), ("utility_weight", [4.0, 5.0])):
        assert captured[f"mass_{name}_list"] == values
        assert captured[f"mass_pilot_{name}_list"] == values
    for name, value in (("privacy_quadratic_weight", 2.0), ("privacy_linear_weight", 0.5),
                        ("useful_quadratic_weight", 1.5), ("useful_linear_weight", 0.75)):
        assert captured[f"mass_{name}"] == value
