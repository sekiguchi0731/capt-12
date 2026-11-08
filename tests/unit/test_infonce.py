from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capt12.comparison.infonce import expected_infonce_and_gradient, sample_source_batches
from capt12.comparison.mass12 import fit_mass12_finite
from capt12.config import validate_config


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


def test_infonce_training_preserves_source_information_and_records_both_stages(tmp_path: Path) -> None:
    from capt12.experiments.prior_art_comparison import _mass_history

    tokens = np.tile(np.arange(4), 10)
    model = fit_mass12_finite(
        tokens, None, np.zeros(len(tokens), dtype=int), np.eye(4), np.zeros((4, 4)),
        privacy_weight=0, utility_weight=1, infonce_weight=1,
        infonce_temperature=0.4, infonce_batch_size=32, infonce_embedding_dim=4,
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
    expected = expected_infonce_and_gradient(
        channel[None, :, :], np.eye(4), masses, model.infonce_embeddings, batches, 0.4
    )[0]
    assert final.post_update_infonce_loss == pytest.approx(expected)
    assert final.post_update_loss == pytest.approx(expected)
    path = tmp_path / "history.csv"
    _mass_history(model, path)
    frame = pd.read_csv(path)
    assert np.isfinite(frame[["pre_update_infonce_loss", "post_update_infonce_loss"]]).all().all()
    assert frame.iloc[-1]["post_update_infonce_loss"] == pytest.approx(expected)
    assert model.provenance()["infonce_upstream_symmetric"] is False


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
    ])
    assert result.exit_code == 0, result.output
    assert captured["mass_infonce_weight"] == 0.25
    assert captured["mass_infonce_temperature"] == 0.4
    assert captured["mass_infonce_batch_size"] == 8
    assert captured["mass_infonce_embedding_dim"] == 3
