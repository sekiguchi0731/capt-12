from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capt12.comparison.infonce import (
    identity_infonce_and_gradient,
    infonce_objective_and_gradient,
    pretrain_infonce_embeddings,
    sample_source_batches,
)
from capt12.comparison.mass12 import fit_mass12_finite
from capt12.config import validate_config


def test_identity_pretraining_gradient_includes_both_embedding_roles_and_duplicates() -> None:
    rng = np.random.default_rng(17)
    embeddings = rng.normal(size=(4, 3))
    masses = np.array([[0.1, 0.2, 0.1, 0], [0.2, 0, 0.4, 0], [0, 0, 0, 0]])
    batches = (np.array([0, 1, 1, 2]), np.array([2, 0, 2]), np.array([], dtype=int))
    temperature = 0.4

    def independent_loss(values: np.ndarray) -> float:
        unit = values / np.linalg.norm(values, axis=1, keepdims=True)
        result = 0.0
        for weights, batch in zip(masses, batches, strict=True):
            if not len(batch):
                continue
            scores = unit[batch] @ unit[batch].T / temperature
            maximum = scores.max(axis=1)
            denominator = maximum + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
            result += weights.sum() * (denominator - np.diag(scores)).mean()
        return float(result)

    loss, gradient = identity_infonce_and_gradient(masses, embeddings, batches, temperature)
    assert loss == pytest.approx(independent_loss(embeddings), abs=1e-12)
    for index in np.ndindex(embeddings.shape):
        plus, minus = embeddings.copy(), embeddings.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numerical = (independent_loss(plus) - independent_loss(minus)) / 2e-5
        assert gradient[index] == pytest.approx(numerical, abs=1e-8)
    np.testing.assert_array_equal(gradient[3], 0)


def test_pretraining_improves_identity_loss_and_is_reproducible() -> None:
    initial = np.random.default_rng(3).normal(0, 0.1, size=(4, 3))
    original = initial.copy()
    masses = np.full((1, 4), 0.25)
    controls = dict(epochs=50, learning_rate=0.03, batch_size=32, temperature=0.5, seed=7)
    trained, history = pretrain_infonce_embeddings(initial, masses, **controls)
    repeated, repeated_history = pretrain_infonce_embeddings(initial, masses, **controls)
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(trained, repeated)
    assert history == repeated_history
    before = identity_infonce_and_gradient(masses, initial, (np.arange(4),), 0.5)[0]
    after = identity_infonce_and_gradient(masses, trained, (np.arange(4),), 0.5)[0]
    assert after < before
    final_batch = sample_source_batches(masses, 32, np.random.default_rng(history[-1].batch_seed))
    assert history[-1].post_update_loss == pytest.approx(
        identity_infonce_and_gradient(masses, trained, final_batch, 0.5)[0]
    )


def test_mass_starts_from_pretrained_weights_and_saves_pre_post_history(tmp_path: Path) -> None:
    from capt12.experiments.prior_art_comparison import _mass_pretrain_history

    tokens = np.tile(np.arange(4), 10)
    decoder = np.array([[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]])
    seed = 3
    model = fit_mass12_finite(
        tokens, None, np.zeros(len(tokens)), decoder, np.zeros((4, 4)),
        infonce_weight=0.7, infonce_temperature=0.5, infonce_batch_size=16,
        infonce_embedding_dim=3, infonce_pretrain_epochs=15,
        infonce_pretrain_learning_rate=0.03, epochs=1, seed=seed,
    )
    masses = np.full((1, 4), 0.25)
    random_weights = np.random.default_rng(np.random.SeedSequence([seed, 701])).normal(
        0, 0.1, size=(4, 3)
    )
    pretrained, history = pretrain_infonce_embeddings(
        random_weights, masses, epochs=15, learning_rate=0.03,
        batch_size=16, temperature=0.5, seed=seed,
    )
    np.testing.assert_allclose(model.infonce_initial_embeddings, pretrained, rtol=0, atol=1e-14)
    assert not np.allclose(pretrained, random_weights)
    assert not np.allclose(model.infonce_embeddings, pretrained)
    assert [value.batch_seed for value in model.infonce_pretrain_history] == [value.batch_seed for value in history]
    np.testing.assert_allclose(
        [value.post_update_loss for value in model.infonce_pretrain_history],
        [value.post_update_loss for value in history], rtol=0, atol=1e-14,
    )
    # Independently reconstruct the first main-training measurement, including
    # the unchanged channel initialization and the saved source/output seeds.
    logits = np.random.default_rng(seed).normal(0, 1e-3, size=(1, 4, 2))
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    first = model.history[0]
    batches = sample_source_batches(masses, 16, np.random.default_rng(first.infonce_batch_seed))
    expected = infonce_objective_and_gradient(
        probabilities, decoder, masses, pretrained, batches, 0.5,
        direction="symmetric", output_seed=first.infonce_output_seed,
    ).loss
    assert first.loss == pytest.approx(0.7 * expected)
    assert first.infonce_loss == pytest.approx(expected)
    path = tmp_path / "pretrain.csv"
    _mass_pretrain_history(model, path)
    frame = pd.read_csv(path)
    assert frame["epoch"].tolist() == list(range(1, 16))
    assert frame.iloc[-1]["post_update_infonce_loss"] == pytest.approx(history[-1].post_update_loss)
    assert model.provenance()["infonce_pretrain_completed_epochs"] == 15


def test_pretraining_defaults_and_disabled_conditions() -> None:
    assert validate_config({"mass_epochs": 1})["mass_infonce_pretrain_epochs"] == 100
    tokens = np.arange(4)
    inputs = (tokens, None, np.zeros(4), np.eye(4), np.zeros((4, 4)))
    default = fit_mass12_finite(*inputs, infonce_weight=1, epochs=1)
    assert len(default.infonce_pretrain_history) == 100
    disabled = fit_mass12_finite(*inputs, infonce_weight=1, infonce_pretrain_epochs=0, epochs=1)
    assert not disabled.infonce_pretrain_history
    random_weights = np.random.default_rng(np.random.SeedSequence([0, 701])).normal(0, 0.1, (4, 32))
    np.testing.assert_array_equal(disabled.infonce_initial_embeddings, random_weights)
    no_infonce = fit_mass12_finite(*inputs, infonce_weight=0, epochs=1)
    assert no_infonce.infonce_embeddings is None
    assert no_infonce.infonce_initial_embeddings is None
    assert not no_infonce.infonce_pretrain_history
    assert no_infonce.provenance()["infonce_pretrain_completed_epochs"] == 0


@pytest.mark.parametrize("overrides", [
    {"infonce_pretrain_epochs": -1}, {"infonce_pretrain_learning_rate": 0},
    {"infonce_pretrain_learning_rate": float("nan")},
    {"infonce_pretrain_learning_rate": float("inf")},
])
def test_invalid_pretraining_controls(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="mass_infonce_pretrain"):
        validate_config({f"mass_{key}": value for key, value in overrides.items()})
    with pytest.raises(ValueError, match="InfoNCE pretraining"):
        fit_mass12_finite(np.arange(4), None, np.zeros(4), np.eye(4), np.zeros((4, 4)), **overrides)
