from __future__ import annotations

import inspect
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from capt12.comparison.mass_nn import (
    MassNN,
    ce_mi_penalties,
    fit_mass_nn,
    resolve_mass_device,
    score_function_loss,
    symmetric_infonce,
)
from capt12.config import validate_config


def _config(**overrides):
    return validate_config({
        "K": 4, "L": 2, "mass_epochs": 2, "mass_batch_size": 8,
        "mass_token_dim": 4, "mass_hidden_dim": 8,
        "mass_attribute_pretrain_epochs": 2, "mass_infonce_pretrain_epochs": 2,
        "mass_infonce_embedding_dim": 4, "mass_eval_noise_samples": 3,
        "mass_learning_rate": 0.01, **overrides,
    })


def test_neural_mi_penalties_match_upstream_equations_and_gradients() -> None:
    sensitive = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    useful = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    loss = ce_mi_penalties({"a": sensitive}, {"y": useful}, {"a": 0.8, "y": 0.9},
                          {"a": 0.1}, {"y": 0.4}, privacy_weight=3, utility_weight=4,
                          privacy_quadratic_weight=2, privacy_linear_weight=0.5,
                          useful_quadratic_weight=1.5, useful_linear_weight=0.75)
    assert float(loss.detach()) == pytest.approx(3 * (2 * 0.3**2 + 0.5 * 0.3) + 4 * (1.5 * 0.2**2 + 0.75 * 0.2))
    loss.backward()
    assert float(sensitive.grad) == pytest.approx(-3 * (4 * 0.3 + 0.5))
    assert float(useful.grad) == pytest.approx(4 * (3 * 0.2 + 0.75))


def test_neural_infonce_matches_official_cross_view_mask() -> None:
    torch.manual_seed(7)
    first = F.normalize(torch.randn(3, 4, dtype=torch.float64), dim=-1)
    second = F.normalize(torch.randn(3, 4, dtype=torch.float64), dim=-1)
    features = torch.cat((first, second))
    scores = features @ features.T / 0.4
    mask = torch.zeros((6, 6), dtype=torch.bool)
    mask[:3, 3:] = True
    mask[3:, :3] = True
    expected = []
    for index in range(6):
        positive = (index + 3) % 6
        expected.append(torch.logsumexp(scores[index, mask[index]], 0) - scores[index, positive])
    loss, reverse, forward = symmetric_infonce(first, second, torch.zeros(3), 0.4)
    assert float(loss) == pytest.approx(float(torch.stack(expected).mean()), abs=1e-12)
    assert float(loss) == pytest.approx(float((forward + reverse) / 2))


def test_categorical_generator_gradient_matches_full_coupled_batch_expectation() -> None:
    logits = torch.tensor([[0.2, -0.3], [-0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(1)
    costs = []
    for outputs in itertools.product(range(2), repeat=2):
        # Includes dependence on both positive and negative sampled outputs.
        info = symmetric_infonce(torch.eye(2, dtype=torch.float64),
                                 torch.eye(2, dtype=torch.float64)[list(outputs)], torch.zeros(2), 0.5)[0]
        classifier = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float64)
        ce = -classifier[list(outputs), [0, 1]].log().mean()
        sensitive = max(math.log(2) - float(ce) - 0.1, 0)
        costs.append((outputs, float(info) + sensitive**2 + sensitive))
    true_loss = sum(probabilities[0, output[0]] * probabilities[1, output[1]] * cost for output, cost in costs)
    expected = torch.autograd.grad(true_loss, logits, retain_graph=True)[0]
    estimated = torch.zeros_like(logits)
    for output, cost in costs:
        probability = (probabilities[0, output[0]] * probabilities[1, output[1]]).detach()
        log_prob = logits.log_softmax(1)[torch.arange(2), list(output)]
        loss = score_function_loss(logits.new_tensor(cost), log_prob, baseline=1.3)
        estimated += probability * torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(estimated, expected, rtol=0, atol=1e-12)


def test_neural_training_pretrains_both_networks_updates_and_roundtrips(tmp_path: Path, monkeypatch) -> None:
    # Exact finite-table MI and CAPT calibration must not be part of this fit.
    monkeypatch.setattr("capt12.comparison.mass12._conditional_mi_and_gradient",
                        lambda *args: pytest.fail("exact MI used in neural training"))
    tokens = np.tile(np.arange(4), 4)
    contexts = np.repeat("public", len(tokens))
    config = _config()
    model = fit_mass_nn(tokens, contexts, {"a": tokens // 2}, {"click": tokens % 2},
                        config=config, loss_m=0.1, loss_n=0.1, privacy_weight=1,
                        utility_weight=1, seed=2, temperature=1)
    assert len(model.history) == 2
    assert [row["stage"] for row in model.pretrain_history] == ["attributes"] * 2 + ["infonce"] * 2
    assert model.generator.output.out_features == 4
    assert "L" not in model.config
    assert "sensitive" not in inspect.signature(model.sample).parameters
    with torch.random.fork_rng():
        torch.manual_seed(2)
        random_model = MassNN(model.config, model.context_values, model.label_values)
    for name in ("features", "attributes"):
        random_state = getattr(random_model, name).state_dict()
        assert any(not torch.equal(value, random_state[key]) for key, value in model.initial_state[name].items()
                   if value.is_floating_point())
    for key, value in random_model.generator.state_dict().items():
        torch.testing.assert_close(value, model.initial_state["generator"][key], rtol=0, atol=0)
    for name in ("generator", "features", "attributes"):
        final = getattr(model, name).state_dict()
        assert any(not torch.equal(value, final[key]) for key, value in model.initial_state[name].items()
                   if value.is_floating_point())
    assert np.isfinite([row["post_update_loss"] for row in model.history]).all()
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = MassNN.load(path)
    np.testing.assert_array_equal(model.sample(tokens, contexts, seed=19), loaded.sample(tokens, contexts, seed=19))
    np.testing.assert_array_equal(model.channel("public"), loaded.channel("public"))
    channel = loaded.channel("public")
    np.testing.assert_allclose(channel.sum(1), 1, atol=1e-12)
    assert channel.shape == (4, 4) and (channel > 0).all()
    assert loaded.provenance()["postprocessing"] == "none"
    assert loaded.provenance()["uses_capt_decoder"] is False
    assert loaded.history == model.history
    with pytest.raises(ValueError, match="D_design"):
        fit_mass_nn(tokens, contexts, {"a": tokens // 2}, {"click": tokens % 2},
                    config=config, loss_m=.1, loss_n=.1, privacy_weight=1, utility_weight=1,
                    seed=2, temperature=1, source_split="D_test")


@pytest.mark.parametrize("overrides", [
    {"mass_mi_conditioning": "invalid"}, {"mass_eval_noise_samples": 0},
    {"mass_token_dim": 1}, {"mass_batch_size": 1}, {"mass_hidden_dim": 0},
    {"mass_weight_decay": float("nan")}, {"mass_attribute_pretrain_epochs": -1},
    {"mass_output_mode": "finite_block"},
    {"mass_device": "cuda:-1"}, {"mass_device": "mps"},
])
def test_neural_controls_reject_invalid_values(overrides) -> None:
    with pytest.raises(ValueError):
        _config(**overrides)


def test_unconditional_classifier_mi_and_singleton_final_batch() -> None:
    tokens = np.arange(9) % 4  # Batch size 8 would otherwise leave one row for BatchNorm.
    contexts = np.array(["a"] * 5 + ["b"] * 4)
    sensitive = (contexts == "b").astype(int)
    config = _config(mass_epochs=1, mass_attribute_pretrain_epochs=0,
                     mass_infonce_pretrain_epochs=0, mass_infonce_weight=0,
                     mass_mi_conditioning="unconditional")
    model = fit_mass_nn(tokens, contexts, {"a": sensitive}, {"click": tokens % 2},
                        config=config, loss_m=0.1, loss_n=0, privacy_weight=1,
                        utility_weight=1, seed=0, temperature=1)
    expected_entropy = -(5 / 9 * math.log(5 / 9) + 4 / 9 * math.log(4 / 9))
    assert model.entropies["a"] == pytest.approx(expected_entropy)
    assert all(network.contexts is None for network in model.attributes.values())
    row = model.history[-1]
    ce = json.loads(row["post_update_attribute_ce_nats"])["a"]
    estimate = json.loads(row["post_update_sensitive_mi_nats"])["a"]
    assert estimate == pytest.approx(expected_entropy - ce)
    assert math.isfinite(row["discriminator_loss"])


def test_full_grid_rejects_different_neural_training_controls(tmp_path: Path, monkeypatch) -> None:
    from capt12.experiments.prior_art_comparison import run_prior_art_comparison

    config = _config(comparison_output_dir=str(tmp_path), prior_art_comparison=True)
    monkeypatch.setattr("capt12.experiments.prior_art_comparison.record_source_provenance",
                        lambda cfg: {**cfg, "source_git_sha": "test-source"})
    (tmp_path / "pilot_summary.json").write_text(json.dumps({
        "source_git_sha": "test-source", "K": 4, "L": 2,
        "mass_training_signature": "different-neural-controls", "full_grid_allowed": True,
    }))
    with pytest.raises(ValueError, match="same neural training controls"):
        run_prior_art_comparison(config, phase="full")


def test_cpu_training_and_portable_load_do_not_fork_cuda(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *args, **kwargs: pytest.fail("CPU path initialized CUDA RNG"))
    tokens = np.tile(np.arange(4), 4)
    contexts = np.repeat("public", len(tokens))
    events = []
    model = fit_mass_nn(tokens, contexts, {"a": tokens // 2}, {"click": tokens % 2},
                        config=_config(), loss_m=0.1, loss_n=0,
                        privacy_weight=1, utility_weight=1, seed=0, temperature=1,
                        progress=events.append)
    assert model.device.type == "cpu"
    assert model.training_runtime["device"] == "cpu"
    assert [event["epoch"] for event in events if event["stage"] == "joint_training"] == [1, 2]
    # Simulate a GPU-trained checkpoint's metadata on this CPU-only host.
    model.config["mass_device"] = "cuda:1"
    model.training_runtime["device"] = "cuda:1"
    path = tmp_path / "portable.pt"
    model.save(path)
    loaded = MassNN.load(path)
    assert loaded.device.type == "cpu"
    assert loaded.training_runtime["device"] == "cuda:1"
    np.testing.assert_array_equal(model.sample(tokens, contexts, seed=3), loaded.sample(tokens, contexts, seed=3))


def test_requested_cuda_fails_explicitly_without_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_mass_device("cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="visible GPU set"):
        resolve_mass_device("cuda:1")


def test_cuda_preflight_precedes_native_experiments(monkeypatch) -> None:
    from capt12.experiments.prior_art_comparison import run_prior_art_comparison

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("capt12.experiments.prior_art_comparison.record_source_provenance",
                        lambda cfg: {**cfg, "source_git_sha": "test-source"})
    monkeypatch.setattr("capt12.experiments.prior_art_comparison._native_run",
                        lambda *args, **kwargs: pytest.fail("CAPT ran before failed CUDA preflight"))
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        run_prior_art_comparison(_config(mass_device="cuda:0"))


@pytest.mark.parametrize("index", [0, 1])
def test_cuda_training_sampling_and_checkpoint_on_explicitly_visible_gpus(index, tmp_path: Path) -> None:
    # Never initialize unallocated devices on a shared machine's default namespace.
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or not torch.cuda.is_available():
        pytest.skip("requires an explicitly allocated CUDA_VISIBLE_DEVICES mask")
    if index >= torch.cuda.device_count():
        pytest.skip("second visible device not allocated")
    tokens = np.tile(np.arange(4), 4)
    contexts = np.repeat("public", len(tokens))
    model = fit_mass_nn(tokens, contexts, {"a": tokens // 2}, {"click": tokens % 2},
                        config=_config(mass_device=f"cuda:{index}"), loss_m=0.1, loss_n=0,
                        privacy_weight=1, utility_weight=1, seed=0, temperature=1)
    assert next(model.generator.parameters()).device == torch.device("cuda", index)
    assert np.isfinite([row["post_update_loss"] for row in model.history]).all()
    outputs = model.sample(tokens, contexts, seed=17)
    np.testing.assert_allclose(model.channel("public").sum(1), 1, atol=1e-10)
    path = tmp_path / "cuda.pt"
    model.save(path)
    loaded = MassNN.load(path, device=f"cuda:{index}")
    np.testing.assert_array_equal(outputs, loaded.sample(tokens, contexts, seed=17))
    assert MassNN.load(path).device.type == "cpu"
