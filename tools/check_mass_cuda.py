"""Small standalone device/training/checkpoint probe; no Criteo data required."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from capt12.comparison.mass_nn import MassNN, fit_mass_nn, resolve_mass_device
from capt12.config import validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    if arguments.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to your allocated physical GPU(s) first.")
    device = resolve_mass_device(arguments.device)
    config = validate_config({
        "K": 32, "L": 8, "mass_device": str(device), "mass_epochs": 2,
        "mass_batch_size": 16, "mass_token_dim": 8, "mass_hidden_dim": 16,
        "mass_attribute_pretrain_epochs": 2, "mass_infonce_pretrain_epochs": 2,
        "mass_infonce_embedding_dim": 4, "mass_eval_noise_samples": 2,
    })
    tokens = np.tile(np.arange(32), 2)
    contexts = np.repeat("public", len(tokens))
    model = fit_mass_nn(tokens, contexts, {"attribute": tokens % 2}, {"click": tokens % 3 == 0},
                        config=config, loss_m=0.1, loss_n=0, privacy_weight=1,
                        utility_weight=1, seed=0, temperature=1)
    outputs = model.sample(tokens, contexts, seed=17)
    channel = model.channel("public")
    assert channel.shape == (32, 32) and np.isfinite(channel).all()
    np.testing.assert_allclose(channel.sum(1), 1, atol=1e-10)
    with tempfile.TemporaryDirectory(prefix="capt-mass-device-") as temporary:
        path = Path(temporary) / "mass.pt"
        model.save(path)
        restored = MassNN.load(path, device=str(device))
        np.testing.assert_array_equal(outputs, restored.sample(tokens, contexts, seed=17))
        # GPU-trained weights must also remain portable to a CPU-only environment.
        assert MassNN.load(path).device.type == "cpu"
    print(json.dumps({
        "status": "ok", "runtime": model.runtime(), "training_runtime": model.training_runtime,
        "torch_cuda_arch_list": torch.cuda.get_arch_list() if device.type == "cuda" else [],
        "gpu_training_and_sampling": device.type == "cuda",
        "history_rows": len(model.history), "checkpoint_roundtrip": True,
        "cpu_checkpoint_load": True,
    }, indent=2))


if __name__ == "__main__":
    main()
