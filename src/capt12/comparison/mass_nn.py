"""MaSS neural training adapted to a fixed, finite token interface.

The module follows jpmorganchase/MaSS at 6fbe9be1: Gaussian-noise residual
generator, pretrained attribute/InfoNCE networks, CE-based MI penalties and
generator/feature then attribute-network updates. Original MaSS emits a
continuous representation. Here a K-way categorical output is necessary for
the token interface; its generator gradient uses the unbiased score-function
estimator, including the dependence of in-batch negatives on the generator.
No CAPT partition, decoder, confidence set or post-training cover is used.
"""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as error:
    raise ImportError("Neural MaSS requires PyTorch; install capt12[mass] (uv sync --extra mass).") from error

from capt12.comparison.mass12 import check_mass_operational_bounds

UPSTREAM_SHA = "6fbe9be1ff155b3fdd2677de7977d801d10000b5"


class TokenGenerator(nn.Module):
    """Upstream residual/noise MLP, with a categorical token output head."""

    def __init__(self, k: int, contexts: int, dim: int) -> None:
        super().__init__()
        self.tokens = nn.Embedding(k, dim)
        self.contexts = nn.Embedding(contexts, dim)
        self.noise_dim = dim
        self.layer1 = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.output = nn.Linear(dim, k)

    def forward(self, tokens: torch.Tensor, contexts: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        original = self.tokens(tokens) + self.contexts(contexts)
        hidden = self.layer1(torch.cat((original, noise), dim=1))
        return self.output(F.relu(hidden + original))


class AttributeNetwork(nn.Module):
    def __init__(self, k: int, contexts: int, dim: int, hidden: int, classes: int) -> None:
        super().__init__()
        self.tokens = nn.Embedding(k, dim)
        self.contexts = nn.Embedding(contexts, dim) if contexts else None
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, max(2, hidden // 2)),
            nn.BatchNorm1d(max(2, hidden // 2)), nn.ReLU(),
            nn.Linear(max(2, hidden // 2), classes),
        )

    def forward(self, tokens: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        values = self.tokens(tokens)
        if self.contexts is not None:
            values = values + self.contexts(contexts)
        return self.mlp(values)


class InfoNCENetwork(nn.Module):
    def __init__(self, k: int, dim: int, hidden: int, feature_dim: int) -> None:
        super().__init__()
        self.tokens = nn.Embedding(k, dim)
        self.encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, feature_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.encoder(self.tokens(tokens))), dim=-1)


def symmetric_infonce(
    source: torch.Tensor, output: torch.Tensor, contexts: torch.Tensor,
    temperature: float, direction: str = "symmetric",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream cross-view-only negatives, optionally stratified by public C."""
    reverse = source.sum() * 0
    forward = reverse
    for context in torch.unique(contexts):
        mask = contexts == context
        first, second = source[mask], output[mask]
        scores = first @ second.T / temperature
        labels = torch.arange(len(first), device=scores.device)
        weight = len(first) / len(source)
        forward = forward + weight * F.cross_entropy(scores, labels)
        reverse = reverse + weight * F.cross_entropy(scores.T, labels)
    loss = (forward + reverse) / 2 if direction == "symmetric" else (
        reverse if direction == "output_to_source" else forward
    )
    return loss, reverse, forward


def ce_mi_penalties(
    sensitive_ce: Mapping[str, torch.Tensor], useful_ce: Mapping[str, torch.Tensor],
    entropies: Mapping[str, float], m: Mapping[str, float], n: Mapping[str, float],
    *, privacy_weight: float = 1, utility_weight: float = 1,
    privacy_quadratic_weight: float = 1, privacy_linear_weight: float = 1,
    useful_quadratic_weight: float = 1, useful_linear_weight: float = 1,
) -> torch.Tensor:
    """Upstream min(CE-H+m,0) and max(CE-H+n,0), with d²+|d|."""
    values = list(sensitive_ce.values()) + list(useful_ce.values())
    result = values[0].sum() * 0
    for name, ce in sensitive_ce.items():
        violation = torch.minimum(ce - entropies[name] + m[name], ce.new_zeros(()))
        result = result + privacy_weight * (
            privacy_quadratic_weight * violation.square() + privacy_linear_weight * violation.abs()
        )
    for name, ce in useful_ce.items():
        violation = torch.maximum(ce - entropies[name] + n[name], ce.new_zeros(()))
        result = result + utility_weight * (
            useful_quadratic_weight * violation.square() + useful_linear_weight * violation.abs()
        )
    return result


def score_function_loss(loss: torch.Tensor, log_probabilities: torch.Tensor, baseline: float) -> torch.Tensor:
    # Sum is essential: loss already averages a coupled batch, including all
    # InfoNCE positives/negatives. Dividing again by B would bias its gradient.
    return (loss.detach() - baseline) * log_probabilities.sum()


def _entropy(labels: np.ndarray, contexts: np.ndarray, conditional: bool) -> float:
    result = 0.0
    for context in (np.unique(contexts) if conditional else [None]):
        selected = labels[contexts == context] if conditional else labels
        _, counts = np.unique(selected, return_counts=True)
        probabilities = counts / counts.sum()
        result += len(selected) / len(labels) * float(-(probabilities * np.log(probabilities)).sum())
    return result


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _batches(order: np.ndarray, size: int) -> list[np.ndarray]:
    batches = [order[start:start + size] for start in range(0, len(order), size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        last = batches.pop()
        batches[-1] = np.concatenate((batches[-1], last))
    return batches


class MassNN:
    def __init__(self, config: dict[str, Any], contexts: tuple[str, ...], classes: dict[str, tuple[str, ...]]) -> None:
        self.config = config
        self.context_values = contexts
        self.label_values = classes
        k, dim, hidden = config["K"], config["mass_token_dim"], config["mass_hidden_dim"]
        self.generator = TokenGenerator(k, len(contexts), dim)
        self.features = InfoNCENetwork(k, dim, hidden, config["mass_infonce_embedding_dim"])
        conditional_contexts = len(contexts) if config["mass_mi_conditioning"] == "conditional" else 0
        # ModuleDict keys are generated, because dataset column names may contain dots.
        self.head_keys = {name: f"attribute_{i}" for i, name in enumerate(classes)}
        self.attributes = nn.ModuleDict({
            self.head_keys[name]: AttributeNetwork(k, conditional_contexts, dim, hidden, max(2, len(levels)))
            for name, levels in classes.items()
        })
        self.history: list[dict[str, Any]] = []
        self.pretrain_history: list[dict[str, Any]] = []
        self.initial_state: dict[str, Any] = {}
        self.entropies: dict[str, float] = {}
        self.m: dict[str, float] = {}
        self.n: dict[str, float] = {}
        self.evaluation_indices: list[int] = []
        self.eval()

    def eval(self) -> None:
        for module in (self.generator, self.features, self.attributes):
            module.eval()

    def _draw(self, tokens: torch.Tensor, contexts: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = torch.Generator().manual_seed(int(seed))
        noise = torch.randn((len(tokens), self.generator.noise_dim), generator=rng)
        logits = self.generator(tokens, contexts, noise) / self.config["mass_temperature"]
        log_probabilities = F.log_softmax(logits, dim=1)
        outputs = torch.multinomial(log_probabilities.exp(), 1, generator=rng).squeeze(1)
        return outputs, log_probabilities.gather(1, outputs[:, None]).squeeze(1)

    @torch.no_grad()
    def sample(self, tokens: np.ndarray, public_context: np.ndarray, *, seed: int) -> np.ndarray:
        self.eval()
        values = np.asarray(tokens, dtype=int)
        if values.ndim != 1 or np.any(values < 0) or np.any(values >= self.config["K"]):
            raise ValueError("tokens are outside the frozen K-token alphabet")
        contexts = np.array([self.context_values.index(str(value)) for value in public_context])
        if contexts.shape != values.shape:
            raise ValueError("public_context must align with tokens")
        return self._draw(torch.tensor(values), torch.tensor(contexts), seed)[0].numpy()

    @torch.no_grad()
    def channel(self, public_context: str, *, noise_samples: int | None = None) -> np.ndarray:
        """MC average over Gaussian input noise; exact categorical expectation per draw.

        This is a utility-evaluation estimate, never a privacy certificate or
        a replacement mechanism. Runtime sample() draws fresh Gaussian noise.
        """
        self.eval()
        k = self.config["K"]
        count = self.config["mass_eval_noise_samples"] if noise_samples is None else noise_samples
        if count < 1:
            raise ValueError("noise_samples must be positive")
        context = self.context_values.index(str(public_context))
        rng = torch.Generator().manual_seed(int(np.random.SeedSequence([
            self.config["mass_seed"], context, 901,
        ]).generate_state(1)[0]))
        result = np.empty((k, k), dtype=float)
        # Bound temporary logits to roughly 4 million entries, also at K=4096.
        chunk = max(1, min(128, 4_000_000 // (k * count)))
        for start in range(0, k, chunk):
            tokens = torch.arange(start, min(start + chunk, k)).repeat_interleave(count)
            contexts = torch.full_like(tokens, context)
            noise = torch.randn((len(tokens), self.generator.noise_dim), generator=rng)
            probabilities = F.softmax(self.generator(tokens, contexts, noise) / self.config["mass_temperature"], dim=1)
            result[start:start + len(tokens) // count] = probabilities.reshape(-1, count, k).mean(1).double().numpy()
        result /= result.sum(axis=1, keepdims=True)
        return result

    def provenance(self) -> dict[str, Any]:
        return {
            "method": "mass_nn", "upstream_sha": UPSTREAM_SHA,
            "architecture": "Gaussian-noise residual generator; attribute and shared InfoNCE MLPs",
            "mi_estimator": "H(label|C)-classifier_cross_entropy" if self.config["mass_mi_conditioning"] == "conditional"
            else "H(label)-classifier_cross_entropy",
            "gradient_estimator": "categorical_score_function_with_past_loss_baseline",
            "postprocessing": "none", "uses_capt_decoder": False,
            "source_split": "D_design", "inference_inputs": ["Z", "B", "S"],
            "uses_sensitive_value_online": False, "formal_capt_certificate": False,
            "utility_channel_estimator": "Monte Carlo Gaussian noise; categorical expectation",
            "config": self.config, "context_values": list(self.context_values),
            "label_values": self.label_values, "entropies": self.entropies,
            "loss_m": self.m, "loss_n": self.n,
        }

    def save(self, path: Path) -> None:
        self.eval()
        torch.save({
            "format_version": 1, "config": self.config, "contexts": self.context_values,
            "classes": self.label_values, "generator": _state(self.generator),
            "features": _state(self.features), "attributes": _state(self.attributes),
            "initial_state": self.initial_state, "entropies": self.entropies,
            "loss_m": self.m, "loss_n": self.n,
            "history": self.history, "pretrain_history": self.pretrain_history,
            "evaluation_indices": self.evaluation_indices,
        }, path)

    @classmethod
    def load(cls, path: Path) -> MassNN:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1:
            raise ValueError("unsupported neural MaSS checkpoint")
        with torch.random.fork_rng():
            model = cls(payload["config"], tuple(payload["contexts"]), payload["classes"])
        for name in ("generator", "features", "attributes"):
            getattr(model, name).load_state_dict(payload[name])
        for name in ("initial_state", "entropies", "history", "pretrain_history", "evaluation_indices"):
            setattr(model, name, payload[name])
        model.m, model.n = payload["loss_m"], payload["loss_n"]
        model.eval()
        return model


class NeuralChannels(Mapping[str, np.ndarray]):
    """Lazy MC utility channels; does not modify the trained mechanism."""

    def __init__(self, model: MassNN) -> None:
        self.model = model

    def __iter__(self) -> Iterator[str]:
        return iter(self.model.context_values)

    def __len__(self) -> int:
        return len(self.model.context_values)

    def __getitem__(self, context: str) -> np.ndarray:
        return self.model.channel(context)


def fit_mass_nn(
    tokens: np.ndarray, public_context: np.ndarray, sensitive_labels: Mapping[str, np.ndarray],
    useful_labels: Mapping[str, np.ndarray], *, config: dict[str, Any],
    loss_m: float, loss_n: float, privacy_weight: float, utility_weight: float,
    seed: int, temperature: float, cost: np.ndarray | None = None,
    source_split: str = "D_design", frozen_context_values: tuple[str, ...] | None = None,
) -> MassNN:
    if source_split != "D_design":
        raise ValueError("neural MaSS training/pretraining requires D_design")
    values = np.asarray(tokens, dtype=int)
    contexts_raw = np.asarray(public_context).astype(str)
    if values.ndim != 1 or len(values) < 2 or contexts_raw.shape != values.shape:
        raise ValueError("neural MaSS requires at least two aligned token/context rows")
    if np.any(values < 0) or np.any(values >= config["K"]):
        raise ValueError("tokens fall outside K")
    labels = {name: np.asarray(array).astype(str) for name, array in {**sensitive_labels, **useful_labels}.items()}
    if not labels or any(array.shape != values.shape for array in labels.values()):
        raise ValueError("attribute labels must align with D_design")
    if set(sensitive_labels) & set(useful_labels):
        raise ValueError("sensitive and useful attribute names must be distinct")
    m = {name: float(loss_m) for name in sensitive_labels}
    n = {name: float(loss_n) for name in useful_labels}
    check_mass_operational_bounds(sensitive_labels, useful_labels, m, n)
    contexts = tuple(sorted(map(str, frozen_context_values or set(contexts_raw))))
    context_array = np.array([contexts.index(value) for value in contexts_raw], dtype=np.int64)
    classes = {name: tuple(sorted(map(str, np.unique(array)))) for name, array in labels.items()}
    cfg = dict(config)
    cfg.update(mass_seed=int(seed), mass_temperature=float(temperature))
    cost_tensor = None
    if cfg["mass_ctr_weight"] > 0:
        if cost is None or np.asarray(cost).shape != (cfg["K"], cfg["K"]):
            raise ValueError("active CTR cost requires a K-by-K matrix")
        cost_tensor = torch.tensor(np.asarray(cost), dtype=torch.float32)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        # Keep checkpoint configs primitive and independent of paths/data rows.
        keys = [key for key in cfg if key.startswith("mass_") and not key.endswith("_list")]
        model = MassNN({"K": int(cfg["K"]), **{key: cfg[key] for key in keys}}, contexts, classes)
        model.m, model.n = m, n
        conditional = cfg["mass_mi_conditioning"] == "conditional"
        model.entropies = {name: _entropy(array, context_array, conditional) for name, array in labels.items()}
        x, c = torch.tensor(values), torch.tensor(context_array)
        y = {name: torch.tensor(np.array([classes[name].index(value) for value in array])) for name, array in labels.items()}
        random = np.random.default_rng(np.random.SeedSequence([seed, 802]))
        batch_size = cfg["mass_batch_size"]
        attr_opt = torch.optim.AdamW(model.attributes.parameters(), lr=cfg["mass_learning_rate"], weight_decay=cfg["mass_weight_decay"])
        feature_opt = torch.optim.AdamW(model.features.parameters(), lr=cfg["mass_infonce_pretrain_learning_rate"], weight_decay=cfg["mass_weight_decay"])

        def indices() -> torch.Tensor:
            return torch.tensor(random.choice(len(values), size=min(batch_size, len(values)), replace=False))

        def cross_entropies(outputs: torch.Tensor, selected: torch.Tensor) -> dict[str, torch.Tensor]:
            return {name: F.cross_entropy(model.attributes[model.head_keys[name]](outputs, c[selected]), target[selected])
                    for name, target in y.items()}

        def contrastive(outputs: torch.Tensor, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            selected = selected[:cfg["mass_infonce_batch_size"]]
            outputs = outputs[:len(selected)]
            groups = c[selected] if conditional else torch.zeros_like(c[selected])
            return symmetric_infonce(model.features(x[selected]), model.features(outputs), groups,
                                     cfg["mass_infonce_temperature"], cfg["mass_infonce_direction"])

        for stage, steps in (("attributes", cfg["mass_attribute_pretrain_epochs"]),
                             ("infonce", cfg["mass_infonce_pretrain_epochs"] if cfg["mass_infonce_weight"] else 0)):
            for step in range(1, steps + 1):
                selected = indices()
                model.eval()
                with torch.no_grad():
                    before = sum(cross_entropies(x[selected], selected).values()) if stage == "attributes" else contrastive(x[selected], selected)[0]
                module, optimizer = (model.attributes, attr_opt) if stage == "attributes" else (model.features, feature_opt)
                module.train()
                loss = sum(cross_entropies(x[selected], selected).values()) if stage == "attributes" else contrastive(x[selected], selected)[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    after = sum(cross_entropies(x[selected], selected).values()) if stage == "attributes" else contrastive(x[selected], selected)[0]
                model.pretrain_history.append({"stage": stage, "update": step,
                                              "pre_update_loss": float(before), "post_update_loss": float(after)})
        model.initial_state = {name: _state(getattr(model, name)) for name in ("generator", "features", "attributes")}
        # Match upstream's batch-size-scaled generator LR and separate attribute optimizer.
        lr = cfg["mass_learning_rate"] * batch_size / 256
        generator_opt = torch.optim.AdamW(
            list(model.generator.parameters()) + list(model.features.parameters()),
            lr=lr, weight_decay=cfg["mass_weight_decay"],
        )
        attr_opt = torch.optim.AdamW(model.attributes.parameters(), lr=lr, weight_decay=cfg["mass_weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_opt, T_max=cfg["mass_epochs"], eta_min=min(lr, 1e-5))
        attr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(attr_opt, T_max=cfg["mass_epochs"], eta_min=min(lr, 1e-5))
        penalty_controls = {name: float(cfg[f"mass_{name}"]) for name in (
            "privacy_quadratic_weight", "privacy_linear_weight", "useful_quadratic_weight", "useful_linear_weight")}

        def objective(selected: torch.Tensor, draw_seed: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
            outputs, log_probabilities = model._draw(x[selected], c[selected], draw_seed)
            ce = cross_entropies(outputs, selected)
            penalty = ce_mi_penalties({name: ce[name] for name in m}, {name: ce[name] for name in n},
                                      model.entropies, m, n, privacy_weight=privacy_weight,
                                      utility_weight=utility_weight, **penalty_controls)
            info = contrastive(outputs, selected) if cfg["mass_infonce_weight"] else (penalty * 0,) * 3
            distortion = cost_tensor[x[selected], outputs].mean() if cost_tensor is not None else penalty * 0
            loss = penalty + cfg["mass_infonce_weight"] * info[0] + cfg["mass_ctr_weight"] * distortion
            return loss, ce, distortion, info, log_probabilities, outputs

        probe = indices()
        model.evaluation_indices = probe.tolist()

        def measure(draw_seed: int) -> dict[str, Any]:
            model.eval()
            with torch.no_grad():
                loss, ce, distortion, info, _, _ = objective(probe, draw_seed)
            return {"loss": float(loss), "distortion": float(distortion),
                    "sensitive_mi_nats": json.dumps({name: model.entropies[name] - float(ce[name]) for name in m}, sort_keys=True),
                    "useful_mi_nats": json.dumps({name: model.entropies[name] - float(ce[name]) for name in n}, sort_keys=True),
                    "attribute_ce_nats": json.dumps({name: float(value) for name, value in ce.items()}, sort_keys=True),
                    "infonce_loss": float(info[0]), "infonce_output_to_source_loss": float(info[1]),
                    "infonce_source_to_output_loss": float(info[2])}

        baseline = 0.0
        for epoch in range(1, cfg["mass_epochs"] + 1):
            evaluation_seed = int(np.random.SeedSequence([seed, epoch, 804]).generate_state(1)[0])
            before = measure(evaluation_seed)
            optimization_losses, classifier_losses = [], []
            for step, batch in enumerate(_batches(random.permutation(len(values)), batch_size)):
                selected = torch.tensor(batch)
                draw_seed = int(np.random.SeedSequence([seed, epoch, step, 803]).generate_state(1)[0])
                model.generator.train()
                model.features.train()
                model.attributes.train()
                loss, _, _, info, log_probabilities, outputs = objective(selected, draw_seed)
                # The finite output is a hard token in both objectives. The
                # score-function term trains G; the ordinary InfoNCE term trains h.
                generator_loss = score_function_loss(loss, log_probabilities, baseline)
                generator_loss = generator_loss + cfg["mass_infonce_weight"] * info[0]
                generator_opt.zero_grad()
                generator_loss.backward()
                generator_opt.step()
                baseline = 0.9 * baseline + 0.1 * float(loss.detach())
                # As upstream, update attribute networks on the outputs drawn
                # before the generator update, detached from G.
                attr_opt.zero_grad()
                classifier_loss = sum(cross_entropies(outputs.detach(), selected).values())
                classifier_loss.backward()
                attr_opt.step()
                optimization_losses.append(float(loss.detach()))
                classifier_losses.append(float(classifier_loss.detach()))
            after = measure(evaluation_seed)
            row: dict[str, Any] = {"epoch": epoch, **before,
                                  **{f"pre_update_{key}": value for key, value in before.items()},
                                  **{f"post_update_{key}": value for key, value in after.items()},
                                  "optimization_loss": float(np.mean(optimization_losses)),
                                  "discriminator_loss": float(np.mean(classifier_losses)),
                                  "evaluation_seed": evaluation_seed,
                                  "learning_rate": generator_opt.param_groups[0]["lr"],
                                  "mi_estimator": "classifier_cross_entropy"}
            model.history.append(row)
            scheduler.step()
            attr_scheduler.step()
        model.eval()
    return model
