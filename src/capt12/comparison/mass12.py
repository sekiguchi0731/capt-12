from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from capt12.mechanisms.lp import validate_channel


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    probabilities = np.exp(scaled)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def _categorical_entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(np.asarray(labels).astype(str), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def _conditional_entropy(values: np.ndarray, conditions: np.ndarray) -> float:
    values = np.asarray(values).astype(str)
    conditions = np.asarray(conditions).astype(str)
    if len(values) != len(conditions) or len(values) == 0:
        raise ValueError("conditional entropy arrays must be nonempty and aligned")
    result = 0.0
    for condition in np.unique(conditions):
        mask = conditions == condition
        result += float(mask.mean()) * _categorical_entropy(values[mask])
    return result


def _label_mapping(
    labels: Mapping[str, np.ndarray] | np.ndarray | None,
    *,
    default_name: str,
    row_count: int,
) -> dict[str, np.ndarray]:
    if labels is None:
        return {}
    raw = labels if isinstance(labels, Mapping) else {default_name: labels}
    result: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        array = np.asarray(values).astype(str)
        if array.shape != (row_count,):
            raise ValueError(f"label {name} must have one value per D_design row")
        result[str(name)] = array
    return result


def _threshold_mapping(
    value: Mapping[str, float] | float | None,
    names: tuple[str, ...],
    *,
    default: float | None = None,
) -> dict[str, float]:
    if not names:
        return {}
    if isinstance(value, Mapping):
        normalized = {str(name): float(threshold) for name, threshold in value.items()}
        missing = sorted(set(names) - set(normalized))
        if missing:
            raise ValueError(f"missing MaSS thresholds for {missing}")
        return {name: normalized[name] for name in names}
    threshold = default if value is None else float(value)
    if threshold is None:
        raise ValueError("MaSS thresholds are required when labels are provided")
    return {name: threshold for name in names}


def check_mass_operational_bounds(
    sensitive_labels: Mapping[str, np.ndarray],
    useful_labels: Mapping[str, np.ndarray],
    loss_m: Mapping[str, float],
    loss_n: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    """Check the paper's m/n feasibility conditions in nats.

    MaSS requires m>=0, n<=H(U), and n<=m+H(U|S).  These are operational
    feasibility checks, not achieved-privacy measurements.
    """
    diagnostics: dict[str, float] = {}
    for name, threshold in loss_m.items():
        if threshold < 0:
            raise ValueError(f"loss-m for {name} must be nonnegative")
    for useful_name, n_value in loss_n.items():
        entropy = _categorical_entropy(useful_labels[useful_name])
        diagnostics[f"entropy:{useful_name}"] = entropy
        if n_value > entropy + tolerance:
            raise ValueError(f"loss-n for {useful_name} exceeds H(U) in nats")
        for sensitive_name, m_value in loss_m.items():
            conditional = _conditional_entropy(
                useful_labels[useful_name], sensitive_labels[sensitive_name]
            )
            diagnostics[f"conditional_entropy:{useful_name}|{sensitive_name}"] = conditional
            if n_value > m_value + conditional + tolerance:
                raise ValueError(
                    f"loss-n for {useful_name} exceeds loss-m for {sensitive_name} + H(U|S)"
                )
    return diagnostics


def _joint_context_label_token(
    tokens: np.ndarray,
    contexts: np.ndarray,
    labels: np.ndarray,
    context_values: tuple[str, ...],
    token_count: int,
) -> np.ndarray:
    label_values = tuple(sorted(np.unique(labels).tolist()))
    label_index = {value: index for index, value in enumerate(label_values)}
    context_index = {value: index for index, value in enumerate(context_values)}
    joint = np.zeros((len(context_values), len(label_values), token_count), dtype=float)
    for token, context, label in zip(tokens, contexts, labels, strict=True):
        joint[context_index[context], label_index[label], token] += 1
    joint /= len(tokens)
    return joint


def _conditional_mi_and_gradient(
    joint_context_label_token: np.ndarray,
    block_probabilities: np.ndarray,
    decoder: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return exact I(label; output | public context) and dI/dpi in nats."""
    joint = np.asarray(joint_context_label_token, dtype=float)
    pi = np.asarray(block_probabilities, dtype=float)
    gradient = np.zeros_like(pi)
    mutual_information = 0.0
    for context in range(joint.shape[0]):
        context_mass = float(joint[context].sum())
        if context_mass <= 0:
            continue
        label_mass = joint[context].sum(axis=1)
        label_output = (joint[context] @ pi[context]) @ decoder
        output_mass = label_output.sum(axis=0)
        denominator = label_mass[:, None] * output_mass[None, :]
        mask = (label_output > 0) & (denominator > 0)
        log_ratio = np.zeros_like(label_output)
        log_ratio[mask] = np.log(
            label_output[mask] * context_mass / denominator[mask]
        )
        mutual_information += float(np.sum(label_output[mask] * log_ratio[mask]))
        output_gradient = log_ratio @ decoder.T
        gradient[context] = joint[context].T @ output_gradient
    return mutual_information, gradient


@dataclass(frozen=True)
class Mass12Epoch:
    epoch: int
    loss: float
    distortion: float
    sensitive_mi_nats: dict[str, float]
    useful_mi_nats: dict[str, float]


@dataclass(frozen=True)
class Mass12FiniteChannel:
    """MaSS-12 finite-output adaptation with no sensitive inference input."""

    logits: np.ndarray
    decoder: np.ndarray
    context_values: tuple[str, ...]
    temperature: float
    history: tuple[Mass12Epoch, ...]
    loss_m: dict[str, float]
    loss_n: dict[str, float]
    seed: int
    objective_name: str
    source_split: str = "D_design"
    method: str = "mass12_raw"
    display_name: str = "MaSS-12 (finite-output adaptation)"
    uses_sensitive_value_online: bool = False
    deployable_under_capt: bool = True

    def __post_init__(self) -> None:
        if self.source_split != "D_design":
            raise ValueError("MaSS-12 may be fitted from D_design only")
        if self.logits.ndim != 3:
            raise ValueError("logits must have shape (context, token, block)")
        if self.decoder.shape != (self.logits.shape[2], self.logits.shape[1]):
            raise ValueError("common decoder shape does not match MaSS-12 logits")
        if len(self.context_values) != self.logits.shape[0]:
            raise ValueError("context vocabulary does not match logits")
        if self.temperature <= 0 or not np.isfinite(self.logits).all():
            raise ValueError("MaSS-12 temperature/logits must be finite and valid")
        if np.any(self.decoder < 0) or not np.allclose(
            self.decoder.sum(axis=1), 1.0, atol=1e-10, rtol=0
        ):
            raise ValueError("MaSS-12 common decoder rows must be stochastic")

    def block_probabilities(self, token: int, public_context: Hashable = "all") -> np.ndarray:
        """Inference path: its only data inputs are Z and public (B,S)."""
        context = str(public_context)
        try:
            context_index = self.context_values.index(context)
        except ValueError as error:
            raise ValueError(f"unknown frozen public context: {context}") from error
        if not 0 <= int(token) < self.logits.shape[1]:
            raise ValueError("token is outside the fixed 12-bit alphabet")
        return _softmax(self.logits[context_index, int(token)], self.temperature)

    def channel(self, public_context: Hashable = "all") -> np.ndarray:
        """Enumerate the exact token channel; no sampling occurs."""
        context = str(public_context)
        try:
            context_index = self.context_values.index(context)
        except ValueError as error:
            raise ValueError(f"unknown frozen public context: {context}") from error
        probabilities = _softmax(self.logits[context_index], self.temperature)
        channel = probabilities @ self.decoder
        validate_channel(channel)
        return channel

    def channels(self) -> dict[str, np.ndarray]:
        return {context: self.channel(context) for context in self.context_values}

    def provenance(self) -> dict[str, object]:
        return {
            "method": self.method,
            "display_name": self.display_name,
            "adaptation": "exact finite-output MI penalties with a frozen CAPT decoder",
            "official_code_reused": False,
            "official_equations_preserved": ["I(O;S)<=m", "I(O;U)>=n", "d^2+|d|"],
            "source_split": self.source_split,
            "inference_inputs": ["Z", "B", "S"],
            "uses_sensitive_value_online": self.uses_sensitive_value_online,
            "temperature": self.temperature,
            "seed": self.seed,
            "objective_name": self.objective_name,
        }


def fit_mass12_finite(
    tokens: np.ndarray,
    public_context: np.ndarray | None,
    sensitive_labels: Mapping[str, np.ndarray] | np.ndarray,
    decoder: np.ndarray,
    cost: np.ndarray,
    *,
    useful_labels: Mapping[str, np.ndarray] | np.ndarray | None = None,
    loss_m: Mapping[str, float] | float = 0.0,
    loss_n: Mapping[str, float] | float | None = None,
    privacy_weight: float = 1.0,
    utility_weight: float = 1.0,
    seed: int = 0,
    epochs: int = 100,
    learning_rate: float = 0.05,
    temperature: float = 1.0,
    source_split: str = "D_design",
    objective_name: str = "teacher_kl",
    frozen_context_values: Sequence[Hashable] | None = None,
) -> Mass12FiniteChannel:
    """Fit the finite-output MaSS adaptation using D_design only.

    Sensitive values are training labels.  The returned inference object has no
    sensitive-label argument and computes Q(o|z,b,s) exactly as pi @ decoder.
    """
    if source_split != "D_design":
        raise ValueError("MaSS-12 training/tuning must use D_design, never D_cert or D_test")
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if privacy_weight < 0 or utility_weight < 0 or privacy_weight + utility_weight <= 0:
        raise ValueError("MaSS loss weights must be nonnegative with positive total weight")

    token_array = np.asarray(tokens, dtype=int)
    if token_array.ndim != 1 or len(token_array) == 0:
        raise ValueError("tokens must be a nonempty one-dimensional D_design array")
    decoder_array = np.asarray(decoder, dtype=float)
    if (
        decoder_array.ndim != 2
        or not np.isfinite(decoder_array).all()
        or np.any(decoder_array < 0)
    ):
        raise ValueError("decoder must be a nonnegative (L,K) matrix")
    if not np.allclose(decoder_array.sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("every decoder row must sum to one")
    block_count, token_count = decoder_array.shape
    if token_array.min() < 0 or token_array.max() >= token_count:
        raise ValueError("D_design tokens fall outside the fixed decoder alphabet")
    cost_array = np.asarray(cost, dtype=float)
    if cost_array.shape != (token_count, token_count) or not np.isfinite(cost_array).all():
        raise ValueError("cost must be a finite K by K matrix")

    if public_context is None:
        contexts = np.full(len(token_array), "all", dtype=str)
    else:
        contexts = np.asarray(public_context).astype(str)
        if contexts.shape != token_array.shape:
            raise ValueError("public_context must align with D_design tokens")
    observed_contexts = set(np.unique(contexts).tolist())
    if frozen_context_values is None:
        context_values = tuple(sorted(observed_contexts))
    else:
        context_values = tuple(sorted({str(value) for value in frozen_context_values}))
        missing_contexts = sorted(observed_contexts - set(context_values))
        if missing_contexts:
            raise ValueError(
                "D_design contains public contexts outside the frozen vocabulary: "
                f"{missing_contexts}"
            )
        if not context_values:
            raise ValueError("frozen_context_values must not be empty")
    context_index = {value: index for index, value in enumerate(context_values)}

    sensitive = _label_mapping(
        sensitive_labels, default_name="sensitive", row_count=len(token_array)
    )
    useful = _label_mapping(useful_labels, default_name="useful", row_count=len(token_array))
    m_values = _threshold_mapping(loss_m, tuple(sensitive), default=0.0)
    n_values = _threshold_mapping(loss_n, tuple(useful)) if useful else {}
    check_mass_operational_bounds(sensitive, useful, m_values, n_values)

    token_context_mass = np.zeros((len(context_values), token_count), dtype=float)
    for token, context in zip(token_array, contexts, strict=True):
        token_context_mass[context_index[context], token] += 1 / len(token_array)
    sensitive_joint = {
        name: _joint_context_label_token(
            token_array, contexts, labels, context_values, token_count
        )
        for name, labels in sensitive.items()
    }
    useful_joint = {
        name: _joint_context_label_token(
            token_array, contexts, labels, context_values, token_count
        )
        for name, labels in useful.items()
    }
    block_cost = cost_array @ decoder_array.T

    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1e-3, size=(len(context_values), token_count, block_count))
    first_moment = np.zeros_like(logits)
    second_moment = np.zeros_like(logits)
    history: list[Mass12Epoch] = []
    for epoch in range(1, epochs + 1):
        probabilities = _softmax(logits, temperature)
        distortion = float(np.sum(token_context_mass[:, :, None] * probabilities * block_cost))
        probability_gradient = utility_weight * token_context_mass[:, :, None] * block_cost
        total_loss = utility_weight * distortion

        sensitive_mi: dict[str, float] = {}
        for name, joint in sensitive_joint.items():
            information, information_gradient = _conditional_mi_and_gradient(
                joint, probabilities, decoder_array
            )
            sensitive_mi[name] = information
            violation = max(information - m_values[name], 0.0)
            total_loss += privacy_weight * (violation * violation + violation)
            if violation > 0:
                probability_gradient += (
                    privacy_weight * (2 * violation + 1) * information_gradient
                )

        useful_mi: dict[str, float] = {}
        for name, joint in useful_joint.items():
            information, information_gradient = _conditional_mi_and_gradient(
                joint, probabilities, decoder_array
            )
            useful_mi[name] = information
            violation = max(n_values[name] - information, 0.0)
            total_loss += utility_weight * (violation * violation + violation)
            if violation > 0:
                probability_gradient -= (
                    utility_weight * (2 * violation + 1) * information_gradient
                )

        centered_gradient = probability_gradient - np.sum(
            probability_gradient * probabilities, axis=-1, keepdims=True
        )
        logits_gradient = probabilities * centered_gradient / temperature
        first_moment = 0.9 * first_moment + 0.1 * logits_gradient
        second_moment = 0.999 * second_moment + 0.001 * np.square(logits_gradient)
        corrected_first = first_moment / (1 - 0.9**epoch)
        corrected_second = second_moment / (1 - 0.999**epoch)
        logits -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        history.append(
            Mass12Epoch(
                epoch=epoch,
                loss=float(total_loss),
                distortion=distortion,
                sensitive_mi_nats=sensitive_mi,
                useful_mi_nats=useful_mi,
            )
        )

    model = Mass12FiniteChannel(
        logits=logits,
        decoder=decoder_array.copy(),
        context_values=context_values,
        temperature=float(temperature),
        history=tuple(history),
        loss_m=m_values,
        loss_n=n_values,
        seed=int(seed),
        objective_name=str(objective_name),
    )
    for context in model.context_values:
        validate_channel(model.channel(context))
    return model
