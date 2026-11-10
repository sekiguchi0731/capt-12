from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


def sample_source_batches(
    token_context_mass: np.ndarray, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, ...]:
    """Draw independent source negatives within each public context."""
    batches = []
    for weights in token_context_mass:
        total = float(weights.sum())
        batches.append(
            rng.choice(len(weights), size=batch_size, p=weights / total)
            if total > 0 else np.array([], dtype=int)
        )
    return tuple(batches)


def expected_infonce_and_gradient(
    block_probabilities: np.ndarray,
    decoder: np.ndarray,
    token_context_mass: np.ndarray,
    embeddings: np.ndarray,
    source_batches: tuple[np.ndarray, ...],
    temperature: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Expected output-to-source InfoNCE and exact channel/embedding gradients.

    Given a batch Z_1,...,Z_B sampled iid within a public context, classify
    which Z_i generated O using a shared, learned unit-normalized embedding:

        E[ -log exp(sim(Z_i,O)/tau) / sum_j exp(sim(Z_j,O)/tau) ].

    The expectation over O is enumerated exactly through pi @ decoder. This
    is the reverse direction of upstream's symmetric in-batch InfoNCE; source
    batches are stochastic. Duplicate tokens remain genuine sampled negatives.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if temperature <= 0 or not np.isfinite(temperature) or np.any(norms <= 1e-12):
        raise ValueError("InfoNCE requires positive temperature and nonzero embeddings")
    features = embeddings / norms
    active_outputs = np.flatnonzero(decoder.sum(axis=0) > 0)
    output_features = features[active_outputs]
    active_decoder = decoder[:, active_outputs]
    probability_gradient = np.zeros_like(block_probabilities)
    feature_gradient = np.zeros_like(features)
    total_loss = 0.0
    for context, sources in enumerate(source_batches):
        context_mass = float(token_context_mass[context].sum())
        if context_mass <= 0 or not len(sources):
            continue
        source_features = features[sources]
        scores = source_features @ output_features.T / temperature
        shifted = scores - scores.max(axis=0, keepdims=True)
        exponentials = np.exp(shifted)
        normalizer = exponentials.sum(axis=0, keepdims=True)
        negative_log_probability = np.log(normalizer) - shifted
        output_probabilities = block_probabilities[context, sources] @ active_decoder
        scale = context_mass / len(sources)
        total_loss += scale * float(np.sum(output_probabilities * negative_log_probability))
        np.add.at(
            probability_gradient[context], sources,
            scale * (negative_log_probability @ active_decoder.T),
        )
        score_gradient = scale * (
            exponentials / normalizer * output_probabilities.sum(axis=0, keepdims=True)
            - output_probabilities
        )
        np.add.at(feature_gradient, sources, score_gradient @ output_features / temperature)
        feature_gradient[active_outputs] += score_gradient.T @ source_features / temperature
    embedding_gradient = (
        feature_gradient - features * np.sum(feature_gradient * features, axis=1, keepdims=True)
    ) / norms
    return float(total_loss), probability_gradient, embedding_gradient


def sample_output_batches(
    block_probabilities: np.ndarray,
    decoder: np.ndarray,
    source_batches: tuple[np.ndarray, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    """Sample independent output candidates from the current categorical Q."""
    batches = []
    for context, sources in enumerate(source_batches):
        probabilities = block_probabilities[context, sources] @ decoder
        cumulative = probabilities.cumsum(axis=1)
        cumulative[:, -1] = 1.0
        batches.append(np.array([
            np.searchsorted(row, uniform, side="right")
            for row, uniform in zip(cumulative, rng.random(len(sources)), strict=True)
        ], dtype=int))
    return tuple(batches)


def source_to_output_infonce_and_gradient(
    block_probabilities: np.ndarray,
    decoder: np.ndarray,
    token_context_mass: np.ndarray,
    embeddings: np.ndarray,
    source_batches: tuple[np.ndarray, ...],
    output_batches: tuple[np.ndarray, ...],
    temperature: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate source-to-output InfoNCE with unbiased categorical gradients.

    For each sampled output slot j, enumerate every possible replacement O_j
    while fixing the other sampled outputs. This conditional expectation gives
    dE[L]/dQ_j directly, including the effect of each negative's distribution.
    Loss and embedding gradients average these conditional expectations over
    j (Rao-Blackwellization). The remaining output sampling is Monte Carlo;
    this is not the derivative of a fixed sampled-output loss.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if temperature <= 0 or not np.isfinite(temperature) or np.any(norms <= 1e-12):
        raise ValueError("InfoNCE requires positive temperature and nonzero embeddings")
    features = embeddings / norms
    active = np.flatnonzero(decoder.sum(axis=0) > 0)
    active_decoder = decoder[:, active]
    output_features = features[active]
    probability_gradient = np.zeros_like(block_probabilities)
    feature_gradient = np.zeros_like(features)
    total_loss = 0.0
    for context, (sources, outputs) in enumerate(zip(source_batches, output_batches, strict=True)):
        mass = float(token_context_mass[context].sum())
        batch_size = len(sources)
        if mass <= 0 or not batch_size:
            continue
        if batch_size < 2 or outputs.shape != sources.shape:
            raise ValueError("forward InfoNCE requires aligned batches of at least two pairs")
        indices = np.searchsorted(active, outputs)
        if np.any(indices >= len(active)) or not np.array_equal(active[indices], outputs):
            raise ValueError("sampled output is outside decoder support")
        source_features = features[sources]
        scores = source_features @ output_features.T / temperature
        sampled_scores = scores[:, indices]
        positives = np.diag(sampled_scores)
        probabilities = block_probabilities[context, sources] @ active_decoder
        score_gradient = np.zeros_like(scores)
        for slot in range(batch_size):
            others = np.arange(batch_size) != slot
            log_denominator = np.logaddexp(
                logsumexp(sampled_scores[:, others], axis=1)[:, None], scores
            )
            positive_scores = np.broadcast_to(positives[:, None], scores.shape).copy()
            positive_scores[slot] = scores[slot]
            conditional_loss = (log_denominator - positive_scores).mean(axis=0)
            total_loss += mass / batch_size * float(probabilities[slot] @ conditional_loss)
            probability_gradient[context, sources[slot]] += mass * (
                conditional_loss @ active_decoder.T
            )
            q = probabilities[slot]
            score_gradient += q[None, :] * np.exp(scores - log_denominator)
            log_q = np.full_like(q, -np.inf)
            positive = q > 0
            log_q[positive] = np.log(q[positive])
            expected_inverse = logsumexp(log_q[None, :] - log_denominator, axis=1)
            other_probabilities = np.exp(sampled_scores[:, others] + expected_inverse[:, None])
            np.add.at(score_gradient.T, indices[others], other_probabilities.T)
            score_gradient[slot] -= q
            score_gradient[np.flatnonzero(others), indices[others]] -= 1
        score_gradient *= mass / batch_size**2
        np.add.at(feature_gradient, sources, score_gradient @ output_features / temperature)
        feature_gradient[active] += score_gradient.T @ source_features / temperature
    embedding_gradient = (
        feature_gradient - features * np.sum(feature_gradient * features, axis=1, keepdims=True)
    ) / norms
    return float(total_loss), probability_gradient, embedding_gradient


@dataclass(frozen=True)
class InfoNCEResult:
    loss: float
    output_to_source_loss: float
    source_to_output_loss: float
    probability_gradient: np.ndarray
    embedding_gradient: np.ndarray


def infonce_objective_and_gradient(
    block_probabilities: np.ndarray,
    decoder: np.ndarray,
    token_context_mass: np.ndarray,
    embeddings: np.ndarray,
    source_batches: tuple[np.ndarray, ...],
    temperature: float,
    *,
    direction: str = "symmetric",
    output_seed: int = 0,
) -> InfoNCEResult:
    """Select either direction or the upstream-style arithmetic mean of both."""
    if direction not in {"symmetric", "output_to_source", "source_to_output"}:
        raise ValueError("unknown InfoNCE direction")
    reverse_loss = forward_loss = 0.0
    probability_gradient = np.zeros_like(block_probabilities)
    embedding_gradient = np.zeros_like(embeddings)
    if direction in {"symmetric", "output_to_source"}:
        reverse_loss, probability_gradient, embedding_gradient = expected_infonce_and_gradient(
            block_probabilities, decoder, token_context_mass, embeddings, source_batches, temperature
        )
    if direction in {"symmetric", "source_to_output"}:
        outputs = sample_output_batches(
            block_probabilities, decoder, source_batches, np.random.default_rng(output_seed)
        )
        forward_loss, forward_pi_gradient, forward_embedding_gradient = source_to_output_infonce_and_gradient(
            block_probabilities, decoder, token_context_mass, embeddings, source_batches,
            outputs, temperature,
        )
        probability_gradient = probability_gradient + forward_pi_gradient
        embedding_gradient = embedding_gradient + forward_embedding_gradient
    scale = 0.5 if direction == "symmetric" else 1.0
    return InfoNCEResult(
        loss=scale * (reverse_loss + forward_loss),
        output_to_source_loss=reverse_loss,
        source_to_output_loss=forward_loss,
        probability_gradient=scale * probability_gradient,
        embedding_gradient=scale * embedding_gradient,
    )
