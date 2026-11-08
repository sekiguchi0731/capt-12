from __future__ import annotations

import numpy as np


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
