from __future__ import annotations

from collections.abc import Callable

import numpy as np


def validate_decoder(decoder: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    decoder = np.asarray(decoder, dtype=float)
    assignment = np.asarray(assignment, dtype=int)
    if np.any(decoder < 0) or not np.allclose(decoder.sum(axis=1), 1):
        raise ValueError("decoder rows must be distributions")
    for block in range(decoder.shape[0]):
        if np.any(decoder[block, assignment != block] > 1e-12):
            raise ValueError("decoder may only emit tokens inside its block")
    return decoder


def uniform_within_block(assignment: np.ndarray, frequencies: np.ndarray, **_: object) -> np.ndarray:
    del frequencies
    l_count = int(np.max(assignment)) + 1
    decoder = np.zeros((l_count, len(assignment)))
    for block in range(l_count):
        members = np.flatnonzero(assignment == block)
        decoder[block, members] = 1 / len(members)
    return validate_decoder(decoder, assignment)


def design_frequency(
    assignment: np.ndarray,
    frequencies: np.ndarray,
    *,
    uniform_floor: float = 1e-8,
    **_: object,
) -> np.ndarray:
    l_count = int(np.max(assignment)) + 1
    decoder = np.zeros((l_count, len(assignment)))
    for block in range(l_count):
        members = np.flatnonzero(assignment == block)
        mass = np.asarray(frequencies, dtype=float)[members] + uniform_floor
        decoder[block, members] = mass / mass.sum()
    return validate_decoder(decoder, assignment)


def pi0_conditional(assignment: np.ndarray, frequencies: np.ndarray, **kwargs) -> np.ndarray:
    return design_frequency(assignment, frequencies, **kwargs)


def utility_medoid(
    assignment: np.ndarray,
    frequencies: np.ndarray,
    *,
    scores: np.ndarray,
    **_: object,
) -> np.ndarray:
    l_count = int(np.max(assignment)) + 1
    decoder = np.zeros((l_count, len(assignment)))
    for block in range(l_count):
        members = np.flatnonzero(assignment == block)
        weights = np.asarray(frequencies)[members]
        weights = weights / weights.sum() if weights.sum() else np.ones(len(members)) / len(members)
        mean = weights @ np.asarray(scores)[members]
        medoid = members[np.argmin(abs(np.asarray(scores)[members] - mean))]
        decoder[block, medoid] = 1.0
    return validate_decoder(decoder, assignment)


DECODER_REGISTRY: dict[str, Callable] = {
    "uniform_within_block": uniform_within_block,
    "pi0_conditional": pi0_conditional,
    "design_frequency": design_frequency,
    "utility_medoid": utility_medoid,
    "point_mass": utility_medoid,
}


def build_decoder(name: str, assignment: np.ndarray, frequencies: np.ndarray, **kwargs) -> np.ndarray:
    split_id = kwargs.pop("split_id", "D_design")
    if split_id != "D_design":
        raise ValueError("decoders may only be fit on D_design")
    return DECODER_REGISTRY[name](assignment, frequencies, **kwargs)
