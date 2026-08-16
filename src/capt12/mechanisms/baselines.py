from __future__ import annotations

import math

import numpy as np

from capt12.mechanisms.lp import validate_channel


def raw_identity(k: int) -> np.ndarray:
    return np.eye(k)


def common_cover(distribution: np.ndarray) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=float)
    if distribution.ndim != 1 or np.any(distribution < 0) or distribution.sum() <= 0:
        raise ValueError("cover distribution must be nonnegative")
    distribution = distribution / distribution.sum()
    return np.tile(distribution, (len(distribution), 1))


def k_ary_rr(k: int, epsilon: float) -> np.ndarray:
    if k < 1 or epsilon < 0:
        raise ValueError("k >= 1 and epsilon >= 0 are required")
    if k == 1:
        return np.ones((1, 1))
    off = 1.0 / (math.exp(epsilon) + k - 1)
    channel = np.full((k, k), off)
    np.fill_diagonal(channel, math.exp(epsilon) * off)
    validate_channel(channel)
    return channel


def scalar_keep_or_cover(cover: np.ndarray, keep_probability: float) -> np.ndarray:
    if not 0 <= keep_probability <= 1:
        raise ValueError("keep_probability must be in [0,1]")
    channel = keep_probability * np.eye(len(cover)) + (1 - keep_probability) * common_cover(cover)
    validate_channel(channel)
    return channel


def tokenwise_keep_or_cover(cover: np.ndarray, keep_probabilities: np.ndarray) -> np.ndarray:
    cover_channel = common_cover(cover)
    keep = np.asarray(keep_probabilities, dtype=float)
    if keep.shape != (len(cover),) or np.any((keep < 0) | (keep > 1)):
        raise ValueError("one keep probability in [0,1] is required per token")
    channel = keep[:, None] * np.eye(len(cover)) + (1 - keep[:, None]) * cover_channel
    validate_channel(channel)
    return channel


BASELINES = {
    "raw_identity": raw_identity,
    "common_cover": common_cover,
    "k_ary_rr": k_ary_rr,
    "scalar_keep_or_cover": scalar_keep_or_cover,
    "tokenwise_keep_or_cover": tokenwise_keep_or_cover,
}

