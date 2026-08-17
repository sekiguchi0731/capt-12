from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass
class ConfidenceBox:
    lower: np.ndarray
    upper: np.ndarray
    nominal: np.ndarray
    method: str
    alpha_familywise: float
    tv_radius: float = 0.0
    experimental: bool = False

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.nominal = np.asarray(self.nominal, dtype=float)
        if not (self.lower.shape == self.upper.shape == self.nominal.shape):
            raise ValueError("confidence box arrays must have equal shapes")
        if np.any(self.lower < 0) or np.any(self.upper > 1) or np.any(self.lower > self.upper):
            raise ValueError("invalid confidence bounds")
        if self.lower.sum() > 1 + 1e-10 or self.upper.sum() < 1 - 1e-10:
            raise ValueError("box does not intersect the simplex")
        if not np.isclose(self.nominal.sum(), 1.0):
            raise ValueError("nominal distribution must sum to one")
        if not 0 <= self.tv_radius <= 1:
            raise ValueError("TV radius must be in [0,1]")

    def to_dict(self) -> dict:
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "nominal": self.nominal.tolist(),
            "method": self.method,
            "alpha_familywise": self.alpha_familywise,
            "tv_radius": self.tv_radius,
            "experimental": self.experimental,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> ConfidenceBox:
        return cls(**dict(value))


def _allocation(alpha: float, group_count: int, cell_count: int, comparisons: int = 1) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    tests = max(1, group_count * cell_count * comparisons)
    return alpha / tests


def point_box(counts: np.ndarray, **_: object) -> ConfidenceBox:
    counts = np.asarray(counts, dtype=float)
    if counts.sum() <= 0:
        raise ValueError("point confidence requires a nonempty group")
    nominal = counts / counts.sum()
    return ConfidenceBox(nominal, nominal, nominal, "point", 1.0)


def full_simplex_box(counts: np.ndarray, **_: object) -> ConfidenceBox:
    """Represent complete uncertainty for an unobserved conditional group."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("full-simplex confidence requires a nonempty alphabet")
    if np.any(counts != 0):
        raise ValueError("full-simplex completion is only valid for a zero-count group")
    nominal = np.ones(len(counts), dtype=float) / len(counts)
    return ConfidenceBox(
        np.zeros(len(counts), dtype=float),
        np.ones(len(counts), dtype=float),
        nominal,
        "full_simplex",
        1.0,
    )


def cp_box(
    counts: np.ndarray,
    *,
    alpha: float = 0.05,
    group_count: int = 1,
    comparisons: int = 1,
    tv_radius: float = 0.0,
) -> ConfidenceBox:
    counts = np.asarray(counts, dtype=int)
    n = int(counts.sum())
    if n <= 0:
        raise ValueError("Clopper-Pearson requires a nonempty group")
    per_cell = _allocation(alpha, group_count, len(counts), comparisons)
    tail = per_cell / 2
    lower = np.where(counts == 0, 0.0, beta.ppf(tail, counts, n - counts + 1))
    upper = np.where(counts == n, 1.0, beta.ppf(1 - tail, counts + 1, n - counts))
    nominal = counts / n
    return ConfidenceBox(lower, upper, nominal, "cp_box", alpha, tv_radius)


def hoeffding_box(
    counts: np.ndarray,
    *,
    alpha: float = 0.05,
    group_count: int = 1,
    comparisons: int = 1,
    tv_radius: float = 0.0,
) -> ConfidenceBox:
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n <= 0:
        raise ValueError("Hoeffding confidence requires a nonempty group")
    per_cell = _allocation(alpha, group_count, len(counts), comparisons)
    radius = np.sqrt(np.log(2 / per_cell) / (2 * n))
    nominal = counts / n
    return ConfidenceBox(
        np.maximum(0.0, nominal - radius),
        np.minimum(1.0, nominal + radius),
        nominal,
        "hoeffding_box",
        alpha,
        tv_radius,
    )


def dp_aware_box(
    noisy_counts: np.ndarray,
    *,
    noise_scale: float,
    alpha: float = 0.05,
    group_count: int = 1,
    comparisons: int = 1,
    tv_radius: float = 0.0,
) -> ConfidenceBox:
    """Experimental conservative tail-plus-Hoeffding construction.

    Noisy counts are never passed to Clopper-Pearson. A simultaneous Laplace
    tail interval is first converted to count bounds, then combined with a
    Hoeffding sampling radius. Certification intentionally rejects this mode.
    """
    noisy = np.asarray(noisy_counts, dtype=float)
    per_cell = _allocation(alpha / 2, group_count, len(noisy), comparisons)
    noise_tail = noise_scale * np.log(1 / per_cell)
    low_count = np.maximum(0.0, noisy - noise_tail)
    high_count = np.maximum(low_count, noisy + noise_tail)
    n_low = max(1.0, low_count.sum())
    n_high = max(n_low, high_count.sum())
    nominal_counts = np.maximum(0.0, noisy)
    if nominal_counts.sum() == 0:
        nominal_counts[:] = 1.0
    nominal = nominal_counts / nominal_counts.sum()
    sampling = np.sqrt(np.log(4 / per_cell) / (2 * n_low))
    lower = np.maximum(0.0, low_count / n_high - sampling)
    upper = np.minimum(1.0, high_count / n_low + sampling)
    return ConfidenceBox(
        lower,
        upper,
        nominal,
        "dp_aware_box_experimental",
        alpha,
        tv_radius,
        experimental=True,
    )


CONFIDENCE_REGISTRY = {
    "point": point_box,
    "full_simplex": full_simplex_box,
    "cp_box": cp_box,
    "hoeffding_box": hoeffding_box,
    "dp_aware_box": dp_aware_box,
}


def confidence_box_from_counts(
    counts: np.ndarray,
    *,
    confidence: str,
    missing_group_policy: str = "force_cover",
    alpha: float = 0.05,
    group_count: int = 1,
    comparisons: int = 1,
    tv_radius: float = 0.0,
) -> ConfidenceBox:
    """Construct the configured finite-sample or zero-count uncertainty set."""
    values = np.asarray(counts)
    if values.sum() == 0:
        if missing_group_policy != "full_simplex":
            raise ValueError("zero-count groups require missing_group_policy=full_simplex")
        return full_simplex_box(values)
    if confidence not in CONFIDENCE_REGISTRY:
        raise ValueError(f"unknown confidence construction: {confidence}")
    factory = CONFIDENCE_REGISTRY[confidence]
    if confidence == "point":
        return factory(values)
    return factory(
        values,
        alpha=alpha,
        group_count=group_count,
        comparisons=comparisons,
        tv_radius=tv_radius,
    )

