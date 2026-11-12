"""Ranking of independently released scores, conditional on a fixed test set.

ROC AUC integrates pairwise ranks exactly for the supplied finite channel.
Finite-sample AP is averaged over independent integer release histograms;
AP of fractional/expected counts is a different estimand and is not used.
If the supplied channel approximates a neural Gaussian mixture, that separate
channel-approximation error is not removed by exact pairwise integration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

RANKING_DEFINITION = "independent_released_scores_v1"

RANKING_RESULT_COLUMNS = (
    "ctr_ranking_definition",
    "ctr_channel_estimator",
    "released_roc_auc",
    "released_average_precision",
    "ctr_ap_mc_replicates",
    "ctr_ap_mc_seed",
    "ctr_ap_mc_std",
    "ctr_ap_mc_se",
    "ctr_ap_mc_values",
    "mean_score_roc_auc",
    "mean_score_average_precision",
)


@dataclass(frozen=True)
class ReleasedRanking:
    roc_auc: float
    average_precision: float
    ap_mc_values: np.ndarray
    ap_mc_std: float
    ap_mc_se: float
    mc_seed: int

    @property
    def mc_replicates(self) -> int:
        return len(self.ap_mc_values)


def ranking_metadata(result: ReleasedRanking) -> dict:
    return {
        "ctr_ranking_definition": RANKING_DEFINITION,
        "released_roc_auc": result.roc_auc,
        "released_average_precision": result.average_precision,
        "ctr_ap_mc_replicates": result.mc_replicates,
        "ctr_ap_mc_seed": result.mc_seed,
        "ctr_ap_mc_std": result.ap_mc_std,
        "ctr_ap_mc_se": result.ap_mc_se,
        "ctr_ap_mc_values": json.dumps(result.ap_mc_values.tolist()),
    }


def _score_counts(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray):
    """Aggregate equal scores across contexts/tokens, preserving exact ties."""
    levels, inverse = np.unique(scores, return_inverse=True)
    pos = np.bincount(inverse, weights=positive, minlength=len(levels))
    neg = np.bincount(inverse, weights=negative, minlength=len(levels))
    return levels, inverse, pos, neg


def weighted_released_auc(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    _, _, pos, neg = _score_counts(scores, positive, negative)
    if pos.sum() <= 0 or neg.sum() <= 0:
        return float("nan")
    strictly_below = np.cumsum(neg) - neg
    return float(
        np.clip(np.sum(pos * (strictly_below + 0.5 * neg)) / (pos.sum() * neg.sum()), 0, 1)
    )


def _histogram_ap(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    # Arrays have shape (replicates, scores), with scores in ascending order.
    pos, neg = positive[:, ::-1], negative[:, ::-1]
    cumulative_positive = np.cumsum(pos, axis=1)
    cumulative_total = np.cumsum(pos + neg, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive, dtype=float),
        where=cumulative_total > 0,
    )
    denominator = pos.sum(axis=1)
    return np.divide(
        np.sum(pos * precision, axis=1),
        denominator,
        out=np.zeros(len(pos), dtype=float),
        where=denominator > 0,
    )


class ReleasedRankingAccumulator:
    """Stream contexts without materializing per-display output probabilities."""

    def __init__(self, *, replicates: int = 8, seed: int = 0) -> None:
        if replicates < 2 or seed < 0:
            raise ValueError(
                "CTR ranking MC requires at least two replicates and a nonnegative seed"
            )
        self.replicates, self.seed = int(replicates), int(seed)
        self.rng = np.random.default_rng(seed)
        self.scores: list[np.ndarray] = []
        self.expected_positive: list[np.ndarray] = []
        self.expected_negative: list[np.ndarray] = []
        self.sampled_positive: list[np.ndarray] = []
        self.sampled_negative: list[np.ndarray] = []

    def add_context(
        self,
        channel: np.ndarray,
        scores: np.ndarray,
        tokens: np.ndarray,
        labels: np.ndarray,
        *,
        source_groups: np.ndarray | None = None,
    ) -> None:
        q, scores = np.asarray(channel, dtype=float), np.asarray(scores, dtype=float)
        tokens, labels = np.asarray(tokens), np.asarray(labels)
        if (
            q.ndim != 2
            or q.shape[1:] != scores.shape
            or not np.isfinite(q).all()
            or np.any(q < 0)
            or not np.allclose(q.sum(1), 1, atol=1e-10, rtol=0)
            or not np.isfinite(scores).all()
        ):
            raise ValueError("released ranking requires finite scores and a row-stochastic channel")
        if (
            tokens.ndim != 1
            or tokens.shape != labels.shape
            or tokens.dtype.kind not in "iu"
            or np.any(tokens < 0)
            or np.any(tokens >= q.shape[0])
            or not set(np.unique(labels)).issubset({0, 1})
        ):
            raise ValueError("released ranking tokens and binary labels must align")
        counts = [np.bincount(tokens[labels == label], minlength=q.shape[0]) for label in (1, 0)]
        if source_groups is not None:
            groups = np.asarray(source_groups)
            if groups.shape != (q.shape[0],):
                raise ValueError("source groups must align with channel input rows")
            _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
            # Source compression is valid only for identical channel rows.
            for group, representative in enumerate(first):
                if not np.all(q[inverse == group] == q[representative]):
                    raise ValueError("source groups combine different release distributions")
            q = q[first]
            counts = [
                np.bincount(inverse, weights=values, minlength=len(first)).astype(np.int64)
                for values in counts
            ]
        expected = [values @ q for values in counts]
        sampled = []
        for values in counts:
            histogram = np.zeros((self.replicates, q.shape[1]), dtype=np.int64)
            for index in np.flatnonzero(values):
                size = int(values[index])
                support = np.flatnonzero(q[index] > 0)
                probability = q[index, support]
                probability = probability / probability.sum()
                if len(support) == 1:
                    histogram[:, support[0]] += size
                elif size >= len(support):
                    histogram[:, support] += self.rng.multinomial(
                        size, probability, size=self.replicates
                    )
                else:
                    # Sparse groups need O(n) draws rather than K binomial draws.
                    cumulative = np.cumsum(probability)
                    cumulative[-1] = 1
                    draws = support[
                        np.searchsorted(
                            cumulative, self.rng.random((self.replicates, size)), side="right"
                        )
                    ]
                    np.add.at(histogram, (np.arange(self.replicates)[:, None], draws), 1)
            sampled.append(histogram)
        self.scores.append(scores.copy())
        self.expected_positive.append(expected[0])
        self.expected_negative.append(expected[1])
        self.sampled_positive.append(sampled[0])
        self.sampled_negative.append(sampled[1])

    def finish(self) -> ReleasedRanking:
        if not self.scores:
            raise ValueError("released ranking has no test contexts")
        scores = np.concatenate(self.scores)
        expected_pos = np.concatenate(self.expected_positive)
        expected_neg = np.concatenate(self.expected_negative)
        _, inverse = np.unique(scores, return_inverse=True)
        size = int(inverse.max()) + 1
        aggregated = []
        for tables in (self.sampled_positive, self.sampled_negative):
            counts = np.concatenate(tables, axis=1)
            aggregated.append(
                np.stack([np.bincount(inverse, weights=row, minlength=size) for row in counts])
            )
        values = _histogram_ap(*aggregated)
        deviation = 0.0 if np.ptp(values) == 0 else float(values.std(ddof=1))
        return ReleasedRanking(
            roc_auc=weighted_released_auc(scores, expected_pos, expected_neg),
            average_precision=float(values.mean()),
            ap_mc_values=values,
            ap_mc_std=deviation,
            ap_mc_se=deviation / np.sqrt(self.replicates),
            mc_seed=self.seed,
        )
