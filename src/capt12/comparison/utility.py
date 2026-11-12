from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from capt12.evaluation.released_ranking import ReleasedRanking, ReleasedRankingAccumulator


def _context_array(
    values: Mapping[str, np.ndarray] | np.ndarray, context: str
) -> np.ndarray:
    return np.asarray(values[context] if isinstance(values, Mapping) else values, dtype=float)


def _channel_array(
    channels: Mapping[str, np.ndarray] | np.ndarray, context: str
) -> np.ndarray:
    return np.asarray(channels[context] if isinstance(channels, Mapping) else channels, dtype=float)


@dataclass(frozen=True)
class FrozenCTRUtility:
    empirical_log_loss: float
    excess_log_loss: float
    teacher_kl: float
    hybrid_objective: float
    roc_auc: float
    pr_auc: float
    distortion_objective: float
    log_loss_ci95_low: float
    log_loss_ci95_high: float
    excess_log_loss_ci95_low: float
    excess_log_loss_ci95_high: float
    row_log_loss: np.ndarray
    row_unsanitized_log_loss: np.ndarray
    mean_score_roc_auc: float
    mean_score_average_precision: float
    released_ranking: ReleasedRanking


@dataclass(frozen=True)
class PairedUtilityDifference:
    mean: float
    ci95_low: float
    ci95_high: float
    bootstrap_replicates: int


def _cluster_bootstrap_mean(
    values: np.ndarray,
    user_day_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    groups = np.asarray(user_day_ids).astype(str)
    if values.shape != groups.shape:
        raise ValueError("bootstrap values and user-day IDs must align")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) == 0 or replicates < 1:
        raise ValueError("cluster bootstrap requires groups and positive replicates")
    totals = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled = rng.integers(0, len(unique), size=len(unique))
        estimates[replicate] = float(totals[sampled].sum() / counts[sampled].sum())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def evaluate_frozen_ctr_utility(
    channels: Mapping[str, np.ndarray] | np.ndarray,
    tokens: np.ndarray,
    public_context: np.ndarray | None,
    labels: np.ndarray,
    service_probabilities: Mapping[str, np.ndarray] | np.ndarray,
    user_day_ids: np.ndarray,
    *,
    distortion_cost: Mapping[str, np.ndarray] | np.ndarray | None = None,
    hybrid_empirical_weight: float = 0.5,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 0,
    ranking_mc_replicates: int = 8,
    ranking_mc_seed: int = 0,
    source_groups: Mapping[str, np.ndarray] | np.ndarray | None = None,
    split_name: str = "D_test",
) -> FrozenCTRUtility:
    """Evaluate released-score CTR utility for the supplied finite channel.

    ROC integrates ranks; AP uses finite independent release histograms. A
    neural caller's Gaussian-channel approximation remains a separate error.
    """
    if split_name != "D_test":
        raise ValueError("privacy-comparison utility is evaluated on D_test only")
    if not 0 <= hybrid_empirical_weight <= 1:
        raise ValueError("hybrid_empirical_weight must be in [0,1]")
    token_array = np.asarray(tokens, dtype=int)
    label_array = np.asarray(labels, dtype=int)
    groups = np.asarray(user_day_ids)
    contexts = (
        np.full(len(token_array), "all", dtype=str)
        if public_context is None
        else np.asarray(public_context).astype(str)
    )
    if not (
        token_array.ndim == 1
        and token_array.shape == label_array.shape == groups.shape == contexts.shape
        and len(token_array) > 0
    ):
        raise ValueError("D_test utility arrays must be nonempty and aligned")
    if not set(np.unique(label_array)).issubset({0, 1}):
        raise ValueError("CTR labels must be binary")

    row_loss = np.empty(len(token_array), dtype=float)
    row_unsanitized = np.empty(len(token_array), dtype=float)
    row_teacher_kl = np.empty(len(token_array), dtype=float)
    row_distortion = np.empty(len(token_array), dtype=float)
    expected_scores = np.empty(len(token_array), dtype=float)
    ranking = ReleasedRankingAccumulator(replicates=ranking_mc_replicates, seed=ranking_mc_seed)
    for context in sorted(np.unique(contexts)):
        mask = contexts == context
        channel = _channel_array(channels, context)
        probabilities = np.clip(_context_array(service_probabilities, context), 1e-12, 1 - 1e-12)
        if channel.shape != (len(probabilities), len(probabilities)):
            raise ValueError(f"channel/service alphabet mismatch for context {context}")
        context_tokens = token_array[mask]
        if context_tokens.min() < 0 or context_tokens.max() >= len(probabilities):
            raise ValueError("D_test token is outside the frozen service alphabet")
        labels_here = label_array[mask]
        loss_one = channel @ -np.log(probabilities)
        loss_zero = channel @ -np.log1p(-probabilities)
        row_loss[mask] = np.where(labels_here == 1, loss_one[context_tokens], loss_zero[context_tokens])
        original = probabilities[context_tokens]
        row_unsanitized[mask] = np.where(
            labels_here == 1, -np.log(original), -np.log1p(-original)
        )
        teacher_by_token = probabilities * (np.log(probabilities) + loss_one) + (
            1 - probabilities) * (np.log1p(-probabilities) + loss_zero)
        row_teacher_kl[mask] = np.maximum(teacher_by_token[context_tokens], 0)
        expected_scores[mask] = (channel @ probabilities)[context_tokens]
        ranking.add_context(channel, probabilities, context_tokens, labels_here,
                            source_groups=None if source_groups is None else np.asarray(
                                source_groups[context] if isinstance(source_groups, Mapping) else source_groups))
        if distortion_cost is None:
            row_distortion[mask] = row_teacher_kl[mask]
        else:
            cost = _context_array(distortion_cost, context)
            if cost.shape != channel.shape:
                raise ValueError("distortion cost does not match channel")
            row_distortion[mask] = np.einsum("ij,ij->i", channel, cost)[context_tokens]

    try:
        roc = float(roc_auc_score(label_array, expected_scores)) if len(np.unique(label_array)) == 2 else float("nan")
        pr = float(average_precision_score(label_array, expected_scores)) if label_array.any() else 0.0
    except ValueError:
        roc = float("nan")
        pr = float("nan")
    low, high = _cluster_bootstrap_mean(
        row_loss, groups, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    excess_low, excess_high = _cluster_bootstrap_mean(
        row_loss - row_unsanitized,
        groups,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    empirical = float(row_loss.mean())
    teacher = float(row_teacher_kl.mean())
    released = ranking.finish()
    return FrozenCTRUtility(
        empirical_log_loss=empirical,
        excess_log_loss=float((row_loss - row_unsanitized).mean()),
        teacher_kl=teacher,
        hybrid_objective=float(
            hybrid_empirical_weight * empirical
            + (1 - hybrid_empirical_weight) * teacher
        ),
        roc_auc=released.roc_auc,
        pr_auc=released.average_precision,
        distortion_objective=float(row_distortion.mean()),
        log_loss_ci95_low=low,
        log_loss_ci95_high=high,
        excess_log_loss_ci95_low=excess_low,
        excess_log_loss_ci95_high=excess_high,
        row_log_loss=row_loss,
        row_unsanitized_log_loss=row_unsanitized,
        mean_score_roc_auc=roc,
        mean_score_average_precision=pr,
        released_ranking=released,
    )


def paired_cluster_bootstrap_difference(
    baseline_row_values: np.ndarray,
    capt_row_values: np.ndarray,
    user_day_ids: np.ndarray,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
) -> PairedUtilityDifference:
    baseline = np.asarray(baseline_row_values, dtype=float)
    capt = np.asarray(capt_row_values, dtype=float)
    if baseline.shape != capt.shape:
        raise ValueError("paired method rows must align")
    difference = baseline - capt
    low, high = _cluster_bootstrap_mean(
        difference,
        np.asarray(user_day_ids),
        replicates=bootstrap_replicates,
        seed=seed,
    )
    return PairedUtilityDifference(
        mean=float(difference.mean()),
        ci95_low=low,
        ci95_high=high,
        bootstrap_replicates=bootstrap_replicates,
    )
