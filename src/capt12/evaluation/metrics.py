from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def target_ctr_weights(labels: np.ndarray, target_ctr: float) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    if not 0 < target_ctr < 1:
        raise ValueError("target CTR must be in (0,1)")
    empirical = y.mean()
    if empirical <= 0 or empirical >= 1:
        return np.ones(len(y))
    weights = np.where(y == 1, target_ctr / empirical, (1 - target_ctr) / (1 - empirical))
    return weights / weights.mean()


def expected_channel_metrics(
    channel: np.ndarray,
    input_weights: np.ndarray,
    cost: np.ndarray,
    token_to_block: np.ndarray | None = None,
) -> dict[str, float]:
    q = np.asarray(channel, dtype=float)
    weights = np.asarray(input_weights, dtype=float)
    weights = weights / weights.sum()
    result = {
        "expected_distortion": float(np.sum(weights[:, None] * q * cost)),
        "exact_token_retention": float(weights @ np.diag(q)),
    }
    if token_to_block is None:
        result["block_retention"] = result["exact_token_retention"]
    else:
        assignment = np.asarray(token_to_block, dtype=int)
        same = assignment[:, None] == assignment[None, :]
        result["block_retention"] = float(np.sum(weights[:, None] * q * same))
    return result


def ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15, weights: np.ndarray | None = None) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    value = 0.0
    for idx in range(bins):
        mask = indices == idx
        if not mask.any():
            continue
        mass = w[mask].sum() / w.sum()
        value += mass * abs(np.average(y[mask], weights=w[mask]) - np.average(p[mask], weights=w[mask]))
    return float(value)


def prediction_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    weighted_loss = float(log_loss(y, p, sample_weight=w, labels=[0, 1]))
    unweighted_loss = float(log_loss(y, p, labels=[0, 1]))
    base = np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6)
    null_loss = float(log_loss(y, np.full(len(y), base), sample_weight=w, labels=[0, 1]))
    try:
        if len(np.unique(y)) < 2:
            raise ValueError("AUC is undefined for a one-class test split")
        roc = float(roc_auc_score(y, p, sample_weight=w))
        pr = float(average_precision_score(y, p, sample_weight=w))
    except ValueError:
        roc = math.nan
        pr = math.nan
    key = lambda name: f"{prefix}{name}"  # noqa: E731
    return {
        key("weighted_log_loss"): weighted_loss,
        key("unweighted_log_loss"): unweighted_loss,
        key("LLHCompVN"): (
            1 - weighted_loss / null_loss
            if null_loss > 0 and len(np.unique(y)) == 2
            else math.nan
        ),
        key("calibration_ratio"): float(np.average(y, weights=w) / max(np.average(p, weights=w), 1e-12)),
        key("ECE"): ece(y, p, weights=w),
        key("ROC_AUC"): roc,
        key("PR_AUC"): pr,
    }


def evaluate_target_ctrs(labels: np.ndarray, probabilities: np.ndarray, targets: list[float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for target in targets:
        result.update(
            prediction_metrics(
                labels,
                probabilities,
                weights=target_ctr_weights(labels, target),
                prefix=f"target_ctr_{target:g}_",
            )
        )
    return result
