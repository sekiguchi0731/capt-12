"""Binary attribute-inference metrics at an explicitly chosen common prior.

Curves describe frozen, learned score thresholds on held-out releases. They
are not the population-optimal Neyman--Pearson curve or a privacy guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta


def precision_recall(curve: pd.DataFrame, prior: float = 0.5) -> pd.DataFrame:
    if not np.isfinite(prior) or not 0 < prior < 1:
        raise ValueError("positive-hypothesis prior must be in (0, 1)")
    result = curve.copy()
    tpr, fpr = result["tpr"].to_numpy(float), result["fpr"].to_numpy(float)
    if (
        not np.isfinite(tpr).all()
        or not np.isfinite(fpr).all()
        or np.any((tpr < 0) | (tpr > 1) | (fpr < 0) | (fpr > 1))
    ):
        raise ValueError("ROC rates must be finite probabilities")
    denominator = prior * tpr + (1 - prior) * fpr
    result["recall"] = tpr
    result["precision"] = np.divide(
        prior * tpr, denominator, out=np.full(len(result), np.nan), where=denominator > 0
    )
    result["positive_prior"] = prior
    return result


def binary_attack_metrics(curve: pd.DataFrame, prior: float = 0.5) -> dict[str, float]:
    pr = precision_recall(curve, prior)
    tpr, fpr = pr["tpr"].to_numpy(float), pr["fpr"].to_numpy(float)
    if (
        len(pr) < 2
        or not np.allclose([tpr[0], fpr[0]], 0)
        or not np.allclose([tpr[-1], fpr[-1]], 1)
        or np.any(np.diff(tpr) < -1e-12)
        or np.any(np.diff(fpr) < -1e-12)
    ):
        raise ValueError("attack curve must be complete and ordered by descending threshold")
    # This Bayes rule is fixed by the TRAINED log likelihood ratio and prior;
    # test labels are not used to optimize its threshold.
    boundary = np.log1p(-prior) - np.log(prior)
    index = np.flatnonzero(pr["threshold"].to_numpy(float) >= boundary)[-1]
    accuracy = prior * tpr[index] + (1 - prior) * (1 - fpr[index])
    baseline = max(prior, 1 - prior)
    return {
        "roc_auc": float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2)),
        "average_precision": float(
            np.sum(np.diff(tpr) * np.nan_to_num(pr["precision"].to_numpy()[1:]))
        ),
        "prior_rule_accuracy": float(accuracy),
        "pair_eau": float(accuracy),
        "prior_only_accuracy": baseline,
        "accuracy_uplift": float(accuracy - baseline),
        # A descriptive, test-selected maximum, NOT a calibrated operating point.
        "max_threshold_advantage": float(np.max(tpr - fpr)),
    }


def fpr_at_recall(curve: pd.DataFrame, recall: np.ndarray) -> np.ndarray:
    """Randomize between adjacent frozen thresholds; retain vertical ROC edges.

    At a repeated recall, take the first (smallest FPR) point. Immediately to
    its right, the segment starts at the last point with that recall. Removing
    duplicate recalls and interpolating would invent an unattainable curve.
    """
    r = np.asarray(recall, dtype=float)
    if np.any((r < 0) | (r > 1)):
        raise ValueError("recall must lie in [0, 1]")
    y, x = curve["tpr"].to_numpy(float), curve["fpr"].to_numpy(float)
    upper = np.searchsorted(y, r, side="left")
    upper = np.minimum(upper, len(y) - 1)
    lower = np.maximum(upper - 1, 0)
    dy = y[upper] - y[lower]
    fraction = np.divide(r - y[lower], dy, out=np.zeros_like(r), where=dy > 0)
    return x[lower] + fraction * (x[upper] - x[lower])


def precision_at_recall(curve: pd.DataFrame, recall: np.ndarray, prior: float) -> np.ndarray:
    fpr = fpr_at_recall(curve, recall)
    r = np.asarray(recall, float)
    denominator = prior * r + (1 - prior) * fpr
    return np.divide(prior * r, denominator, out=np.full_like(r, np.nan), where=denominator > 0)


def pr_plot_points(curve: pd.DataFrame, prior: float = 0.5) -> pd.DataFrame:
    """PR interpolation is nonlinear; interpolate ROC decision mixtures first."""
    knots = curve["tpr"].to_numpy(float)
    recall = np.unique(
        np.r_[
            np.geomspace(1e-8, 1, 300),
            np.linspace(0.001, 1, 400),
            knots,
            np.nextafter(knots[knots > 0], 1),
        ]
    )
    recall = recall[(recall > 0) & (recall <= 1)]
    return pd.DataFrame({"recall": recall, "precision": precision_at_recall(curve, recall, prior)})


def compare_pr_curves(
    capt: pd.DataFrame, other: pd.DataFrame, prior: float, tolerance: float = 1e-10
) -> dict[str, float | bool]:
    # Check every knot AND both one-sided limits, including near recall zero.
    # On open intervals FPR differences are affine, so endpoint checks suffice
    # for an order statement (without using a coarse plotting grid).
    knots = np.unique(np.r_[capt["tpr"], other["tpr"], 0.0, 1.0])
    probes = np.unique(
        np.r_[
            knots,
            np.nextafter(knots[knots > 0], 0),
            np.nextafter(knots[knots > 0], 1),
            (knots[:-1] + knots[1:]) / 2,
        ]
    )
    probes = probes[(probes > 0) & (probes <= 1)]
    cp = precision_at_recall(capt, probes, prior)
    op = precision_at_recall(other, probes, prior)
    difference = cp - op
    index = int(np.argmax(difference))

    # Check the exact affine FPR order at all left/right limits. Precision
    # probes supply readable witnesses; they do not decide dominance.
    def limits(curve, right):
        y, x = curve["tpr"].to_numpy(float), curve["fpr"].to_numpy(float)
        positions = np.searchsorted(y, knots, side="right" if right else "left")
        if right:
            positions -= 1
        positions = np.clip(positions, 0, len(y) - 1)
        answer = fpr_at_recall(curve, knots)
        hit = y[positions] == knots
        answer[hit] = x[positions[hit]]
        return answer

    gaps = np.r_[
        (limits(other, False) - limits(capt, False))[knots > 0],
        (limits(other, True) - limits(capt, True))[knots < 1],
    ]
    return {
        "capt_pr_no_higher": bool(gaps.max() <= tolerance),
        "max_other_minus_capt_fpr": float(gaps.max()),
        "max_capt_minus_other_precision": float(difference[index]),
        "witness_recall": float(probes[index]),
        "min_capt_minus_other_precision": float(difference.min()),
    }


def probability_intervals(
    successes: np.ndarray, trials: np.ndarray, tail_alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    s, n = np.asarray(successes), np.asarray(trials)
    if not 0 < tail_alpha < 0.5 or np.any(n <= 0) or np.any(s < 0) or np.any(s > n):
        raise ValueError("invalid binomial counts or confidence allocation")
    low, high = np.zeros(s.shape), np.ones(s.shape)
    selected = s > 0
    low[selected] = beta.ppf(tail_alpha, s[selected], n[selected] - s[selected] + 1)
    selected = s < n
    high[selected] = beta.ppf(1 - tail_alpha, s[selected] + 1, n[selected] - s[selected])
    return low, high


def rare_score_buckets(
    curve: pd.DataFrame, positive: int, negative: int, max_buckets: int = 16
) -> pd.DataFrame:
    """Choose extreme learned-score buckets, before inspecting their test counts."""
    if max_buckets < 2:
        raise ValueError("at least two score buckets must be considered")
    size = len(curve) - 1
    half = max_buckets // 2
    indices = np.unique(
        np.r_[np.arange(min(half, size)), np.arange(max(0, size - (max_buckets - half)), size)]
    )
    return pd.DataFrame(
        {
            "score": curve["threshold"].to_numpy()[1:][indices],
            "positive_successes": np.diff(curve["tp"].to_numpy())[indices],
            "negative_successes": np.diff(curve["fp"].to_numpy())[indices],
            "positive_trials": positive,
            "negative_trials": negative,
        }
    )


def audit_score_buckets(events: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Simultaneous one-sided D-infinity witnesses across ALL supplied events.

    A zero empirical denominator is reported as such, never as infinite true
    epsilon. The bounds require independent test user-days within each group.
    """
    if not 0 < confidence < 1:
        raise ValueError("audit confidence must be in (0, 1)")
    result = events.copy()
    if result.empty:
        return result
    tail = (1 - confidence) / (4 * len(result))
    for label in ("positive", "negative"):
        s, n = result[f"{label}_successes"].to_numpy(), result[f"{label}_trials"].to_numpy()
        low, high = probability_intervals(s, n, tail)
        result[f"{label}_probability"] = s / n
        result[f"{label}_ci_low"], result[f"{label}_ci_high"] = low, high
    p, q = result["positive_probability"], result["negative_probability"]
    # NaN explicitly denotes an unobserved empirical denominator.
    result["empirical_log_ratio"] = np.log(
        np.divide(p, q, out=np.full(len(p), np.nan), where=(p > 0) & (q > 0))
    )
    result["zero_negative_count"] = result["negative_successes"].eq(0)
    result["zero_positive_count"] = result["positive_successes"].eq(0)
    for name, numerator, denominator in (
        ("forward", result["positive_ci_low"], result["negative_ci_high"]),
        ("reverse", result["negative_ci_low"], result["positive_ci_high"]),
    ):
        ratio = np.divide(numerator, denominator, out=np.zeros(len(result)), where=denominator > 0)
        result[f"{name}_epsilon_lower"] = np.log(np.maximum(1, ratio))
    result["epsilon_lower"] = result[["forward_epsilon_lower", "reverse_epsilon_lower"]].max(axis=1)
    result["confidence"] = confidence
    result["simultaneous_event_count"] = len(result)
    result["tail_alpha"] = tail
    result["true_infinite_epsilon_proven"] = False
    return result
