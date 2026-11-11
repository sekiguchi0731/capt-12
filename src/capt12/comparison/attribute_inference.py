"""Replaceable attacks on attributes, conditional on public context/profile.

These are empirical attack measurements, not epsilon estimates or certificates.
Only D_attack_train determines the family, scores, and operating thresholds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain, permutations
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import beta

from capt12.experiments.sampling import stable_hash64
from capt12.privacy.adjacency import Group


class TokenAttack(Protocol):
    """A frozen score for each nominal output token; larger favors the positive group."""

    scores: np.ndarray


@dataclass(frozen=True)
class LikelihoodRatioAttack:
    scores: np.ndarray


def fit_likelihood_ratio(
    positive: np.ndarray,
    negative: np.ndarray,
    k: int,
    smoothing: float,
) -> TokenAttack:
    if k < 1 or not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("K and likelihood-ratio smoothing must be positive")
    if not len(positive) or not len(negative):
        raise ValueError("both attack-fit groups must contain observations")
    counts = [np.bincount(values, minlength=k) for values in (positive, negative)]
    probabilities = [(value + smoothing) / (value.sum() + k * smoothing) for value in counts]
    return LikelihoodRatioAttack(np.log(probabilities[0]) - np.log(probabilities[1]))


# Add a factory here to change attacks without changing sampling, evaluation,
# confidence intervals, artifact schemas, or mechanism training.
ATTACK_FACTORIES: dict[str, Callable[[np.ndarray, np.ndarray, int, float], TokenAttack]] = {
    "likelihood_ratio": fit_likelihood_ratio,
}


def attack_fit_mask(ids: np.ndarray, *, seed: int, calibration_fraction: float) -> np.ndarray:
    if not 0 < calibration_fraction < 1:
        raise ValueError("attack calibration fraction must be in (0, 1)")
    ranks = stable_hash64(ids, seed=seed, namespace="attribute-attack-fit-calibration")
    return ranks.astype(np.float64) / 2.0**64 >= calibration_fraction


def _curve(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> pd.DataFrame:
    """All score thresholds are frozen before test labels are inspected; ties stay together."""
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores) - 1]
    tp = np.r_[0, np.cumsum(np.bincount(positive, minlength=len(scores))[order])[ends]]
    fp = np.r_[0, np.cumsum(np.bincount(negative, minlength=len(scores))[order])[ends]]
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "tp": tp,
            "fp": fp,
            "tpr": tp / len(positive),
            "fpr": fp / len(negative),
        }
    )


def calibrate_operating_point(
    scores: np.ndarray,
    negative: np.ndarray,
    target_fpr: float,
) -> tuple[float, float]:
    """Neyman-Pearson boundary with randomized decisions at tied scores.

    Randomization is part of the *attacker's decision*, never mechanism output.
    It hits target FPR on calibration data; deployment/test FPR is not guaranteed.
    """
    counts = np.bincount(negative, minlength=len(scores))
    levels, inverse = np.unique(scores, return_inverse=True)
    levels = levels[::-1]
    mass = np.bincount(inverse, weights=counts, minlength=len(levels))[::-1]
    boundary = int(np.searchsorted(np.cumsum(mass), target_fpr * len(negative)))
    before = mass[:boundary].sum()
    probability = (target_fpr * len(negative) - before) / mass[boundary]
    return float(levels[boundary]), float(np.clip(probability, 0, 1))


def _interval(successes: int, trials: int, tail_alpha: float) -> tuple[float, float]:
    return (
        0.0 if successes == 0 else float(beta.ppf(tail_alpha, successes, trials - successes + 1)),
        1.0
        if successes == trials
        else float(beta.ppf(1 - tail_alpha, successes + 1, trials - successes)),
    )


@dataclass
class AttributeInferenceResult:
    pairs: pd.DataFrame
    roc: pd.DataFrame
    score_tables: dict[str, np.ndarray]
    summary: dict

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.pairs.to_csv(path / "pairs.csv", index=False)
        self.roc.to_csv(path / "roc.csv", index=False)
        np.savez_compressed(path / "attacker_scores.npz", **self.score_tables)
        render_attribute_roc(self.roc, self.summary, path)
        (path / "summary.json").write_text(
            json.dumps(self.summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def render_attribute_roc(roc: pd.DataFrame, summary: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    grid = np.linspace(0, 1, 1001)
    interpolated = []
    for _, curve in roc.groupby("pair_id", sort=True):
        for axis in axes:
            axis.plot(curve["fpr"], curve["tpr"], alpha=0.12, color="#2563A6", linewidth=0.8)
        interpolated.append(np.interp(grid, curve["fpr"], curve["tpr"]))
    for axis in axes:
        if interpolated:
            axis.plot(
                grid,
                np.mean(interpolated, axis=0),
                color="#2563A6",
                label="Mean across eligible pairs",
            )
        axis.plot([0, 1], [0, 1], "--", color="#9CA3AF", label="No output information")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.grid(alpha=0.2)
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_xlim(-0.02, 1.02)
    axes[1].set_xlim(0, min(1.0, max(0.05, 2 * summary["target_fpr"])))
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"Held-out attribute attack ROC: {summary['evaluated_pairs']}/{summary['eligible_pairs']} eligible pairs"
    )
    fig.text(
        0.5,
        0.01,
        "Thin lines are individual ordered pairs. This is an empirical attack evaluation, not a certificate.",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path / "roc.png", dpi=150)
    fig.savefig(path / "roc.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def evaluate_attribute_inference(
    attack_outputs: np.ndarray,
    attack_groups: Sequence[Group],
    attack_ids: np.ndarray,
    test_outputs: np.ndarray,
    test_groups: Sequence[Group],
    test_ids: np.ndarray,
    *,
    expected_groups: Sequence[Group],
    k: int,
    method: str = "likelihood_ratio",
    smoothing: float = 1.0,
    target_fpr: float = 0.01,
    confidence: float = 0.95,
    calibration_fraction: float = 0.5,
    seed: int = 0,
    adjacency_mode: str = "tuple_adjacent",
    fit_split: str = "D_attack_train",
    test_split: str = "D_test",
) -> AttributeInferenceResult:
    if fit_split != "D_attack_train" or test_split != "D_test":
        raise ValueError("attribute attacks fit only on D_attack_train and evaluate only on D_test")
    if method not in ATTACK_FACTORIES:
        raise ValueError(f"unsupported attribute attack method: {method}")
    if k < 1 or not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("K and attack smoothing must be positive")
    if adjacency_mode not in {"tuple_adjacent", "joint_all_pairs"}:
        raise ValueError("attribute attack adjacency must be tuple_adjacent or joint_all_pairs")
    if not 0 < target_fpr < 1 or not 0 < confidence < 1 or seed < 0:
        raise ValueError("attack FPR/confidence must be in (0, 1), and seed nonnegative")
    for outputs, groups, ids in (
        (attack_outputs, attack_groups, attack_ids),
        (test_outputs, test_groups, test_ids),
    ):
        if not (len(outputs) == len(groups) == len(ids)) or len(outputs) == 0:
            raise ValueError("attack inputs must be nonempty aligned arrays")
        if (
            np.asarray(outputs).dtype.kind not in "iu"
            or np.any(outputs < 0)
            or np.any(outputs >= k)
        ):
            raise ValueError("attack output tokens must be integers in [0, K)")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("attribute attacks require one release per user-day")
    if set(map(str, attack_ids)).intersection(map(str, test_ids)):
        raise ValueError("attack fit/calibration and test user-days must be disjoint")
    fit = attack_fit_mask(attack_ids, seed=seed, calibration_fraction=calibration_fraction)
    known = {group.key(): group for group in expected_groups}
    if any(group.key() not in known for group in chain(attack_groups, test_groups)):
        raise ValueError("observed attribute groups are outside the frozen support")

    def positions(groups: Sequence[Group]) -> dict[str, np.ndarray]:
        indices: dict[str, list[int]] = {}
        for index, group in enumerate(groups):
            indices.setdefault(group.key(), []).append(index)
        return {key: np.asarray(value, dtype=int) for key, value in indices.items()}

    attack_index, test_index = positions(attack_groups), positions(test_groups)
    empty = np.array([], dtype=int)
    scopes: dict[tuple[str, str], list[Group]] = {}
    for group in sorted(known.values(), key=lambda group: group.key()):
        scopes.setdefault((group.profile, str(group.context)), []).append(group)
    family = []
    for groups in scopes.values():
        for positive, negative in permutations(groups, 2):
            changed = [
                name
                for name, a, b in zip(
                    positive.attributes, positive.values, negative.values, strict=True
                )
                if a != b
            ]
            if not changed or (adjacency_mode == "tuple_adjacent" and len(changed) != 1):
                continue
            family.append((positive, negative, changed))
    # Four one-sided endpoints per pair give simultaneous coverage for both
    # operating-point rates across this mechanism's full pair/context family.
    tail_alpha = (1 - confidence) / (4 * max(1, len(family)))
    coins = np.random.default_rng(seed).random(len(test_outputs))
    rows, curves, tables, eligible_keys = [], [], {}, []
    for positive, negative, changed in family:
        keys = (positive.key(), negative.key())
        pair_id = hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:20]
        selected = [attack_index.get(key, empty) for key in keys]
        fitted = [indices[fit[indices]] for indices in selected]
        calibrated = [indices[~fit[indices]] for indices in selected]
        tested = [test_index.get(key, empty) for key in keys]
        row = {
            "pair_id": pair_id,
            "profile": positive.profile,
            "context": str(positive.context),
            "positive_values": json.dumps(positive.values),
            "negative_values": json.dumps(negative.values),
            "changed_attributes": json.dumps(changed),
            "status": "complete",
            "fit_positive": len(fitted[0]),
            "fit_negative": len(fitted[1]),
            "calibration_positive": len(calibrated[0]),
            "calibration_negative": len(calibrated[1]),
            "test_positive": len(tested[0]),
            "test_negative": len(tested[1]),
        }
        if any(not len(value) for value in fitted):
            row["status"] = "missing_fit_group"
        elif any(not len(value) for value in calibrated):
            row["status"] = "missing_calibration_group"
        else:
            eligible_keys.append(keys)
            attack = ATTACK_FACTORIES[method](
                attack_outputs[fitted[0]], attack_outputs[fitted[1]], k, smoothing
            )
            scores = np.asarray(attack.scores, dtype=float)
            if scores.shape != (k,) or not np.isfinite(scores).all():
                raise ValueError("token attacker must return K finite scores")
            tables[pair_id] = scores
            threshold, tie_probability = calibrate_operating_point(
                scores, attack_outputs[calibrated[1]], target_fpr
            )
            row.update(threshold=threshold, tie_probability=tie_probability)
            if any(not len(value) for value in tested):
                row["status"] = "missing_test_group"
            else:
                curve = _curve(scores, test_outputs[tested[0]], test_outputs[tested[1]])
                curve.insert(0, "pair_id", pair_id)
                curves.append(curve)
                counts = []
                for indices in tested:
                    selected_scores = scores[test_outputs[indices]]
                    counts.append(
                        int(
                            np.sum(
                                (selected_scores > threshold)
                                | (
                                    (selected_scores == threshold)
                                    & (coins[indices] < tie_probability)
                                )
                            )
                        )
                    )
                x, y = curve["fpr"].to_numpy(), curve["tpr"].to_numpy()
                row["roc_auc"] = float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))
                for label, successes, indices in zip(("tpr", "fpr"), counts, tested, strict=True):
                    row[label] = successes / len(indices)
                    row[f"{label}_successes"] = successes
                    row[f"{label}_ci_low"], row[f"{label}_ci_high"] = _interval(
                        successes, len(indices), tail_alpha
                    )
        rows.append(row)
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        pairs = pd.DataFrame(columns=["pair_id", "status", "roc_auc", "tpr", "fpr"])
    completed = pairs.loc[pairs["status"] == "complete"]
    status = (
        "no_eligible_pairs"
        if not eligible_keys
        else ("complete" if len(completed) == len(eligible_keys) else "incomplete_test_coverage")
    )
    summary = {
        "schema_version": 1,
        "method": method,
        "status": status,
        "K": k,
        "smoothing": smoothing,
        "target_fpr": target_fpr,
        "confidence": confidence,
        "calibration_fraction": calibration_fraction,
        "seed": seed,
        "adjacency_mode": adjacency_mode,
        "expected_pairs": len(family),
        "eligible_pairs": len(eligible_keys),
        "evaluated_pairs": len(completed),
        "scope_sha256": hashlib.sha256(
            json.dumps(eligible_keys, sort_keys=True).encode()
        ).hexdigest(),
        "attack_fit_rows": int(fit.sum()),
        "attack_calibration_rows": int((~fit).sum()),
        "test_rows": len(test_outputs),
        "fit_split": fit_split,
        "test_split": test_split,
        "certificate": False,
        "epsilon_estimated": False,
        "interval_scope": "simultaneous operating-point TPR/FPR intervals over frozen ordered pairs within this mechanism cell; user_day_iid assumption; not the whole grid",
        "interpretation": "empirical attribute inference; test FPR can differ from calibration target; no worst-case privacy guarantee",
    }
    for metric in ("roc_auc", "tpr", "fpr"):
        for aggregate in ("mean", "worst"):
            summary[f"{aggregate}_{metric}"] = (
                float(completed[metric].mean() if aggregate == "mean" else completed[metric].max())
                if status == "complete"
                else None
            )
    return AttributeInferenceResult(
        pairs,
        pd.concat(curves, ignore_index=True)
        if curves
        else pd.DataFrame(columns=["pair_id", "threshold", "tp", "fp", "tpr", "fpr"]),
        tables,
        summary,
    )
