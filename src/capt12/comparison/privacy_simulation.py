"""Exact MI-only counterexamples, NOT a trained MaSS/CAPT experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

from capt12.comparison.attack_metrics import pr_plot_points


def example_channel(
    reveal_probability: float, *, epsilon: float | None = None, floor_mix: float = 0.0
) -> np.ndarray:
    """Z=(A,U), O=(U,marker); independent balanced A,U. No online A argument.

    The MI-only channel occasionally retains the A-bit already present in Z.
    The reference randomizes this bit using binary epsilon-RR. Both preserve
    U; identical uniform contamination gives them identical CTR utility too.
    """
    if not 0 <= reveal_probability <= 1 or not 0 <= floor_mix < 1:
        raise ValueError("reveal probability and floor mixture must be probabilities")
    if epsilon is not None and (not np.isfinite(epsilon) or epsilon < 0):
        raise ValueError("reference epsilon must be finite and nonnegative")
    keep = 1.0 if epsilon is None else expit(epsilon)
    channel = np.zeros((4, 6))
    for a in range(2):
        for u in range(2):
            z = 2 * a + u
            channel[z, u] = 1 - reveal_probability
            channel[z, 2 + 2 * a + u] = reveal_probability * keep
            channel[z, 2 + 2 * (1 - a) + u] = reveal_probability * (1 - keep)
    return (1 - floor_mix) * channel + floor_mix / 6


def output_profiles(channel: np.ndarray) -> np.ndarray:
    q = np.asarray(channel, float)
    if q.shape != (4, 6) or np.any(q < 0) or not np.allclose(q.sum(1), 1):
        raise ValueError("expected a row-stochastic 4-by-6 channel")
    return np.stack([q[:2].mean(0), q[2:].mean(0)])


def exact_binary_metrics(profiles: np.ndarray, prior: float = 0.5) -> dict:
    q = np.asarray(profiles, float)
    if q.ndim != 2 or q.shape[0] != 2 or np.any(q < 0) or not np.allclose(q.sum(1), 1):
        raise ValueError("two normalized output profiles are required")
    if not 0 < prior < 1:
        raise ValueError("prior must be in (0,1)")
    priors = np.array([1 - prior, prior])
    marginal = priors @ q
    mask = q > 0
    ratio = np.divide(q, marginal, out=np.ones_like(q), where=marginal > 0)
    information = float(np.sum((priors[:, None] * q)[mask] * np.log(ratio[mask])))
    epsilon = 0.0
    for p, denominator in ((q[0], q[1]), (q[1], q[0])):
        if np.any((p > 0) & (denominator == 0)):
            epsilon = float("inf")
        elif np.isfinite(epsilon):
            positive = p > 0
            epsilon = max(epsilon, float(np.log(p[positive] / denominator[positive]).max()))
    # Optimal EAU is available here because the distributions are known.
    eau = float(np.maximum((1 - prior) * q[0], prior * q[1]).sum())
    return {
        "mutual_information_nats": max(0.0, information),
        "exact_epsilon": epsilon,
        "exact_optimal_eau": eau,
        "prior_only_accuracy": max(prior, 1 - prior),
    }


def exact_roc(profiles: np.ndarray) -> pd.DataFrame:
    negative, positive = np.asarray(profiles)
    support = (positive + negative) > 0
    p, q = positive[support], negative[support]
    ratio = np.divide(p, q, out=np.full_like(p, np.inf), where=q > 0)
    levels = np.unique(ratio)[::-1]
    tpr = np.r_[0, np.cumsum([p[ratio == level].sum() for level in levels])]
    fpr = np.r_[0, np.cumsum([q[ratio == level].sum() for level in levels])]
    return pd.DataFrame({"tpr": np.clip(tpr, 0, 1), "fpr": np.clip(fpr, 0, 1)})


def exact_privacy_profile(profiles: np.ndarray, epsilons: np.ndarray) -> np.ndarray:
    negative, positive = np.asarray(profiles)
    values = np.asarray(epsilons, float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("finite nonnegative epsilons required")
    return np.maximum(
        np.maximum(positive - np.exp(values[:, None]) * negative, 0).sum(1),
        np.maximum(negative - np.exp(values[:, None]) * positive, 0).sum(1),
    )


def expected_ctr_logloss(channel: np.ndarray) -> float:
    # Y|U is Bernoulli(0.1) or Bernoulli(0.9); the fixed service reads output U.
    source_p = np.array([0.1, 0.9, 0.1, 0.9])
    output_p = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    cost = -source_p[:, None] * np.log(output_p) - (1 - source_p[:, None]) * np.log1p(-output_p)
    return float(np.sum(channel * cost) / 4)


def run_mi_counterexample(
    output: Path,
    *,
    epsilon: float = 1.0,
    reveal_probability: float = 0.01,
    positive_floor: float = 1e-9,
) -> Path:
    if not 0 < reveal_probability < 1 or not 0 < positive_floor < 1:
        raise ValueError("example reveal probability and positive floor must be in (0,1)")
    output.mkdir(parents=True, exist_ok=True)
    q_grid = np.unique(np.r_[0, np.geomspace(1e-6, 0.5, 40), reveal_probability])
    rows, channels, curves, profiles = [], {}, {}, {}
    for floor in (0.0, positive_floor):
        for mechanism, eps in (
            ("MI-only admissible channel", None),
            ("Attribute-private reference", epsilon),
        ):
            for reveal in q_grid:
                channel = example_channel(float(reveal), epsilon=eps, floor_mix=floor)
                result = exact_binary_metrics(output_profiles(channel))
                rows.append(
                    {
                        "mechanism": mechanism,
                        "floor_mix": floor,
                        "reveal_probability": reveal,
                        **result,
                        "expected_ctr_logloss": expected_ctr_logloss(channel),
                        "trained_mass": False,
                        "trained_capt": False,
                    }
                )
                if reveal == reveal_probability:
                    key = f"{mechanism} | floor={floor:g}"
                    channels[key] = channel
                    profiles[key] = output_profiles(channel)
                    curves[key] = pr_plot_points(exact_roc(profiles[key]))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "exact_metrics.csv", index=False)
    np.savez_compressed(output / "known_channels.npz", **channels)
    for key, curve in curves.items():
        curve["scenario"] = key
    pd.concat(curves.values(), ignore_index=True).to_csv(output / "exact_pr.csv", index=False)
    eps_grid = np.linspace(0, 25, 251)
    profile_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "scenario": key,
                    "epsilon": eps_grid,
                    "exact_delta": exact_privacy_profile(value, eps_grid),
                }
            )
            for key, value in profiles.items()
        ],
        ignore_index=True,
    )
    profile_frame.to_csv(output / "exact_privacy_profiles.csv", index=False)
    _figures(frame, curves, profile_frame, output, reveal_probability)
    metadata = {
        "reference_epsilon": epsilon,
        "example_reveal_probability": reveal_probability,
        "positive_floor_mix": positive_floor,
        "distribution_known": True,
        "trained_mass": False,
        "trained_capt": False,
        "source": "constructive MI-only counterexample",
        "interpretation": "MI-only objectives permit rare severe leakage; this is not evidence that a particular learned MaSS model has infinite epsilon",
        "strictly_positive_case": "uniform mixing makes all entries positive and epsilon finite, but potentially very large",
        "utility": "same fixed CTR service and identical exact expected logloss at matched floor mixture",
    }
    (output / "simulation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Exact MI-only privacy counterexample\n\n"
        "A and U are independent balanced bits; Z=(A,U). With probability q the MI-only channel reveals the A-bit already in Z, otherwise only U. "
        "Thus I(A;O)=q log(2) can be arbitrarily small, while for every q>0 its worst-case attribute epsilon is infinite. "
        "The reference applies epsilon-RR to the revealed bit, preserving U and the exact same CTR utility. "
        "Positive uniform contamination gives both mechanisms identical utility and makes their supports strictly positive; the MI-only epsilon then becomes finite but can remain large.\n\n"
        "These are known channels, not MaSS or CAPT trained on simulated data. They establish a limitation of MI-only requirements, not a failure of every MI optimizer.\n"
    )
    return output


def _figures(frame, curves, profiles, output, reveal):
    from capt12.comparison.privacy_comparison import _save

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    for key, curve in curves.items():
        for axis in axes:
            axis.plot(curve.recall, curve.precision, marker="o", markersize=3, label=key)
    for axis in axes:
        axis.axhline(0.5, linestyle="--", color="#999999")
        axis.set(xlabel="Recall", ylabel="Exact optimal attack precision", ylim=(0.45, 1.03))
        axis.grid(alpha=0.2)
    axes[0].set_xlim(0, 1)
    axes[1].set_xscale("log")
    axes[1].set_xlim(1e-6, 1)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        f"Known-channel counterexample: rare release probability q={reveal:g}", fontsize=12
    )
    fig.text(
        0.5,
        0.015,
        "Constructed channels; not trained MaSS/CAPT. Equal fixed-service CTR log loss at matched floor mixture.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save(fig, output, "simulation_pr")
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    for key, group in profiles.groupby("scenario", sort=False):
        axis.plot(group.epsilon, group.exact_delta, label=key)
    axis.set(
        xlabel="Candidate privacy budget epsilon", ylabel="Exact privacy profile delta(epsilon)"
    )
    axis.legend(fontsize=7, frameon=False)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, output, "simulation_privacy_profile")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for (mechanism, floor), group in frame.groupby(["mechanism", "floor_mix"], sort=False):
        label = f"{mechanism} | floor={floor:g}"
        axes[0].plot(group.mutual_information_nats, group.expected_ctr_logloss, label=label)
        axes[1].plot(group.reveal_probability, group.mutual_information_nats, label=label)
        axes[2].plot(group.exact_optimal_eau, group.expected_ctr_logloss, label=label)
    axes[0].set(xlabel="Exact sensitive MI (nats)", ylabel="Exact expected CTR log loss")
    axes[1].set(xlabel="Rare release probability q", ylabel="Exact sensitive MI (nats)")
    axes[1].set_xscale("log")
    axes[2].set(xlabel="Exact optimal EAU", ylabel="Exact expected CTR log loss")
    for axis in (axes[0], axes[2]):
        axis.set_ylim(0.3, 0.35)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, output, "simulation_utility_mi")
