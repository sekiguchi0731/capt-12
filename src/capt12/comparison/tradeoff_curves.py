"""Parameter-ordered observations and same-EAU, piecewise-linear comparisons.

Segments are visual interpolation, not attainable mixtures or fitted mechanisms.
Keep folds/vertical segments instead of silently replacing a trace by its envelope.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Controls of the learned transformation; never group using measured losses/MI.
MASS_CONTROLS = (
    "mass_seed",
    "mass_backend",
    "mass_mi_conditioning",
    "mass_m",
    "mass_n",
    "mass_privacy_weight",
    "mass_utility_weight",
    "mass_ctr_weight",
    "mass_temperature",
    "mass_infonce_weight",
    "mass_infonce_direction",
    "mass_infonce_pretrain_epochs",
    "mass_infonce_pretrain_learning_rate",
    "mass_privacy_quadratic_weight",
    "mass_privacy_linear_weight",
    "mass_useful_quadratic_weight",
    "mass_useful_linear_weight",
    "mass_attribute_pretrain_epochs",
    "mass_batch_size",
    "mass_learning_rate",
    "mass_weight_decay",
    "mass_token_dim",
    "mass_hidden_dim",
    "mass_eval_noise_samples",
)
AXES = ("m", "n", "privacy_weight", "utility_weight", "temperature")


def parameter_traces(summary: pd.DataFrame, mass_axis: str = "m") -> pd.DataFrame:
    if mass_axis not in AXES:
        raise ValueError(f"MaSS curve axis must be one of {AXES}")
    records = []
    for row in summary.to_dict("records"):
        is_mass = row["method"] == "mass12_raw"
        axis = f"mass_{mass_axis}" if is_mass else "target_epsilon"
        controls = {key: row.get(key) for key in MASS_CONTROLS if key != axis} if is_mass else {}
        identity = [row["cell_id"], row["method"], controls]
        trace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        records.append(
            {**row, "trace_id": trace, "parameter": axis, "parameter_value": float(row[axis])}
        )
    result = pd.DataFrame(records).sort_values(["trace_id", "parameter_value", "model_id"])
    unique = []
    for _, group in result.groupby(["trace_id", "parameter_value"], sort=False):
        if len(group) > 1:
            identity = "model_sha256" if group.method.iloc[0] == "mass12_raw" else "channel_sha256"
            metrics = group[["mean_pair_eau", "worst_pair_eau", "empirical_log_loss"]].to_numpy(
                float
            )
            if (
                identity not in group
                or group[identity].isna().any()
                or group[identity].nunique() != 1
                or not np.allclose(metrics, metrics[0], rtol=1e-12, atol=1e-12)
            ):
                raise ValueError(
                    "duplicate sweep parameter with different mechanisms or measurements; select a consistent training/attack replicate"
                )
        representative = group.iloc[0].to_dict()
        representative["replicated_model_ids"] = json.dumps(group.model_id.tolist())
        unique.append(representative)
    result = pd.DataFrame(unique)
    result["trace_position"] = result.groupby("trace_id").cumcount()
    return result


def horizontal_intersections(trace: pd.DataFrame, metric: str, eau: float) -> list[dict]:
    """All intersections, preserving nonmonotonic and vertical portions."""
    values = trace.sort_values("trace_position").to_dict("records")
    answers = []
    for a, b in zip(values[:-1], values[1:], strict=True):
        x0, x1 = a[metric], b[metric]
        if not min(x0, x1) <= eau <= max(x0, x1):
            continue
        if x0 == x1:
            # A vertical trace has multiple utilities at one EAU; retain both ends.
            weights = [0.0, 1.0]
        else:
            weights = [(eau - x0) / (x1 - x0)]
        for weight in weights:
            answers.append(
                {
                    "left_model_id": a["model_id"],
                    "right_model_id": b["model_id"],
                    "right_weight": weight,
                    "logloss": (1 - weight) * a["empirical_log_loss"]
                    + weight * b["empirical_log_loss"],
                    "left_parameter": a["parameter_value"],
                    "right_parameter": b["parameter_value"],
                }
            )
    if len(values) == 1 and eau == values[0][metric]:
        a = values[0]
        answers.append(
            {
                "left_model_id": a["model_id"],
                "right_model_id": a["model_id"],
                "right_weight": 0.0,
                "logloss": a["empirical_log_loss"],
                "left_parameter": a["parameter_value"],
                "right_parameter": a["parameter_value"],
            }
        )
    return answers


def same_eau_comparisons(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell, frame in points.groupby("cell_id"):
        capts = [g for _, g in frame[frame.method.eq("capt")].groupby("trace_id")]
        masses = [g for _, g in frame[frame.method.eq("mass12_raw")].groupby("trace_id")]
        for capt in capts:
            for mass in masses:
                for metric in ("mean_pair_eau", "worst_pair_eau"):
                    lo, hi = (
                        max(capt[metric].min(), mass[metric].min()),
                        min(capt[metric].max(), mass[metric].max()),
                    )
                    if lo > hi:
                        continue
                    grid = np.unique(np.r_[np.linspace(lo, hi, 101), capt[metric], mass[metric]])
                    for eau in grid[(grid >= lo) & (grid <= hi)]:
                        for c in horizontal_intersections(capt, metric, eau):
                            for m in horizontal_intersections(mass, metric, eau):
                                rows.append(
                                    {
                                        "cell_id": cell,
                                        "metric": metric,
                                        "eau": eau,
                                        "capt_trace": capt.trace_id.iloc[0],
                                        "mass_trace": mass.trace_id.iloc[0],
                                        **{f"capt_{k}": v for k, v in c.items()},
                                        **{f"mass_{k}": v for k, v in m.items()},
                                        "mass_minus_capt_logloss": m["logloss"] - c["logloss"],
                                        "interpretation": "descriptive segment interpolation, not a trained setting or confidence claim",
                                    }
                                )
    return pd.DataFrame(
        rows,
        columns=list(rows[0])
        if rows
        else ["cell_id", "metric", "eau", "mass_minus_capt_logloss", "interpretation"],
    )


def write_tradeoff_curves(summary: pd.DataFrame, output: Path, mass_axis: str = "m") -> None:
    points = parameter_traces(summary, mass_axis)
    points.to_csv(output / "tradeoff_points.csv", index=False)
    matched = same_eau_comparisons(points)
    matched.to_csv(output / "same_eau_logloss.csv", index=False)
    colors = {"capt": "#2563A6", "mass12_raw": "#D97706", "optimal_ldp": "#666666"}
    for cell, frame in points.groupby("cell_id"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for _, trace in frame.groupby("trace_id", sort=False):
            first = trace.iloc[0]
            label = (
                (
                    f"MaSS {mass_axis} sweep; "
                    + ", ".join(
                        f"{key.removeprefix('mass_')}={first[key]:g}"
                        for key in (
                            "mass_m",
                            "mass_n",
                            "mass_privacy_weight",
                            "mass_utility_weight",
                            "mass_seed",
                        )
                        if key != f"mass_{mass_axis}" and key in first and pd.notna(first[key])
                    )
                )
                if first.method == "mass12_raw"
                else first.method.upper()
            )
            for axis, metric in zip(axes, ("mean_pair_eau", "worst_pair_eau"), strict=True):
                axis.plot(
                    trace[metric],
                    trace.empirical_log_loss,
                    "o-",
                    color=colors.get(first.method),
                    label=label,
                    linewidth=1.3,
                )
                for row in trace.to_dict("records"):
                    name = mass_axis if first.method == "mass12_raw" else "eps"
                    axis.annotate(
                        f"{name}={row['parameter_value']:g}",
                        (row[metric], row["empirical_log_loss"]),
                        xytext=(4, 5),
                        textcoords="offset points",
                        fontsize=7,
                    )
                axis.set(
                    xlabel=("Mean" if metric.startswith("mean") else "Largest")
                    + " pairwise EAU (lower is better)",
                    ylabel="Held-out CTR log loss (lower is better)",
                )
                axis.grid(alpha=0.2)
        axes[-1].legend(fontsize=6, frameon=False)
        fig.suptitle("Observed parameter sweeps: CAPT vs MaSS")
        fig.text(
            0.5,
            0.01,
            "Lines follow parameter order; folds are retained. Segment interpolation is descriptive only.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        for ext in ("png", "pdf"):
            fig.savefig(output / "figures" / f"eau_logloss_curve_{cell}.{ext}", dpi=180)
        plt.close(fig)
