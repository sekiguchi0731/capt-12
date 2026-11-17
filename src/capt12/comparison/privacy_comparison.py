"""Compare saved attribute attacks on common held-out attribute pairs.

No mechanism, attacker, or threshold is retrained or selected on test labels.
Copied remote runs are supported; the original artifacts are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capt12.comparison.attack_metrics import (
    audit_score_buckets,
    binary_attack_metrics,
    compare_pr_curves,
    pr_plot_points,
    precision_at_recall,
    precision_recall,
    rare_score_buckets,
)
from capt12.utils.artifacts import sha256_file

CELL_KEYS = ["cost", "seed", "K", "L"]
COLORS = {"capt": "#2563A6", "optimal_ldp": "#555555", "mass12_raw": "#D97706"}
PAPER = "https://petsymposium.org/popets/2019/popets-2019-0045.pdf"
EAU_SOURCE = "IPSJ-TOD1902006, Definition 3.2.1 (printed p. 56)"


@dataclass
class SavedAttack:
    row: dict
    path: Path
    pairs: pd.DataFrame
    curves: dict[str, pd.DataFrame]
    summary: dict

    @property
    def model_id(self) -> str:
        return self.row["comparison_model_id"]

    @property
    def label(self) -> str:
        if self.row["method"] == "mass12_raw":
            return (
                f"MaSS m={self.row['mass_m']:g}, n={self.row.get('mass_n', 0):g}, "
                f"wp={self.row.get('mass_privacy_weight', 1):g}, wu={self.row.get('mass_utility_weight', 1):g}, s={self.row.get('mass_seed', 0):g}"
            )
        name = "CAPT" if self.row["method"] == "capt" else str(self.row["display_name"])
        return f"{name} eps={self.row['target_epsilon']:g}"


def _attack_path(results: Path, raw: str, attack_id: str) -> Path:
    candidates = [Path(raw)]
    normalized = raw.replace("\\", "/")
    suffix = normalized.split("/runs/", 1)[-1] if "/runs/" in normalized else None
    for parent in [results.parent, *list(results.parents)[:4]]:
        candidates.append(parent / "attribute_attacks" / attack_id)
        if suffix is not None:
            candidates.append(parent / "runs" / suffix)
    for candidate in candidates:
        if (candidate / "pairs.csv").is_file() and (candidate / "roc.csv").is_file():
            return candidate.resolve()
    raise ValueError(f"cannot locate saved attribute attack {attack_id} relative to {results}")


def load_saved_attacks(results_paths: list[Path]) -> tuple[list[SavedAttack], list[dict]]:
    attacks, sources = [], []
    seen = set()
    for source in results_paths:
        source = source.resolve()
        frame = pd.read_csv(source, float_precision="round_trip")
        required = {
            *CELL_KEYS,
            "target_epsilon",
            "method",
            "empirical_log_loss",
            "attribute_attack_id",
            "attribute_attack_path",
        }
        if required - set(frame):
            raise ValueError(f"results missing columns: {sorted(required - set(frame))}")
        sources.append({"path": str(source), "sha256": sha256_file(source)})
        for row in frame.to_dict("records"):
            if pd.isna(row["attribute_attack_id"]):
                raise ValueError("all compared rows require saved attribute attack artifacts")
            identity = [
                *[row[key] for key in CELL_KEYS],
                row["target_epsilon"],
                row["method"],
                row["attribute_attack_id"],
            ]
            key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
            if key in seen:
                continue
            seen.add(key)
            row["comparison_model_id"] = key
            path = _attack_path(
                source, str(row["attribute_attack_path"]), str(row["attribute_attack_id"])
            )
            summary = json.loads((path / "summary.json").read_text())
            mechanism = (
                row.get("model_sha256")
                if row["method"] == "mass12_raw"
                else row.get("channel_sha256")
            )
            if summary.get("mechanism_identity", {}).get("mechanism") != mechanism:
                raise ValueError("saved attack mechanism identity does not match results")
            pairs = pd.read_csv(path / "pairs.csv", dtype={"pair_id": str, "context": str})
            roc = pd.read_csv(
                path / "roc.csv", dtype={"pair_id": str}, float_precision="round_trip"
            )
            if pairs.pair_id.duplicated().any():
                raise ValueError("duplicate attribute pair IDs")
            if set(roc.pair_id) != set(pairs.loc[pairs.status.eq("complete"), "pair_id"]):
                raise ValueError("ROC pairs do not match completed pair records")
            curves = {
                str(key): value.reset_index(drop=True)
                for key, value in roc.groupby("pair_id", sort=False)
            }
            with np.load(path / "attacker_scores.npz", allow_pickle=False) as frozen:
                for pair_id, curve in curves.items():
                    pair = pairs.loc[pairs.pair_id.eq(pair_id)].iloc[0]
                    expected = np.r_[np.inf, np.unique(frozen[pair_id])[::-1]]
                    if not np.array_equal(curve.threshold.to_numpy(), expected):
                        raise ValueError("ROC thresholds differ from the saved frozen attacker")
                    for count, rate, total in (
                        ("tp", "tpr", pair.test_positive),
                        ("fp", "fpr", pair.test_negative),
                    ):
                        values = curve[count].to_numpy()
                        if (
                            np.any(values < 0)
                            or np.any(values != np.floor(values))
                            or np.any(np.diff(values) < 0)
                            or values[0] != 0
                            or values[-1] != total
                            or not np.allclose(curve[rate], values / total, atol=1e-14, rtol=1e-14)
                        ):
                            raise ValueError(
                                "ROC release counts/rates do not match saved group support"
                            )
            attacks.append(
                SavedAttack(row, path, pairs.set_index("pair_id", drop=False), curves, summary)
            )
            sources.append(
                {
                    "attack_id": row["attribute_attack_id"],
                    "path": str(path),
                    "pairs_sha256": sha256_file(path / "pairs.csv"),
                    "roc_sha256": sha256_file(path / "roc.csv"),
                    "summary_sha256": sha256_file(path / "summary.json"),
                    "attacker_scores_sha256": sha256_file(path / "attacker_scores.npz"),
                }
            )
    if not attacks:
        raise ValueError("no saved attacks to compare")
    return attacks, sources


def _cell_id(row: dict) -> str:
    return hashlib.sha256(json.dumps([row[k] for k in CELL_KEYS]).encode()).hexdigest()[:12]


def _batches(items, method):
    """Keep common CAPT/LDP references on every page of a large MaSS grid."""
    references = [item for item in items if method(item) != "mass12_raw"]
    variants = [item for item in items if method(item) == "mass12_raw"]
    if len(items) <= 8:
        return [items]
    if not variants:
        return [references[start : start + 8] for start in range(0, len(references), 8)]
    if len(references) > 4:
        pages = []
        for start in range(0, len(references), 4):
            chunk = references[start : start + 4]
            count = 8 - len(chunk)
            pages.extend(chunk + variants[i : i + count] for i in range(0, len(variants), count))
        return pages
    count = max(1, 8 - len(references))
    return [
        references + variants[start : start + count] for start in range(0, len(variants), count)
    ]


def _save(fig, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=160)
    fig.savefig(output / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def _pr_figures(models, common, grid_curves, output: Path, cell: str, prior: float):
    # Each series uses the SAME pairs, equal pair weights and the SAME prior.
    minimum_support = int(
        models[0].pairs.loc[common, ["test_positive", "test_negative"]].min().min()
    )
    for aggregate in ("mean", "worst"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
        for model in models:
            values = grid_curves[model.model_id]
            precision = values.mean(0) if aggregate == "mean" else values.max(0)
            grid = np.unique(np.r_[np.geomspace(1e-6, 1, 180), np.linspace(0.01, 1, 250)])
            for axis in axes:
                axis.plot(grid, precision, label=model.label, linewidth=1.4)
        for axis in axes:
            axis.axhline(prior, color="#999999", linestyle="--", label="Prior-only guessing")
            axis.set_xlabel("Recall")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.2)
        axes[0].set_xlim(0, 1)
        axes[1].set_xscale("log")
        axes[1].set_xlim(1e-6, 1)
        axes[0].set_ylabel("Attack precision (lower means weaker measured attack)")
        axes[1].legend(fontsize=7, frameon=False)
        title = "Macro mean" if aggregate == "mean" else "Pointwise worst-pair envelope"
        fig.suptitle(
            f"{title} PR | {len(common)} common ordered pairs | prior={prior:g}", fontsize=11
        )
        fig.text(
            0.5,
            0.015,
            f"Held-out learned attacks; minimum test-group n={minimum_support}. Small groups can show perfect precision by chance; no population dominance certificate.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        _save(fig, output, f"pr_{aggregate}_{cell}")
    # Representative pairs are selected by CALIBRATION support, not attack success.
    reference = models[0].pairs.loc[common]
    ranking = reference[["calibration_positive", "calibration_negative"]].min(axis=1)
    selected = ranking.sort_values(ascending=False, kind="stable").index[:3]
    fig, axes = plt.subplots(1, len(selected), figsize=(5.2 * len(selected), 4), squeeze=False)
    for axis, pair_id in zip(axes[0], selected, strict=True):
        pair = reference.loc[pair_id]
        for model in models:
            pr = pr_plot_points(model.curves[pair_id], prior)
            axis.plot(pr.recall, pr.precision, label=model.label, linewidth=1.2)
        axis.axhline(prior, color="#999999", linestyle="--")
        axis.set(xlabel="Recall", ylabel="Attack precision", xlim=(0, 1), ylim=(0, 1.02))
        axis.set_title(
            f"Context {pair['context']}\n{pair['positive_values']} vs {pair['negative_values']}",
            fontsize=8,
        )
        axis.grid(alpha=0.2)
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.suptitle("Pairs with largest minimum calibration-group support", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, output, f"pr_pairs_{cell}")


def _scalar_figures(summary: pd.DataFrame, output: Path):
    pages = []
    for cell, group in summary.groupby("cell_id", sort=True):
        batches = _batches(group.to_dict("records"), lambda row: row["method"])
        for page, records in enumerate(batches):
            frame = pd.DataFrame(records)
            frame["cell_id"] = cell if len(batches) == 1 else f"{cell}_page{page + 1}"
            pages.append(frame)
    _scalar_pages(pd.concat(pages, ignore_index=True), output)


def _scalar_pages(summary: pd.DataFrame, output: Path):
    for cell, frame in summary.groupby("cell_id", sort=True):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
        for _, row in frame.iterrows():
            for axis, metric, label in zip(
                axes,
                ("mean_pair_eau", "worst_pair_eau", "mean_average_precision"),
                ("Mean pairwise EAU", "Largest pairwise EAU", "Mean attack Average Precision"),
                strict=True,
            ):
                artist = axis.scatter(row[metric], row.empirical_log_loss, label=row["label"], s=34)
                if "eau" in metric:
                    axis.hlines(
                        row.empirical_log_loss,
                        row[f"{metric}_ci_low"],
                        row[f"{metric}_ci_high"],
                        linewidth=1,
                        colors=artist.get_facecolor()[0],
                    )
                axis.set_xlabel(label + " (lower is better)")
                axis.grid(alpha=0.2)
        axes[0].set_ylabel("Held-out CTR log loss (lower is better)")
        axes[-1].legend(fontsize=7, frameon=False)
        fig.suptitle(
            f"Common-pair privacy/utility | prior={frame.positive_prior.iloc[0]:g}", fontsize=11
        )
        fig.text(
            0.5,
            0.015,
            f"EAU intervals: {100 * frame.confidence.iloc[0]:g}% simultaneous bounds for fixed learned rules; a single seed does not establish a general result.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        _save(fig, output, f"privacy_utility_{cell}")


def compare_saved_privacy(
    results_paths: list[Path],
    output: Path,
    *,
    prior: float = 0.5,
    confidence: float = 0.95,
    max_buckets: int = 16,
    min_test_count: int = 1,
    mass_curve_axis: str = "m",
) -> Path:
    """No Criteo data or torch is needed: use saved release-count curves only."""
    precision_recall(pd.DataFrame({"tpr": [0, 1], "fpr": [0, 1]}), prior)
    if min_test_count < 1 or max_buckets < 2 or not 0 < confidence < 1:
        raise ValueError("invalid comparison support, bucket count, or confidence")
    output = output.resolve()
    if any(output == path.resolve().parent for path in results_paths):
        raise ValueError("comparison output must be separate from original results")
    attacks, sources = load_saved_attacks(results_paths)
    cells = {}
    for model in attacks:
        cells.setdefault(_cell_id(model.row), []).append(model)
    output.mkdir(parents=True, exist_ok=True)
    pairs_out, coverage, events_out, comparisons, metadata_cells = [], [], [], [], []
    grids, cell_common = {}, {}
    grid = np.unique(np.r_[np.geomspace(1e-6, 1, 180), np.linspace(0.01, 1, 250)])
    pr_out = []
    for cell, models in cells.items():
        if len({model.summary["scope_sha256"] for model in models}) != 1:
            raise ValueError("different frozen eligible attribute-pair scopes cannot be compared")
        for key in (
            "method",
            "K",
            "smoothing",
            "seed",
            "calibration_fraction",
            "target_fpr",
            "fit_split",
            "test_split",
        ):
            if len({model.summary[key] for model in models}) != 1:
                raise ValueError(
                    f"attack fitting/calibration controls differ within the comparison: {key}"
                )
        common = None
        for model in models:
            selected = model.pairs.loc[
                model.pairs.status.eq("complete")
                & (model.pairs.test_positive >= min_test_count)
                & (model.pairs.test_negative >= min_test_count)
            ]
            keys = set(selected.index)
            common = keys if common is None else common.intersection(keys)
        common = sorted(common)
        if not common:
            raise ValueError(f"no common evaluated attribute pairs in cell {cell}")
        cell_common[cell] = common
        common_hash = hashlib.sha256(json.dumps(common).encode()).hexdigest()
        reference = models[0].pairs.loc[common]
        for model in models:
            # IDs alone are not enough to establish comparable population strata.
            columns = [
                "profile",
                "context",
                "positive_values",
                "negative_values",
                "test_positive",
                "test_negative",
                "calibration_positive",
                "calibration_negative",
            ]
            pd.testing.assert_frame_equal(reference[columns], model.pairs.loc[common, columns])
            coverage.append(
                {
                    "cell_id": cell,
                    "model_id": model.model_id,
                    "label": model.label,
                    "expected_pairs": model.summary["expected_pairs"],
                    "eligible_pairs": model.summary["eligible_pairs"],
                    "evaluated_pairs": model.summary["evaluated_pairs"],
                    "common_pairs": len(common),
                    "common_scope_sha256": common_hash,
                }
            )
            interpolated = []
            for pair_id in common:
                curve, pair = model.curves[pair_id], model.pairs.loc[pair_id]
                metrics = binary_attack_metrics(curve, prior)
                record = {
                    "cell_id": cell,
                    "model_id": model.model_id,
                    "pair_id": pair_id,
                    "positive_prior": prior,
                    **{
                        key: pair[key]
                        for key in (
                            "profile",
                            "context",
                            "positive_values",
                            "negative_values",
                            "test_positive",
                            "test_negative",
                        )
                    },
                    **metrics,
                    "calibrated_tpr": pair["tpr"],
                    "calibrated_fpr": pair["fpr"],
                }
                pairs_out.append(record)
                pr = precision_recall(curve, prior)
                pr["pair_id"] = pair_id
                pr.insert(0, "model_id", model.model_id)
                pr.insert(0, "cell_id", cell)
                pr_out.append(pr)
                interpolated.append(precision_at_recall(curve, grid, prior))
                bucket = rare_score_buckets(
                    curve, int(pair.test_positive), int(pair.test_negative), max_buckets
                )
                bucket["event_type"] = "extreme_frozen_score_bucket"
                # Tail unions can have substantially more power than a single
                # output-score bucket. Select boundaries using trained score
                # ranks only, never by their held-out counts.
                levels = len(curve) - 1
                ranks = np.unique(np.linspace(1, levels, min(max_buckets, levels), dtype=int))
                tail = curve.iloc[ranks]
                for complement in (False, True):
                    bucket = pd.concat([bucket, pd.DataFrame({
                        "score": tail.threshold.to_numpy(),
                        "positive_successes": (int(pair.test_positive) - tail.tp if complement else tail.tp).to_numpy(),
                        "negative_successes": (int(pair.test_negative) - tail.fp if complement else tail.fp).to_numpy(),
                        "positive_trials": int(pair.test_positive),
                        "negative_trials": int(pair.test_negative),
                        "event_type": "frozen_score_lower_tail" if complement else "frozen_score_upper_tail",
                    })], ignore_index=True)
                boundary = np.log1p(-prior) - np.log(prior)
                decision = curve.loc[curve.threshold >= boundary].iloc[-1]
                bayes = pd.DataFrame(
                    [
                        {
                            "score": boundary,
                            "event_type": "fixed_prior_bayes_rule",
                            "positive_successes": int(decision.tp),
                            "negative_successes": int(decision.fp),
                            "positive_trials": int(pair.test_positive),
                            "negative_trials": int(pair.test_negative),
                        }
                    ]
                )
                candidate = pd.concat([bucket, bayes], ignore_index=True)
                candidate["cell_id"], candidate["model_id"], candidate["pair_id"] = (
                    cell,
                    model.model_id,
                    pair_id,
                )
                events_out.append(candidate)
            grids[model.model_id] = np.stack(interpolated)
        for capt in [model for model in models if model.row["method"] == "capt"]:
            for other in [model for model in models if model.row["method"] != "capt"]:
                for pair_id in common:
                    comparisons.append(
                        {
                            "cell_id": cell,
                            "capt_id": capt.model_id,
                            "other_id": other.model_id,
                            "pair_id": pair_id,
                            **compare_pr_curves(capt.curves[pair_id], other.curves[pair_id], prior),
                        }
                    )
        metadata_cells.append(
            {
                "cell_id": cell,
                **{key: models[0].row[key] for key in CELL_KEYS},
                "common_pair_count": len(common),
                "common_scope_sha256": common_hash,
            }
        )
    pairs_frame = pd.DataFrame(pairs_out)
    audit = audit_score_buckets(pd.concat(events_out, ignore_index=True), confidence)
    bayes = audit.loc[audit.event_type.eq("fixed_prior_bayes_rule")].copy()
    bayes["pair_eau_ci_low"] = prior * bayes.positive_ci_low + (1 - prior) * (
        1 - bayes.negative_ci_high
    )
    bayes["pair_eau_ci_high"] = prior * bayes.positive_ci_high + (1 - prior) * (
        1 - bayes.negative_ci_low
    )
    pairs_frame = pairs_frame.merge(
        bayes[["model_id", "pair_id", "pair_eau_ci_low", "pair_eau_ci_high"]],
        on=["model_id", "pair_id"],
        validate="one_to_one",
    )
    summaries = []
    for model in attacks:
        measured = pairs_frame.loc[pairs_frame.model_id.eq(model.model_id)]
        record = {
            **model.row,
            "cell_id": _cell_id(model.row),
            "model_id": model.model_id,
            "label": model.label,
            "positive_prior": prior,
            "confidence": confidence,
            "common_pairs": len(measured),
        }
        for metric in (
            "pair_eau",
            "average_precision",
            "accuracy_uplift",
            "roc_auc",
            "calibrated_tpr",
            "calibrated_fpr",
        ):
            record[f"mean_{metric}"], record[f"worst_{metric}"] = (
                measured[metric].mean(),
                measured[metric].max(),
            )
        for end in ("low", "high"):
            record[f"mean_pair_eau_ci_{end}"] = measured[f"pair_eau_ci_{end}"].mean()
            record[f"worst_pair_eau_ci_{end}"] = measured[f"pair_eau_ci_{end}"].max()
        record["event_epsilon_lower"] = audit.loc[
            audit.model_id.eq(model.model_id), "epsilon_lower"
        ].max()
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    pair_comparisons = pd.DataFrame(comparisons)
    dominance = []
    if len(pair_comparisons):
        indexed = summary.set_index("model_id")
        for (cell, capt_id, other_id), group in pair_comparisons.groupby(
            ["cell_id", "capt_id", "other_id"], sort=True
        ):
            capt, other = indexed.loc[capt_id], indexed.loc[other_id]
            dominance.append(
                {
                    "cell_id": cell,
                    "capt_id": capt_id,
                    "other_id": other_id,
                    "other_label": other["label"],
                    "common_pairs": len(group),
                    "fraction_pairs_capt_pr_no_higher": group.capt_pr_no_higher.mean(),
                    "capt_pr_no_higher_on_all_common_pairs": bool(group.capt_pr_no_higher.all()),
                    "other_minus_capt_logloss": other.empirical_log_loss - capt.empirical_log_loss,
                    "capt_logloss_no_higher": bool(
                        capt.empirical_log_loss <= other.empirical_log_loss
                    ),
                    "observed_joint_dominance": bool(
                        group.capt_pr_no_higher.all()
                        and capt.empirical_log_loss <= other.empirical_log_loss
                    ),
                    "population_dominance_proven": False,
                }
            )
    pairs_frame.to_csv(output / "pair_metrics.csv", index=False)
    summary.to_csv(output / "mechanism_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage.csv", index=False)
    pd.concat(pr_out, ignore_index=True).to_csv(output / "pr_curves.csv", index=False)
    pair_comparisons.to_csv(output / "pair_pr_order.csv", index=False)
    pd.DataFrame(dominance).to_csv(output / "dominance_summary.csv", index=False)
    audit.to_csv(output / "event_audit.csv", index=False)
    for cell, models in cells.items():
        batches = _batches(models, lambda model: model.row["method"])
        for page, batch in enumerate(batches):
            name = cell if len(batches) == 1 else f"{cell}_page{page + 1}"
            _pr_figures(batch, cell_common[cell], grids, output / "figures", name, prior)
    _scalar_figures(summary, output / "figures")
    from capt12.comparison.tradeoff_curves import write_tradeoff_curves

    write_tradeoff_curves(summary, output, mass_curve_axis)
    _mass_capt_joint_comparison(audit, summary).to_csv(output / "mass_capt_leakage_utility.csv", index=False)
    _audit_figure(audit, summary, output / "figures")
    manifest = {
        "schema_version": 1,
        "sources": sources,
        "evaluation_code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("privacy_comparison.py", "attack_metrics.py", "tradeoff_curves.py", "privacy_simulation.py")
        },
        "cells": metadata_cells,
        "positive_prior": prior,
        "confidence": confidence,
        "min_test_count": min_test_count,
        "max_extreme_score_buckets_per_pair": max_buckets,
        "mechanisms_retrained": False,
        "attacks_retrained": False,
        "eau_source": EAU_SOURCE,
        "pr_reference": PAPER,
        "eau_definition": "P(fixed learned binary attribute-pair attacker(O)=hypothesis), with common chosen prior; not supremum over attackers or multiclass population EAU",
        "aggregation": "equal-weight macro mean or largest value over the SAME observed ordered pairs, conditional on public context/profile; not population-frequency weighted",
        "curve_interpretation": "empirical frozen-score thresholds; adjacent threshold randomization; no confidence band or population PR-dominance claim",
        "pr_order_fpr_numerical_tolerance": 1e-10,
        "utility_scope": "full held-out D_test with a common frozen CTR service; attack comparisons cover common observed attribute pairs",
        "budget_comparison": "CAPT budgets share each cost/frozen-seed/K/L cell; MaSS rows are not assigned a certified epsilon",
        "audit_interpretation": "simultaneous Clopper-Pearson event probability bounds over all supplied cells/models/pairs/events, under test user-day independence; lower witnesses only, never certificates",
        "eau_interval_scope": "same joint event-family confidence allocation as event audit",
        "coverage_interpretation": "unobserved/excluded pairs remain outside empirical comparison, not zero leakage",
        "true_infinite_epsilon_proven": False,
        "mass_curve_axis": mass_curve_axis,
        "same_eau_interpretation": "all intersections of parameter-ordered observed segments within their overlap; descriptive interpolation only, no extrapolation or envelope selection",
    }
    (output / "comparison.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_report(summary, pd.DataFrame(coverage), pd.DataFrame(dominance), output, prior)
    return output


def _mass_capt_joint_comparison(audit, summary):
    """Separate MaSS lower witnesses from CAPT upper certificates.

    Two lower bounds alone do not establish privacy ordering. The CAPT side
    remains conditional on its certificate's sampling and shift assumptions.
    """
    records = []
    for cell, frame in summary.groupby("cell_id"):
        for capt in frame.loc[frame.method.eq("capt")].to_dict("records"):
            valid = capt.get("certificate_valid") in (True, "True")
            upper = float(capt.get("certified_upper_epsilon", np.nan))
            for mass in frame.loc[frame.method.eq("mass12_raw")].to_dict("records"):
                events = audit.loc[audit.model_id.eq(mass["model_id"])]
                witness = events.loc[events.epsilon_lower.idxmax()]
                separates = bool(valid and np.isfinite(upper) and witness.epsilon_lower > upper)
                utility = bool(capt["empirical_log_loss"] <= mass["empirical_log_loss"])
                records.append({
                    "cell_id": cell, "capt_id": capt["model_id"], "mass_id": mass["model_id"],
                    "mass_label": mass["label"], "capt_target_epsilon": capt["target_epsilon"],
                    "capt_reported_certificate_valid": valid, "capt_certified_upper_epsilon": upper,
                    "mass_epsilon_lower": witness.epsilon_lower, "witness_pair_id": witness.pair_id,
                    "witness_event_type": witness.event_type, "witness_score": witness.score,
                    "witness_positive_trials": witness.positive_trials,
                    "witness_negative_trials": witness.negative_trials,
                    "witness_positive_successes": witness.positive_successes,
                    "witness_negative_successes": witness.negative_successes,
                    "mass_lower_exceeds_capt_upper": separates,
                    "capt_logloss_no_higher": utility,
                    "mass_minus_capt_logloss": mass["empirical_log_loss"]-capt["empirical_log_loss"],
                    "joint_separation_observed": separates and utility,
                    "scope": "conditional on CAPT certificate assumptions and simultaneous held-out audit coverage; utility on full D_test",
                })
    return pd.DataFrame(records)


def _audit_figure(audit, summary, output):
    for cell, group in summary.groupby("cell_id", sort=True):
        batches = _batches(group.to_dict("records"), lambda row: row["method"])
        for page, records in enumerate(batches):
            frame = pd.DataFrame(records)
            selected = audit.loc[audit.model_id.isin(frame.model_id)]
            name = "rare_events" if summary.cell_id.nunique() == 1 else f"rare_events_{cell}"
            if len(batches) > 1:
                name += f"_page{page + 1}"
            _audit_page(selected, frame, output, name)


def _audit_page(audit, summary, output, name):
    fig, axis = plt.subplots(figsize=(8.5, 4.6))
    for row in summary.to_dict("records"):
        events = audit.loc[
            audit.model_id.eq(row["model_id"]) & audit.event_type.eq("extreme_frozen_score_bucket")
        ]
        mass = events[["positive_probability", "negative_probability"]].max(axis=1)
        keep = mass > 0
        axis.scatter(
            mass[keep],
            events.loc[keep, "epsilon_lower"],
            label=row["label"],
            s=9,
            alpha=0.35,
            rasterized=True,
        )
    axis.set_xscale("log")
    axis.set(
        xlabel="Larger empirical group probability of score-bucket event",
        ylabel="Simultaneous lower bound on attribute privacy loss",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, frameon=False)
    fig.suptitle("Held-out rare-score-event audit", fontsize=12)
    fig.text(
        0.5,
        0.015,
        "A zero observed count does not prove infinite epsilon. A zero lower bound does not prove privacy.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save(fig, output, name)


def _write_report(summary, coverage, dominance, output, prior):
    lines = [
        "# Saved attribute-privacy comparison",
        "",
        f"Positive hypothesis prior: {prior:g}. Lower attack precision / EAU / AP and lower CTR log loss are favorable.",
        "",
        "EAU follows IPSJ-TOD1902006 Definition 3.2.1 (printed p. 56): attack correctness probability. Here it is explicitly a binary pairwise, common-prior EAU for a learned likelihood-ratio decision rule fixed before test. It is not the optimal attack supremum, an individual IEAU, or an unconditional multiclass accuracy.",
        "",
        "The 2019 hypothesis-testing paper motivates PR relations (Section 3, Theorem 2; Section 5.1 for prior sensitivity). Its analytic optimal attacks on neighboring databases are adapted here to learned attacks on attribute values within a fixed public context/profile.",
        "",
        "All curves and scalar comparisons within a cell use exactly the same completed attribute pairs. Excluded groups remain unassessed. Macro curves are equal-pair averages; the worst-pair envelope can switch pair at each recall. Neither is a single population-optimal attacker.",
        "",
        "PR order is checked pair by pair at every piecewise-linear ROC knot and one-sided limit, not just at plotting-grid points. Apparent order is descriptive: there is no simultaneous confidence band for PR or evidence of dominance over every possible attacker.",
        "",
        "EAU intervals and event privacy-loss lower bounds share a simultaneous binomial confidence allocation over all evaluated events. Test user-days must be independent within groups. A zero denominator count cannot establish infinite population privacy loss.",
        "",
        "## Mechanisms",
        "",
        "| Mechanism | Common pairs | Mean pair EAU | Worst pair EAU | Mean AP | CTR log loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['label']} | {row['common_pairs']} | {row['mean_pair_eau']:.6f} | {row['worst_pair_eau']:.6f} | {row['mean_average_precision']:.6f} | {row['empirical_log_loss']:.6f} |"
        )
    lines += [
        "",
        "## Coverage",
        "",
        "~~~text",
        coverage[
            ["label", "expected_pairs", "eligible_pairs", "evaluated_pairs", "common_pairs"]
        ].to_string(index=False),
        "~~~",
        "",
        "## Observed comparisons",
        "",
        "~~~text\n" + dominance.to_string(index=False) + "\n~~~"
        if len(dominance)
        else "No matching CAPT reference.",
        "",
        "The parameter-sweep EAU/logloss curves connect observed settings in parameter order. same_eau_logloss.csv retains every segment intersection in the overlapping EAU range. Interpolated utilities are descriptive estimates, not new trained settings; folds and vertical segments are retained and there is no extrapolation.",
        "",
        "The main eau_logloss_curve figure uses one axis pair: mean pairwise EAU versus CTR logloss. Mean is the equal-pair average over the common evaluated attribute pairs/contexts. The separate supplementary eau_logloss_largest figure uses the largest observed pairwise EAU; small groups can make this maximum noisy. These aggregates are across pairs, not seeds, and the mean is not population-frequency weighted.",
        "",
        "mass_capt_leakage_utility.csv checks whether a MaSS simultaneous held-out leakage lower witness exceeds the reported CAPT certificate upper bound while CAPT has no higher full-test log loss. Comparing two audit lower bounds alone cannot establish privacy ordering. This check is conditional on the CAPT certificate assumptions. These results do not establish a population privacy guarantee for MaSS.",
        "",
    ]
    lines += ["## Figures", ""]
    cells = set(summary.cell_id)
    for path in sorted((output / "figures").glob("*.png")):
        if path.stem == "rare_events" or any(cell in path.stem for cell in cells):
            lines.append(f"- [{path.stem}]({path.relative_to(output).as_posix()})")
    lines += [
        "",
        "The generic MI counterexample is excluded by default; it is not an actual MaSS/CAPT experiment.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
