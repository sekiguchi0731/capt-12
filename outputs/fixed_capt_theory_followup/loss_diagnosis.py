"""Reproduce the exploratory comparison from the repository root using .venv/bin/python."""

import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import xlogy

from capt12.comparison.optimal_ldp.numeric import DOWN, UP, decimal_ratio, float_bound, log_bound
from capt12.comparison.optimal_ldp.retraining import fit_ctr
from capt12.comparison.optimal_ldp.theory import Bounds, weighted_bounds
from capt12.comparison.optimal_ldp.utility import (
    RandomizedResponse,
    bayes_loss_bounds,
    joint_probabilities,
    solve_spr,
)
from capt12.experiments.fixed_capt_ctr_comparison import (
    ComparisonSettings,
    _check_context_groups,
    _encode_tokens,
    _fixed_input_columns,
    _load_selected_displays,
)
from capt12.experiments.fixed_capt_theory import _load_bundle, _read_result_files
from capt12.models.reference import ReferenceModel

OUTPUT = Path(__file__).resolve().parent
THEORY = OUTPUT.parent / "fixed_capt_theory_grid81" / "theory.json"
PRIOR_WEIGHTS = (0.0, 1.0, 10.0, 100.0, math.inf)


def histograms(theory, saved, certificate, cells):
    config = certificate.resolved_config
    models = Path(theory["certificate_dir"]) / "models"
    mapper = joblib.load(models / "category_mapper.joblib")
    encoder = joblib.load(models / "encoder.joblib")
    k = theory["K"]
    bcol = theory["context_column"]
    profile = theory["profile"]
    label = config["label_col"]
    counts = {}
    seen = pd.Index([], dtype=str)
    for split in ("train", "validation"):
        joint = {b: np.zeros((k, 2), dtype=np.int64) for b in cells}
        days = saved["splits"][f"D_predict_{split}"]
        for day in days:
            if split == "train" and day == days[-1]:
                counts["inner_fit"] = {b: values.copy() for b, values in joint.items()}
            frame, seen = _load_selected_displays(
                Path(theory["data_root"]),
                [day],
                [*_fixed_input_columns(config), label],
                mapper,
                id_col=config["id_col"],
                user_col=config["user_col"],
                seen_ids=seen,
            )
            for value, group in frame.groupby(bcol, sort=False):
                b = str(value)
                _check_context_groups(group, profile.split("+"), profile, b, cells[b]["groups"])
                y = group[label].to_numpy()
                assert np.isin(y, [0, 1]).all()
                for start in range(0, len(group), 16384):
                    take = slice(start, start + 16384)
                    z = _encode_tokens(encoder, group.iloc[take], k)
                    joint[b] += np.bincount(2 * z + y[take].astype(int), minlength=2 * k).reshape(
                        k, 2
                    )
            del frame
        counts[split] = joint
    return counts


def threshold_rr(scores, counts):
    """Train only the cut on the supplied scores and law. Tied scores stay together."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(order)]
    prefixes = np.vstack((np.zeros(2, dtype=int), counts[order].cumsum(axis=0)))
    left = prefixes[cuts]
    right = counts.sum(axis=0) - left
    flip = 1 / (1 + math.e)

    def loss(mass):
        probability = np.divide(
            mass,
            mass.sum(1, keepdims=True),
            out=np.zeros_like(mass),
            where=mass.sum(1, keepdims=True) > 0,
        )
        return -np.sum(xlogy(mass, probability), axis=1)

    losses = loss((1 - flip) * left + flip * right) + loss(flip * left + (1 - flip) * right)
    cut = cuts[np.argmin(losses)]
    assignment = np.ones(len(order), dtype=int)
    assignment[order[:cut]] = 0
    return RandomizedResponse(assignment, 2, 1.0)


def by_bin(assignment, counts, n):
    totals = np.zeros((n, 2), dtype=np.int64)
    np.add.at(totals, assignment, counts)
    return totals


def smoothed_scores(reference, counts, weight):
    if math.isinf(weight):
        return reference
    totals = counts.sum(axis=1)
    return np.divide(
        counts[:, 1] + weight * reference,
        totals + weight,
        out=reference.copy(),
        where=totals + weight > 0,
    )


def fitted_risk(channel, train, evaluate):
    return fit_ctr(channel, train, train, pseudocounts=(1.0,), eta=0.01).loss(evaluate)


def select_prior_and_route(counts, cells, scores):
    """Learn one prior strength and all public-context routes strictly within predictor-training days."""
    tuning = {}
    for b, cell in cells.items():
        fit = counts["inner_fit"][b]
        tune = counts["train"][b] - fit
        n = int(tune.sum())
        if not fit.sum() or not n:
            # Missing contexts always take the original-token LDP branch.
            tuning[b] = (n, [math.log(2)] * len(PRIOR_WEIGHTS), math.log(2))
            continue
        risks = [
            fitted_risk(threshold_rr(smoothed_scores(scores[b], fit, s), fit), fit, tune)
            for s in PRIOR_WEIGHTS
        ]
        tuning[b] = (n, risks, fitted_risk(cell["channel"], fit, tune))
    total = sum(x[0] for x in tuning.values())
    losses = [
        math.fsum(n * risks[i] for n, risks, _ in tuning.values()) / total
        for i in range(len(PRIOR_WEIGHTS))
    ]
    index = int(np.argmin(losses))
    routes = {
        b: ("capt" if capt < risks[index] else "ldp") for b, (n, risks, capt) in tuning.items()
    }
    return PRIOR_WEIGHTS[index], routes, dict(zip(map(str, PRIOR_WEIGHTS), losses, strict=True))


def raw_epsilon(channel):
    # Include every reachable block over the entire raw alphabet even if validation count is zero.
    rows = [channel.rows[c] for c in np.unique(channel.assignment)]
    ratios = [max(column) / min(column) for column in zip(*rows, strict=True) if max(column) > 0]
    return Bounds(
        *(float_bound(log_bound(decimal_ratio(max(ratios), ctx), ctx), ctx) for ctx in (DOWN, UP))
    )


def summarize(rows):
    n = sum(r["validation_n"] for r in rows)
    weights = [Fraction(r["validation_n"], n) for r in rows]

    def avg(key):
        return math.fsum(float(w) * r[key] for w, r in zip(weights, rows, strict=True))

    def interval(key):
        return weighted_bounds(
            [Bounds(r[key + "_lower"], r[key + "_upper"]) for r in rows], weights
        )

    def by_context(op):
        return vars(weighted_bounds([op(r) for r in rows], weights))

    def exact(row, name):
        return Bounds(row[name + "_lower"], row[name + "_upper"])

    def v_z(row):
        return Bounds(
            row["epsilon_1_v_z_lower"], min(row["epsilon_1_v_z_upper"], row["oracle_binary_upper"])
        )

    def gap_vs_capt(row):
        return v_z(row).minus(exact(row, "d_capt"))

    def oracle_gain(row, relative_to):
        gap = gap_vs_capt(row)
        if relative_to == "capt":
            return Bounds(max(0.0, -gap.upper), max(0.0, -gap.lower))
        return Bounds(max(0.0, gap.lower), max(0.0, gap.upper))

    out = {
        key: avg(key)
        for key in (
            "h_y_given_z",
            "h_y_given_c",
            "d_capt",
            "i_y_z",
            "i_y_z_given_c",
            "i_y_c_given_release",
            "simple_reference_rr_train_to_val_loss",
            "capt_train_to_val_loss",
            "tuned_rr_train_to_val_loss",
            "train_selected_hybrid_loss",
            "plugin_selected_hybrid_loss",
        )
    }
    out.update(
        contexts=len(rows),
        validation_n=n,
        tokens_seen_once_share=avg("single_observation_mass"),
        train_gate_capt_contexts=[
            r["context_index"] for r in rows if r["selected_on_train"] == "capt"
        ],
        v_z_grid=vars(interval("epsilon_1_v_z")),
        v_z=by_context(v_z),
        v_c=vars(interval("v_c")),
        oracle_binary=vars(interval("oracle_binary")),
        gap=by_context(gap_vs_capt),
        ldp_penalty_grid=vars(interval("ldp_penalty")),
        v_c_minus_v_z=by_context(lambda r: exact(r, "v_c").minus(v_z(r))),
        v_c_minus_capt=vars(interval("v_c_minus_capt")),
        binary_minus_v_z=vars(interval("binary_minus_v_z")),
        hybrid_loss=by_context(
            lambda r: Bounds(
                min(r["d_capt_lower"], v_z(r).lower),
                min(r["d_capt_upper"], v_z(r).upper),
            )
        ),
        hybrid_gain_over_ldp=by_context(lambda r: oracle_gain(r, "ldp")),
        hybrid_gain_over_capt=by_context(lambda r: oracle_gain(r, "capt")),
        block_larger=sum(r["i_y_z_given_c"] > r["i_y_c_given_release"] for r in rows),
        capt_beats_block_ldp=sum(r["v_c_minus_capt_lower"] > 0 for r in rows),
        block_ldp_beats_capt=sum(r["v_c_minus_capt_upper"] < 0 for r in rows),
        raw_ldp_certified_contexts=[
            r["context_index"] for r in rows if r["raw_epsilon_upper"] <= 1
        ],
        full_simplex_edge_contexts=[r["context_index"] for r in rows if r["simplex_edges"] > 0],
        no_noise_block_beats_optimal_ldp=sum(
            r["h_y_given_c"] + 1e-12 < r["epsilon_1_v_z_lower"] for r in rows
        ),
        repair_max=max(r["repair_weight"] for r in rows),
    )
    return out


def run():
    theory = json.loads(THEORY.read_text())
    saved = _read_result_files(Path(theory["results_dir"]))
    first = next(iter(saved.values()))
    bundle = Path(theory["certificate_dir"])
    certificate, cells = _load_bundle(saved, bundle, ComparisonSettings(**first["settings"]))
    counts = histograms(theory, first, certificate, cells)
    reference = ReferenceModel.load(bundle / "models" / "reference.joblib")
    scores = {
        b: np.asarray(
            reference.predict(
                pd.DataFrame({"__token__": np.arange(theory["K"]), theory["context_column"]: b})
            ),
            dtype=float,
        )
        for b in cells
    }
    weight, routes, prior_risks = select_prior_and_route(counts, cells, scores)
    details = pd.read_parquet(bundle / "context_results.parquet").set_index("public_context")
    rows = []
    for original in theory["contexts"]:
        b = original["public_context_value"]
        train, validation = (counts[split][b] for split in ("train", "validation"))
        assert train.sum() == original["train_n"] and validation.sum() == original["validation_n"]
        if not validation.sum():
            continue
        channel = cells[b]["channel"]
        blocks = by_bin(channel.assignment, validation, len(channel.rows))
        hc = Bounds(*bayes_loss_bounds(joint_probabilities(blocks)))
        assert hc.lower - 1e-15 <= original["h_y_given_c"] <= hc.upper + 1e-15
        cap = Bounds(*bayes_loss_bounds(channel.exact_output_joint(validation)))
        assert abs(cap.midpoint - original["d_capt"]) < 1e-15
        vc = solve_spr(blocks, 1.0)
        score = scores[b]
        practical = threshold_rr(score, train)
        tuned = threshold_rr(smoothed_scores(score, train, weight), train)

        raw_n = validation.sum(axis=1)
        empirical_score = np.divide(
            validation[:, 1], raw_n, out=np.full(len(raw_n), 0.5), where=raw_n > 0
        )
        binary = threshold_rr(empirical_score, validation)
        chosen_bins = by_bin(binary.assignment, validation, 2)
        # The binary SPR enclosure is exact for this fixed two-bin representation.
        binary_spr = solve_spr(chosen_bins, 1.0)
        released = binary.output_counts(validation)
        direct_risk = (
            -xlogy(released, released / released.sum(1, keepdims=True)).sum() / released.sum()
        )
        assert abs(binary_spr.estimate - direct_risk) < 2e-14
        original_vz = Bounds(original["epsilon_1_v_z_lower"], original["epsilon_1_v_z_upper"])
        block_ldp, binary_risk = (
            Bounds(vc.lower, vc.upper),
            Bounds(binary_spr.lower, binary_spr.upper),
        )
        assert original_vz.lower <= min(block_ldp.upper, binary_risk.upper) + 1e-14
        hz = Bounds(*bayes_loss_bounds(joint_probabilities(validation)))
        data = details.loc[b]
        row = {
            key: original[key]
            for key in (
                "context_index",
                "public_context_value",
                "validation_n",
                "validation_weight",
                "train_n",
                "epsilon_1_empirical_decision",
                "epsilon_1_v_z_lower",
                "epsilon_1_v_z_upper",
                "epsilon_1_gap_lower",
                "epsilon_1_gap_upper",
                "d_capt",
                "d_capt_lower",
                "d_capt_upper",
                "h_y_given_z",
                "h_y_given_c",
                "i_y_z",
                "i_y_z_given_c",
                "i_y_c_given_release",
            )
        }
        capt_risk = fitted_risk(channel, train, validation)
        tuned_risk = fitted_risk(tuned, train, validation)
        row.update(
            capt_train_to_val_loss=capt_risk,
            simple_reference_rr_train_to_val_loss=fitted_risk(practical, train, validation),
            tuned_rr_train_to_val_loss=tuned_risk,
            selected_on_train=routes[b],
            train_selected_hybrid_loss=capt_risk if routes[b] == "capt" else tuned_risk,
            plugin_selected_hybrid_loss=capt_risk
            if original["epsilon_1_empirical_decision"] == "empirical_plugin_win"
            else tuned_risk,
            distinct_observed_z=int((raw_n > 0).sum()),
            single_observation_mass=float((raw_n == 1).sum() / raw_n.sum()),
            block_spr_outputs=int(vc.channel.outputs),
            simplex_edges=int(data["full_simplex_ordered_edge_count"]),
            repair_weight=float(data["repair_lambda"]),
        )
        intervals = {
            "ldp_penalty": original_vz.minus(hz),
            "raw_epsilon": raw_epsilon(channel),
            "v_c": block_ldp,
            "oracle_binary": binary_risk,
            "v_c_minus_v_z": block_ldp.minus(original_vz),
            "v_c_minus_capt": block_ldp.minus(cap),
            "binary_minus_v_z": binary_risk.minus(original_vz),
        }
        for key, interval in intervals.items():
            for side, value in vars(interval).items():
                row[f"{key}_{side}"] = (
                    max(0.0, value) if key in ("v_c_minus_v_z", "binary_minus_v_z") else value
                )
        rows.append(row)

    out = {"all": summarize(rows)}
    for status in ("win", "loss", "inconclusive"):
        out[status] = summarize(
            [r for r in rows if r["epsilon_1_empirical_decision"] == "empirical_plugin_" + status]
        )
    out["training"] = {
        "fit_days": first["splits"]["D_predict_train"][:-1],
        "prior_and_gate_selection_day": first["splits"]["D_predict_train"][-1],
        "selection_losses": prior_risks,
        "chosen_prior_weight": str(weight),
        "fallback_gate": "ldp",
        "predictor_pseudocount": 1.0,
        "predictor_clip": 0.01,
        "holdout_days": theory["validation_days"],
    }
    with (OUTPUT / "context-diagnosis.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print("Validation contexts", len(rows), "user-days", out["all"]["validation_n"])


if __name__ == "__main__":
    run()
