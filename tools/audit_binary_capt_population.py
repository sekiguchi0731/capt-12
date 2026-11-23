"""Audit the L32 binary-direct-cp run on the original user-days without refitting."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from capt12.comparison.binary_capt import design_label_counts
from capt12.comparison.optimal_ldp.numeric import (
    DOWN,
    UP,
    decimal_ratio,
    float_bound,
    multinomial_tv_radius,
)
from capt12.comparison.optimal_ldp.population import (
    binary_loss_cells,
    check_binary_cp_interval,
    frozen_capt_ucb,
    ldp_population_lower,
)
from capt12.comparison.optimal_ldp.retraining import (
    FixedBlocks,
    _continuity_limits,
    _entropy_radius,
)
from capt12.comparison.optimal_ldp.theory import Bounds, weighted_bounds
from capt12.comparison.optimal_ldp.utility import grid_bounds
from capt12.confidence.boxes import cp_box, full_simplex_box
from capt12.data.loader import assert_disjoint_splits, load_parquet_sample
from capt12.experiments.fixed_capt_ctr_comparison import (
    _certified_block_beta,
    _encode_tokens,
    _fixed_input_columns,
    _load_selected_displays,
    _verified_fixed_channel,
)
from capt12.utils.artifacts import sha256_file

ALPHA_CAPT = ALPHA_LDP = Fraction(3, 2000)
GRID = tuple(i / 64 for i in range(65))
BUDGET = 1.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_sources(results: Path, run: Path):
    sources = {}
    for path in sorted(run.glob("certificate-*-context-*.json")):
        idx = int(path.stem.rsplit("-", 1)[1])
        cert = json.loads(path.read_text())
        b = str(cert["resolved_config"]["public_context_value"])
        require(
            b not in sources and idx not in [s[0] for s in sources.values()], "duplicate context"
        )
        result = results / f"{idx:02d}.json"
        rec = json.loads(result.read_text()) if result.exists() else {}
        if rec:
            require(
                Path(rec["certificate"]).name == path.name
                and str(rec["context"]) == b
                and rec["certificate_sha256"] == sha256_file(path),
                "result/certificate mismatch",
            )
        sources[b] = (idx, path, cert, rec)
    require(bool(sources), "no source certificates")
    return sources


def raw_sample(sources, run: Path, data: Path):
    """Recover eligibility before test, then check exact original-token counts and day IDs."""
    cfg = next(iter(sources.values()))[2]["resolved_config"]
    record = next(rec for _, _, _, rec in sources.values() if rec and rec.get("binary_channel"))
    splits = record["splits"]
    roles = {
        "train": splits["D_predict_train"],
        "validation": splits["D_predict_validation"],
        "test": splits["D_test"],
    }
    base = {k: splits[k] for k in ("D_model", "D_design", "D_cert")}
    assert_disjoint_splits({**base, **roles})
    require(
        sorted(splits["D_attack_train"]) == sorted(roles["train"] + roles["validation"]),
        "predictor splits must partition the attack-training period",
    )
    files = {
        "encoder": run / "models/encoder.joblib",
        "mapper": run / "models/category_mapper.joblib",
    }
    hashes = {name: sha256_file(path) for name, path in files.items()}
    id_col, user, label, context = (
        cfg["id_col"],
        cfg["user_col"],
        cfg["label_col"],
        cfg["context_cols"][0],
    )
    k = cfg["K"]
    for _, _, cert, rec in sources.values():
        other = cert["resolved_config"]
        require(
            all(
                other[key] == cfg[key]
                for key in (
                    "K",
                    "L",
                    "context_cols",
                    "profiles",
                    "id_col",
                    "user_col",
                    "label_col",
                    "phi_source_cols",
                    "contribution_policy",
                    "sampling_assumption",
                )
            )
            and all(cert["component_hashes"][key] == v for key, v in hashes.items())
            and all(
                cert["split_identifiers"][key] == splits[key] for key in cert["split_identifiers"]
            ),
            "source contexts differ in maps, features, sampling or periods",
        )
        if rec and rec.get("binary_channel"):
            require(
                rec["splits"] == splits
                and rec["K"] == k
                and rec["context_column"] == context
                and rec["profile"] == cfg["profiles"][0]
                and not rec["observed_profile_selection"]
                and rec["capt_mechanism"] == "binary-direct-cp",
                "inconsistent comparison population",
            )
    mapper, encoder = joblib.load(files["mapper"]), joblib.load(files["encoder"])
    sizes = {role: Counter() for role in roles}
    counts = {b: np.zeros((k, 2), dtype=np.int64) for b in sources}
    seen = pd.Index([], dtype=str)
    test_ids = set()
    test_displays = set()
    days = []
    # The selection uses only the display ID and day-local user ID before context filtering.
    for role, dates in roles.items():
        for day in dates:
            cols = (
                [*_fixed_input_columns(cfg), label]
                if role == "test"
                else list(dict.fromkeys([id_col, user, *mapper.mappings]))
            )
            frame, seen = _load_selected_displays(
                data, [day], cols, mapper, id_col=id_col, user_col=user, seen_ids=seen
            )
            context_values = frame[context].astype(str)
            sizes[role].update(context_values.value_counts().to_dict())
            require(set(sizes[role]) <= sources.keys(), "observed an unrepresented public context")
            if role != "test":
                continue
            eligible = {b for b in sources if sizes["train"][b] and sizes["validation"][b]}
            selected = frame.loc[context_values.isin(eligible)]
            overlap = len(test_ids.intersection(selected[user]))
            test_ids.update(selected[user])
            test_displays.update(selected[id_col])
            days.append(
                {
                    "day": day,
                    "eligible": len(selected),
                    "all": len(frame),
                    "eligible_ids_seen_on_previous_test_day": overlap,
                }
            )
            for b, group in frame.groupby(context, sort=False):
                for start in range(0, len(group), 16384):
                    part = group.iloc[start : start + 16384]
                    z = _encode_tokens(encoder, part, k)
                    y = part[label].to_numpy()
                    require(np.isin(y, [0, 1]).all(), "nonbinary test outcome")
                    counts[str(b)] += np.bincount(2 * z + y.astype(int), minlength=2 * k).reshape(
                        k, 2
                    )
            print(f"Rebuilt test day {day}, {len(selected):,} eligible user-days", flush=True)
    # IDs are deliberately rotated. These counts check released identifiers only.
    prior_matches = {}
    for day in sorted(
        sum((list(x) for x in (*base.values(), roles["train"], roles["validation"])), [])
    ):
        prior = load_parquet_sample(data_root=data, days=[day], columns=[user, id_col])
        prior_matches[day] = len(test_ids.intersection(prior[user].unique()))
        require(
            test_displays.isdisjoint(prior[id_col]), "eligible test display reused in a prior role"
        )
    eligible = {b for b in sources if sizes["train"][b] and sizes["validation"][b]}
    require(
        eligible == {b for b, (_, _, _, rec) in sources.items() if rec.get("binary_channel")},
        "frozen predictors do not cover exactly the pretest-eligible contexts",
    )
    manifest_path = data.parent / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for day in sorted(prior_matches.keys() | set(roles["test"])):
        items = [v for v in manifest["selected_files"] if v["day"] == day]
        paths = list((data / f"day_int={day}").glob("*.parquet"))
        require(
            len(items) == len(paths) == 1
            and Path(items[0]["path"]).name == paths[0].name
            and items[0]["size"] == paths[0].stat().st_size,
            "local shard/manifest mismatch",
        )
    meta = dict(
        splits=splits,
        model_sha256=hashes,
        selection=manifest["selection"],
        manifest_sha256=sha256_file(manifest_path),
        test_days=days,
        test_ids_matching_earlier_days=prior_matches,
        prior_displays_matching_eligible_test=0,
        distinct_eligible_day_local_ids=len(test_ids),
        privacy_sampling=cfg["sampling_assumption"],
        observed_profile_selection=False,
        profile=record["profile"],
        context_column=context,
        online_memoization=record["online_memoization"],
        population=record["population"],
        saved_utility_assumption=record["utility_assumption"],
    )
    return eligible, counts, sizes, meta


def verify_release(path, rec, design_path, design_hash):
    cert, parent, _, edges, _ = _verified_fixed_channel(path)
    spec, calibration = rec["binary_channel"], rec["calibration"]
    cfg = cert.resolved_config
    alpha = float(cfg["alpha_cert"])
    groups = cert.histogram_counts
    family = len(parent.rows) - 1
    require(
        cert.confidence["coverage_basis"] == "finite_sample"
        and cert.tv_radius == 0
        and calibration["block_shift_tv"] == 0
        and calibration["raw_shift_assumption"] is None
        and rec["settings"]["raw_shift_tv"] is None
        and spec["split"] == "D_design"
        and spec["design_sha256"] == design_hash
        and spec["alpha_context"] == alpha == calibration["attribute_bound_alpha"]
        and spec["threshold_family_size"] == family
        and edges
        and all(e.epsilon == BUDGET for e in edges),
        "unsupported certification scope",
    )
    design = design_label_counts(design_path, rec["context"], parent.assignment, len(parent.rows))
    totals = design.sum(axis=1)
    order = np.argsort(
        np.divide(design[:, 1], totals, out=np.full(len(totals), 0.5), where=totals > 0),
        kind="stable",
    )
    high = np.isin(np.arange(len(totals)), spec["high_blocks"])
    require(
        0 < high.sum() < len(high)
        and set(order[-high.sum() :]) == set(spec["high_blocks"])
        and int(totals.sum()) == spec["design_displays"],
        "cut outside frozen design family",
    )
    boxes = {}
    # The two cells are complements, so there is only one binomial parameter per group/cut.
    tail = Fraction(alpha) / (2 * len(groups) * family)
    for g, hist in groups.items():
        v = np.array(hist)
        bins = np.array([v[~high].sum(), v[high].sum()])
        n, k = int(bins.sum()), int(bins[1])
        if n:
            # cp_box treats both cells as separate tests. Propose endpoints at
            # (2/3)*tail and validate both actual tails below tail using directed arithmetic.
            # Coverage uses only that validation, independent of the quantile library.
            box = cp_box(bins, alpha=4 * alpha / 3, group_count=len(groups), comparisons=family)
            low = max(Fraction(box.lower[1]), 1 - Fraction(box.upper[0]))
            upper = min(Fraction(box.upper[1]), 1 - Fraction(box.lower[0]))
            check_binary_cp_interval(n, k, low, upper, tail)
        else:
            box = full_simplex_box(bins)
        boxes[g] = box
    probs = list(map(Fraction, spec["probability_of_one_low_high"]))
    require(len(probs) == 2 and all(0 <= q <= 1 for q in probs), "invalid release probabilities")
    bit = high[parent.assignment]
    channel = FixedBlocks(bit.astype(int), tuple((1 - q, q) for q in probs))
    beta = _certified_block_beta(channel, boxes, edges)
    lam = calibration["beta_R"]
    require(
        beta <= lam <= BUDGET
        and min(bit.sum(), (~bit).sum()) >= 2
        and calibration["tv_method"]
        == calibration["uniform_method"]
        == "exact_disjoint_token_preimages"
        and calibration["delta_tv_upper"] == calibration["j_beta_upper"] == 1
        and all(
            rec["comparisons"][key]["lambda"] == calibration[field] == lam
            for key, field in (
                ("uniform_calibrated", "uniform_budget"),
                ("tv_calibrated", "tv_budget"),
            )
        ),
        "calibrated budgets outside verified common envelope",
    )
    return bit, alpha, beta


def audit(results: Path, run: Path, data: Path, saved_counts: Path | None):
    sources = read_sources(results, run)
    eligible, counts, sizes, sampling = raw_sample(sources, run, data)
    if saved_counts is not None:
        with np.load(saved_counts, allow_pickle=False) as saved:
            require(
                set(saved.files) == eligible
                and all(np.array_equal(saved[b], counts[b]) for b in eligible),
                "saved and reconstructed original-token test histograms differ",
            )
    design = run / "mechanism/frozen_design.npz"
    design_hash = sha256_file(design)
    rows, all_cells, intervals = [], [], []
    privacy_error = Fraction(0)
    n = sum(sizes["test"][b] for b in eligible)
    for b, (idx, path, _, rec) in sources.items():
        if b not in eligible:
            continue
        ns = {role: int(values[b]) for role, values in sizes.items()}
        require(
            ns == rec["utility_sample_sizes"] and counts[b].sum() == ns["test"],
            "different saved sample sizes",
        )
        bit, alpha, verified_beta = verify_release(path, rec, design, design_hash)
        spec = rec["binary_channel"]
        cells = binary_loss_cells(
            counts[b], bit, spec["probability_of_one_low_high"], spec["fitted_output_ctr"]
        )
        all_cells.extend(cells)
        privacy_error += Fraction(alpha)
        # An eligible future-positive context with no test observations still enters the support.
        if ns["test"]:
            vz = grid_bounds(counts[b], BUDGET, GRID)
            intervals.append((ns["test"], Bounds(vz.lower, vz.upper)))
            loss = weighted_bounds(
                [c.loss for c in cells if c.count],
                [Fraction(c.count, ns["test"]) for c in cells if c.count],
            )
            require(
                abs(loss.midpoint - rec["capt_test_log_loss"]) < 1e-12, "saved CAPT loss mismatch"
            )
        else:
            vz = loss = None
        rows.append(
            dict(
                index=idx,
                context=b,
                **ns,
                test_weight=ns["test"] / n,
                lambda_tv=rec["comparisons"]["tv_calibrated"]["lambda"],
                lambda_uniform=rec["comparisons"]["uniform_calibrated"]["lambda"],
                beta_R=rec["calibration"]["beta_R"],
                audited_beta_upper=verified_beta,
                alpha_privacy=alpha,
                empirical_capt=loss.midpoint if loss else None,
                empirical_v1_lower=vz.lower if vz else None,
                empirical_v1_upper=vz.upper if vz else None,
                result_sha256=sha256_file(results / f"{idx:02d}.json"),
                certificate_sha256=rec["certificate_sha256"],
            )
        )
        print(
            f"Verified frozen binary CAPT and 65-point LDP grid for context {idx:02d}", flush=True
        )
    total_alpha = privacy_error + ALPHA_CAPT + ALPHA_LDP
    require(
        total_alpha <= Fraction(1, 20), "eligible privacy family exhausts joint confidence budget"
    )
    empirical = weighted_bounds([v for _, v in intervals], [Fraction(nb, n) for nb, _ in intervals])
    cap = frozen_capt_ucb(all_cells, ALPHA_CAPT)
    lower, correction = ldp_population_lower(empirical.lower, n, ALPHA_LDP)
    margin = Bounds(lower, lower).minus(Bounds(cap.upper, cap.upper)).lower
    k = len(next(iter(counts.values())))
    alphabet = 2 * k * len(eligible)
    tv = multinomial_tv_radius(n, alphabet, float_bound(decimal_ratio(ALPHA_LDP, DOWN), DOWN))
    entropy_penalty = _entropy_radius(tv)
    tv_lower, _ = _continuity_limits(empirical.lower, empirical.upper, entropy_penalty)
    neff = 1 / sum((d["eligible"] / n) ** 2 for d in sampling["test_days"])
    day_ucb = min(
        cap.support.upper,
        cap.mean.upper
        + (cap.support.upper - cap.support.lower)
        * math.sqrt(math.log(1 / float(ALPHA_CAPT)) / (2 * neff)),
    )
    sensitivity = {}
    for m in (1, 10, 100, 1000, 1_000_000):
        u = frozen_capt_ucb(all_cells, ALPHA_CAPT / m).upper
        sensitivity[str(m)] = dict(
            capt_ucb=u, margin_lower=Bounds(lower, lower).minus(Bounds(u, u)).lower
        )
    excluded = {
        b: dict(index=sources[b][0], **{role: int(values[b]) for role, values in sizes.items()})
        for b in sorted(sources.keys() - eligible)
    }
    summary = dict(
        population_decision="assumptions_not_enabled",
        retrospective_holdout=True,
        physical_user_independence_verified=False,
        sampling=sampling,
        excluded=excluded,
        contexts=len(eligible),
        n=n,
        K=k,
        scores_and_channel_refitted=False,
        grid_points=len(GRID),
        common_ldp_envelope=BUDGET,
        counts_sha256=sha256_file(saved_counts) if saved_counts else None,
        design_sha256=design_hash,
        alpha_privacy_upper=float_bound(decimal_ratio(privacy_error, UP), UP),
        alpha_capt=float(ALPHA_CAPT),
        alpha_ldp=float(ALPHA_LDP),
        joint_coverage_lower=float_bound(decimal_ratio(1 - total_alpha, DOWN), DOWN),
        capt=asdict(cap),
        empirical_v1=asdict(empirical),
        empirical_gap=asdict(empirical.minus(cap.mean)),
        ldp_correction_upper=correction,
        ldp_population_lower_conditional=lower,
        population_margin_lower_conditional=margin,
        conditional_inequality_holds=(margin > 0),
        pooled_tv=dict(
            joint_alphabet=alphabet,
            radius=tv,
            entropy_correction_upper=float_bound(entropy_penalty, UP),
            ldp_lower=tv_lower,
        ),
        independent_days_sensitivity=dict(effective_days=neff, capt_hoeffding_ucb_approx=day_ucb),
        prespecified_family_sensitivity=sensitivity,
    )
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/binary_direct_cp_test_l32"))
    parser.add_argument(
        "--run", type=Path, default=Path("outputs/context_entropy_l32_runs/8ee23db7e6b8e659")
    )
    parser.add_argument("--data", type=Path, default=Path("data/CriteoPrivateAd_release/data"))
    parser.add_argument(
        "--counts", type=Path, help="also verify previously saved (Z,Y) test histograms"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/binary_capt_population_audit"))
    args = parser.parse_args()
    summary, rows = audit(args.results, args.run, args.data, args.counts)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "contexts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        f"N={summary['n']:,}  conditional margin >= {summary['population_margin_lower_conditional']:.9f}"
        f"  population_decision={summary['population_decision']}"
    )


if __name__ == "__main__":
    main()
