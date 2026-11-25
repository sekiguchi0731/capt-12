"""Reproduce the frozen Criteo comparison and separate per-stage empirical losses.

From the repository root: uv run python tools/diagnose_criteo_replication.py
Use --baseline-only to analyze archived original test counts without replication shards.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import zipfile
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

from capt12.comparison.optimal_ldp.retraining import (
    FixedBlocks,
    Predictor,
    fixed_predictor_loss_bounds,
)
from capt12.comparison.optimal_ldp.theory import Bounds, weighted_bounds
from capt12.comparison.optimal_ldp.utility import (
    bayes_loss_bounds,
    grid_bounds,
    joint_probabilities,
)
from capt12.experiments import criteo_replication as rep
from capt12.utils.artifacts import sha256_file

DATA = Path("data/CriteoPrivateAd_release")
RUN = Path("outputs/context_entropy_l32_runs/8ee23db7e6b8e659")
BASE = Path("outputs/criteo_replication")
OLD_COUNTS = Path("outputs/binary_direct_cp_test_analysis/test_counts.npz")
COMPONENTS = (
    "token_bayes",
    "block_loss",
    "threshold_loss",
    "randomization_loss",
    "predictor_excess",
)


def read(path):
    return json.loads(path.read_text())


def manifests(plan, cache):
    """Scan workspace and embedded review manifests, binding named raw files by basename."""
    candidates = {f["path"].split("/")[-1]: f for f in plan["selected"]}
    paths = {p.name: p for p in Path("data").glob("*/data/day_int=*/*.parquet")}
    inventory, hits = [], {}
    pattern = re.compile(rb"part-[\w.\-]+\.parquet")

    days_by_name = {}

    def inspect(name, raw):
        for day, shard in re.findall(rb"day_int=(\d+)/([\w.\-]+\.parquet)", raw):
            days_by_name[shard.decode()] = int(day)
        names = sorted({s.decode() for s in pattern.findall(raw)})
        inventory.append(dict(manifest=name, sha256=hashlib.sha256(raw).hexdigest(), shards=names))
        for shard in names:
            hits.setdefault(shard, []).append(name)

    for folder in (
        Path("data"),
        Path("outputs"),
        Path("reports"),
        Path("results"),
        Path("configs"),
        Path("docs"),
        Path("tools"),
        Path("src"),
        Path("tests"),
    ):
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if "manifest" in path.name.lower() and path.suffix.lower() in {
                ".json",
                ".txt",
                ".yaml",
                ".yml",
                ".csv",
            }:
                inspect(str(path), path.read_bytes())
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as z:
                    for member in z.namelist():
                        if "manifest" in Path(member).name.lower() and not member.endswith("/"):
                            inspect(f"{path}!{member}", z.read(member))
    objects = subprocess.check_output(["git", "rev-list", "--objects", "--all"], text=True)
    for record in objects.splitlines():
        oid, _, name = record.partition(" ")
        if "manifest" in Path(name).name.lower() and Path(name).suffix.lower() in {
            ".json",
            ".yaml",
            ".yml",
            ".csv",
            ".txt",
        }:
            raw = subprocess.check_output(["git", "cat-file", "blob", oid])
            inspect(f"git:{oid}:{name}", raw)
    # A differently named local file may refer to the same bytes. Bind each
    # historical shard to its day and include every one on the comparison dates.
    historical = {}
    known = {f["path"].split("/")[-1]: f for f in plan["previous"]}
    # This specific prior exposure was identified by content on the analysis machine.
    # A remote checkout may have retained only the replication cache.
    smoke = "part-00016-b8d2009e-5149-4859-8b07-5d93b432e66e-c000.gz.parquet"
    smoke_hash = "e2bd46685b0f4c5071bdb82e8df1e68279592c91932bbab68546cf268174cf8c"
    if candidates[smoke]["sha256"] != smoke_hash:
        raise ValueError("registered day-28 smoke content mismatch")
    if smoke not in hits:
        hits[smoke] = ["verified original smoke_source_manifest.json on analysis machine"]
    for name in hits:
        if name in days_by_name and days_by_name[name] not in rep.DAYS:
            continue
        local = paths.get(name)
        if not local:
            if name not in known and name not in candidates:
                raise ValueError(f"unresolved historical shard: {name}")
            f = (known | candidates)[name]
            local = DATA / f["path"] if name in known else cache / f["sha256"]
            day = f["day"]
        else:
            day = int(local.parent.name.split("=")[1])
        if day in rep.DAYS:
            digest = sha256_file(local)
            if name in known or name in candidates:
                assert digest == (known | candidates)[name]["sha256"], name
            historical[digest] = dict(day=day, path=str(local), name=name, manifests=hits[name])
    for f in plan["previous"]:
        assert f["sha256"] in historical
    return inventory, historical


def entropy(counts):
    return Bounds(*bayes_loss_bounds(joint_probabilities(counts))).midpoint


def profile(hist, spec, assignment, grid, ldp=None):
    n = int(hist.sum())
    high = np.isin(np.arange(len(hist)), spec["high_tokens"])
    block = np.zeros((int(assignment.max()) + 1, 2), dtype=np.int64)
    np.add.at(block, assignment, hist)
    bins = np.stack([hist[~high].sum(axis=0), hist[high].sum(axis=0)])
    channel = FixedBlocks(
        high.astype(int), tuple((1 - q, q) for q in map(Fraction, spec["release_one"]))
    )
    joint = channel.exact_output_joint(hist)
    learned = Bounds(
        *fixed_predictor_loss_bounds(Predictor(channel, np.array(spec["ctr"]), 0, 0), joint)
    )
    bayes = Bounds(*bayes_loss_bounds(joint))
    hz, hc, ht = map(entropy, (hist, block, bins))
    hy = entropy(hist.sum(axis=0, keepdims=True))
    if ldp is None:
        v = grid_bounds(hist, spec["budgets"]["uniform_calibrated"], grid)
        ldp = Bounds(v.lower, v.upper)
    row = dict(
        n=n,
        ctr=hist[:, 1].sum() / n,
        high_probability=bins[1].sum() / n,
        token_bayes=hz,
        block_loss=hc - hz,
        threshold_loss=ht - hc,
        randomization_loss=bayes.midpoint - ht,
        predictor_excess=learned.midpoint - bayes.midpoint,
        capt_bayes=bayes.midpoint,
        fitted=learned.midpoint,
        entropy_y=hy,
        token_mi=hy - hz,
        release_mi=hy - bayes.midpoint,
        retention_sdpi_margin=(hy - bayes.midpoint)
        - math.tanh(spec["budgets"]["uniform_calibrated"] / 2) ** 2 * (hy - hz),
    )
    for bit, (by, output) in enumerate(zip(bins, joint, strict=True)):
        row[f"threshold_ctr_{bit}"] = float(by[1] / by.sum()) if by.sum() else None
        row[f"output_ctr_{bit}"] = float(output[1] / sum(output)) if sum(output) else None
        row[f"frozen_ctr_{bit}"] = spec["ctr"][bit]
    for name, x in [
        ("v", ldp),
        ("ldp_information_loss", ldp.minus(Bounds(hz, hz))),
        ("bayes_gap", ldp.minus(bayes)),
        ("fitted_gap", ldp.minus(learned)),
    ]:
        row[name + "_lower"] = x.lower
        row[name + "_upper"] = x.upper
    row["residual"] = learned.midpoint - math.fsum(row[k] for k in COMPONENTS)
    if abs(row["residual"]) > 1e-12 or min(row[k] for k in COMPONENTS) < -1e-12:
        raise ValueError("loss decomposition did not telescope")
    return row


def write_table(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def summarize(name, hist, plan, assigns, out, checks=None, grid_saved=None):
    rows = []
    for b, s in plan["contexts"].items():
        data = hist[b]
        assert data.shape == (plan["config"]["K"], 2) and data.sum() > 0
        r = dict(
            context=b,
            index=s["index"],
            **profile(
                data, s, assigns[b], plan["grid"], Bounds(**grid_saved[b]) if grid_saved else None
            ),
        )
        if checks:
            target = checks[b]
            assert r["n"] == target["n"] and abs(r["fitted"] - target["fitted"]) < 1e-12, (
                b,
                r["n"],
                target,
            )
        rows.append(r)
    n = sum(x["n"] for x in rows)
    for r in rows:
        r["weight"] = r["n"] / n
    agg = {
        k: math.fsum(r["weight"] * r[k] for r in rows)
        for k in rows[0]
        if k not in {"context", "index", "n", "weight"}
        and not k.startswith(("threshold_ctr_", "output_ctr_", "frozen_ctr_"))
    }
    weights = [Fraction(r["n"], n) for r in rows]
    for key in ["v", "bayes_gap", "fitted_gap"]:
        interval = weighted_bounds(
            [Bounds(r[key + "_lower"], r[key + "_upper"]) for r in rows], weights
        )
        agg[key + "_lower"], agg[key + "_upper"] = interval.lower, interval.upper
    agg["n"] = n
    write_table(out / (name + "_contexts.csv"), rows)
    (out / (name + "_aggregate.json")).write_text(json.dumps(agg, indent=2))
    return rows, agg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Fetch missing registered replication shards into --cache.",
    )
    parser.add_argument("--cache", type=Path, default=Path("data/criteo_replication_cache"))
    parser.add_argument("--output", type=Path, default=BASE / "diagnosis")
    args = parser.parse_args()
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    p = read(BASE / "plan.json")
    report = read(BASE / "result.json")
    rep._match(BASE / "plan.json", report["plan_sha256"])
    assert rep._code_hash() == p["code_sha256"] and tuple(p["grid"]) == rep.GRID
    if args.download_missing and not args.baseline_only:
        for item in p["selected"]:
            rep._fetch(p["revision"], item, args.cache)
    inventory, history = manifests(p, args.cache)
    overlap = [f["path"] for f in p["selected"] if f["sha256"] in history]
    (out / "exposures.json").write_text(
        json.dumps(dict(inventory=inventory, historical=history, overlap=overlap), indent=2)
    )
    assigns, checks = {}, {}
    for b, s in p["contexts"].items():
        rpath = Path("outputs/binary_direct_cp_test_l32") / f"{s['index']:02}.json"
        rep._match(rpath, s["result_sha256"])
        r = read(rpath)
        path = RUN / Path(r["certificate"]).name
        rep._match(path, s["certificate_sha256"])
        assigns[b] = np.asarray(read(path)["assignment"], int)
        assert (
            np.flatnonzero(np.isin(assigns[b], r["binary_channel"]["high_blocks"])).tolist()
            == s["high_tokens"]
        )
        checks[b] = dict(n=r["utility_sample_sizes"]["test"], fitted=r["capt_test_log_loss"])
    if OLD_COUNTS.is_file():
        with np.load(OLD_COUNTS, allow_pickle=False) as saved_counts:
            hist = {b: saved_counts[b] for b in p["contexts"]}
    else:
        hist, groups = blank(p)
        models = load_models(p)
        for day in rep.DAYS:
            local = [DATA / f["path"] for f in p["previous"] if f["day"] == day]
            sample_day(local, [], set(), day, p, models, hist, groups)
    summarize("original", hist, p, assigns, out, checks)
    # Keep the published original as the reference and provide a symmetric
    # sensitivity analysis on original test days touched by the other manifests.
    old_sha = {f["sha256"] for f in p["previous"]}
    removed, removed_groups = blank(p)
    audit_days = []
    for day in rep.DAYS:
        users = extra_users(history, old_sha, day, p["config"]["user_col"])
        if not users:
            continue
        local = [DATA / f["path"] for f in p["previous"] if f["day"] == day]
        total, groups = blank(p)
        audit_days.append(
            sample_day(
                local, [], users, day, p, load_models(p), total, groups, removed, removed_groups
            )
        )
    adjusted = {b: hist[b] - removed[b] for b in hist}
    assert all((table >= 0).all() for table in adjusted.values())
    summarize("original_smoke_excluded", adjusted, p, assigns, out)
    (out / "original_smoke_exclusions.json").write_text(json.dumps(audit_days, indent=2))
    if not args.baseline_only:
        replay(p, history, assigns, report, args.cache, out)


def blank(plan):
    return (
        {b: np.zeros((plan["config"]["K"], 2), dtype="int64") for b in plan["contexts"]},
        {
            b: {g: np.zeros(2, dtype="int64") for g in s["groups"]}
            for b, s in plan["contexts"].items()
        },
    )


def load_models(plan):
    objects = []
    for name, filename in [("mapper", "category_mapper.joblib"), ("encoder", "encoder.joblib")]:
        path = RUN / "models" / filename
        rep._match(path, plan["models"][name]["sha256"])
        objects.append(joblib.load(path))
    return objects


def sample_day(
    fresh, old, exposed_users, day, plan, models, full, groups, removed=None, excluded_groups=None
):
    """Selecting a display and then dropping whole user-days commutes with dropping them first."""
    cfg = plan["config"]
    event = cfg["id_col"]
    user = cfg["user_col"]
    if old:
        winners, audit = rep._winners(old, fresh, day, user, event)
    else:
        import pandas as pd

        ids = pd.concat(
            [pq.ParquetFile(f).read(columns=[event, user]).to_pandas() for f in fresh],
            ignore_index=True,
        )
        assert not ids.isna().any().any() and ids[event].is_unique
        selected = rep.select_one_display_per_user_day(
            ids.astype(str).assign(day_int=day), user_col=user, day_col="day_int", id_col=event
        )
        winners = set(selected[event])
        audit = {"day": day, "new_user_days": len(winners)}
    mapper, encoder = models
    columns = list(
        dict.fromkeys([*rep._fixed_input_columns(cfg), *mapper.mappings, cfg["label_col"]])
    )
    excluded = eligible = excluded_eligible = 0
    for path in fresh:
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=16384):
            frame = batch.to_pandas()
            ids = frame[event].astype(str)
            keep = ids.map(winners.__contains__)
            if not keep.any():
                continue
            winners.difference_update(ids[keep])
            part = mapper.transform(frame.loc[keep].assign(day_int=day))
            eligible += rep._accumulate(part, cfg, plan["contexts"], encoder, full, groups)
            if exposed_users:
                flagged = part[user].astype(str).isin(exposed_users)
                excluded += int(flagged.sum())
                if flagged.any():
                    excluded_eligible += rep._accumulate(
                        part[flagged], cfg, plan["contexts"], encoder, removed, excluded_groups
                    )
    assert not winners
    return dict(
        **audit,
        eligible=eligible,
        extra_excluded=excluded,
        extra_excluded_eligible=excluded_eligible,
    )


def extra_users(history, old_sha, day, user):
    users = set()
    for digest, entry in history.items():
        if entry["day"] == day and digest not in old_sha:
            column = pq.ParquetFile(entry["path"]).read(columns=[user]).column(0)
            assert not column.null_count
            users.update(map(str, column.unique().to_pylist()))
    return users


def replay(p, history, assigns, saved, cache, out):
    models = load_models(p)
    full, groups = blank(p)
    removed, removed_groups = blank(p)
    prev_sha = {f["sha256"] for f in p["previous"]}
    days = []
    for day in rep.DAYS:
        fresh = []
        for f in p["selected"]:
            if f["day"] != day:
                continue
            path = (
                Path(history[f["sha256"]]["path"])
                if f["sha256"] in history
                else cache / f["sha256"]
            )
            rep._match(path, f["sha256"])
            fresh.append(path)
        old = [Path(history[f["sha256"]]["path"]) for f in p["previous"] if f["day"] == day]
        users = extra_users(history, prev_sha, day, p["config"]["user_col"])
        dayrow = sample_day(
            fresh, old, users, day, p, models, full, groups, removed, removed_groups
        )
        expected = next(d for d in saved["days"] if d["day"] == day)
        assert all(
            dayrow[k] == expected[k]
            for k in (
                "raw",
                "eligible",
                "new_user_days",
                "excluded_seen_user_days",
                "overlapping_displays",
            )
        ), dayrow
        days.append(dayrow)
        print(dayrow, flush=True)
    clean = {b: full[b] - removed[b] for b in full}
    clean_groups = {
        b: {g: v - removed_groups[b][g] for g, v in values.items()} for b, values in groups.items()
    }
    for b, hist in clean.items():
        assert hist.min() >= 0 and sum(int(x[0]) for x in clean_groups[b].values()) == int(
            hist.sum()
        )
        assert int(hist.sum()) == int(full[b].sum() - removed[b].sum())
    assert sum(int(v.sum()) for v in removed.values()) == sum(
        d["extra_excluded_eligible"] for d in days
    )
    checks = {
        x["context"]: dict(n=x["n"], fitted=x["capt_loss"]["lower"]) for x in saved["contexts"]
    }
    for name, tables in [("replication_full", full), ("replication_clean", clean)]:
        np.savez_compressed(out / (name + "_counts.npz"), **tables)
    # All three comparisons use the original per-context lambda and 65-point bounds.
    original_replay = rep._summarize(p, full, groups)
    for key in (
        "contexts",
        "optimal_ldp",
        "capt_loss",
        "ldp_minus_capt",
        "empirical_attribute_verified",
    ):
        assert original_replay[key] == saved[key], key
    corrected = rep._summarize(p, clean, clean_groups)
    (out / "corrected_result.json").write_text(
        json.dumps(dict(plan_sha256=saved["plan_sha256"], days=days, **corrected), indent=2)
    )
    summarize(
        "replication_full",
        full,
        p,
        assigns,
        out,
        checks,
        {x["context"]: x["optimal_ldp"] for x in original_replay["contexts"]},
    )
    summarize(
        "replication_clean",
        clean,
        p,
        assigns,
        out,
        grid_saved={x["context"]: x["optimal_ldp"] for x in corrected["contexts"]},
    )
    contrasts(out, "original")
    contrasts(out, "original_smoke_excluded")


def contrasts(out, reference):
    def rows_for(name):
        with (out / (name + "_contexts.csv")).open() as handle:
            return {
                x["context"]: {k: float(v) for k, v in x.items() if k != "context" and v != ""}
                for x in csv.DictReader(handle)
            }

    original = rows_for(reference)
    for period in ("replication_full", "replication_clean"):
        current = rows_for(period)
        rows = []
        for b, x in current.items():
            old = original[b]

            def mid(row, key):
                return (row[key + "_lower"] + row[key + "_upper"]) / 2

            # Reference distribution convention: change weights first, then within-context losses.
            row = {
                "context": b,
                "index": int(x["index"]),
                "weight": x["weight"],
                "mix": (x["weight"] - old["weight"]) * mid(old, "fitted_gap"),
                "delta_ldp_penalty": mid(x, "ldp_information_loss")
                - mid(old, "ldp_information_loss"),
            }
            for key in COMPONENTS[1:]:
                row["minus_delta_" + key] = old[key] - x[key]
            for key in (
                "ctr",
                "high_probability",
                "threshold_ctr_0",
                "threshold_ctr_1",
                "output_ctr_0",
                "output_ctr_1",
                "token_mi",
                "release_mi",
                "retention_sdpi_margin",
            ):
                row["delta_" + key] = x[key] - old[key] if key in x and key in old else None
            for objective in ("fitted_gap", "bayes_gap"):
                for side in ("lower", "upper"):
                    row[objective + "_" + side] = x[objective + "_" + side]
                row[objective + "_sign"] = (
                    "win"
                    if x[objective + "_lower"] > 0
                    else "loss"
                    if x[objective + "_upper"] < 0
                    else "inconclusive"
                )
            row["delta_fitted_gap"] = mid(x, "fitted_gap") - mid(old, "fitted_gap")
            rows.append(row)
        terms = ["delta_ldp_penalty", *["minus_delta_" + k for k in COMPONENTS[1:]]]
        for row in rows:
            assert abs(sum(row[k] for k in terms) - row["delta_fitted_gap"]) < 1e-12
        aggregate = {k: math.fsum(row["weight"] * row[k] for row in rows) for k in terms}
        aggregate["mix"] = math.fsum(row["mix"] for row in rows)
        aggregate["total_gap_change"] = sum(aggregate.values())
        write_table(out / (period + "_vs_" + reference + "_change.csv"), rows)
        (out / (period + "_vs_" + reference + "_change.json")).write_text(
            json.dumps(aggregate, indent=2)
        )


if __name__ == "__main__":
    main()
