"""Prospective, frozen binary-CAPT replication on additional Criteo user-days."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from capt12.comparison.optimal_ldp.numeric import UP, decimal_ratio, float_bound, log_bound
from capt12.comparison.optimal_ldp.population import (
    binary_loss_cells,
    frozen_capt_ucb,
    ldp_population_lower,
)
from capt12.comparison.optimal_ldp.theory import Bounds, weighted_bounds
from capt12.comparison.optimal_ldp.utility import grid_bounds
from capt12.experiments.fixed_capt_ctr_comparison import _encode_tokens, _fixed_input_columns
from capt12.experiments.sampling import select_one_display_per_user_day
from capt12.privacy.adjacency import Group
from capt12.utils.artifacts import sha256_file

REPO = "criteo/CriteoPrivateAd"
DAYS = tuple(range(25, 31))
GRID = tuple(i / 64 for i in range(65))


def _save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as tmp:
        try:
            json.dump(value, tmp, indent=2, allow_nan=False)
            tmp.write("\n")
            tmp.flush()
            os.link(tmp.name, path)  # publish complete file, never overwrite
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def _read(path):
    return json.loads(Path(path).read_text())


def _match(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")


def _code_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        digest.update(f"{file.relative_to(root)} {sha256_file(file)}\n".encode())
    return digest.hexdigest()


def _inventory():
    url = f"https://huggingface.co/api/datasets/{REPO}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _select_shards(info, manifest, per_day):
    if manifest["repository"] != REPO or not re.fullmatch("[0-9a-f]{40}", info["sha"]):
        raise ValueError("expected a pinned CriteoPrivateAd release")
    inventory = {f["rfilename"]: f for f in info["siblings"]}
    seen = {f["path"]: f for f in manifest["selected_files"]}
    seen_hashes = set()
    for path, old in seen.items():
        f = inventory[path]
        if f["blobId"] != old["blob_id"] or f["size"] != old["size"] or not f.get("lfs"):
            raise ValueError(f"previous release differs at {path}")
        seen_hashes.add(f["lfs"]["sha256"])
    selected, previous = [], []
    for day in DAYS:
        prefix = f"data/day_int={day}/"
        old = [f for path, f in seen.items() if path.startswith(prefix)]
        if not old or any(f["day"] != day for f in old):
            raise ValueError(f"missing prior exposure inventory on day {day}")
        previous.extend(
            dict(day=day, path=f["path"], sha256=inventory[f["path"]]["lfs"]["sha256"]) for f in old
        )
        candidates = sorted(
            (
                f
                for path, f in inventory.items()
                if path.startswith(prefix)
                and path.endswith(".parquet")
                and f["size"] >= 50_000_000
                and f.get("lfs")
                and path not in seen
                and f["lfs"]["sha256"] not in seen_hashes
            ),
            key=lambda f: (f["size"], f["rfilename"]),
        )
        if not candidates or (per_day is not None and not 1 <= per_day <= len(candidates)):
            raise ValueError(
                f"requested {per_day} fresh shards, found {len(candidates)} on day {day}"
            )
        for f in candidates[:per_day]:
            selected.append(
                dict(day=day, path=f["rfilename"], size=f["size"], sha256=f["lfs"]["sha256"])
            )
            seen_hashes.add(f["lfs"]["sha256"])
    if len({f["sha256"] for f in selected}) != len(selected):
        raise ValueError("duplicate holdout content")
    return previous, selected


def _sources(results, run):
    models = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in (
            ("mapper", run / "models/category_mapper.joblib"),
            ("encoder", run / "models/encoder.joblib"),
        )
    }
    contexts = {}
    config = None
    for idx in range(31):  # Published L32 cohort, fixed before opening the replication set
        source = results / f"{idx:02d}.json"
        r = _read(source)
        path = run / Path(r["certificate"]).name
        _match(path, r["certificate_sha256"])
        c = _read(path)
        cfg = c["resolved_config"]
        b = str(r["context"])
        spec = r["binary_channel"]
        beta = r["calibration"]["beta_R"]
        budgets = {
            key: r["comparisons"][key]["lambda"]
            for key in ("equal_epsilon", "tv_calibrated", "uniform_calibrated")
        }
        # In this release both calibrations have worst-case factor 1. Thus lambda
        # <= beta guarantees the attribute budget for ANY new pair of group laws.
        if not (
            b == str(cfg["public_context_value"])
            and b not in contexts
            and r["capt_mechanism"] == "binary-direct-cp"
            and int(path.stem.rsplit("-", 1)[1]) == idx
            and tuple(r["splits"]["D_test"]) == DAYS
            and r["utility_sample_sizes"]["train"]
            and r["utility_sample_sizes"]["validation"]
            and r["profile"] == cfg["profiles"][0]
            and r["context_column"] == cfg["context_cols"][0]
            and r["K"] == cfg["K"] == len(c["assignment"])
            and r["L"] == cfg["L"] == 32
            and 0 <= budgets["tv_calibrated"] == budgets["uniform_calibrated"] == beta <= 1
            and all(m["sha256"] == c["component_hashes"][key] for key, m in models.items())
        ):
            raise ValueError(f"inconsistent frozen L32 record {idx}")
        common = {
            key: cfg[key]
            for key in (
                "K",
                "profiles",
                "context_cols",
                "id_col",
                "user_col",
                "label_col",
                "phi_source_cols",
            )
        }
        if config is not None and common != config:
            raise ValueError("contexts use different input mappings")
        config = common
        contexts[b] = dict(
            index=idx,
            result_sha256=sha256_file(source),
            certificate_sha256=r["certificate_sha256"],
            beta_R=beta,
            budgets=budgets,
            high_tokens=np.flatnonzero(np.isin(c["assignment"], spec["high_blocks"])).tolist(),
            release_one=spec["probability_of_one_low_high"],
            ctr=spec["fitted_output_ctr"],
            edges=[[e["left"], e["right"]] for e in c["adjacency"]],
            groups=list(c["boxes"]),
        )
    return config, models, contexts


def prepare(
    results: Path, run: Path, source_manifest: Path, plan: Path, shards_per_day: int | None
):
    if plan.exists():
        raise FileExistsError(plan)
    info = _inventory()
    previous, selected = _select_shards(info, _read(source_manifest), shards_per_day)
    root = source_manifest.resolve().parent
    for day in DAYS:
        known = {root / f["path"] for f in previous if f["day"] == day}
        if set((root / f"data/day_int={day}").rglob("*.parquet")) != known:
            raise ValueError(
                f"historical data directory differs from exposure manifest on day {day}"
            )
    for f in previous:
        _match(root / f["path"], f["sha256"])
    config, models, contexts = _sources(results, run)
    _save_new(
        plan,
        dict(
            version=1,
            repository=REPO,
            revision=info["sha"],
            code_sha256=_code_hash(),
            original_root=str(root),
            manifest_sha256=sha256_file(source_manifest),
            previous=previous,
            selected=selected,
            config=config,
            models=models,
            contexts=contexts,
            grid=list(GRID),
            primary="uniform_calibrated",
        ),
    )
    return sha256_file(plan)


def _winners(previous, fresh, day, user, event):
    def read_ids(path):
        ids = pq.ParquetFile(path).read(columns=[user, event]).to_pandas()
        if ids.isna().any().any():
            raise ValueError("missing user or display identifier")
        return ids.astype(str)

    old = pd.concat([read_ids(path) for path in previous], ignore_index=True)
    new = pd.concat([read_ids(path) for path in fresh], ignore_index=True)
    if new[event].duplicated().any() or new[event].isin(old[event]).any():
        raise ValueError(f"duplicate display IDs on day {day}")
    overlap = new[user].isin(old[user])
    keep = select_one_display_per_user_day(
        new.loc[~overlap].assign(day_int=day), user_col=user, day_col="day_int", id_col=event
    )
    return set(keep[event]), dict(
        day=day,
        raw=len(new),
        overlapping_displays=int(overlap.sum()),
        excluded_seen_user_days=new.loc[overlap, user].nunique(),
        new_user_days=len(keep),
    )


def _accumulate(frame, cfg, contexts, encoder, counts, groups):
    context = cfg["context_cols"][0]
    profile = cfg["profiles"][0]
    attrs = profile.split("+")
    eligible = 0
    for b, part in frame.groupby(context, sort=False):
        b = str(b)
        if b not in contexts:
            continue
        z = _encode_tokens(encoder, part, cfg["K"])
        y = part[cfg["label_col"]]
        if not y.isin([0, 1]).all():
            raise ValueError("nonbinary or missing click label")
        np.add.at(counts[b], (z, y.to_numpy(dtype=int)), 1)
        high = np.isin(z, contexts[b]["high_tokens"])
        for values, positions in part.groupby(attrs, sort=False, dropna=False).indices.items():
            group = Group(profile, values if isinstance(values, tuple) else (values,), b).key()
            if group not in contexts[b]["groups"]:
                raise ValueError(f"unknown protected group {group}")
            groups[b][group] += np.array([len(positions), high[positions].sum()], dtype=np.int64)
        eligible += len(part)
    return eligible


def _privacy(spec, groups):
    # A group absent from the finite logs may have ANY high-token probability.
    low, high = map(Fraction, spec["release_one"])
    ranges = {}
    for g, (n, k) in groups.items():
        if n:
            p = low + (high - low) * Fraction(int(k), int(n))
            ranges[g] = (p, p)
        else:
            ranges[g] = (min(low, high), max(low, high))
    ratio = Fraction(1)
    for left, right in spec["edges"]:
        a, b = ranges[left]
        c, d = ranges[right]
        for num, den in ((b, c), (1 - a, 1 - d)):
            if num and not den:
                return None
            if den:
                ratio = max(ratio, num / den)
    return float_bound(log_bound(decimal_ratio(ratio, UP), UP), UP)


def _summarize(plan, counts, groups):
    rows, all_cells, bounds, weights = [], [], [], []
    n = sum(int(v.sum()) for v in counts.values())
    if not n:
        raise ValueError("no eligible replication user-days")
    privacy_ok = True
    for b, spec in plan["contexts"].items():
        hist = counts[b]
        nb = int(hist.sum())
        high = np.isin(np.arange(len(hist)), spec["high_tokens"])
        cells = binary_loss_cells(hist, high, spec["release_one"], spec["ctr"])
        all_cells.extend(cells)
        row = dict(
            context=b,
            index=spec["index"],
            n=nb,
            weight=nb / n,
            beta_R=spec["beta_R"],
            budgets=spec["budgets"],
            absent_groups=sum(int(v[0] == 0) for v in groups[b].values()),
        )
        if nb:
            beta = _privacy(spec, groups[b])
            row.update(
                empirical_attribute_beta_upper=beta,
                attribute_budget_verified=beta is not None and beta <= spec["beta_R"],
            )
            privacy_ok &= row["attribute_budget_verified"]
            vz = grid_bounds(hist, spec["budgets"][plan["primary"]], plan["grid"])
            interval = Bounds(vz.lower, vz.upper)
            bounds.append(interval)
            weights.append(Fraction(nb, n))
            loss = weighted_bounds(
                [c.loss for c in cells if c.count],
                [Fraction(c.count, nb) for c in cells if c.count],
            )
            row.update(
                capt_loss=asdict(loss),
                optimal_ldp=asdict(interval),
                ldp_minus_capt=asdict(interval.minus(loss)),
            )
        rows.append(row)
    vz = weighted_bounds(bounds, weights)
    loss = weighted_bounds(
        [c.loss for c in all_cells if c.count], [Fraction(c.count, n) for c in all_cells if c.count]
    )
    gap = vz.minus(loss)
    sensitivity = None
    if n >= 2:
        delta = Fraction(3, 2000)
        cap = frozen_capt_ucb(all_cells, delta)
        lower, _ = ldp_population_lower(vz.lower, n, delta)
        sensitivity = dict(
            delta_per_side=float(delta),
            capt_upper=cap.upper,
            ldp_lower=lower,
            positive=cap.upper < lower,
        )
    utility = "win" if gap.lower > 0 else "loss" if gap.upper < 0 else "inconclusive"
    return dict(
        contexts=rows,
        eligible_user_days=n,
        capt_loss=asdict(loss),
        optimal_ldp=asdict(vz),
        ldp_minus_capt=asdict(gap),
        empirical_utility=utility,
        empirical_comparison=utility if privacy_ok else "attribute_budget_not_verified",
        empirical_attribute_verified=privacy_ok,
        population_status="assumptions_not_enabled",
        independent_unit_sensitivity=sensitivity,
    )


def _fetch(revision, item, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / item["sha256"]
    if not path.is_file():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{item['path']}"
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
            partial = Path(tmp.name)
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, tmp, length=1024 * 1024)
                tmp.flush()
                _match(partial, item["sha256"])
                os.replace(partial, path)
            finally:
                partial.unlink(missing_ok=True)
    _match(path, item["sha256"])
    return path


def evaluate(plan_path: Path, expected_sha256: str, output: Path, cache: Path):
    if output.exists():
        raise FileExistsError(output)
    _match(plan_path, expected_sha256)
    p = _read(plan_path)
    if p["version"] != 1 or p["code_sha256"] != _code_hash():
        raise ValueError("replication code changed since registration")
    for model in p["models"].values():
        _match(model["path"], model["sha256"])
    mapper, encoder = (joblib.load(p["models"][key]["path"]) for key in ("mapper", "encoder"))
    cfg, contexts = p["config"], p["contexts"]
    counts = {b: np.zeros((cfg["K"], 2), dtype=np.int64) for b in contexts}
    groups = {b: {g: np.zeros(2, dtype=np.int64) for g in s["groups"]} for b, s in contexts.items()}
    days = []
    for day in DAYS:
        old, new = [], []
        for f in p["previous"]:
            if f["day"] == day:
                path = Path(p["original_root"]) / f["path"]
                _match(path, f["sha256"])
                old.append(path)
        for f in p["selected"]:
            if f["day"] == day:
                new.append(_fetch(p["revision"], f, cache))
        event, user = cfg["id_col"], cfg["user_col"]
        winners, audit = _winners(old, new, day, user, event)
        eligible = 0
        columns = list(
            dict.fromkeys([*_fixed_input_columns(cfg), *mapper.mappings, cfg["label_col"]])
        )
        for path in new:
            for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=16384):
                frame = batch.to_pandas()
                ids = frame[event].astype(str)
                chosen = ids.map(winners.__contains__)
                if chosen.any():
                    winners.difference_update(ids[chosen])
                    selected = mapper.transform(frame.loc[chosen].assign(day_int=day))
                    eligible += _accumulate(selected, cfg, contexts, encoder, counts, groups)
        if winners:
            raise ValueError("selected displays missing from second pass")
        days.append(
            dict(**audit, eligible=eligible, outside_cohort=audit["new_user_days"] - eligible)
        )
        print(f"Evaluated replication day {day}: {eligible} eligible user-days", flush=True)
    report = _summarize(p, counts, groups)
    _save_new(output, dict(plan_sha256=expected_sha256, days=days, **report))
