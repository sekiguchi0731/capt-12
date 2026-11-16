"""Re-solve CAPT epsilon sweeps with an existing encoder/CTR model/partition.

No MaSS retraining, synthetic distributions, or test-dependent channel selection.
The held-out Criteo features are encoded in batches; only compact records survive.
Run with nohup; use the original results together with the new results for plotting.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import yaml

from capt12.certification.artifact import (
    Certificate,
    hash_array,
    make_certificate,
    verify_certificate,
)
from capt12.certification.robust import (
    repair_robust_channel_uniform,
    solve_robust_block_lp,
    verify_robust_channel,
)
from capt12.comparison.privacy_comparison import compare_saved_privacy
from capt12.confidence.boxes import ConfidenceBox
from capt12.data.loader import iter_parquet_batches
from capt12.experiments.context_stratified import _context_objectives
from capt12.experiments.prior_art_comparison import (
    _attach_attribute_attack,
    _channel_sha256,
    _columns,
    _context_values,
    _empty_row,
    _sample_factorized,
    _token_channels,
)
from capt12.experiments.sampling import select_one_display_per_user_day
from capt12.privacy.adjacency import Group, build_adjacency
from capt12.utils.artifacts import require_clean_worktree, sha256_file
from capt12.utils.progress import ProgressLogger


def compact_split(config, native, split, progress):
    """Preserve source order and global user-day selection across batch boundaries."""
    mapper = joblib.load(native / "models/category_mapper.joblib")
    encoder = joblib.load(native / "models/encoder.joblib")
    keep = list(
        dict.fromkeys(
            [
                config.get("id_col", "id"),
                config.get("user_col", "user_id"),
                config.get("label_col", "is_clicked"),
                "day_int",
                *config["profiles"][0].split("+"),
                *config["context_cols"],
            ]
        )
    )
    frames = []
    total = 0
    for batch in iter_parquet_batches(
        config["data_root"],
        columns=_columns(config),
        days=config["splits"][split],
        batch_size=16384,
    ):
        batch = mapper.transform(batch)
        tokens = encoder.transform(batch)
        compact = batch[keep].copy()
        compact["__token__"] = tokens
        frames.append(compact)
        total += len(batch)
        if len(frames) % 25 == 0:
            progress.emit("encoding", split=split, rows=total)
    if not frames:
        raise ValueError(f"no Criteo rows in {split}")
    return pd.concat(frames, ignore_index=True)


def exact_logloss(channels, data):
    total = 0.0
    for context in np.unique(data.test_context):
        positions = np.flatnonzero(data.test_context == context)
        probabilities = data.service_probabilities[str(context)]
        block_zero = channels.decoder @ -np.log1p(-probabilities)
        block_one = channels.decoder @ -np.log(probabilities)
        rows = channels.input_to_block[str(context)]
        zero, one = rows @ block_zero, rows @ block_one
        z, y = data.test_tokens[positions], data.test_labels[positions]
        total += np.sum(np.where(y == 1, one[z], zero[z]))
    return float(total / len(data.test_tokens))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--epsilons", default="0.1,0.25,0.5,2,4")
    args = parser.parse_args()
    epsilons = [float(v) for v in args.epsilons.split(",")]
    if (
        not epsilons
        or len(set(epsilons)) != len(epsilons)
        or any(not np.isfinite(e) or e <= 0 for e in epsilons)
    ):
        parser.error("epsilons must be distinct, finite and positive")
    source = args.results.resolve()
    native = args.native.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "sweep.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sha = require_clean_worktree()
    identity = {
        "source_results_sha256": sha256_file(source),
        "source_native": str(native),
        "data_root": str(args.data_root.resolve()),
        "epsilons": epsilons,
        "source_git_sha": sha,
    }
    manifest_path = output / "sweep.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["identity"] != identity:
        raise ValueError(
            "existing sweep belongs to different sources/controls; choose a new output directory"
        )
    manifest_path.write_text(json.dumps({"identity": identity, "status": "running"}, indent=2))
    progress = ProgressLogger(output, name="frozen-capt-sweep")
    original_rows = pd.read_csv(source, float_precision="round_trip")
    candidates = original_rows.loc[original_rows.method.eq("capt")]
    if len(candidates) != 1:
        raise ValueError("source results must contain exactly one CAPT row")
    original = candidates.iloc[0].to_dict()
    # Locate the comparison run from the actual saved attack directory, including copied runs.
    from capt12.comparison.privacy_comparison import load_saved_attacks

    models, _ = load_saved_attacks([source])
    attack = next(m for m in models if m.row["method"] == "capt")
    config = yaml.safe_load((attack.path.parent.parent / "resolved_config.yaml").read_text())
    config.update(
        data_root=str(args.data_root.resolve()),
        source_git_sha=sha,
        source_worktree_clean=True,
        require_clean_worktree=True,
    )
    config.setdefault("attribute_attack_prior", 0.5)
    cert_paths = sorted(native.glob("certificate-*-context-*.json"))
    if not cert_paths:
        raise ValueError("source native run has no context certificates")
    originals = []
    for path in cert_paths:
        verified = verify_certificate(path)
        if not verified.valid:
            raise ValueError(f"source certificate failed verification: {path}: {verified}")
        originals.append(Certificate.read(path))
    source_manifest = json.loads((native / "mechanism/context_channel_manifest.json").read_text())
    source_channels = _token_channels(
        np.asarray([c.channel for c in originals]),
        np.asarray([str(c.resolved_config["public_context_value"]) for c in originals]),
        np.asarray(originals[0].assignment),
        np.asarray(originals[0].decoder),
    )
    if _channel_sha256(source_channels) != original["channel_sha256"]:
        raise ValueError("native certificates do not match the CAPT mechanism in source results")
    design_name = originals[0].resolved_config["context_design"]
    if any(c.resolved_config["context_design"] != design_name for c in originals):
        raise ValueError("source has multiple designs")
    config["context_utility_objective"] = originals[0].resolved_config["context_utility_objective"]
    if float(config.get("context_r_pooling_weight", 0)) != 0:
        raise ValueError("this sweep currently requires unpooled context channels")
    support = json.loads((native / "frozen_support.json").read_text())
    frozen = {}
    with np.load(native / "mechanism/frozen_design.npz") as arrays:
        for key in (
            "assignment",
            "frequencies",
            "context_levels",
            "token_context_weights",
            "token_context_label_count",
            "token_context_label_sum",
        ):
            frozen[key] = arrays[key]
    frozen["cartesian_support"] = {tuple(v) for v in support["G_cart"]}
    frozen["design_only_probability"] = {
        tuple(k.split("|")): v for k, v in support["D_design_probability"].items()
    }
    frozen["reference"] = joblib.load(native / "models/reference.joblib")
    assignment, decoder = np.array(originals[0].assignment), np.array(originals[0].decoder)
    design = SimpleNamespace(
        assignment=assignment, decoder=decoder, block_weights=np.ones(len(decoder))
    )
    objectives = _context_objectives(frozen, design, config)
    objectives = {o.context: o for o in objectives}
    progress.emit("source_verified", certificates=len(originals))
    # Encode each split once, independent of epsilon; retain the full test cohort.
    test = compact_split(config, native, "D_test", progress)
    train = select_one_display_per_user_day(
        compact_split(config, native, "D_attack_train", progress),
        user_col=config["user_col"],
        day_col="day_int",
        id_col=config["id_col"],
    )
    positions = select_one_display_per_user_day(
        test.assign(__position=np.arange(len(test))),
        user_col=config["user_col"],
        day_col="day_int",
        id_col=config["id_col"],
    ).__position.to_numpy()

    def context(f):
        return _context_values(f, config["context_cols"]).astype(str).to_numpy()

    def epochs(f):
        return (f.day_int.astype(str) + "|" + f[config["user_col"]].astype(str)).to_numpy()

    data = SimpleNamespace(
        test_frame=test,
        test_tokens=test.__token__.to_numpy(),
        test_context=context(test),
        test_labels=test[config["label_col"]].to_numpy(),
        test_user_days=epochs(test),
        attack_frame=train,
        attack_tokens=train.__token__.to_numpy(),
        attack_context=context(train),
        attack_user_days=epochs(train),
        attack_test_positions=positions,
        expected_attack_groups=tuple(
            Group(config["profiles"][0], tuple(map(str, g[:-1])), str(g[-1]))
            for g in support["G_cart"]
        ),
        service_probabilities={k: o.token_probabilities for k, o in objectives.items()},
    )
    records = []
    for epsilon in epsilons:
        destination = output / f"epsilon-{epsilon:g}"
        destination.mkdir(exist_ok=True)
        row_path = destination / "result.json"
        if row_path.exists():
            records.append(json.loads(row_path.read_text()))
            continue
        for folder in ("models", "mechanism"):
            (destination / folder).mkdir(exist_ok=True)
        for name in ("encoder", "reference", "category_mapper"):
            shutil.copy2(native / f"models/{name}.joblib", destination / f"models/{name}.joblib")
        solutions = []
        manifest = copy.deepcopy(source_manifest)
        entry = manifest["designs"][design_name]
        for stale in ("ldp_contexts", "constant_contexts", "unpooled_contexts"):
            entry.pop(stale, None)
        for cert in originals:
            context_value = str(cert.resolved_config["public_context_value"])
            boxes = {k: ConfidenceBox.from_dict(v) for k, v in cert.boxes.items()}
            groups = [Group(g["profile"], tuple(g["values"]), g["context"]) for g in cert.groups]
            pairs = build_adjacency(
                groups, config["adjacency"], epsilon, config.get("epsilon_by_attr", {})
            )
            objective = objectives[context_value]
            progress.emit("solve", epsilon=epsilon, context=context_value)
            solution, _ = solve_robust_block_lp(
                objective.block_cost,
                objective.block_weights,
                boxes,
                pairs,
                max_iterations=int(config["max_cutting_plane_iterations"]),
                progress=progress,
                heartbeat_seconds=30,
                cut_formulation=config["robust_cut_formulation"],
            )
            repair = repair_robust_channel_uniform(
                solution.channel,
                boxes,
                pairs,
                safety_margin=float(config.get("certificate_repair_margin", 1e-10)),
            )
            entry["contexts"][context_value] = hash_array(repair.channel)
            entry["context_repairs"][context_value].update(mixing_weight=repair.mixing_weight)
            entry["context_repairs"][context_value].pop("pre_repair_objective", None)
            entry["context_repairs"][context_value].pop("released_channel_objective", None)
            solutions.append((cert, boxes, groups, pairs, solution, repair))
        manifest_file = destination / "mechanism/context_channel_manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))
        channels_array = np.stack([s[-1].channel for s in solutions])
        contexts_array = np.array(
            [str(s[0].resolved_config["public_context_value"]) for s in solutions]
        )
        np.savez_compressed(
            destination / f"mechanism/context_channels-{design_name}.npz",
            contexts=contexts_array,
            channels=channels_array,
            assignment=assignment,
            decoder=decoder,
        )
        upper = 0.0
        for index, (old, boxes, groups, pairs, solution, repair) in enumerate(solutions):
            cell_config = dict(old.resolved_config)
            cell_config.update(epsilon=epsilon, source_git_sha=sha, source_worktree_clean=True)
            coverage = copy.deepcopy(old.coverage)
            coverage["context_r_pooling"]["unpooled_released_channel_hash"] = hash_array(
                repair.channel
            )
            coverage["post_solve_repair"].update(
                mixing_weight=repair.mixing_weight,
                pre_repair_infinite_constraint_count=repair.pre_infinite_constraint_count,
                post_repair_zero_denominator_positive_numerator_count=repair.post_zero_denominator_positive_numerator_count,
                released_channel_min_entry=repair.min_entry,
                released_channel_max_row_sum_error=repair.max_row_sum_error,
                conservative_decimal_verification=asdict(repair.conservative_verification),
            )
            certificate = make_certificate(
                config=cell_config,
                channel=repair.channel,
                boxes=boxes,
                adjacency=pairs,
                verification=verify_robust_channel(repair.channel, boxes, pairs),
                solver=solution.solver,
                component_hashes={
                    **{k: old.component_hashes[k] for k in ("encoder", "model", "mapper")},
                    "context_channel_manifest": sha256_file(manifest_file),
                },
                split_identifiers=old.split_identifiers,
                dp_parameters=old.dp_parameters,
                histogram_counts=old.histogram_counts,
                assignment=assignment,
                decoder=decoder,
                coverage=coverage,
                groups=groups,
            )
            cert_path = destination / f"certificate-{design_name}-context-{index:02d}.json"
            certificate.write(cert_path)
            verified = verify_certificate(cert_path)
            if not verified.valid:
                raise RuntimeError(f"sweep certificate failed: {verified}")
            upper = max(upper, repair.conservative_verification.realized_epsilon)
        channels = _token_channels(channels_array, contexts_array, assignment, decoder)
        row = _empty_row()
        for key in ("method", "display_name", "cost", "seed", "K", "L"):
            row[key] = original[key]
        row.update(
            target_epsilon=epsilon,
            certified_upper_epsilon=upper,
            certificate_valid=True,
            certificate_path=str(destination),
            channel_sha256=_channel_sha256(channels),
            empirical_log_loss=exact_logloss(channels, data),
        )
        _attach_attribute_attack(
            row,
            config,
            destination,
            data,
            lambda z, c, s, channels=channels: _sample_factorized(channels, z, c, s),
        )
        row_path.write_text(json.dumps(row, indent=2, default=str))
        records.append(row)
        pd.DataFrame(records).to_csv(output / "results.csv", index=False)
        progress.emit("point_complete", epsilon=epsilon, logloss=row["empirical_log_loss"])
    pd.DataFrame(records).to_csv(output / "results.csv", index=False)
    compare_saved_privacy([source, output / "results.csv"], output / "comparison")
    manifest_path.write_text(json.dumps({"identity": identity, "status": "complete"}, indent=2))
    progress.emit("complete", output=str(output))


if __name__ == "__main__":
    main()
