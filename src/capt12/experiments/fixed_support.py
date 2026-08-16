from __future__ import annotations

import gc
import json
import math
import resource
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from capt12.certification.artifact import hash_json, make_certificate, verify_certificate
from capt12.certification.robust import solve_robust_block_lp, verify_robust_channel
from capt12.confidence.boxes import CONFIDENCE_REGISTRY, ConfidenceBox
from capt12.config import validate_config
from capt12.data.loader import (
    assert_disjoint_splits,
    assert_no_row_overlap,
    load_parquet_sample,
)
from capt12.data.preprocessing import (
    FrozenCategoryMapper,
    build_group_histograms,
    expected_tuple_grid,
)
from capt12.decoders.registry import build_decoder
from capt12.distortions.registry import block_cost_matrix, token_cost_matrix
from capt12.encoders.base import make_encoder
from capt12.experiments.sampling import (
    FULL_LABEL,
    select_one_display_per_user_day,
    stable_nested_positions,
)
from capt12.mechanisms.baselines import common_cover
from capt12.models.reference import named_reference_model
from capt12.partitions.registry import build_nested_partitions, build_partition
from capt12.pipeline import record_source_provenance
from capt12.privacy.adjacency import Group, build_adjacency, disconnected_hybrid_components
from capt12.utils.artifacts import environment, finish_run, prepare_run, sha256_file


def _context_values(frame: pd.DataFrame, contexts: Sequence[str]) -> pd.Series:
    if contexts:
        return frame[list(contexts)].astype(str).agg("|".join, axis=1)
    return pd.Series("all", index=frame.index)


def _tuple_counter(
    frame: pd.DataFrame,
    *,
    profile: str,
    contexts: Sequence[str],
) -> Counter[tuple[str, ...]]:
    attributes = profile.split("+") if profile else []
    values = frame[attributes].astype(str).copy()
    values["__context__"] = _context_values(frame, contexts)
    grouped = values.value_counts(sort=False, dropna=False)
    return Counter(
        {
            tuple(map(str, key if isinstance(key, tuple) else (key,))): int(count)
            for key, count in grouped.items()
        }
    )


def _groups_from_tuples(profile: str, tuples: Iterable[tuple[str, ...]]) -> list[Group]:
    attribute_count = len(profile.split("+")) if profile else 0
    return [Group(profile, tuple(value[:attribute_count]), value[-1]) for value in sorted(tuples)]


def _expected_frame_from_cartesian(
    tuples: Iterable[tuple[str, ...]],
    *,
    profile: str,
    context: str,
) -> pd.DataFrame:
    attributes = profile.split("+") if profile else []
    columns = [*attributes, context]
    return pd.DataFrame(list(sorted(tuples)), columns=columns)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value if value > 10_000_000 else value * 1024


def _fixed_design(
    config: dict[str, Any],
    path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    profile = config["profiles"][0]
    profile_attributes = profile.split("+")
    contexts = list(config.get("context_cols", []))
    sensitive = list(config.get("sensitive_cols", []))
    source = list(config.get("phi_source_cols", []))
    label = config.get("label_col", "is_clicked")
    id_col = config.get("id_col", "id")
    user_col = config.get("user_col", "user_id")
    columns = list(
        dict.fromkeys([id_col, user_col, label, *profile_attributes, *contexts, *source])
    )

    model_frame = load_parquet_sample(
        data_root=config["data_root"],
        columns=columns,
        days=config["splits"]["D_model"],
    )
    mapper = FrozenCategoryMapper(int(config.get("max_context_cardinality", 32))).fit(
        model_frame, [*profile_attributes, *contexts], split_id="D_model"
    )
    model_frame = mapper.transform(model_frame)
    encoder = make_encoder(
        config.get("phi", "hash"),
        int(config.get("K", 64)),
        seed=int(config.get("frozen_design_seed", config.get("seed", 0))),
        sensitive_policy=config.get("phi_sensitive_policy", "exclude"),
        column=config.get("precomputed_token_col", "token"),
    )
    encoder.fit(
        model_frame,
        source,
        split_id="D_model",
        sensitive_columns=sensitive,
        labels=model_frame[label].to_numpy(),
    )
    model_tokens = encoder.transform(model_frame)
    model_frame["__token__"] = model_tokens
    reference = named_reference_model(
        config.get("fixed_model", "ctr_model"),
        eta=float(config.get("distortion_clip", 1e-6)),
    ).fit(
        model_frame,
        label,
        ["__token__", *contexts],
        split_id="D_model",
        sensitive_columns=sensitive,
    )
    model_support = _tuple_counter(model_frame, profile=profile, contexts=contexts)
    domains = {
        column: set(model_frame[column].astype(str).unique().tolist())
        for column in [*profile.split("+"), *contexts]
    }

    joblib.dump(reference, path / "models" / "reference.joblib")
    joblib.dump(encoder, path / "models" / "encoder.joblib")
    joblib.dump(mapper, path / "models" / "category_mapper.joblib")
    model_rows = len(model_frame)
    del model_frame, model_tokens
    gc.collect()

    design_frame = load_parquet_sample(
        data_root=config["data_root"],
        columns=columns,
        days=config["splits"]["D_design"],
    )
    design_frame = mapper.transform(design_frame)
    design_tokens = encoder.transform(design_frame)
    design_frame["__token__"] = design_tokens
    design_frame["__ref_probability__"] = reference.predict(design_frame)
    design_support = _tuple_counter(design_frame, profile=profile, contexts=contexts)
    for column in domains:
        domains[column].update(design_frame[column].astype(str).unique().tolist())
    combined_support = model_support + design_support

    k = int(config.get("K", 64))
    frequencies = np.bincount(design_tokens, minlength=k).astype(float)
    frequencies = (frequencies + 1e-12) / (frequencies.sum() + k * 1e-12)
    objective_weights = (
        np.ones(k) / k if config.get("pi_weighting", "empirical") == "uniform" else frequencies
    )
    token_scores = np.full(k, design_frame["__ref_probability__"].mean())
    for token in range(k):
        mask = design_tokens == token
        if mask.any():
            token_scores[token] = design_frame.loc[mask, "__ref_probability__"].mean()
    l_count = int(config.get("L", min(16, k)))
    if config.get("nested_partitions", False):
        assignment = build_nested_partitions(np.argsort(token_scores), [l_count])[l_count]
    else:
        assignment = build_partition(
            config.get("partition", "frequency_balanced"),
            frequencies,
            l_count,
            scores=token_scores,
            seed=int(config.get("frozen_design_seed", config.get("seed", 0))),
        )
    decoder = build_decoder(
        config.get("decoder", "design_frequency"),
        assignment,
        frequencies,
        scores=token_scores,
    )
    design_context = _context_values(design_frame, contexts)
    context_levels = sorted(design_context.unique().tolist())
    context_index = {value: idx for idx, value in enumerate(context_levels)}
    token_context_scores = np.tile(token_scores[:, None], (1, len(context_levels)))
    token_context_weights = np.zeros_like(token_context_scores)
    design_probabilities = design_frame["__ref_probability__"].to_numpy()
    context_array = design_context.to_numpy()
    for token in range(k):
        token_mask = design_tokens == token
        for context_value, context_idx in context_index.items():
            mask = token_mask & (context_array == context_value)
            token_context_weights[token, context_idx] = int(mask.sum())
            if mask.any():
                token_context_scores[token, context_idx] = design_probabilities[mask].mean()
    token_context_weights[token_context_weights.sum(axis=1) == 0] = 1.0
    token_cost = token_cost_matrix(
        token_context_scores,
        token_context_weights,
        config.get("distortion", "bernoulli_kl"),
        float(config.get("distortion_clip", 1e-6)),
    )
    cost_weights = (
        np.ones(k) / k if config.get("cost_aggregation", "empirical") == "uniform" else frequencies
    )
    block_cost = block_cost_matrix(token_cost, assignment, decoder, cost_weights)
    block_weights = np.bincount(assignment, weights=objective_weights, minlength=l_count)

    domain_frame = pd.DataFrame(
        {column: pd.Series(sorted(values)) for column, values in domains.items()}
    )
    cartesian_support = {
        tuple(map(str, value))
        for value in expected_tuple_grid(
            domain_frame,
            profile,
            contexts,
            include_fallback_levels=True,
        )
    }
    observed_design_support = set(combined_support)
    design_total = sum(combined_support.values())
    design_probability = {group: count / design_total for group, count in combined_support.items()}
    structural_candidates = cartesian_support - observed_design_support
    design_groups = _groups_from_tuples(profile, observed_design_support)
    design_adjacency = build_adjacency(
        design_groups,
        config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
        float(config.get("epsilon", 1.0)),
        config.get("epsilon_by_attr", {}),
    )
    design_connectivity_gaps = disconnected_hybrid_components(design_groups, design_adjacency)
    cartesian_groups = _groups_from_tuples(profile, cartesian_support)
    cartesian_adjacency = build_adjacency(
        cartesian_groups,
        config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
        float(config.get("epsilon", 1.0)),
        config.get("epsilon_by_attr", {}),
    )
    design_adjacency_hash = hash_json([asdict(pair) for pair in design_adjacency])
    cartesian_adjacency_hash = hash_json([asdict(pair) for pair in cartesian_adjacency])
    support_payload = {
        "profile": profile,
        "G_cart": [list(value) for value in sorted(cartesian_support)],
        "G_design": [list(value) for value in sorted(observed_design_support)],
        "G_cart_minus_G_design": [list(value) for value in sorted(structural_candidates)],
        "design_probability": {
            "|".join(value): probability
            for value, probability in sorted(design_probability.items())
        },
        "G_design_adjacency_count": len(design_adjacency),
        "G_design_adjacency_hash": design_adjacency_hash,
        "G_cart_adjacency_count": len(cartesian_adjacency),
        "G_cart_adjacency_hash": cartesian_adjacency_hash,
    }
    support_path = path / "frozen_support.json"
    support_path.write_text(json.dumps(support_payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        path / "mechanism" / "frozen_design.npz",
        frequencies=frequencies,
        objective_weights=objective_weights,
        token_scores=token_scores,
        assignment=assignment,
        decoder=decoder,
        token_cost=token_cost,
        block_cost=block_cost,
        block_weights=block_weights,
        common_cover_distribution=block_weights / block_weights.sum(),
    )
    design_rows = len(design_frame)
    del design_frame, design_tokens, design_probabilities, context_array
    gc.collect()

    frozen = {
        "mapper": mapper,
        "encoder": encoder,
        "reference": reference,
        "assignment": assignment,
        "decoder": decoder,
        "block_cost": block_cost,
        "block_weights": block_weights,
        "cartesian_support": cartesian_support,
        "design_support": observed_design_support,
        "design_probability": design_probability,
        "support_hash": sha256_file(support_path),
        "mapper_hash": sha256_file(path / "models" / "category_mapper.joblib"),
        "encoder_hash": sha256_file(path / "models" / "encoder.joblib"),
        "model_hash": sha256_file(path / "models" / "reference.joblib"),
        "design_connectivity_gaps": design_connectivity_gaps,
        "design_adjacency_count": len(design_adjacency),
        "design_adjacency_hash": design_adjacency_hash,
        "cartesian_adjacency_count": len(cartesian_adjacency),
        "cartesian_adjacency_hash": cartesian_adjacency_hash,
        "model_rows": model_rows,
        "design_rows": design_rows,
    }
    support_table = pd.DataFrame(
        [
            {
                "group": "|".join(group),
                "in_G_cart": True,
                "in_G_design": group in observed_design_support,
                "structural_zero_candidate": group in structural_candidates,
                "design_count": combined_support.get(group, 0),
                "design_probability": design_probability.get(group, 0.0),
            }
            for group in sorted(cartesian_support)
        ]
    )
    return frozen, support_table


def _confidence_boxes(
    hist: Any,
    adjacency: Sequence[Any],
    config: dict[str, Any],
) -> dict[str, ConfidenceBox]:
    confidence_name = config.get("confidence", "cp_box")
    factory = CONFIDENCE_REGISTRY[confidence_name]
    boxes: dict[str, ConfidenceBox] = {}
    for key, counts in hist.counts.items():
        kwargs = {
            "alpha": float(config.get("alpha_cert", 0.05)),
            "group_count": len(hist.counts),
            "comparisons": max(1, len(adjacency)),
            "tv_radius": float(config.get("shift_tv", 0.0)),
        }
        boxes[key] = factory(counts, **kwargs) if confidence_name != "point" else factory(counts)
    return boxes


def _evaluate_cell(
    *,
    config: dict[str, Any],
    path: Path,
    frozen: dict[str, Any],
    cert_frame: pd.DataFrame,
    cert_tokens: np.ndarray,
    seed: int,
    size_label: str,
    size_target: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    profile = config["profiles"][0]
    contexts = list(config.get("context_cols", []))
    assignment = frozen["assignment"]
    l_count = int(config["L"])
    expected_frame = _expected_frame_from_cartesian(
        frozen["cartesian_support"],
        profile=profile,
        context=contexts[0],
    )
    hist = build_group_histograms(
        cert_frame,
        cert_tokens,
        profile=profile,
        context_columns=contexts,
        alphabet_size=l_count,
        token_to_block=assignment,
        min_group_count=int(config.get("min_group_count", 20)),
        rare_group_policy=config.get("rare_group_policy", "force_cover"),
        expected_frame=expected_frame,
        include_fallback_levels=False,
    )
    observed_support = {
        (*tuple(map(str, group.values)), str(group.context)) for group in hist.groups
    }
    design_missing = frozen["design_support"] - observed_support
    structural_observed = (frozen["cartesian_support"] - frozen["design_support"]).intersection(
        observed_support
    )
    adjacency = build_adjacency(
        hist.groups,
        config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
        float(config.get("epsilon", 1.0)),
        config.get("epsilon_by_attr", {}),
    )
    connectivity_gaps = disconnected_hybrid_components(hist.groups, adjacency)
    if connectivity_gaps:
        hist.force_cover = True
    boxes = _confidence_boxes(hist, adjacency, config)
    fallback = bool(hist.force_cover or not adjacency)
    if fallback:
        block_channel = common_cover(frozen["block_weights"])
        verification = verify_robust_channel(
            block_channel,
            boxes,
            adjacency,
            tolerance=float(config.get("solver_tolerance", 1e-8)),
        )
        solver = {
            "status": "forced_cover_incomplete_group_coverage",
            "objective": float(
                np.sum(frozen["block_weights"][:, None] * block_channel * frozen["block_cost"])
            ),
            "runtime_seconds": 0.0,
            "iterations": 0,
            "primal_gap": 0.0,
            "dual_gap": None,
            "message": "fixed Cartesian support is incomplete or rare",
            "variable_count": l_count * l_count,
            "constraint_count": len(adjacency) * l_count + l_count,
            "estimated_memory_bytes": l_count * l_count * 8,
        }
    else:
        solution, verification = solve_robust_block_lp(
            frozen["block_cost"],
            frozen["block_weights"],
            boxes,
            adjacency,
            tolerance=float(config.get("solver_tolerance", 1e-8)),
            time_limit=config.get("time_limit"),
        )
        solver = asdict(solution.solver)
        if solution.channel is None:
            fallback = True
            block_channel = common_cover(frozen["block_weights"])
            verification = verify_robust_channel(block_channel, boxes, adjacency)
        else:
            block_channel = solution.channel
    cover_channel = common_cover(frozen["block_weights"])
    u_capt = -float(np.sum(frozen["block_weights"][:, None] * block_channel * frozen["block_cost"]))
    u_cover = -float(
        np.sum(frozen["block_weights"][:, None] * cover_channel * frozen["block_cost"])
    )
    nontrivial = bool(np.max(np.abs(block_channel - block_channel[0][None, :])) > 1e-10)
    counts_by_tuple = {
        (*tuple(map(str, group.values)), str(group.context)): int(hist.counts[group.key()].sum())
        for group in hist.groups
    }
    design_counts = np.asarray(
        [counts_by_tuple.get(group, 0) for group in frozen["design_support"]],
        dtype=int,
    )
    design_rare_count = int(np.sum(design_counts < int(config.get("min_group_count", 20))))
    design_support_ready = bool(
        not design_missing and design_rare_count == 0 and frozen["design_connectivity_gaps"] == 0
    )
    probabilities = np.asarray(list(frozen["design_probability"].values()))
    n = len(cert_frame)
    predicted_missing_exact = float(np.sum(np.power(1 - probabilities, n)))
    predicted_missing_exp = float(np.sum(np.exp(-n * probabilities)))
    min_probability = float(probabilities.min())
    group_count = len(probabilities)
    n_all_observed_95 = int(math.ceil(math.log(group_count / 0.05) / min_probability))
    n_expected_20 = int(math.ceil(int(config.get("min_group_count", 20)) / min_probability))

    cell_config = dict(config)
    cell_config["fixed_support_experiment"] = {
        "profile": profile,
        "cert_sampling_seed": seed,
        "cert_size_label": size_label,
        "cert_target_user_days": size_target,
        "cert_realized_user_days": n,
        "sampling": "stable_blake2b_prefix_over_user_day",
        "contribution_selection": "stable_blake2b_min_event_per_user_day",
        "expected_support": "frozen_cartesian_from_full_D_model_union_D_design",
        "support_hash": frozen["support_hash"],
        "mapper_hash": frozen["mapper_hash"],
        "cartesian_adjacency_hash": frozen["cartesian_adjacency_hash"],
        "design_adjacency_hash": frozen["design_adjacency_hash"],
    }
    coverage = {
        "expected_group_count": hist.expected_group_count,
        "observed_group_count": len(hist.groups),
        "missing_group_count": hist.missing_group_count,
        "missing_groups": list(hist.missing_groups),
        "rare_group_count": hist.rare_group_count,
        "hybrid_connectivity_gaps": connectivity_gaps,
        "requires_universal_cover": hist.force_cover,
        "policy": config.get("rare_group_policy", "force_cover"),
        "profile": profile,
        "bundle_profiles": [profile],
        "frozen_support_hash": frozen["support_hash"],
        "frozen_mapper_hash": frozen["mapper_hash"],
        "frozen_cartesian_adjacency_hash": frozen["cartesian_adjacency_hash"],
        "frozen_design_adjacency_hash": frozen["design_adjacency_hash"],
        "cartesian_minus_design_count": len(frozen["cartesian_support"] - frozen["design_support"]),
        "design_minus_cert_count": len(design_missing),
    }
    certificate = make_certificate(
        config=cell_config,
        channel=block_channel,
        boxes=boxes,
        adjacency=adjacency,
        verification=verification,
        solver=solver,
        component_hashes={
            "encoder": frozen["encoder_hash"],
            "model": frozen["model_hash"],
        },
        split_identifiers=config["splits"],
        dp_parameters={
            "epsilon": config.get("dp_hist_epsilon"),
            "delta": config.get("dp_hist_delta"),
            "contribution_policy": config.get("contribution_policy", "one-display-per-uuid-day"),
        },
        histogram_counts=hist.counts,
        assignment=assignment,
        decoder=frozen["decoder"],
        coverage=coverage,
        groups=hist.groups,
    )
    certificate_path = path / f"certificate-seed{seed}-n{size_label}.json"
    certificate.write(certificate_path)
    independent = verify_certificate(certificate_path)
    if not independent.valid:
        raise RuntimeError(f"independent certificate verification failed: {certificate_path}")
    elapsed = time.perf_counter() - started
    return {
        "profile": profile,
        "cert_sampling_seed": seed,
        "cert_size_label": size_label,
        "cert_target_user_days": size_target,
        "cert_user_days": n,
        "G_cart_count": len(frozen["cartesian_support"]),
        "G_design_count": len(frozen["design_support"]),
        "G_cert_count": len(observed_support),
        "cartesian_minus_design_count": len(frozen["cartesian_support"] - frozen["design_support"]),
        "cartesian_minus_design_observed_in_cert_count": len(structural_observed),
        "design_minus_cert_count": len(design_missing),
        "cartesian_minus_cert_count": hist.missing_group_count,
        "cartesian_missing_fraction": hist.missing_group_count / len(frozen["cartesian_support"]),
        "design_missing_fraction": len(design_missing) / len(frozen["design_support"]),
        "rare_group_count_observed": hist.rare_group_count,
        "rare_group_mass": hist.rare_group_mass,
        "design_groups_below_min_count": design_rare_count,
        "min_design_group_count": int(design_counts.min()),
        "min_observed_group_count": int(min(counts_by_tuple.values(), default=0)),
        "hybrid_connectivity_gaps_observed": connectivity_gaps,
        "hybrid_path_complete_observed": connectivity_gaps == 0,
        "hybrid_connectivity_gaps_design": frozen["design_connectivity_gaps"],
        "hybrid_path_complete_design": frozen["design_connectivity_gaps"] == 0,
        "design_support_ready_without_boundary_certificate": design_support_ready,
        "predicted_design_missing_exact_from_design_p": predicted_missing_exact,
        "predicted_design_missing_exp_bound_from_design_p": predicted_missing_exp,
        "min_design_probability": min_probability,
        "n_all_design_groups_observed_95_union_bound": n_all_observed_95,
        "n_min_expected_count_20": n_expected_20,
        "solver_status": solver["status"],
        "solver_runtime_seconds": solver["runtime_seconds"],
        "cell_runtime_seconds": elapsed,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "universal_cover_activated": fallback and not nontrivial,
        "mechanism_fallback_share": 1.0 if fallback else 0.0,
        "nontrivial_channel": nontrivial,
        "U_CAPT": u_capt,
        "U_common_cover": u_cover,
        "U_gain_over_common_cover": u_capt - u_cover,
        "certificate_valid": independent.valid,
        "certificate_checked_constraints": independent.checked_constraints,
        "certificate_max_violation": independent.max_violation,
        "certificate_path": certificate_path.name,
        "frozen_support_hash": frozen["support_hash"],
        "frozen_mapper_hash": frozen["mapper_hash"],
        "frozen_cartesian_adjacency_hash": frozen["cartesian_adjacency_hash"],
        "frozen_design_adjacency_hash": frozen["design_adjacency_hash"],
        "frozen_encoder_hash": frozen["encoder_hash"],
        "frozen_model_hash": frozen["model_hash"],
    }


def _plot_results(results: pd.DataFrame, path: Path) -> None:
    blue = "#2563A6"
    orange = "#D97706"
    grey = "#666666"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ordered_labels = [
        str(value)
        for value in sorted(value for value in results["cert_target_user_days"].dropna().unique())
    ] + [FULL_LABEL]
    label_to_x = {label: idx for idx, label in enumerate(ordered_labels)}
    plot_frame = results.copy()
    plot_frame["x"] = plot_frame["cert_size_label"].map(label_to_x)
    x = np.arange(len(ordered_labels))

    for seed, subset in plot_frame.groupby("cert_sampling_seed"):
        subset = subset.sort_values("x")
        axes[0, 0].plot(
            subset["x"],
            subset["design_minus_cert_count"],
            color=blue,
            alpha=0.3,
            marker="o",
            label=f"sampling missing, seed {seed}",
        )
    structural = int(results["cartesian_minus_design_count"].iloc[0])
    axes[0, 0].axhline(
        structural,
        color=orange,
        linestyle="--",
        label="Cartesian − design (fixed candidates)",
    )
    axes[0, 0].set_title("Frozen-support missing tuple counts")
    axes[0, 0].set_ylabel("tuple count")
    axes[0, 0].legend(fontsize=7, ncol=2)

    grouped = plot_frame.groupby("x", sort=True)
    median_cart = grouped["cartesian_missing_fraction"].median()
    median_design = grouped["design_missing_fraction"].median()
    axes[0, 1].plot(
        x,
        median_cart.reindex(x),
        color=orange,
        marker="s",
        label="Cartesian missing fraction",
    )
    axes[0, 1].plot(
        x,
        median_design.reindex(x),
        color=blue,
        marker="o",
        label="design-support missing fraction",
    )
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].set_title("Median missing fractions across 3 seeds")
    axes[0, 1].set_ylabel("fraction")
    axes[0, 1].legend(fontsize=8)

    for seed, subset in plot_frame.groupby("cert_sampling_seed"):
        subset = subset.sort_values("x")
        axes[1, 0].plot(
            subset["x"],
            subset["min_design_group_count"],
            color=blue,
            alpha=0.35,
            marker="o",
            label=f"seed {seed}",
        )
    axes[1, 0].axhline(
        20,
        color=orange,
        linestyle="--",
        label="min_group_count=20",
    )
    axes[1, 0].set_yscale("symlog", linthresh=1)
    axes[1, 0].set_title("Minimum count over frozen design support")
    axes[1, 0].set_ylabel("minimum count")
    axes[1, 0].legend(fontsize=7, ncol=2)

    fallback = grouped["mechanism_fallback_share"].mean().reindex(x)
    nontrivial = grouped["nontrivial_channel"].mean().reindex(x)
    axes[1, 1].plot(x, fallback, color=orange, marker="s", label="fallback share")
    axes[1, 1].plot(x, nontrivial, color=blue, marker="o", label="nontrivial fraction")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].set_title("Certified mechanism outcome")
    axes[1, 1].set_ylabel("fraction across seeds")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(x, ordered_labels, rotation=0)
        axis.set_xlabel("nested D_cert user-days")
        axis.grid(alpha=0.25, color=grey)
    fig.suptitle("Criteo constrained_2: frozen design and nested certificate samples")
    fig.tight_layout()
    fixed_time = datetime(2000, 1, 1, tzinfo=UTC)
    fig.savefig(
        path / "fixed_support_diagnostic.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "CAPT-12",
            "Producer": "CAPT-12",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    fig.savefig(
        path / "fixed_support_diagnostic.png",
        bbox_inches="tight",
        dpi=220,
        metadata={"Software": "CAPT-12"},
    )
    plt.close(fig)


def _write_report(
    results: pd.DataFrame,
    support: pd.DataFrame,
    path: Path,
) -> None:
    full = results.loc[results["cert_size_label"] == FULL_LABEL]
    structural = int(results["cartesian_minus_design_count"].iloc[0])
    report = f"""# Criteo fixed-support nested D_cert experiment

## Scope

- Profile: `features_kv_bits_constrained_2`
- Frozen once from the full D_model and D_design splits: mapper, encoder, reference model, K/L, partition, decoder, common-cover distribution, cost, objective weights, expected support, and adjacency rule.
- Varied only a stable BLAKE2b-ranked nested prefix of one-display-per-user-day D_cert for seeds 0, 1, and 2.
- Sizes: 5k, 10k, 25k, 50k, 100k, and full.

## Support definitions

- G_cart: frozen Cartesian product used by the current fail-closed coverage rule.
- G_design: joint tuples observed in full D_model union D_design after the frozen mapper.
- G_cert(n): tuples observed in the nested contributed certificate sample.

G_cart has {int(results["G_cart_count"].iloc[0])} tuples; G_design has {int(results["G_design_count"].iloc[0])}. The fixed difference G_cart minus G_design contains {structural} structural-zero candidates.

## Full-split result

Across the three sampling seeds, full D_cert contains {int(full["cert_user_days"].min())} user-days. Cartesian missing ranges from {int(full["cartesian_minus_cert_count"].min())} to {int(full["cartesian_minus_cert_count"].max())}; design-support missing ranges from {int(full["design_minus_cert_count"].min())} to {int(full["design_minus_cert_count"].max())}.

All {len(results)} cells have valid certificates. Certified CAPT remains the input-independent common cover whenever the current Cartesian support is incomplete or rare. `design_support_ready_without_boundary_certificate` is diagnostic only: it must not be interpreted as a sound supported-only certificate because supported/unsupported boundary constraints are not implemented.

## Interpretation

This experiment separates a fixed Cartesian-minus-design component from a sampling component. A persistent Cartesian-minus-design difference is evidence that adding D_cert rows alone cannot satisfy the current Cartesian complete-coverage gate. It does not establish that those tuples are impossible in the deployment population, and it does not authorize replacing G_cart by G_design without a runtime support rule and boundary constraints.

The probability-based sample-size projections use frequencies from D_model union D_design and therefore require temporal stationarity to predict D_cert coverage. They are diagnostics, not certificate guarantees.
"""
    (path / "fixed_support_report.md").write_text(report)


def run_fixed_support_scaling(config: dict[str, Any]) -> Path:
    """Run one frozen Criteo design against nested D_cert user-day samples."""
    config = validate_config(config)
    required_splits = {"D_model", "D_design", "D_cert", "D_attack_train", "D_test"}
    if set(config.get("splits", {})) != required_splits:
        raise ValueError(
            f"fixed-support scaling requires exactly five splits: {sorted(required_splits)}"
        )
    assert_disjoint_splits(config["splits"])
    profiles = list(config.get("profiles", []))
    contexts = list(config.get("context_cols", []))
    if profiles != ["features_kv_bits_constrained_2"]:
        raise ValueError(
            "fixed-support scaling currently requires only features_kv_bits_constrained_2"
        )
    if len(contexts) != 1:
        raise ValueError("fixed-support scaling currently requires exactly one context")
    experiment = dict(config.get("fixed_support_scaling", {}))
    sizes = [int(value) for value in experiment.get("cert_sizes", [])]
    seeds = [int(value) for value in experiment.get("cert_sampling_seeds", [])]
    if not sizes or not seeds:
        raise ValueError("fixed_support_scaling requires cert_sizes and cert_sampling_seeds")
    config = dict(config)
    config["frozen_design_seed"] = int(experiment.get("frozen_design_seed", 0))
    config["fixed_support_scaling"] = {
        "cert_sizes": sorted(set(sizes)),
        "cert_sampling_seeds": sorted(set(seeds)),
        "frozen_design_seed": config["frozen_design_seed"],
    }
    config = record_source_provenance(config)
    path = prepare_run(config)
    started = time.perf_counter()
    frozen, support_table = _fixed_design(config, path)
    support_table.to_csv(path / "tables" / "frozen_support.csv", index=False)
    support_table.to_parquet(path / "frozen_support.parquet", index=False)

    profile = profiles[0]
    profile_attributes = profile.split("+")
    source = list(config.get("phi_source_cols", []))
    user_col = config.get("user_col", "user_id")
    id_col = config.get("id_col", "id")
    cert_columns = list(dict.fromkeys([id_col, user_col, *profile_attributes, *contexts, *source]))
    cert_source = load_parquet_sample(
        data_root=config["data_root"],
        columns=cert_columns,
        days=config["splits"]["D_cert"],
    )
    # The other splits are not loaded, but the configured day sets remain disjoint.
    assert_no_row_overlap({"D_cert": cert_source}, id_col)
    cert_source = frozen["mapper"].transform(cert_source)
    contributed = select_one_display_per_user_day(
        cert_source,
        user_col=user_col,
        day_col="day_int",
        id_col=id_col,
    )
    cert_tokens_all = frozen["encoder"].transform(contributed)
    rows: list[dict[str, Any]] = []
    nested_checks: dict[str, bool] = {}
    for seed in sorted(set(seeds)):
        positions = stable_nested_positions(
            contributed,
            sizes=sizes,
            seed=seed,
            user_col=user_col,
            day_col="day_int",
        )
        prior: set[int] = set()
        nested = True
        for label in [*map(str, sorted(set(sizes))), FULL_LABEL]:
            current = set(map(int, positions[label]))
            nested = nested and prior.issubset(current)
            prior = current
            selected = positions[label]
            target = None if label == FULL_LABEL else int(label)
            rows.append(
                _evaluate_cell(
                    config=config,
                    path=path,
                    frozen=frozen,
                    cert_frame=contributed.iloc[selected].reset_index(drop=True),
                    cert_tokens=cert_tokens_all[selected],
                    seed=seed,
                    size_label=label,
                    size_target=target,
                )
            )
        nested_checks[str(seed)] = nested
    results = pd.DataFrame(rows)
    results.to_csv(path / "tables" / "fixed_support_scaling.csv", index=False)
    results.to_parquet(path / "fixed_support_scaling.parquet", index=False)
    _plot_results(results, path)
    _write_report(results, support_table, path)
    total_seconds = time.perf_counter() - started
    metadata = {
        "experiment": "criteo_fixed_support_nested_D_cert",
        "source_git_sha": config["source_git_sha"],
        "profile": profile,
        "model_rows": frozen["model_rows"],
        "design_rows": frozen["design_rows"],
        "cert_source_rows": len(cert_source),
        "cert_contributed_user_days": len(contributed),
        "cert_sizes": sorted(set(sizes)),
        "cert_sampling_seeds": sorted(set(seeds)),
        "nested_checks": nested_checks,
        "frozen_support_hash": frozen["support_hash"],
        "frozen_mapper_hash": frozen["mapper_hash"],
        "frozen_encoder_hash": frozen["encoder_hash"],
        "frozen_model_hash": frozen["model_hash"],
        "wall_seconds": total_seconds,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "created_at": datetime.now(UTC).isoformat(),
        "environment": environment(),
    }
    (path / "fixed_support_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    finish_run(
        path,
        {
            "dataset": "criteo",
            "experiment": metadata["experiment"],
            "profile": profile,
            "result_rows": len(results),
            "certificates": results["certificate_path"].tolist(),
            "nested_checks": nested_checks,
        },
    )
    return path
