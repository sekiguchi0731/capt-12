from __future__ import annotations

import heapq
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from capt12.audit.lower import lower_audit
from capt12.bounds.theorem4 import theorem4_envelope
from capt12.certification.artifact import make_certificate
from capt12.certification.robust import solve_robust_block_lp, verify_robust_channel
from capt12.confidence.boxes import CONFIDENCE_REGISTRY, ConfidenceBox
from capt12.config import run_id, validate_config
from capt12.data.contributions import apply_contribution_policy
from capt12.data.inspect import day_from_path, discover_parquet
from capt12.data.loader import assert_no_row_overlap, load_temporal_splits
from capt12.data.preprocessing import (
    FrozenCategoryMapper,
    build_group_histograms,
    build_marginal_histograms,
)
from capt12.data.profiles import assign_simulated_profiles
from capt12.data.synthetic import generate_synthetic, theorem4_counterexample
from capt12.decoders.registry import build_decoder
from capt12.distortions.registry import block_cost_matrix, token_cost_matrix
from capt12.encoders.base import PrecomputedEncoder, ScoreEncoder, make_encoder
from capt12.evaluation.metrics import (
    evaluate_target_ctrs,
    expected_channel_metrics,
    prediction_metrics,
)
from capt12.mechanisms.baselines import (
    common_cover,
    k_ary_rr,
    raw_identity,
    scalar_keep_or_cover,
    tokenwise_keep_or_cover,
)
from capt12.mechanisms.lp import (
    lift_block_channel,
    privacy_violations,
    solve_block_lp,
    solve_full_lp,
)
from capt12.models.reference import (
    ReferenceModel,
    named_reference_model,
    validate_external_reference_inputs,
)
from capt12.partitions.registry import build_nested_partitions, build_partition
from capt12.privacy.adjacency import (
    build_adjacency,
    disconnected_hybrid_components,
)
from capt12.privacy.profile import evaluate_profile_privacy
from capt12.utils.artifacts import finish_run, prepare_run, sha256_file


def _aggregate_distribution(distribution: np.ndarray, assignment: np.ndarray, l_count: int) -> np.ndarray:
    return np.bincount(assignment, weights=distribution, minlength=l_count)


def _singleton_decoder(k: int) -> np.ndarray:
    return np.eye(k)


def _metric_row(
    *,
    mechanism: str,
    channel: np.ndarray | None,
    weights: np.ndarray,
    cost: np.ndarray,
    target_epsilon: float,
    certified_epsilon: float | None,
    l_count: int,
    k: int,
    solver: Any | None = None,
    assignment: np.ndarray | None = None,
    certified: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mechanism": mechanism,
        "K": k,
        "L": l_count,
        "target_epsilon": target_epsilon,
        "certificate_epsilon": certified_epsilon,
        "certified": certified,
        "feasible": channel is not None,
        "is_bound": mechanism == "theorem4_envelope",
    }
    if channel is not None:
        row.update(expected_channel_metrics(channel, weights, cost, assignment))
        row["utility_retention"] = row["exact_token_retention"]
        row["mechanism_table_size"] = int(channel.size)
        row["nontrivial_channel"] = bool(
            np.max(np.abs(channel - channel[0][None, :])) > 1e-10
        )
    if solver is not None:
        row.update(
            {
                "solver_status": solver.status,
                "solver_runtime": solver.runtime_seconds,
                "solver_iterations": solver.iterations,
                "variable_count": solver.variable_count,
                "constraint_count": solver.constraint_count,
                "estimated_memory": solver.estimated_memory_bytes,
                "primal_gap": solver.primal_gap,
                "dual_gap": solver.dual_gap,
            }
        )
    return row


def run_synthetic(config: dict[str, Any]) -> tuple[Path, pd.DataFrame]:
    config = validate_config(config)
    path = prepare_run(config)
    k = int(config.get("K", 6))
    l_count = int(config.get("L", min(3, k)))
    seed = int(config.get("seed", 0))
    epsilon = float(config.get("epsilon", 0.5))
    population = generate_synthetic(int(config.get("n_rows", 3000)), k, seed, epsilon)
    frequencies = np.bincount(population.frame["token"], minlength=k).astype(float)
    frequencies /= frequencies.sum()
    objective_weights = (
        np.ones(k) / k
        if config.get("pi_weighting", "empirical") == "uniform"
        else frequencies
    )
    if config.get("nested_partitions", False):
        assignment = build_nested_partitions(np.argsort(population.token_scores), [l_count])[
            l_count
        ]
    else:
        assignment = build_partition(
            config.get("partition", "frequency_balanced"),
            frequencies,
            l_count,
            scores=population.token_scores,
            group_distributions=np.vstack(list(population.distributions.values())),
            seed=seed,
        )
    decoder = build_decoder(
        config.get("decoder", "design_frequency"),
        assignment,
        frequencies,
        scores=population.token_scores,
    )
    block_groups = {
        key: _aggregate_distribution(value, assignment, l_count)
        for key, value in population.distributions.items()
    }
    distortion = config.get("distortion", "retention")
    context_probabilities = np.c_[population.token_scores, np.clip(population.token_scores * 1.15, 1e-6, 1 - 1e-6)]
    context_weights = np.ones_like(context_probabilities) / 2
    token_cost = token_cost_matrix(context_probabilities, context_weights, distortion)
    cost_weights = (
        np.ones(k) / k
        if config.get("cost_aggregation", "empirical") == "uniform"
        else frequencies
    )
    block_cost = block_cost_matrix(token_cost, assignment, decoder, cost_weights)
    block_weights = np.bincount(assignment, weights=objective_weights, minlength=l_count)
    tolerance = float(config.get("solver_tolerance", 1e-8))
    block_solution = solve_block_lp(
        block_cost,
        block_weights,
        block_groups,
        population.adjacency,
        tolerance=tolerance,
    )
    if block_solution.channel is None:
        raise RuntimeError(block_solution.solver.message)
    token_channel = lift_block_channel(block_solution.channel, assignment, decoder)
    point_boxes = {
        key: ConfidenceBox(value, value, value, "point", 1.0) for key, value in block_groups.items()
    }
    verification = verify_robust_channel(block_solution.channel, point_boxes, population.adjacency, tolerance=tolerance)
    certificate = make_certificate(
        config=config,
        channel=block_solution.channel,
        boxes=point_boxes,
        adjacency=population.adjacency,
        verification=verification,
        solver=block_solution.solver,
        component_hashes={},
        split_identifiers={"D_model": "synthetic-days-1:6", "D_design": "7:12", "D_cert": "13:18"},
        assignment=assignment,
        decoder=decoder,
        coverage={"known_population": True, "requires_universal_cover": False},
        groups=population.groups,
    )
    certificate.write(path / "certificate.json")
    np.savez_compressed(path / "mechanism" / "capt_block.npz", channel=block_solution.channel, assignment=assignment, decoder=decoder)
    mechanisms = config.get(
        "mechanisms",
        ["raw_identity", "common_cover", "k_ary_rr", "scalar_keep_or_cover", "tokenwise_keep_or_cover", "capt_block", "capt_full"],
    )
    rows: list[dict[str, Any]] = []
    candidates = {
        "raw_identity": raw_identity(k),
        "common_cover": common_cover(frequencies),
        "k_ary_rr": k_ary_rr(k, epsilon),
        "scalar_keep_or_cover": scalar_keep_or_cover(frequencies, min(1.0, math.expm1(epsilon) / (math.expm1(epsilon) + k))),
        "tokenwise_keep_or_cover": tokenwise_keep_or_cover(frequencies, np.full(k, min(1.0, epsilon / (1 + epsilon)))),
        "capt_block": token_channel,
    }
    for name in mechanisms:
        if name in candidates:
            channel = candidates[name]
            violations = privacy_violations(channel, population.distributions, population.adjacency)
            solver = block_solution.solver if name == "capt_block" else None
            rows.append(
                _metric_row(
                    mechanism=name,
                    channel=channel,
                    weights=objective_weights,
                    cost=token_cost,
                    target_epsilon=epsilon,
                    certified_epsilon=verification.realized_epsilon if name == "capt_block" else (epsilon if not violations else None),
                    l_count=l_count if name == "capt_block" else k,
                    k=k,
                    solver=solver,
                    assignment=assignment if name == "capt_block" else None,
                    certified=(name == "capt_block" and verification.valid) or (not violations and name in {"common_cover", "k_ary_rr"}),
                )
            )
        elif name == "capt_full":
            full = solve_full_lp(
                token_cost,
                objective_weights,
                population.distributions,
                population.adjacency,
                full_max_k=int(config.get("full_max_k", 256)),
                force=bool(config.get("force_full", False)),
                tolerance=tolerance,
            )
            full_verification = (
                verify_robust_channel(
                    full.channel,
                    {
                        key: ConfidenceBox(value, value, value, "point", 1.0)
                        for key, value in population.distributions.items()
                    },
                    population.adjacency,
                    tolerance=tolerance,
                )
                if full.channel is not None
                else None
            )
            full_eligible = bool(
                full.channel is not None
                and full.solver.status == "optimal"
                and full_verification is not None
                and full_verification.valid
            )
            full_row = _metric_row(
                mechanism="capt_full",
                channel=full.channel,
                weights=objective_weights,
                cost=token_cost,
                target_epsilon=epsilon,
                certified_epsilon=(
                    full_verification.realized_epsilon
                    if full_eligible and full_verification is not None
                    else None
                ),
                l_count=k,
                k=k,
                solver=full.solver,
                certified=full_eligible,
            )
            full_row.update(
                {
                    "full_verification_valid": bool(
                        full_verification is not None and full_verification.valid
                    ),
                    "full_oracle_optimized": full_eligible,
                    "full_comparison_eligible": full_eligible,
                }
            )
            rows.append(full_row)
    if distortion == "retention":
        envelope = theorem4_envelope(
            population.distributions, population.adjacency, objective_weights
        )
        rows.append(
            {
                "mechanism": "theorem4_envelope",
                "K": k,
                "L": k,
                "target_epsilon": epsilon,
                "utility_retention": envelope.utility,
                "exact_token_retention": envelope.utility,
                "is_bound": True,
                "certified": False,
                "feasible": False,
                "bound_label": "population envelope",
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["seed"] = seed
    metrics["distortion"] = distortion
    metrics["partition"] = config.get("partition", "frequency_balanced")
    metrics["decoder"] = config.get("decoder", "design_frequency")
    metrics["nested_partition"] = bool(config.get("nested_partitions", False))
    metrics["pi_weighting"] = config.get("pi_weighting", "empirical")
    metrics["profile_weighting"] = config.get("profile_weighting", "global_design")
    metrics["cost_aggregation"] = config.get("cost_aggregation", "empirical")
    metrics["confidence"] = config.get("confidence", "point")
    metrics["case"] = "standard"
    metrics["certificate_audit_gap"] = np.nan
    metrics.to_parquet(path / "metrics.parquet", index=False)
    metrics.to_csv(path / "tables" / "metrics.csv", index=False)
    finish_run(path, {"dataset": "synthetic", "rows": len(population.frame)})
    return path, metrics


def run_theorem4_grid(config: dict[str, Any], resume: bool = False) -> pd.DataFrame:
    base = dict(config)
    all_rows: list[pd.DataFrame] = []
    dimensions = itertools.product(
        base.get("seeds", [base.get("seed", 0)]),
        base.get("epsilon_list", [base.get("epsilon", 0.0)]),
        base.get("L_list", [base.get("L", base.get("K", 3))]),
        base.get("partition_list", [base.get("partition", "frequency_balanced")]),
        base.get("decoder_list", [base.get("decoder", "design_frequency")]),
    )
    for seed, epsilon, l_count, partition, decoder_name in dimensions:
        k = int(base.get("K", 3))
        if l_count > k:
            continue
        cfg = {
            **base,
            "dataset": "synthetic",
            "seed": int(seed),
            "epsilon": float(epsilon),
            "L": int(l_count),
            "partition": str(partition),
            "decoder": str(decoder_name),
            "distortion": "retention",
            "mechanisms": ["common_cover", "capt_block", "capt_full"],
        }
        target = (
            Path(cfg.get("output_dir", "outputs/runs"))
            / run_id(validate_config(cfg))
            / "metrics.parquet"
        )
        if resume and target.exists():
            frame = pd.read_parquet(target)
        else:
            _, frame = run_synthetic(cfg)
        all_rows.append(frame)
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    # Always include the requested strict counterexample as a standardized run.
    distributions, adjacency = theorem4_counterexample()
    weights = np.ones(3) / 3
    full = solve_full_lp(1 - np.eye(3), weights, distributions, adjacency, full_max_k=256)
    full_verification = verify_robust_channel(
        full.channel,
        {
            key: ConfidenceBox(value, value, value, "point", 1.0)
            for key, value in distributions.items()
        },
        adjacency,
    )
    envelope = theorem4_envelope(distributions, adjacency, weights)
    counter_rows = pd.DataFrame(
        [
            {
                "mechanism": mechanism,
                "K": 3,
                "L": 3,
                "target_epsilon": 0.0,
                "utility_retention": utility,
                "exact_token_retention": utility,
                "seed": 0,
                "distortion": "retention",
                "case": "theorem4_counterexample",
                "partition": str(partition),
                "decoder": str(decoder_name),
                "confidence": "point",
                "feasible": mechanism == "capt_full",
                "full_verification_valid": (
                    full_verification.valid if mechanism == "capt_full" else math.nan
                ),
                "full_oracle_optimized": (
                    full.solver.status == "optimal" and full_verification.valid
                    if mechanism == "capt_full"
                    else math.nan
                ),
                "full_comparison_eligible": (
                    full.solver.status == "optimal" and full_verification.valid
                    if mechanism == "capt_full"
                    else math.nan
                ),
                "bound_label": (
                    "population envelope" if mechanism == "theorem4_envelope" else None
                ),
            }
            for partition in base.get(
                "partition_list", [base.get("partition", "frequency_balanced")]
            )
            for decoder_name in base.get(
                "decoder_list", [base.get("decoder", "design_frequency")]
            )
            for mechanism, utility in [
                ("capt_full", 1 - full.solver.objective),
                ("theorem4_envelope", envelope.utility),
            ]
        ]
    )
    result = pd.concat([result, counter_rows], ignore_index=True)
    summary = Path(base.get("output_dir", "outputs/runs")) / "synthetic_theorem4_results.parquet"
    summary.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(summary, index=False)
    return result


def _fit_reference(config: dict[str, Any], frames: dict[str, pd.DataFrame], feature_columns: list[str]) -> ReferenceModel:
    path = config.get("fixed_model_path")
    if path:
        return validate_external_reference_inputs(
            ReferenceModel.load(path),
            allowed_feature_columns=feature_columns,
            sensitive_columns=config.get("sensitive_cols", []),
        )
    model = named_reference_model(
        config.get("fixed_model", "ctr_model"),
        eta=float(config.get("distortion_clip", 1e-6)),
        prediction_column=config.get("prediction_column"),
    )
    if model.prediction_column is not None:
        return validate_external_reference_inputs(
            model,
            allowed_feature_columns=feature_columns,
            sensitive_columns=config.get("sensitive_cols", []),
            prediction_manifest_path=config.get("prediction_manifest_path"),
        )
    return model.fit(
        frames["D_model"],
        config.get("label_col", "is_clicked"),
        feature_columns,
        split_id="D_model",
        sensitive_columns=config.get("sensitive_cols", []),
    )


def _sample_outputs(tokens: np.ndarray, channel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.asarray([rng.choice(channel.shape[1], p=channel[token]) for token in tokens], dtype=int)


def _audit_with_context_strata(
    attack: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_col: str,
    output_col: str,
    contexts: list[str],
    alpha: float,
):
    return lower_audit(
        attack,
        test,
        group_col=group_col,
        output_col=output_col,
        context_cols=contexts,
        alpha=alpha,
    )


def _audit_day_robustness(
    attack: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_col: str,
    output_col: str,
    contexts: list[str],
    alpha: float,
    day_col: str = "day_int",
) -> tuple[int, float, float]:
    """Descriptive day-wise sensitivity check; not a clustered confidence interval."""
    if day_col not in test:
        return 0, math.nan, math.nan
    values = sorted(test[day_col].dropna().unique().tolist())
    results = [
        lower_audit(
            attack,
            test.loc[test[day_col] == value],
            group_col=group_col,
            output_col=output_col,
            context_cols=contexts,
            alpha=alpha,
        ).epsilon_lower
        for value in values
    ]
    if not results:
        return 0, math.nan, math.nan
    return len(results), float(np.min(results)), float(np.max(results))


def _certified_hybrid_path_upper(
    witness: dict[str, Any],
    *,
    profile: str,
    groups: list[Any],
    adjacency: list[Any],
) -> float | None:
    """Return the shortest embedded certificate path for an audit witness."""
    left_values = tuple(map(str, witness.get("left_group_values", [])))
    right_values = tuple(map(str, witness.get("right_group_values", [])))
    context = str(witness.get("context", "all"))
    lookup = {
        (group.profile, str(group.context), tuple(map(str, group.values))): group.key()
        for group in groups
    }
    left = lookup.get((profile, context, left_values))
    right = lookup.get((profile, context, right_values))
    if left is None or right is None:
        return None
    graph: dict[str, list[tuple[str, float]]] = {}
    for pair in adjacency:
        graph.setdefault(pair.left, []).append((pair.right, float(pair.epsilon)))
    distances = {left: 0.0}
    queue = [(0.0, left)]
    while queue:
        distance, node = heapq.heappop(queue)
        if node == right:
            return distance
        if distance > distances.get(node, math.inf):
            continue
        for neighbor, weight in graph.get(node, []):
            candidate = distance + weight
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return None


def _audit_certificate_comparison(
    audit: Any,
    *,
    profile: str,
    groups: list[Any],
    adjacency: list[Any],
    config: dict[str, Any],
    universal_upper: float | None,
) -> tuple[float | None, float, bool]:
    """Return a comparable path bound only under an explicit population bridge.

    The lower audit is evaluated on D_test, while finite-sample confidence sets
    describe D_cert.  A tuple-adjacent hybrid bound is therefore numerically
    comparable only for an input-independent channel or when the run explicitly
    declares a stationarity/shift-coverage assumption.
    """
    if universal_upper is not None:
        upper = universal_upper
        comparable = True
    elif audit.witness is None:
        return None, math.nan, False
    else:
        path_upper = _certified_hybrid_path_upper(
            audit.witness,
            profile=profile,
            groups=groups,
            adjacency=adjacency,
        )
        if path_upper is None:
            return None, math.nan, False
        upper = path_upper
        comparable = bool(config.get("audit_bridge_evidence")) and config.get(
            "audit_population_assumption"
        ) in {
            "stationary",
            "covered_by_shift_set",
        }
    gap = upper - audit.epsilon_lower if comparable else None
    return gap, upper, comparable


def run_criteo(config: dict[str, Any], *, max_rows: int | None = None) -> tuple[Path, pd.DataFrame]:
    config = validate_config(config)
    required_splits = {"D_model", "D_design", "D_cert", "D_attack_train", "D_test"}
    if set(config.get("splits", {})) != required_splits:
        raise ValueError(f"Criteo requires exactly five splits: {sorted(required_splits)}")
    path = prepare_run(config)
    sensitive = list(config.get("sensitive_cols", []))
    contexts = list(config.get("context_cols", []))
    source = list(config.get("phi_source_cols", []))
    label = config.get("label_col", "is_clicked")
    id_col = config.get("id_col", "id")
    user_col = config.get("user_col", "user_id")
    optional_input_columns = [
        value
        for value in [config.get("prediction_column"), config.get("precomputed_token_col")]
        if value
    ]
    columns = list(
        dict.fromkeys(
            [id_col, user_col, label, *sensitive, *contexts, *source, *optional_input_columns]
        )
    )
    frames = load_temporal_splits(
        config["data_root"],
        config["splits"],
        columns=columns,
        max_rows_per_split=max_rows or config.get("max_rows"),
        sample_frac=float(config.get("sample_frac", 1.0)),
        seed=int(config.get("seed", 0)),
        id_col=id_col,
    )
    if any(frame.empty for frame in frames.values()):
        raise ValueError("at least one temporal split is empty")
    assert_no_row_overlap(frames, id_col)
    mapper = FrozenCategoryMapper(int(config.get("max_context_cardinality", 32))).fit(
        frames["D_model"], [*sensitive, *contexts], split_id="D_model"
    )
    frames = {name: mapper.transform(frame) for name, frame in frames.items()}
    encoder_score_model = _fit_reference(config, frames, source)
    for frame in frames.values():
        frame["__encoder_score__"] = encoder_score_model.predict(frame)
    k = int(config.get("K", 64))
    encoder = make_encoder(
        config.get("phi", "hash"),
        k,
        seed=int(config.get("seed", 0)),
        sensitive_policy=config.get("phi_sensitive_policy", "exclude"),
        column=config.get("precomputed_token_col", "token"),
    )
    if isinstance(encoder, ScoreEncoder):
        encoder.fit_scores(
            frames["D_model"]["__encoder_score__"].to_numpy(), split_id="D_model"
        )
        tokens = {
            name: encoder.transform_scores(frame["__encoder_score__"].to_numpy())
            for name, frame in frames.items()
        }
    elif isinstance(encoder, PrecomputedEncoder):
        encoder.fit(frames["D_model"])
        tokens = {name: encoder.transform(frame) for name, frame in frames.items()}
    else:
        encoder.fit(
            frames["D_model"],
            source,
            split_id="D_model",
            sensitive_columns=sensitive,
            labels=frames["D_model"][label].to_numpy(),
        )
        tokens = {name: encoder.transform(frame) for name, frame in frames.items()}
    for name, frame in frames.items():
        frame["__token__"] = tokens[name]
    if config.get("fixed_model_path") or config.get("fixed_model") == "precomputed":
        model = encoder_score_model
    else:
        model = named_reference_model(
            config.get("fixed_model", "ctr_model"),
            eta=float(config.get("distortion_clip", 1e-6)),
        ).fit(
            frames["D_model"],
            label,
            ["__token__", *contexts],
            split_id="D_model",
            sensitive_columns=sensitive,
        )
    for frame in frames.values():
        frame["__ref_probability__"] = model.predict(frame)
    joblib.dump(model, path / "models" / "reference.joblib")
    joblib.dump(encoder, path / "models" / "encoder.joblib")
    frequencies = np.bincount(tokens["D_design"], minlength=k).astype(float)
    frequencies = (frequencies + 1e-12) / (frequencies.sum() + k * 1e-12)
    objective_weights = (
        np.ones(k) / k
        if config.get("pi_weighting", "empirical") == "uniform"
        else frequencies
    )
    token_scores = np.full(k, frames["D_design"]["__ref_probability__"].mean())
    for token in range(k):
        mask = tokens["D_design"] == token
        if mask.any():
            token_scores[token] = frames["D_design"].loc[mask, "__ref_probability__"].mean()
    profile = config.get("profiles", [sensitive[0]])[0]
    design_hist = build_group_histograms(
        frames["D_design"],
        tokens["D_design"],
        profile=profile,
        context_columns=contexts,
        alphabet_size=k,
        min_group_count=1,
        rare_group_policy="force_cover",
    )
    design_distributions = np.vstack(
        [counts / counts.sum() for counts in design_hist.counts.values()]
    )
    l_count = int(config.get("L", min(16, k)))
    if config.get("nested_partitions", False):
        assignment = build_nested_partitions(np.argsort(token_scores), [l_count])[l_count]
    else:
        assignment = build_partition(
            config.get("partition", "frequency_balanced"),
            frequencies,
            l_count,
            scores=token_scores,
            group_distributions=design_distributions,
            seed=int(config.get("seed", 0)),
        )
    decoder = build_decoder(
        config.get("decoder", "design_frequency"), assignment, frequencies, scores=token_scores
    )
    design_context = (
        frames["D_design"][contexts].astype(str).agg("|".join, axis=1)
        if contexts
        else pd.Series("all", index=frames["D_design"].index)
    )
    context_levels = sorted(design_context.unique().tolist())
    context_index = {value: idx for idx, value in enumerate(context_levels)}
    token_context_scores = np.tile(token_scores[:, None], (1, len(context_levels)))
    token_context_weights = np.zeros_like(token_context_scores)
    design_probabilities = frames["D_design"]["__ref_probability__"].to_numpy()
    for token in range(k):
        token_mask = tokens["D_design"] == token
        for context_value, context_idx in context_index.items():
            mask = token_mask & (design_context.to_numpy() == context_value)
            token_context_weights[token, context_idx] = int(mask.sum())
            if mask.any():
                token_context_scores[token, context_idx] = design_probabilities[mask].mean()
    zero_rows = token_context_weights.sum(axis=1) == 0
    token_context_weights[zero_rows] = 1.0
    token_cost = token_cost_matrix(
        token_context_scores,
        token_context_weights,
        config.get("distortion", "bernoulli_kl"),
        float(config.get("distortion_clip", 1e-6)),
    )
    cost_weights = (
        np.ones(k) / k
        if config.get("cost_aggregation", "empirical") == "uniform"
        else frequencies
    )
    block_cost = block_cost_matrix(token_cost, assignment, decoder, cost_weights)
    block_weights = np.bincount(assignment, weights=objective_weights, minlength=l_count)
    all_rows: list[dict[str, Any]] = []
    certificates = []
    profile_channels: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(int(config.get("seed", 0)))
    profiles = config.get("profiles", [profile])
    contribution_policy = config.get("contribution_policy", "one-display-per-uuid-day")
    cert_frame = apply_contribution_policy(
        frames["D_cert"],
        user_col=user_col,
        day_col="day_int",
        policy=contribution_policy,
        cap=int(config.get("contribution_cap", 1)),
        seed=int(config.get("seed", 0)),
    )
    cert_tokens = tokens["D_cert"][cert_frame.index.to_numpy()]
    audit_frames: dict[str, pd.DataFrame] = {}
    audit_tokens: dict[str, np.ndarray] = {}
    for split_name in ("D_attack_train", "D_test"):
        audit_frame = apply_contribution_policy(
            frames[split_name],
            user_col=user_col,
            day_col="day_int",
            policy=contribution_policy,
            cap=int(config.get("contribution_cap", 1)),
            seed=int(config.get("seed", 0)),
        )
        audit_frames[split_name] = audit_frame
        audit_tokens[split_name] = tokens[split_name][audit_frame.index.to_numpy()]
    expected_group_frame = pd.concat(
        [frames["D_model"], frames["D_design"]], ignore_index=True
    )
    for profile in profiles:
        histogram_builder = (
            build_marginal_histograms
            if config.get("adjacency", config.get("privacy_scope")) == "marginal"
            else build_group_histograms
        )
        hist = histogram_builder(
            cert_frame,
            cert_tokens,
            profile=profile,
            context_columns=contexts,
            alphabet_size=l_count,
            token_to_block=assignment,
            min_group_count=int(config.get("min_group_count", 20)),
            rare_group_policy=config.get("rare_group_policy", "force_cover"),
            expected_frame=expected_group_frame,
            include_fallback_levels=True,
        )
        adjacency = build_adjacency(
            hist.groups,
            config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
            float(config.get("epsilon", 1.0)),
            config.get("epsilon_by_attr", {}),
        )
        connectivity_gaps = disconnected_hybrid_components(hist.groups, adjacency)
        if connectivity_gaps:
            rare_policy = config.get("rare_group_policy", "force_cover")
            if rare_policy == "force_cover":
                hist.force_cover = True
            else:
                raise ValueError(
                    f"protected tuple graph has {connectivity_gaps} disconnected hybrid components; "
                    "use force_cover or provide a full-support certificate split"
                )
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
        mechanism_used_fallback = False
        if hist.force_cover or not adjacency:
            mechanism_used_fallback = True
            block_channel = common_cover(block_weights)
            verification = verify_robust_channel(
                block_channel, boxes, adjacency, tolerance=float(config.get("solver_tolerance", 1e-8))
            )
            solver_info = {
                "status": (
                    "forced_cover_incomplete_group_coverage"
                    if hist.force_cover
                    and (hist.missing_group_count or connectivity_gaps)
                    else "forced_cover_rare_group"
                    if hist.force_cover
                    else "cover_no_adjacent_pairs"
                ),
                "objective": float(np.sum(block_weights[:, None] * block_channel * block_cost)),
                "runtime_seconds": 0.0,
                "iterations": 0,
                "primal_gap": 0.0,
                "dual_gap": None,
                "variable_count": l_count * l_count,
                "constraint_count": len(adjacency) * l_count + l_count,
                "estimated_memory_bytes": l_count * l_count * 8,
                "message": "explicit group-coverage fallback",
            }
        else:
            solution, verification = solve_robust_block_lp(
                block_cost,
                block_weights,
                boxes,
                adjacency,
                tolerance=float(config.get("solver_tolerance", 1e-8)),
                time_limit=config.get("time_limit"),
            )
            if solution.channel is None:
                # Cover is always mathematically feasible; preserve failure as numerical status.
                mechanism_used_fallback = True
                block_channel = common_cover(block_weights)
                verification = verify_robust_channel(block_channel, boxes, adjacency)
            else:
                block_channel = solution.channel
            solver_info = asdict(solution.solver)
        token_channel = lift_block_channel(block_channel, assignment, decoder)
        profile_channels[profile] = token_channel
        certificate_eligible = verification.valid and not any(
            box.experimental for box in boxes.values()
        )
        attack = audit_frames["D_attack_train"].copy()
        test = audit_frames["D_test"].copy()
        attack["__output__"] = _sample_outputs(
            audit_tokens["D_attack_train"], token_channel, rng
        )
        test["__output__"] = _sample_outputs(audit_tokens["D_test"], token_channel, rng)
        attrs = profile.split("+")
        attack["__audit_group__"] = pd.Series(
            list(map(tuple, attack[attrs].astype(str).to_numpy())), index=attack.index
        )
        test["__audit_group__"] = pd.Series(
            list(map(tuple, test[attrs].astype(str).to_numpy())), index=test.index
        )
        audit = _audit_with_context_strata(
            attack,
            test,
            group_col="__audit_group__",
            output_col="__output__",
            contexts=contexts,
            alpha=float(config.get("alpha_cert", 0.05)),
        )
        audit_day_count, audit_day_min, audit_day_max = _audit_day_robustness(
            attack,
            test,
            group_col="__audit_group__",
            output_col="__output__",
            contexts=contexts,
            alpha=float(config.get("alpha_cert", 0.05)),
        )
        universal_cover = bool(
            np.allclose(block_channel, block_channel[0][None, :], atol=1e-10, rtol=0)
        )
        audit_gap, audit_upper, audit_comparable = _audit_certificate_comparison(
            audit,
            profile=profile,
            groups=hist.groups,
            adjacency=adjacency,
            config=config,
            universal_upper=0.0 if universal_cover else None,
        )
        if not certificate_eligible:
            audit_gap, audit_upper, audit_comparable = None, math.nan, False
        expected_scores = token_channel[tokens["D_test"]] @ token_scores
        row = _metric_row(
            mechanism="capt_block",
            channel=token_channel,
            weights=objective_weights,
            cost=token_cost,
            target_epsilon=float(config.get("epsilon", 1.0)),
            certified_epsilon=(verification.realized_epsilon if certificate_eligible else None),
            l_count=l_count,
            k=k,
            assignment=assignment,
            certified=certificate_eligible,
        )
        row.update(
            prediction_metrics(frames["D_test"][label].to_numpy(), expected_scores)
        )
        row.update(
            evaluate_target_ctrs(
                frames["D_test"][label].to_numpy(),
                expected_scores,
                list(config.get("target_ctr_list", [0.001, 0.005, 0.01])),
            )
        )
        row.update(
            {
                "profile": profile,
                "lower_audit_epsilon": audit.epsilon_lower,
                "audit_upper_epsilon": audit_upper,
                "witness_matched_upper_epsilon": audit_upper,
                "certificate_global_upper_epsilon": (
                    verification.realized_epsilon if certificate_eligible else math.nan
                ),
                "certificate_audit_gap": audit_gap,
                "audit_certificate_comparable": audit_comparable,
                "audit_events_tested": audit.events_tested,
                "audit_bounds_tested": audit.bounds_tested,
                "audit_per_bound_alpha": audit.per_bound_alpha,
                "audit_candidate_events": audit.candidate_events,
                "audit_candidate_groups": audit.candidate_groups,
                "audit_candidate_contexts": audit.candidate_contexts,
                "audit_day_count": audit_day_count,
                "audit_day_min_epsilon": audit_day_min,
                "audit_day_max_epsilon": audit_day_max,
                "audit_sampling_unit": "user-day privacy epoch",
                "sampling_assumption": config.get("sampling_assumption"),
                "rare_group_mass": hist.rare_group_mass,
                "universal_cover_activated": bool(
                    mechanism_used_fallback and universal_cover
                ),
                "mechanism_fallback_share": 1.0 if mechanism_used_fallback else 0.0,
                "rare_group_count": hist.rare_group_count,
                "missing_group_count": hist.missing_group_count,
                "expected_group_count": hist.expected_group_count,
                "hybrid_connectivity_gaps": connectivity_gaps,
                "group_sample_size": len(cert_frame),
                "solver_status": solver_info["status"],
                "solver_runtime": solver_info["runtime_seconds"],
                "variable_count": solver_info["variable_count"],
                "constraint_count": solver_info["constraint_count"],
                "estimated_memory": solver_info["estimated_memory_bytes"],
                "partition": config.get("partition", "frequency_balanced"),
                "decoder": config.get("decoder", "design_frequency"),
                "phi": config.get("phi", "hash"),
                "fixed_model": config.get("fixed_model", "ctr_model"),
                "distortion": config.get("distortion", "bernoulli_kl"),
                "confidence": confidence_name,
                "tv_radius": float(config.get("shift_tv", 0.0)),
                "seed": int(config.get("seed", 0)),
                "nested_partition": bool(config.get("nested_partitions", False)),
                "contribution_policy": contribution_policy,
                "certificate_width": max(
                    (float(np.max(box.upper - box.lower)) for box in boxes.values()),
                    default=0.0,
                ),
                "alpha_cert": float(config.get("alpha_cert", 0.05)),
                "dp_hist_epsilon": config.get("dp_hist_epsilon"),
                "pi_weighting": config.get("pi_weighting", "empirical"),
                "profile_weighting": config.get("profile_weighting", "global_design"),
                "cost_aggregation": config.get("cost_aggregation", "empirical"),
            }
        )
        all_rows.append(row)
        requested_mechanisms = set(
            config.get(
                "mechanisms",
                [
                    "raw_identity",
                    "common_cover",
                    "k_ary_rr",
                    "scalar_keep_or_cover",
                    "tokenwise_keep_or_cover",
                    "capt_block",
                    "capt_full",
                ],
            )
        )
        baseline_channels = {
            "raw_identity": raw_identity(k),
            "common_cover": common_cover(frequencies),
            "k_ary_rr": k_ary_rr(k, float(config.get("epsilon", 1.0))),
            "scalar_keep_or_cover": scalar_keep_or_cover(
                frequencies,
                min(
                    1.0,
                    math.expm1(float(config.get("epsilon", 1.0)))
                    / (math.expm1(float(config.get("epsilon", 1.0))) + k),
                ),
            ),
            "tokenwise_keep_or_cover": tokenwise_keep_or_cover(
                frequencies,
                np.full(
                    k,
                    min(
                        1.0,
                        float(config.get("epsilon", 1.0))
                        / (1 + float(config.get("epsilon", 1.0))),
                    ),
                ),
            ),
        }
        for baseline_name, baseline_channel in baseline_channels.items():
            if baseline_name not in requested_mechanisms:
                continue
            baseline_attack = audit_frames["D_attack_train"].copy()
            baseline_test = audit_frames["D_test"].copy()
            baseline_attack["__output__"] = _sample_outputs(
                audit_tokens["D_attack_train"], baseline_channel, rng
            )
            baseline_test["__output__"] = _sample_outputs(
                audit_tokens["D_test"], baseline_channel, rng
            )
            baseline_attack["__audit_group__"] = pd.Series(
                list(map(tuple, baseline_attack[attrs].astype(str).to_numpy())),
                index=baseline_attack.index,
            )
            baseline_test["__audit_group__"] = pd.Series(
                list(map(tuple, baseline_test[attrs].astype(str).to_numpy())),
                index=baseline_test.index,
            )
            baseline_audit = _audit_with_context_strata(
                baseline_attack,
                baseline_test,
                group_col="__audit_group__",
                output_col="__output__",
                contexts=contexts,
                alpha=float(config.get("alpha_cert", 0.05)),
            )
            analytic_certificate = baseline_name in {"common_cover", "k_ary_rr"}
            analytic_epsilon = (
                0.0
                if baseline_name == "common_cover"
                else float(config.get("epsilon", 1.0))
                if baseline_name == "k_ary_rr"
                else None
            )
            baseline_gap, baseline_upper, baseline_comparable = (
                _audit_certificate_comparison(
                    baseline_audit,
                    profile=profile,
                    groups=hist.groups,
                    adjacency=adjacency,
                    config=config,
                    universal_upper=analytic_epsilon,
                )
            )
            if not analytic_certificate:
                baseline_gap, baseline_upper, baseline_comparable = None, math.nan, False
            baseline_scores = baseline_channel[tokens["D_test"]] @ token_scores
            baseline_row = _metric_row(
                mechanism=baseline_name,
                channel=baseline_channel,
                weights=objective_weights,
                cost=token_cost,
                target_epsilon=float(config.get("epsilon", 1.0)),
                certified_epsilon=analytic_epsilon,
                l_count=k,
                k=k,
                certified=analytic_certificate,
            )
            baseline_row.update(
                prediction_metrics(
                    frames["D_test"][label].to_numpy(), baseline_scores
                )
            )
            baseline_row.update(
                evaluate_target_ctrs(
                    frames["D_test"][label].to_numpy(),
                    baseline_scores,
                    list(config.get("target_ctr_list", [0.001, 0.005, 0.01])),
                )
            )
            baseline_row.update(
                {
                    key: row[key]
                    for key in [
                        "profile",
                        "rare_group_mass",
                        "rare_group_count",
                        "missing_group_count",
                        "expected_group_count",
                        "hybrid_connectivity_gaps",
                        "group_sample_size",
                        "partition",
                        "decoder",
                        "phi",
                        "fixed_model",
                        "distortion",
                        "confidence",
                        "tv_radius",
                        "seed",
                        "nested_partition",
                        "contribution_policy",
                        "certificate_width",
                        "alpha_cert",
                        "dp_hist_epsilon",
                        "pi_weighting",
                        "profile_weighting",
                        "cost_aggregation",
                        "sampling_assumption",
                        "audit_sampling_unit",
                    ]
                }
            )
            baseline_row.update(
                {
                    "lower_audit_epsilon": baseline_audit.epsilon_lower,
                    "audit_upper_epsilon": baseline_upper,
                    "witness_matched_upper_epsilon": baseline_upper,
                    "certificate_global_upper_epsilon": (
                        analytic_epsilon if analytic_certificate else math.nan
                    ),
                    "certificate_audit_gap": baseline_gap,
                    "audit_certificate_comparable": baseline_comparable,
                    "audit_events_tested": baseline_audit.events_tested,
                    "audit_bounds_tested": baseline_audit.bounds_tested,
                    "audit_per_bound_alpha": baseline_audit.per_bound_alpha,
                    "audit_candidate_events": baseline_audit.candidate_events,
                    "audit_candidate_groups": baseline_audit.candidate_groups,
                    "audit_candidate_contexts": baseline_audit.candidate_contexts,
                    "audit_day_count": math.nan,
                    "audit_day_min_epsilon": math.nan,
                    "audit_day_max_epsilon": math.nan,
                    "universal_cover_activated": math.nan,
                    "mechanism_fallback_share": math.nan,
                    "solver_status": "analytic" if analytic_certificate else "uncertified_baseline",
                    "solver_runtime": 0.0,
                    "variable_count": 0,
                    "constraint_count": 0,
                    "estimated_memory": int(baseline_channel.nbytes),
                }
            )
            all_rows.append(baseline_row)
        if "capt_full" in requested_mechanisms:
            full_limit = int(config.get("full_max_k", 256))
            force_full = bool(config.get("force_full", False))
            full_channel = None
            full_status = "skipped_safety_limit"
            full_runtime = 0.0
            full_variables = k * k
            full_constraints = 0
            full_memory = k * k * 24
            full_certified = False
            full_epsilon = None
            full_verification = None
            full_used_fallback = False
            if k <= full_limit or force_full:
                full_hist = histogram_builder(
                    cert_frame,
                    cert_tokens,
                    profile=profile,
                    context_columns=contexts,
                    alphabet_size=k,
                    min_group_count=int(config.get("min_group_count", 20)),
                    rare_group_policy=config.get("rare_group_policy", "force_cover"),
                    expected_frame=expected_group_frame,
                    include_fallback_levels=True,
                )
                full_adjacency = build_adjacency(
                    full_hist.groups,
                    config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
                    float(config.get("epsilon", 1.0)),
                    config.get("epsilon_by_attr", {}),
                )
                full_boxes = {}
                for key, counts in full_hist.counts.items():
                    kwargs = {
                        "alpha": float(config.get("alpha_cert", 0.05)),
                        "group_count": len(full_hist.counts),
                        "comparisons": max(1, len(full_adjacency)),
                        "tv_radius": float(config.get("shift_tv", 0.0)),
                    }
                    full_boxes[key] = (
                        factory(counts, **kwargs)
                        if confidence_name != "point"
                        else factory(counts)
                    )
                if full_hist.force_cover or not full_adjacency:
                    full_used_fallback = True
                    full_channel = common_cover(frequencies)
                    full_verification = verify_robust_channel(
                        full_channel,
                        full_boxes,
                        full_adjacency,
                        tolerance=float(config.get("solver_tolerance", 1e-8)),
                    )
                    full_status = (
                        "not_optimized_force_cover_policy"
                        if full_hist.force_cover
                        else "cover_no_adjacent_pairs"
                    )
                    full_certified = full_verification.valid and not any(
                        box.experimental for box in full_boxes.values()
                    )
                    full_epsilon = (
                        full_verification.realized_epsilon if full_certified else None
                    )
                else:
                    full_solution, full_verification = solve_robust_block_lp(
                        token_cost,
                        objective_weights,
                        full_boxes,
                        full_adjacency,
                        tolerance=float(config.get("solver_tolerance", 1e-8)),
                        time_limit=config.get("time_limit"),
                    )
                    full_channel = full_solution.channel
                    full_status = full_solution.solver.status
                    full_runtime = full_solution.solver.runtime_seconds
                    full_variables = full_solution.solver.variable_count
                    full_constraints = full_solution.solver.constraint_count
                    full_memory = full_solution.solver.estimated_memory_bytes
                    full_certified = full_verification.valid and not any(
                        box.experimental for box in full_boxes.values()
                    )
                    full_epsilon = (
                        full_verification.realized_epsilon if full_certified else None
                    )
                    if full_channel is None:
                        full_used_fallback = True
            full_row = _metric_row(
                mechanism="capt_full",
                channel=full_channel,
                weights=objective_weights,
                cost=token_cost,
                target_epsilon=float(config.get("epsilon", 1.0)),
                certified_epsilon=full_epsilon,
                l_count=k,
                k=k,
                certified=full_certified,
            )
            full_row.update(
                {
                    key: row[key]
                    for key in [
                        "profile",
                        "rare_group_mass",
                        "rare_group_count",
                        "missing_group_count",
                        "expected_group_count",
                        "hybrid_connectivity_gaps",
                        "group_sample_size",
                        "partition",
                        "decoder",
                        "phi",
                        "fixed_model",
                        "distortion",
                        "confidence",
                        "tv_radius",
                        "seed",
                        "nested_partition",
                        "contribution_policy",
                        "certificate_width",
                        "alpha_cert",
                        "dp_hist_epsilon",
                        "pi_weighting",
                        "profile_weighting",
                        "cost_aggregation",
                        "sampling_assumption",
                        "audit_sampling_unit",
                    ]
                }
            )
            full_row.update(
                {
                    "solver_status": full_status,
                    "solver_runtime": full_runtime,
                    "variable_count": full_variables,
                    "constraint_count": full_constraints,
                    "estimated_memory": full_memory,
                    "full_verification_valid": bool(
                        full_verification is not None and full_verification.valid
                    ),
                    "full_oracle_optimized": bool(
                        full_status == "optimal"
                        and full_channel is not None
                        and full_verification is not None
                        and full_verification.valid
                    ),
                    # Finite-sample token and block boxes are different uncertainty
                    # sets. Do not present that optimized token LP as the matched
                    # Theorem-4 full optimum for a block-box CAPT run.
                    "full_comparison_eligible": False,
                    "universal_cover_activated": bool(
                        full_used_fallback
                        and full_channel is not None
                        and np.allclose(
                            full_channel,
                            full_channel[0][None, :],
                            atol=1e-10,
                            rtol=0,
                        )
                    ),
                    "mechanism_fallback_share": (
                        1.0 if full_used_fallback else 0.0
                    ),
                    "certificate_global_upper_epsilon": (
                        full_epsilon if full_certified else math.nan
                    ),
                }
            )
            if full_channel is not None:
                full_scores = full_channel[tokens["D_test"]] @ token_scores
                full_row.update(
                    prediction_metrics(
                        frames["D_test"][label].to_numpy(), full_scores
                    )
                )
                full_row.update(
                    evaluate_target_ctrs(
                        frames["D_test"][label].to_numpy(),
                        full_scores,
                        list(config.get("target_ctr_list", [0.001, 0.005, 0.01])),
                    )
                )
            all_rows.append(full_row)
        if certificate_eligible:
            certificate = make_certificate(
                config=config,
                channel=block_channel,
                boxes=boxes,
                adjacency=adjacency,
                verification=verification,
                solver=solver_info,
                component_hashes={
                    "encoder": sha256_file(path / "models" / "encoder.joblib"),
                    "model": sha256_file(path / "models" / "reference.joblib"),
                },
                split_identifiers=config["splits"],
                dp_parameters={
                    "epsilon": config.get("dp_hist_epsilon"),
                    "delta": config.get("dp_hist_delta"),
                    "contribution_policy": config.get("contribution_policy", "one-display-per-uuid-day"),
                },
                histogram_counts=hist.counts,
                assignment=assignment,
                decoder=decoder,
                coverage={
                    "expected_group_count": hist.expected_group_count,
                    "observed_group_count": len(hist.groups),
                    "missing_group_count": hist.missing_group_count,
                    "missing_groups": list(hist.missing_groups),
                    "rare_group_count": hist.rare_group_count,
                    "hybrid_connectivity_gaps": connectivity_gaps,
                    "requires_universal_cover": hist.force_cover,
                    "policy": config.get("rare_group_policy", "force_cover"),
                    "profile": profile,
                    "bundle_profiles": list(profiles),
                },
                groups=hist.groups,
            )
            cert_path = path / ("certificate.json" if len(profiles) == 1 else f"certificate-{profile.replace('+', '_')}.json")
            certificate.write(cert_path)
            certificates.append(str(cert_path.name))
    metrics = pd.DataFrame(all_rows)
    protect_profile = config.get("protect_profile", "off")
    if protect_profile not in {False, None, "off"}:
        if protect_profile == "counterfactual":
            diagnostic = evaluate_profile_privacy(
                profile_channels,
                mode="counterfactual",
                common_input_distribution=objective_weights,
            )
        elif protect_profile == "observational":
            assigned = assign_simulated_profiles(
                frames["D_test"],
                profiles=profiles,
                user_col=user_col,
                epoch_col="day_int",
                probabilities=config.get("profile_probs"),
                seed=int(config.get("seed", 0)),
            )
            observed = {}
            for profile_name in profiles:
                selected = assigned.to_numpy() == profile_name
                counts = np.bincount(tokens["D_test"][selected], minlength=k).astype(float)
                observed[profile_name] = (
                    counts / counts.sum() if counts.sum() else objective_weights
                )
            diagnostic = evaluate_profile_privacy(
                profile_channels,
                mode="observational",
                observational_input_distributions=observed,
            )
        else:
            raise ValueError("protect_profile must be off, counterfactual, or observational")
        metrics.loc[metrics["mechanism"] == "capt_block", "profile_privacy_epsilon"] = (
            diagnostic.epsilon
        )
        (path / "profile_privacy.json").write_text(
            json.dumps(asdict(diagnostic), indent=2, default=str) + "\n"
        )
    metrics.to_parquet(path / "metrics.parquet", index=False)
    metrics.to_csv(path / "tables" / "metrics.csv", index=False)
    np.savez_compressed(path / "mechanism" / "capt_block.npz", assignment=assignment, decoder=decoder)
    file_metadata = [
        {
            "path": str(file),
            "size": file.stat().st_size,
            "mtime_ns": file.stat().st_mtime_ns,
            "day_int": day_from_path(file),
            "rows": pq.ParquetFile(file).metadata.num_rows,
        }
        for file in discover_parquet(config["data_root"])
    ]
    (path / "input_files.json").write_text(json.dumps(file_metadata, indent=2) + "\n")
    finish_run(path, {"dataset": "criteo", "split_rows": {name: len(frame) for name, frame in frames.items()}, "certificates": certificates})
    return path, metrics


def run_pipeline(config: dict[str, Any], *, max_rows: int | None = None) -> tuple[Path, pd.DataFrame]:
    if config.get("dataset", "synthetic") == "criteo":
        return run_criteo(config, max_rows=max_rows)
    return run_synthetic(config)
