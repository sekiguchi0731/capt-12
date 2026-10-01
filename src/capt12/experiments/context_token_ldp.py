from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from capt12.certification.artifact import hash_array
from capt12.config import run_id
from capt12.distortions.registry import conditional_utility_cost
from capt12.experiments.context_global_token_ldp import (
    _ldp_max_violation,
    _realized_ldp_epsilon,
    _repair_ldp_uniform,
)
from capt12.mechanisms.lp import lift_block_channel, solve_ldp_block_lp, validate_channel
from capt12.models.reference import TOKEN_REFERENCE_FEATURE_SCHEMA
from capt12.utils.artifacts import sha256_file

_VERSION = 1


def verify_pure_ldp_decimal(channel: np.ndarray, epsilon: float) -> dict[str, Any]:
    """Verify pure epsilon-LDP on the serialized binary64 channel.

    For each output column, comparing its maximum and minimum entry is
    equivalent to checking every ordered pair of input rows.  Decimal.from_float
    preserves the exact stored binary64 values, so this is independent of the
    floating-point diagnostic used by the repair routine.
    """
    matrix = np.asarray(channel, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("pure-LDP verification requires a square channel")
    if not np.isfinite(matrix).all():
        raise ValueError("pure-LDP verification requires finite entries")
    if np.min(matrix) < 0:
        raise ValueError("pure-LDP verification requires nonnegative entries")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("pure-LDP verification requires a finite nonnegative epsilon")
    validate_channel(matrix, tolerance=1e-12)
    with localcontext() as context:
        context.prec = 80
        target_epsilon = Decimal(str(float(epsilon)))
        exp_epsilon = target_epsilon.exp()
        worst = Decimal("-Infinity")
        realized = Decimal(0)
        positive_over_zero = 0
        for output in range(matrix.shape[1]):
            values = [Decimal.from_float(float(value)) for value in matrix[:, output]]
            maximum = max(values)
            minimum = min(values)
            worst = max(worst, maximum - exp_epsilon * minimum)
            if maximum > 0:
                if minimum <= 0:
                    realized = Decimal("Infinity")
                    positive_over_zero += 1
                elif realized.is_finite():
                    realized = max(realized, (maximum / minimum).ln())
    valid = bool(worst <= 0 and realized <= target_epsilon and positive_over_zero == 0)
    return {
        "valid": valid,
        "max_additive_violation": float(worst),
        "realized_epsilon": float(realized),
        "positive_over_zero_output_count": positive_over_zero,
        "arithmetic": "Decimal.from_float(binary64), precision=80",
        "equivalent_ordered_constraints": int(
            matrix.shape[0] * (matrix.shape[0] - 1) * matrix.shape[1]
        ),
    }


def _emit(event: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[context_token_ldp] [{event}] {details}".rstrip(), flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _solver_progress(event: str, fields: dict[str, Any]) -> None:
    if event in {
        "lp_problem_build_finished",
        "lp_solver_started",
        "lp_solver_heartbeat",
        "lp_solver_retry_started",
        "lp_solver_finished",
    }:
        _emit(event, **fields)


def _context_objective_inputs(
    mechanism_run: Path,
    artifacts: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    frozen_path = mechanism_run / "mechanism" / "frozen_design.npz"
    with np.load(frozen_path, allow_pickle=False) as stored:
        frozen = {name: np.asarray(stored[name]).copy() for name in stored.files}
    required = {
        "frequencies",
        "context_levels",
        "token_context_weights",
        "token_context_label_count",
        "token_context_label_sum",
        "reference_feature_schema",
    }
    missing = required - set(frozen)
    if missing:
        raise ValueError(f"frozen design is missing context-token-LDP inputs: {sorted(missing)}")
    if str(frozen["reference_feature_schema"].item()) != TOKEN_REFERENCE_FEATURE_SCHEMA:
        raise ValueError("context-token-LDP requires the categorical-token reference model")

    config = artifacts["resolved"]
    arrays = artifacts["arrays"]
    contexts = arrays["contexts"].astype(str)
    if not len(contexts) or len(set(contexts.tolist())) != len(contexts):
        raise ValueError("serialized public contexts must be nonempty and unique")
    token_count = len(arrays["assignment"])
    if token_count < 1:
        raise ValueError("context-token-LDP requires a nonempty token alphabet")
    if len(config.get("context_cols", ())) != 1:
        raise ValueError("context-token-LDP requires exactly one public context column")
    context_column = str(config["context_cols"][0])
    probability_grid = np.stack(
        [
            artifacts["reference"].predict(
                pd.DataFrame(
                    {
                        "__token__": np.arange(token_count, dtype=int),
                        context_column: context,
                    }
                )
            )
            for context in contexts
        ]
    )
    if (
        probability_grid.shape != (len(contexts), token_count)
        or not np.isfinite(probability_grid).all()
        or np.min(probability_grid) < 0
        or np.max(probability_grid) > 1
    ):
        raise ValueError("reference probability grid is invalid")
    stored_contexts = list(map(str, frozen["context_levels"].tolist()))
    if len(set(stored_contexts)) != len(stored_contexts):
        raise ValueError("frozen context levels must be unique")
    stored_index = {context: index for index, context in enumerate(stored_contexts)}
    stored_weights = np.asarray(frozen["token_context_weights"], dtype=float)
    stored_counts = np.asarray(frozen["token_context_label_count"], dtype=float)
    stored_sums = np.asarray(frozen["token_context_label_sum"], dtype=float)
    expected_shape = (token_count, len(stored_contexts))
    for name, value in (
        ("token_context_weights", stored_weights),
        ("token_context_label_count", stored_counts),
        ("token_context_label_sum", stored_sums),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {expected_shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains a non-finite entry")
    if np.min(stored_weights) < 0:
        raise ValueError("token_context_weights contains a negative entry")
    global_weights = np.asarray(frozen["frequencies"], dtype=float)
    if (
        global_weights.shape != (token_count,)
        or not np.isfinite(global_weights).all()
        or np.min(global_weights) < 0
        or global_weights.sum() <= 0
    ):
        raise ValueError("frozen token frequencies are missing or invalid")
    global_weights = global_weights / global_weights.sum()

    objective = str(config.get("context_utility_objective", "teacher_kl"))
    eta = float(config.get("distortion_clip", 1e-6))
    hybrid_weight = float(config.get("hybrid_empirical_weight", 0.5))
    costs = np.empty((len(contexts), token_count, token_count), dtype=float)
    weights = np.empty((len(contexts), token_count), dtype=float)
    for context_index, context in enumerate(contexts):
        stored_context_index = stored_index.get(str(context))
        if stored_context_index is None:
            label_counts = np.zeros(token_count, dtype=float)
            label_sums = np.zeros(token_count, dtype=float)
            token_weights = global_weights.copy()
        else:
            label_counts = stored_counts[:, stored_context_index]
            label_sums = stored_sums[:, stored_context_index]
            token_weights = stored_weights[:, stored_context_index].copy()
            if token_weights.sum() <= 0:
                token_weights = global_weights.copy()
            else:
                token_weights /= token_weights.sum()
        token_cost, _ = conditional_utility_cost(
            probability_grid[context_index],
            label_counts,
            label_sums,
            objective=objective,
            eta=eta,
            hybrid_empirical_weight=hybrid_weight,
        )
        costs[context_index] = token_cost
        weights[context_index] = token_weights
    if not np.isfinite(costs).all() or not np.isfinite(weights).all():
        raise ValueError("context-token-LDP objective contains a non-finite entry")
    return contexts, costs, weights, objective


def _checkpoint(
    checkpoint_dir: Path,
    index: int,
    context: str,
    objective_hash: str,
    epsilon: float,
    token_count: int,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    channel_path = checkpoint_dir / f"context-{index:02d}.npz"
    metadata_path = checkpoint_dir / f"context-{index:02d}.json"
    if not channel_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(channel_path, allow_pickle=False) as stored:
            channel = np.asarray(stored["channel"], dtype=float).copy()
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    expected = {
        "version": _VERSION,
        "context": context,
        "context_index": index,
        "objective_hash": objective_hash,
        "epsilon": epsilon,
        "K": token_count,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    if channel.shape != (token_count, token_count):
        return None
    try:
        validate_channel(channel, tolerance=1e-12)
    except ValueError:
        return None
    if _ldp_max_violation(channel, epsilon) > -1e-12:
        return None
    exact = verify_pure_ldp_decimal(channel, epsilon)
    if not exact["valid"]:
        return None
    if metadata.get("channel_hash") != hash_array(channel):
        return None
    return channel, metadata


def _completed(output_dir: Path, signature: dict[str, Any]) -> bool:
    metadata_path = output_dir / "context_token_ldp_metadata.json"
    channel_path = output_dir / "context_token_ldp_channels.npz"
    manifest_path = output_dir / "manifest.json"
    if not all(path.is_file() for path in (metadata_path, channel_path, manifest_path)):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    internally_consistent = (
        manifest.get("status") == "complete"
        and metadata.get("status") == "complete"
        and metadata.get("signature") == signature
        and manifest.get("signature") == signature
        and manifest.get("metadata_sha256") == sha256_file(metadata_path)
        and metadata.get("channel_sha256") == sha256_file(channel_path)
        and manifest.get("channel_sha256") == metadata.get("channel_sha256")
    )
    if not internally_consistent:
        return False
    try:
        loaded_contexts, _, _ = load_context_token_ldp_channels(output_dir)
    except (OSError, RuntimeError, ValueError, KeyError):
        return False
    return loaded_contexts.tolist() == list(map(str, signature["contexts"]))


def solve_context_token_ldp_channels(
    mechanism_run: str | Path,
    *,
    output_root: str | Path = "outputs/context_token_ldp_channels",
    solver_time_limit: float = 1800,
) -> Path:
    """Solve and strictly repair one unrestricted KxK LDP channel per public context."""
    if not math.isfinite(solver_time_limit) or solver_time_limit <= 0:
        raise ValueError("context-token-LDP solver time limit must be positive")
    mechanism_run = Path(mechanism_run).resolve()
    # Local import avoids a module-level cycle: fixed-test evaluation invokes this solver.
    from capt12.experiments.context_fixed_test_seeds import _load_fixed_artifacts

    artifacts = _load_fixed_artifacts(mechanism_run, load_runtime_models=False)
    config = artifacts["resolved"]
    metadata = artifacts["metadata"]
    arrays = artifacts["arrays"]
    contexts, costs, weights, objective = _context_objective_inputs(mechanism_run, artifacts)
    token_count = costs.shape[1]
    objective_hash = hash_array(
        np.concatenate(
            [costs.reshape(len(contexts), -1), weights],
            axis=1,
        )
    )
    epsilon = float(config["epsilon"])
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("context-token-LDP requires epsilon > 0")
    requested_repair_margin = float(config.get("certificate_repair_margin", 1e-10))
    if not math.isfinite(requested_repair_margin) or requested_repair_margin <= 0:
        raise ValueError("context-token-LDP repair margin must be finite and positive")
    repair_margin = max(requested_repair_margin, 1e-12)
    signature = {
        "experiment": "context_specific_full_token_ldp",
        "version": _VERSION,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "mechanism_run_id": mechanism_run.name,
        "mechanism_source_git_sha": metadata["source_git_sha"],
        "frozen_design_seed": int(config["frozen_design_seed"]),
        "epsilon": epsilon,
        "K": token_count,
        "utility_objective": objective,
        "objective_hash": objective_hash,
        "contexts": contexts.tolist(),
        "strict_uniform_repair_margin": repair_margin,
    }
    output_dir = Path(output_root) / run_id(signature)
    if _completed(output_dir, signature):
        return output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    channels = np.empty_like(costs)
    context_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, context in enumerate(contexts):
        context_hash = hash_array(np.concatenate([costs[index].reshape(-1), weights[index]]))
        restored = _checkpoint(
            checkpoint_dir,
            index,
            str(context),
            context_hash,
            epsilon,
            token_count,
        )
        if restored is not None:
            channels[index], record = restored
            context_records.append(record)
            _emit(
                "context_reused",
                context=context,
                context_index=index,
                context_count=len(contexts),
            )
            continue

        _emit(
            "context_solve_started",
            context=context,
            context_index=index,
            context_count=len(contexts),
            K=token_count,
            constraints=token_count * (token_count - 1) * token_count,
        )
        solution = solve_ldp_block_lp(
            costs[index],
            weights[index],
            epsilon,
            tolerance=1e-10,
            time_limit=solver_time_limit,
            progress=_solver_progress,
            progress_label=f"context-token-ldp/context={context}",
            heartbeat_seconds=30,
        )
        if solution.channel is None:
            raise RuntimeError(
                f"context-token-LDP solve failed for context {context}: {solution.solver.message}"
            )
        raw_objective = float(np.sum(weights[index, :, None] * solution.channel * costs[index]))
        repaired_block, _, _, _ = _repair_ldp_uniform(
            arrays["ldp_channels"][index],
            epsilon,
            margin=repair_margin,
        )
        lifted_block = lift_block_channel(
            repaired_block,
            arrays["assignment"],
            arrays["decoder"],
        )
        block_comparator_objective = float(
            np.sum(weights[index, :, None] * lifted_block * costs[index])
        )
        objective_tolerance = max(1e-8, 1e-7 * abs(block_comparator_objective))
        if raw_objective > block_comparator_objective + objective_tolerance:
            raise RuntimeError(
                "context-token-LDP optimum is worse than its feasible block-LDP "
                f"comparator for {context}: token={raw_objective}, "
                f"block={block_comparator_objective}"
            )
        channel, repair_lambda, pre_repair, post_repair = _repair_ldp_uniform(
            solution.channel,
            epsilon,
            margin=repair_margin,
        )
        validate_channel(channel, tolerance=1e-12)
        independent_violation = _ldp_max_violation(channel, epsilon)
        if independent_violation > -1e-12:
            raise RuntimeError(
                f"strict context-token-LDP verification failed for {context}: "
                f"{independent_violation}"
            )
        realized_epsilon = _realized_ldp_epsilon(channel)
        if not math.isfinite(realized_epsilon) or realized_epsilon > epsilon + 1e-10:
            raise RuntimeError(
                f"context-token-LDP realized epsilon is invalid for {context}: {realized_epsilon}"
            )
        exact_verification = verify_pure_ldp_decimal(channel, epsilon)
        if not exact_verification["valid"]:
            raise RuntimeError(
                f"Decimal context-token-LDP verification failed for {context}: {exact_verification}"
            )
        repaired_objective = float(np.sum(weights[index, :, None] * channel * costs[index]))
        channels[index] = channel
        record = {
            "version": _VERSION,
            "context": str(context),
            "context_index": index,
            "objective_hash": context_hash,
            "epsilon": epsilon,
            "K": token_count,
            "channel_hash": hash_array(channel),
            "solver": asdict(solution.solver),
            "raw_objective": raw_objective,
            "repaired_objective": repaired_objective,
            "strict_block_ldp_comparator_objective": block_comparator_objective,
            "raw_token_minus_block_objective": (raw_objective - block_comparator_objective),
            "repair_lambda": repair_lambda,
            "pre_repair_max_additive_violation": pre_repair,
            "max_additive_violation": post_repair,
            "independent_max_additive_violation": independent_violation,
            "realized_epsilon": realized_epsilon,
            "decimal_verification": exact_verification,
            "row_sum_max_error": float(np.max(np.abs(channel.sum(axis=1) - 1))),
            "minimum_entry": float(np.min(channel)),
        }
        channel_checkpoint = checkpoint_dir / f"context-{index:02d}.npz"
        metadata_checkpoint = checkpoint_dir / f"context-{index:02d}.json"
        _write_npz(channel_checkpoint, channel=channel)
        _write_json(metadata_checkpoint, record)
        context_records.append(record)
        _emit(
            "context_solve_finished",
            context=context,
            context_index=index,
            solver_seconds=f"{solution.solver.runtime_seconds:.1f}",
            repair_lambda=f"{repair_lambda:.3g}",
            realized_epsilon=f"{realized_epsilon:.12g}",
        )

    channel_path = output_dir / "context_token_ldp_channels.npz"
    _write_npz(channel_path, contexts=contexts, channels=channels)
    result_metadata = {
        "status": "complete",
        "signature": signature,
        "mechanism_run_path": str(mechanism_run),
        "channel_file": channel_path.name,
        "channel_sha256": sha256_file(channel_path),
        "strict_uniform_repair_margin": repair_margin,
        "context_records": context_records,
        "wall_seconds": time.perf_counter() - started,
    }
    metadata_path = output_dir / "context_token_ldp_metadata.json"
    _write_json(metadata_path, result_metadata)
    _write_json(
        output_dir / "manifest.json",
        {
            "status": "complete",
            "signature": signature,
            "channel_sha256": sha256_file(channel_path),
            "metadata_sha256": sha256_file(metadata_path),
        },
    )
    _emit(
        "finished",
        output=output_dir,
        contexts=len(contexts),
        wall_seconds=f"{result_metadata['wall_seconds']:.1f}",
    )
    return output_dir


def load_context_token_ldp_channels(
    artifact_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    artifact_dir = Path(artifact_dir)
    metadata_path = artifact_dir / "context_token_ldp_metadata.json"
    channel_path = artifact_dir / "context_token_ldp_channels.npz"
    manifest_path = artifact_dir / "manifest.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"context-token-LDP artifact metadata is unreadable: {artifact_dir}"
        ) from error
    if metadata.get("status") != "complete" or manifest.get("status") != "complete":
        raise RuntimeError(f"context-token-LDP artifact is incomplete: {artifact_dir}")
    signature = metadata.get("signature")
    if not isinstance(signature, dict) or manifest.get("signature") != signature:
        raise RuntimeError("context-token-LDP manifest signature does not match metadata")
    if manifest.get("metadata_sha256") != sha256_file(metadata_path):
        raise RuntimeError("context-token-LDP metadata hash does not match its manifest")
    try:
        actual_channel_sha256 = sha256_file(channel_path)
    except OSError as error:
        raise RuntimeError("context-token-LDP channel file is unreadable") from error
    if (
        metadata.get("channel_sha256") != actual_channel_sha256
        or manifest.get("channel_sha256") != actual_channel_sha256
    ):
        raise RuntimeError("context-token-LDP channel hash does not match its metadata")
    try:
        with np.load(channel_path, allow_pickle=False) as stored:
            contexts = np.asarray(stored["contexts"]).astype(str)
            channels = np.asarray(stored["channels"], dtype=float).copy()
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError("context-token-LDP channel file is malformed") from error
    expected_contexts = signature.get("contexts")
    if (
        not isinstance(expected_contexts, list)
        or contexts.tolist() != list(map(str, expected_contexts))
        or len(set(contexts.tolist())) != len(contexts)
    ):
        raise RuntimeError("context-token-LDP contexts do not match the signed order")
    try:
        token_count = int(signature["K"])
        epsilon = float(signature["epsilon"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("context-token-LDP signature has invalid K or epsilon") from error
    if (
        channels.ndim != 3
        or channels.shape[0] != len(contexts)
        or channels.shape[1:] != (token_count, token_count)
    ):
        raise RuntimeError("context-token-LDP channel tensor has an invalid shape")
    records = metadata.get("context_records")
    if not isinstance(records, list) or len(records) != len(contexts):
        raise RuntimeError("context-token-LDP context records are incomplete")
    for index, (context, channel, record) in enumerate(
        zip(contexts, channels, records, strict=True)
    ):
        if not isinstance(record, dict) or any(
            (
                record.get("context") != str(context),
                record.get("context_index") != index,
                record.get("K") != token_count,
                record.get("epsilon") != epsilon,
                record.get("channel_hash") != hash_array(channel),
            )
        ):
            raise RuntimeError(f"context-token-LDP record does not match channel {index}")
        try:
            validate_channel(channel, tolerance=1e-12)
        except ValueError as error:
            raise RuntimeError(
                f"context-token-LDP channel {index} is not row-stochastic"
            ) from error
        if _ldp_max_violation(channel, epsilon) > -1e-12:
            raise RuntimeError(f"context-token-LDP channel {index} lacks the signed strict slack")
        verification = verify_pure_ldp_decimal(channel, epsilon)
        if not verification["valid"]:
            raise RuntimeError(f"context-token-LDP channel {index} failed Decimal verification")
    return contexts, channels, metadata
