from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from capt12.certification.robust import VerificationResult, verify_robust_channel
from capt12.comparison.artifacts import (
    RESULT_COLUMNS,
    render_prior_art_figures,
    write_method_contracts,
    write_review_packet,
)
from capt12.comparison.calibration import calibrate_common_cover, design_cover_distribution
from capt12.comparison.certificate import write_factorized_certificate
from capt12.comparison.contracts import default_method_contracts
from capt12.comparison.mass12 import Mass12FiniteChannel, _softmax, fit_mass12_finite
from capt12.comparison.utility import FrozenCTRUtility, evaluate_frozen_ctr_utility
from capt12.config import run_id, validate_config
from capt12.data.loader import load_parquet_sample
from capt12.data.preprocessing import build_group_histograms
from capt12.experiments.context_stratified import (
    _split_histogram_by_context,
    run_context_stratified_diagnostic,
)
from capt12.experiments.fixed_support import _context_values, _expected_frame_from_cartesian
from capt12.experiments.sampling import select_one_display_per_user_day
from capt12.experiments.simplex_completion import _build_boxes
from capt12.pipeline import record_source_provenance
from capt12.privacy.adjacency import AdjacentPair, Group, build_adjacency
from capt12.utils.artifacts import finish_run, prepare_run
from capt12.utils.progress import ProgressLogger, process_memory_bytes


_ORCHESTRATOR_KEYS = {
    "comparison_L_list",
    "comparison_output_dir",
    "comparison_pilot_seeds",
    "cost_list",
    "epsilon_list",
    "formal_comparison_epsilon",
    "prior_art_comparison",
    "seeds",
}


@dataclass(frozen=True)
class PreparedComparisonData:
    design_frame: pd.DataFrame
    design_tokens: np.ndarray
    design_context: np.ndarray
    cert_frame: pd.DataFrame
    cert_tokens: np.ndarray
    test_frame: pd.DataFrame
    test_tokens: np.ndarray
    test_context: np.ndarray
    test_labels: np.ndarray
    test_user_days: np.ndarray
    service_probabilities: dict[str, np.ndarray]
    representation_token_cost: np.ndarray


class FactorizedChannels(Mapping[str, np.ndarray]):
    """Lazy exact Q mapping that keeps at most one K-by-K context channel live."""

    def __init__(
        self,
        input_to_block: Mapping[str, np.ndarray],
        decoder: np.ndarray,
        *,
        cover_distribution: np.ndarray | None = None,
        mixing_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.input_to_block = {
            str(context): np.asarray(value, dtype=float)
            for context, value in input_to_block.items()
        }
        self.decoder = np.asarray(decoder, dtype=float)
        output_count = self.decoder.shape[1]
        self.cover = (
            np.ones(output_count, dtype=float) / output_count
            if cover_distribution is None
            else np.asarray(cover_distribution, dtype=float)
        )
        self.mixing = {
            context: float((mixing_weights or {}).get(context, 0.0))
            for context in self.input_to_block
        }

    def __getitem__(self, context: str) -> np.ndarray:
        key = str(context)
        weight = self.mixing[key]
        raw = self.input_to_block[key] @ self.decoder
        return (1 - weight) * raw + weight * self.cover[None, :]

    def __iter__(self) -> Iterator[str]:
        return iter(self.input_to_block)

    def __len__(self) -> int:
        return len(self.input_to_block)


def _completed(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _native_config(
    config: dict[str, Any],
    output_dir: Path,
    *,
    cost: str,
    block_count: int,
    seed: int,
    epsilon: float,
) -> dict[str, Any]:
    result = {
        key: value
        for key, value in config.items()
        if key not in _ORCHESTRATOR_KEYS and not key.startswith("mass_")
    }
    result.update(
        {
            "output_dir": str(output_dir),
            "context_utility_objective": cost,
            "context_representation_mode": "objective_aligned",
            "context_designs": [f"joint_kmedoids_cost_medoid_L{block_count}"],
            "frozen_design_seed": int(seed),
            "seed": int(seed),
            "epsilon": float(epsilon),
            "K": 4096,
            "L": int(block_count),
        }
    )
    return validate_config(result)


def _native_run(
    config: dict[str, Any],
    output_dir: Path,
    *,
    cost: str,
    block_count: int,
    seed: int,
    epsilon: float,
) -> Path:
    cell = _native_config(
        config,
        output_dir,
        cost=cost,
        block_count=block_count,
        seed=seed,
        epsilon=epsilon,
    )
    expected = dict(cell)
    expected.update(
        {
            "require_clean_worktree": True,
            "source_worktree_clean": True,
            "source_git_sha": config["source_git_sha"],
        }
    )
    path = Path(cell["output_dir"]) / run_id(expected)
    return path if _completed(path) else run_context_stratified_diagnostic(cell)


def _columns(config: dict[str, Any]) -> list[str]:
    profile = config["profiles"][0]
    return list(
        dict.fromkeys(
            [
                config.get("id_col", "id"),
                config.get("user_col", "user_id"),
                config.get("label_col", "is_clicked"),
                *profile.split("+"),
                *config.get("sensitive_cols", []),
                *config.get("context_cols", []),
                *config.get("phi_source_cols", []),
            ]
        )
    )


def _probability_grid(
    reference: Any,
    contexts: list[str],
    context_column: str,
    k: int,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for context in contexts:
        result[context] = np.asarray(
            reference.predict(
                pd.DataFrame(
                    {
                        "__token__": np.arange(k, dtype=int),
                        context_column: context,
                    }
                )
            ),
            dtype=float,
        )
    return result


def _prepare_data(config: dict[str, Any], native_path: Path) -> PreparedComparisonData:
    mapper = joblib.load(native_path / "models" / "category_mapper.joblib")
    encoder = joblib.load(native_path / "models" / "encoder.joblib")
    reference = joblib.load(native_path / "models" / "reference.joblib")
    columns = _columns(config)
    contexts = list(config.get("context_cols", []))
    context_column = contexts[0]
    id_col = config.get("id_col", "id")
    user_col = config.get("user_col", "user_id")
    label_col = config.get("label_col", "is_clicked")

    design_frame = mapper.transform(
        load_parquet_sample(
            data_root=config["data_root"],
            columns=columns,
            days=config["splits"]["D_design"],
        )
    )
    design_tokens = encoder.transform(design_frame)
    design_context = _context_values(design_frame, contexts).astype(str).to_numpy()

    cert_source = load_parquet_sample(
        data_root=config["data_root"],
        columns=columns,
        days=config["splits"]["D_cert"],
    )
    cert_source = select_one_display_per_user_day(
        cert_source,
        user_col=user_col,
        day_col="day_int",
        id_col=id_col,
    )
    cert_frame = mapper.transform(cert_source)
    cert_tokens = encoder.transform(cert_frame)

    test_frame = mapper.transform(
        load_parquet_sample(
            data_root=config["data_root"],
            columns=columns,
            days=config["splits"]["D_test"],
        )
    )
    test_tokens = encoder.transform(test_frame)
    test_context = _context_values(test_frame, contexts).astype(str).to_numpy()
    test_user_days = (
        test_frame["day_int"].astype(str) + "|" + test_frame[user_col].astype(str)
    ).to_numpy()
    with np.load(native_path / "mechanism" / "frozen_design.npz") as frozen:
        context_values = sorted(
            set(map(str, frozen["context_levels"].tolist())).union(np.unique(test_context))
        )
        representation_token_cost = np.asarray(
            frozen["representation_token_cost"], dtype=float
        ).copy()
    return PreparedComparisonData(
        design_frame=design_frame,
        design_tokens=np.asarray(design_tokens, dtype=int),
        design_context=design_context,
        cert_frame=cert_frame,
        cert_tokens=np.asarray(cert_tokens, dtype=int),
        test_frame=test_frame,
        test_tokens=np.asarray(test_tokens, dtype=int),
        test_context=test_context,
        test_labels=test_frame[label_col].to_numpy(dtype=int),
        test_user_days=test_user_days,
        service_probabilities=_probability_grid(reference, context_values, context_column, 4096),
        representation_token_cost=representation_token_cost,
    )


def _boxes_by_context(
    config: dict[str, Any],
    native_path: Path,
    data: PreparedComparisonData,
    *,
    target_epsilon: float,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[AdjacentPair]],
    dict[str, list[Group]],
]:
    support = json.loads((native_path / "frozen_support.json").read_text(encoding="utf-8"))
    expected = _expected_frame_from_cartesian(
        {tuple(map(str, value)) for value in support["G_cart"]},
        profile=config["profiles"][0],
        context=config["context_cols"][0],
    )
    hist = build_group_histograms(
        data.cert_frame,
        data.cert_tokens,
        profile=config["profiles"][0],
        context_columns=config.get("context_cols", []),
        alphabet_size=4096,
        min_group_count=int(config.get("min_group_count", 20)),
        rare_group_policy=config.get("rare_group_policy", "confidence_box"),
        missing_group_policy=config.get("missing_group_policy", "full_simplex"),
        expected_frame=expected,
        include_fallback_levels=False,
    )
    split = _split_histogram_by_context(hist.groups, hist.counts)
    boxes: dict[str, dict[str, Any]] = {}
    adjacency: dict[str, list[AdjacentPair]] = {}
    groups: dict[str, list[Group]] = {}
    context_count = len(split)
    for context, (context_groups, counts) in split.items():
        pairs = build_adjacency(
            context_groups,
            config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
            float(target_epsilon),
            config.get("epsilon_by_attr", {}),
        )
        cell_config = dict(config)
        cell_config["alpha_cert"] = float(config.get("alpha_cert", 0.05)) / context_count
        boxes[context] = _build_boxes(
            counts,
            adjacency_count=len(pairs),
            config=cell_config,
        )
        adjacency[context] = pairs
        groups[context] = context_groups
    return boxes, adjacency, groups


def _token_channels(
    block_channels: np.ndarray,
    contexts: np.ndarray,
    assignment: np.ndarray,
    decoder: np.ndarray,
) -> FactorizedChannels:
    factors = {
        str(context): channel[assignment]
        for context, channel in zip(contexts, block_channels, strict=True)
    }
    return FactorizedChannels(factors, decoder)


def _channel_sha256(channels: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for context in sorted(channels):
        channel = channels[context]
        digest.update(context.encode())
        digest.update(np.ascontiguousarray(channel, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _evaluate(
    channels: Mapping[str, np.ndarray],
    data: PreparedComparisonData,
    config: dict[str, Any],
    *,
    seed: int,
) -> FrozenCTRUtility:
    return evaluate_frozen_ctr_utility(
        channels,
        data.test_tokens,
        data.test_context,
        data.test_labels,
        data.service_probabilities,
        data.test_user_days,
        distortion_cost=data.representation_token_cost,
        hybrid_empirical_weight=float(config.get("hybrid_empirical_weight", 0.5)),
        bootstrap_replicates=int(config.get("utility_cluster_bootstrap_replicates", 2000)),
        bootstrap_seed=int(seed),
        split_name="D_test",
    )


def _empty_row() -> dict[str, Any]:
    row: dict[str, Any] = {name: math.nan for name in RESULT_COLUMNS}
    for name in (
        "certificate_valid",
        "constant_channel",
        "formal_comparable",
        "deployable_under_capt",
        "uses_sensitive_value_online",
        "oracle",
    ):
        row[name] = False
    row["exclusion_reason"] = ""
    row["certificate_path"] = ""
    row["lower_audit_path"] = ""
    row["worst_witness"] = ""
    return row


def _utility_row(row: dict[str, Any], utility: FrozenCTRUtility) -> None:
    row.update(
        {
            "empirical_log_loss": utility.empirical_log_loss,
            "excess_ctr_log_loss": utility.excess_log_loss,
            "excess_ctr_log_loss_ci95_low": utility.excess_log_loss_ci95_low,
            "excess_ctr_log_loss_ci95_high": utility.excess_log_loss_ci95_high,
            "teacher_kl": utility.teacher_kl,
            "hybrid_objective": utility.hybrid_objective,
            "roc_auc": utility.roc_auc,
            "pr_auc": utility.pr_auc,
            "distortion_objective": utility.distortion_objective,
        }
    )


def _result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    def normalized(value: Any) -> Any:
        try:
            return None if pd.isna(value) else value
        except (TypeError, ValueError):
            return value

    return tuple(
        normalized(row.get(name))
        for name in (
            "method",
            "cost",
            "seed",
            "mass_seed",
            "target_epsilon",
            "L",
            "mass_m",
            "mass_n",
            "mass_privacy_weight",
            "mass_utility_weight",
            "mass_temperature",
        )
    )


def _write_checkpoint(records: dict[tuple[Any, ...], dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(records.values(), columns=RESULT_COLUMNS)
    if len(frame):
        frame = frame.sort_values(
            ["cost", "L", "seed", "target_epsilon", "method"], kind="stable"
        )
    frame.to_csv(path, index=False)
    return frame


def _native_rows(
    config: dict[str, Any],
    comparison_path: Path,
    native_path: Path,
    data: PreparedComparisonData,
    *,
    cost: str,
    block_count: int,
    seed: int,
    target_epsilon: float,
) -> list[dict[str, Any]]:
    design_name = f"joint_kmedoids_cost_medoid_L{block_count}"
    with np.load(
        native_path / "mechanism" / f"context_channels-{design_name}.npz"
    ) as payload:
        contexts = np.asarray(payload["contexts"], dtype=str).copy()
        assignment = np.asarray(payload["assignment"], dtype=int).copy()
        decoder = np.asarray(payload["decoder"], dtype=float).copy()
        capt_block = np.asarray(payload["channels"], dtype=float).copy()
        ldp_block = np.asarray(payload["ldp_channels"], dtype=float).copy()
    methods = {
        "capt": ("CAPT-12", capt_block),
        "optimal_ldp": ("optimal LDP", ldp_block),
    }
    boxes, adjacency, groups = _boxes_by_context(
        config, native_path, data, target_epsilon=target_epsilon
    )
    rows: list[dict[str, Any]] = []
    for method, (display_name, block_channels) in methods.items():
        token_channels = _token_channels(block_channels, contexts, assignment, decoder)
        utility = _evaluate(token_channels, data, config, seed=seed)
        if method == "capt":
            summary = pd.read_csv(native_path / "tables" / "certificate_summary.csv")
            valid = bool(
                summary["certificate_valid"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
                .all()
            )
            realized = float(summary["certificate_conservative_realized_epsilon"].max())
            certificate_path = str(native_path)
        else:
            factors = {
                str(context): block_channel[assignment]
                for context, block_channel in zip(contexts, block_channels, strict=True)
            }
            certificate_path_obj = (
                comparison_path
                / "certificates"
                / f"ldp-{cost}-L{block_count}-seed{seed}-eps{target_epsilon:g}.json"
            )
            certificate_path_obj, verification = write_factorized_certificate(
                certificate_path_obj,
                input_to_block=factors,
                decoder=decoder,
                cover_distribution=np.ones(4096, dtype=float) / 4096,
                mixing_weights={context: 0.0 for context in factors},
                boxes=boxes,
                adjacency=adjacency,
                groups=groups,
                adjacency_spec={
                    "mode": config.get(
                        "adjacency", config.get("privacy_scope", "tuple_adjacent")
                    ),
                    "epsilon": target_epsilon,
                    "epsilon_by_attr": config.get("epsilon_by_attr", {}),
                },
                metadata={
                    "method": method,
                    "source_git_sha": config["source_git_sha"],
                    "native_run": str(native_path),
                    "target_epsilon": target_epsilon,
                },
            )
            valid = verification.valid
            realized = verification.realized_epsilon
            certificate_path = str(certificate_path_obj)
        row = _empty_row()
        row.update(
            {
                "method": method,
                "display_name": display_name,
                "cost": cost,
                "seed": seed,
                "mass_seed": math.nan,
                "target_epsilon": target_epsilon,
                "L": block_count,
                "raw_upper_epsilon": realized,
                "certified_upper_epsilon": realized,
                "certificate_valid": valid,
                "constant_channel": bool(
                    all(
                        np.allclose(value, value[0][None, :], atol=1e-10, rtol=0)
                        for value in token_channels.values()
                    )
                ),
                "formal_comparable": valid,
                "deployable_under_capt": True,
                "uses_sensitive_value_online": False,
                "oracle": False,
                "channel_sha256": _channel_sha256(token_channels),
                "certificate_path": certificate_path,
            }
        )
        _utility_row(row, utility)
        rows.append(row)
    rows[1]["paired_excess_log_loss_difference_vs_capt"] = (
        rows[1]["excess_ctr_log_loss"] - rows[0]["excess_ctr_log_loss"]
    )
    return rows


def _mass_history(model: Mass12FiniteChannel, path: Path) -> None:
    rows = []
    for value in model.history:
        rows.append(
            {
                "epoch": value.epoch,
                "loss": value.loss,
                "distortion": value.distortion,
                "sensitive_mi_nats": json.dumps(value.sensitive_mi_nats, sort_keys=True),
                "useful_mi_nats": json.dumps(value.useful_mi_nats, sort_keys=True),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _verification_summary(
    channels: Mapping[str, np.ndarray],
    boxes: dict[str, dict[str, Any]],
    adjacency: dict[str, list[AdjacentPair]],
) -> tuple[float, dict[str, VerificationResult]]:
    results = {
        context: verify_robust_channel(channels[context], boxes[context], adjacency[context])
        for context in sorted(channels)
    }
    return max((value.realized_epsilon for value in results.values()), default=0.0), results


def _mass_rows(
    config: dict[str, Any],
    comparison_path: Path,
    native_path: Path,
    data: PreparedComparisonData,
    common_cover: np.ndarray,
    boxes: dict[str, dict[str, Any]],
    adjacency: dict[str, list[AdjacentPair]],
    groups: dict[str, list[Group]],
    *,
    cost: str,
    block_count: int,
    frozen_seed: int,
    mass_seed: int,
    loss_m: float,
    loss_n: float,
    privacy_weight: float,
    utility_weight: float,
    temperature: float,
) -> list[dict[str, Any]]:
    design_name = f"joint_kmedoids_cost_medoid_L{block_count}"
    with np.load(
        native_path / "mechanism" / f"context_channels-{design_name}.npz"
    ) as payload:
        decoder = np.asarray(payload["decoder"], dtype=float).copy()
    formal_epsilon = float(config.get("formal_comparison_epsilon", 1.0))
    contexts = tuple(sorted(boxes))
    sensitive = {
        attribute: data.design_frame[attribute].astype(str).to_numpy()
        for attribute in config["profiles"][0].split("+")
    }
    useful = {"click": data.design_frame[config.get("label_col", "is_clicked")].to_numpy()}
    model = fit_mass12_finite(
        data.design_tokens,
        data.design_context,
        sensitive,
        decoder,
        data.representation_token_cost,
        useful_labels=useful,
        loss_m=loss_m,
        loss_n=loss_n,
        privacy_weight=privacy_weight,
        utility_weight=utility_weight,
        seed=mass_seed,
        epochs=int(config.get("mass_epochs", 100)),
        learning_rate=float(config.get("mass_learning_rate", 0.05)),
        temperature=temperature,
        source_split="D_design",
        objective_name=cost,
        frozen_context_values=contexts,
    )
    stem = (
        f"mass-{cost}-L{block_count}-fseed{frozen_seed}-mseed{mass_seed}"
        f"-m{loss_m:g}-n{loss_n:g}-pw{privacy_weight:g}-uw{utility_weight:g}-t{temperature:g}"
    )
    model_path = comparison_path / "models" / f"{stem}.npz"
    probabilities = {
        context: _softmax(model.logits[index], model.temperature)
        for index, context in enumerate(model.context_values)
    }
    np.savez_compressed(
        model_path,
        contexts=np.asarray(model.context_values, dtype=str),
        logits=model.logits,
        decoder=model.decoder,
        temperature=np.asarray(model.temperature),
    )
    _mass_history(model, comparison_path / "tables" / f"{stem}-history.csv")

    raw_channels = FactorizedChannels(probabilities, decoder)
    raw_upper, raw_results = _verification_summary(raw_channels, boxes, adjacency)
    mixing: dict[str, float] = {}
    calibration_provenance: dict[str, dict[str, object]] = {}
    calibration_tv: dict[str, float] = {}
    frequencies_by_context = {
        context: np.bincount(
            data.design_tokens[data.design_context == context], minlength=4096
        ).astype(float)
        for context in contexts
    }
    global_frequencies = np.bincount(data.design_tokens, minlength=4096).astype(float)
    for context in contexts:
        weights = frequencies_by_context[context]
        if weights.sum() == 0:
            weights = global_frequencies
        calibration = calibrate_common_cover(
            raw_channels[context],
            common_cover,
            boxes[context],
            adjacency[context],
            target_epsilon=formal_epsilon,
            bisection_tolerance=float(config.get("cover_bisection_tolerance", 1e-8)),
            verifier_tolerance=float(config.get("solver_tolerance", 1e-8)),
            input_weights=weights,
            cost=data.representation_token_cost,
        )
        mixing[context] = calibration.mixing_weight
        calibration_provenance[context] = calibration.provenance()
        calibration_tv[context] = calibration.max_row_tv_change
    calibrated_channels = FactorizedChannels(
        probabilities,
        decoder,
        cover_distribution=common_cover,
        mixing_weights=mixing,
    )
    calibrated_upper, _ = _verification_summary(calibrated_channels, boxes, adjacency)

    raw_certificate = ""
    raw_valid = False
    if math.isfinite(raw_upper):
        claimed = max(raw_upper, 0.0) + 1e-12
        raw_adjacency = {
            context: [replace(pair, epsilon=claimed) for pair in pairs]
            for context, pairs in adjacency.items()
        }
        raw_path, raw_verification = write_factorized_certificate(
            comparison_path / "certificates" / f"{stem}-raw.json",
            input_to_block=probabilities,
            decoder=decoder,
            cover_distribution=common_cover,
            mixing_weights={context: 0.0 for context in contexts},
            boxes=boxes,
            adjacency=raw_adjacency,
            groups=groups,
            adjacency_spec={
                "mode": config.get(
                    "adjacency", config.get("privacy_scope", "tuple_adjacent")
                ),
                "epsilon": claimed,
                "epsilon_by_attr": {},
            },
            metadata={
                **model.provenance(),
                "variant": "raw",
                "source_git_sha": config["source_git_sha"],
                "claimed_epsilon": claimed,
                "measured_upper_epsilon": raw_upper,
                "model_path": str(model_path),
            },
        )
        raw_certificate = str(raw_path)
        raw_valid = raw_verification.valid
    calibrated_path, calibrated_verification = write_factorized_certificate(
        comparison_path / "certificates" / f"{stem}-calibrated.json",
        input_to_block=probabilities,
        decoder=decoder,
        cover_distribution=common_cover,
        mixing_weights=mixing,
        boxes=boxes,
        adjacency=adjacency,
        groups=groups,
        adjacency_spec={
            "mode": config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
            "epsilon": formal_epsilon,
            "epsilon_by_attr": config.get("epsilon_by_attr", {}),
        },
        metadata={
            **model.provenance(),
            "variant": "common-cover calibrated",
            "source_git_sha": config["source_git_sha"],
            "target_epsilon": formal_epsilon,
            "model_path": str(model_path),
            "calibrations": {
                context: calibration_provenance[context] for context in contexts
            },
        },
    )
    raw_utility = _evaluate(raw_channels, data, config, seed=frozen_seed)
    calibrated_utility = _evaluate(
        calibrated_channels, data, config, seed=frozen_seed
    )
    rows: list[dict[str, Any]] = []
    for method, display_name, channels, utility, certified, valid, certificate in (
        (
            "mass12_raw",
            "MaSS-12 (finite-output adaptation)",
            raw_channels,
            raw_utility,
            raw_upper,
            raw_valid,
            raw_certificate,
        ),
        (
            "mass12_calibrated",
            "MaSS-12 + certified cover calibration",
            calibrated_channels,
            calibrated_utility,
            calibrated_upper,
            calibrated_verification.valid,
            str(calibrated_path),
        ),
    ):
        row = _empty_row()
        row.update(
            {
                "method": method,
                "display_name": display_name,
                "cost": cost,
                "seed": frozen_seed,
                "mass_seed": mass_seed,
                "target_epsilon": formal_epsilon,
                "L": block_count,
                "mass_m": loss_m,
                "mass_n": loss_n,
                "mass_privacy_weight": privacy_weight,
                "mass_utility_weight": utility_weight,
                "mass_temperature": temperature,
                "raw_upper_epsilon": raw_upper,
                "certified_upper_epsilon": certified,
                "certificate_valid": valid,
                "cover_lambda": (0.0 if method == "mass12_raw" else max(mixing.values())),
                "cover_row_tv": (
                    0.0
                    if method == "mass12_raw"
                    else max(calibration_tv.values())
                ),
                "constant_channel": bool(
                    all(
                        np.allclose(value, value[0][None, :], atol=1e-10, rtol=0)
                        for value in channels.values()
                    )
                ),
                # VerificationResult exposes the worst constraint and aggregate
                # count checked, but not the exact number of positive violations.
                # Do not substitute a context count for a constraint count.
                "violating_constraint_count": math.nan,
                "worst_witness": json.dumps(
                    {
                        context: result.worst_case
                        for context, result in raw_results.items()
                        if result.worst_case is not None
                    },
                    sort_keys=True,
                ),
                "formal_comparable": valid,
                "deployable_under_capt": True,
                "uses_sensitive_value_online": False,
                "oracle": False,
                "channel_sha256": _channel_sha256(channels),
                "certificate_path": certificate,
            }
        )
        _utility_row(row, utility)
        rows.append(row)
    return rows


def _phase_values(
    config: dict[str, Any], phase: str
) -> tuple[list[str], list[int], list[int], list[float]]:
    costs = list(config.get("cost_list", ["empirical_logloss"]))
    seeds = list(map(int, config.get("seeds", [0])))
    formal = float(config.get("formal_comparison_epsilon", 1.0))
    if phase == "pilot":
        pilot_seeds = list(map(int, config.get("comparison_pilot_seeds", seeds[:1])))
        return costs, [16], pilot_seeds, [formal]
    if phase != "full":
        raise ValueError("phase must be pilot or full")
    block_counts = list(map(int, config.get("comparison_L_list", [8, 16, 32])))
    epsilons = list(map(float, config.get("epsilon_list", [formal])))
    if formal not in epsilons:
        epsilons.append(formal)
    return costs, sorted(set(block_counts)), seeds, sorted(set(epsilons))


def run_prior_art_comparison(
    config: dict[str, Any],
    *,
    phase: str = "pilot",
    resume: bool = True,
) -> Path:
    """Run native CAPT/LDP cells and the D_design-trained MaSS adaptation.

    This is the only supported runner for `prior_art_comparison: true` configs.
    It deliberately does not dispatch through the ordinary CAPT grid.
    """
    config = record_source_provenance(validate_config(config))
    config = dict(config)
    publication = Path(config.get("comparison_output_dir", "outputs/prior_art_comparison"))
    if phase == "full" and bool(config.get("require_mass_pilot_for_full", True)):
        pilot_summary_path = publication / "pilot_summary.json"
        if not pilot_summary_path.is_file():
            raise ValueError(
                "full comparison requires a completed pilot_summary.json; run --phase pilot first"
            )
        pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8"))
        if pilot_summary.get("source_git_sha") != config["source_git_sha"]:
            raise ValueError("full comparison and MaSS pilot must use the same source Git SHA")
        if not bool(pilot_summary.get("full_grid_allowed")):
            raise ValueError(
                "MaSS pilot did not show the predeclared raw-epsilon variation required "
                "for the full grid"
            )
    config["comparison_phase"] = phase
    config["output_dir"] = str(
        publication / "runs"
    )
    path = prepare_run(config)
    progress = ProgressLogger(path, name=path.name)
    progress.emit_environment()
    started = time.perf_counter()
    costs, block_counts, seeds, epsilons = _phase_values(config, phase)
    progress.emit(
        "prior_art_comparison_started",
        phase=phase,
        costs=costs,
        block_counts=block_counts,
        seeds=seeds,
        epsilon_values=epsilons,
        **process_memory_bytes(),
    )

    checkpoint = path / "tables" / "results.csv"
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    if resume and checkpoint.is_file():
        previous = pd.read_csv(checkpoint)
        for raw in previous.to_dict(orient="records"):
            records[_result_key(raw)] = raw

    native_paths: dict[tuple[str, int, int, float], Path] = {}
    current_data_key: tuple[str, int, int] | None = None
    current_data: PreparedComparisonData | None = None
    for cost, block_count, seed, epsilon in product(costs, block_counts, seeds, epsilons):
        progress.emit(
            "native_cell_started",
            cost=cost,
            L=block_count,
            seed=seed,
            epsilon=epsilon,
        )
        native = _native_run(
            config,
            path / "native_runs",
            cost=cost,
            block_count=block_count,
            seed=seed,
            epsilon=epsilon,
        )
        native_paths[(cost, block_count, seed, epsilon)] = native
        data_key = (cost, block_count, seed)
        if data_key != current_data_key:
            current_data = _prepare_data(config, native)
            current_data_key = data_key
        if current_data is None:
            raise RuntimeError("comparison data cache was not initialized")
        data = current_data
        rows = _native_rows(
            config,
            path,
            native,
            data,
            cost=cost,
            block_count=block_count,
            seed=seed,
            target_epsilon=epsilon,
        )
        for row in rows:
            records[_result_key(row)] = row
        _write_checkpoint(records, checkpoint)
        progress.emit("native_cell_finished", run=str(native), result_rows=len(rows))

    current_data = None
    formal = float(config.get("formal_comparison_epsilon", 1.0))
    reference_data = _prepare_data(
        config,
        native_paths[(costs[0], block_counts[0], seeds[0], formal)],
    )
    common_cover = design_cover_distribution(
        np.bincount(reference_data.design_tokens, minlength=4096),
        floor=float(config.get("cover_probability_floor", 1e-12)),
        source_split="D_design",
    )
    reference_data = None
    np.save(path / "mechanism" / "common_token_cover.npy", common_cover)
    (path / "mechanism" / "common_token_cover.json").write_text(
        json.dumps(
            {
                "source_split": "D_design",
                "reference_frozen_design_seed": seeds[0],
                "shared_across_methods_and_seeds": True,
                "sha256": hashlib.sha256(common_cover.tobytes()).hexdigest(),
                "floor": float(config.get("cover_probability_floor", 1e-12)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    prefix = "mass_pilot_" if phase == "pilot" else "mass_"
    mass_grid = product(
        config.get(f"{prefix}seed_list", config.get("mass_seed_list", [0])),
        config.get(f"{prefix}m_list", config.get("mass_m_list", [0.0])),
        config.get(f"{prefix}n_list", config.get("mass_n_list", [0.0])),
        config.get(
            f"{prefix}privacy_weight_list",
            config.get("mass_privacy_weight_list", [1.0]),
        ),
        config.get(
            f"{prefix}utility_weight_list",
            config.get("mass_utility_weight_list", [1.0]),
        ),
        config.get(
            f"{prefix}temperature_list",
            config.get("mass_temperature_list", [1.0]),
        ),
    )
    mass_controls = list(mass_grid)
    for cost, block_count, frozen_seed in product(costs, block_counts, seeds):
        native = native_paths[(cost, block_count, frozen_seed, formal)]
        data = _prepare_data(config, native)
        boxes, adjacency, groups = _boxes_by_context(
            config,
            native,
            data,
            target_epsilon=formal,
        )
        for mass_seed, loss_m, loss_n, privacy_weight, utility_weight, temperature in mass_controls:
            probe = _empty_row()
            probe.update(
                {
                    "method": "mass12_calibrated",
                    "cost": cost,
                    "seed": frozen_seed,
                    "mass_seed": mass_seed,
                    "target_epsilon": formal,
                    "L": block_count,
                    "mass_m": loss_m,
                    "mass_n": loss_n,
                    "mass_privacy_weight": privacy_weight,
                    "mass_utility_weight": utility_weight,
                    "mass_temperature": temperature,
                }
            )
            if resume and _result_key(probe) in records:
                continue
            progress.emit(
                "mass_fit_started",
                cost=cost,
                L=block_count,
                frozen_seed=frozen_seed,
                mass_seed=mass_seed,
                loss_m=loss_m,
                loss_n=loss_n,
                privacy_weight=privacy_weight,
                utility_weight=utility_weight,
                temperature=temperature,
            )
            rows = _mass_rows(
                config,
                path,
                native,
                data,
                common_cover,
                boxes,
                adjacency,
                groups,
                cost=cost,
                block_count=block_count,
                frozen_seed=frozen_seed,
                mass_seed=int(mass_seed),
                loss_m=float(loss_m),
                loss_n=float(loss_n),
                privacy_weight=float(privacy_weight),
                utility_weight=float(utility_weight),
                temperature=float(temperature),
            )
            for row in rows:
                records[_result_key(row)] = row
            _write_checkpoint(records, checkpoint)
            progress.emit("mass_fit_finished", result_rows=len(rows), **process_memory_bytes())

    frame = _write_checkpoint(records, checkpoint)
    publication.mkdir(parents=True, exist_ok=True)
    frame.to_csv(publication / "results.csv", index=False)
    if phase == "pilot":
        raw = frame.loc[
            (frame["method"] == "mass12_raw")
            & np.isfinite(frame["raw_upper_epsilon"].astype(float))
        ]
        ranges = {
            str(cost): float(group["raw_upper_epsilon"].max() - group["raw_upper_epsilon"].min())
            for cost, group in raw.groupby("cost", sort=True)
        }
        threshold = float(config.get("mass_pilot_min_epsilon_range", 1e-6))
        full_allowed = bool(ranges) and set(ranges) == set(costs) and all(
            value > threshold for value in ranges.values()
        )
        (publication / "pilot_summary.json").write_text(
            json.dumps(
                {
                    "source_git_sha": config["source_git_sha"],
                    "raw_epsilon_range_by_cost": ranges,
                    "minimum_required_range": threshold,
                    "full_grid_allowed": full_allowed,
                    "rule": "every configured cost must show finite raw achieved-epsilon variation",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    contracts = default_method_contracts()
    write_method_contracts(contracts, path / "method_contracts.json")
    write_method_contracts(contracts, publication / "method_contracts.json")
    figure_paths = render_prior_art_figures(frame, contracts, publication / "figures")
    write_review_packet(
        publication,
        {
            "source_git_sha": config["source_git_sha"],
            "comparison_run": str(path),
            "phase": phase,
            "figure_count": len(figure_paths),
        },
        exclude_top_level=("runs",),
    )
    finish_run(
        path,
        {
            "source_git_sha": config["source_git_sha"],
            "phase": phase,
            "result_rows": len(frame),
            "wall_seconds": time.perf_counter() - started,
        },
    )
    progress.emit(
        "prior_art_comparison_finished",
        result_rows=len(frame),
        publication_path=str(publication),
        wall_seconds=time.perf_counter() - started,
        **process_memory_bytes(),
    )
    return path
