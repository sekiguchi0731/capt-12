from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
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
from capt12.comparison.certificate import write_factorized_certificate
from capt12.comparison.contracts import default_method_contracts
from capt12.comparison.mass12 import Mass12FiniteChannel
from capt12.comparison.utility import FrozenCTRUtility, evaluate_frozen_ctr_utility
from capt12.config import comparison_design_name, run_id, validate_config
from capt12.data.loader import load_parquet_sample
from capt12.data.preprocessing import build_group_histograms
from capt12.evaluation.released_ranking import ranking_metadata
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
    attack_frame: pd.DataFrame
    attack_tokens: np.ndarray
    attack_context: np.ndarray
    attack_user_days: np.ndarray
    attack_test_positions: np.ndarray
    expected_attack_groups: tuple[Group, ...]


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
        if key not in _ORCHESTRATOR_KEYS and not key.startswith(("mass_", "attribute_attack_"))
    }
    result.update(
        {
            "output_dir": str(output_dir),
            "context_utility_objective": cost,
            "context_representation_mode": "objective_aligned",
            "context_designs": [comparison_design_name(int(config["K"]), block_count)],
            "frozen_design_seed": int(seed),
            "seed": int(seed),
            "epsilon": float(epsilon),
            "K": int(config["K"]),
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


def _prepare_data(config: dict[str, Any], native_path: Path, *, evaluation_only: bool = False) -> PreparedComparisonData:
    mapper = joblib.load(native_path / "models" / "category_mapper.joblib")
    encoder = joblib.load(native_path / "models" / "encoder.joblib")
    reference = joblib.load(native_path / "models" / "reference.joblib")
    columns = _columns(config)
    contexts = list(config.get("context_cols", []))
    context_column = contexts[0]
    id_col = config.get("id_col", "id")
    user_col = config.get("user_col", "user_id")
    label_col = config.get("label_col", "is_clicked")

    if evaluation_only:
        design_frame = cert_frame = pd.DataFrame(columns=[*columns, "day_int"])
        design_tokens = cert_tokens = np.array([], dtype=int)
        design_context = np.array([], dtype=str)
    else:
        design_frame = mapper.transform(load_parquet_sample(
            data_root=config["data_root"], columns=columns, days=config["splits"]["D_design"],
        ))
        design_tokens = encoder.transform(design_frame)
        design_context = _context_values(design_frame, contexts).astype(str).to_numpy()
        cert_source = select_one_display_per_user_day(load_parquet_sample(
            data_root=config["data_root"], columns=columns, days=config["splits"]["D_cert"],
        ), user_col=user_col, day_col="day_int", id_col=id_col)
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
    attack_frame = mapper.transform(select_one_display_per_user_day(
        load_parquet_sample(data_root=config["data_root"], columns=columns,
                            days=config["splits"]["D_attack_train"]),
        user_col=user_col, day_col="day_int", id_col=id_col,
    )) if not evaluation_only and config["attribute_attack_method"] != "none" else test_frame.iloc[:0].copy()
    attack_tokens = np.asarray(encoder.transform(attack_frame), dtype=int) if len(attack_frame) else np.array([], dtype=int)
    attack_context = _context_values(attack_frame, contexts).astype(str).to_numpy()
    test_positions = select_one_display_per_user_day(
        test_frame.assign(__attack_position=np.arange(len(test_frame))),
        user_col=user_col, day_col="day_int", id_col=id_col,
    )["__attack_position"].to_numpy(dtype=int)
    support = json.loads((native_path / "frozen_support.json").read_text(encoding="utf-8"))
    expected_attack_groups = tuple(
        Group(config["profiles"][0], tuple(map(str, group[:-1])), str(group[-1]))
        for group in support["G_cart"]
    )
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
        service_probabilities=_probability_grid(reference, context_values, context_column, int(config["K"])),
        representation_token_cost=representation_token_cost,
        attack_frame=attack_frame, attack_tokens=attack_tokens, attack_context=attack_context,
        attack_user_days=(attack_frame["day_int"].astype(str) + "|" + attack_frame[user_col].astype(str)).to_numpy(),
        attack_test_positions=test_positions, expected_attack_groups=expected_attack_groups,
    )


def _sample_factorized(
    channels: FactorizedChannels, tokens: np.ndarray, contexts: np.ndarray, seed: int,
) -> np.ndarray:
    """Sample the deployed two-stage mechanism without allocating N-by-K probabilities."""
    rng = np.random.default_rng(seed)
    result = np.empty(len(tokens), dtype=int)
    for context in np.unique(contexts):
        positions = np.flatnonzero(contexts == context)
        factors = channels.input_to_block[str(context)]
        if channels.mixing[str(context)] != 0:
            raise ValueError("attribute comparison expects unwrapped native channels")
        blocks = np.empty(len(positions), dtype=int)
        selected_tokens = tokens[positions]
        for token, selected in pd.Series(selected_tokens).groupby(selected_tokens, sort=True).indices.items():
            blocks[selected] = rng.choice(factors.shape[1], len(selected), p=factors[token])
        for block in np.unique(blocks):
            selected = positions[blocks == block]
            result[selected] = rng.choice(channels.decoder.shape[1], len(selected), p=channels.decoder[block])
    return result


def _sample_mass(model: Any, tokens: np.ndarray, contexts: np.ndarray, seed: int) -> np.ndarray:
    # Runtime samples, streamed to bound the K-logit memory on full Criteo splits.
    batch = max(1, min(4096, 4_000_000 // int(model.config["K"])))
    result = np.empty(len(tokens), dtype=int)
    for index, start in enumerate(range(0, len(tokens), batch)):
        draw_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
        result[start:start + batch] = model.sample(tokens[start:start + batch], contexts[start:start + batch], seed=draw_seed)
    return result


def _attach_attribute_attack(
    row: dict[str, Any], config: dict[str, Any], comparison_path: Path,
    data: PreparedComparisonData, sample: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
) -> None:
    from capt12.comparison.attribute_inference import evaluate_attribute_inference

    for name in RESULT_COLUMNS:
        if name.startswith("attribute_attack_"):
            row[name] = math.nan
    method = config["attribute_attack_method"]
    row["attribute_attack_method"] = method
    if method == "none":
        row["attribute_attack_status"] = "disabled"
        return
    controls = {key: value for key, value in config.items() if key.startswith("attribute_attack_")}
    identity = {"schema_version": 1, "mechanism": row.get("model_sha256") if row["method"] == "mass12_raw" else row["channel_sha256"],
                "cost": row["cost"], "frozen_seed": int(row["seed"]), "controls": controls,
                "source_git_sha": config.get("source_git_sha"), "splits": config["splits"],
                "profile": config["profiles"][0], "adjacency": config.get("adjacency", config.get("privacy_scope", "tuple_adjacent"))}
    identity["mass_execution_device"] = config.get("mass_device", "cpu") if row["method"] == "mass12_raw" else "cpu"
    attack_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = comparison_path / "attribute_attacks" / attack_id
    summary_path = path / "summary.json"
    if all((path / name).is_file() for name in ("summary.json", "pairs.csv", "roc.csv", "attacker_scores.npz", "roc.png", "roc.pdf")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        seed = int(np.random.SeedSequence([int(config["attribute_attack_seed"]), int(row["seed"])]).generate_state(1)[0])
        indices = data.attack_test_positions
        attack_outputs = sample(data.attack_tokens, data.attack_context, seed)
        test_outputs = sample(data.test_tokens[indices], data.test_context[indices], seed ^ 0xFFFFFFFF)
        registry = {(str(group.context), *group.values): group for group in data.expected_attack_groups}

        def groups(frame: pd.DataFrame, contexts: np.ndarray) -> list[Group]:
            values = frame[config["profiles"][0].split("+")].astype(str).to_numpy()
            return [registry[(str(context), *map(str, value))] for context, value in zip(contexts, values, strict=True)]

        result = evaluate_attribute_inference(
            attack_outputs, groups(data.attack_frame, data.attack_context), data.attack_user_days,
            test_outputs, groups(data.test_frame.iloc[indices], data.test_context[indices]), data.test_user_days[indices],
            expected_groups=data.expected_attack_groups, k=int(config["K"]), method=method,
            smoothing=float(config["attribute_attack_smoothing"]),
            target_fpr=float(config["attribute_attack_target_fpr"]),
            confidence=float(config["attribute_attack_confidence"]),
            calibration_fraction=float(config["attribute_attack_calibration_fraction"]), seed=seed,
            adjacency_mode=identity["adjacency"],
        )
        summary = result.summary
        summary["mechanism_identity"] = identity
        summary["attack_id"] = attack_id
        summary["release_sampling"] = "one actual mechanism draw per selected user-day; MaSS uses fresh Gaussian and categorical draws, not the utility MC channel"
        summary["release_sha256"] = hashlib.sha256(attack_outputs.tobytes() + test_outputs.tobytes()).hexdigest()
        summary["references"] = {
            "hypothesis_testing": "https://petsymposium.org/popets/2019/popets-2019-0045.pdf",
            "sections": ["2.2 Definition 3 / Equation 1", "3 Theorem 2", "5.3.1", "6"],
            "adaptation": "estimated conditional token distributions and held-out attribute-pair ROC; not a reproduction of the paper's analytic DP comparisons",
        }
        result.save(path)
    row.update(attribute_attack_id=attack_id, attribute_attack_path=str(path),
               attribute_attack_method=method, attribute_attack_status=summary["status"])
    for key in ("scope_sha256", "target_fpr", "expected_pairs", "eligible_pairs", "evaluated_pairs",
                "mean_roc_auc", "worst_roc_auc", "mean_tpr", "worst_tpr", "mean_fpr", "worst_fpr"):
        row[f"attribute_attack_{key}"] = summary[key]


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
        alphabet_size=int(config["K"]),
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
    source_groups = None
    if isinstance(channels, FactorizedChannels):
        source_groups = {context: np.unique(factors, axis=0, return_inverse=True)[1]
                         for context, factors in channels.input_to_block.items()}
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
        ranking_mc_replicates=int(config["ctr_ranking_mc_replicates"]),
        ranking_mc_seed=int(np.random.SeedSequence([int(config["ctr_ranking_mc_seed"]), seed]).generate_state(1)[0]),
        source_groups=source_groups,
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
            "mean_score_roc_auc": utility.mean_score_roc_auc,
            "mean_score_average_precision": utility.mean_score_average_precision,
            **ranking_metadata(utility.released_ranking),
            "ctr_channel_estimator": "gaussian_mc" if row.get("mass_backend") == "neural_token" else "exact",
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
            "mass_backend",
            "mass_control_sha256",
            "target_epsilon",
            "K",
            "L",
            "mass_m",
            "mass_n",
            "mass_privacy_weight",
            "mass_utility_weight",
            "mass_ctr_weight",
            "mass_temperature",
            "mass_infonce_weight",
            "mass_infonce_direction",
            "mass_infonce_pretrain_epochs",
            "mass_infonce_pretrain_learning_rate",
            "mass_privacy_quadratic_weight", "mass_privacy_linear_weight",
            "mass_useful_quadratic_weight", "mass_useful_linear_weight",
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
    design_name = comparison_design_name(int(config["K"]), block_count)
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
                cover_distribution=np.ones(int(config["K"]), dtype=float) / int(config["K"]),
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
                "K": int(config["K"]),
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
        _attach_attribute_attack(row, config, comparison_path, data,
                                 partial(_sample_factorized, token_channels))
        rows.append(row)
    rows[1]["paired_excess_log_loss_difference_vs_capt"] = (
        rows[1]["excess_ctr_log_loss"] - rows[0]["excess_ctr_log_loss"]
    )
    return rows


def _mass_history(model: Mass12FiniteChannel, path: Path) -> None:
    """Write both update stages, retaining the legacy pre-update columns."""
    rows = []
    for value in model.history:
        rows.append(
            {
                "epoch": value.epoch,
                "mass_ctr_weight": model.ctr_weight,
                "mass_infonce_pretrain_epochs": model.infonce_pretrain_epochs,
                "mass_infonce_pretrain_completed_epochs": len(model.infonce_pretrain_history),
                "loss": value.loss,
                "distortion": value.distortion,
                "sensitive_mi_nats": json.dumps(value.sensitive_mi_nats, sort_keys=True),
                "useful_mi_nats": json.dumps(value.useful_mi_nats, sort_keys=True),
                "pre_update_loss": value.loss,
                "pre_update_distortion": value.distortion,
                "pre_update_sensitive_mi_nats": json.dumps(
                    value.sensitive_mi_nats, sort_keys=True
                ),
                "pre_update_useful_mi_nats": json.dumps(value.useful_mi_nats, sort_keys=True),
                "post_update_loss": value.post_update_loss,
                "post_update_distortion": value.post_update_distortion,
                "post_update_sensitive_mi_nats": json.dumps(
                    value.post_update_sensitive_mi_nats, sort_keys=True
                ),
                "post_update_useful_mi_nats": json.dumps(
                    value.post_update_useful_mi_nats, sort_keys=True
                ),
                "infonce_loss": value.infonce_loss,
                "pre_update_infonce_loss": value.infonce_loss,
                "post_update_infonce_loss": value.post_update_infonce_loss,
                "infonce_batch_seed": value.infonce_batch_seed,
                "infonce_output_seed": value.infonce_output_seed,
                "pre_update_infonce_output_to_source_loss": value.infonce_output_to_source_loss,
                "post_update_infonce_output_to_source_loss": value.post_update_infonce_output_to_source_loss,
                "pre_update_infonce_source_to_output_loss": value.infonce_source_to_output_loss,
                "post_update_infonce_source_to_output_loss": value.post_update_infonce_source_to_output_loss,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _mass_pretrain_history(model: Mass12FiniteChannel, path: Path) -> None:
    columns = ["epoch", "pre_update_infonce_loss", "post_update_infonce_loss",
               "infonce_batch_seed", "learning_rate", "temperature"]
    rows = [
        {
            "epoch": value.epoch,
            "pre_update_infonce_loss": value.loss,
            "post_update_infonce_loss": value.post_update_loss,
            "infonce_batch_seed": value.batch_seed,
            "learning_rate": model.infonce_pretrain_learning_rate,
            "temperature": model.infonce_temperature,
        }
        for value in model.infonce_pretrain_history
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


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
    config: dict[str, Any], comparison_path: Path, data: PreparedComparisonData,
    *, cost: str, block_count: int, frozen_seed: int, mass_seed: int,
    loss_m: float, loss_n: float, privacy_weight: float, utility_weight: float,
    temperature: float,
    progress: ProgressLogger | None = None,
) -> list[dict[str, Any]]:
    from capt12.comparison.mass_nn import MassNN, NeuralChannels, fit_mass_nn
    from capt12.utils.artifacts import sha256_file

    sensitive = {attribute: data.design_frame[attribute].astype(str).to_numpy()
                 for attribute in config["profiles"][0].split("+")}
    useful = {"click": data.design_frame[config.get("label_col", "is_clicked")].to_numpy()}
    contexts = tuple(sorted(data.service_probabilities))
    controls = {
        **{key: value for key, value in config.items() if key.startswith("mass_") and not key.endswith("_list")},
        "K": int(config["K"]), "loss_m": loss_m, "loss_n": loss_n,
        "privacy_weight": privacy_weight, "utility_weight": utility_weight,
        "temperature": temperature, "mass_seed": mass_seed,
    }
    control_sha = hashlib.sha256(json.dumps(controls, sort_keys=True).encode()).hexdigest()
    # Neither CAPT L nor its decoder is a training input/cache key. Reuse the
    # same frozen neural model when paired with different CAPT block counts.
    stem = f"mass-nn-{cost}-fseed{frozen_seed}-mseed{mass_seed}-{control_sha[:12]}"
    model_path = comparison_path / "models" / f"{stem}.pt"
    if model_path.is_file():
        model = MassNN.load(model_path, device=config.get("mass_device", "cpu"))
    else:
        model = fit_mass_nn(
            data.design_tokens, data.design_context, sensitive, useful, config=config,
            loss_m=loss_m, loss_n=loss_n, privacy_weight=privacy_weight,
            utility_weight=utility_weight, seed=mass_seed, temperature=temperature,
            cost=data.representation_token_cost if config["mass_ctr_weight"] else None,
            frozen_context_values=contexts,
            progress=(lambda event: progress.emit("mass_training_progress", **event)) if progress else None,
        )
        model.save(model_path)
        pd.DataFrame(model.history).to_csv(comparison_path / "tables" / f"{stem}-history.csv", index=False)
        pd.DataFrame(model.pretrain_history, columns=["stage", "update", "pre_update_loss", "post_update_loss"]).to_csv(
            comparison_path / "tables" / f"{stem}-pretrain.csv", index=False,
        )
        (comparison_path / "models" / f"{stem}.json").write_text(
            json.dumps(model.provenance(), indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    channels = NeuralChannels(model)
    utility = _evaluate(channels, data, config, seed=frozen_seed)
    final = model.history[-1]
    sensitive_mi = json.loads(final["post_update_sensitive_mi_nats"])
    useful_mi = json.loads(final["post_update_useful_mi_nats"])
    row = _empty_row()
    row.update({
        "method": "mass12_raw", "display_name": "MaSS neural (token interface)",
        "mass_backend": "neural_token", "mass_control_sha256": control_sha,
        "mass_device": str(model.device),
        "mass_torch_version": model.training_runtime.get("torch_version", "unknown"),
        "mass_cuda_version": model.training_runtime.get("cuda_runtime"),
        "cost": cost, "seed": frozen_seed, "mass_seed": mass_seed,
        "K": int(config["K"]), "mass_output_count": int(config["K"]),
        # L labels only the CAPT reference for pairing; it does not constrain MaSS.
        "L": block_count, "target_epsilon": float(config.get("formal_comparison_epsilon", 1.0)),
        "mass_m": loss_m, "mass_n": loss_n, "mass_privacy_weight": privacy_weight,
        "mass_utility_weight": utility_weight, "mass_temperature": temperature,
        "mass_ctr_weight": config["mass_ctr_weight"],
        **{key: config[key] for key in (
            "mass_infonce_weight", "mass_infonce_direction", "mass_infonce_pretrain_epochs",
            "mass_infonce_pretrain_learning_rate", "mass_privacy_quadratic_weight",
            "mass_privacy_linear_weight", "mass_useful_quadratic_weight", "mass_useful_linear_weight",
            "mass_batch_size", "mass_learning_rate", "mass_weight_decay", "mass_token_dim",
            "mass_hidden_dim", "mass_attribute_pretrain_epochs", "mass_eval_noise_samples", "mass_mi_conditioning",
        )},
        "mass_sensitive_mi_estimate_nats": max(sensitive_mi.values(), default=0.0),
        "mass_useful_mi_estimate_nats": min(useful_mi.values(), default=0.0),
        "mass_sensitive_mi_by_attribute_nats": json.dumps(sensitive_mi, sort_keys=True),
        "mass_useful_mi_by_attribute_nats": json.dumps(useful_mi, sort_keys=True),
        "mass_classifier_ce_nats": final["post_update_attribute_ce_nats"],
        "mass_final_loss": final["post_update_loss"],
        "model_path": str(model_path), "model_sha256": sha256_file(model_path),
        "cover_lambda": 0.0, "cover_row_tv": 0.0,
        "formal_comparable": False, "certificate_valid": False,
        "deployable_under_capt": True, "uses_sensitive_value_online": False, "oracle": False,
        "exclusion_reason": "Own CE-based MI objective; no CAPT certificate or post-training randomization",
    })
    _utility_row(row, utility)
    _attach_attribute_attack(row, config, comparison_path, data,
                             partial(_sample_mass, model))
    return [row]


def _phase_values(
    config: dict[str, Any], phase: str
) -> tuple[list[str], list[int], list[int], list[float]]:
    costs = list(config.get("cost_list", ["empirical_logloss"]))
    seeds = list(map(int, config.get("seeds", [0])))
    formal = float(config.get("formal_comparison_epsilon", 1.0))
    if phase == "pilot":
        pilot_seeds = list(map(int, config.get("comparison_pilot_seeds", seeds[:1])))
        return costs, [int(config["L"])], pilot_seeds, [formal]
    if phase != "full":
        raise ValueError("phase must be pilot or full")
    block_counts = list(map(int, config.get("comparison_L_list", [int(config["L"])])))
    epsilons = list(map(float, config.get("epsilon_list", [formal])))
    if formal not in epsilons:
        epsilons.append(formal)
    return costs, sorted(set(block_counts)), seeds, sorted(set(epsilons))


def _mass_training_signature(config: dict[str, Any]) -> str:
    controls = {key: value for key, value in config.items()
                if key.startswith("mass_") and not key.endswith("_list")
                and not key.startswith("mass_pilot_")}
    return hashlib.sha256(json.dumps(controls, sort_keys=True).encode()).hexdigest()


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
    from capt12.comparison.mass_nn import resolve_mass_device

    # Check the chosen CUDA build/kernel before spending time on CAPT/data loading.
    selected_device = resolve_mass_device(config["mass_device"])
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
        if pilot_summary.get("mass_training_signature") != _mass_training_signature(config):
            raise ValueError("full comparison and MaSS pilot must use the same neural training controls")
        if (int(pilot_summary.get("K", 4096)), int(pilot_summary.get("L", 16))) != (
            int(config["K"]), int(config["L"])
        ):
            raise ValueError("full comparison and MaSS pilot must use the same K and primary L")
        if pilot_summary.get("mass_ctr_weight") != config["mass_ctr_weight"]:
            raise ValueError("full comparison and MaSS pilot must use the same mass_ctr_weight")
        for key in ("mass_infonce_pretrain_epochs", "mass_infonce_pretrain_learning_rate"):
            if pilot_summary.get(key) != config[key]:
                raise ValueError(f"full comparison and MaSS pilot must use the same {key}")
        if not bool(pilot_summary.get("full_grid_allowed")):
            raise ValueError(
                "MaSS pilot did not show the predeclared classifier-MI variation required "
                "for the full grid"
            )
    config["comparison_phase"] = phase
    config["output_dir"] = str(
        publication / "runs"
    )
    path = prepare_run(config)
    progress = ProgressLogger(path, name=path.name)
    progress.emit_environment()
    progress.emit("mass_device_selected", device=str(selected_device))
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
            progress.emit("comparison_data_prepare_started", native_run=str(native))
            current_data = _prepare_data(config, native)
            current_data_key = data_key
            progress.emit("comparison_data_prepare_finished", design_rows=len(current_data.design_frame),
                          cert_rows=len(current_data.cert_frame), test_rows=len(current_data.test_frame),
                          attack_rows=len(current_data.attack_frame), **process_memory_bytes())
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
        progress.emit("mass_data_prepare_started", native_run=str(native))
        data = _prepare_data(config, native)
        progress.emit("mass_data_prepare_finished", design_rows=len(data.design_frame), **process_memory_bytes())
        for mass_seed, loss_m, loss_n, privacy_weight, utility_weight, temperature in mass_controls:
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
                ctr_weight=float(config.get("mass_ctr_weight", 0.0)),
                temperature=temperature,
            )
            rows = _mass_rows(
                config,
                path,
                data,
                cost=cost,
                block_count=block_count,
                frozen_seed=frozen_seed,
                mass_seed=int(mass_seed),
                loss_m=float(loss_m),
                loss_n=float(loss_n),
                privacy_weight=float(privacy_weight),
                utility_weight=float(utility_weight),
                temperature=float(temperature),
                progress=progress,
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
            & np.isfinite(frame["mass_sensitive_mi_estimate_nats"].astype(float))
        ]
        ranges = {
            str(cost): float(group["mass_sensitive_mi_estimate_nats"].max() - group["mass_sensitive_mi_estimate_nats"].min())
            for cost, group in raw.groupby("cost", sort=True)
        }
        threshold = float(config["mass_pilot_min_mi_range"])
        full_allowed = bool(ranges) and set(ranges) == set(costs) and all(
            value > threshold for value in ranges.values()
        )
        (publication / "pilot_summary.json").write_text(
            json.dumps(
                {
                    "source_git_sha": config["source_git_sha"],
                    "mass_training_signature": _mass_training_signature(config),
                    "K": int(config["K"]),
                    "L": int(config["L"]),
                    "mass_ctr_weight": float(config["mass_ctr_weight"]),
                    "mass_infonce_pretrain_epochs": int(config["mass_infonce_pretrain_epochs"]),
                    "mass_infonce_pretrain_learning_rate": float(config["mass_infonce_pretrain_learning_rate"]),
                    "classifier_mi_range_by_cost": ranges,
                    "minimum_required_range": threshold,
                    "full_grid_allowed": full_allowed,
                    "rule": "every configured cost must show finite classifier-MI variation across pilot controls",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    contracts = default_method_contracts(token_count=int(config["K"]))
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


def _saved_native_utility_rows(run_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Expose an existing standalone CAPT run for utility-only reevaluation."""
    design = config["context_designs"][0]
    path = run_dir / "mechanism" / f"context_channels-{design}.npz"
    if not path.is_file():
        raise ValueError("run has neither comparison results nor saved native context channels")
    with np.load(path) as payload:
        rows = []
        for method, key in (("capt", "channels"), ("optimal_ldp", "ldp_channels")):
            channels = _token_channels(payload[key], payload["contexts"], payload["assignment"], payload["decoder"])
            row = _empty_row()
            row.update(method=method, display_name="CAPT" if method == "capt" else "LDP (native blocks)",
                       cost=config.get("context_utility_objective", "teacher_kl"),
                       seed=int(config.get("frozen_design_seed", config.get("seed", 0))),
                       K=int(config["K"]), L=len(payload["decoder"]), target_epsilon=float(config["epsilon"]),
                       channel_sha256=_channel_sha256(channels), certificate_path=str(run_dir) if method == "capt" else "",
                       ctr_ranking_definition="legacy_mean_score")
            certificate_table = run_dir / "tables" / "certificate_summary.csv"
            if method == "capt" and certificate_table.is_file():
                certificate = pd.read_csv(certificate_table)
                valid = bool(certificate["certificate_valid"].astype(str).str.lower().eq("true").all())
                row.update(certificate_valid=valid, formal_comparable=valid,
                           certified_upper_epsilon=float(certificate["certificate_conservative_realized_epsilon"].max()))
            rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _reevaluate_saved_comparison(
    run_dir: Path, overrides: dict[str, Any], output_dir: Path | None = None,
    *, utility: bool = False,
) -> Path:
    """Evaluate saved mechanisms under new utility/attack controls, without retraining.

    Original comparison tables, model files, and certificates are never rewritten.
    """
    import yaml

    from capt12.comparison.artifacts import render_attribute_inference_figures, validate_results
    from capt12.utils.artifacts import sha256_file

    source_config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    config = record_source_provenance(validate_config({**source_config, "prior_art_comparison": True, **overrides}))
    controls = {key: value for key, value in config.items() if key.startswith("ctr_ranking_" if utility else "attribute_attack_")}
    if utility:
        controls["mass_eval_noise_samples"] = config["mass_eval_noise_samples"]
    signature = hashlib.sha256(json.dumps({"controls": controls, "mass_device": config["mass_device"], "evaluation_source": config["source_git_sha"]}, sort_keys=True).encode()).hexdigest()[:16]
    output = output_dir or run_dir / ("utility_evaluations" if utility else "attack_evaluations") / signature
    if output.resolve() == run_dir.resolve():
        raise ValueError("reevaluation requires a separate output directory")
    if (output / "results.csv").exists() and not (output / "evaluation.json").is_file():
        raise ValueError("reevaluation output directory would overwrite an original comparison table")
    output.mkdir(parents=True, exist_ok=True)
    comparison_table = run_dir / "tables" / "results.csv"
    if comparison_table.is_file():
        frame = validate_results(pd.read_csv(comparison_table))
    elif utility:
        frame = _saved_native_utility_rows(run_dir, config)
    else:
        raise ValueError("attribute reevaluation requires a completed comparison results.csv")
    references = frame.loc[frame["method"] == "capt"]
    rows = []
    previous_native, data = None, None
    for row in frame.to_dict(orient="records"):
        matching = references
        for key in ("cost", "seed", "K", "L", "target_epsilon"):
            matching = matching.loc[matching[key] == row[key]]
        if len(matching) != 1:
            raise ValueError("each reevaluation row requires exactly one matching saved CAPT reference")
        native = Path(matching.iloc[0]["certificate_path"])
        if previous_native != native:
            data = _prepare_data(config, native, evaluation_only=utility)
            previous_native = native
        if data is None:
            raise RuntimeError("evaluation data were not loaded")
        if row["method"] == "mass12_raw" and row["mass_backend"] == "neural_token":
            from capt12.comparison.mass_nn import MassNN, NeuralChannels

            if sha256_file(Path(row["model_path"])) != row["model_sha256"]:
                raise ValueError("saved MaSS model differs from the recorded model SHA")
            model = MassNN.load(Path(row["model_path"]), device=config["mass_device"])
            sampler = partial(_sample_mass, model)
            if utility:
                model.config["mass_eval_noise_samples"] = config["mass_eval_noise_samples"]
                row["mass_eval_noise_samples"] = config["mass_eval_noise_samples"]
                channels = NeuralChannels(model)
        elif row["method"] in {"capt", "optimal_ldp"}:
            design = (comparison_design_name(int(row["K"]), int(row["L"]))
                      if comparison_table.is_file() else source_config["context_designs"][0])
            with np.load(native / "mechanism" / f"context_channels-{design}.npz") as payload:
                channels = _token_channels(payload["channels" if row["method"] == "capt" else "ldp_channels"],
                                           payload["contexts"], payload["assignment"], payload["decoder"])
            if _channel_sha256(channels) != row["channel_sha256"]:
                raise ValueError("saved native channel differs from the recorded channel SHA")
            sampler = partial(_sample_factorized, channels)
        else:
            raise ValueError(f"unsupported saved mechanism for reevaluation: {row['method']}")
        if utility:
            _utility_row(row, _evaluate(channels, data, config, seed=int(row["seed"])))
        else:
            _attach_attribute_attack(row, config, output, data, sampler)
        rows.append(row)
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    result.to_csv(output / "results.csv", index=False)
    render_attribute_inference_figures(result, output / "figures")
    (output / "evaluation.json").write_text(json.dumps({
        "comparison_run": str(run_dir), "training_source_git_sha": source_config.get("source_git_sha"),
        "evaluation_source_git_sha": config["source_git_sha"], "controls": controls,
        "mass_execution_device": config["mass_device"],
        "mechanisms_retrained": False,
        "evaluation_kind": "released_ctr_utility" if utility else "attribute_attack",
        "certificates_recomputed": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def reevaluate_attribute_attacks(run_dir: Path, overrides: dict[str, Any], output_dir: Path | None = None) -> Path:
    return _reevaluate_saved_comparison(run_dir, overrides, output_dir)


def reevaluate_ctr_utility(run_dir: Path, overrides: dict[str, Any], output_dir: Path | None = None) -> Path:
    return _reevaluate_saved_comparison(run_dir, overrides, output_dir, utility=True)
