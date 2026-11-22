"""One-context post-training evaluation of a saved context CAPT certificate."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from capt12.certification.artifact import Certificate, _rebuild_boxes, verify_certificate
from capt12.comparison.binary_capt import design_label_counts, fit_binary_capt
from capt12.comparison.optimal_ldp.calibration import TokenTVBall, calibrate
from capt12.comparison.optimal_ldp.numeric import ONE, UP, decimal_ratio, float_bound, log_bound
from capt12.comparison.optimal_ldp.retraining import (
    FixedBlocks,
    evaluate_comparison,
    fit_ctr,
    fixed_predictor_loss_bounds,
    fixed_reference_loss,
    prepare_comparison,
)
from capt12.comparison.optimal_ldp.theory import (
    Bounds,
    empirical_ldp_oracle,
    information_decomposition,
)
from capt12.data.loader import assert_disjoint_splits, load_parquet_sample
from capt12.data.preprocessing import build_group_histograms
from capt12.experiments.sampling import select_one_display_per_user_day
from capt12.models.reference import ReferenceModel
from capt12.privacy.adjacency import AdjacentPair, Group
from capt12.utils.artifacts import git_sha, sha256_file


@dataclass(frozen=True)
class ComparisonSettings:
    predictor_train_days: tuple[int, ...]
    predictor_validation_days: tuple[int, ...]
    eta: float = 0.01
    pseudocounts: tuple[float, ...] = (1, 10, 100)
    ctr_grid: tuple[float, ...] = tuple(i / 16 for i in range(17))
    exact_max_k: int = 64
    exact_calibration_limit: int = 12
    raw_shift_tv: float | None = None
    iid_utility_target: bool = False
    alpha_raw_calibration: float = 0.01
    alpha_test: float = 0.01
    alpha_utility: float = 0.01

    @classmethod
    def read(cls, path: Path) -> ComparisonSettings:
        return cls(**yaml.safe_load(path.read_text()))

    def validate(self, splits: dict) -> None:
        assert_disjoint_splits(splits)
        utility = {
            "D_predict_train": self.predictor_train_days,
            "D_predict_validation": self.predictor_validation_days,
        }
        assert_disjoint_splits(utility)
        allowed = set(splits["D_attack_train"])
        for days in utility.values():
            if not days or not set(days) <= allowed:
                raise ValueError(
                    "predictor train and validation must use separate D_attack_train days"
                )
        if not 0 <= self.exact_max_k <= 64 or not 0 <= self.exact_calibration_limit <= 16:
            raise ValueError("exact_max_k must be in [0,64]; exact_calibration_limit in [0,16]")
        if not 2 <= len(self.ctr_grid) <= 65:
            raise ValueError("ctr_grid needs between 2 and 65 points")
        if self.raw_shift_tv is not None and not 0 <= self.raw_shift_tv <= 1:
            raise ValueError("raw_shift_tv must be null or in [0,1]")
        if not isinstance(self.iid_utility_target, bool) or any(
            not 0 < alpha < 1
            for alpha in (self.alpha_raw_calibration, self.alpha_test, self.alpha_utility)
        ):
            raise ValueError("invalid statistical settings")


def _certified_block_beta(capt: FixedBlocks, boxes: dict, edges: list[AdjacentPair]) -> float:
    """Recompute the common budget over the verified boxes using exact vertex arithmetic."""
    if not edges:
        return 0.0
    bounds = {
        key: (tuple(map(Fraction, box.lower)), tuple(map(Fraction, box.upper)))
        for key, box in boxes.items()
    }
    if any(box.tv_radius for box in boxes.values()):
        raise ValueError("the repaired block verifier requires zero block TV radius")

    def support(column: tuple[Fraction, ...], key: str, order: list[int]) -> Fraction:
        lower, upper = bounds[key]
        remaining = 1 - sum(lower)
        value = sum(w * lo for w, lo in zip(column, lower, strict=True))
        for i in order:
            addition = min(remaining, upper[i] - lower[i])
            value += addition * column[i]
            remaining -= addition
            if not remaining:
                break
        if remaining:
            raise ValueError("infeasible verified block confidence box")
        return value

    maximum_ratio = Fraction(1)
    for column in zip(*capt.rows, strict=True):
        order = sorted(range(len(column)), key=column.__getitem__)
        maximum = {key: support(column, key, order[::-1]) for key in {e.left for e in edges}}
        minimum = {key: support(column, key, order) for key in {e.right for e in edges}}
        for edge in edges:
            numerator, denominator = maximum[edge.left], minimum[edge.right]
            if numerator and not denominator:
                raise ValueError("no finite scalar CAPT certificate for normalized rows")
            if denominator:
                maximum_ratio = max(maximum_ratio, numerator / denominator)
    ratio = decimal_ratio(maximum_ratio, UP)
    return float_bound(log_bound(max(ONE, ratio), UP), UP)


def _token_balls(
    frame: pd.DataFrame,
    tokens: np.ndarray,
    certificate: Certificate,
    settings: ComparisonSettings,
    assignment: np.ndarray,
) -> tuple[dict[str, TokenTVBall], dict[str, int]]:
    config = certificate.resolved_config
    profile, context_columns = config["profiles"][0], config["context_cols"]
    raw = build_group_histograms(
        frame,
        tokens,
        profile=profile,
        context_columns=context_columns,
        alphabet_size=len(assignment),
        rare_group_policy="confidence_box",
    ).counts
    stored = certificate.histogram_counts
    if stored is None or set(raw) - set(stored):
        raise ValueError("D_cert groups differ from the verified certificate")
    alpha_group = math.nextafter(settings.alpha_raw_calibration / len(stored), 0)
    balls, sizes = {}, {}
    for key, block_counts in stored.items():
        counts = raw.get(key, np.zeros(len(assignment), dtype=int))
        aggregated = np.zeros(len(block_counts), dtype=np.int64)
        np.add.at(aggregated, assignment, counts)
        if not np.array_equal(aggregated, block_counts):
            raise ValueError("D_cert token counts do not reproduce the certified block counts")
        balls[key] = TokenTVBall.from_counts(
            counts, alpha=alpha_group, shift_tv=settings.raw_shift_tv
        )
        sizes[key] = int(counts.sum())
    return balls, sizes


def _balls_respect_block_boxes(
    balls: dict[str, TokenTVBall], boxes: dict, assignment: np.ndarray
) -> bool:
    for key, ball in balls.items():
        aggregate = [Fraction(0)] * len(boxes[key].lower)
        for block, mass in zip(assignment, ball.center, strict=True):
            aggregate[block] += mass
        for mass, lower, upper in zip(aggregate, boxes[key].lower, boxes[key].upper, strict=True):
            if max(0, mass - ball.radius) < Fraction(lower) or min(
                1, mass + ball.radius
            ) > Fraction(upper):
                return False
    return True


def _json_finite(value):
    if isinstance(value, float) and math.isinf(value):
        return "infinity" if value > 0 else "-infinity"
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(v) for v in value]
    return value


def _verified_fixed_channel(certificate_path: Path):
    verification = verify_certificate(certificate_path)
    if not verification.valid:
        raise ValueError(f"CAPT certificate failed verification: {verification.worst_case}")
    certificate = Certificate.read(certificate_path)
    config = certificate.resolved_config
    if (
        config.get("dataset") != "criteo"
        or config.get("public_context_policy") != "stratified"
        or len(config["profiles"]) != 1
        or len(config["context_cols"]) != 1
        or "post_solve_repair" not in certificate.coverage
        or not {"encoder", "mapper", "model"} <= certificate.component_hashes.keys()
    ):
        raise ValueError("expected one verified repaired Criteo context with bound runtime models")

    assignment = np.asarray(certificate.assignment, dtype=int)
    k = len(assignment)
    if not 1 <= k <= 4096:
        raise ValueError("common original-token/release alphabet must have 1 to 4096 symbols")
    capt = FixedBlocks.from_arrays(assignment, np.asarray(certificate.channel))
    edges = [
        AdjacentPair(item["left"], item["right"], item["epsilon"]) for item in certificate.adjacency
    ]
    boxes, error = _rebuild_boxes(certificate, edges)
    if error or boxes is None:
        raise ValueError(f"could not rebuild verified block sets: {error}")
    beta = _certified_block_beta(capt, boxes, edges)
    if beta > certificate.target_epsilon:
        raise ValueError("normalized fixed channel exceeds the certificate's scalar target")
    return certificate, capt, boxes, edges, beta


def _fixed_input_columns(config):
    id_col, user_col = config.get("id_col", "id"), config.get("user_col", "user_id")
    categories = list(dict.fromkeys([*config["profiles"][0].split("+"), *config["context_cols"]]))
    columns = list(
        dict.fromkeys([id_col, user_col, *categories, *config.get("phi_source_cols", [])])
    )
    if config.get("label_col", "is_clicked") in columns:
        raise ValueError("privacy calibration and encoding cannot depend on the click label")
    return columns


def _binary_from_certificate(path, certificate, parent, edges):
    if certificate.confidence["coverage_basis"] != "finite_sample" or certificate.tv_radius:
        raise ValueError("binary-direct-cp requires finite-sample counts and zero block TV shift")
    config = certificate.resolved_config
    design_path = path.parent / "mechanism/frozen_design.npz"
    design_hash = sha256_file(design_path)
    manifest = path.parent / "sol_review_bundle_manifest.json"
    if manifest.is_file():
        for item in json.loads(manifest.read_text())["files"]:
            if item["path"] == "mechanism/frozen_design.npz" and item["sha256"] != design_hash:
                raise ValueError("frozen D_design table differs from the source run manifest")
    design = design_label_counts(
        design_path, str(config["public_context_value"]), parent.assignment, len(parent.rows)
    )
    candidate = fit_binary_capt(
        parent.assignment, design, certificate.histogram_counts, edges, float(config["alpha_cert"])
    )
    # Verify each ordered edge at its declared target, including heterogeneous budgets.
    beta = 0.0
    for target in sorted({e.epsilon for e in edges}):
        bound = _certified_block_beta(
            candidate.channel, candidate.boxes, [e for e in edges if e.epsilon == target]
        )
        if bound > target:
            raise ValueError("binary CAPT failed exact verification of normalized rows")
        beta = max(beta, bound)
    metadata = {
        "high_blocks": candidate.high_blocks,
        "probability_of_one_low_high": candidate.probabilities,
        "design_sha256": design_hash,
        "design_displays": int(design.sum()),
        "split": "D_design",
        "threshold_family_size": max(1, len(parent.rows) - 1),
        "alpha_context": float(config["alpha_cert"]),
        "confidence_method": "simultaneous two-cell Clopper-Pearson over groups and thresholds",
    }
    return candidate.channel, candidate.boxes, beta, metadata


def _load_selected_displays(data_root, days, columns, mapper, *, id_col, user_col, seen_ids):
    frame = load_parquet_sample(data_root=data_root, columns=columns, days=days)
    if frame.empty or set(frame["day_int"]) != set(days):
        raise ValueError(f"missing or empty source days: {days}")
    if frame[[id_col, user_col]].isna().any().any():
        raise ValueError("display and user identifiers must be present")
    ids = pd.Index(frame[id_col].astype(str))
    if ids.has_duplicates or not ids.intersection(seen_ids).empty:
        raise ValueError("duplicate display ID within or across loaded splits")
    selected = select_one_display_per_user_day(
        mapper.transform(frame),
        user_col=user_col,
        day_col="day_int",
        id_col=id_col,
    )
    return selected, seen_ids.append(ids)


def _check_context_groups(frame, attributes, profile, context, group_keys):
    observed_groups = {
        Group(profile, values, context).key()
        for values in frame[attributes].drop_duplicates().itertuples(index=False, name=None)
    }
    if observed_groups - group_keys:
        raise ValueError("selected context contains groups outside the verified attribute domain")


def _encode_tokens(encoder, frame, k):
    tokens = np.asarray(encoder.transform(frame), dtype=int) if len(frame) else np.empty(0, int)
    if tokens.shape != (len(frame),) or (tokens < 0).any() or (tokens >= k).any():
        raise ValueError("frozen encoder produced tokens outside the certified alphabet")
    return tokens


def run_fixed_capt_ctr_comparison(
    certificate_path: Path,
    settings: ComparisonSettings,
    output: Path,
    *,
    data_root: Path | None = None,
    mechanism: str = "fixed",
) -> dict:
    started = time.perf_counter()
    if mechanism not in {"fixed", "binary-direct-cp"}:
        raise ValueError(f"unknown CAPT mechanism: {mechanism}")
    binary = mechanism == "binary-direct-cp"
    if binary and output.exists():
        raise FileExistsError(f"binary comparison requires a new output file: {output}")
    certificate, parent, boxes, edges, beta = _verified_fixed_channel(certificate_path)
    config = certificate.resolved_config
    splits = certificate.split_identifiers
    settings.validate(splits)
    capt = parent
    if binary:
        capt, boxes, beta, binary_metadata = _binary_from_certificate(
            certificate_path, certificate, parent, edges
        )
    profile = config["profiles"][0]
    context_column = config["context_cols"][0]
    context = str(config["public_context_value"])
    id_col, user_col = config.get("id_col", "id"), config.get("user_col", "user_id")
    label = config.get("label_col", "is_clicked")
    attributes = profile.split("+")
    columns = _fixed_input_columns(config)
    if data_root is None:
        data_root = Path(config["data_root"])
    files = certificate_path.parent / "models"
    mapper = joblib.load(files / "category_mapper.joblib")
    encoder = joblib.load(files / "encoder.joblib")
    reference = ReferenceModel.load(files / "reference.joblib")
    assignment = capt.assignment
    k = len(assignment)
    seen_ids = pd.Index([], dtype=str)

    def load(days, *, utility: bool):
        nonlocal seen_ids
        frame, seen_ids = _load_selected_displays(
            data_root,
            days,
            [*columns, label] if utility else columns,
            mapper,
            id_col=id_col,
            user_col=user_col,
            seen_ids=seen_ids,
        )
        frame = frame.loc[frame[context_column].astype(str) == context].reset_index(drop=True)
        _check_context_groups(frame, attributes, profile, context, boxes.keys())
        return frame, _encode_tokens(encoder, frame, k)

    def utility_counts(days):
        counts = np.zeros((k, 2), dtype=np.int64)
        for day in days:
            frame, tokens = load([day], utility=True)
            labels = frame[label].to_numpy()
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("utility requires binary click labels")
            counts += np.bincount(2 * tokens + labels.astype(int), minlength=2 * k).reshape(k, 2)
        return counts

    balls, cert_sizes = None, None
    if settings.raw_shift_tv is not None and edges:
        frame, tokens = load(splits["D_cert"], utility=False)
        balls, cert_sizes = _token_balls(frame, tokens, certificate, settings, parent.assignment)
        del frame, tokens
    calibration = calibrate(
        beta,
        balls,
        [(edge.left, edge.right) for edge in edges],
        alphabet_size=k,
        exact_limit=settings.exact_calibration_limit,
        restricted_by_block_certificate=(
            balls is not None and not _balls_respect_block_boxes(balls, boxes, assignment)
        ),
    )
    if binary and not balls and edges and min(np.bincount(assignment, minlength=2)) >= 2:
        # For either group, any allowed event masses can be lifted onto a
        # different representative token in each bin. Disjoint support attains 1.
        calibration = replace(
            calibration,
            tv_method="exact_disjoint_token_preimages",
            uniform_method="exact_disjoint_token_preimages",
        )
    calibrated_seconds = time.perf_counter() - started
    budgets = {
        "equal_epsilon": float(config["epsilon"]),
        "tv_calibrated": calibration.tv_budget,
        "uniform_calibrated": calibration.uniform_budget,
    }
    train = utility_counts(settings.predictor_train_days)
    validation = utility_counts(settings.predictor_validation_days)
    fitted = prepare_comparison(
        capt,
        budgets,
        train,
        validation,
        grid=settings.ctr_grid,
        exact_max_k=settings.exact_max_k,
        pseudocounts=settings.pseudocounts,
        eta=settings.eta,
    )
    if binary:
        parent_predictor = fit_ctr(
            parent, train, validation, pseudocounts=settings.pseudocounts, eta=settings.eta
        )
    fitted_seconds = time.perf_counter() - started - calibrated_seconds
    scores = reference.predict(pd.DataFrame({"__token__": np.arange(k), context_column: context}))
    # Load D_test only after all privacy, mechanism, CTR and smoothing choices.
    test = utility_counts(splits["D_test"])
    result = evaluate_comparison(
        fitted,
        test,
        iid_target=settings.iid_utility_target,
        alpha_test=settings.alpha_test,
        alpha_utility=settings.alpha_utility,
    )
    for name, comparison in result["comparisons"].items():
        comparison["common_attribute_beta_uniformly_guaranteed"] = (
            name != "equal_epsilon" or budgets[name] <= calibration.uniform_budget
        )
    if binary:
        info = information_decomposition(test, capt)
        oracle = empirical_ldp_oracle(info, settings.ctr_grid, settings.exact_max_k)
        fitted_test = Bounds(
            *fixed_predictor_loss_bounds(fitted.capt, capt.exact_output_joint(test))
        )
        result.update(
            binary_channel={**binary_metadata, "fitted_output_ctr": fitted.capt.ctr.tolist()},
            parent_retrained_test_log_loss=parent_predictor.loss(test),
            test_plugin_capt_bayes=asdict(info.entropy_release),
        )
        for name, comparison in result["comparisons"].items():
            vz = oracle(budgets[name])
            comparison.update(
                test_ldp_fitted_minus_capt_fitted=comparison["ldp_test_log_loss"]
                - result["capt_test_log_loss"],
                test_plugin_v_z=asdict(vz),
                test_plugin_v_z_minus_fitted_capt=asdict(vz.minus(fitted_test)),
                test_plugin_v_z_minus_capt_bayes=asdict(vz.minus(info.entropy_release)),
            )
    privacy_alpha = float(config["alpha_cert"]) + (settings.alpha_raw_calibration if balls else 0)
    result.update(
        certificate=str(certificate_path.resolve()),
        certificate_sha256=sha256_file(certificate_path),
        certificate_source_git_sha=certificate.code_git_sha,
        evaluation_source_git_sha=git_sha(),
        data_root=str(data_root.resolve()),
        profile=profile,
        context_column=context_column,
        context=context,
        K=k,
        L=len(parent.rows),
        capt_mechanism=mechanism,
        capt_output_count=len(capt.rows),
        population="selected Criteo log user-days in the fixed context with the specified profile applied",
        observed_profile_selection=False,
        sampling_unit="one stable-hash-selected display per user-day before public context filtering",
        online_memoization="single release utility; online reuses the first release per epoch/profile/context",
        utility_assumption=(
            "conditionally iid selected user-days in validation and test from the same target law"
            if settings.iid_utility_target
            else "empirical temporal holdout only"
        ),
        splits={
            **splits,
            "D_predict_train": settings.predictor_train_days,
            "D_predict_validation": settings.predictor_validation_days,
        },
        utility_sample_sizes={
            "train": int(train.sum()),
            "validation": int(validation.sum()),
            "test": int(test.sum()),
        },
        settings=asdict(settings),
        timings_seconds={
            "verification_and_calibration": calibrated_seconds,
            "predictor_fit_and_validation": fitted_seconds,
        },
        calibration={
            **asdict(calibration),
            "beta_R": beta,
            "beta_R_basis": "upper-rounded common scalar over reconstructed block confidence sets",
            "uncertainty_sets": (
                "certified block-set token preimages intersected with simultaneous raw-token TV balls"
                if balls
                else "full original-token preimages of verified block confidence sets"
            ),
            "raw_shift_assumption": settings.raw_shift_tv,
            "cert_group_sizes": cert_sizes,
            "attribute_edges": certificate.adjacency,
            "attribute_bound_alpha": privacy_alpha,
            "verified_block_confidence_methods": sorted({box.method for box in boxes.values()}),
            "block_shift_tv": certificate.tv_radius,
        },
        simultaneous_error_upper=(
            min(1.0, privacy_alpha + settings.alpha_test + settings.alpha_utility)
            if settings.iid_utility_target
            else None
        ),
        capt_fixed_reference_test_log_loss=fixed_reference_loss(
            parent,
            np.asarray(certificate.decoder),
            scores,
            test,
        ),
        elapsed_seconds=time.perf_counter() - started,
    )
    if binary:
        result["parent_fixed_reference_test_log_loss"] = result[
            "capt_fixed_reference_test_log_loss"
        ]
        result["capt_fixed_reference_test_log_loss"] = None
    result = _json_finite(result)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x" if binary else "w", encoding="utf-8") as fh:
        fh.write(payload)
    return result
