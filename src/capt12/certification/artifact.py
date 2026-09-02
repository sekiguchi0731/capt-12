from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from capt12.certification.robust import (
    VerificationResult,
    verify_robust_channel,
    verify_robust_channel_conservative,
)
from capt12.confidence.boxes import (
    CONFIDENCE_REGISTRY,
    ConfidenceBox,
    confidence_box_from_counts,
)
from capt12.decoders.registry import validate_decoder
from capt12.mechanisms.lp import lift_block_channel
from capt12.models.reference import TOKEN_REFERENCE_FEATURE_SCHEMA
from capt12.privacy.adjacency import AdjacentPair, Group, build_adjacency
from capt12.utils.artifacts import require_clean_worktree, sha256_file

DEFAULT_VERIFICATION_TOLERANCE = 1e-8


@dataclass
class Certificate:
    version: int
    created_at: str
    code_git_sha: str
    resolved_config: dict[str, Any]
    component_hashes: dict[str, str]
    split_identifiers: dict[str, Any]
    confidence: dict[str, Any]
    dp_parameters: dict[str, Any]
    tv_radius: float
    target_epsilon: float
    realized_worst_case_epsilon: float
    worst_case: dict[str, Any] | None
    solver: dict[str, Any]
    constraint_verification: dict[str, Any]
    channel: list[list[float]]
    boxes: dict[str, dict[str, Any]]
    adjacency: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    histogram_counts: dict[str, list[int]] | None
    assignment: list[int] | None
    decoder: list[list[float]] | None
    coverage: dict[str, Any]
    verification_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> Certificate:
        value = json.loads(Path(path).read_text())
        value.setdefault("histogram_counts", None)
        value.setdefault("groups", [])
        value.setdefault("assignment", None)
        value.setdefault("decoder", None)
        value.setdefault("coverage", {})
        value.setdefault("verification_scope", "legacy_embedded_constraints_only")
        value.setdefault(
            "constraint_verification", value.pop("independent_verification", {})
        )
        return cls(**value)


def hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_certificate(
    *,
    config: dict[str, Any],
    channel: np.ndarray,
    boxes: Mapping[str, ConfidenceBox],
    adjacency: Sequence[AdjacentPair],
    verification: VerificationResult,
    solver: Any,
    component_hashes: dict[str, str] | None = None,
    split_identifiers: dict[str, Any] | None = None,
    dp_parameters: dict[str, Any] | None = None,
    histogram_counts: Mapping[str, np.ndarray] | None = None,
    assignment: np.ndarray | None = None,
    decoder: np.ndarray | None = None,
    coverage: dict[str, Any] | None = None,
    groups: Sequence[Group] | None = None,
) -> Certificate:
    config = dict(config)
    if config.get("rare_group_policy") == "merge_to_other":
        raise ValueError(
            "merge_to_other cannot produce a certificate without a frozen runtime "
            "coarsening map"
        )
    if not verification.valid:
        raise ValueError("cannot serialize a certificate that failed robust constraint verification")
    strict_verification = verify_robust_channel(
        channel,
        boxes,
        adjacency,
        tolerance=DEFAULT_VERIFICATION_TOLERANCE,
    )
    if not strict_verification.valid:
        raise ValueError("channel fails the verifier-owned certificate tolerance")
    verification = strict_verification
    if any(box.experimental for box in boxes.values()):
        raise ValueError("experimental confidence boxes cannot be serialized as certificates")
    methods = {box.method for box in boxes.values()}
    if "point" in methods and config.get("dataset", "synthetic") not in {
        "synthetic",
        "synthetic_theorem4",
    }:
        raise ValueError("point confidence certificates require a known synthetic population")
    if config.get("dataset", "synthetic") not in {
        "synthetic",
        "synthetic_theorem4",
    }:
        if config.get("sampling_assumption") != "user_day_iid":
            raise ValueError(
                "sampled-data certificates require the explicit user_day_iid assumption"
            )
        if config.get("contribution_policy", "one-display-per-uuid-day") != (
            "one-display-per-uuid-day"
        ):
            raise ValueError(
                "user_day_iid certificates require one-display-per-uuid-day contributions"
            )
    source_sha = require_clean_worktree()
    declared_sha = config.get("source_git_sha")
    if declared_sha not in {None, source_sha}:
        raise ValueError(
            "resolved source_git_sha does not match the clean checkout used to create "
            "the certificate"
        )
    config["require_clean_worktree"] = True
    config["source_worktree_clean"] = True
    config["source_git_sha"] = source_sha
    if assignment is None or decoder is None:
        raise ValueError("certificates require embedded assignment and common decoder")
    if not groups:
        raise ValueError("certificates require structured protected groups")
    if {group.key() for group in groups} != set(boxes):
        raise ValueError("protected-group keys must match confidence-box keys")
    expected_adjacency = build_adjacency(
        groups,
        config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
        float(config.get("epsilon", 1.0)),
        config.get("epsilon_by_attr", {}),
    )
    if set(expected_adjacency) != set(adjacency):
        raise ValueError("adjacency does not match structured groups and resolved config")
    serialized_boxes = {key: box.to_dict() for key, box in boxes.items()}
    serialized_adjacency = [asdict(pair) for pair in adjacency]
    serialized_groups = [asdict(group) for group in (groups or [])]
    serialized_counts = (
        {key: np.asarray(value, dtype=int).tolist() for key, value in histogram_counts.items()}
        if histogram_counts is not None
        else None
    )
    serialized_coverage = dict(coverage or {})
    serialized_splits = dict(split_identifiers or {})
    serialized_dp = dict(dp_parameters or {})
    hashes = dict(component_hashes or {})
    hashes["channel"] = hash_array(channel)
    hashes["config"] = hash_json(config)
    hashes["boxes"] = hash_json(serialized_boxes)
    hashes["adjacency"] = hash_json(serialized_adjacency)
    hashes["groups"] = hash_json(serialized_groups)
    hashes["coverage"] = hash_json(serialized_coverage)
    hashes["split_identifiers"] = hash_json(serialized_splits)
    hashes["dp_parameters"] = hash_json(serialized_dp)
    if serialized_counts is not None:
        hashes["histogram_counts"] = hash_json(serialized_counts)
    validate_decoder(decoder, assignment)
    lift_block_channel(channel, assignment, decoder)
    hashes["partition"] = hash_array(assignment)
    hashes["decoder"] = hash_array(decoder)
    return Certificate(
        version=2,
        created_at=datetime.now(UTC).isoformat(),
        code_git_sha=source_sha,
        resolved_config=config,
        component_hashes=hashes,
        split_identifiers=serialized_splits,
        confidence={
            "methods": sorted(methods),
            "overall_level": (
                1.0 if methods == {"point"} else 1 - float(config.get("alpha_cert", 0.05))
            ),
            "coverage_basis": (
                "known_synthetic_population" if methods == {"point"} else "finite_sample"
            ),
        },
        dp_parameters=serialized_dp,
        tv_radius=max((box.tv_radius for box in boxes.values()), default=0.0),
        target_epsilon=max((pair.epsilon for pair in adjacency), default=0.0),
        realized_worst_case_epsilon=verification.realized_epsilon,
        worst_case=verification.worst_case,
        solver=asdict(solver) if hasattr(solver, "__dataclass_fields__") else dict(solver),
        constraint_verification=asdict(verification),
        channel=np.asarray(channel).tolist(),
        boxes=serialized_boxes,
        adjacency=serialized_adjacency,
        groups=serialized_groups,
        histogram_counts=serialized_counts,
        assignment=np.asarray(assignment, dtype=int).tolist() if assignment is not None else None,
        decoder=np.asarray(decoder, dtype=float).tolist() if decoder is not None else None,
        coverage=serialized_coverage,
        verification_scope="self_contained_bundle_consistency_and_robust_constraints",
    )


def _invalid(error: str) -> VerificationResult:
    return VerificationResult(False, float("inf"), float("inf"), {"error": error}, 0)


def _same_box(left: ConfidenceBox, right: ConfidenceBox) -> bool:
    return (
        left.method == right.method
        and left.experimental == right.experimental
        and np.isclose(left.alpha_familywise, right.alpha_familywise)
        and np.isclose(left.tv_radius, right.tv_radius)
        and np.allclose(left.lower, right.lower, atol=1e-12, rtol=0)
        and np.allclose(left.upper, right.upper, atol=1e-12, rtol=0)
        and np.allclose(left.nominal, right.nominal, atol=1e-12, rtol=0)
    )


def _verify_component_hashes(certificate: Certificate, path: Path) -> str | None:
    embedded: dict[str, str] = {
        "channel": hash_array(np.asarray(certificate.channel, dtype=float)),
        "config": hash_json(certificate.resolved_config),
        "boxes": hash_json(certificate.boxes),
        "adjacency": hash_json(certificate.adjacency),
        "groups": hash_json(certificate.groups),
        "coverage": hash_json(certificate.coverage),
        "split_identifiers": hash_json(certificate.split_identifiers),
        "dp_parameters": hash_json(certificate.dp_parameters),
    }
    if certificate.histogram_counts is not None:
        embedded["histogram_counts"] = hash_json(certificate.histogram_counts)
    if certificate.assignment is not None:
        embedded["partition"] = hash_array(np.asarray(certificate.assignment, dtype=int))
    if certificate.decoder is not None:
        embedded["decoder"] = hash_array(np.asarray(certificate.decoder, dtype=float))
    external = {
        "encoder": path.parent / "models" / "encoder.joblib",
        "model": path.parent / "models" / "reference.joblib",
        "mapper": path.parent / "models" / "category_mapper.joblib",
        "context_channel_manifest": (
            path.parent / "mechanism" / "context_channel_manifest.json"
        ),
    }
    for name, expected in certificate.component_hashes.items():
        if name in embedded:
            actual = embedded[name]
        elif name in external:
            component_path = external[name]
            if not component_path.is_file():
                return f"certificate bundle is missing component: {component_path}"
            actual = sha256_file(component_path)
        else:
            return f"certificate contains an unsupported, unverifiable component hash: {name}"
        if actual != expected:
            return f"component hash mismatch: {name}"
    required = {
        "channel",
        "config",
        "partition",
        "decoder",
        "boxes",
        "adjacency",
        "groups",
        "coverage",
        "split_identifiers",
        "dp_parameters",
    }
    if certificate.histogram_counts is not None:
        required.add("histogram_counts")
    config = certificate.resolved_config
    if config.get("sensitive_fallback_policy") == "unified_unknown":
        required.add("mapper")
    if config.get("public_context_policy") == "stratified":
        required.add("context_channel_manifest")
    missing = required - set(certificate.component_hashes)
    if missing:
        return f"certificate is missing required component hashes: {sorted(missing)}"
    return None


def _verify_profile_bundle(certificate: Certificate, path: Path) -> str | None:
    profiles = certificate.coverage.get("bundle_profiles", [])
    if len(profiles) <= 1:
        return None
    expected_decoder = certificate.component_hashes.get("decoder")
    expected_partition = certificate.component_hashes.get("partition")
    for profile in profiles:
        sibling_path = path.parent / f"certificate-{str(profile).replace('+', '_')}.json"
        if not sibling_path.is_file():
            return f"profile certificate bundle is incomplete: {sibling_path.name}"
        sibling = Certificate.read(sibling_path)
        if sibling.version != certificate.version:
            return f"profile certificate version mismatch: {sibling_path.name}"
        if sibling.component_hashes.get("decoder") != expected_decoder:
            return f"profile certificates do not share one common decoder: {sibling_path.name}"
        if sibling.component_hashes.get("partition") != expected_partition:
            return f"profile certificates do not share one partition: {sibling_path.name}"
        if sibling.component_hashes.get("config") != certificate.component_hashes.get("config"):
            return f"profile certificate config mismatch: {sibling_path.name}"
    return None


def _verify_context_channel_manifest(
    certificate: Certificate, path: Path
) -> str | None:
    config = certificate.resolved_config
    if config.get("public_context_policy") != "stratified":
        return None
    manifest_path = path.parent / "mechanism" / "context_channel_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return f"context-channel manifest cannot be read: {error}"
    contexts = list(config.get("context_cols", []))
    if len(contexts) != 1:
        return "context-channel manifest requires exactly one public context column"
    context_column = contexts[0]
    selector = ["Z", "profile", context_column]
    if manifest.get("profile") != config.get("profiles", [None])[0]:
        return "context-channel manifest profile does not match the certificate"
    if manifest.get("public_context_column") != context_column:
        return "context-channel manifest public context does not match the certificate"
    if manifest.get("channel_selector_inputs") != selector:
        return "context-channel manifest selector is not Z/profile/public context"
    if manifest.get("protected_value_used_online") is not False:
        return "context-channel manifest permits a protected online selector"
    if manifest.get("runtime_mapper_hash") != certificate.component_hashes.get("mapper"):
        return "context-channel manifest runtime mapper does not match the certificate"
    if int(manifest.get("version", 0)) >= 4:
        if manifest.get("reference_model_hash") != certificate.component_hashes.get("model"):
            return "context-channel manifest reference model does not match the certificate"
        if manifest.get("reference_feature_schema") != TOKEN_REFERENCE_FEATURE_SCHEMA:
            return "context-channel manifest does not declare categorical-token f_ref"
        if manifest.get("reference_feature_columns") != ["__token__", context_column]:
            return "context-channel manifest reference inputs do not match token and context"
        if manifest.get("reference_categorical_columns") != ["__token__"]:
            return "context-channel manifest does not mark the token categorical"
    if manifest.get("sensitive_coarsening") != {
        "missing": "__UNKNOWN__",
        "unseen": "__UNKNOWN__",
    }:
        return "context-channel manifest does not declare unified sensitive fallback"
    repair = certificate.coverage.get("post_solve_repair")
    if repair is not None and manifest.get("released_channel_policy") != (
        "uniform_full_support_postsolve_repair"
    ):
        return "context-channel manifest does not identify the repaired released channel"
    design_name = config.get("context_design")
    design = manifest.get("designs", {}).get(design_name)
    if not isinstance(design, dict):
        return "certificate design is missing from the context-channel manifest"
    channel_hash = hash_array(np.asarray(certificate.channel, dtype=float))
    context_value = str(config.get("public_context_value", ""))
    if design.get("contexts", {}).get(context_value) != channel_hash:
        return "context-channel manifest does not bind the certified context channel"
    if repair is not None:
        manifest_repair = design.get("context_repairs", {}).get(context_value, {})
        if manifest_repair.get("method") != repair.get("method"):
            return "context-channel manifest repair method does not match the certificate"
        if not np.isclose(
            float(manifest_repair.get("mixing_weight", -1)),
            float(repair.get("mixing_weight", -2)),
            atol=0,
            rtol=0,
        ):
            return "context-channel manifest repair weight does not match the certificate"
    if design.get("assignment_hash") != certificate.component_hashes.get("partition"):
        return "context-channel manifest partition does not match the certificate"
    if design.get("decoder_hash") != certificate.component_hashes.get("decoder"):
        return "context-channel manifest decoder does not match the certificate"
    dimension = len(certificate.channel)
    if int(design.get("L", -1)) != dimension:
        return "context-channel manifest dimension does not match the certificate"
    expected_entries = len(design.get("contexts", {})) * dimension**2
    if int(design.get("table_entries", -1)) != expected_entries:
        return "context-channel manifest table size is inconsistent"
    return None


def _verify_provenance_metadata(certificate: Certificate) -> str | None:
    config = certificate.resolved_config
    if config.get("require_clean_worktree") is not True:
        return "certificate does not require a clean source worktree"
    if config.get("source_worktree_clean") is not True:
        return "certificate was not recorded as coming from a clean source worktree"
    source_sha = config.get("source_git_sha")
    if not source_sha or source_sha == "unknown":
        return "certificate is missing a committed source Git SHA"
    if source_sha != certificate.code_git_sha:
        return "certificate source Git SHA does not match code_git_sha"
    if config.get("rare_group_policy") == "merge_to_other":
        return "merge_to_other certificates are disabled without runtime coarsening"
    if config.get("dataset", "synthetic") in {"synthetic", "synthetic_theorem4"}:
        return None
    if config.get("sampling_assumption") != "user_day_iid":
        return "sampled-data certificate lacks the explicit user_day_iid assumption"
    if config.get("contribution_policy", "one-display-per-uuid-day") != (
        "one-display-per-uuid-day"
    ):
        return "user_day_iid inference requires one-display-per-uuid-day contributions"
    if certificate.split_identifiers != config.get("splits", {}):
        return "split identifiers do not match resolved config"
    expected_dp = {
        "epsilon": config.get("dp_hist_epsilon"),
        "delta": config.get("dp_hist_delta"),
        "contribution_policy": config.get(
            "contribution_policy", "one-display-per-uuid-day"
        ),
    }
    if certificate.dp_parameters != expected_dp:
        return "DP/contribution metadata do not match resolved config"
    coverage = certificate.coverage
    if config.get("public_context_policy") == "stratified":
        contexts = list(config.get("context_cols", []))
        selector = config.get("channel_selector_inputs", [])
        expected_selector = ["Z", "profile", *contexts]
        if selector != expected_selector:
            return "context-stratified channel selector does not match Z/profile/public context"
        protected = set(config.get("profiles", [""])[0].split("+"))
        if protected.intersection(selector):
            return "context-stratified channel selector uses a protected value"
        context_value = str(config.get("public_context_value", ""))
        if not context_value or str(coverage.get("public_context_value", "")) != context_value:
            return "public-context certificate scope is missing or inconsistent"
        group_contexts = {
            str(group.get("context", "all")) for group in certificate.groups
        }
        if group_contexts != {context_value}:
            return "protected groups do not match the declared public-context scope"
        if config.get("sensitive_fallback_policy") != "unified_unknown":
            return "context-stratified certificate requires unified sensitive fallback"
        if config.get("sensitive_unknown_value") != "__UNKNOWN__":
            return "context-stratified certificate has an unsupported unknown secret value"
    expected = int(coverage.get("expected_group_count", -1))
    observed = int(coverage.get("observed_group_count", -1))
    missing = int(coverage.get("missing_group_count", -1))
    missing_groups = coverage.get("missing_groups", [])
    if min(expected, observed, missing) < 0 or expected != observed + missing:
        return "group coverage counts are inconsistent"
    if len(missing_groups) != missing:
        return "missing-group list does not match coverage count"
    missing_policy = config.get("missing_group_policy", "force_cover")
    rare_policy = config.get("rare_group_policy", "force_cover")
    needs_cover = bool(
        (missing and missing_policy != "full_simplex")
        or (int(coverage.get("rare_group_count", 0)) and rare_policy == "force_cover")
        or int(coverage.get("hybrid_connectivity_gaps", 0))
    )
    if bool(coverage.get("requires_universal_cover")) != needs_cover:
        return "universal-cover requirement does not match the configured coverage policy"
    return None


def _rebuild_adjacency(certificate: Certificate) -> tuple[list[AdjacentPair] | None, str | None]:
    if not certificate.groups:
        return None, "certificate is missing structured protected groups"
    groups = [
        Group(value["profile"], tuple(value["values"]), value.get("context", "all"))
        for value in certificate.groups
    ]
    if {group.key() for group in groups} != set(certificate.boxes):
        return None, "protected-group keys do not match confidence-box keys"
    config = certificate.resolved_config
    try:
        rebuilt = build_adjacency(
            groups,
            config.get("adjacency", config.get("privacy_scope", "tuple_adjacent")),
            float(config.get("epsilon", 1.0)),
            config.get("epsilon_by_attr", {}),
        )
    except (TypeError, ValueError) as error:
        return None, f"adjacency reconstruction failed: {error}"
    embedded = [
        AdjacentPair(
            value["left"],
            value["right"],
            value["epsilon"],
            tuple(value.get("changed_attributes", ())),
        )
        for value in certificate.adjacency
    ]
    if set(rebuilt) != set(embedded):
        return None, "embedded adjacency does not match groups/config"
    return rebuilt, None


def _rebuild_boxes(
    certificate: Certificate, adjacency: Sequence[AdjacentPair]
) -> tuple[dict[str, ConfidenceBox] | None, str | None]:
    boxes = {key: ConfidenceBox.from_dict(value) for key, value in certificate.boxes.items()}
    if any(box.experimental for box in boxes.values()):
        return None, "experimental confidence boxes are not certifiable"
    methods = {box.method for box in boxes.values()}
    if sorted(methods) != certificate.confidence.get("methods"):
        return None, "confidence method summary does not match embedded boxes"
    expected_basis = "known_synthetic_population" if methods == {"point"} else "finite_sample"
    if certificate.confidence.get("coverage_basis") != expected_basis:
        return None, "confidence coverage basis does not match embedded boxes"
    expected_level = (
        1.0
        if methods == {"point"}
        else 1 - float(certificate.resolved_config.get("alpha_cert", 0.05))
    )
    if not np.isclose(certificate.confidence.get("overall_level"), expected_level):
        return None, "confidence level summary does not match resolved config"
    if not np.isclose(
        certificate.tv_radius,
        max((box.tv_radius for box in boxes.values()), default=0.0),
    ):
        return None, "TV radius summary does not match embedded boxes"
    if "point" in methods:
        if certificate.resolved_config.get("dataset", "synthetic") not in {
            "synthetic",
            "synthetic_theorem4",
        }:
            return None, "point confidence is only valid for known synthetic populations"
        return boxes, None
    if certificate.histogram_counts is None:
        return None, "sampled-data certificate is missing histogram counts"
    confidence_name = certificate.resolved_config.get("confidence", "cp_box")
    if confidence_name not in CONFIDENCE_REGISTRY:
        return None, f"unknown confidence construction: {confidence_name}"
    if confidence_name == "dp_aware_box":
        return None, "experimental DP-aware confidence cannot be certified"
    if set(certificate.histogram_counts) != set(boxes):
        return None, "histogram-count keys do not match confidence-box keys"
    factory = CONFIDENCE_REGISTRY[confidence_name]
    missing_policy = certificate.resolved_config.get("missing_group_policy", "force_cover")
    missing_keys = set(certificate.coverage.get("missing_groups", []))
    rebuilt: dict[str, ConfidenceBox] = {}
    for key, raw_counts in certificate.histogram_counts.items():
        counts = np.asarray(raw_counts, dtype=int)
        try:
            if counts.sum() == 0:
                if missing_policy != "full_simplex" or key not in missing_keys:
                    return None, (
                        "zero-count confidence group is not authorized for "
                        f"full-simplex completion: {key}"
                    )
                rebuilt[key] = confidence_box_from_counts(
                    counts,
                    confidence=confidence_name,
                    missing_group_policy=missing_policy,
                )
            else:
                if key in missing_keys:
                    return None, f"listed missing group has positive histogram count: {key}"
                rebuilt[key] = factory(
                    counts,
                    alpha=float(certificate.resolved_config.get("alpha_cert", 0.05)),
                    group_count=len(certificate.histogram_counts),
                    comparisons=max(1, len(adjacency)),
                    tv_radius=float(certificate.resolved_config.get("shift_tv", 0.0)),
                )
        except (TypeError, ValueError) as error:
            return None, f"confidence box reconstruction failed for {key}: {error}"
        if not _same_box(rebuilt[key], boxes[key]):
            return None, f"confidence box does not match embedded counts/config: {key}"
    return rebuilt, None


def _verify_certificate(path: str | Path, tolerance: float | None = None) -> VerificationResult:
    path = Path(path)
    certificate = Certificate.read(path)
    if certificate.version < 2:
        return _invalid("legacy certificates lack the inputs required for bundle verification")
    if certificate.verification_scope != "self_contained_bundle_consistency_and_robust_constraints":
        return _invalid("certificate declares an unsupported verification scope")
    tol = DEFAULT_VERIFICATION_TOLERANCE if tolerance is None else float(tolerance)
    if not 0 <= tol <= DEFAULT_VERIFICATION_TOLERANCE:
        return _invalid(
            f"verification tolerance must be between 0 and {DEFAULT_VERIFICATION_TOLERANCE:g}"
        )
    hash_error = _verify_component_hashes(certificate, path)
    if hash_error:
        return _invalid(hash_error)
    bundle_error = _verify_profile_bundle(certificate, path)
    if bundle_error:
        return _invalid(bundle_error)
    context_bundle_error = _verify_context_channel_manifest(certificate, path)
    if context_bundle_error:
        return _invalid(context_bundle_error)
    metadata_error = _verify_provenance_metadata(certificate)
    if metadata_error:
        return _invalid(metadata_error)
    channel = np.asarray(certificate.channel, dtype=float)
    if certificate.assignment is None or certificate.decoder is None:
        return _invalid("certificate is missing partition/decoder deployment data")
    assignment = np.asarray(certificate.assignment, dtype=int)
    decoder = np.asarray(certificate.decoder, dtype=float)
    try:
        validate_decoder(decoder, assignment)
        lift_block_channel(channel, assignment, decoder)
    except ValueError as error:
        return _invalid(f"invalid block-to-token deployment lift: {error}")
    if certificate.coverage.get("requires_universal_cover") and not np.allclose(
        channel, channel[0][None, :], atol=tol, rtol=0
    ):
        return _invalid("incomplete group coverage requires an input-independent common cover")
    adjacency, adjacency_error = _rebuild_adjacency(certificate)
    if adjacency_error or adjacency is None:
        return _invalid(adjacency_error or "adjacency reconstruction failed")
    boxes, box_error = _rebuild_boxes(certificate, adjacency)
    if box_error or boxes is None:
        return _invalid(box_error or "confidence box reconstruction failed")
    expected_target = max((pair.epsilon for pair in adjacency), default=0.0)
    if not np.isclose(certificate.target_epsilon, expected_target):
        return _invalid("target epsilon summary does not match reconstructed adjacency")
    result = verify_robust_channel(channel, boxes, adjacency, tolerance=tol)
    repair = certificate.coverage.get("post_solve_repair")
    if repair is not None:
        if not isinstance(repair, dict) or repair.get("method") != (
            "uniform_full_support_mixing"
        ):
            return _invalid("certificate declares an unsupported post-solve repair")
        conservative = verify_robust_channel_conservative(channel, boxes, adjacency)
        if not conservative.valid:
            return _invalid("repaired channel fails tolerance-free Decimal privacy verification")
        if not np.isfinite(conservative.realized_epsilon):
            return _invalid("repaired channel has a non-finite pure-epsilon ratio")
        if conservative.realized_epsilon > expected_target:
            return _invalid("repaired channel exceeds its target pure-epsilon budget")
        declared = repair.get("conservative_decimal_verification", {})
        if declared.get("valid") is not True:
            return _invalid("certificate does not declare a valid conservative repair check")
        if int(declared.get("checked_constraints", -1)) != conservative.checked_constraints:
            return _invalid("conservative repair constraint count does not match")
        if int(repair.get("post_repair_zero_denominator_positive_numerator_count", -1)) != 0:
            return _invalid("certificate repair leaves a positive-over-zero ratio")
    if result.valid and not np.isclose(
        certificate.realized_worst_case_epsilon,
        result.realized_epsilon,
        atol=tol,
        rtol=0,
    ):
        return _invalid("realized epsilon summary does not match recomputed constraints")
    return result


def verify_certificate(path: str | Path, tolerance: float | None = None) -> VerificationResult:
    try:
        return _verify_certificate(path, tolerance)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return _invalid(f"malformed or incomplete certificate bundle: {error}")
