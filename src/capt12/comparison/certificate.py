from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from capt12.certification.artifact import hash_array
from capt12.certification.robust import VerificationResult, verify_robust_channel
from capt12.confidence.boxes import ConfidenceBox
from capt12.mechanisms.lp import validate_channel
from capt12.privacy.adjacency import AdjacentPair, Group, build_adjacency
from capt12.utils.artifacts import sha256_file


@dataclass(frozen=True)
class FactorizedCertificateVerification:
    valid: bool
    realized_epsilon: float
    max_violation: float
    checked_constraints: int
    context_results: dict[str, dict[str, Any]]
    error: str | None = None


def _released_channel(
    input_to_block: np.ndarray,
    decoder: np.ndarray,
    cover: np.ndarray,
    mixing_weight: float,
) -> np.ndarray:
    raw = np.asarray(input_to_block, dtype=float) @ np.asarray(decoder, dtype=float)
    channel = (1 - float(mixing_weight)) * raw + float(mixing_weight) * np.asarray(
        cover, dtype=float
    )[None, :]
    validate_channel(channel)
    return channel


def _verification_payload(value: VerificationResult) -> dict[str, Any]:
    return {
        "valid": bool(value.valid),
        "realized_epsilon": float(value.realized_epsilon),
        "max_violation": float(value.max_violation),
        "checked_constraints": int(value.checked_constraints),
        "worst_case": value.worst_case,
    }


def write_factorized_certificate(
    manifest_path: str | Path,
    *,
    input_to_block: Mapping[str, np.ndarray],
    decoder: np.ndarray,
    cover_distribution: np.ndarray,
    mixing_weights: Mapping[str, float],
    boxes: Mapping[str, Mapping[str, ConfidenceBox]],
    adjacency: Mapping[str, Sequence[AdjacentPair]],
    groups: Mapping[str, Sequence[Group]],
    adjacency_spec: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tolerance: float = 1e-8,
) -> tuple[Path, FactorizedCertificateVerification]:
    """Write a compact, independently verifiable certificate for Q=(pi D) wrapper.

    The artifact stores the K-by-L input-to-block factors and the shared L-by-K
    decoder, not a redundant K-by-K JSON matrix.  Verification reconstructs
    every exact token channel and runs the existing robust support verifier.
    """
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    contexts = tuple(sorted(map(str, input_to_block)))
    if (
        not contexts
        or set(contexts) != set(map(str, boxes))
        or set(contexts) != set(map(str, adjacency))
        or set(contexts) != set(map(str, groups))
    ):
        raise ValueError("factorized certificate contexts must be nonempty and aligned")
    if set(contexts) != set(map(str, mixing_weights)):
        raise ValueError("every certificate context requires one mixing weight")

    factors = [np.asarray(input_to_block[context], dtype=float) for context in contexts]
    if len({value.shape for value in factors}) != 1:
        raise ValueError("all factorized channels must share one K-by-L shape")
    k, block_count = factors[0].shape
    decoder_array = np.asarray(decoder, dtype=float)
    cover = np.asarray(cover_distribution, dtype=float)
    if decoder_array.shape != (block_count, k):
        raise ValueError("decoder must have shape L-by-K")
    if cover.shape != (k,) or np.any(cover <= 0) or not np.isclose(cover.sum(), 1.0):
        raise ValueError("certificate cover must be a full-support K-vector")

    group_keys: list[str] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    nominal: list[np.ndarray] = []
    box_metadata: list[dict[str, Any]] = []
    context_payload: dict[str, dict[str, Any]] = {}
    context_results: dict[str, dict[str, Any]] = {}
    all_valid = True
    realized = -math.inf
    max_violation = -math.inf
    checked = 0
    for context, factor in zip(contexts, factors, strict=True):
        groups_here = list(groups[context])
        if {group.key() for group in groups_here} != set(boxes[context]):
            raise ValueError("structured groups must match factorized certificate boxes")
        start = len(group_keys)
        for key in sorted(boxes[context]):
            box = boxes[context][key]
            if box.lower.shape != (k,):
                raise ValueError("confidence-box alphabet does not match factorized channel")
            group_keys.append(key)
            lower.append(box.lower)
            upper.append(box.upper)
            nominal.append(box.nominal)
            box_metadata.append(
                {
                    "method": box.method,
                    "alpha_familywise": float(box.alpha_familywise),
                    "tv_radius": float(box.tv_radius),
                    "experimental": bool(box.experimental),
                }
            )
        weight = float(mixing_weights[context])
        if not 0 <= weight <= 1:
            raise ValueError("mixing weights must lie in [0,1]")
        channel = _released_channel(factor, decoder_array, cover, weight)
        result = verify_robust_channel(
            channel,
            boxes[context],
            adjacency[context],
            tolerance=tolerance,
        )
        payload = _verification_payload(result)
        context_results[context] = payload
        all_valid &= result.valid
        realized = max(realized, result.realized_epsilon)
        max_violation = max(max_violation, result.max_violation)
        checked += result.checked_constraints
        context_payload[context] = {
            "group_start": start,
            "group_stop": len(group_keys),
            "mixing_weight": weight,
            "input_to_block_sha256": hash_array(factor),
            "released_channel_sha256": hash_array(channel),
            "adjacency": [asdict(value) for value in adjacency[context]],
            "groups": [asdict(value) for value in groups_here],
            "verification": payload,
        }
    if not all_valid:
        raise ValueError("cannot serialize a factorized certificate that fails verification")
    if realized == -math.inf:
        realized = 0.0
    if max_violation == -math.inf:
        max_violation = 0.0

    arrays_path = destination.with_suffix(".npz")
    np.savez_compressed(
        arrays_path,
        contexts=np.asarray(contexts, dtype=str),
        input_to_block=np.stack(factors),
        decoder=decoder_array,
        cover_distribution=cover,
        group_keys=np.asarray(group_keys, dtype=str),
        lower=np.stack(lower),
        upper=np.stack(upper),
        nominal=np.stack(nominal),
    )
    manifest = {
        "schema_version": 1,
        "certificate_type": "factorized_finite_output_robust_profile_privacy",
        "created_at": datetime.now(UTC).isoformat(),
        "factorization": "Q_b=(1-lambda_b)(pi_b D)+lambda_b 1 mu^T",
        "arrays_file": arrays_path.name,
        "arrays_sha256": sha256_file(arrays_path),
        "decoder_sha256": hash_array(decoder_array),
        "cover_sha256": hash_array(cover),
        "K": k,
        "L": block_count,
        "contexts": context_payload,
        "adjacency_spec": dict(adjacency_spec),
        "box_metadata": box_metadata,
        "metadata": dict(metadata),
        "verification": {
            "valid": True,
            "realized_epsilon": realized,
            "max_violation": max_violation,
            "checked_constraints": checked,
            "tolerance": tolerance,
        },
    }
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verification = verify_factorized_certificate(destination, tolerance=tolerance)
    if not verification.valid:
        raise RuntimeError(
            f"independent factorized certificate verification failed: {destination}"
        )
    return destination, verification


def verify_factorized_certificate(
    manifest_path: str | Path,
    *,
    tolerance: float | None = None,
) -> FactorizedCertificateVerification:
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported factorized certificate schema")
        arrays_path = path.parent / str(manifest["arrays_file"])
        if sha256_file(arrays_path) != manifest["arrays_sha256"]:
            raise ValueError("factorized certificate array hash mismatch")
        with np.load(arrays_path, allow_pickle=False) as payload:
            contexts = tuple(map(str, payload["contexts"].tolist()))
            factors = np.asarray(payload["input_to_block"], dtype=float)
            decoder = np.asarray(payload["decoder"], dtype=float)
            cover = np.asarray(payload["cover_distribution"], dtype=float)
            group_keys = tuple(map(str, payload["group_keys"].tolist()))
            lower = np.asarray(payload["lower"], dtype=float)
            upper = np.asarray(payload["upper"], dtype=float)
            nominal = np.asarray(payload["nominal"], dtype=float)
        if hash_array(decoder) != manifest["decoder_sha256"] or hash_array(cover) != manifest[
            "cover_sha256"
        ]:
            raise ValueError("factorized certificate component hash mismatch")
        if factors.shape != (len(contexts), int(manifest["K"]), int(manifest["L"])):
            raise ValueError("factorized certificate factor shape mismatch")
        expected_box_shape = (len(group_keys), int(manifest["K"]))
        if not (lower.shape == upper.shape == nominal.shape == expected_box_shape):
            raise ValueError("factorized certificate box shape mismatch")
        box_metadata = manifest["box_metadata"]
        if len(box_metadata) != len(group_keys):
            raise ValueError("factorized certificate box metadata mismatch")
        verification_tolerance = (
            float(manifest["verification"]["tolerance"])
            if tolerance is None
            else float(tolerance)
        )
        results: dict[str, dict[str, Any]] = {}
        valid = True
        realized = -math.inf
        max_violation = -math.inf
        checked = 0
        for index, context in enumerate(contexts):
            item = manifest["contexts"][context]
            start, stop = int(item["group_start"]), int(item["group_stop"])
            boxes_here = {
                group_keys[position]: ConfidenceBox(
                    lower[position],
                    upper[position],
                    nominal[position],
                    **box_metadata[position],
                )
                for position in range(start, stop)
            }
            adjacency_here = [
                AdjacentPair(
                    value["left"],
                    value["right"],
                    float(value["epsilon"]),
                    tuple(value.get("changed_attributes", ())),
                )
                for value in item["adjacency"]
            ]
            groups_here = [
                Group(
                    value["profile"],
                    tuple(value["values"]),
                    value.get("context", "all"),
                )
                for value in item["groups"]
            ]
            if {group.key() for group in groups_here} != set(boxes_here):
                raise ValueError("factorized certificate group/box mismatch")
            spec = manifest["adjacency_spec"]
            expected_adjacency = build_adjacency(
                groups_here,
                str(spec["mode"]),
                float(spec["epsilon"]),
                {
                    str(key): float(value)
                    for key, value in spec.get("epsilon_by_attr", {}).items()
                },
            )
            if set(expected_adjacency) != set(adjacency_here):
                raise ValueError("factorized certificate adjacency does not match groups/spec")
            factor = factors[index]
            if hash_array(factor) != item["input_to_block_sha256"]:
                raise ValueError("factorized certificate input factor hash mismatch")
            channel = _released_channel(
                factor,
                decoder,
                cover,
                float(item["mixing_weight"]),
            )
            if hash_array(channel) != item["released_channel_sha256"]:
                raise ValueError("factorized certificate released-channel hash mismatch")
            result = verify_robust_channel(
                channel,
                boxes_here,
                adjacency_here,
                tolerance=verification_tolerance,
            )
            results[context] = _verification_payload(result)
            valid &= result.valid
            realized = max(realized, result.realized_epsilon)
            max_violation = max(max_violation, result.max_violation)
            checked += result.checked_constraints
        if realized == -math.inf:
            realized = 0.0
        if max_violation == -math.inf:
            max_violation = 0.0
        recorded = manifest["verification"]
        if int(recorded["checked_constraints"]) != checked or not np.isclose(
            float(recorded["realized_epsilon"]), realized, equal_nan=True
        ):
            raise ValueError("factorized certificate recorded verification mismatch")
        return FactorizedCertificateVerification(
            valid=bool(valid),
            realized_epsilon=float(realized),
            max_violation=float(max_violation),
            checked_constraints=int(checked),
            context_results=results,
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        return FactorizedCertificateVerification(
            valid=False,
            realized_epsilon=math.inf,
            max_violation=math.inf,
            checked_constraints=0,
            context_results={},
            error=str(error),
        )
