from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def parse_csv_list(
    value: str | list[Any] | tuple[Any, ...] | None,
    cast: type = str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[Any]:
    """Parse comma-separated CLI/config values, deduplicate, sort and validate."""
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple)) else value.split(",")
    parsed: list[Any] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        converted = cast(text)
        if minimum is not None and converted < minimum:
            raise ValueError(f"{converted} is below minimum {minimum}")
        if maximum is not None and converted > maximum:
            raise ValueError(f"{converted} exceeds maximum {maximum}")
        parsed.append(converted)
    return sorted(set(parsed))


def parse_profiles(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list) else value.split(";")
    profiles = ["+".join(x.strip() for x in str(v).split("+") if x.strip()) for v in raw]
    profiles = [v for v in profiles if v]
    if any(not profile for profile in profiles):
        raise ValueError("profiles must contain non-empty attribute names")
    return list(dict.fromkeys(profiles))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        elif value is not None:
            result[key] = value
    return result


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    base_config = config.pop("base_config", None)
    if base_config:
        base_path = Path(base_config)
        if not base_path.is_absolute() and not base_path.exists():
            base_path = path.parent / base_path.name
        config = deep_merge(load_config(base_path), config)
    config = deep_merge(config, overrides or {})
    return validate_config(config)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    for key in ("K", "L"):
        if key in cfg:
            cfg[key] = int(cfg[key])
    if "K" in cfg and not 1 <= cfg["K"] <= 4096:
        raise ValueError("K must satisfy 1 <= K <= 4096")
    if "L" in cfg and "K" in cfg and not 1 <= cfg["L"] <= cfg["K"]:
        raise ValueError("L must satisfy 1 <= L <= K")
    for key, cast, low, high in (
        ("K_list", int, 1, 4096),
        ("L_list", int, 1, 4096),
        ("epsilon_list", float, 0, None),
        ("shift_tv_list", float, 0, 1),
        ("seeds", int, 0, None),
        ("target_ctr_list", float, 0, 1),
    ):
        if key in cfg:
            cfg[key] = parse_csv_list(cfg[key], cast, minimum=low, maximum=high)
    if "profiles" in cfg:
        cfg["profiles"] = parse_profiles(cfg["profiles"])
    for l_value in cfg.get("L_list", []):
        if "K" in cfg and l_value > cfg["K"] and not cfg.get("K_list"):
            raise ValueError(f"invalid combination L={l_value} > K={cfg['K']}")
    if cfg.get("adjacency", cfg.get("privacy_scope", "tuple_adjacent")) not in {
        "tuple_adjacent",
        "marginal",
        "joint_all_pairs",
    }:
        raise ValueError("adjacency must be tuple_adjacent, marginal, or joint_all_pairs")
    rare_group_policy = cfg.get("rare_group_policy", "force_cover")
    if rare_group_policy not in {
        "fail",
        "merge_to_other",
        "force_cover",
    }:
        raise ValueError("rare_group_policy must be fail, merge_to_other, or force_cover")
    if rare_group_policy == "merge_to_other":
        raise ValueError(
            "merge_to_other is disabled until the same frozen coarsening map is "
            "applied at runtime; use fail or force_cover"
        )
    if cfg.get("dataset", "synthetic") not in {
        "synthetic",
        "synthetic_theorem4",
    } and cfg.get("confidence") == "point":
        raise ValueError(
            "point confidence is only valid for a known synthetic population; "
            "use cp_box or hoeffding_box for sampled data"
        )
    if cfg.get("profile_weighting", "global_design") != "global_design":
        raise ValueError(
            "profile_weighting is not identifiable without observed profile assignments; "
            "the supported utility objective is global_design"
        )
    if cfg.get("audit_population_assumption") not in {
        None,
        "stationary",
        "covered_by_shift_set",
    }:
        raise ValueError(
            "audit_population_assumption must be stationary or covered_by_shift_set"
        )
    if cfg.get("audit_population_assumption") and not cfg.get("audit_bridge_evidence"):
        raise ValueError(
            "audit_population_assumption requires audit_bridge_evidence; a config label "
            "alone does not establish a population bridge"
        )
    if cfg.get("dataset", "synthetic") not in {
        "synthetic",
        "synthetic_theorem4",
    }:
        if cfg.get("sampling_assumption") != "user_day_iid":
            raise ValueError(
                "finite-sample Criteo certificates and lower audits require the explicit "
                "sampling_assumption: user_day_iid"
            )
        if cfg.get("contribution_policy", "one-display-per-uuid-day") != (
            "one-display-per-uuid-day"
        ):
            raise ValueError(
                "user_day_iid inference requires one-display-per-uuid-day contributions"
            )
    solver_tolerance = float(cfg.get("solver_tolerance", 1e-8))
    if not 0 < solver_tolerance <= 1e-8:
        raise ValueError("solver_tolerance for certificate runs must be in (0, 1e-8]")
    return cfg


def canonical_json(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def run_id(config: dict[str, Any], length: int = 16) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:length]
