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
    if cfg.get("require_clean_worktree") is False:
        raise ValueError("certificate-producing runs cannot disable the clean-worktree requirement")
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
        "confidence_box",
    }:
        raise ValueError(
            "rare_group_policy must be fail, merge_to_other, force_cover, or confidence_box"
        )
    if rare_group_policy == "merge_to_other":
        raise ValueError(
            "merge_to_other is disabled until the same frozen coarsening map is "
            "applied at runtime; use fail or force_cover"
        )
    missing_group_policy = cfg.get("missing_group_policy", "force_cover")
    if missing_group_policy not in {"fail", "force_cover", "full_simplex"}:
        raise ValueError("missing_group_policy must be fail, force_cover, or full_simplex")
    if missing_group_policy == "full_simplex" and rare_group_policy != "confidence_box":
        raise ValueError(
            "full-simplex completion requires rare_group_policy: confidence_box so "
            "every observed group keeps its finite-sample confidence set"
        )
    sensitive_fallback = cfg.get("sensitive_fallback_policy", "separate_missing_other")
    if sensitive_fallback not in {"separate_missing_other", "unified_unknown"}:
        raise ValueError(
            "sensitive_fallback_policy must be separate_missing_other or unified_unknown"
        )
    if (
        sensitive_fallback == "unified_unknown"
        and cfg.get("sensitive_unknown_value", "__UNKNOWN__") != "__UNKNOWN__"
    ):
        raise ValueError("unified sensitive fallback must use __UNKNOWN__")
    public_context_policy = cfg.get("public_context_policy", "shared")
    if public_context_policy not in {"shared", "stratified"}:
        raise ValueError("public_context_policy must be shared or stratified")
    if public_context_policy == "stratified" and len(cfg.get("context_cols", [])) != 1:
        raise ValueError("stratified public-context CAPT currently requires one context column")
    if (
        cfg.get("dataset", "synthetic")
        not in {
            "synthetic",
            "synthetic_theorem4",
        }
        and cfg.get("confidence") == "point"
    ):
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
        raise ValueError("audit_population_assumption must be stationary or covered_by_shift_set")
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
    heartbeat_seconds = float(cfg.get("solver_heartbeat_seconds", 60.0))
    if heartbeat_seconds <= 0:
        raise ValueError("solver_heartbeat_seconds must be positive")
    if not isinstance(cfg.get("solver_verbose", False), bool):
        raise ValueError("solver_verbose must be a boolean")
    if cfg.get("time_limit") is not None:
        time_limit = float(cfg["time_limit"])
        if time_limit <= 0:
            raise ValueError("time_limit must be positive")
        cfg["time_limit"] = time_limit
    cut_formulation = cfg.get("robust_cut_formulation", "paired_witness")
    if cut_formulation not in {"paired_witness", "shared_support_bounds"}:
        raise ValueError("robust_cut_formulation must be paired_witness or shared_support_bounds")
    if not isinstance(cfg.get("resume_cutting_plane", False), bool):
        raise ValueError("resume_cutting_plane must be a boolean")
    checkpoint_every = int(cfg.get("cutting_plane_checkpoint_every", 1))
    if checkpoint_every < 1:
        raise ValueError("cutting_plane_checkpoint_every must be positive")
    cfg["cutting_plane_checkpoint_every"] = checkpoint_every
    if "context_designs" in cfg:
        valid_context_designs = {
            "joint_kmedoids_cost_medoid_L8",
            "joint_kmedoids_cost_medoid_L16",
            "joint_kmedoids_cost_medoid_L32",
            "singleton_identity_L64",
        }
        context_designs = list(dict.fromkeys(map(str, cfg["context_designs"])))
        if len(context_designs) != 1 or context_designs[0] not in valid_context_designs:
            raise ValueError(
                "context_designs must select exactly one prescribed L8, L16, L32, or L64 design"
            )
        cfg["context_designs"] = context_designs
    if "certificate_channel_repair" in cfg:
        if cfg["certificate_channel_repair"] != "uniform_full_support_mixing":
            raise ValueError("certificate_channel_repair must be uniform_full_support_mixing")
        repair_margin = float(cfg.get("certificate_repair_margin", 1e-10))
        if not 0 < repair_margin <= 1e-8:
            raise ValueError("certificate_repair_margin must be in (0, 1e-8]")
        cfg["certificate_repair_margin"] = repair_margin
    if "frozen_design_seed" in cfg:
        frozen_design_seed = int(cfg["frozen_design_seed"])
        if frozen_design_seed < 0:
            raise ValueError("frozen_design_seed must be nonnegative")
        cfg["frozen_design_seed"] = frozen_design_seed
    context_experiment = public_context_policy == "stratified" or any(
        key in cfg
        for key in (
            "context_utility_objective",
            "context_representation_mode",
            "hybrid_empirical_weight",
            "alpha_audit",
            "audit_seed",
        )
    )
    if context_experiment:
        utility_objective = str(cfg.get("context_utility_objective", "teacher_kl"))
        valid_utility_objectives = {
            "teacher_kl",
            "empirical_logloss",
            "hybrid_logloss_kl",
        }
        if utility_objective not in valid_utility_objectives:
            raise ValueError(
                "context_utility_objective must be teacher_kl, empirical_logloss, "
                "or hybrid_logloss_kl"
            )
        empirical_weight = float(cfg.get("hybrid_empirical_weight", 0.5))
        if not 0 <= empirical_weight <= 1:
            raise ValueError("hybrid_empirical_weight must be in [0, 1]")
        cfg["context_utility_objective"] = utility_objective
        cfg["hybrid_empirical_weight"] = empirical_weight
        representation_mode = str(cfg.get("context_representation_mode", "teacher_kl_fixed"))
        if representation_mode not in {"teacher_kl_fixed", "objective_aligned"}:
            raise ValueError(
                "context_representation_mode must be teacher_kl_fixed or objective_aligned"
            )
        cfg["context_representation_mode"] = representation_mode
        alpha_audit = float(cfg.get("alpha_audit", cfg.get("alpha_cert", 0.05)))
        if not 0 < alpha_audit < 1:
            raise ValueError("alpha_audit must be in (0, 1)")
        cfg["alpha_audit"] = alpha_audit
        audit_seed = int(cfg.get("audit_seed", 2026))
        if audit_seed < 0:
            raise ValueError("audit_seed must be nonnegative")
        cfg["audit_seed"] = audit_seed
    return cfg


def canonical_json(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def run_id(config: dict[str, Any], length: int = 16) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:length]
