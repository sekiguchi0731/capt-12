from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from capt12.config import run_id, validate_config
from capt12.pipeline import run_pipeline, run_theorem4_grid

MECHANISM_ALIASES = {
    "cover": "common_cover",
    "krr": "k_ary_rr",
    "koc": "scalar_keep_or_cover",
}


def normalize_mechanisms(values: list[str]) -> list[str]:
    return [MECHANISM_ALIASES.get(value, value) for value in values]


def run_grid(
    config: dict[str, Any],
    *,
    resume: bool = False,
    dry_run: bool = False,
    max_rows: int | None = None,
) -> pd.DataFrame:
    if config.get("dataset") == "synthetic_theorem4":
        if dry_run:
            return pd.DataFrame([{"status": "dry_run", "dataset": "synthetic_theorem4"}])
        return run_theorem4_grid(config, resume=resume)
    dimensions = {
        "seed": config.get("seeds", [config.get("seed", 0)]),
        "phi": config.get("phi_list", [config.get("phi", "hash")]),
        "K": config.get("K_list", [config.get("K", 64)]),
        "L": config.get("L_list", [config.get("L", 16)]),
        "epsilon": config.get("epsilon_list", [config.get("epsilon", 1.0)]),
        "distortion": config.get("distortion_list", [config.get("distortion", "bernoulli_kl")]),
        "partition": config.get("partition_list", [config.get("partition", "frequency_balanced")]),
        "decoder": config.get("decoder_list", [config.get("decoder", "design_frequency")]),
        "shift_tv": config.get("shift_tv_list", [config.get("shift_tv", 0.0)]),
    }
    mechanisms = normalize_mechanisms(config.get("mechanism_list", config.get("mechanisms", ["capt_block"])))
    rows = []
    keys = list(dimensions)
    for values in itertools.product(*(dimensions[key] for key in keys)):
        overrides = dict(zip(keys, values, strict=True))
        cfg = {**config, **overrides, "mechanisms": mechanisms}
        if int(cfg["L"]) > int(cfg["K"]):
            rows.append({**overrides, "status": "skipped", "skip_reason": "L>K"})
            continue
        if "theorem4_envelope" in mechanisms and cfg.get("distortion") != "retention":
            rows.append(
                {
                    **overrides,
                    "status": "skipped",
                    "skip_reason": "theorem4_envelope is valid only for exact-token retention",
                }
            )
            continue
        try:
            cfg = validate_config(cfg)
        except ValueError as error:
            rows.append({**overrides, "status": "skipped", "skip_reason": str(error)})
            continue
        target = Path(cfg.get("output_dir", "outputs/runs")) / run_id(cfg) / "metrics.parquet"
        if dry_run:
            rows.append({**overrides, "status": "dry_run", "run_id": run_id(cfg)})
        elif resume and target.exists():
            result = pd.read_parquet(target)
            result["grid_status"] = "resumed"
            rows.extend(result.to_dict("records"))
        else:
            _, result = run_pipeline(cfg, max_rows=max_rows)
            result["grid_status"] = "complete"
            rows.extend(result.to_dict("records"))
    result = pd.DataFrame(rows)
    output = Path(config.get("output_dir", "outputs/runs"))
    output.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        result.to_parquet(output / "grid_results.parquet", index=False)
    (output / "grid_manifest.json").write_text(
        json.dumps(
            {
                "runs_or_skips": len(result),
                "dry_run": dry_run,
                "resume": resume,
                "dimensions": {key: list(value) for key, value in dimensions.items()},
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return result

