from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer

from capt12.certification.artifact import verify_certificate as verify_certificate_file
from capt12.confidence.boxes import CONFIDENCE_REGISTRY
from capt12.config import load_config, parse_csv_list, parse_profiles, run_id
from capt12.data.inspect import inspect_parquet, write_inspection_json
from capt12.decoders.registry import DECODER_REGISTRY
from capt12.distortions.registry import DISTORTION_REGISTRY
from capt12.encoders.base import ENCODER_REGISTRY
from capt12.grid import run_grid
from capt12.mechanisms.baselines import BASELINES
from capt12.models.reference import MODEL_REGISTRY
from capt12.partitions.registry import PARTITION_REGISTRY
from capt12.pipeline import run_pipeline
from capt12.utils.artifacts import capt12_source_root

app = typer.Typer(no_args_is_help=True, help="CAPT-12 reproducible research CLI")


def _load(path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return load_config(
        path, {key: value for key, value in (overrides or {}).items() if value is not None}
    )


def _resolve_git_commit(revision: str) -> str:
    try:
        root = capt12_source_root()
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", f"{revision}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot resolve Git commit: {revision}") from error
    return result.stdout.strip()


@app.command("resolve-run-id")
def resolve_run_id(
    revision: str = typer.Argument(..., help="Git commit, short SHA, tag, or HEAD"),
    config: Path = typer.Option(..., "--config", exists=True),
    frozen_design_seed: int | None = typer.Option(
        None,
        "--frozen-design-seed",
        min=0,
        help="Override the frozen encoder/partition design seed.",
    ),
    epsilon: float | None = typer.Option(
        None,
        "--epsilon",
        min=0,
        help="Override the privacy budget (context experiments require epsilon > 0).",
    ),
    utility_objective: str | None = typer.Option(
        None,
        "--utility-objective",
        help="Context LP objective: teacher_kl, empirical_logloss, or hybrid_logloss_kl.",
    ),
    representation_mode: str | None = typer.Option(
        None,
        "--representation-mode",
        help="Block/decoder cost: teacher_kl_fixed or objective_aligned.",
    ),
    hybrid_empirical_weight: float | None = typer.Option(
        None,
        "--hybrid-empirical-weight",
        min=0,
        max=1,
        help="Empirical-label weight for hybrid_logloss_kl.",
    ),
    context_r_pooling_weight: float | None = typer.Option(
        None,
        "--context-r-pooling-weight",
        min=0,
        max=1,
        help=(
            "Convex shrinkage rho for context R: (1-rho)R_context + "
            "rho R_shared; 0 keeps independent context R and 1 uses the shared target."
        ),
    ),
) -> None:
    """Predict the deterministic output ID for a config and source commit."""
    try:
        source_git_sha = _resolve_git_commit(revision)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="revision") from error
    resolved = _load(
        config,
        {
            "frozen_design_seed": frozen_design_seed,
            "epsilon": epsilon,
            "context_utility_objective": utility_objective,
            "context_representation_mode": representation_mode,
            "hybrid_empirical_weight": hybrid_empirical_weight,
            "context_r_pooling_weight": context_r_pooling_weight,
        },
    )
    # Match record_source_provenance() exactly without requiring the requested
    # revision to be the currently checked-out HEAD.
    resolved["require_clean_worktree"] = True
    resolved["source_worktree_clean"] = True
    resolved["source_git_sha"] = source_git_sha
    identifier = run_id(resolved)
    output_path = Path(resolved.get("output_dir", "outputs/runs")) / identifier
    typer.echo(
        json.dumps(
            {
                "config": str(config),
                "requested_revision": revision,
                "source_git_sha": source_git_sha,
                "run_id": identifier,
                "output_path": str(output_path),
                "output_exists": output_path.exists(),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("inspect-data")
def inspect_data(
    data_root: Path = typer.Option(Path("data/CriteoPrivateAd_release/data"), "--data-root"),
    output: Path | None = typer.Option(None, "--output"),
    sample_rows_per_file: int = typer.Option(2048, "--sample-rows-per-file", min=0),
) -> None:
    """Inspect Parquet footers/schema and a bounded sample without a full load."""
    inspection = inspect_parquet(data_root, sample_rows_per_file)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_inspection_json(inspection, output)
    typer.echo(json.dumps(inspection.to_dict(), indent=2, sort_keys=True))


@app.command("list-components")
def list_components() -> None:
    components = {
        "encoders": sorted(ENCODER_REGISTRY),
        "reference_models": sorted(MODEL_REGISTRY),
        "distortions": sorted(DISTORTION_REGISTRY),
        "partitions": sorted(PARTITION_REGISTRY),
        "decoders": sorted(DECODER_REGISTRY),
        "confidence": sorted(CONFIDENCE_REGISTRY),
        "baselines": sorted(BASELINES),
        "mechanisms": ["capt_block", "capt_full"],
        "bounds": ["theorem4_envelope"],
    }
    typer.echo(json.dumps(components, indent=2))


def _stage_command(config: Path, max_rows: int | None, stage: str) -> None:
    cfg = _load(config, {"max_rows": max_rows})
    path, metrics = run_pipeline(cfg, max_rows=max_rows)
    typer.echo(json.dumps({"stage": stage, "run": str(path), "rows": len(metrics)}, indent=2))


@app.command("prepare")
def prepare(
    config: Path = typer.Option(..., "--config", exists=True),
    data_root: Path | None = typer.Option(None, "--data-root"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    max_rows: int | None = typer.Option(None, "--max-rows", min=1),
    sample_frac: float | None = typer.Option(None, "--sample-frac", min=0, max=1),
) -> None:
    cfg = _load(
        config,
        {
            "data_root": str(data_root) if data_root else None,
            "output_dir": str(output_dir) if output_dir else None,
            "max_rows": max_rows,
            "sample_frac": sample_frac,
        },
    )
    path, metrics = run_pipeline(cfg, max_rows=max_rows)
    typer.echo(json.dumps({"stage": "prepare", "run": str(path), "rows": len(metrics)}, indent=2))


@app.command("train-ref-model")
def train_ref_model(
    config: Path = typer.Option(..., "--config", exists=True),
    max_rows: int | None = typer.Option(None, "--max-rows"),
) -> None:
    _stage_command(config, max_rows, "train-ref-model")


@app.command("reference-benchmark")
def reference_benchmark(
    config: Path = typer.Option(..., "--config", exists=True),
    output: Path = typer.Option(
        Path("outputs/reference_benchmarks/nonprivate_k64.json"),
        "--output",
    ),
    max_rows_per_day: int | None = typer.Option(None, "--max-rows-per-day", min=1),
) -> None:
    """Evaluate the frozen, unsanitized K-token f_ref on temporal D_test."""
    from capt12.experiments.reference_benchmark import (
        benchmark_nonprivate_reference,
        write_reference_benchmark,
    )

    result = benchmark_nonprivate_reference(
        _load(config),
        max_rows_per_day=max_rows_per_day,
    )
    written = write_reference_benchmark(result, output)
    typer.echo(json.dumps({**result, "output": str(written)}, indent=2, sort_keys=True))


@app.command("build-encoder")
def build_encoder(
    config: Path = typer.Option(..., "--config", exists=True),
    max_rows: int | None = typer.Option(None, "--max-rows"),
) -> None:
    _stage_command(config, max_rows, "build-encoder")


@app.command("solve")
def solve(
    config: Path = typer.Option(..., "--config", exists=True),
    max_rows: int | None = typer.Option(None, "--max-rows"),
    force_full: bool = typer.Option(False, "--force-full"),
) -> None:
    cfg = _load(config, {"max_rows": max_rows, "force_full": force_full})
    path, metrics = run_pipeline(cfg, max_rows=max_rows)
    typer.echo(
        json.dumps(
            {
                "stage": "solve",
                "run": str(path),
                "solver_status": metrics.get("solver_status", []).tolist()
                if "solver_status" in metrics
                else [],
            },
            indent=2,
        )
    )


@app.command("certify")
def certify(
    config: Path = typer.Option(..., "--config", exists=True),
    max_rows: int | None = typer.Option(None, "--max-rows"),
) -> None:
    cfg = _load(config, {"max_rows": max_rows})
    if cfg.get("confidence") == "dp_aware_box":
        raise typer.BadParameter("dp_aware_box is experimental and cannot emit a certified result")
    path, metrics = run_pipeline(cfg, max_rows=max_rows)
    typer.echo(
        json.dumps(
            {
                "stage": "certify",
                "run": str(path),
                "certified": bool(metrics.get("certified", False).all()),
            },
            indent=2,
        )
    )


@app.command("audit")
def audit(
    config: Path = typer.Option(..., "--config", exists=True),
    max_rows: int | None = typer.Option(None, "--max-rows"),
) -> None:
    _stage_command(config, max_rows, "audit")


@app.command("evaluate")
def evaluate(
    config: Path = typer.Option(..., "--config", exists=True),
    max_rows: int | None = typer.Option(None, "--max-rows"),
) -> None:
    _stage_command(config, max_rows, "evaluate")


@app.command("run-grid")
def run_grid_command(
    config: Path = typer.Option(..., "--config", exists=True),
    data_root: Path | None = typer.Option(None, "--data-root"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    seed: int | None = typer.Option(None, "--seed"),
    seeds: str | None = typer.Option(None, "--seeds"),
    max_rows: int | None = typer.Option(None, "--max-rows"),
    sample_frac: float | None = typer.Option(None, "--sample-frac"),
    fixed_model: str | None = typer.Option(None, "--fixed-model"),
    fixed_model_path: Path | None = typer.Option(None, "--fixed-model-path"),
    phi: str | None = typer.Option(None, "--phi"),
    phi_list: str | None = typer.Option(None, "--phi-list"),
    phi_source_cols: str | None = typer.Option(None, "--phi-source-cols"),
    phi_sensitive_policy: str | None = typer.Option(None, "--phi-sensitive-policy"),
    k: int | None = typer.Option(None, "--K"),
    k_list: str | None = typer.Option(None, "--K-list", "--K_list"),
    l_count: int | None = typer.Option(None, "--L"),
    l_list: str | None = typer.Option(None, "--L-list", "--L_list"),
    sensitive_cols: str | None = typer.Option(None, "--sensitive-cols"),
    profiles: str | None = typer.Option(None, "--profiles"),
    profile_assignment: str | None = typer.Option(None, "--profile-assignment"),
    profile_probs: str | None = typer.Option(None, "--profile-probs"),
    context_cols: str | None = typer.Option(None, "--context-cols"),
    max_context_cardinality: int | None = typer.Option(None, "--max-context-cardinality"),
    privacy_scope: str | None = typer.Option(None, "--privacy-scope"),
    adjacency: str | None = typer.Option(None, "--adjacency"),
    epsilon: float | None = typer.Option(None, "--epsilon"),
    epsilon_list: str | None = typer.Option(None, "--epsilon-list"),
    epsilon_by_attr: str | None = typer.Option(None, "--epsilon-by-attr"),
    protect_profile: str | None = typer.Option(None, "--protect-profile"),
    profile_epsilon: float | None = typer.Option(None, "--profile-epsilon"),
    distortion: str | None = typer.Option(None, "--distortion"),
    distortion_list: str | None = typer.Option(None, "--distortion-list"),
    distortion_clip: float | None = typer.Option(None, "--distortion-clip"),
    partition: str | None = typer.Option(None, "--partition"),
    partition_list: str | None = typer.Option(None, "--partition-list"),
    nested_partitions: bool | None = typer.Option(
        None, "--nested-partitions/--non-nested-partitions"
    ),
    decoder: str | None = typer.Option(None, "--decoder"),
    decoder_list: str | None = typer.Option(None, "--decoder-list"),
    mechanism_list: str | None = typer.Option(None, "--mechanism-list"),
    confidence: str | None = typer.Option(None, "--confidence"),
    alpha_cert: float | None = typer.Option(None, "--alpha-cert"),
    dp_hist_epsilon: float | None = typer.Option(None, "--dp-hist-epsilon"),
    dp_hist_delta: float | None = typer.Option(None, "--dp-hist-delta"),
    shift_tv_list: str | None = typer.Option(None, "--shift-tv-list"),
    min_group_count: int | None = typer.Option(None, "--min-group-count"),
    rare_group_policy: str | None = typer.Option(None, "--rare-group-policy"),
    contribution_policy: str | None = typer.Option(None, "--contribution-policy"),
    solver: str | None = typer.Option(None, "--solver"),
    solver_tolerance: float | None = typer.Option(None, "--solver-tolerance"),
    time_limit: float | None = typer.Option(None, "--time-limit"),
    full_max_k: int | None = typer.Option(None, "--full-max-k"),
    force_full: bool = typer.Option(False, "--force-full"),
    jobs: int | None = typer.Option(None, "--jobs"),
    resume: bool = typer.Option(False, "--resume"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    target_ctr_list: str | None = typer.Option(None, "--target-ctr-list"),
    mass_m_list: str | None = typer.Option(None, "--mass-m-list", "--mass_m_list"),
    mass_n_list: str | None = typer.Option(None, "--mass-n-list", "--mass_n_list"),
    mass_privacy_weight_list: str | None = typer.Option(
        None, "--mass-privacy-weight-list", "--mass_privacy_weight_list"
    ),
    mass_utility_weight_list: str | None = typer.Option(
        None, "--mass-utility-weight-list", "--mass_utility_weight_list"
    ),
    mass_seed_list: str | None = typer.Option(
        None, "--mass-seed-list", "--mass_seed_list"
    ),
    mass_epochs: int | None = typer.Option(None, "--mass-epochs", "--mass_epochs", min=1),
    mass_output_mode: str | None = typer.Option(
        None, "--mass-output-mode", "--mass_output_mode"
    ),
    mass_temperature_list: str | None = typer.Option(
        None, "--mass-temperature-list", "--mass_temperature_list"
    ),
) -> None:
    overrides: dict[str, Any] = {
        "data_root": str(data_root) if data_root else None,
        "output_dir": str(output_dir) if output_dir else None,
        "seed": seed,
        "seeds": parse_csv_list(seeds, int, minimum=0) if seeds else None,
        "max_rows": max_rows,
        "sample_frac": sample_frac,
        "fixed_model": fixed_model,
        "fixed_model_path": str(fixed_model_path) if fixed_model_path else None,
        "phi": phi,
        "phi_list": parse_csv_list(phi_list) if phi_list else None,
        "phi_source_cols": parse_csv_list(phi_source_cols) if phi_source_cols else None,
        "phi_sensitive_policy": phi_sensitive_policy,
        "K": k,
        "K_list": parse_csv_list(k_list, int, minimum=1, maximum=4096) if k_list else None,
        "L": l_count,
        "L_list": parse_csv_list(l_list, int, minimum=1, maximum=4096) if l_list else None,
        "sensitive_cols": parse_csv_list(sensitive_cols) if sensitive_cols else None,
        "profiles": parse_profiles(profiles) if profiles else None,
        "profile_assignment": profile_assignment,
        "profile_probs": parse_csv_list(profile_probs, float, minimum=0, maximum=1)
        if profile_probs
        else None,
        "context_cols": parse_csv_list(context_cols) if context_cols else None,
        "max_context_cardinality": max_context_cardinality,
        "privacy_scope": privacy_scope,
        "adjacency": adjacency or privacy_scope,
        "epsilon": epsilon,
        "epsilon_list": parse_csv_list(epsilon_list, float, minimum=0) if epsilon_list else None,
        "epsilon_by_attr": json.loads(epsilon_by_attr) if epsilon_by_attr else None,
        "protect_profile": protect_profile,
        "profile_epsilon": profile_epsilon,
        "distortion": distortion,
        "distortion_list": parse_csv_list(distortion_list) if distortion_list else None,
        "distortion_clip": distortion_clip,
        "partition": partition,
        "partition_list": parse_csv_list(partition_list) if partition_list else None,
        "nested_partitions": nested_partitions,
        "decoder": decoder,
        "decoder_list": parse_csv_list(decoder_list) if decoder_list else None,
        "mechanism_list": parse_csv_list(mechanism_list) if mechanism_list else None,
        "confidence": confidence,
        "alpha_cert": alpha_cert,
        "dp_hist_epsilon": dp_hist_epsilon,
        "dp_hist_delta": dp_hist_delta,
        "shift_tv_list": parse_csv_list(shift_tv_list, float, minimum=0, maximum=1)
        if shift_tv_list
        else None,
        "min_group_count": min_group_count,
        "rare_group_policy": rare_group_policy,
        "contribution_policy": contribution_policy,
        "solver": solver,
        "solver_tolerance": solver_tolerance,
        "time_limit": time_limit,
        "full_max_k": full_max_k,
        "force_full": force_full,
        "jobs": jobs,
        "target_ctr_list": parse_csv_list(target_ctr_list, float, minimum=0, maximum=1)
        if target_ctr_list
        else None,
        "mass_m_list": parse_csv_list(mass_m_list, float, minimum=0)
        if mass_m_list
        else None,
        "mass_n_list": parse_csv_list(mass_n_list, float, minimum=0)
        if mass_n_list
        else None,
        "mass_privacy_weight_list": parse_csv_list(
            mass_privacy_weight_list, float, minimum=0
        )
        if mass_privacy_weight_list
        else None,
        "mass_utility_weight_list": parse_csv_list(
            mass_utility_weight_list, float, minimum=0
        )
        if mass_utility_weight_list
        else None,
        "mass_seed_list": parse_csv_list(mass_seed_list, int, minimum=0)
        if mass_seed_list
        else None,
        "mass_epochs": mass_epochs,
        "mass_output_mode": mass_output_mode,
        "mass_temperature_list": parse_csv_list(
            mass_temperature_list, float, minimum=0
        )
        if mass_temperature_list
        else None,
    }
    cfg = _load(config, overrides)
    if cfg.get("prior_art_comparison", False) or any(
        key.startswith("mass_") for key in cfg
    ):
        raise typer.BadParameter(
            "prior-art/MaSS configs cannot run through run-grid; use "
            "`capt12 prior-art-comparison --config ...`"
        )
    result = run_grid(cfg, resume=resume, dry_run=dry_run, max_rows=max_rows)
    typer.echo(
        json.dumps(
            {
                "rows": len(result),
                "complete": int((result.get("grid_status") == "complete").sum())
                if "grid_status" in result
                else 0,
                "skipped": int((result.get("status") == "skipped").sum())
                if "status" in result
                else 0,
            },
            indent=2,
        )
    )


@app.command("plot")
def plot(
    suite: str = typer.Option("paper", "--suite"),
    input_path: Path = typer.Option(Path("outputs/runs"), "--input"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    if suite != "paper":
        raise typer.BadParameter("only --suite paper is currently supported")
    from capt12.plots.paper import plot_paper_suite

    paths = plot_paper_suite(input_path, output_dir)
    typer.echo(
        json.dumps(
            {"generated": len(paths), "output": str(output_dir or input_path / "paper_figures")},
            indent=2,
        )
    )


@app.command("prior-art-comparison")
def prior_art_comparison(
    config: Path = typer.Option(..., "--config", exists=True),
    phase: str = typer.Option(
        "pilot",
        "--phase",
        help="pilot runs epsilon=1/L=16; full adds the epsilon and L grids.",
    ),
    comparison_output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Publication results directory (defaults to comparison_output_dir in config).",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Run CAPT/LDP and the finite-output MaSS comparison end to end."""
    if phase not in {"pilot", "full"}:
        raise typer.BadParameter("phase must be pilot or full", param_hint="--phase")
    cfg = _load(
        config,
        {
            "comparison_output_dir": (
                str(comparison_output_dir) if comparison_output_dir is not None else None
            )
        },
    )
    if not cfg.get("prior_art_comparison", False):
        raise typer.BadParameter(
            "dedicated comparison runner requires prior_art_comparison: true",
            param_hint="--config",
        )
    from capt12.experiments.prior_art_comparison import run_prior_art_comparison

    path = run_prior_art_comparison(cfg, phase=phase, resume=resume)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "phase": phase,
                "run": str(path),
                "results": str(
                    Path(cfg.get("comparison_output_dir", "outputs/prior_art_comparison"))
                    / "results.csv"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("render-prior-art-comparison")
def render_prior_art_comparison(
    results: Path = typer.Option(..., "--results", exists=True),
    method_contracts: Path = typer.Option(..., "--method-contracts", exists=True),
    output_dir: Path = typer.Option(..., "--output-dir"),
) -> None:
    """Render validated prior-art figures after comparison runs finish."""
    import pandas as pd

    from capt12.comparison.artifacts import (
        figure_hash_manifest,
        load_method_contracts,
        render_prior_art_figures,
    )

    paths = render_prior_art_figures(
        pd.read_csv(results), load_method_contracts(method_contracts), output_dir
    )
    typer.echo(
        json.dumps(
            {
                "generated": len(paths),
                "output_dir": str(output_dir),
                "sha256": figure_hash_manifest(paths),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.command("verify-prior-art-certificate")
def verify_prior_art_certificate(
    certificate: Path = typer.Argument(..., exists=True),
    tolerance: float | None = typer.Option(
        None,
        "--verification-tolerance",
        min=0,
        max=1e-8,
    ),
) -> None:
    """Reconstruct Q from a compact comparison bundle and recheck every constraint."""
    from capt12.comparison.certificate import verify_factorized_certificate

    result = verify_factorized_certificate(certificate, tolerance=tolerance)
    typer.echo(json.dumps(asdict(result), indent=2, sort_keys=True))
    if not result.valid:
        raise typer.Exit(code=1)


@app.command("verify-certificate")
def verify_certificate(
    certificate: Path = typer.Argument(..., exists=True),
    tolerance: float | None = typer.Option(
        None,
        "--verification-tolerance",
        "--solver-tolerance",
        min=0,
        max=1e-8,
    ),
) -> None:
    """Verify certificate bundle consistency and robust constraints.

    This does not authenticate the raw data provenance; use a trusted signed
    manifest for that deployment concern.
    """
    result = verify_certificate_file(certificate, tolerance)
    typer.echo(json.dumps(result.__dict__, indent=2, sort_keys=True))
    if not result.valid:
        raise typer.Exit(code=1)


@app.command("fixed-support-scaling")
def fixed_support_scaling(
    config: Path = typer.Option(..., "--config", exists=True),
) -> None:
    """Run a frozen Criteo design against nested D_cert user-day samples."""
    from capt12.experiments.fixed_support import run_fixed_support_scaling

    path = run_fixed_support_scaling(_load(config))
    typer.echo(json.dumps({"status": "ok", "run": str(path)}, indent=2))


@app.command("simplex-completion")
def simplex_completion(
    config: Path = typer.Option(..., "--config", exists=True),
) -> None:
    """Compare full-simplex CAPT with cover and LDP baselines on full D_cert."""
    from capt12.experiments.simplex_completion import run_simplex_completion

    path = run_simplex_completion(_load(config))
    typer.echo(json.dumps({"status": "ok", "run": str(path)}, indent=2))


@app.command("utility-design")
def utility_design(
    config: Path = typer.Option(..., "--config", exists=True),
) -> None:
    """Screen utility-aware designs, then compare informative CAPT with LDP."""
    from capt12.experiments.utility_design import run_utility_design_diagnostic

    path = run_utility_design_diagnostic(_load(config))
    typer.echo(json.dumps({"status": "ok", "run": str(path)}, indent=2))


@app.command("context-stratified")
def context_stratified(
    config: Path = typer.Option(..., "--config", exists=True),
    frozen_design_seed: int | None = typer.Option(
        None,
        "--frozen-design-seed",
        min=0,
        help="Override frozen_design_seed while keeping the prescribed L condition.",
    ),
    epsilon: float | None = typer.Option(
        None,
        "--epsilon",
        min=0,
        help="Privacy budget; must be strictly positive for certificate-safe repair.",
    ),
    utility_objective: str | None = typer.Option(
        None,
        "--utility-objective",
        help="LP objective: teacher_kl, empirical_logloss, or hybrid_logloss_kl.",
    ),
    representation_mode: str | None = typer.Option(
        None,
        "--representation-mode",
        help="Block/decoder cost: teacher_kl_fixed or objective_aligned.",
    ),
    hybrid_empirical_weight: float | None = typer.Option(
        None,
        "--hybrid-empirical-weight",
        min=0,
        max=1,
        help="Empirical-label weight for hybrid_logloss_kl.",
    ),
    context_r_pooling_weight: float | None = typer.Option(
        None,
        "--context-r-pooling-weight",
        min=0,
        max=1,
        help=(
            "Convex shrinkage rho for context R: (1-rho)R_context + "
            "rho R_shared; 0 keeps independent context R and 1 uses the shared target."
        ),
    ),
) -> None:
    """Run one prescribed public-context CAPT condition for one design seed."""
    from capt12.experiments.context_stratified import (
        run_context_stratified_diagnostic,
    )

    path = run_context_stratified_diagnostic(
        _load(
            config,
            {
                "frozen_design_seed": frozen_design_seed,
                "epsilon": epsilon,
                "context_utility_objective": utility_objective,
                "context_representation_mode": representation_mode,
                "hybrid_empirical_weight": hybrid_empirical_weight,
                "context_r_pooling_weight": context_r_pooling_weight,
            },
        )
    )
    typer.echo(json.dumps({"status": "ok", "run": str(path)}, indent=2))


@app.command("context-seed-stability")
def context_seed_stability(
    config: Path = typer.Option(..., "--config", exists=True),
    mechanism_seed_mode: str = typer.Option(
        "per-seed",
        "--mechanism-seed-mode",
        help=(
            "per-seed rebuilds and reoptimizes the complete mechanism for every design "
            "seed; fixed builds/reuses one complete canonical mechanism."
        ),
    ),
    frozen_design_seeds: str | None = typer.Option(
        None,
        "--frozen-design-seeds",
        help=(
            "Comma-separated design seeds for per-seed mode (default: 0,1,2); "
            "not allowed in fixed mode."
        ),
    ),
    fixed_mechanism_seed: int | None = typer.Option(
        None,
        "--fixed-mechanism-seed",
        min=0,
        help=(
            "Canonical design seed for fixed mode. The encoder, partition, decoder, and "
            "all context-specific R matrices are fixed together."
        ),
    ),
    test_seeds: str | None = typer.Option(
        None,
        "--test-seeds",
        help=(
            "Comma-separated Monte Carlo release seeds for D_test after the complete "
            "mechanism is fixed; fixed mode only and at least two distinct seeds."
        ),
    ),
    test_baselines: str = typer.Option(
        "block-ldp",
        "--test-baselines",
        help=(
            "Comma-separated fixed-test baselines: block-ldp, rr, or both. "
            "CAPT is always included; fixed mode with --test-seeds only."
        ),
    ),
    epsilon: float | None = typer.Option(
        None,
        "--epsilon",
        min=0,
        help="Privacy budget used by the selected run or runs; must be strictly positive.",
    ),
    utility_objective: str | None = typer.Option(
        None,
        "--utility-objective",
        help="LP objective used by the selected run or runs.",
    ),
    representation_mode: str | None = typer.Option(
        None,
        "--representation-mode",
        help="Block/decoder cost used by the selected run or runs.",
    ),
    hybrid_empirical_weight: float | None = typer.Option(
        None,
        "--hybrid-empirical-weight",
        min=0,
        max=1,
        help="Empirical-label weight for hybrid_logloss_kl.",
    ),
    context_r_pooling_weight: float | None = typer.Option(
        None,
        "--context-r-pooling-weight",
        min=0,
        max=1,
        help=(
            "Convex shrinkage rho for every context R: (1-rho)R_context + "
            "rho R_shared; included in the fixed mechanism when fixed mode is used."
        ),
    ),
) -> None:
    """Run per-seed design sensitivity or one fixed canonical mechanism."""
    from capt12.experiments.context_seed_stability import (
        run_context_fixed_mechanism,
        run_context_seed_stability,
    )

    mode = mechanism_seed_mode.strip().lower()
    if mode not in {"per-seed", "fixed"}:
        raise typer.BadParameter(
            "must be per-seed or fixed",
            param_hint="--mechanism-seed-mode",
        )
    resolved = _load(
        config,
        {
            "context_utility_objective": utility_objective,
            "epsilon": epsilon,
            "context_representation_mode": representation_mode,
            "hybrid_empirical_weight": hybrid_empirical_weight,
            "context_r_pooling_weight": context_r_pooling_weight,
        },
    )
    if mode == "fixed":
        if frozen_design_seeds is not None:
            raise typer.BadParameter(
                "cannot be used when --mechanism-seed-mode=fixed",
                param_hint="--frozen-design-seeds",
            )
        if fixed_mechanism_seed is None:
            raise typer.BadParameter(
                "is required when --mechanism-seed-mode=fixed",
                param_hint="--fixed-mechanism-seed",
            )
        try:
            parsed_test_seeds = parse_csv_list(test_seeds, int, minimum=0)
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--test-seeds") from error
        from capt12.experiments.context_fixed_test_seeds import normalize_test_baselines

        try:
            parsed_test_baselines = normalize_test_baselines(test_baselines.split(","))
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--test-baselines") from error
        if test_seeds is not None and len(parsed_test_seeds) < 2:
            raise typer.BadParameter(
                "requires at least two distinct seeds",
                param_hint="--test-seeds",
            )
        path = run_context_fixed_mechanism(resolved, fixed_mechanism_seed)
        test_seed_path = None
        if parsed_test_seeds:
            from capt12.experiments.context_fixed_test_seeds import (
                run_fixed_mechanism_test_seeds,
            )

            test_seed_path = run_fixed_mechanism_test_seeds(
                path,
                parsed_test_seeds,
                baselines=parsed_test_baselines,
            )
        payload = {
            "status": "ok",
            "mechanism_seed_mode": mode,
            "fixed_mechanism_seed": fixed_mechanism_seed,
            "run": str(path),
            "sol_review_bundle": str(path / "sol_review_bundle.zip"),
        }
        if test_seed_path is not None:
            payload.update(
                {
                    "test_seeds": parsed_test_seeds,
                    "test_baselines": parsed_test_baselines,
                    "test_seed_evaluation": str(test_seed_path),
                }
            )
        typer.echo(json.dumps(payload, indent=2))
        return

    if test_seeds is not None:
        raise typer.BadParameter(
            "can only be used when --mechanism-seed-mode=fixed",
            param_hint="--test-seeds",
        )
    if fixed_mechanism_seed is not None:
        raise typer.BadParameter(
            "can only be used when --mechanism-seed-mode=fixed",
            param_hint="--fixed-mechanism-seed",
        )
    try:
        seeds = parse_csv_list(frozen_design_seeds or "0,1,2", int, minimum=0)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--frozen-design-seeds") from error
    path = run_context_seed_stability(resolved, seeds)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "mechanism_seed_mode": mode,
                "summary": str(path),
                "sol_review_bundle": str(path / "sol_seed_stability_review_bundle.zip"),
            },
            indent=2,
        )
    )


@app.command("context-cost-comparison")
def context_cost_comparison(
    summary_dirs: list[Path] = typer.Option(
        ...,
        "--summary-dir",
        exists=True,
        file_okay=False,
        help="Seed-summary directory; repeat once for each of the three utility objectives.",
    ),
    output_root: Path = typer.Option(
        Path("outputs/context_cost_comparisons"),
        "--output-root",
    ),
) -> None:
    """Validate and compare three objective-aligned cost families."""
    from capt12.experiments.context_cost_comparison import run_context_cost_comparison

    path = run_context_cost_comparison(summary_dirs, output_root)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "comparison": str(path),
                "sol_review_bundle": str(path / "sol_context_cost_comparison_bundle.zip"),
            },
            indent=2,
        )
    )


@app.command("context-epsilon-grid")
def context_epsilon_grid(
    config: Path = typer.Option(..., "--config", exists=True),
    epsilon_values: str = typer.Option(
        "0.5,1,2",
        "--epsilon-values",
        help="Comma-separated positive privacy budgets.",
    ),
    frozen_design_seeds: str = typer.Option(
        "0,1,2,3,4",
        "--frozen-design-seeds",
        help="Comma-separated paired frozen-design seeds used at every epsilon.",
    ),
    utility_objective: str | None = typer.Option(
        None,
        "--utility-objective",
        help="LP objective used throughout the grid.",
    ),
    representation_mode: str | None = typer.Option(
        None,
        "--representation-mode",
        help="Block/decoder representation used throughout the grid.",
    ),
    hybrid_empirical_weight: float | None = typer.Option(
        None,
        "--hybrid-empirical-weight",
        min=0,
        max=1,
    ),
    context_r_pooling_weight: float | None = typer.Option(
        None,
        "--context-r-pooling-weight",
        min=0,
        max=1,
        help="Fixed convex context-R pooling weight used throughout the epsilon grid.",
    ),
    output_root: Path = typer.Option(
        Path("outputs/context_epsilon_grids"),
        "--output-root",
    ),
) -> None:
    """Run/reuse a matched positive-epsilon grid and create one Sol bundle."""
    from capt12.experiments.context_epsilon_grid import run_context_epsilon_grid

    try:
        epsilons = parse_csv_list(epsilon_values, float, minimum=0)
        if any(value <= 0 for value in epsilons):
            raise ValueError("epsilon values must be strictly positive")
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--epsilon-values") from error
    try:
        seeds = parse_csv_list(frozen_design_seeds, int, minimum=0)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--frozen-design-seeds") from error
    path = run_context_epsilon_grid(
        _load(
            config,
            {
                "context_utility_objective": utility_objective,
                "context_representation_mode": representation_mode,
                "hybrid_empirical_weight": hybrid_empirical_weight,
                "context_r_pooling_weight": context_r_pooling_weight,
            },
        ),
        seeds,
        epsilons,
        output_root=output_root,
    )
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "grid": str(path),
                "sol_review_bundle": str(path / "sol_context_epsilon_grid_bundle.zip"),
            },
            indent=2,
        )
    )


@app.command("context-cost-stability")
def context_cost_stability(
    config: Path = typer.Option(..., "--config", exists=True),
    frozen_design_seeds: str = typer.Option(
        "0,1,2,3,4",
        "--frozen-design-seeds",
        help="Comma-separated seeds used by every objective family.",
    ),
    epsilon: float | None = typer.Option(
        None,
        "--epsilon",
        min=0,
        help="Privacy budget shared by all objectives and seeds.",
    ),
    hybrid_empirical_weight: float = typer.Option(
        0.5,
        "--hybrid-empirical-weight",
        min=0,
        max=1,
    ),
    context_r_pooling_weight: float | None = typer.Option(
        None,
        "--context-r-pooling-weight",
        min=0,
        max=1,
        help="Fixed convex context-R pooling weight used for every objective and seed.",
    ),
    output_root: Path = typer.Option(
        Path("outputs/context_cost_comparisons"),
        "--output-root",
    ),
) -> None:
    """Run/reuse all three aligned costs and create one integrated bundle."""
    from capt12.experiments.context_cost_comparison import run_context_cost_stability

    try:
        seeds = parse_csv_list(frozen_design_seeds, int, minimum=0)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--frozen-design-seeds") from error
    path = run_context_cost_stability(
        _load(
            config,
            {
                "epsilon": epsilon,
                "context_r_pooling_weight": context_r_pooling_weight,
            },
        ),
        seeds,
        hybrid_empirical_weight=hybrid_empirical_weight,
        output_root=output_root,
    )
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "comparison": str(path),
                "sol_review_bundle": str(path / "sol_context_cost_comparison_bundle.zip"),
            },
            indent=2,
        )
    )


@app.command("context-paper-figure")
def context_paper_figure(
    epsilon_grid_bundle: Path = typer.Option(
        ...,
        "--epsilon-grid-bundle",
        exists=True,
        dir_okay=False,
        help="Certified L=16 epsilon-grid Sol bundle.",
    ),
    block_comparison_bundles: list[Path] = typer.Option(
        ...,
        "--block-comparison-bundle",
        exists=True,
        dir_okay=False,
        help="Certified cost-comparison bundle; repeat for each available L.",
    ),
    output_root: Path = typer.Option(
        Path("outputs/paper_figures/context_capt_main"),
        "--output-root",
    ),
) -> None:
    """Render the two-panel certified frontier and block-size paper figure."""
    from capt12.experiments.context_paper_figure import run_context_paper_figure

    path = run_context_paper_figure(
        epsilon_grid_bundle,
        block_comparison_bundles,
        output_root,
    )
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "paper_figure": str(path),
                "review_bundle": str(path / "context_capt_main_figure_bundle.zip"),
            },
            indent=2,
        )
    )


@app.command("context-fixed-test-seeds")
def context_fixed_test_seeds(
    run: Path = typer.Option(
        ...,
        "--run",
        exists=True,
        file_okay=False,
        help="Completed fixed CAPT mechanism run; it is not rebuilt or reoptimized.",
    ),
    test_seeds: str = typer.Option(
        "0,1,2,3,4",
        "--test-seeds",
        help="Comma-separated Monte Carlo release seeds; at least two distinct seeds.",
    ),
    baselines: str = typer.Option(
        "block-ldp,rr",
        "--baselines",
        help="Comma-separated baselines: block-ldp, rr, or both. CAPT is always included.",
    ),
    output_root: Path = typer.Option(
        Path("outputs/context_fixed_test_seed_evaluations"),
        "--output-root",
    ),
) -> None:
    """Evaluate fixed CAPT against selectable block-LDP and K-ary RR baselines."""
    from capt12.experiments.context_fixed_test_seeds import (
        normalize_test_baselines,
        run_fixed_mechanism_test_seeds,
    )

    try:
        parsed_seeds = parse_csv_list(test_seeds, int, minimum=0)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--test-seeds") from error
    if len(parsed_seeds) < 2:
        raise typer.BadParameter("requires at least two distinct seeds", param_hint="--test-seeds")
    try:
        parsed_baselines = normalize_test_baselines(baselines.split(","))
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--baselines") from error
    path = run_fixed_mechanism_test_seeds(
        run,
        parsed_seeds,
        baselines=parsed_baselines,
        output_root=output_root,
    )
    baseline_methods = {
        "block-ldp": "context_ldp",
        "rr": "kary_rr",
    }
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "run": str(run),
                "test_seeds": parsed_seeds,
                "baselines": parsed_baselines,
                "comparison_methods": [
                    "context_capt",
                    *[baseline_methods[name] for name in parsed_baselines],
                ],
                "test_seed_evaluation": str(path),
                "figure": str(path / "figures" / "fixed_capt_ldp_baseline_comparison.png"),
            },
            indent=2,
        )
    )


@app.command("context-render-fixed-test-figure")
def context_render_fixed_test_figure(
    evaluation: Path = typer.Option(
        ...,
        "--evaluation",
        exists=True,
        file_okay=False,
        help="Completed context-fixed-test-seeds evaluation directory.",
    ),
) -> None:
    """Regenerate the fixed-test comparison PNG/PDF without rerunning evaluation."""
    from capt12.experiments.context_fixed_test_seeds import (
        render_fixed_test_seed_figure,
    )

    png, pdf = render_fixed_test_seed_figure(evaluation)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "evaluation": str(evaluation),
                "png": str(png),
                "pdf": str(pdf),
            },
            indent=2,
        )
    )


@app.command("context-global-token-ldp")
def context_global_token_ldp(
    epsilon_grid_bundle: Path = typer.Option(
        ...,
        "--epsilon-grid-bundle",
        exists=True,
        dir_okay=False,
    ),
    block_comparison_bundles: list[Path] = typer.Option(
        ...,
        "--block-comparison-bundle",
        exists=True,
        dir_okay=False,
        help="Repeat for the certified L=8 and L=16 comparison bundles.",
    ),
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        exists=True,
        file_okay=False,
        help="Override the D_test data root stored in the source runs.",
    ),
    solver_time_limit: float = typer.Option(
        300,
        "--solver-time-limit",
        min=1,
        help="Per-attempt HiGHS time limit in seconds.",
    ),
    output_root: Path = typer.Option(
        Path("outputs/global_token_ldp_comparisons"),
        "--output-root",
    ),
) -> None:
    """Solve and compare the unrestricted K=64 global optimal-LDP baseline."""
    from capt12.experiments.context_global_token_ldp import (
        run_global_token_ldp_comparison,
    )

    path = run_global_token_ldp_comparison(
        epsilon_grid_bundle,
        block_comparison_bundles,
        data_root=data_root,
        solver_time_limit=solver_time_limit,
        output_root=output_root,
    )
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "comparison": str(path),
                "sol_review_bundle": str(path / "sol_global_token_ldp_comparison_bundle.zip"),
            },
            indent=2,
        )
    )


@app.command("context-fixed-global-token-ldp")
def context_fixed_global_token_ldp(
    run: Path = typer.Option(
        ...,
        "--run",
        exists=True,
        file_okay=False,
        help="Completed fixed-mechanism CAPT run directory.",
    ),
    fixed_evaluation: Path = typer.Option(
        ...,
        "--fixed-evaluation",
        exists=True,
        file_okay=False,
        help="Fixed test-seed evaluation belonging to --run.",
    ),
    solver_time_limit: float = typer.Option(
        1800,
        "--solver-time-limit",
        min=1,
        help="HiGHS time limit in seconds for the matched K=64 global-LDP solve.",
    ),
) -> None:
    """Compare a fixed CAPT run with a matched unrestricted global token-LDP."""
    from capt12.experiments.context_fixed_global_ldp import (
        run_fixed_capt_global_ldp_comparison,
    )

    path = run_fixed_capt_global_ldp_comparison(
        run,
        fixed_evaluation,
        solver_time_limit=solver_time_limit,
    )
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "comparison": str(path),
                "figure": str(path / "figures" / "fixed_capt_vs_global_ldp_absolute.png"),
            },
            indent=2,
        )
    )


@app.command("context-cost-global-ldp-figures")
def context_cost_global_ldp_figures(
    cost_comparison_dirs: list[Path] = typer.Option(
        ...,
        "--cost-comparison-dir",
        exists=True,
        file_okay=False,
        help="Repeat for each completed L-specific three-cost comparison directory.",
    ),
    global_comparison_dir: Path = typer.Option(
        ...,
        "--global-comparison-dir",
        exists=True,
        file_okay=False,
        help="Completed global token-level optimal-LDP comparison directory.",
    ),
) -> None:
    """Add non-overwriting global-LDP figures and review bundles."""
    from capt12.experiments.context_cost_global_ldp_figure import (
        add_global_ldp_figures,
    )

    bundles = add_global_ldp_figures(cost_comparison_dirs, global_comparison_dir)
    typer.echo(
        json.dumps(
            {
                "status": "ok",
                "bundles": [str(path) for path in bundles],
            },
            indent=2,
        )
    )


@app.command("smoke")
def smoke(
    config: Path = typer.Option(..., "--config", exists=True),
    data_root: Path | None = typer.Option(None, "--data-root"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    seed: int | None = typer.Option(None, "--seed"),
    max_rows: int | None = typer.Option(None, "--max-rows", min=1),
) -> None:
    cfg = _load(
        config,
        {
            "data_root": str(data_root) if data_root else None,
            "output_dir": str(output_dir) if output_dir else None,
            "seed": seed,
            "max_rows": max_rows,
        },
    )
    path, metrics = run_pipeline(cfg, max_rows=max_rows)
    typer.echo(
        json.dumps({"status": "ok", "run": str(path), "metric_rows": len(metrics)}, indent=2)
    )


if __name__ == "__main__":
    app()
