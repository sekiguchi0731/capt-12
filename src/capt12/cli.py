from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from capt12.certification.artifact import verify_certificate as verify_certificate_file
from capt12.confidence.boxes import CONFIDENCE_REGISTRY
from capt12.config import load_config, parse_csv_list, parse_profiles
from capt12.data.inspect import inspect_parquet, write_inspection_json
from capt12.decoders.registry import DECODER_REGISTRY
from capt12.distortions.registry import DISTORTION_REGISTRY
from capt12.encoders.base import ENCODER_REGISTRY
from capt12.grid import run_grid
from capt12.mechanisms.baselines import BASELINES
from capt12.models.reference import MODEL_REGISTRY
from capt12.partitions.registry import PARTITION_REGISTRY
from capt12.pipeline import run_pipeline

app = typer.Typer(no_args_is_help=True, help="CAPT-12 reproducible research CLI")


def _load(path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return load_config(path, {key: value for key, value in (overrides or {}).items() if value is not None})


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
    cfg = _load(config, {"data_root": str(data_root) if data_root else None, "output_dir": str(output_dir) if output_dir else None, "max_rows": max_rows, "sample_frac": sample_frac})
    path, metrics = run_pipeline(cfg, max_rows=max_rows)
    typer.echo(json.dumps({"stage": "prepare", "run": str(path), "rows": len(metrics)}, indent=2))


@app.command("train-ref-model")
def train_ref_model(config: Path = typer.Option(..., "--config", exists=True), max_rows: int | None = typer.Option(None, "--max-rows")) -> None:
    _stage_command(config, max_rows, "train-ref-model")


@app.command("build-encoder")
def build_encoder(config: Path = typer.Option(..., "--config", exists=True), max_rows: int | None = typer.Option(None, "--max-rows")) -> None:
    _stage_command(config, max_rows, "build-encoder")


@app.command("solve")
def solve(config: Path = typer.Option(..., "--config", exists=True), max_rows: int | None = typer.Option(None, "--max-rows"), force_full: bool = typer.Option(False, "--force-full")) -> None:
    cfg = _load(config, {"max_rows": max_rows, "force_full": force_full})
    path, metrics = run_pipeline(cfg, max_rows=max_rows)
    typer.echo(json.dumps({"stage": "solve", "run": str(path), "solver_status": metrics.get("solver_status", []).tolist() if "solver_status" in metrics else []}, indent=2))


@app.command("certify")
def certify(config: Path = typer.Option(..., "--config", exists=True), max_rows: int | None = typer.Option(None, "--max-rows")) -> None:
    cfg = _load(config, {"max_rows": max_rows})
    if cfg.get("confidence") == "dp_aware_box":
        raise typer.BadParameter("dp_aware_box is experimental and cannot emit a certified result")
    path, metrics = run_pipeline(cfg, max_rows=max_rows)
    typer.echo(json.dumps({"stage": "certify", "run": str(path), "certified": bool(metrics.get("certified", False).all())}, indent=2))


@app.command("audit")
def audit(config: Path = typer.Option(..., "--config", exists=True), max_rows: int | None = typer.Option(None, "--max-rows")) -> None:
    _stage_command(config, max_rows, "audit")


@app.command("evaluate")
def evaluate(config: Path = typer.Option(..., "--config", exists=True), max_rows: int | None = typer.Option(None, "--max-rows")) -> None:
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
    nested_partitions: bool | None = typer.Option(None, "--nested-partitions/--non-nested-partitions"),
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
        "profile_probs": parse_csv_list(profile_probs, float, minimum=0, maximum=1) if profile_probs else None,
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
        "shift_tv_list": parse_csv_list(shift_tv_list, float, minimum=0, maximum=1) if shift_tv_list else None,
        "min_group_count": min_group_count,
        "rare_group_policy": rare_group_policy,
        "contribution_policy": contribution_policy,
        "solver": solver,
        "solver_tolerance": solver_tolerance,
        "time_limit": time_limit,
        "full_max_k": full_max_k,
        "force_full": force_full,
        "jobs": jobs,
        "target_ctr_list": parse_csv_list(target_ctr_list, float, minimum=0, maximum=1) if target_ctr_list else None,
    }
    cfg = _load(config, overrides)
    result = run_grid(cfg, resume=resume, dry_run=dry_run, max_rows=max_rows)
    typer.echo(json.dumps({"rows": len(result), "complete": int((result.get("grid_status") == "complete").sum()) if "grid_status" in result else 0, "skipped": int((result.get("status") == "skipped").sum()) if "status" in result else 0}, indent=2))


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
    typer.echo(json.dumps({"generated": len(paths), "output": str(output_dir or input_path / "paper_figures")}, indent=2))


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


@app.command("smoke")
def smoke(
    config: Path = typer.Option(..., "--config", exists=True),
    data_root: Path | None = typer.Option(None, "--data-root"),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
    seed: int | None = typer.Option(None, "--seed"),
    max_rows: int | None = typer.Option(None, "--max-rows", min=1),
) -> None:
    cfg = _load(config, {"data_root": str(data_root) if data_root else None, "output_dir": str(output_dir) if output_dir else None, "seed": seed, "max_rows": max_rows})
    path, metrics = run_pipeline(cfg, max_rows=max_rows)
    typer.echo(json.dumps({"status": "ok", "run": str(path), "metric_rows": len(metrics)}, indent=2))


if __name__ == "__main__":
    app()
