"""Diagnose why a context-stratified CAPT run does not beat global token-level LDP.

Usage (from the repository root, on the machine that holds ``outputs/``)::

    uv run python scripts/diagnose_capt_vs_global_ldp.py \
        --run outputs/context_stratified_l32_runs/6478476eb2a9f5b2 \
        --fixed-eval outputs/context_fixed_test_seed_evaluations/173e4c1da0bfc76e

    # Also re-solve global K=64 LDP (and optional per-context K=64 LDP) on the
    # run's own encoder/reference/D_design statistics and evaluate on D_test:
    uv run python scripts/diagnose_capt_vs_global_ldp.py --run ... --solve-global
    uv run python scripts/diagnose_capt_vs_global_ldp.py --run ... --solve-global --context-token-ldp

Everything is reported as held-out expected randomized log loss in micro-nats
per display (lower is better), and the CAPT-vs-global gap is decomposed into:

    global K=64 LDP
      -> shared_ldp      (same L-block partition/medoid decoder, one channel): partition penalty
      -> context_ldp     (one block-LDP channel per public context): context-stratification gain
      -> capt_pre_repair (robust A_S|B constraint instead of LDP): privacy-relaxation gain
      -> context_capt    (uniform full-support repair): repair cost
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

MICRO = 1e6


def _find(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(root.rglob(Path(name).name))
    matches = [path for path in matches if path.as_posix().endswith(name)]
    return matches[0] if matches else None


def _run_dir(root: Path) -> Path:
    marker = _find(root, "context_stratified_metadata.json")
    if marker is None:
        raise SystemExit(f"no context_stratified_metadata.json under {root}")
    return marker.parent


def _section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def _weighted(values: pd.Series, weights: pd.Series) -> float:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0 else math.nan


def report_config(run: Path) -> dict[str, Any]:
    _section("1. Run configuration (what exactly was compared)")
    config = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
    keys = [
        "K",
        "L",
        "epsilon",
        "profiles",
        "context_cols",
        "context_designs",
        "context_utility_objective",
        "context_representation_mode",
        "confidence",
        "alpha_cert",
        "missing_group_policy",
        "rare_group_policy",
        "sensitive_fallback_policy",
        "adjacency",
        "privacy_scope",
        "frozen_design_seed",
        "certificate_channel_repair",
        "time_limit",
        "max_cutting_plane_iterations",
    ]
    for key in keys:
        if key in config:
            print(f"  {key:32s} {config[key]}")
    objective = str(config.get("context_utility_objective", "teacher_kl"))
    representation = str(config.get("context_representation_mode", "teacher_kl_fixed"))
    if objective != "empirical_logloss" or representation != "objective_aligned":
        print(
            "\n  !! CAPT R was optimized for "
            f"'{objective}' with '{representation}' partition/decoder, but the global "
            "token-LDP baseline is optimized directly for empirical_logloss with an "
            "objective-aligned cost, and both are scored on held-out empirical log loss.\n"
            "     This alone is an objective-mismatch handicap for CAPT (see section 5)."
        )
    return config


def report_test_metrics(run: Path) -> pd.DataFrame:
    _section("2. Held-out D_test metrics stored by the run (analytic, seed-invariant)")
    metrics = pd.read_csv(run / "tables" / "test_metrics.csv")
    columns = [
        "design",
        "method",
        "expected_randomized_log_loss",
        "mixture_mean_log_loss",
        "ROC_AUC",
        "PR_AUC",
        "ECE",
    ]
    frame = metrics[[column for column in columns if column in metrics]].copy()
    reference = frame.loc[frame["method"] == "context_capt", "expected_randomized_log_loss"]
    if len(reference):
        frame["minus_capt_micro"] = MICRO * (
            frame["expected_randomized_log_loss"] - float(reference.iloc[0])
        )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.9g}"))
    aggregate_path = run / "tables" / "aggregate_results.csv"
    if aggregate_path.is_file():
        aggregate = pd.read_csv(aggregate_path)
        print("\n  In-sample design objective (D_design; lower is better):")
        print(
            aggregate[
                ["method", "aggregate_objective_value", "gain_over_context_constant"]
            ].to_string(index=False, float_format=lambda value: f"{value:.9g}")
        )
    return metrics


def report_contexts(run: Path) -> pd.DataFrame:
    _section("3. Per-context diagnosis: where does CAPT actually differ from LDP?")
    frame = pd.read_csv(run / "tables" / "context_results.csv")
    mass = frame["design_mass"]
    degraded = frame["ldp_degraded"].astype(bool)
    identical = frame["capt_equals_context_ldp_channel"].astype(bool)
    print(f"  contexts                                   {len(frame)}")
    print(
        f"  contexts with a full-simplex/full-simplex edge (CAPT == pure LDP by construction): "
        f"{int(degraded.sum())}  (design mass {float(mass[degraded].sum()):.4f})"
    )
    print(
        f"  contexts whose released CAPT channel equals context LDP: {int(identical.sum())} "
        f"(design mass {float(mass[identical].sum()):.4f})"
    )
    print(
        "  mass-weighted CAPT advantage over context LDP on D_design objective: "
        f"pre-repair {MICRO * _weighted(frame['capt_advantage_over_context_ldp_pre_repair'], mass):.6g}"
        f" micro, released {MICRO * _weighted(frame['capt_advantage_over_context_ldp'], mass):.6g}"
        " micro"
    )
    print(
        f"  repair lambda: max {frame['repair_lambda'].max():.3g}, "
        f"mass-weighted {_weighted(frame['repair_lambda'], mass):.3g}"
    )
    print(
        f"  missing (full-simplex) groups total {int(frame['missing_group_count'].sum())}, "
        f"rare groups total {int(frame['rare_group_count'].sum())}, "
        f"expected groups total {int(frame['expected_group_count'].sum())}"
    )
    columns = [
        "public_context",
        "design_mass",
        "expected_group_count",
        "missing_group_count",
        "rare_group_count",
        "full_simplex_ordered_edge_count",
        "ldp_degraded",
        "capt_equals_context_ldp_channel",
        "excess_context_ldp",
        "excess_context_capt",
        "capt_advantage_over_context_ldp_pre_repair",
        "capt_advantage_over_context_ldp",
        "repair_lambda",
    ]
    shown = frame.sort_values("design_mass", ascending=False)[columns].copy()
    for column in (
        "excess_context_ldp",
        "excess_context_capt",
        "capt_advantage_over_context_ldp_pre_repair",
        "capt_advantage_over_context_ldp",
    ):
        shown[column] = MICRO * shown[column]
    print("\n  (excess/advantage columns in micro-nats; sorted by design mass)")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(shown.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    return frame


def report_fixed_eval(fixed: Path | None) -> None:
    if fixed is None:
        return
    _section("4. Fixed-mechanism test-seed evaluation (Monte Carlo only)")
    summary_path = _find(fixed, "tables/test_seed_summary.csv")
    if summary_path is None:
        print(f"  no tables/test_seed_summary.csv under {fixed}")
        return
    summary = pd.read_csv(summary_path)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.9g}"))
    print(
        "\n  NOTE: 'context_ldp' here is the per-context L-block LDP on the SAME partition,"
        " not global K=64 token LDP. The test seeds only redraw the released token from a"
        " fixed channel; they add Monte Carlo noise, not information about design variance."
    )


def _probability_grid(reference: Any, contexts: list[str], column: str, k: int) -> np.ndarray:
    return np.stack(
        [
            reference.predict(pd.DataFrame({"__token__": np.arange(k, dtype=int), column: context}))
            for context in contexts
        ]
    )


def _evaluate(
    channels: dict[str, np.ndarray],
    grid: np.ndarray,
    contexts: list[str],
    context_values: np.ndarray,
    tokens: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    from capt12.evaluation.metrics import prediction_metrics

    scores = np.empty(len(tokens), dtype=float)
    loss = 0.0
    covered = 0
    for index, context in enumerate(contexts):
        mask = context_values == context
        if not mask.any():
            continue
        channel = channels[context]
        probabilities = grid[index]
        loss_one = channel @ -np.log(np.clip(probabilities, 1e-6, 1))
        loss_zero = channel @ -np.log(np.clip(1 - probabilities, 1e-6, 1))
        inputs = tokens[mask]
        scores[mask] = (channel @ probabilities)[inputs]
        loss += float(np.where(labels[mask] == 1, loss_one[inputs], loss_zero[inputs]).sum())
        covered += int(mask.sum())
    if covered != len(tokens):
        raise RuntimeError("D_test contains contexts absent from the run")
    metrics = prediction_metrics(labels, scores)
    return {
        "expected_randomized_log_loss": loss / len(tokens),
        "ROC_AUC": float(metrics["ROC_AUC"]),
        "PR_AUC": float(metrics["PR_AUC"]),
        "ECE": float(metrics["ECE"]),
    }


def _solve_token_ldp(cost: np.ndarray, weights: np.ndarray, epsilon: float, limit: float):
    from capt12.experiments.context_global_token_ldp import _repair_ldp_uniform
    from capt12.mechanisms.lp import solve_ldp_block_lp

    solution = solve_ldp_block_lp(cost, weights, epsilon, tolerance=1e-10, time_limit=limit)
    if solution.channel is None:
        raise RuntimeError(solution.solver.message)
    repaired, _, _, _ = _repair_ldp_uniform(solution.channel, epsilon, margin=1e-12)
    return repaired


def solve_global(
    run: Path,
    config: dict[str, Any],
    metrics: pd.DataFrame,
    *,
    data_root: Path | None,
    context_token_ldp: bool,
    time_limit: float,
) -> None:
    from capt12.data.loader import load_parquet_sample
    from capt12.distortions.registry import conditional_utility_cost
    from capt12.experiments.fixed_support import _context_aggregated_representation_cost
    from capt12.mechanisms.lp import lift_block_channel
    from capt12.models.reference import ReferenceModel

    _section("5. Re-solved baselines on this run's encoder/reference/D_design")
    epsilon = float(config["epsilon"])
    column = str(config["context_cols"][0])
    label = str(config.get("label_col", "is_clicked"))
    eta = float(config.get("distortion_clip", 1e-6))
    hybrid = float(config.get("hybrid_empirical_weight", 0.5))
    run_objective = str(config.get("context_utility_objective", "teacher_kl"))
    with np.load(run / "mechanism" / "frozen_design.npz") as stored:
        frozen = {name: np.asarray(stored[name]).copy() for name in stored.files}
    channel_file = sorted((run / "mechanism").glob("context_channels-*.npz"))[0]
    with np.load(channel_file) as stored:
        stored_channels = {name: np.asarray(stored[name]).copy() for name in stored.files}
    contexts = [str(value) for value in stored_channels["contexts"]]
    assignment = stored_channels["assignment"]
    decoder = stored_channels["decoder"]
    k = len(assignment)
    reference = ReferenceModel.load(run / "models" / "reference.joblib")
    mapper = joblib.load(run / "models" / "category_mapper.joblib")
    encoder = joblib.load(run / "models" / "encoder.joblib")

    columns = list(
        dict.fromkeys(
            [
                label,
                *str(config["profiles"][0]).split("+"),
                column,
                *map(str, config.get("phi_source_cols", [])),
            ]
        )
    )
    frame = load_parquet_sample(
        data_root=data_root or Path(config["data_root"]),
        columns=columns,
        days=config["splits"]["D_test"],
    )
    frame = mapper.transform(frame)
    tokens = encoder.transform(frame)
    labels = frame[label].to_numpy(dtype=int)
    context_values = frame[column].astype(str).to_numpy()
    grid = _probability_grid(reference, contexts, column, k)
    print(f"  D_test rows {len(frame):,}; contexts {len(contexts)}; K={k}; epsilon={epsilon:g}")

    levels = [str(value) for value in frozen["context_levels"]]
    level_grid = _probability_grid(reference, levels, column, k).T
    counts = np.asarray(frozen["token_context_label_count"], dtype=float)
    sums = np.asarray(frozen["token_context_label_sum"], dtype=float)
    fallback = np.asarray(frozen["token_context_weights"], dtype=float)
    weights = np.asarray(frozen["objective_weights"], dtype=float)

    rows: list[dict[str, Any]] = []

    def add(name: str, channels: dict[str, np.ndarray]) -> None:
        rows.append(
            {"method": name, **_evaluate(channels, grid, contexts, context_values, tokens, labels)}
        )

    identity = np.eye(k)
    add("no_privacy_identity_K64", {context: identity for context in contexts})
    add(
        f"no_privacy_partition_L{decoder.shape[0]}",
        {
            context: lift_block_channel(np.eye(decoder.shape[0]), assignment, decoder)
            for context in contexts
        },
    )
    add(
        "context_capt(stored)",
        {
            context: lift_block_channel(stored_channels["channels"][index], assignment, decoder)
            for index, context in enumerate(contexts)
        },
    )
    add(
        "context_ldp(stored)",
        {
            context: lift_block_channel(stored_channels["ldp_channels"][index], assignment, decoder)
            for index, context in enumerate(contexts)
        },
    )
    objectives = sorted({"empirical_logloss", run_objective})
    for objective in objectives:
        cost, _ = _context_aggregated_representation_cost(
            level_grid,
            counts,
            sums,
            objective=objective,
            eta=eta,
            hybrid_empirical_weight=hybrid,
            empty_token_context_weights=fallback,
        )
        print(f"  solving global K={k} LDP for {objective} ...", flush=True)
        channel = _solve_token_ldp(cost, weights, epsilon, time_limit)
        add(f"global_token_ldp_K64[{objective}]", {context: channel for context in contexts})
        if context_token_ldp:
            per_context: dict[str, np.ndarray] = {}
            level_index = {value: index for index, value in enumerate(levels)}
            for index, context in enumerate(contexts):
                if context in level_index and counts[:, level_index[context]].sum() > 0:
                    column_index = level_index[context]
                    token_weights = fallback[:, column_index]
                    context_counts = counts[:, column_index]
                    context_sums = sums[:, column_index]
                else:
                    token_weights = weights
                    context_counts = np.zeros(k)
                    context_sums = np.zeros(k)
                context_cost, _ = conditional_utility_cost(
                    grid[index],
                    context_counts,
                    context_sums,
                    objective=objective,
                    eta=eta,
                    hybrid_empirical_weight=hybrid,
                )
                print(f"  solving context K={k} LDP [{objective}] {context} ...", flush=True)
                per_context[context] = _solve_token_ldp(
                    context_cost, token_weights / token_weights.sum(), epsilon, time_limit
                )
            add(f"context_token_ldp_K64[{objective}]", per_context)

    result = pd.DataFrame(rows)
    stored = metrics.set_index("method")["expected_randomized_log_loss"]
    for method in ("context_capt", "context_ldp"):
        recomputed = float(
            result.loc[
                result["method"] == f"{method}(stored)", "expected_randomized_log_loss"
            ].iloc[0]
        )
        gap = MICRO * (recomputed - float(stored[method]))
        flag = "" if abs(gap) < 1e-3 else "  !! evaluation mismatch; decomposition unreliable"
        print(f"  consistency {method}: recomputed - test_metrics.csv = {gap:+.6f} micro{flag}")
    capt = float(
        result.loc[result["method"] == "context_capt(stored)", "expected_randomized_log_loss"].iloc[
            0
        ]
    )
    result["minus_capt_micro"] = MICRO * (result["expected_randomized_log_loss"] - capt)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.9g}"))
    out = run / "tables" / "diagnose_capt_vs_global_ldp.csv"
    result.to_csv(out, index=False)
    print(f"\n  written {out}")

    _section("6. Decomposition of (global LDP - CAPT) in micro-nats (positive = CAPT better)")
    loss = metrics.set_index("method")["expected_randomized_log_loss"]
    for objective in objectives:
        global_loss = float(
            result.loc[
                result["method"] == f"global_token_ldp_K64[{objective}]",
                "expected_randomized_log_loss",
            ].iloc[0]
        )
        steps = [
            (
                "partition penalty   global K64 -> shared_ldp(L)",
                global_loss,
                loss.get("shared_ldp"),
            ),
            (
                "context gain        shared_ldp -> context_ldp",
                loss.get("shared_ldp"),
                loss.get("context_ldp"),
            ),
            (
                "privacy relaxation  context_ldp -> capt_pre_repair",
                loss.get("context_ldp"),
                loss.get("context_capt_pre_repair"),
            ),
            (
                "repair cost         capt_pre_repair -> capt",
                loss.get("context_capt_pre_repair"),
                loss.get("context_capt"),
            ),
        ]
        print(f"\n  global baseline objective: {objective}")
        total = 0.0
        for name, before, after in steps:
            if before is None or after is None:
                print(f"    {name:52s} n/a")
                continue
            value = MICRO * (float(before) - float(after))
            total += value
            print(f"    {name:52s} {value:+12.4f}")
        print(f"    {'TOTAL = global - capt':52s} {total:+12.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fixed-eval", type=Path)
    parser.add_argument("--solve-global", action="store_true")
    parser.add_argument("--context-token-ldp", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--time-limit", type=float, default=600)
    args = parser.parse_args()
    run = _run_dir(args.run)
    print(f"run directory: {run}")
    config = report_config(run)
    metrics = report_test_metrics(run)
    report_contexts(run)
    report_fixed_eval(args.fixed_eval)
    metadata = json.loads((run / "context_stratified_metadata.json").read_text(encoding="utf-8"))
    for key in (
        "max_repair_lambda",
        "mass_weighted_repair_lambda",
        "conservative_max_realized_epsilon",
    ):
        if key in metadata:
            print(f"  metadata {key}: {metadata[key]}")
    if args.solve_global:
        solve_global(
            run,
            config,
            metrics,
            data_root=args.data_root,
            context_token_ldp=args.context_token_ldp,
            time_limit=args.time_limit,
        )
    else:
        print(
            "\n(re-run with --solve-global to compute the matched global K=64 LDP and the"
            " full decomposition; add --context-token-ldp for per-context K=64 LDP)"
        )


if __name__ == "__main__":
    main()
