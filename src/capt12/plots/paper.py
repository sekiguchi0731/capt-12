from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = sns.color_palette("colorblind")
PLOT_RANDOM_SEED = 0
FIXED_ARTIFACT_TIME = datetime(2000, 1, 1, tzinfo=UTC)
matplotlib.rcParams["svg.hashsalt"] = "capt12-paper-artifacts"
FIGURE_NAMES = {
    1: "privacy_utility_frontier",
    2: "theorem4_gap_decomposition",
    3: "scaling_with_L",
    4: "privacy_audit_diagnostic",
    5: "sample_complexity_robustness",
    6: "profile_attribute_heatmap",
    7: "reference_score_surrogate_utility",
    8: "ablation",
}


def _result_source_git_sha(frame: pd.DataFrame, source: Path) -> str:
    if "source_git_sha" not in frame.columns:
        raise ValueError(
            f"result table has no source_git_sha provenance: {source}; "
            "regenerate it with the current pipeline"
        )
    values = frame["source_git_sha"]
    if frame.empty or values.isna().any():
        raise ValueError(
            f"result table has incomplete source_git_sha provenance: {source}; "
            "regenerate it with the current pipeline"
        )
    source_shas = {str(value).strip() for value in values}
    if "" in source_shas or len(source_shas) != 1:
        raise ValueError(
            f"result table mixes source_git_sha values: {source} "
            f"({sorted(source_shas)!r})"
        )
    return next(iter(source_shas))


def read_results(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if path.is_file():
        frame = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        )
        _result_source_git_sha(frame, path)
        return frame
    files = sorted(path.rglob("metrics.parquet"))
    summary = path / "synthetic_theorem4_results.parquet"
    if summary.exists():
        files.append(summary)
    frames = []
    source_shas: dict[str, list[str]] = {}
    seen = set()
    for file in files:
        resolved = file.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        frame = pd.read_parquet(file)
        source_sha = _result_source_git_sha(frame, file)
        source_shas.setdefault(source_sha, []).append(str(file))
        frame["source_file"] = str(file)
        frames.append(frame)
    if not frames:
        csvs = sorted(path.rglob("metrics.csv"))
        for file in csvs:
            frame = pd.read_csv(file)
            source_sha = _result_source_git_sha(frame, file)
            source_shas.setdefault(source_sha, []).append(str(file))
            frames.append(frame.assign(source_file=str(file)))
    if not frames:
        raise FileNotFoundError(f"no standardized metrics tables under {path}")
    if len(source_shas) != 1:
        details = "; ".join(
            f"{source_sha}: {', '.join(source_files)}"
            for source_sha, source_files in sorted(source_shas.items())
        )
        raise ValueError(
            "refusing to aggregate result tables from different source_git_sha "
            f"values; select a single-source directory ({details})"
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def _numeric(frame: pd.DataFrame, name: str, default: float = math.nan) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def theorem4_gap_table(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "experiment_series" in frame and (
        frame["experiment_series"] == "synthetic_theorem4"
    ).any():
        frame = frame[frame["experiment_series"] == "synthetic_theorem4"]
    if "source_file" in frame and frame["source_file"].astype(str).str.contains(
        "synthetic_theorem4_results.parquet"
    ).any():
        frame = frame[
            frame["source_file"]
            .astype(str)
            .str.contains("synthetic_theorem4_results.parquet")
        ]
    if "distortion" in frame:
        frame = frame[frame["distortion"] == "retention"]
    if "case" in frame:
        frame["case"] = frame["case"].fillna("standard")
    match_keys = [
        key
        for key in [
            "K",
            "target_epsilon",
            "seed",
            "case",
            "partition",
            "decoder",
            "distortion",
            "pi_weighting",
            "cost_aggregation",
            "confidence",
            "problem_signature",
        ]
        if key in frame
    ]
    capt_rows = frame[frame["mechanism"] == "capt_block"]
    if capt_rows.empty or "L" not in capt_rows:
        return pd.DataFrame()
    if "decoder_cover_retention" in capt_rows:
        decoder_cover = (
            capt_rows.groupby([*match_keys, "L"], dropna=False)[
                "decoder_cover_retention"
            ]
            .mean()
            .rename("U_decoder_cover")
            .reset_index()
        )
    else:
        decoder_cover = capt_rows[[*match_keys, "L"]].drop_duplicates()
        decoder_cover["U_decoder_cover"] = np.nan
    capt = (
        capt_rows.groupby([*match_keys, "L"], dropna=False)["utility_retention"]
        .mean()
        .rename("U_CAPT")
        .reset_index()
    )
    full_mask = frame["mechanism"] == "capt_full"
    for eligibility in (
        "full_oracle_optimized",
        "full_verification_valid",
        "full_comparison_eligible",
        "feasible",
    ):
        if eligibility not in frame:
            full_mask &= False
        else:
            full_mask &= frame[eligibility].fillna(False).astype(bool)
    full = (
        frame[full_mask]
        .groupby(match_keys, dropna=False)["utility_retention"]
        .mean()
        .rename("U_full")
        .reset_index()
    )
    envelope = (
        frame[frame["mechanism"] == "theorem4_envelope"]
        .groupby(match_keys, dropna=False)["utility_retention"]
        .mean()
        .rename("U_envelope")
        .reset_index()
    )
    result = (
        capt.merge(full, on=match_keys, how="left")
        .merge(envelope, on=match_keys, how="left")
        .merge(decoder_cover, on=[*match_keys, "L"], how="left")
    )
    denom_full = result["U_full"] - result["U_decoder_cover"]
    denom_env = result["U_envelope"] - result["U_decoder_cover"]
    result["attainment_full"] = np.where(abs(denom_full) > 1e-12, (result["U_CAPT"] - result["U_decoder_cover"]) / denom_full, np.nan)
    result["attainment_envelope"] = np.where(abs(denom_env) > 1e-12, (result["U_CAPT"] - result["U_decoder_cover"]) / denom_env, np.nan)
    result["converse_looseness"] = np.where(abs(denom_env) > 1e-12, (result["U_envelope"] - result["U_full"]) / denom_env, np.nan)
    for column in ["attainment_full", "attainment_envelope", "converse_looseness"]:
        result.loc[np.isclose(result[column], 0.0, atol=1e-12), column] = 0.0
        result.loc[np.isclose(result[column], 1.0, atol=1e-12), column] = 1.0
    return result


def _save(fig: plt.Figure, source: pd.DataFrame, output: Path, name: str, caption: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    source.to_csv(output / f"{name}_source.csv", index=False)
    (output / f"{name}_caption.txt").write_text(caption.strip() + "\n")
    paths = []
    for suffix in ("pdf", "svg", "png"):
        target = output / f"{name}.{suffix}"
        if suffix == "pdf":
            metadata = {
                "Creator": "CAPT-12",
                "Producer": "CAPT-12",
                "CreationDate": FIXED_ARTIFACT_TIME,
                "ModDate": FIXED_ARTIFACT_TIME,
            }
        elif suffix == "svg":
            metadata = {"Creator": "CAPT-12", "Date": "2000-01-01T00:00:00Z"}
        else:
            metadata = {"Software": "CAPT-12"}
        fig.savefig(target, bbox_inches="tight", dpi=220, metadata=metadata)
        paths.append(target)
    plt.close(fig)
    return paths


def _empty_or_axis(ax: plt.Axes, frame: pd.DataFrame, required: list[str]) -> bool:
    missing = [column for column in required if column not in frame or frame[column].dropna().empty]
    if missing:
        ax.text(0.5, 0.5, f"No applicable standardized results\nmissing: {', '.join(missing)}", ha="center", va="center")
        ax.set_axis_off()
        return True
    return False


def figure1(frame: pd.DataFrame, output: Path) -> list[Path]:
    source = frame.copy()
    source["epsilon_x"] = _numeric(source, "certificate_epsilon").fillna(_numeric(source, "target_epsilon"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics = ["expected_distortion", "LLHCompVN", "weighted_log_loss", "calibration_ratio"]
    for ax, metric in zip(axes.flat, metrics, strict=True):
        if _empty_or_axis(ax, source, ["epsilon_x", metric, "mechanism"]):
            continue
        sns.lineplot(data=source, x="epsilon_x", y=metric, hue="mechanism", style="L" if "L" in source else None, estimator="mean", errorbar=("ci", 95), seed=PLOT_RANDOM_SEED, palette="colorblind", ax=ax)
        uncertified = source.get("certified", pd.Series(True, index=source.index)).fillna(False) == False  # noqa: E712
        ax.scatter(source.loc[uncertified, "epsilon_x"], _numeric(source.loc[uncertified], metric), marker="x", color="black", s=28, label="uncertified/infeasible")
        ax.set_title(metric.replace("_", " "))
    fig.suptitle("Privacy–utility frontier")
    return _save(fig, source, output, FIGURE_NAMES[1], "Target or certified epsilon versus frozen reference-score surrogate utility. Crosses are uncertified/infeasible. These panels do not establish deployed CTR or auction-utility preservation.")


def figure2(frame: pd.DataFrame, output: Path) -> list[Path]:
    source = theorem4_gap_table(frame)
    condition_keys = [
        key
        for key in ["K", "target_epsilon", "case", "partition", "decoder"]
        if key in source
    ]
    conditions = (
        source[condition_keys].drop_duplicates().to_dict("records")
        if condition_keys and not source.empty
        else [{}]
    )
    fig, axes = plt.subplots(
        max(1, len(conditions)),
        2,
        figsize=(11, max(4, 3.3 * len(conditions))),
        squeeze=False,
    )
    for row_idx, condition in enumerate(conditions):
        subset = source
        for key, value in condition.items():
            subset = subset[subset[key] == value]
        if not _empty_or_axis(
            axes[row_idx, 0], subset, ["L", "U_CAPT", "U_full", "U_envelope"]
        ):
            melted = subset.melt(
                id_vars=[column for column in [*condition_keys, "seed", "L"] if column in subset],
                value_vars=["U_CAPT", "U_full", "U_envelope", "U_decoder_cover"],
                var_name="quantity",
                value_name="retention",
            )
            sns.lineplot(
                data=melted,
                x="L",
                y="retention",
                hue="quantity",
                marker="o",
                errorbar=("pi", 100),
                seed=PLOT_RANDOM_SEED,
                palette="colorblind",
                ax=axes[row_idx, 0],
            )
        if not _empty_or_axis(
            axes[row_idx, 1],
            subset,
            ["L", "attainment_full", "attainment_envelope", "converse_looseness"],
        ):
            melted = subset.melt(
                id_vars=[column for column in [*condition_keys, "seed", "L"] if column in subset],
                value_vars=[
                    "attainment_full",
                    "attainment_envelope",
                    "converse_looseness",
                ],
                var_name="gap",
                value_name="fraction",
            )
            sns.lineplot(
                data=melted,
                x="L",
                y="fraction",
                hue="gap",
                marker="o",
                errorbar=("pi", 100),
                seed=PLOT_RANDOM_SEED,
                palette="colorblind",
                ax=axes[row_idx, 1],
            )
        short_values = []
        if "K" in condition:
            short_values.append(f"K={condition['K']:g}")
        if "target_epsilon" in condition:
            short_values.append(f"ε={condition['target_epsilon']:g}")
        for key in ["case", "partition", "decoder"]:
            if key in condition and source[key].nunique(dropna=False) > 1:
                short_values.append(f"{key}={condition[key]}")
        title = ", ".join(short_values) or "matched setting"
        axes[row_idx, 0].set_title(f"Retention — {title}", fontsize=10)
        axes[row_idx, 1].set_title(f"Gap fractions — {title}", fontsize=10)
    fig.suptitle("Theorem-4 gap decomposition (retention only)", y=0.995)
    fig.subplots_adjust(hspace=0.55, top=0.96)
    caption = "Known-population synthetic results only. Each row fixes K, epsilon, case, partition, and decoder. Markers are means over the three seeds and bands show the observed seed minimum and maximum; no bootstrap confidence interval is claimed. U_full is included only when the full LP is feasible, solver-optimal, independently verified, and marked comparison-eligible. U_CAPT(L) <= U_full <= U_envelope; L=K reaches full, not necessarily the envelope. Gap attainment is normalized from U_decoder_cover, the best input-independent cover representable by that row's fixed partition and common decoder, so it is a within-CAPT-class baseline rather than the separate token-level common-cover mechanism. Attainment_full is N/A when U_full equals U_decoder_cover because the normalizing gap is zero."
    return _save(fig, source, output, FIGURE_NAMES[2], caption)


def figure3(frame: pd.DataFrame, output: Path) -> list[Path]:
    source = frame.copy()
    if "nested_partition" in source:
        source["partition_mode"] = source["nested_partition"].fillna(False).map(
            {True: "nested", False: "non-nested"}
        )
    gap = theorem4_gap_table(frame)
    if not gap.empty:
        merge_keys = [
            key
            for key in [
                "K",
                "L",
                "target_epsilon",
                "seed",
                "case",
                "partition",
                "decoder",
                "distortion",
                "pi_weighting",
                "cost_aggregation",
                "confidence",
                "problem_signature",
            ]
            if key in source and key in gap
        ]
        if merge_keys:
            source = source.merge(gap[merge_keys + ["attainment_full", "attainment_envelope"]], on=merge_keys, how="left")
    metrics = ["expected_distortion", "attainment_full", "attainment_envelope", "certificate_epsilon", "certificate_width", "solver_runtime", "estimated_memory", "variable_count", "constraint_count", "certified"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    for ax, metric in zip(axes.flat, metrics, strict=True):
        if _empty_or_axis(ax, source, ["L", metric]):
            continue
        hue = "partition_mode" if "partition_mode" in source else "mechanism"
        sns.lineplot(data=source, x="L", y=metric, hue=hue, marker="o", errorbar=("ci", 95), seed=PLOT_RANDOM_SEED, palette="colorblind", ax=ax)
        ax.set_title(metric.replace("_", " "))
    fig.suptitle("Scaling with L")
    return _save(fig, source, output, FIGURE_NAMES[3], "Scaling panels distinguish nested and non-nested partitions. Utility monotonicity is not claimed for independently fitted non-nested partitions.")


def figure4(frame: pd.DataFrame, output: Path) -> list[Path]:
    source = frame[frame.get("mechanism", pd.Series(index=frame.index)) == "capt_block"].copy()
    if "experiment_series" in source and (
        source["experiment_series"] == "criteo_smoke"
    ).any():
        source = source[source["experiment_series"] == "criteo_smoke"]
    source["profile"] = source.get("profile", pd.Series("all", index=source.index)).fillna("all")
    fig, ax = plt.subplots(figsize=(9, 4))
    if not _empty_or_axis(
        ax,
        source,
        [
            "group_sample_size",
            "profile",
            "lower_audit_epsilon",
            "certificate_global_upper_epsilon",
        ],
    ):
        melted = source.melt(
            id_vars=[
                column
                for column in [
                    "group_sample_size",
                    "profile",
                    "seed",
                    "universal_cover_activated",
                    "audit_certificate_comparable",
                ]
                if column in source
            ],
            value_vars=[
                "lower_audit_epsilon",
                "witness_matched_upper_epsilon",
                "certificate_global_upper_epsilon",
            ],
            var_name="quantity",
            value_name="epsilon",
        ).dropna(subset=["epsilon"])
        sns.scatterplot(
            data=melted,
            x="group_sample_size",
            y="epsilon",
            hue="quantity",
            style="profile",
            palette="colorblind",
            ax=ax,
        )
        ax.axhline(0.0, color="0.3", linewidth=0.8)
        ax.set_title("Criteo CAPT audit diagnostic (profile-wide common cover)")
        ax.set_ylabel("epsilon (numerical zero at common cover)")
    return _save(fig, source, output, FIGURE_NAMES[4], "Per-run CAPT audit diagnostic; mechanisms and profiles are not averaged together. The lower witness uses a D_attack_train-fixed family and alpha/(2T) one-sided allocation under the declared user-day i.i.d. assumption. A witness-matched upper is recorded only when the witness endpoints have an embedded certified path; its gap is comparable only with documented population-bridge evidence. The global certificate upper is shown separately. Current Criteo common-cover rows are fail-safe outcomes, not evidence of a nontrivial two-sided bracket.")


def figure5(frame: pd.DataFrame, output: Path) -> list[Path]:
    source = frame.copy()
    axes_names = ["group_sample_size", "L", "dp_hist_epsilon", "alpha_cert", "tv_radius"]
    metrics = ["expected_distortion", "certificate_width", "nontrivial_channel"]
    fig, axes = plt.subplots(len(metrics), len(axes_names), figsize=(15, 8))
    for row, metric in enumerate(metrics):
        for col, x_name in enumerate(axes_names):
            ax = axes[row, col]
            if _empty_or_axis(ax, source, [x_name, metric]):
                continue
            sns.lineplot(data=source, x=x_name, y=metric, marker="o", errorbar=("ci", 95), seed=PLOT_RANDOM_SEED, color=PALETTE[col % len(PALETTE)], ax=ax)
    fig.suptitle("Sample complexity and robustness")
    return _save(fig, source, output, FIGURE_NAMES[5], "Utility, certificate width/gap, and certificate success versus D_cert size, L, DP epsilon, confidence level, and TV shift. DP-aware points remain experimental unless explicitly certified by a future validated method.")


def figure6(frame: pd.DataFrame, output: Path) -> list[Path]:
    source = frame.copy()
    source["profile"] = source.get("profile", pd.Series("all", index=source.index)).fillna("all")
    source["context"] = source.get("context", pd.Series("all", index=source.index)).fillna("all")
    metrics = ["certificate_epsilon", "lower_audit_epsilon", "expected_distortion", "group_sample_size", "rare_group_mass"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    for ax, metric in zip(axes, metrics, strict=True):
        if _empty_or_axis(ax, source, ["profile", "context", metric]):
            continue
        table = source.pivot_table(index="profile", columns="context", values=metric, aggfunc="mean")
        sns.heatmap(table, cmap="viridis", ax=ax, cbar=True)
        ax.set_title(metric.replace("_", " "))
    return _save(fig, source, output, FIGURE_NAMES[6], "Certified epsilon, lower witness, frozen reference-score surrogate utility, sample size, and rare-group mass by anonymous profile/attribute/context bucket. rare_group_mass is not the mechanism fallback share; profile-wide fallback is reported separately.")


def figure7(frame: pd.DataFrame, output: Path) -> list[Path]:
    source = frame.copy()
    target_columns = [column for column in source if column.startswith("target_ctr_") and column.endswith(("LLHCompVN", "weighted_log_loss", "calibration_ratio"))]
    metrics = ["LLHCompVN", "calibration_ratio", "weighted_log_loss", *target_columns[:3]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, metric in zip(axes.flat, metrics[:6], strict=False):
        if _empty_or_axis(ax, source, ["mechanism", metric]):
            continue
        sns.barplot(data=source, x="mechanism", y=metric, hue="L" if "L" in source else None, errorbar=("ci", 95), seed=PLOT_RANDOM_SEED, palette="colorblind", ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(metric)
    return _save(fig, source, output, FIGURE_NAMES[7], "Frozen reference-score surrogate diagnostics: LLHCompVN, calibration ratio, weighted log-loss, and configured target-CTR reweighting. No downstream CTR model is retrained on sanitized (O,B), so this figure does not establish deployed bidding or CTR utility.")


def figure8(frame: pd.DataFrame, output: Path) -> list[Path]:
    source = frame.copy()
    dimensions = ["phi", "fixed_model", "distortion", "partition", "decoder", "L", "confidence", "tv_radius"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, dimension in zip(axes.flat, dimensions, strict=True):
        metric = "expected_distortion" if "expected_distortion" in source else "utility_retention"
        if _empty_or_axis(ax, source, [dimension, metric]):
            continue
        sns.pointplot(data=source, x=dimension, y=metric, errorbar=("ci", 95), seed=PLOT_RANDOM_SEED, color=PALETTE[0], ax=ax)
        ax.tick_params(axis="x", rotation=35)
        ax.set_title(dimension)
    return _save(fig, source, output, FIGURE_NAMES[8], "Ablations for frozen encoder, reference model, distortion, partition, common decoder, L, confidence construction, and TV shift.")


def _appendix(frame: pd.DataFrame, output: Path) -> list[Path]:
    generated = []
    specifications: list[tuple[str, str, str]] = [
        ("block_channel_heatmap", "L", "certificate_epsilon"),
        ("token_block_assignment", "K", "L"),
        ("block_frequency", "L", "group_sample_size"),
        ("randomization_memoization", "mechanism", "exact_token_retention"),
        ("solver_convergence", "solver_iterations", "solver_runtime"),
        ("theorem4_counterexample", "mechanism", "utility_retention"),
    ]
    for name, x_name, y_name in specifications:
        source = frame.copy()
        if name == "theorem4_counterexample" and "case" in source:
            source = source[source["case"] == "theorem4_counterexample"]
        fig, ax = plt.subplots(figsize=(6, 4))
        if not _empty_or_axis(ax, source, [x_name, y_name]):
            if pd.api.types.is_numeric_dtype(source[x_name]):
                sns.lineplot(data=source, x=x_name, y=y_name, hue="mechanism" if "mechanism" in source else None, marker="o", errorbar=("ci", 95), seed=PLOT_RANDOM_SEED, palette="colorblind", ax=ax)
            else:
                sns.barplot(data=source, x=x_name, y=y_name, hue="L" if "L" in source else None, errorbar=("ci", 95), seed=PLOT_RANDOM_SEED, palette="colorblind", ax=ax)
        generated.extend(_save(fig, source, output, f"appendix_{name}", f"Appendix diagnostic: {name.replace('_', ' ')}. Source values come from standardized result tables."))
    return generated


def plot_paper_suite(input_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    frame = read_results(input_path)
    output = Path(output_dir) if output_dir else Path(input_path) / "paper_figures"
    functions: list[Callable[[pd.DataFrame, Path], list[Path]]] = [figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8]
    generated = []
    for function in functions:
        generated.extend(function(frame, output))
    generated.extend(_appendix(frame, output))
    return generated
