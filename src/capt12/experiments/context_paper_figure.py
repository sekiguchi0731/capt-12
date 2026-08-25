from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from capt12.config import run_id
from capt12.utils.artifacts import git_sha, sha256_file

_OBJECTIVE = "empirical_logloss"
_REPRESENTATION = "objective_aligned"
_FIGURE_VERSION = 1
_BLUE = "#2563A6"
_INK = "#1F2937"
_GREY = "#6B7280"
_LIGHT_BLUE = "#93B7D5"


def _read_zip_csv(path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        try:
            payload = archive.read(member)
        except KeyError as error:
            raise ValueError(f"missing {member} in {path}") from error
    return pd.read_csv(io.BytesIO(payload))


def _read_zip_json(path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        try:
            payload = archive.read(member)
        except KeyError as error:
            raise ValueError(f"missing {member} in {path}") from error
    return json.loads(payload)


def _as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"invalid boolean values in {series.name}")
    return normalized == "true"


def _load_frontier(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = _read_zip_json(path, "grid/context_epsilon_grid_metadata.json")
    frame = _read_zip_csv(path, "grid/tables/epsilon_seed_results.csv")
    if int(metadata["L"]) != 16:
        raise ValueError("main frontier requires L=16")
    if metadata["context_utility_objective"] != _OBJECTIVE:
        raise ValueError(f"main frontier requires {_OBJECTIVE}")
    if metadata["context_representation_mode"] != _REPRESENTATION:
        raise ValueError(f"main frontier requires {_REPRESENTATION}")
    if set(frame["L"].astype(int)) != {16}:
        raise ValueError("epsilon-grid table mixes block sizes")
    if set(frame["utility_objective"].astype(str)) != {_OBJECTIVE}:
        raise ValueError("epsilon-grid table mixes utility objectives")
    if set(frame["representation_mode"].astype(str)) != {_REPRESENTATION}:
        raise ValueError("epsilon-grid table mixes representation modes")
    if not _as_boolean(frame["all_certificates_valid"]).all():
        raise ValueError("epsilon-grid table contains an invalid certificate")
    realized = frame["conservative_max_realized_epsilon"].to_numpy(float)
    targets = frame["epsilon"].to_numpy(float)
    if not np.isfinite(realized).all() or np.any(realized > targets):
        raise ValueError("epsilon-grid table fails conservative pure-epsilon verification")
    frame = frame.copy()
    frame["robust_certified_upper_epsilon"] = realized
    frame["optimal_ldp_logloss_gain_micro"] = (
        -1e6 * frame["test_capt_minus_ldp_expected_randomized_log_loss"].to_numpy(float)
    )
    return frame, metadata


def _load_block_bundle(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = _read_zip_json(path, "comparison/context_cost_comparison_metadata.json")
    frame = _read_zip_csv(path, "comparison/tables/objective_seed_results.csv")
    selected = frame.loc[frame["utility_objective"].astype(str) == _OBJECTIVE].copy()
    if len(selected) == 0:
        raise ValueError(f"{path} does not contain {_OBJECTIVE}")
    if set(selected["representation_mode"].astype(str)) != {_REPRESENTATION}:
        raise ValueError(f"{path} is not objective-aligned")
    if not _as_boolean(selected["all_certificates_valid"]).all():
        raise ValueError(f"{path} contains an invalid certificate")
    block_count = int(metadata["L"])
    if set(selected["L"].astype(int)) != {block_count}:
        raise ValueError(f"{path} has inconsistent block counts")
    if float(metadata["epsilon"]) != 1.0 or set(selected["epsilon"].astype(float)) != {1.0}:
        raise ValueError("block-size sensitivity requires epsilon=1")
    selected["optimal_ldp_logloss_gain_micro"] = (
        -1e6 * selected["test_capt_minus_ldp_expected_randomized_log_loss"].to_numpy(float)
    )
    return selected, metadata


def _load_block_sensitivity(paths: list[Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(paths) < 2:
        raise ValueError("block-size sensitivity requires at least two comparison bundles")
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for path in paths:
        frame, item = _load_block_bundle(path)
        frames.append(frame)
        metadata.append(item)
    block_counts = [int(item["L"]) for item in metadata]
    if len(set(block_counts)) != len(block_counts):
        raise ValueError("duplicate block size in sensitivity inputs")
    source_shas = {str(item["experiment_source_git_sha"]) for item in metadata}
    if len(source_shas) != 1:
        raise ValueError("block-size inputs must share one experiment source SHA")
    seed_sets = {
        tuple(sorted(frame["frozen_design_seed"].astype(int).tolist())) for frame in frames
    }
    if len(seed_sets) != 1:
        raise ValueError("block-size inputs must share one paired seed family")
    reference_encoders: dict[int, str] | None = None
    for frame in frames:
        encoders = {
            int(row.frozen_design_seed): str(row.encoder_sha256) for row in frame.itertuples()
        }
        if reference_encoders is None:
            reference_encoders = encoders
        elif encoders != reference_encoders:
            raise ValueError("encoder hashes differ across block sizes")
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["frozen_design_seed", "L"]
    )
    return combined, {
        "block_counts": sorted(block_counts),
        "source_git_sha": next(iter(source_shas)),
        "frozen_design_seeds": list(next(iter(seed_sets))),
    }


def _plot(frontier: pd.DataFrame, blocks: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.05),
        constrained_layout=True,
        sharey=True,
    )

    axis = axes[0]
    for _, group in frontier.groupby("frozen_design_seed", sort=True):
        group = group.sort_values("epsilon")
        axis.plot(
            group["robust_certified_upper_epsilon"],
            group["optimal_ldp_logloss_gain_micro"],
            color=_LIGHT_BLUE,
            lw=0.75,
            alpha=0.65,
            zorder=1,
        )
        axis.scatter(
            group["robust_certified_upper_epsilon"],
            group["optimal_ldp_logloss_gain_micro"],
            s=22,
            facecolor="white",
            edgecolor=_BLUE,
            linewidth=0.8,
            alpha=0.8,
            zorder=2,
        )
    means = (
        frontier.groupby("epsilon", as_index=False)
        .agg(
            robust_certified_upper_epsilon=("robust_certified_upper_epsilon", "mean"),
            optimal_ldp_logloss_gain_micro=("optimal_ldp_logloss_gain_micro", "mean"),
        )
        .sort_values("robust_certified_upper_epsilon")
    )
    axis.plot(
        means["robust_certified_upper_epsilon"],
        means["optimal_ldp_logloss_gain_micro"],
        color=_BLUE,
        marker="o",
        markersize=4.8,
        lw=2.1,
        zorder=3,
    )
    axis.axhline(0, color=_GREY, lw=1, ls="--")
    axis.set_title(
        "(a) Certified privacy–utility trade-off\n" r"$L=16$",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel(r"Robust certified upper $\bar{\epsilon}$")
    axis.set_ylabel(
        r"Improvement over optimal $\epsilon$-LDP"
        "\n"
        r"($\mu$nats/display; higher is better)"
    )
    axis.grid(axis="y", color="#E5E7EB", lw=0.6)

    axis = axes[1]
    seeds = sorted(blocks["frozen_design_seed"].astype(int).unique())
    offsets = dict(zip(seeds, np.linspace(-0.28, 0.28, len(seeds)), strict=True))
    for seed in seeds:
        group = blocks.loc[blocks["frozen_design_seed"] == seed].sort_values("L")
        x = group["L"].to_numpy(float) + offsets[seed]
        y = group["optimal_ldp_logloss_gain_micro"].to_numpy(float)
        axis.plot(x, y, color=_LIGHT_BLUE, lw=0.75, alpha=0.65, zorder=1)
        axis.scatter(
            x,
            y,
            s=24,
            facecolor="white",
            edgecolor=_BLUE,
            linewidth=0.9,
            zorder=2,
        )
    block_counts = sorted(blocks["L"].astype(int).unique())
    means = blocks.groupby("L")["optimal_ldp_logloss_gain_micro"].mean()
    for block_count in block_counts:
        axis.hlines(
            means.loc[block_count],
            block_count - 0.52,
            block_count + 0.52,
            color=_BLUE,
            lw=2.4,
            zorder=4,
        )
    axis.axhline(0, color=_GREY, lw=1, ls="--")
    axis.set_title(
        "(b) Block-size sensitivity\n" r"$\epsilon=1$",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("Number of blocks L")
    axis.set_xticks(block_counts, [str(value) for value in block_counts])
    axis.grid(axis="y", color="#E5E7EB", lw=0.6)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=_LIGHT_BLUE,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=_BLUE,
            markeredgewidth=0.9,
            lw=0.75,
            label="Paired frozen-design seed",
        ),
        Line2D([0], [0], color=_BLUE, lw=2.4, label="Seed mean"),
        Line2D(
            [0],
            [0],
            color=_GREY,
            lw=1,
            ls="--",
            label=r"Parity with optimal $\epsilon$-LDP",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.4,
    )

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fixed_metadata = {
        "Creator": "capt12 context-paper-figure",
        "CreationDate": datetime(2000, 1, 1, tzinfo=UTC),
    }
    fig.savefig(
        figure_dir / "context_capt_main.pdf",
        bbox_inches="tight",
        metadata=fixed_metadata,
    )
    fig.savefig(
        figure_dir / "context_capt_main.png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "capt12 context-paper-figure"},
    )
    plt.close(fig)


def _write_bundle(output_dir: Path) -> Path:
    bundle = output_dir / "context_capt_main_figure_bundle.zip"
    temporary = output_dir / ".context_capt_main_figure_bundle.zip.tmp"
    external_manifest = output_dir / "context_capt_main_figure_bundle_manifest.json"
    members = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path not in {bundle, temporary, external_manifest}
    )
    fixed_timestamp = (2000, 1, 1, 0, 0, 0)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in members:
                info = zipfile.ZipInfo(path.relative_to(output_dir).as_posix(), fixed_timestamp)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        temporary.replace(bundle)
    finally:
        temporary.unlink(missing_ok=True)
    return bundle


def run_context_paper_figure(
    epsilon_grid_bundle: Path,
    block_comparison_bundles: list[Path],
    output_root: Path = Path("outputs/paper_figures/context_capt_main"),
) -> Path:
    """Create the two-panel paper figure from independently certified bundles."""
    frontier, frontier_metadata = _load_frontier(epsilon_grid_bundle)
    blocks, block_metadata = _load_block_sensitivity(block_comparison_bundles)
    frontier_seeds = tuple(
        int(value) for value in sorted(frontier["frozen_design_seed"].astype(int).unique())
    )
    if frontier_seeds != tuple(block_metadata["frozen_design_seeds"]):
        raise ValueError("frontier and block-size panels must share one seed family")
    signature = {
        "experiment": "context_capt_main_paper_figure",
        "version": _FIGURE_VERSION,
        "objective": _OBJECTIVE,
        "representation": _REPRESENTATION,
        "epsilon_grid_bundle_sha256": sha256_file(epsilon_grid_bundle),
        "block_comparison_bundle_sha256": [
            sha256_file(path) for path in block_comparison_bundles
        ],
    }
    output_dir = output_root / run_id(signature)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    frontier.to_csv(output_dir / "tables" / "frontier_seed_points.csv", index=False)
    blocks.to_csv(output_dir / "tables" / "block_sensitivity_seed_points.csv", index=False)
    metadata = {
        **signature,
        "analysis_git_sha": git_sha(),
        "frontier_source_git_sha": frontier_metadata["source_git_sha"],
        "block_sensitivity_source_git_sha": block_metadata["source_git_sha"],
        "epsilon_values": sorted(frontier["epsilon"].astype(float).unique()),
        "block_counts": block_metadata["block_counts"],
        "frozen_design_seeds": list(frontier_seeds),
        "all_certificates_valid": True,
        "mass_method_included": False,
        "missing_block_counts": [32] if 32 not in block_metadata["block_counts"] else [],
        "input_paths": {
            "epsilon_grid_bundle": str(epsilon_grid_bundle.resolve()),
            "block_comparison_bundles": [
                str(path.resolve()) for path in block_comparison_bundles
            ],
        },
    }
    (output_dir / "context_capt_main_figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    caption = (
        "Held-out utility of CAPT relative to the context-wise optimal epsilon-LDP "
        "baseline under the matching block representation, decoder, and "
        "empirical-log-loss objective. Positive values favor CAPT. (a) Privacy-utility "
        "trade-off for L=16; every CAPT mechanism satisfies its robust certificate "
        "epsilon_bar <= epsilon for epsilon in {0.5, 1, 2}. (b) Sensitivity to the "
        "number of blocks at epsilon=1. Open circles and thin lines show paired "
        "frozen-design seeds, and thick lines indicate seed means. Seeds perturb the "
        "mechanism design and are not independent test samples; no seed-level "
        "confidence interval is implied. L=32 and MaSS are omitted because no "
        "completed, formally matched result was available.\n"
    )
    (output_dir / "caption.txt").write_text(caption, encoding="utf-8")
    latex_caption = r"""\caption{
Held-out utility of CAPT relative to the context-wise optimal
$\epsilon$-LDP baseline under the matching block representation,
decoder, and empirical-log-loss objective.
Positive values favor CAPT.
(a) Privacy--utility trade-off for $L=16$; every CAPT mechanism
satisfies its robust certificate
$\bar{\epsilon}\leq\epsilon$ for
$\epsilon\in\{0.5,1,2\}$.
(b) Sensitivity to the number of blocks at $\epsilon=1$.
Open circles and thin lines show paired frozen-design seeds, and
thick lines indicate seed means.
Seeds perturb the mechanism design and are not independent test
samples; no seed-level confidence interval is implied.
}
"""
    (output_dir / "caption.tex").write_text(latex_caption, encoding="utf-8")
    _plot(frontier, blocks, output_dir)
    bundle = _write_bundle(output_dir)
    manifest = {
        "bundle": bundle.name,
        "bundle_size_bytes": bundle.stat().st_size,
        "bundle_sha256": sha256_file(bundle),
        "files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path != bundle
        ],
    }
    (output_dir / "context_capt_main_figure_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir
