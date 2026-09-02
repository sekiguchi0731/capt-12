from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from capt12.plots.paper import read_results


def _write_metrics(path: Path, source_shas: list[str] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {"mechanism": ["capt_block"]}
    if source_shas is not None:
        data["source_git_sha"] = source_shas
        data["mechanism"] = ["capt_block"] * len(source_shas)
    pd.DataFrame(data).to_parquet(path, index=False)


def test_read_results_accepts_one_source_sha(tmp_path: Path) -> None:
    first = tmp_path / "run-a" / "metrics.parquet"
    second = tmp_path / "run-b" / "metrics.parquet"
    _write_metrics(first, ["a" * 40])
    _write_metrics(second, ["a" * 40])

    frame = read_results(tmp_path)

    assert set(frame["source_git_sha"]) == {"a" * 40}
    assert set(frame["source_file"]) == {str(first), str(second)}


def test_read_results_rejects_different_source_shas(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "run-a" / "metrics.parquet", ["a" * 40])
    _write_metrics(tmp_path / "run-b" / "metrics.parquet", ["b" * 40])

    with pytest.raises(ValueError, match="different source_git_sha"):
        read_results(tmp_path)


def test_read_results_checks_csv_fallback(tmp_path: Path) -> None:
    metrics = tmp_path / "run" / "tables" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "mechanism": ["capt_block", "common_cover"],
            "source_git_sha": ["a" * 40, "b" * 40],
        }
    ).to_csv(metrics, index=False)

    with pytest.raises(ValueError, match="mixes source_git_sha"):
        read_results(tmp_path)


@pytest.mark.parametrize("source_shas", [None, ["a" * 40, "b" * 40]])
def test_read_results_rejects_untraceable_table(
    tmp_path: Path, source_shas: list[str] | None
) -> None:
    metrics = tmp_path / "run" / "metrics.parquet"
    _write_metrics(metrics, source_shas)

    with pytest.raises(ValueError, match="source_git_sha"):
        read_results(tmp_path)
