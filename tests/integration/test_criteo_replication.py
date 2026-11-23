import hashlib
import json

import joblib
import pandas as pd
import pytest
from typer.testing import CliRunner

from capt12.cli import app
from capt12.data.preprocessing import FrozenCategoryMapper
from capt12.encoders.base import PrecomputedEncoder
from capt12.experiments import criteo_replication as rep
from capt12.utils.artifacts import sha256_file


def test_replication_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(rep, "DAYS", (25,))
    run, results, cache = (tmp_path / name for name in ("run", "results", "cache"))
    for path in (run / "models", results, cache):
        path.mkdir(parents=True)
    mapper = FrozenCategoryMapper(mappings={"a": {"g", "h"}, "b": set(map(str, range(31)))})
    files = dict(mapper="category_mapper.joblib", encoder="encoder.joblib")
    for key, obj in (("mapper", mapper), ("encoder", PrecomputedEncoder(2, "z"))):
        joblib.dump(obj, run / "models" / files[key])
    hashes = {key: sha256_file(run / "models" / name) for key, name in files.items()}
    for i in range(31):
        b = str(i)
        g, h = f"a|{b}|g", f"a|{b}|h"
        cfg = dict(
            K=2,
            L=32,
            public_context_value=b,
            profiles=["a"],
            context_cols=["b"],
            id_col="id",
            user_col="u",
            label_col="y",
            phi_source_cols=["z"],
        )
        path = run / f"certificate-demo-context-{i:02d}.json"
        path.write_text(
            json.dumps(
                dict(
                    resolved_config=cfg,
                    assignment=[0, 1],
                    component_hashes=hashes,
                    boxes={g: {}, h: {}},
                    adjacency=[dict(left=g, right=h), dict(left=h, right=g)],
                )
            )
        )
        rec = dict(
            certificate=str(path),
            certificate_sha256=sha256_file(path),
            context=b,
            K=2,
            L=32,
            splits={"D_test": [25]},
            profile="a",
            context_column="b",
            utility_sample_sizes={"train": 1, "validation": 1},
            capt_mechanism="binary-direct-cp",
            comparisons={
                key: {"lambda": 1}
                for key in ("equal_epsilon", "tv_calibrated", "uniform_calibrated")
            },
            calibration={"beta_R": 1},
            binary_channel=dict(
                high_blocks=[1],
                probability_of_one_low_high=[0.4, 0.6],
                fitted_output_ctr=[0.2, 0.8],
            ),
        )
        (results / f"{i:02d}.json").write_text(json.dumps(rec))
    day_dir = tmp_path / "data/day_int=25"
    day_dir.mkdir(parents=True)
    old = day_dir / "old.parquet"
    pd.DataFrame({"u": ["old"], "id": ["old_display"]}).to_parquet(old)

    def item(path):
        return dict(
            rfilename=f"data/day_int=25/{path.name}",
            size=60_000_000,
            blobId="git-" + path.name,
            lfs={"sha256": sha256_file(path)},
        )

    old_item = item(old)
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                repository=rep.REPO,
                selected_files=[
                    dict(
                        day=25,
                        path=old_item["rfilename"],
                        size=old_item["size"],
                        blob_id=old_item["blobId"],
                    )
                ],
            )
        )
    )
    inventory = [old_item]
    for shard, records in enumerate(
        (
            [("old", "1", "0", "g", 1, 1), ("fresh", "2", "0", "g", 1, 1)],
            [
                ("fresh", "3", "0", "g", 1, 1),
                ("other", "4", "1", "h", 0, 0),
                ("outside", "5", "out", "g", 1, 0),
            ],
        )
    ):
        file = tmp_path / f"new-{shard}.parquet"
        pd.DataFrame(records, columns=["u", "id", "b", "a", "z", "y"]).to_parquet(file)
        inventory.append(item(file))
        file.rename(cache / inventory[-1]["lfs"]["sha256"])
    monkeypatch.setattr(rep, "_inventory", lambda: dict(sha="a" * 40, siblings=inventory))
    runner = CliRunner()
    plan, output = tmp_path / "plan.json", tmp_path / "report.json"
    args = [
        "criteo-replication-plan",
        "--plan",
        str(plan),
        "--run",
        str(run),
        "--results",
        str(results),
        "--source-manifest",
        str(manifest),
    ]
    first = runner.invoke(app, args, catch_exceptions=False)
    assert first.exit_code == 0
    digest = first.stdout.strip()
    assert digest == hashlib.sha256(plan.read_bytes()).hexdigest()
    assert runner.invoke(app, args).exit_code != 0
    command = [
        "criteo-replication-run",
        "--plan",
        str(plan),
        "--cache",
        str(cache),
        "--output",
        str(output),
        "--plan-sha256",
    ]
    assert runner.invoke(app, [*command, "0" * 64]).exit_code != 0
    assert not output.exists()
    assert runner.invoke(app, [*command, digest], catch_exceptions=False).exit_code == 0
    report = json.loads(output.read_text())
    assert report["days"] == [
        dict(
            day=25,
            raw=5,
            overlapping_displays=1,
            excluded_seen_user_days=1,
            new_user_days=3,
            eligible=2,
            outside_cohort=1,
        )
    ]
    assert [r["n"] for r in report["contexts"]] == [1, 1] + [0] * 29
    assert report["eligible_user_days"] == 2
    assert report["empirical_attribute_verified"]
    assert report["optimal_ldp"]["lower"] <= 0 <= report["optimal_ldp"]["upper"]
    assert report["population_status"] == "assumptions_not_enabled"
    assert runner.invoke(app, [*command, digest]).exit_code != 0


def test_unobserved_group_and_direction():
    spec = dict(release_one=[0.1, 0.9], edges=[["a", "b"]])
    groups = {"a": (1, 1), "b": (0, 0)}
    assert rep._privacy(spec, groups) >= 2.19
    groups["b"] = (1, 1)
    assert rep._privacy(spec, groups) == 0
    spec["release_one"] = [0, 1]
    groups["b"] = (0, 0)
    assert rep._privacy(spec, groups) is None


def test_corrupt_cached_file(tmp_path):
    key = "0" * 64
    (tmp_path / key).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rep._fetch("a" * 40, {"sha256": key}, tmp_path)


def test_utility_win_requires_attribute_budget():
    import numpy as np

    spec = dict(
        index=0,
        high_tokens=[1],
        ctr=[0.01, 0.99],
        release_one=[0, 1],
        beta_R=0,
        budgets={"uniform_calibrated": 0},
        edges=[["g", "h"]],
        groups=["g", "h"],
    )
    plan = dict(contexts={"b": spec}, grid=rep.GRID, primary="uniform_calibrated")
    counts = {"b": np.array([[50, 0], [0, 50]])}
    grouped = {"b": {"g": (50, 50), "h": (50, 0)}}
    report = rep._summarize(plan, counts, grouped)
    assert report["ldp_minus_capt"]["lower"] > 0.68
    assert report["empirical_utility"] == "win"
    assert report["empirical_comparison"] == "attribute_budget_not_verified"
    grouped["b"] = {"g": (50, 25), "h": (50, 25)}
    assert rep._summarize(plan, counts, grouped)["empirical_comparison"] == "win"
