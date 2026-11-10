from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from capt12.cli import app


@pytest.mark.parametrize(
    ("arguments", "k", "l_count", "ctr_weight"),
    [(["--K", "32", "--L", "3"], 32, 3, 0.0),
     (["--input-bits", "2", "--output-bits", "2", "--mass-ctr-weight", "0.7"], 4, 4, 0.7)],
)
def test_comparison_cli_smoke_with_shared_alphabet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    arguments: list[str], k: int, l_count: int, ctr_weight: float,
) -> None:
    # Only source provenance is substituted for a test checkout. Data loading,
    # model fitting, both optimizers, calibration, and certificates are real.
    monkeypatch.setattr("capt12.pipeline.require_clean_worktree", lambda: "test-source-sha")
    monkeypatch.setattr(
        "capt12.certification.artifact.require_clean_worktree", lambda: "test-source-sha"
    )
    data_root = tmp_path / "data"
    rng = np.random.default_rng(19)
    for day in range(1, 6):
        tokens = np.tile(np.arange(k), 12)
        frame = pd.DataFrame({
            "id": [f"{day}-{i}" for i in range(len(tokens))],
            "user_id": [f"u{i}" for i in range(len(tokens))],
            "token": tokens,
            "is_clicked": (rng.random(len(tokens)) < (0.1 + 0.8 * tokens / (k - 1))).astype(int),
            "features_kv_bits_constrained_2": np.repeat(["a", "b"], len(tokens) // 2),
            "features_ctx_not_constrained_0": "public",
        })
        partition = data_root / f"day_int={day}"
        partition.mkdir(parents=True)
        frame.to_parquet(partition / "part.parquet", index=False)
    publication = tmp_path / "comparison"
    config = {
        "base_config": str(Path("configs/criteo_prior_art_privacy_matched.yaml").resolve()),
        "data_root": str(data_root),
        "comparison_output_dir": str(publication),
        "phi": "hash", "phi_source_cols": ["token"],
        "sensitive_cols": ["features_kv_bits_constrained_2"],
        "splits": dict(zip(
            ["D_model", "D_design", "D_cert", "D_attack_train", "D_test"],
            [[day] for day in range(1, 6)], strict=True,
        )),
        "cost_list": ["teacher_kl"],
        "mass_epochs": 2, "mass_pilot_m_list": [0.1], "mass_pilot_n_list": [0.0],
        "mass_pilot_privacy_weight_list": [1.0], "mass_pilot_utility_weight_list": [1.0],
        "utility_cluster_bootstrap_replicates": 2,
        "solver_verbose": False,
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = CliRunner().invoke(
        app, ["prior-art-comparison", "--config", str(config_path), *arguments,
              "--mass-n-list", "0.01", "--mass-m-list", "0.1",
              "--mass-privacy-quadratic-weight", "2.0", "--mass-privacy-linear-weight", "0.5",
              "--mass-useful-quadratic-weight", "1.5", "--mass-useful-linear-weight", "0.75"],
    )
    assert result.exit_code == 0, f"{result.output[-5000:]}\n{result.exception!r}"
    results = pd.read_csv(publication / "results.csv")
    assert set(results["method"]) == {"capt", "optimal_ldp", "mass12_raw", "mass12_calibrated"}
    assert set(results["K"]) == {k}
    assert set(results["L"]) == {l_count}
    mass_rows = results.loc[results["method"].str.startswith("mass12")]
    assert set(mass_rows["mass_n"]) == {0.01}
    assert set(mass_rows["mass_infonce_direction"]) == {"symmetric"}
    assert set(mass_rows["mass_ctr_weight"]) == {ctr_weight}
    assert np.isfinite(mass_rows["empirical_log_loss"]).all()
    assert (mass_rows["empirical_log_loss"] > 0).all()
    assert results.loc[results["method"] == "mass12_calibrated", "certificate_valid"].all()
    model_paths = list((publication / "runs").glob("*/models/mass-*.npz"))
    assert len(model_paths) == 1
    with np.load(model_paths[0]) as model:
        assert model["logits"].shape[1:] == (k, l_count)
        assert model["decoder"].shape == (l_count, k)
        assert model["infonce_embeddings"].shape == (k, 32)
        assert float(model["infonce_weight"]) == 1.0
        assert float(model["ctr_weight"]) == ctr_weight
        assert np.isfinite(model["infonce_embeddings"]).all()
        assert str(model["infonce_direction"]) == "symmetric"
        assert float(model["privacy_quadratic_weight"]) == 2.0
        assert float(model["privacy_linear_weight"]) == 0.5
        if l_count == k:
            np.testing.assert_array_equal(model["decoder"], np.eye(k))
    native_paths = list((publication / "runs").glob("*/native_runs/*/mechanism/context_channels-*.npz"))
    assert len(native_paths) == 1
    with np.load(native_paths[0]) as native:
        assert native["decoder"].shape == (l_count, k)
    histories = list((publication / "runs").glob("*/tables/mass-*-history.csv"))
    assert len(histories) == 1
    history = pd.read_csv(histories[0])
    assert set(history["mass_ctr_weight"]) == {ctr_weight}
    assert (history["pre_update_infonce_loss"] > 0).all()
    assert (history["post_update_infonce_loss"] > 0).all()
    for stage in ("pre_update", "post_update"):
        np.testing.assert_allclose(
            history[f"{stage}_infonce_loss"],
            (history[f"{stage}_infonce_output_to_source_loss"]
             + history[f"{stage}_infonce_source_to_output_loss"]) / 2,
        )
        sensitive = history[f"{stage}_sensitive_mi_nats"].map(
            lambda value: max(next(iter(json.loads(value).values())) - 0.1, 0)
        )
        useful = history[f"{stage}_useful_mi_nats"].map(
            lambda value: max(0.01 - json.loads(value)["click"], 0)
        )
        np.testing.assert_allclose(
            history[f"{stage}_loss"],
            ctr_weight * history[f"{stage}_distortion"]
            + 2 * sensitive**2 + 0.5 * sensitive
            + 1.5 * useful**2 + 0.75 * useful
            + history[f"{stage}_infonce_loss"],
        )
    for certificate in mass_rows["certificate_path"]:
        if isinstance(certificate, str) and certificate:
            assert json.loads(Path(certificate).read_text())["metadata"]["ctr_weight"] == ctr_weight
    summary = json.loads((publication / "pilot_summary.json").read_text())
    assert summary["mass_ctr_weight"] == ctr_weight
    contracts = json.loads((publication / "method_contracts.json").read_text())["methods"]
    assert all(item["output_bits"] == (k - 1).bit_length()
               for item in contracts if item["single_output"])
