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
    ("arguments", "k", "l_count", "ctr_weight", "pretrain_epochs"),
    [(["--K", "32", "--L", "3"], 32, 3, 0.0, 2),
     (["--input-bits", "2", "--output-bits", "2", "--mass-ctr-weight", "0.7",
       "--mass-infonce-pretrain-epochs", "0"], 4, 4, 0.7, 0),
     (["--K", "8", "--L", "2", "--L-list", "2,3", "--phase", "full"], 8, (2, 3), 0.0, 2)],
)
def test_comparison_cli_smoke_with_shared_alphabet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    arguments: list[str], k: int, l_count: int | tuple[int, ...], ctr_weight: float, pretrain_epochs: int,
) -> None:
    # Only source provenance is substituted for a test checkout. Data loading,
    # model fitting, neural/attribute optimizers, native certificates, and rendering are real.
    monkeypatch.setattr("capt12.pipeline.require_clean_worktree", lambda: "test-source-sha")
    monkeypatch.setattr(
        "capt12.certification.artifact.require_clean_worktree", lambda: "test-source-sha"
    )
    monkeypatch.setattr(
        "capt12.comparison.calibration.calibrate_common_cover",
        lambda *args, **kwargs: pytest.fail("MaSS cover calibration must not run"),
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
        "mass_epochs": 2, "mass_batch_size": 32, "mass_token_dim": 8,
        "mass_hidden_dim": 16, "mass_attribute_pretrain_epochs": 2,
        "mass_infonce_pretrain_epochs": 2, "mass_infonce_batch_size": 16,
        "mass_infonce_embedding_dim": 4, "mass_eval_noise_samples": 2,
        "mass_pilot_m_list": [0.1], "mass_pilot_n_list": [0.0],
        "mass_pilot_privacy_weight_list": [1.0], "mass_pilot_utility_weight_list": [1.0],
        "utility_cluster_bootstrap_replicates": 2,
        "solver_verbose": False,
        "seeds": [0], "epsilon_list": [1.0], "require_mass_pilot_for_full": False,
        "mass_privacy_weight_list": [1.0], "mass_utility_weight_list": [1.0],
        "mass_temperature_list": [1.0], "mass_seed_list": [0],
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
    assert set(results["method"]) == {"capt", "optimal_ldp", "mass12_raw"}
    assert set(results["K"]) == {k}
    levels = {l_count} if isinstance(l_count, int) else set(l_count)
    assert set(results["L"]) == levels
    mass = results.loc[results["method"] == "mass12_raw"]
    assert len(mass) == len(levels)
    assert mass["model_path"].nunique() == 1
    assert mass["model_sha256"].nunique() == 1
    assert mass["empirical_log_loss"].nunique() == 1
    assert set(mass["mass_output_count"]) == {k}
    assert set(mass["mass_backend"]) == {"neural_token"}
    assert set(mass["mass_n"]) == {0.01}
    assert set(mass["mass_ctr_weight"]) == {ctr_weight}
    assert set(mass["mass_infonce_pretrain_epochs"]) == {pretrain_epochs}
    assert np.isfinite(mass["empirical_log_loss"]).all()
    assert np.isfinite(mass["mass_sensitive_mi_estimate_nats"]).all()
    assert not mass["formal_comparable"].any()
    assert not mass["certificate_valid"].any()
    assert mass["certificate_path"].isna().all()
    assert (mass["cover_lambda"] == 0).all()
    assert (mass["cover_row_tv"] == 0).all()
    assert mass["certified_upper_epsilon"].isna().all()
    assert not list((publication / "runs").glob("*/certificates/mass-*"))
    assert not list((publication / "runs").glob("*/mechanism/common_token_cover.*"))
    from capt12.comparison.mass_nn import MassNN
    model_paths = list((publication / "runs").glob("*/models/mass-nn-*.pt"))
    assert len(model_paths) == 1
    model = MassNN.load(model_paths[0])
    assert model.generator.output.out_features == k
    assert model.features.head[-1].out_features == 4
    assert "L" not in model.config
    assert "decoder" not in model.config
    assert model.provenance()["postprocessing"] == "none"
    assert model.provenance()["uses_capt_decoder"] is False
    histories = list((publication / "runs").glob("*/tables/mass-nn-*-history.csv"))
    assert len(histories) == 1
    history = pd.read_csv(histories[0])
    assert len(history) == 2
    for stage in ("pre_update", "post_update"):
        assert np.isfinite(history[f"{stage}_loss"]).all()
        assert np.isfinite(history[f"{stage}_infonce_loss"]).all()
        np.testing.assert_allclose(history[f"{stage}_infonce_loss"],
                                  (history[f"{stage}_infonce_output_to_source_loss"]
                                   + history[f"{stage}_infonce_source_to_output_loss"]) / 2)
        sensitive = history[f"{stage}_sensitive_mi_nats"].map(
            lambda value: max(next(iter(json.loads(value).values())) - 0.1, 0)
        )
        useful = history[f"{stage}_useful_mi_nats"].map(
            lambda value: max(0.01 - json.loads(value)["click"], 0)
        )
        np.testing.assert_allclose(
            history[f"{stage}_loss"],
            2 * sensitive**2 + 0.5 * sensitive + 1.5 * useful**2 + 0.75 * useful
            + history[f"{stage}_infonce_loss"] + ctr_weight * history[f"{stage}_distortion"],
            rtol=1e-6,
        )
    pretrain = pd.read_csv(next((publication / "runs").glob("*/tables/mass-nn-*-pretrain.csv")))
    assert len(pretrain.loc[pretrain["stage"] == "infonce"]) == pretrain_epochs
    assert len(pretrain.loc[pretrain["stage"] == "attributes"]) == 2
    assert np.isfinite(pretrain[["pre_update_loss", "post_update_loss"]]).all().all()
    contracts = json.loads((publication / "method_contracts.json").read_text())["methods"]
    mass_contract = next(item for item in contracts if item["method"] == "mass12_raw")
    assert not mass_contract["formal_comparable"]
    assert all(item["method"] != "mass12_calibrated" for item in contracts)
    assert mass_contract["output_bits"] == (k - 1).bit_length()
    assert (publication / "figures/main_privacy_utility_comparison.png").is_file()
