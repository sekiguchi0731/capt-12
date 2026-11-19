from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from capt12.certification.artifact import hash_array, make_certificate
from capt12.certification.robust import repair_robust_channel_uniform
from capt12.cli import app
from capt12.comparison.optimal_ldp.retraining import FixedBlocks
from capt12.comparison.optimal_ldp.utility import bayes_loss_bounds, grid_bounds, solve_spr
from capt12.confidence.boxes import cp_box
from capt12.data.preprocessing import FrozenCategoryMapper
from capt12.encoders.base import PrecomputedEncoder
from capt12.experiments import fixed_capt_ctr_comparison as experiment
from capt12.mechanisms.lp import SolverInfo
from capt12.models.reference import TOKEN_REFERENCE_FEATURE_SCHEMA, ReferenceModel
from capt12.privacy.adjacency import Group, build_adjacency
from capt12.utils.artifacts import sha256_file


def test_k64_fixed_capt_ctr_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    k, blocks, seed = 64, 2, 12064
    rng = np.random.default_rng(seed)
    splits = {
        "D_model": [1],
        "D_design": [7],
        "D_cert": [13],
        "D_attack_train": [19, 22],
        "D_test": [25],
    }
    source = tmp_path / "data"
    ctr = np.r_[np.linspace(0.02, 0.15, 32), np.linspace(0.37, 0.62, 32)]
    assignment = np.arange(k) // 32
    decoder = (assignment[None, :] == np.arange(2)[:, None]).astype(float) / 32
    tokens = np.arange(k)
    slot = (tokens // 16) % 2
    laws = np.stack(
        [
            np.where(assignment == group, 0.6, 0.4) * np.where(slot == group, 0.8, 0.2) / 16
            for group in (0, 1)
        ]
    )
    population = laws.mean(axis=0)
    true_weights = population[:, None] * np.column_stack((1 - ctr, ctr))
    frames = {}
    for day, n in [(1, 3500), (13, 6000), (19, 5000), (22, 5000), (25, 5000)]:
        a = rng.integers(0, 2, n)
        z = np.empty(n, dtype=int)
        for group in (0, 1):
            z[a == group] = rng.choice(k, (a == group).sum(), p=laws[group])
        frame = pd.DataFrame(
            {
                "id": [f"{day}:{i}" for i in range(n)],
                "user_id": [f"u{i}" for i in range(n)],
                "A": a,
                "B": "ctx",
                "z": z,
                "y": rng.binomial(1, ctr[z]),
            }
        )
        path = source / f"day_int={day}"
        path.mkdir(parents=True)
        frame.to_parquet(path / "sample.parquet", index=False)
        frames[day] = frame

    models = tmp_path / "models"
    models.mkdir()
    mapper = FrozenCategoryMapper(closed_domain_columns=frozenset(["A"])).fit(frames[1], ["A", "B"])
    joblib.dump(mapper, models / "category_mapper.joblib")
    joblib.dump(PrecomputedEncoder(k, "z"), models / "encoder.joblib")
    reference = ReferenceModel("synthetic").fit(
        mapper.transform(frames[1]).rename(columns={"z": "__token__"}),
        "y",
        ["__token__", "B"],
        categorical_columns=["__token__"],
    )
    reference.save(models / "reference.joblib")

    groups = [Group("A", (str(group),), "ctx") for group in (0, 1)]
    edges = build_adjacency(groups, epsilon=0.75)
    alpha_cert = 0.01
    group_counts = {
        group.key(): np.bincount(assignment[frames[13].loc[frames[13]["A"] == i, "z"]], minlength=2)
        for i, group in enumerate(groups)
    }
    boxes = {
        key: cp_box(value, alpha=alpha_cert, group_count=2, comparisons=2)
        for key, value in group_counts.items()
    }
    repair = repair_robust_channel_uniform(np.array([[0.9, 0.1], [0.1, 0.9]]), boxes, edges)
    channel = repair.channel
    hashes = {
        "encoder": sha256_file(models / "encoder.joblib"),
        "mapper": sha256_file(models / "category_mapper.joblib"),
        "model": sha256_file(models / "reference.joblib"),
    }
    manifest_path = tmp_path / "mechanism" / "context_channel_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "version": 6,
                "profile": "A",
                "public_context_column": "B",
                "protected_value_used_online": False,
                "channel_selector_inputs": ["Z", "profile", "B"],
                "runtime_mapper_hash": hashes["mapper"],
                "reference_model_hash": hashes["model"],
                "reference_feature_schema": TOKEN_REFERENCE_FEATURE_SCHEMA,
                "reference_feature_columns": ["__token__", "B"],
                "reference_categorical_columns": ["__token__"],
                "sensitive_domain_policy": "closed_domain",
                "sensitive_coarsening": {"missing": "reject", "unseen": "reject"},
                "released_channel_policy": "uniform_full_support_postsolve_repair",
                "designs": {
                    "pilot": {
                        "L": blocks,
                        "table_entries": blocks**2,
                        "contexts": {"ctx": hash_array(channel)},
                        "assignment_hash": hash_array(assignment),
                        "decoder_hash": hash_array(decoder),
                        "context_repairs": {
                            "ctx": {
                                "method": "uniform_full_support_mixing",
                                "mixing_weight": repair.mixing_weight,
                            }
                        },
                    }
                },
            }
        )
    )
    hashes["context_channel_manifest"] = sha256_file(manifest_path)

    monkeypatch.setattr(
        "capt12.certification.artifact.require_clean_worktree", lambda: "fixture-sha"
    )
    certificate = make_certificate(
        config={
            "dataset": "criteo",
            "profiles": ["A"],
            "context_cols": ["B"],
            "context_design": "pilot",
            "public_context_value": "ctx",
            "public_context_policy": "stratified",
            "sensitive_fallback_policy": "closed_domain",
            "sampling_assumption": "user_day_iid",
            "channel_selector_inputs": ["Z", "profile", "B"],
            "confidence": "cp_box",
            "alpha_cert": alpha_cert,
            "epsilon": 0.75,
            "phi_source_cols": ["z"],
            "label_col": "y",
            "splits": splits,
            "data_root": str(source),
        },
        channel=channel,
        boxes=boxes,
        adjacency=edges,
        groups=groups,
        verification=repair.post_verification,
        solver=SolverInfo("optimal", 0, 0.0, 1),
        component_hashes=hashes,
        split_identifiers=splits,
        histogram_counts=group_counts,
        dp_parameters={
            "epsilon": None,
            "delta": None,
            "contribution_policy": "one-display-per-uuid-day",
        },
        assignment=assignment,
        decoder=decoder,
        coverage={
            "public_context_value": "ctx",
            "expected_group_count": 2,
            "observed_group_count": 2,
            "missing_group_count": 0,
            "post_solve_repair": {
                "method": "uniform_full_support_mixing",
                "mixing_weight": repair.mixing_weight,
                "post_repair_zero_denominator_positive_numerator_count": 0,
                "conservative_decimal_verification": asdict(repair.conservative_verification),
            },
        },
    )
    certificate_path = tmp_path / "certificate.json"
    certificate.write(certificate_path)
    initial_hash = sha256_file(certificate_path)
    requested_days = []
    fitted_models = []
    load = experiment.load_parquet_sample
    prepare = experiment.prepare_comparison

    def record_load(**kwargs):
        requested_days.append(tuple(kwargs["days"]))
        assert (13 in kwargs["days"]) == ("y" not in kwargs["columns"])
        return load(**kwargs)

    def record_prepare(*args, **kwargs):
        assert requested_days == [(13,), (19,), (22,)]
        fitted_models.append(prepare(*args, **kwargs))
        return fitted_models[-1]

    monkeypatch.setattr(experiment, "load_parquet_sample", record_load)
    monkeypatch.setattr(experiment, "prepare_comparison", record_prepare)
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        "predictor_train_days: [19]\npredictor_validation_days: [22]\n"
        "raw_shift_tv: 0.0\niid_utility_target: true\n"
    )
    output = tmp_path / "comparison.json"
    cli = CliRunner().invoke(
        app,
        [
            "fixed-capt-ctr-comparison",
            "--certificate",
            str(certificate_path),
            "--settings",
            str(settings),
            "--output",
            str(output),
        ],
        catch_exceptions=False,
    )
    assert cli.exit_code == 0, cli.output
    assert requested_days == [(13,), (19,), (22,), (25,)]
    assert sha256_file(certificate_path) == initial_hash

    report = json.loads(output.read_text())
    calibration = report["calibration"]
    assert calibration["beta_R"] < calibration["tv_budget"] <= calibration["uniform_budget"]
    assert calibration["uniform_method"].endswith("outer_bound_for_intersection")
    assert report["capt_fixed_reference_test_log_loss"] > 0
    # The laws are known for this fixture, so we can evaluate population Bayes
    # values independently of the split-derived estimates and confidence bounds.
    capt = FixedBlocks.from_arrays(assignment, channel)
    rounded_truth = np.rint(true_weights * 10**12).astype(np.int64)
    true_capt = bayes_loss_bounds(capt.exact_output_joint(rounded_truth))[0]
    assert report["capt_test_loss_ucb"] >= true_capt
    assert fitted_models[0].capt.loss(rounded_truth) >= true_capt
    truth = {}
    for name, summary in report["comparisons"].items():
        true_v = solve_spr(true_weights, summary["lambda"])
        lo, hi = summary["population_v_z_bounds"]
        assert lo <= true_v.lower <= true_v.upper <= hi
        assert summary["ldp_output_count"] <= k
        assert fitted_models[0].ldp[name].loss(rounded_truth) >= true_v.lower - 1e-12
        truth[name] = {"v_z": true_v.estimate, "gap": true_v.estimate - true_capt}
    print(
        json.dumps(
            {
                "fixture_seed": seed,
                "beta_R": calibration["beta_R"],
                "tv_upper": calibration["delta_tv_upper"],
                "j_upper": calibration["j_beta_upper"],
                "capt_population_bayes": true_capt,
                "capt_test": report["capt_test_log_loss"],
                "capt_fixed_reference_test": report["capt_fixed_reference_test_log_loss"],
                "capt_test_ucb": report["capt_test_loss_ucb"],
                "seconds": report["elapsed_seconds"],
                "known_fixture_optimum": truth,
                "comparisons": report["comparisons"],
            },
            indent=2,
        )
    )


def test_k4096_grid_kernel_only():
    rng = np.random.default_rng(4096)
    weights = rng.integers(1, 40, size=(4096, 2))
    result = grid_bounds(weights, 1.0, np.linspace(0, 1, 17))
    assert 0 < result.lower <= result.upper <= math.log(2)
    assert result.channel.assignment.size == 4096
    assert result.channel.outputs <= 16
    np.testing.assert_allclose(
        result.channel.output_counts(weights).sum(axis=0), weights.sum(axis=0)
    )
