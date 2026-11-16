from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from capt12.comparison.privacy_comparison import _mass_capt_joint_comparison
from capt12.comparison.tradeoff_curves import (
    horizontal_intersections,
    parameter_traces,
    same_eau_comparisons,
)
from capt12.experiments.prior_art_comparison import FactorizedChannels


def test_same_eau_retains_folds_and_all_intersections_without_extrapolation():
    rows = []
    for method, points in (
        ("capt", [(0.5, 0.55, 0.5), (1, 0.8, 0.3)]),
        ("mass12_raw", [(0, 0.6, 0.7), (0.1, 0.75, 0.8), (0.5, 0.65, 0.6)]),
    ):
        for i, (parameter, eau, loss) in enumerate(points):
            rows.append(
                dict(
                    cell_id="cell",
                    method=method,
                    model_id=f"{method}{i}",
                    target_epsilon=parameter,
                    mass_m=parameter,
                    mass_n=0,
                    mean_pair_eau=eau,
                    worst_pair_eau=eau,
                    empirical_log_loss=loss,
                )
            )
    traces = parameter_traces(pd.DataFrame(rows))
    mass = traces[traces.method.eq("mass12_raw")]
    assert mass.mean_pair_eau.tolist() == [0.6, 0.75, 0.65]  # parameter order, not EAU order
    hits = horizontal_intersections(mass, "mean_pair_eau", 0.7)
    assert len(hits) == 2
    np.testing.assert_allclose([h["logloss"] for h in hits], [0.7666666667, 0.7])
    matched = same_eau_comparisons(traces)
    assert matched.eau.min() == 0.6 and matched.eau.max() == 0.75
    at_knot = matched[(matched.eau == 0.65) & matched.metric.eq("mean_pair_eau")]
    assert len(at_knot) >= 2
    assert set(at_knot.mass_left_model_id) == {"mass12_raw0", "mass12_raw1"}
    assert not same_eau_comparisons(
        traces.assign(
            mean_pair_eau=np.where(traces.method.eq("capt"), 0.9, 0.5),
            worst_pair_eau=np.where(traces.method.eq("capt"), 0.9, 0.5),
        )
    ).size


def test_vertical_segment_and_fixed_other_mass_controls():
    frame = pd.DataFrame(
        [
            dict(
                cell_id="c",
                method="mass12_raw",
                model_id=str(i),
                mass_m=i,
                mass_n=n,
                mean_pair_eau=0.5,
                worst_pair_eau=0.5,
                empirical_log_loss=loss,
            )
            for i, n, loss in [(0, 0, 0.7), (1, 0, 0.6), (2, 0.1, 0.4)]
        ]
    )
    traces = parameter_traces(frame)
    assert traces.trace_id.nunique() == 2
    selected = traces[traces.mass_n.eq(0)]
    assert {h["logloss"] for h in horizontal_intersections(selected, "mean_pair_eau", 0.5)} == {
        0.6,
        0.7,
    }
    with pytest.raises(ValueError, match="axis"):
        parameter_traces(frame, "bogus")


def test_lower_bounds_alone_do_not_order_privacy():
    summary = pd.DataFrame(
        [
            dict(
                cell_id="c",
                method="capt",
                model_id="c",
                certificate_valid=True,
                certified_upper_epsilon=1,
                target_epsilon=1,
                empirical_log_loss=0.3,
            ),
            dict(
                cell_id="c", method="mass12_raw", model_id="m", label="mass", empirical_log_loss=0.4
            ),
        ]
    )
    event = dict(
        model_id="m",
        epsilon_lower=0.9,
        pair_id="pair",
        event_type="tail",
        score=2,
        positive_trials=1000,
        negative_trials=1000,
        positive_successes=20,
        negative_successes=0,
    )
    result = _mass_capt_joint_comparison(pd.DataFrame([event]), summary)
    assert not result.joint_separation_observed.any()
    result = _mass_capt_joint_comparison(pd.DataFrame([{**event, "epsilon_lower": 1.2}]), summary)
    assert result.joint_separation_observed.all()
    summary.loc[summary.method.eq("capt"), "empirical_log_loss"] = 0.5
    assert not _mass_capt_joint_comparison(
        pd.DataFrame([{**event, "epsilon_lower": 1.2}]), summary
    ).joint_separation_observed.any()


def test_frozen_sweep_logloss_is_expected_loss_of_released_tokens():
    path = Path(__file__).resolve().parents[2] / "tools/sweep_capt_frozen.py"
    spec = importlib.util.spec_from_file_location("frozen_sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probabilities = np.array([0.1, 0.8])
    channels = FactorizedChannels({"c": np.array([[0.25, 0.75], [0.6, 0.4]])}, np.eye(2))
    data = SimpleNamespace(
        test_context=np.array(["c"] * 3),
        test_tokens=np.array([0, 1, 0]),
        test_labels=np.array([1, 0, 0]),
        service_probabilities={"c": probabilities},
    )
    expected = (
        sum(
            channels["c"][z] @ (-np.log(probabilities) if y else -np.log1p(-probabilities))
            for z, y in zip(data.test_tokens, data.test_labels, strict=True)
        )
        / 3
    )
    assert module.exact_logloss(channels, data) == pytest.approx(expected)


def test_duplicate_references_from_parallel_jobs_keep_provenance_and_require_equal_measurements():
    row = dict(
        cell_id="c",
        method="capt",
        model_id="old",
        target_epsilon=1,
        channel_sha256="same",
        mean_pair_eau=0.5,
        worst_pair_eau=0.6,
        empirical_log_loss=0.3,
    )
    frame = pd.DataFrame([row, {**row, "model_id": "new"}])
    traces = parameter_traces(frame)
    assert len(traces) == 1
    assert set(__import__("json").loads(traces.replicated_model_ids.iloc[0])) == {"old", "new"}
    for key, value in (("channel_sha256", "other"), ("mean_pair_eau", 0.51)):
        with pytest.raises(ValueError, match="different mechanisms or measurements"):
            parameter_traces(pd.DataFrame([row, {**row, "model_id": "new", key: value}]))


def test_parallel_result_collection_keeps_only_mass_without_modifying_original(tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/launch_b200_replicates.py"
    spec = importlib.util.spec_from_file_location("eau_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "results.csv"
    pd.DataFrame(
        [
            dict(method="capt", model_sha256="c", empirical_log_loss=0.4),
            dict(method="mass12_raw", model_sha256="m", empirical_log_loss=0.5),
        ]
    ).to_csv(source, index=False)
    before = source.read_bytes()
    collected = pd.read_csv(module.collect_mass_results(source))
    assert collected.method.tolist() == ["mass12_raw"]
    assert collected.model_sha256.tolist() == ["m"]
    assert before == source.read_bytes()
