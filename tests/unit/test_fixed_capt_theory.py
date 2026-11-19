from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from capt12.comparison.optimal_ldp.retraining import FixedBlocks
from capt12.comparison.optimal_ldp.theory import (
    Bounds,
    empirical_decision,
    empirical_ldp_oracle,
    find_crossing,
    information_decomposition,
    sdpi_retention_check,
    weighted_bounds,
)
from capt12.comparison.optimal_ldp.utility import solve_spr
from capt12.experiments import fixed_capt_theory as runner


def _channel(rows, assignment=(0, 1)):
    return FixedBlocks.from_arrays(np.asarray(assignment), np.asarray(rows, dtype=float))


def test_information_decomposition_with_unobserved_token_and_block():
    counts = np.array([[8, 2], [2, 8], [30, 0], [0, 0]])
    capt = _channel([[0.75, 0.2, 0.05], [0.1, 0.8, 0.1], [0, 0, 1]], (0, 0, 1, 2))
    metric = information_decomposition(counts, capt).metrics()
    assert metric["h_y_given_z"] == pytest.approx(
        -0.4 * (0.2 * math.log(0.2) + 0.8 * math.log(0.8))
    )
    assert metric["h_y_given_c"] == pytest.approx(0.4 * math.log(2))
    assert 0 < metric["i_y_z_given_c"] < metric["i_y_z"]
    assert metric["i_y_c_given_release"] > 0
    assert abs(metric["decomposition_residual"]) < 2e-16
    assert metric["h_y_given_z"] + metric["i_y_z_given_c"] + metric[
        "i_y_c_given_release"
    ] == pytest.approx(metric["d_capt"], abs=2e-16)


@pytest.mark.parametrize("budget", [0, 0.2, 1, 3, math.inf])
def test_cached_grid_bounds_cover_exact_spr(budget):
    counts = np.array([[30, 1], [4, 12], [0, 2], [0, 0]])
    info = information_decomposition(counts, _channel([[0.8, 0.2], [0.2, 0.8]], (0, 0, 1, 0)))
    cert = empirical_ldp_oracle(info, [0, 0.15, 0.5, 1], exact_max_k=2)(budget)
    answer = solve_spr(counts, budget)
    assert cert.lower <= answer.upper
    assert cert.upper >= answer.lower


def test_known_binary_crossing_and_certificate_gate():
    counts = np.array([[10, 0], [0, 10]])
    info = information_decomposition(counts, _channel([[0.8, 0.2], [0.2, 0.8]]))
    oracle = empirical_ldp_oracle(info, [0, 0.5, 1], exact_max_k=64)

    def cross(beta, **kwargs):
        return find_crossing(
            oracle,
            info.entropy_release,
            beta,
            released_informative=info.released_informative,
            preserves_raw_information=info.preserves_raw_information,
            **kwargs,
        )

    interval = cross(1.0, tolerance=1e-6)
    assert interval.lower < math.log(4) < interval.upper
    assert interval.upper - interval.lower < 2e-6
    assert interval.win_interval_empty is False
    assert cross(1.5).win_interval_empty is True
    beyond = cross(0.5, initial=0.2, cap=0.3)
    assert (beyond.status, beyond.lower, beyond.upper) == ("finite_above_search_cap", 0.3, None)
    assert beyond.win_interval_empty is False  # the certificate point is also checked
    assert empirical_decision(info.entropy_release, oracle(1), privacy_eligible=False) == (
        "empirical_plugin_inconclusive"
    )


def test_finite_crossing_with_unresolved_upper_is_distinct_from_infinity():
    def oracle(eps):
        return Bounds(max(0, 1 - eps), 1)

    interval = find_crossing(
        oracle,
        Bounds(0.3, 0.3),
        0.8,
        released_informative=True,
        preserves_raw_information=False,
        initial=0.5,
        cap=2,
    )
    assert interval.status == "finite_unresolved_at_search_cap"
    assert interval.lower <= 0.7
    assert interval.upper is None
    assert interval.win_interval_empty is None


@pytest.mark.parametrize(
    ("counts", "rows", "expected"),
    [
        ([[10, 0], [0, 10]], [[1, 0], [0, 1]], "infinite"),
        ([[2, 3], [4, 6]], [[1, 0], [0, 1]], "zero"),
        ([[10, 0], [0, 10]], [[0.5, 0.5], [0.5, 0.5]], "zero"),
    ],
)
def test_exact_crossing_endpoints_and_uninformative_source(counts, rows, expected):
    info = information_decomposition(np.asarray(counts), _channel(rows))
    interval = find_crossing(
        empirical_ldp_oracle(info, [0, 1], 64),
        info.entropy_release,
        0.5,
        released_informative=info.released_informative,
        preserves_raw_information=info.preserves_raw_information,
    )
    assert interval.status == expected
    assert interval.lower == interval.upper == (math.inf if expected == "infinite" else 0)
    if counts[0] == [2, 3]:
        assert info.metrics()["information_retention"] is None
        assert sdpi_retention_check(info, 1, 0)["status"] == "sufficient_condition_inconclusive"


def test_outward_comparison_and_sdpi_boundary():
    midpoint = 0.3
    wide = Bounds(midpoint, math.nextafter(midpoint, math.inf))
    touching = Bounds(wide.upper, math.nextafter(wide.upper, math.inf))
    assert (
        empirical_decision(wide, touching, privacy_eligible=True) == "empirical_plugin_inconclusive"
    )
    assert touching.minus(wide).lower == 0
    assert (
        empirical_decision(touching, wide, privacy_eligible=True) == "empirical_plugin_inconclusive"
    )
    exact_weighted = weighted_bounds([Bounds(0.1, 0.1), wide], [Fraction(1, 3), Fraction(2, 3)])
    assert Fraction(exact_weighted.lower) <= (Fraction(0.1) + 2 * Fraction(0.3)) / 3
    assert Fraction(exact_weighted.upper) >= (Fraction(0.1) + 2 * Fraction(wide.upper)) / 3
    perfect = information_decomposition(np.array([[5, 0], [0, 5]]), _channel(np.eye(2)))
    assert sdpi_retention_check(perfect, 1, 1)["status"] == "sufficient_condition_met"
    assert (
        sdpi_retention_check(perfect, math.inf, 0)["status"] == "sufficient_condition_inconclusive"
    )


def test_zero_validation_skipped_and_excluded_from_validation_weights(tmp_path: Path, monkeypatch):
    common = {
        "settings": {"predictor_train_days": [19], "predictor_validation_days": [22]},
        "K": 2,
        "L": 2,
        "profile": "A",
        "context_column": "B",
        "certificate": "/saved/cert.json",
    }
    specs = {}
    for i, context in enumerate(["seen", "__MISSING__", "__OTHER__"]):
        specs[context] = {
            "index": i,
            "certificate": tmp_path / f"cert-{i}.json",
            "channel": _channel(np.eye(2)),
            "epsilon_cert": 0.5,
            "epsilon_design": 1.0,
            "epsilon_certificate_target": 1.0,
            "budgets": dict.fromkeys(runner.METHODS, 1.0),
            "saved": None,
        }
    sizes = {
        "seen": {"train": 5, "validation": 5, "test": 8},
        "__MISSING__": {"train": 0, "validation": 0, "test": 0},
        "__OTHER__": {"train": 0, "validation": 0, "test": 3},
    }
    joint = {"seen": np.array([[2, 0], [0, 3]])}
    cert = SimpleNamespace(
        resolved_config={"data_root": "local"}, split_identifiers={"D_test": [25]}
    )
    monkeypatch.setattr(runner, "_read_result_files", lambda _: {0: common})
    monkeypatch.setattr(runner, "_load_bundle", lambda *_: (cert, specs))
    monkeypatch.setattr(runner, "_collect_selected_counts", lambda *_, **__: (sizes, joint))
    out = tmp_path / "fresh"
    result = runner.run_fixed_capt_theory(tmp_path, out)
    seen, missing, other = result["contexts"]
    assert seen["validation_weight"] == 1 and seen["test_share_all_certified_contexts"] == 8 / 11
    assert missing["status"] == other["status"] == "skipped"
    assert missing["skip_reason"] == other["skip_reason"] == "no_validation_examples"
    assert other["test_n"] == 3
    assert result["aggregate"]["epsilon_1_empirical_decision"] == "empirical_plugin_win"
    assert result["aggregate"]["population_decision"] == "assumptions_not_enabled"
    assert json.loads((out / "theory.json").read_text()) == result
    with pytest.raises(FileExistsError):
        runner.run_fixed_capt_theory(tmp_path, out)
