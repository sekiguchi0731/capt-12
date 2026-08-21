from __future__ import annotations

import json
import time

import numpy as np

import capt12.mechanisms.lp as lp_module
from capt12.certification.robust import solve_robust_block_lp
from capt12.confidence.boxes import ConfidenceBox
from capt12.mechanisms.lp import solve_ldp_block_lp
from capt12.privacy.adjacency import AdjacentPair
from capt12.utils.progress import ProgressLogger


def test_progress_logger_flushes_readable_and_structured_logs(tmp_path, capsys) -> None:
    progress = ProgressLogger(tmp_path, name="test-run")
    progress.emit("stage_started", item=3, message="human readable")

    terminal = capsys.readouterr().out
    assert "[stage_started]" in terminal
    assert "item=3" in terminal
    assert "message=\"human readable\"" in terminal
    assert "[stage_started]" in (tmp_path / "progress.log").read_text()
    payload = json.loads((tmp_path / "progress.jsonl").read_text())
    assert payload["event"] == "stage_started"
    assert payload["item"] == 3
    assert payload["run"] == "test-run"
    assert payload["attempt"].endswith(f"-pid{payload['pid']}")


def test_lp_progress_reports_problem_solver_and_heartbeat(monkeypatch) -> None:
    real_linprog = lp_module.linprog

    def slow_linprog(*args, **kwargs):
        time.sleep(0.03)
        return real_linprog(*args, **kwargs)

    monkeypatch.setattr(lp_module, "linprog", slow_linprog)
    events: list[tuple[str, dict]] = []

    solution = solve_ldp_block_lp(
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        np.array([0.4, 0.6]),
        1.0,
        progress=lambda event, fields: events.append((event, dict(fields))),
        progress_label="test/ldp",
        heartbeat_seconds=0.005,
    )

    assert solution.channel is not None
    names = [event for event, _ in events]
    assert names[:3] == [
        "ldp_constraint_build_started",
        "ldp_constraint_build_finished",
        "lp_problem_build_started",
    ]
    assert "lp_problem_build_finished" in names
    assert "lp_solver_started" in names
    assert "lp_solver_heartbeat" in names
    assert names[-1] == "lp_solver_finished"
    finished = next(fields for event, fields in events if event == "lp_solver_finished")
    assert finished["label"] == "test/ldp"
    assert finished["variable_count"] == 4
    assert finished["total_constraint_count"] == 6
    assert finished["success"] is True


def test_robust_progress_reports_cutting_plane_and_verification() -> None:
    first = np.array([0.8, 0.2])
    second = np.array([0.2, 0.8])
    boxes = {
        "first": ConfidenceBox(first, first, first, "point", 1.0),
        "second": ConfidenceBox(second, second, second, "point", 1.0),
    }
    adjacency = [
        AdjacentPair("first", "second", 0.5),
        AdjacentPair("second", "first", 0.5),
    ]
    events: list[tuple[str, dict]] = []

    solution, verification = solve_robust_block_lp(
        1 - np.eye(2),
        np.array([0.5, 0.5]),
        boxes,
        adjacency,
        progress=lambda event, fields: events.append((event, dict(fields))),
        progress_label="test/robust",
    )

    assert solution.channel is not None
    assert verification.valid
    names = [event for event, _ in events]
    assert names[0] == "robust_solve_started"
    assert "cutting_plane_iteration_started" in names
    assert "support_oracle_scan_started" in names
    assert "support_oracle_scan_finished" in names
    assert names[-1] == "robust_verification_finished"
