from __future__ import annotations

import pytest

from capt12.utils import artifacts


def test_require_clean_worktree_rejects_dirty_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifacts, "git_sha", lambda: "abc123")
    monkeypatch.setattr(
        artifacts,
        "git_worktree_changes",
        lambda: [" M src/capt12/pipeline.py"],
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        artifacts.require_clean_worktree()


def test_require_clean_worktree_returns_committed_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifacts, "git_sha", lambda: "abc123")
    monkeypatch.setattr(artifacts, "git_worktree_changes", lambda: [])
    assert artifacts.require_clean_worktree() == "abc123"
