from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

import capt12
from capt12.utils import artifacts


def test_require_clean_worktree_rejects_dirty_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifacts, "capt12_source_root", lambda: Path("/source"))
    monkeypatch.setattr(artifacts, "git_sha", lambda source_root=None: "abc123")
    monkeypatch.setattr(
        artifacts,
        "git_worktree_changes",
        lambda source_root=None: [" M src/capt12/pipeline.py"],
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        artifacts.require_clean_worktree()


def test_require_clean_worktree_returns_committed_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifacts, "capt12_source_root", lambda: Path("/source"))
    monkeypatch.setattr(artifacts, "git_sha", lambda source_root=None: "abc123")
    monkeypatch.setattr(
        artifacts, "git_worktree_changes", lambda source_root=None: []
    )
    assert artifacts.require_clean_worktree() == "abc123"


def test_clean_unrelated_cwd_cannot_replace_capt12_source_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sha = artifacts.git_sha()
    assert source_sha != "unknown"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    subprocess.run(["git", "init", "-q", str(unrelated)], check=True)
    (unrelated / "README.md").write_text("unrelated repository\n")
    subprocess.run(["git", "-C", str(unrelated), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(unrelated),
            "-c",
            "user.name=CAPT test",
            "-c",
            "user.email=capt-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "Initialize unrelated repository",
        ],
        check=True,
    )
    unrelated_sha = subprocess.check_output(
        ["git", "-C", str(unrelated), "rev-parse", "HEAD"], text=True
    ).strip()
    assert unrelated_sha != source_sha
    monkeypatch.chdir(unrelated)
    checked_roots: list[Path | None] = []

    def clean_status(source_root: Path | None = None) -> list[str]:
        checked_roots.append(source_root)
        return []

    monkeypatch.setattr(artifacts, "git_worktree_changes", clean_status)

    assert artifacts.git_sha() == source_sha
    assert artifacts.require_clean_worktree() == source_sha
    assert checked_roots == [artifacts.capt12_source_root()]


def test_source_root_rejects_package_without_git_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "site-packages" / "capt12"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("# installed wheel fixture\n")
    monkeypatch.setattr(capt12, "__file__", str(package_file))

    with pytest.raises(RuntimeError, match="build attestation"):
        artifacts.capt12_source_root()


def test_sol_review_bundle_is_deterministic_and_excludes_runtime_noise(tmp_path) -> None:
    run_path = tmp_path / "run-123"
    (run_path / "tables").mkdir(parents=True)
    (run_path / "checkpoints").mkdir()
    (run_path / "manifest.json").write_text('{"status":"complete"}\n')
    (run_path / "tables" / "results.csv").write_text("metric,value\nU,1\n")
    (run_path / "certificate-context-00.json").write_text('{"valid":true}\n')
    (run_path / "progress.log").write_text("verbose runtime log\n")
    (run_path / "progress.jsonl").write_text("{}\n")
    (run_path / "checkpoints" / "cuts.npz").write_bytes(b"checkpoint")

    bundle = artifacts.write_sol_review_bundle(run_path)
    first_hash = artifacts.sha256_file(bundle)
    bundle = artifacts.write_sol_review_bundle(run_path)
    assert artifacts.sha256_file(bundle) == first_hash

    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    prefix = "run-123/"
    assert prefix + "manifest.json" in names
    assert prefix + "tables/results.csv" in names
    assert prefix + "certificate-context-00.json" in names
    assert prefix + "sol_review_bundle_manifest.json" in names
    assert all("checkpoints" not in name for name in names)
    assert all("progress." not in name for name in names)
    bundle_manifest = json.loads(
        (run_path / "sol_review_bundle_manifest.json").read_text()
    )
    assert {item["path"] for item in bundle_manifest["files"]} >= {
        "manifest.json",
        "tables/results.csv",
        "certificate-context-00.json",
    }
