from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from capt12.config import run_id


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_worktree_changes() -> list[str]:
    """Return tracked and untracked non-ignored changes affecting provenance."""
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot establish Git worktree provenance") from error
    return [line for line in output.splitlines() if line.strip()]


def require_clean_worktree() -> str:
    """Require a clean, committed source tree and return its Git SHA."""
    sha = git_sha()
    if sha == "unknown":
        raise RuntimeError("certified runs require a committed Git revision")
    changes = git_worktree_changes()
    if changes:
        preview = ", ".join(changes[:5])
        raise RuntimeError(
            "certified runs require a clean Git worktree; commit or discard source "
            f"changes first ({preview})"
        )
    return sha


def environment() -> dict[str, Any]:
    wanted = ["numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "typer"]
    versions = {}
    for name in wanted:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": git_sha(),
        "dependencies": versions,
    }


def prepare_run(config: dict[str, Any]) -> Path:
    root = Path(config.get("output_dir", "outputs/runs"))
    path = root / run_id(config)
    path.mkdir(parents=True, exist_ok=True)
    (path / "models").mkdir(exist_ok=True)
    (path / "mechanism").mkdir(exist_ok=True)
    (path / "tables").mkdir(exist_ok=True)
    (path / "figures").mkdir(exist_ok=True)
    with (path / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    with (path / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment(), handle, indent=2, sort_keys=True)
    manifest = {"run_id": path.name, "status": "started", "git_sha": git_sha()}
    with (path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def finish_run(path: Path, extra: dict[str, Any] | None = None) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(extra or {})
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
