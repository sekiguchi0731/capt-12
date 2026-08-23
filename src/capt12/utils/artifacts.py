from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

import capt12
from capt12.config import run_id


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_source_file(module: Any) -> Path | None:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.suffix == ".pyc":
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            return None
    return path.resolve()


def capt12_source_root() -> Path:
    """Return the Git root that actually contains the imported CAPT sources."""
    package_file = _module_source_file(capt12)
    if package_file is None:
        raise RuntimeError("cannot locate the imported capt12 package source")
    try:
        output = subprocess.check_output(
            ["git", "-C", str(package_file.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "certified runs require imported capt12 sources in a Git worktree; "
            "installed wheels require a build attestation"
        ) from error
    root = Path(output.strip()).resolve()
    if not package_file.is_relative_to(root):
        raise RuntimeError("imported capt12 package is outside its reported Git root")

    core_sources = {
        package_file,
        Path(__file__).resolve(),
        package_file.parent / "certification" / "artifact.py",
    }
    if any(not source.is_file() for source in core_sources):
        raise RuntimeError("required CAPT certificate sources are unavailable")
    loaded_sources = core_sources | {
        source
        for name, module in sys.modules.items()
        if (name == "capt12" or name.startswith("capt12."))
        and (source := _module_source_file(module)) is not None
    }
    if not loaded_sources or any(not source.is_relative_to(root) for source in loaded_sources):
        raise RuntimeError("loaded capt12 modules do not share one source Git worktree")
    relative_sources = [str(source.relative_to(root)) for source in sorted(loaded_sources)]
    try:
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", *relative_sources],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "loaded capt12 modules are not tracked files in the source Git worktree"
        ) from error
    return root


def git_sha(source_root: Path | None = None) -> str:
    try:
        root = source_root or capt12_source_root()
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        return "unknown"


def git_worktree_changes(source_root: Path | None = None) -> list[str]:
    """Return tracked and untracked non-ignored changes affecting provenance."""
    root = source_root or capt12_source_root()
    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot establish Git worktree provenance") from error
    return [line for line in output.splitlines() if line.strip()]


def require_clean_worktree() -> str:
    """Require a clean, committed source tree and return its Git SHA."""
    source_root = capt12_source_root()
    sha = git_sha(source_root)
    if sha == "unknown":
        raise RuntimeError("certified runs require a committed Git revision")
    changes = git_worktree_changes(source_root)
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


def write_sol_review_bundle(
    path: str | Path,
    *,
    bundle_name: str = "sol_review_bundle.zip",
) -> Path:
    """Create a deterministic, compact review bundle for a completed run."""
    run_path = Path(path)
    if not run_path.is_dir():
        raise ValueError(f"run directory does not exist: {run_path}")
    manifest_path = run_path / "sol_review_bundle_manifest.json"
    bundle_path = run_path / bundle_name
    temporary_path = run_path / f".{bundle_name}.tmp"
    excluded_names = {
        "progress.log",
        "progress.jsonl",
        bundle_name,
        temporary_path.name,
        manifest_path.name,
    }

    def included(candidate: Path) -> bool:
        relative = candidate.relative_to(run_path)
        return (
            candidate.is_file()
            and "checkpoints" not in relative.parts
            and candidate.name not in excluded_names
            and not candidate.name.endswith("_review_bundle.zip")
        )

    files = sorted(
        (candidate for candidate in run_path.rglob("*") if included(candidate)),
        key=lambda candidate: candidate.relative_to(run_path).as_posix(),
    )
    manifest = {
        "version": 1,
        "purpose": "ChatGPT Sol review of CAPT result, certificate, and deployment bundle",
        "run_id": run_path.name,
        "archive_root": run_path.name,
        "excluded": ["checkpoints/**", "progress.log", "progress.jsonl", "*_review_bundle.zip"],
        "files": [
            {
                "path": candidate.relative_to(run_path).as_posix(),
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
            }
            for candidate in files
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    files.append(manifest_path)
    fixed_timestamp = (2000, 1, 1, 0, 0, 0)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for candidate in files:
                relative = candidate.relative_to(run_path).as_posix()
                info = zipfile.ZipInfo(f"{run_path.name}/{relative}", fixed_timestamp)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with candidate.open("rb") as source, archive.open(info, "w") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        temporary_path.replace(bundle_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return bundle_path
