from __future__ import annotations

import json
import os
import platform
import resource
import sys
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def process_memory_bytes() -> dict[str, int | None]:
    """Return best-effort current and peak resident memory without extra dependencies."""
    current: int | None = None
    peak: int | None = None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if peak is None:
        # Linux reports KiB; macOS reports bytes.
        peak = int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    return {"rss_bytes": current, "peak_rss_bytes": peak}


def _display_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), separators=(",", ":"), default=str)
    if isinstance(value, Mapping):
        return json.dumps(dict(value), separators=(",", ":"), sort_keys=True, default=str)
    text = str(value)
    return json.dumps(text) if any(character.isspace() for character in text) else text


class ProgressLogger:
    """Flush progress to stdout, a readable log, and structured JSONL."""

    def __init__(self, run_path: Path, *, name: str) -> None:
        self.run_path = Path(run_path)
        self.name = name
        self.attempt = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-pid{os.getpid()}"
        self.started = time.monotonic()
        self.log_path = self.run_path / "progress.log"
        self.jsonl_path = self.run_path / "progress.jsonl"
        self._lock = threading.Lock()

    def __call__(self, event: str, fields: Mapping[str, Any]) -> None:
        self.emit(event, **dict(fields))

    def emit(self, event: str, **fields: Any) -> None:
        now = datetime.now(UTC)
        payload = {
            "timestamp_utc": now.isoformat(),
            "elapsed_seconds": time.monotonic() - self.started,
            "pid": os.getpid(),
            "run": self.name,
            "attempt": self.attempt,
            "event": event,
            **fields,
        }
        prefix = (
            f"[{now.isoformat()}] [+{payload['elapsed_seconds']:.1f}s] "
            f"[pid={payload['pid']}] [attempt={self.attempt}] [{event}]"
        )
        suffix = " ".join(f"{key}={_display_value(value)}" for key, value in fields.items())
        line = f"{prefix} {suffix}".rstrip()
        with self._lock:
            print(line, flush=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def emit_environment(self) -> None:
        self.emit(
            "runtime_environment",
            host=platform.node(),
            platform=platform.platform(),
            python=sys.version.split()[0],
            cpu_count=os.cpu_count(),
            **process_memory_bytes(),
        )
