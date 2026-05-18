"""End-to-end smoke test: ``python -m app.main --mode PAPER --dry-run``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dry_run_smoke(tmp_path) -> None:
    """The dry-run boot must exit 0 and emit a structured log line."""
    env = {
        "MODE": "PAPER",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'smoke.db'}",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "PYTHONPATH": str(REPO_ROOT),
        "LOG_JSON": "true",
    }
    result = subprocess.run(
        [sys.executable, "-m", "app.main", "--mode", "PAPER", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "boot" in result.stdout
    assert "dry_run.complete" in result.stdout


def test_dry_run_refuses_live_without_adapter(tmp_path) -> None:
    env = {
        "MODE": "LIVE",
        "LIVE_ADAPTER_CONFIRMED": "false",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'smoke.db'}",
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "PYTHONPATH": str(REPO_ROOT),
    }
    result = subprocess.run(
        [sys.executable, "-m", "app.main", "--mode", "LIVE", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 2
    assert "LIVE mode is locked" in result.stderr
