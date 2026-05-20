"""Read/write markdown memory files used by autonomous workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from config.settings import Settings

_RESEARCH_HEADER = re.compile(
    r"^##\s+(\d{4}-\d{2}-\d{2})\s+",
    re.MULTILINE,
)

_DEFAULT_STRATEGY = """# Trading Strategy

This document describes the bot's active playbook. Workflows read it at
pre-market; update it manually when you change rules or filters.

## Universe
- Trade only symbols listed in ``ENABLED_SYMBOLS``.
- Respect per-symbol and portfolio trade caps from settings.

## Setup bias
- Prefer VWAP + EMA pullback continuation in the direction of the session trend.
- Skip counter-trend entries during the first 15 minutes unless ORB confirms.

## Risk
- One position at a time unless multi-symbol caps allow more.
- Default decision when uncertain: **hold**.
"""

_DEFAULT_RESEARCH_LOG = """# Research Log

Dated pre-market research entries are appended below. Market-open refuses to
trade without a section for today's date.
"""

_DEFAULT_TRADE_LOG = """# Trade Log

End-of-day snapshots and workflow trade notes are appended below.
"""

_DEFAULT_WEEKLY_REVIEW = """# Weekly Review

Friday (or forced) weekly summaries are appended below.
"""


@dataclass(frozen=True)
class MemoryPaths:
    root: Path
    strategy: Path
    research_log: Path
    trade_log: Path
    weekly_review: Path


def memory_paths(settings: Settings) -> MemoryPaths:
    root = Path(settings.WORKFLOW_MEMORY_DIR)
    return MemoryPaths(
        root=root,
        strategy=root / "TRADING-STRATEGY.md",
        research_log=root / "RESEARCH-LOG.md",
        trade_log=root / "TRADE-LOG.md",
        weekly_review=root / "WEEKLY-REVIEW.md",
    )


def ensure_memory_files(settings: Settings) -> MemoryPaths:
    paths = memory_paths(settings)
    paths.root.mkdir(parents=True, exist_ok=True)
    _ensure_file(paths.strategy, _DEFAULT_STRATEGY)
    _ensure_file(paths.research_log, _DEFAULT_RESEARCH_LOG)
    _ensure_file(paths.trade_log, _DEFAULT_TRADE_LOG)
    _ensure_file(paths.weekly_review, _DEFAULT_WEEKLY_REVIEW)
    return paths


def _ensure_file(path: Path, default_content: str) -> None:
    if not path.exists():
        path.write_text(default_content.strip() + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_tail(path: Path, *, max_lines: int = 80) -> str:
    text = read_text(path)
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def has_research_for_date(path: Path, session_date: str) -> bool:
    text = read_text(path)
    return f"## {session_date}" in text


def extract_research_section(path: Path, session_date: str) -> str:
    text = read_text(path)
    marker = f"## {session_date}"
    if marker not in text:
        return ""
    start = text.index(marker)
    rest = text[start + len(marker) :]
    nxt = rest.find("\n## ")
    body = rest if nxt < 0 else rest[:nxt]
    return marker + body


def append_section(path: Path, section: str) -> None:
    section = section.strip()
    if not section:
        return
    existing = read_text(path)
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + "\n" + section + "\n", encoding="utf-8")


def list_research_dates(path: Path) -> list[str]:
    text = read_text(path)
    return sorted(_RESEARCH_HEADER.findall(text))


def is_friday(session_date: str) -> bool:
    return date.fromisoformat(session_date).weekday() == 4
