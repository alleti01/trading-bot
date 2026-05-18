"""Static check: ``agents/`` must not import ``execution/`` or ``risk/``.

This is the architectural safety property the spec requires: LLM agents
can summarize, explain, and report — they must never reach into the parts
of the codebase that route trades or set risk limits.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "agents"

FORBIDDEN_PATTERNS = [
    re.compile(r"^\s*from\s+execution(\.|\s)", re.MULTILINE),
    re.compile(r"^\s*import\s+execution(\.|\s|$)", re.MULTILINE),
    re.compile(r"^\s*from\s+risk(\.|\s)", re.MULTILINE),
    re.compile(r"^\s*import\s+risk(\.|\s|$)", re.MULTILINE),
]


@pytest.mark.parametrize(
    "py_file",
    sorted(p for p in AGENTS_DIR.rglob("*.py")),
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_agent_module_has_no_forbidden_imports(py_file: Path) -> None:
    src = py_file.read_text()
    for pattern in FORBIDDEN_PATTERNS:
        assert not pattern.search(src), (
            f"{py_file.relative_to(REPO_ROOT)} imports a forbidden package "
            f"(execution/risk). Agents are advisory only."
        )
