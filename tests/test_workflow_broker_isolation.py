"""Workflows must use only the broker router (no broker-specific imports)."""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"

# Module-level imports forbidden in workflows/. The router and the
# abstract base are the only allowed integrations modules.
_FORBIDDEN_PATTERNS = (
    re.compile(r"^\s*from\s+integrations\.alpaca_paper_client\s+import"),
    re.compile(r"^\s*from\s+integrations\.tradovate_demo_client\s+import"),
    re.compile(r"^\s*from\s+integrations\.mock_broker\s+import"),
    re.compile(r"^\s*import\s+integrations\.alpaca_paper_client"),
    re.compile(r"^\s*import\s+integrations\.tradovate_demo_client"),
    re.compile(r"^\s*import\s+integrations\.mock_broker"),
)


def test_workflows_only_use_broker_router() -> None:
    """Fail loudly if any workflow file imports a broker-specific class."""
    offenders: list[str] = []
    for py_file in WORKFLOWS_DIR.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pat in _FORBIDDEN_PATTERNS:
                if pat.match(line):
                    offenders.append(
                        f"{py_file.relative_to(WORKFLOWS_DIR.parent)}:{line_no}: {line.strip()}"
                    )
    assert not offenders, (
        "Workflows must call broker_router only. Found broker-specific "
        "imports:\n  " + "\n  ".join(offenders)
    )


def test_workflows_use_build_broker_or_router() -> None:
    """Spot-check that at least one workflow module wires through the router."""
    found = False
    for py_file in WORKFLOWS_DIR.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "build_broker" in text or "BrokerRouter" in text:
            found = True
            break
    assert found, "No workflow file wires through integrations.broker_router."
