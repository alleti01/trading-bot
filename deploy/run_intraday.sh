#!/usr/bin/env bash
# Launch the autonomous intraday paper loop.
# Edit REPO_DIR if you move the project. Keeps the Mac awake (caffeinate)
# and the venv activated. The loop idles outside market hours on its own.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/Users/anishalleti/tradeify trading bot}"
cd "$REPO_DIR"

# shellcheck disable=SC1091
source ".venv/bin/activate"

# --no-workflow-dry-run places Alpaca PAPER bracket orders on approved
# setups. Drop the flag for a dry run (no orders).
MODE_FLAG="${1:---no-workflow-dry-run}"

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i python -m app.main --workflow-intraday "$MODE_FLAG"
else
  exec python -m app.main --workflow-intraday "$MODE_FLAG"
fi
