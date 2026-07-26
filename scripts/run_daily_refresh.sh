#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
elif [[ -x "/opt/anaconda3/bin/python3" ]]; then
  PYTHON_BIN="/opt/anaconda3/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

cd "$PROJECT_DIR"
"$PYTHON_BIN" pipeline.py --period 5y >> "$LOG_DIR/daily_refresh.log" 2>&1
