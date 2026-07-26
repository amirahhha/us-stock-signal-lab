#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h:h}"

if [[ -x "${PROJECT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
elif [[ -x "/opt/anaconda3/bin/python3" ]]; then
  PYTHON_BIN="/opt/anaconda3/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" -m streamlit run app.py
