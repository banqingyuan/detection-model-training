#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y python3 python3-venv python3-pip git curl
  else
    echo "python3 is not available and apt-get is missing" >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

echo "Bootstrap complete. Activate with: source .venv/bin/activate"

