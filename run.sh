#!/usr/bin/env bash
set -euo pipefail

LOGFILE="${1:-train.log}"

echo "==> Starting run at $(date)"
echo "==> Log file: $LOGFILE"

if [ -d ".venv" ]; then
  echo "==> Activating existing venv .venv"
  source .venv/bin/activate
else
  echo "==> Creating venv at .venv"
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
fi

pip install -r requirements.txt

echo "==> Running: python -u src/train.py"
python -u src/train.py > "$LOGFILE" 2>&1

echo "==> Finished at $(date). Logs saved to $LOGFILE"
