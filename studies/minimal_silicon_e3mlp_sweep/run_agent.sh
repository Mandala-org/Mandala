#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <entity/project/sweep_id> [count]" >&2
  exit 1
fi

SWEEP_ID="$1"
COUNT="${2:-1}"

cd "$(dirname "$0")/../.."
source mandala-venv/bin/activate
export PYTHONUNBUFFERED=1

for ((i=1; i<=COUNT; i++)); do
  echo "[agent] starting $i/$COUNT for $SWEEP_ID"
  wandb agent "$SWEEP_ID"
done
