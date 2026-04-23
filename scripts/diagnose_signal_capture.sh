#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

echo "=== signal diagnostic wrapper ==="
echo "wrapper_pid=$$"
echo "parent_pid=${PPID:-<unknown>}"
echo "hostname=$(hostname)"
echo "date=$(date -Is)"
echo "command=$*"
echo

child_pid=""

forward_signal() {
  local sig="$1"
  echo "[trap] received ${sig} at $(date -Is)"
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    echo "[trap] forwarding ${sig} to child pid=${child_pid}"
    kill -s "$sig" "$child_pid" 2>/dev/null || true
  fi
}

trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM

"$@" &
child_pid=$!
echo "child_pid=$child_pid"

wait "$child_pid"
status=$?
echo "[wrapper] child exited with status=$status at $(date -Is)"
exit "$status"
