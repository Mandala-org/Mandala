#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Backward-compatible wrapper with an explicit production name.
exec bash data/pyscf_baseline/run_h2o_original_kmesh_match_openmx.sh "$@"
