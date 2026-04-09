#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

SNAPSHOT="${SNAPSHOT:-2700K}"
CUTOFF_RADIUS="${CUTOFF_RADIUS:-7.0}"
L_MAX="${L_MAX:-4}"
N_RADIAL="${N_RADIAL:-64}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

python studies/e3mlp_investigation/scripts/build_silicon_pair_cache.py \
  --snapshot "${SNAPSHOT}" \
  --cutoff-radius "${CUTOFF_RADIUS}" \
  --l-max "${L_MAX}" \
  --n-radial "${N_RADIAL}"
