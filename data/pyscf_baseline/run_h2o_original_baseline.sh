#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/mandala-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/mandala-venv/bin/activate"
fi

RUN_NAME="${RUN_NAME:-h2o_original_rhf_openmx_like}"
INFO_PATH="${INFO_PATH:-data/small/H2O/original/H2O.info.out}"
OUTPUT_DIR="${OUTPUT_DIR:-data/pyscf_baseline/results}"
METHOD="${METHOD:-heuristic}"
KMESH="${KMESH:-1 1 1}"

read -r -a KMESH_ARR <<< "${KMESH}"
if [[ "${#KMESH_ARR[@]}" -ne 3 ]]; then
  echo "KMESH must contain exactly 3 integers, got: '${KMESH}'" >&2
  exit 2
fi

cmd=(
  python data/pyscf_baseline/calc_pyscf_baseline.py
  --info-path "${INFO_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --method "${METHOD}"
  --basis-mode openmx_like
  --kmesh "${KMESH_ARR[0]}" "${KMESH_ARR[1]}" "${KMESH_ARR[2]}"
  --overwrite
)

# Optional parse-only mode.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  cmd+=(--dry-run)
fi

cmd+=("$@")
"${cmd[@]}"
