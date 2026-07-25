#!/usr/bin/env bash
set -euo pipefail

ROOT="sweeps/paper_ablations"
OUT="${ROOT}/repairs"
GENERATOR="scripts/create_wandb_repair_sweep.py"

python -u "${GENERATOR}" \
  --source "${ROOT}/paper_siox_mature_energy_guidance_12h.yaml" \
  --output "${OUT}/paper_siox_mature_energy_guidance_12h_repair2_seed42.yaml" \
  --name "paper_siox_mature_energy_guidance_12h_repair2_seed42" \
  --value 'seed=42' \
  --value 'loss-coef-observables=0.001'

printf '\nRepair sweep YAMLs are ready under %s\n' "${OUT}"
