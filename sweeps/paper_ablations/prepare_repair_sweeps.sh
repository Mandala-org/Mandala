#!/usr/bin/env bash
set -euo pipefail

ROOT="sweeps/paper_ablations"
OUT="${ROOT}/repairs"
GENERATOR="scripts/create_wandb_repair_sweep.py"

python -u "${GENERATOR}" \
  --source "${ROOT}/paper_zncusnses_mature_full_network_spectral_12h.yaml" \
  --output "${OUT}/paper_zncusnses_mature_full_network_spectral_12h_repair_seed41.yaml" \
  --name "paper_zncusnses_mature_full_network_spectral_12h_repair_seed41" \
  --values 'seed=[41]' \
  --values 'spectral-loss-coef=[0.0, 0.001]'

python -u "${GENERATOR}" \
  --source "${ROOT}/paper_siox_mature_energy_guidance_12h.yaml" \
  --output "${OUT}/paper_siox_mature_energy_guidance_12h_repair_seed42.yaml" \
  --name "paper_siox_mature_energy_guidance_12h_repair_seed42" \
  --value 'seed=42' \
  --value 'loss-coef-observables=0.001'

python -u "${GENERATOR}" \
  --source "${ROOT}/paper_silicon_node_aggregation_12h.yaml" \
  --output "${OUT}/paper_silicon_node_aggregation_12h_repair.yaml" \
  --name "paper_silicon_node_aggregation_12h_repair" \
  --value 'evaluate-test-after-fit=false'

python -u "${GENERATOR}" \
  --source "${ROOT}/paper_siox_mature_energy_guidance_3em5_12h.yaml" \
  --output "${OUT}/paper_siox_mature_energy_guidance_3em5_12h_repair.yaml" \
  --name "paper_siox_mature_energy_guidance_3em5_12h_repair" \
  --values 'seed=[41, 43, 44]'

python -u "${GENERATOR}" \
  --source "${ROOT}/paper_zncusnses_big_mature_spectral_24h.yaml" \
  --output "${OUT}/paper_zncusnses_big_mature_spectral_24h_control_repair.yaml" \
  --name "paper_zncusnses_big_mature_spectral_24h_control_repair" \
  --value 'spectral-loss-coef=0.0' \
  --values 'seed=[43, 44]'

printf '\nRepair sweep YAMLs are ready under %s\n' "${OUT}"
