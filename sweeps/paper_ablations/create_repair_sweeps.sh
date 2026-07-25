#!/usr/bin/env bash
set -euo pipefail

ROOT="sweeps/paper_ablations/repairs"

for sweep in \
  paper_zncusnses_mature_full_network_spectral_12h_repair_seed41.yaml \
  paper_siox_mature_energy_guidance_12h_repair_seed42.yaml \
  paper_silicon_node_aggregation_12h_repair.yaml \
  paper_siox_mature_energy_guidance_3em5_12h_repair.yaml \
  paper_zncusnses_big_mature_spectral_24h_control_repair.yaml
do
  printf '\n=== Creating %s ===\n' "${sweep}"
  wandb sweep "${ROOT}/${sweep}"
done
