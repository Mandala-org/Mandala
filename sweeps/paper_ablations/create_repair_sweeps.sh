#!/usr/bin/env bash
set -euo pipefail

ROOT="sweeps/paper_ablations/repairs"

for sweep in \
  paper_siox_mature_energy_guidance_12h_repair2_seed42.yaml
do
  printf '\n=== Creating %s ===\n' "${sweep}"
  wandb sweep "${ROOT}/${sweep}"
done
