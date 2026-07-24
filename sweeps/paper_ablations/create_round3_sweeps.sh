#!/usr/bin/env bash
set -euo pipefail

ROOT="sweeps/paper_ablations"

for sweep in \
  paper_siox_mature_energy_guidance_3em5_12h.yaml
do
  printf '\n=== Creating %s ===\n' "${sweep}"
  wandb sweep "${ROOT}/${sweep}"
done
