#!/usr/bin/env bash
set -euo pipefail

ROOT="sweeps/paper_ablations"

for sweep in \
  paper_zncusnses_envelope_47h.yaml \
  paper_zncusnses_mature_head_spectral_12h.yaml \
  paper_silicon_sleek75_energy_guidance_47h.yaml \
  paper_siox_mature_energy_guidance_12h.yaml
do
  printf '\n=== Creating %s ===\n' "${sweep}"
  wandb sweep "${ROOT}/${sweep}"
done
