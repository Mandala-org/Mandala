#!/usr/bin/env bash
set -euo pipefail

ROOT="sweeps/paper_ablations"

for sweep in \
  paper_additional_seeds_siox_mature_energy_guidance_12h.yaml \
  paper_additional_seeds_zncusnses_mature_full_network_spectral_12h.yaml \
  paper_additional_seeds_zncusnses_envelope_12h.yaml \
  paper_additional_seeds_zncusnses_pair_radial_mlp_12h.yaml \
  paper_additional_seeds_silicon_node_aggregation_12h.yaml
do
  printf '\n=== Creating %s ===\n' "${sweep}"
  wandb sweep "${ROOT}/${sweep}"
done
