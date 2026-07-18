#!/usr/bin/env bash
set -euo pipefail

ROOT="sweeps/paper_ablations"

for sweep in \
  paper_zncusnses_envelope_12h.yaml \
  paper_zncusnses_pair_radial_mlp_12h.yaml \
  paper_silicon_edge_sh_square_12h.yaml \
  paper_zncusnses_node_aggregation_12h.yaml \
  paper_zncusnses_shifted_self_12h.yaml \
  paper_siox_envelope_12h.yaml \
  paper_silicon_spectral_guidance_12h.yaml \
  paper_zncusnses_spectral_guidance_12h.yaml \
  paper_siox_energy_guidance_12h.yaml
do
  printf '\n=== Creating %s ===\n' "${sweep}"
  wandb sweep "${ROOT}/${sweep}"
done
