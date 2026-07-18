#!/usr/bin/env bash
set -euo pipefail

ROOT="sweeps/paper_ablations"

for sweep in \
  paper_zncusnses_envelope_11h5.yaml \
  paper_zncusnses_pair_radial_mlp_11h5.yaml \
  paper_zncusnses_edge_sh_square_11h5.yaml \
  paper_zncusnses_node_aggregation_11h5.yaml \
  paper_zncusnses_shifted_self_11h5.yaml \
  paper_siox_envelope_11h5.yaml \
  paper_siox_spectral_guidance_11h5.yaml \
  paper_silicon_observable_guidance_11h5.yaml
do
  printf '\n=== Creating %s ===\n' "${sweep}"
  wandb sweep "${ROOT}/${sweep}"
done
