#!/usr/bin/env bash
set -euo pipefail

cd /home/brzoza73/casus/mandala

python -u scripts/pair_data/prepare_sio2.py \
  --input /bigdata/casus/wdm/hamiltonian_learning/data/graph_data_SiO2.zip \
  --extraction-dir /bigdata/casus/wdm/hamiltonian_learning/data/pair_descriptor_hamiltonian/sio2_source_v1 \
  --output-dir /home/brzoza73/casus/mandala/artifacts/pair_stage1/sio2_data_audit_v1 \
  --seed 42 \
  --train-ratio 0.8 \
  --validation-ratio 0.1 \
  --distance-bin-width-angstrom 0.05 \
  --benchmark-mae-mev 2.29 \
  --absolute-tail-mae-limit-mev 0.1 \
  --retained-squared-fraction 0.999 \
  --descriptor-cutoffs-angstrom 4.0 6.0 8.0 10.0 \
  --nao-max 14 \
  --torch-threads 1 \
  --resume
