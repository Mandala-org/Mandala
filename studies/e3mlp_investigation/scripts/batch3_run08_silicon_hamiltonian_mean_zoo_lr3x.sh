#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
source mandala-venv/bin/activate

RUN_NAME="${RUN_NAME:-batch3_silicon_hamiltonian_mean_zoo_lr3x_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
VARIANTS="${VARIANTS:-normact,gate,gatemagnitudes,resnormact,resgatemagnitudes,film,bilinear}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-studies/e3mlp_investigation/cache/mplconfig}"

echo "[batch3_run08] starting silicon hamiltonian mean zoo lr3x: ${RUN_NAME}"
IFS=',' read -r -a variant_list <<< "${VARIANTS}"
for variant in "${variant_list[@]}"; do
  subrun="${RUN_NAME}_${variant}"
  edge_batch_size=1024
  if [[ "${variant}" == "bilinear" ]]; then
    edge_batch_size=128
  fi
  echo "[batch3_run08] variant=${variant} subrun=${subrun} edge_batch_size=${edge_batch_size}"
  python -u studies/e3mlp_investigation/scripts/run_silicon_nognn_study.py \
    --train-snapshots 2700K \
    --eval-snapshots 900K \
    --target hamiltonian \
    --aggregation mean \
    --architecture single \
    --variant "${variant}" \
    --depth 4 \
    --hidden-irreps-preset silicon \
    --num-steps 80 \
    --lr 9e-4 \
    --loss-kind huber \
    --topk 8 \
    --device "${DEVICE}" \
    --output-scale 0.5 \
    --weight-init-scale 0.5 \
    --residual-scale 0.05 \
    --pre-norm \
    --edge-batch-size "${edge_batch_size}" \
    --run-name "${subrun}"
done
echo "[batch3_run08] finished: ${RUN_NAME}"
