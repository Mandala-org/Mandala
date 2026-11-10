#!/usr/bin/env bash
set -euo pipefail
# Preserve this already-running legacy chain's immutable source and recipe.
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
frozen="$root/.cache/publication_retrain_v2/legacy_continuation_source"
checkpoint=/bigdata/casus/wdm/hamiltonian_learning/models/checkpoints/publication_retrain_v2/sio2/latest_checkpoint.pt
if [[ -f "$checkpoint" ]]; then
    test -f "$frozen/scripts/reproduce_examples/sio2_v2.json"
    cd "$frozen"
    export PYTHONPATH="$frozen/src:$frozen${PYTHONPATH:+:$PYTHONPATH}"
    exec bash scripts/reproduce_examples/train_sio2_v2.sh "$@" --action continue
fi
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR=/tmp/mandala-publication-mpl
cd "$root"
exec python -u scripts/reproduce_examples/train.py --recipe scripts/reproduce_examples/sio2_v2.json "$@"
