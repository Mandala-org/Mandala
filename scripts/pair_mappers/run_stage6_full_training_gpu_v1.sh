#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point: the joint onsite-weight sweep was superseded by
# fully independent onsite/offsite optimization before it was launched.
exec bash scripts/pair_mappers/run_stage6_split_neural_gpu_v1.sh
