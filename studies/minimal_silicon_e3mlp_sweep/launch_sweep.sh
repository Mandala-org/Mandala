#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source mandala-venv/bin/activate
export PYTHONUNBUFFERED=1

wandb sweep studies/minimal_silicon_e3mlp_sweep/sweep_density_energy_bayes.yaml
