#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

bash studies/e3mlp_investigation/scripts/batch2_run01_silicon_hamiltonian_mean.sh
bash studies/e3mlp_investigation/scripts/batch2_run02_silicon_hamiltonian_attention.sh
bash studies/e3mlp_investigation/scripts/batch2_run03_silicon_density_mean.sh
