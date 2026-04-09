#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

bash studies/e3mlp_investigation/scripts/batch1_run01_stability_core.sh
bash studies/e3mlp_investigation/scripts/batch1_run02_stability_advanced.sh
bash studies/e3mlp_investigation/scripts/batch1_run03_synth_mixed.sh
bash studies/e3mlp_investigation/scripts/batch1_run04_synth_quadratic.sh
bash studies/e3mlp_investigation/scripts/batch1_run05_synth_linear_baseline.sh
