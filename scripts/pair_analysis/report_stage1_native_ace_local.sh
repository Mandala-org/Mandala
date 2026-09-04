#!/usr/bin/env bash
set -euo pipefail

cd /home/bartek/casus/mandala
mkdir -p artifacts/pair_stage1/stage1_benchmark_report_v1

mandala-venv/bin/python -u scripts/pair_analysis/report_stage1_native_ace.py \
  --audit-summary artifacts/pair_stage1/sio2_data_audit_v1/summary.json \
  --cache-summary artifacts/pair_stage1/sio2_baseline_cache_v1/summary.json \
  --core-summary artifacts/pair_stage1/native_ace_core_h100_v1/summary.json \
  --baseline-summary artifacts/pair_stage1/native_ace_sio2_h100_v1/summary.json \
  --output-dir artifacts/pair_stage1/stage1_benchmark_report_v1
