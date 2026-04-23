#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

STUDY_YAML="${1:-sweeps/train_silicon_density_energy_bayes_short_optuna.yaml}"

echo "=== Mandala cluster cache diagnostic ==="
echo "root_dir=$ROOT_DIR"
echo "study_yaml=$STUDY_YAML"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date=$(date -Is)"
echo "python=$(command -v python || true)"
echo "SCRATCH=${SCRATCH:-<unset>}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-<unset>}"
echo

PRIMARY_VENV="$HOME/casus/mandala-venv/bin/activate"
LOCAL_VENV="$ROOT_DIR/mandala-venv/bin/activate"

if [[ -f "$PRIMARY_VENV" ]]; then
  # shellcheck disable=SC1090
  source "$PRIMARY_VENV"
  echo "Activated venv: $HOME/casus/mandala-venv"
elif [[ -f "$LOCAL_VENV" ]]; then
  # shellcheck disable=SC1091
  source "$LOCAL_VENV"
  echo "Activated venv: $ROOT_DIR/mandala-venv"
else
  echo "WARNING: no venv activate script found at:"
  echo "  $PRIMARY_VENV"
  echo "  $LOCAL_VENV"
fi

python - <<'PY' "$STUDY_YAML"
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path.cwd()
sys.path.append(str(ROOT))

from net.common import Config
from scripts.dataset import build_datasets_from_yaml


def canonical(name: str) -> str:
    return name.replace("-", "_")


def value_from_spec(spec):
    if isinstance(spec, dict):
        if "value" in spec:
            return spec["value"]
        return None
    return spec


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


study_yaml = Path(sys.argv[1])
parsed = OmegaConf.to_container(OmegaConf.load(study_yaml), resolve=True)
if not isinstance(parsed, dict):
    raise RuntimeError(f"Expected mapping at top level of {study_yaml}")

params = parsed.get("parameters", {})
if not isinstance(params, dict):
    raise RuntimeError("Expected 'parameters' in study yaml")

cfg = Config()
fixed_args: dict[str, object] = {}

for raw_name, spec in params.items():
    value = value_from_spec(spec)
    if value is None:
        continue
    name = canonical(raw_name)
    fixed_args[name] = value
    if hasattr(cfg, name):
        setattr(cfg, name, value)

print("=== Resolved dataset-relevant config ===")
for key in (
    "dataset_kind",
    "data_path",
    "snapshot_cache_dir",
    "checkpoint_dir",
    "seed",
    "num_train",
    "num_val",
    "cutoff_radius",
    "apply_cutoff_to_targets",
    "require_exact_edge_match",
    "precompute_edge_features",
    "matrix_targets",
    "train_target",
    "symmetrize_hamiltonian_targets",
    "separate_shifted_self",
):
    print(f"{key}={fixed_args.get(key, getattr(cfg, key, '<missing>'))}")
print()

cache_root = cfg.snapshot_cache_dir
if cache_root is None:
    print("ERROR: snapshot_cache_dir is not set; cache is disabled.")
    raise SystemExit(2)

cache_root = Path(str(cache_root)).expanduser()
pre_dir = cache_root / "preprocessed_samples"
print(f"cache_root={cache_root}")
print(f"preprocessed_cache_dir={pre_dir}")
print(f"cache_root_exists={cache_root.exists()}")
print(f"pre_dir_exists={pre_dir.exists()}")
print(f"cache_root_file_count_before={count_files(cache_root)}")
print(f"pre_dir_file_count_before={count_files(pre_dir)}")
print()

def build_once(tag: str):
    print(f"=== Dataset build: {tag} ===")
    train_ds, val_ds, mapper = build_datasets_from_yaml(
        parsed,
        cfg,
        overrides=fixed_args,
        convention=str(fixed_args.get("convention", "e3nn")),
    )
    print(
        f"{tag}: train_len={len(train_ds)}, val_len={len(val_ds)}, "
        f"mapper_edge_types={len(mapper.edge_types)}"
    )
    print(
        f"{tag}: train snapshot cache hits={train_ds.snapshot_cache_hits}, "
        f"misses={train_ds.snapshot_cache_misses}"
    )
    print(
        f"{tag}: train preprocessed cache hits={train_ds.preprocessed_cache_hits}, "
        f"misses={train_ds.preprocessed_cache_misses}"
    )
    print(
        f"{tag}: val snapshot cache hits={val_ds.snapshot_cache_hits}, "
        f"misses={val_ds.snapshot_cache_misses}"
    )
    print(
        f"{tag}: val preprocessed cache hits={val_ds.preprocessed_cache_hits}, "
        f"misses={val_ds.preprocessed_cache_misses}"
    )
    return train_ds, val_ds


build_once("first")
print()
print(f"cache_root_file_count_after_first={count_files(cache_root)}")
print(f"pre_dir_file_count_after_first={count_files(pre_dir)}")
print()

build_once("second")
print()
print(f"cache_root_file_count_after_second={count_files(cache_root)}")
print(f"pre_dir_file_count_after_second={count_files(pre_dir)}")
print()

if pre_dir.exists():
    files = sorted(p for p in pre_dir.iterdir() if p.is_file())
    print("=== Sample preprocessed cache files ===")
    for path in files[:10]:
        print(path)
else:
    print("No preprocessed cache directory exists after two builds.")
PY
