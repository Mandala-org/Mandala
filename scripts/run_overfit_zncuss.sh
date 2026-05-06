#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source "$repo_root/mandala-venv/bin/activate"

# Create a throwaway staging tree containing only the ZnCuSeS snapshot so the
# usual dataset discovery code sees exactly one sample.
staging_root="$(mktemp -d /tmp/mandala_zncuss_overfit.XXXXXX)"
mkdir -p "$staging_root"
ln -s "$repo_root/data/small/ZnCuSeS" "$staging_root/ZnCuSeS"

export WANDB_MODE=disabled
export STAGING_ROOT="$staging_root"

python -u - <<'PY'
from argparse import Namespace
from pathlib import Path
import os

from scripts.train import run_training

staging_root = Path(os.environ["STAGING_ROOT"])

args = Namespace(
    dataset_kind="siox",
    data_path=str(staging_root),
    num_train=1,
    num_val=0,
    val_fraction=0.0,
    convention="e3nn",
    precision="32-true",
    checkpoint_dir="checkpoints/overfit_zncuss",
    run_name="overfit-zncuss-one-snapshot",
    wandb_mode="disabled",
    wandb_project=None,
    log_artifacts=False,
    generate_video=False,
    # Keep the full snapshot; the largest edge length we observed is ~10.58 Å.
    cutoff_radius=11.0,
    max_epochs=500,
    batch_size=1,
    lr=1e-3,
    use_lr_scheduler=False,
    revert_on_spike=False,
    benchmark=False,
    log_every_n_steps=1,
    num_workers=0,
    seed=42,
    dtype="float32",
    gpus=0,
    hidden_base_dim=32,
    l_max=2,
    num_layers_gnn=3,
    n_radial=32,
    dropout=0.0,
    grad_clip_val=0.0,
    loss_coef_observables=1e-4,
    loss_coef_forces=0.0,
    loss_coef_stress=0.0,
    train_on_forces=False,
    train_on_stress=False,
    train_on_energy=True,
    train_on_num_electrons=True,
    enable_forces=False,
    enable_stress=False,
    log_model=False,
    log_data=False,
    safety_checks=False,
    snapshot_cache_dir="/tmp/mandala_zncuss_snapshot_cache",
    dataset_device=None,
    apply_cutoff_to_targets=True,
    require_exact_edge_match=True,
    matrix_targets=["hamiltonian", "overlap", "density"],
)

metrics = run_training(args)
print("Final metrics:", metrics)
PY
