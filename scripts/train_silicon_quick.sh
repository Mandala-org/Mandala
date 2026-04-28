#!/bin/bash
# train_silicon_quick.sh
# Training script for Silicon with DeepH-E3 architecture
# Uses settings from sweeps/quick_test_new.yaml

set -e  # Exit on error

# Activate virtual environment
source mandala-venv/bin/activate

# Run training with exact settings from quick_test_new.yaml
python3 -u scripts/train_silicon.py \
  --gpus=1 \
  --num_workers=4 \
  --n_snapshots_per_temp=1 \
  --val_n_snapshots=0 \
  --max_epochs=2000 \
  --min_temp=300 \
  --max_temp=300 \
  --batch_size=1 \
  --accumulate_grad_batches=1 \
  --log_every_n_steps=10 \
  --tp_type="separate_weight" \
  --use_self_connection=True \
  --cutoff_radius=7.5 \
  --l_max=4 \
  --hidden_base_dim=64 \
  --emb_use_odd_features=True \
  --num_layers_gnn=2 \
  --node_update_message_agg="sum" \
  --edge_update_residual=True \
  --node_update_residual=True \
  --n_radial=64 \
  --radial_layers="[128]" \
  --neck_depth=1 \
  --head_e3mlp_layers=1 \
  --head_use_mlp_log_scale=False \
  --nonlin_kind="normact" \
  --activation_scalar="silu" \
  --activation_gate="sigmoid" \
  --norm_kind="component" \
  --lr=3.0e-4 \
  --use_lr_scheduler=True \
  --lr_scheduler_factor=0.5 \
  --lr_scheduler_patience=60 \
  --lr_scheduler_target="train/loss_total" \
  --dropout=0.0 \
  --train_target="matrix" \
  --loss_l1_fraction=0.0 \
  --matrix_targets="[hamiltonian]" \
  --train_on_energy=False \
  --train_on_num_electrons=False \
  --enable_energy=False \
  --enable_num_electrons=False \
  --loss_coef_observables=0.0 \
  --train_on_forces=False \
  --train_on_stress=False \
  --enable_forces=False \
  --enable_stress=False \
  --symmetrize_output=True \
  --precompute_edge_features=True \
  --safety_checks=True \
  --verbosity=2

echo ""
echo "Training complete!"
