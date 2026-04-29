from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.factory import DatasetFactory
from net.common import Config
from net.e3gnn import E3GNN

# %%
matrix_path = REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.matrix"
info_path = REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.info.out"

cfg = Config(
    cutoff_radius=7.0,
    matrix_targets=["hamiltonian", "overlap", "density"],
    train_target="matrix",
    num_layers_gnn=1,
    hidden_base_dim=16,
    l_max=2,
    n_radial=16,
    radial_layers=(32,),
    head_e3mlp_layers=1,
    internal_e3mlp_layers=0,
    neck_depth=1,
    tp_type="separate_weight",
    e3mlp_variant="basic",
    head_e3mlp_variant="film",
    separate_shifted_self=True,
    head_use_tensor_square=False,
    verbosity=0,
)

factory = DatasetFactory(cfg, convention="e3nn")
factory.add_snapshot(matrix_path, info_path, purpose="train")
train_ds, _, mapper = factory.create()
x, y = train_ds[0]

model = E3GNN(mapper=mapper, cfg=cfg)
t0 = time.perf_counter()
preds = model(x)
t1 = time.perf_counter()

# %%
print("Model knobs")
print("tp_type:", cfg.tp_type)
print("e3mlp_variant:", cfg.e3mlp_variant)
print("head_e3mlp_variant:", cfg.head_e3mlp_variant)
print("parameter_count:", sum(p.numel() for p in model.parameters()))
print("forward_time_sec:", round(t1 - t0, 4))

print("Head outputs")
for name, pred in preds.items():
    print(
        name,
        "H-H shape:",
        tuple(pred["H-H"].shape),
        "H-O shape:",
        tuple(pred["H-O"].shape),
    )

# %%
cfg_alt = Config(
    cutoff_radius=7.0,
    matrix_targets=["hamiltonian"],
    train_target="matrix",
    num_layers_gnn=1,
    hidden_base_dim=16,
    l_max=2,
    n_radial=16,
    radial_layers=(32,),
    head_e3mlp_layers=1,
    internal_e3mlp_layers=0,
    neck_depth=1,
    tp_type="fully_connected",
    e3mlp_variant="gate",
    head_e3mlp_variant="gate_magnitudes",
    separate_shifted_self=False,
    head_use_tensor_square=True,
    verbosity=0,
)

model_alt = E3GNN(mapper=mapper, cfg=cfg_alt)
preds_alt = model_alt(x)

print("Alternative knobs")
print("tp_type:", cfg_alt.tp_type)
print("e3mlp_variant:", cfg_alt.e3mlp_variant)
print("head_e3mlp_variant:", cfg_alt.head_e3mlp_variant)
print("parameter_count:", sum(p.numel() for p in model_alt.parameters()))
print("hamiltonian H-H shape:", tuple(preds_alt["hamiltonian"]["H-H"].shape))
