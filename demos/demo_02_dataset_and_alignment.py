from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from data.factory import DatasetFactory
from net.common import Config

# %%
matrix_path = REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.matrix"
info_path = REPO_ROOT / "data" / "small" / "H2O" / "original" / "H2O.info.out"

cfg_matrix = Config(
    cutoff_radius=7.0,
    allow_openmx_positions_box_from_out=True,
    train_target="matrix",
    matrix_targets=["hamiltonian", "overlap", "density"],
    apply_cutoff_to_targets=True,
    require_exact_edge_match=True,
    precompute_edge_features=True,
    safety_checks=True,
    verbosity=0,
)

factory_matrix = DatasetFactory(cfg_matrix, convention="e3nn")
factory_matrix.add_snapshot(matrix_path, info_path, purpose="train")
train_matrix_ds, _, mapper = factory_matrix.create()
x_matrix, y_matrix = train_matrix_ds[0]

# %%
cfg_irreps = Config(
    cutoff_radius=7.0,
    allow_openmx_positions_box_from_out=True,
    train_target="irreps",
    matrix_targets=["hamiltonian", "overlap", "density"],
    apply_cutoff_to_targets=True,
    require_exact_edge_match=True,
    precompute_edge_features=True,
    safety_checks=True,
    verbosity=0,
)

factory_irreps = DatasetFactory(cfg_irreps, convention="e3nn")
factory_irreps.add_snapshot(matrix_path, info_path, purpose="train")
train_irreps_ds, _, _ = factory_irreps.create()
x_irreps, y_irreps = train_irreps_ds[0]

# %%
print("Matrix-target sample")
print("edge_index shape:", tuple(x_matrix["edge_index"].shape))
print("num_self_edges:", x_matrix["num_self_edges"])
print("pred_pair_edges_static keys:", sorted(x_matrix["pred_pair_edges_static"].keys()))
print("hamiltonian target type:", type(y_matrix["hamiltonian"]).__name__)

print("Irrep-target sample")
print("hamiltonian target type:", type(y_irreps["hamiltonian"]).__name__)
print("hamiltonian vector keys:", sorted(y_irreps["hamiltonian"].pair_vectors.keys()))
print("pred_trace_alignment keys:", sorted(x_irreps["pred_trace_alignment"].keys()))
