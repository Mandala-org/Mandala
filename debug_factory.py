"""Debug script for test_factory dimension mismatch."""

import torch
from pathlib import Path

from net.common import Config
from data.factory import DatasetFactory
from net.e3gnn import E3GNN

# Setup same as test
PAIR_TRAIN_1 = (
    Path("data/small/H2O/original/H2O.matrix"),
    Path("data/small/H2O/original/H2O.info.out"),
)
PAIR_TRAIN_2 = (
    Path("data/big/silicon/900K/Si_DM"),
    Path("data/big/silicon/900K/info.txt"),
)
PAIR_VAL_1 = (
    Path("data/small/H2O/original/H2O.matrix"),
    Path("data/small/H2O/original/H2O.info.out"),
)

cfg = Config(
    cutoff_radius=7.0,
    device="cpu",
    train_target="matrix",
    dtype=torch.float32,
    safety_checks=True,
)

print("Creating dataset factory...")
fac = DatasetFactory(cfg)
fac.add_snapshot(*PAIR_TRAIN_1, purpose="train")
fac.add_snapshot(*PAIR_TRAIN_2, purpose="train")
fac.add_snapshot(*PAIR_VAL_1, purpose="val")
train_ds, val_ds, mapper = fac.create()

print("\n" + "=" * 70)
print("Mapper Information:")
print(f"  Elements: {mapper.orbital_cfg.elements()}")
print(f"  Number of elements: {len(mapper.orbital_cfg.elements())}")
print(f"  Edge types: {len(mapper.edge_types)} unique pairs")
print()

# Get first sample
sample = train_ds[0]
x, y = sample

print("Sample 'x' keys:", list(x.keys()))
print(f"  node_type_idx.shape: {x['node_type_idx'].shape}")
print(f"  edge_type_idx.shape: {x['edge_type_idx'].shape}")
print(f"  edge_index.shape: {x['edge_index'].shape}")
print(f"  edge_sh.shape: {x['edge_sh'].shape}")
print(f"  edge_length_emb.shape: {x['edge_length_emb'].shape}")
print(f"  atoms: {x['atoms']}")
print()

# Create model
print("Creating E3GNN model...")
model_cfg = Config(dropout=0.0, safety_checks=True, l_max=cfg.l_max)
model = E3GNN(mapper=mapper, cfg=model_cfg)

print("\nModel Configuration:")
print(f"  cfg.l_max: {model_cfg.l_max}")
print(f"  cfg.hidden_base_dim: {model_cfg.hidden_base_dim}")
print(f"  cfg.n_radial: {model_cfg.n_radial}")
print()

print("Model Irreps:")
print(f"  hidden_irreps: {model.hidden_irreps}")
print(f"  hidden_irreps.dim: {model.hidden_irreps.dim}")
print(f"  node_enc.irreps_out: {model.node_enc.irreps_out}")
print(f"  node_enc.irreps_out.dim: {model.node_enc.irreps_out.dim}")
print(f"  sh_irreps: {model.sh_irreps}")
print(f"  sh_irreps.dim: {model.sh_irreps.dim}")
print()

print("MessageBlock[0]:")
mp_blk = model.mp_blocks[0]
print(f"  node_irreps: {mp_blk.node_irreps}")
print(f"  node_irreps.dim: {mp_blk.node_irreps.dim}")
print(f"  edge_irreps: {mp_blk.edge_irreps}")
print(f"  edge_irreps.dim: {mp_blk.edge_irreps.dim}")
print()

print("EdgeUpdateBlock:")
edge_blk = mp_blk.edge_upd
print(f"  node_irreps: {edge_blk.node_irreps}")
print(f"  node_irreps.dim: {edge_blk.node_irreps.dim}")
print(f"  edge_irreps: {edge_blk.edge_irreps}")
print(f"  edge_irreps.dim: {edge_blk.edge_irreps.dim}")
print(f"  irreps_out: {edge_blk.irreps_out}")
print(f"  irreps_out.dim: {edge_blk.irreps_out.dim}")
print()

# Try encoding
print("Encoding inputs...")
node = model.node_enc(x["node_type_idx"])
edge = model.edge_enc(x["edge_type_idx"], x["edge_length_emb"], x["edge_sh"])

print(
    f"  node.shape: {node.shape} (expected: [{x['node_type_idx'].shape[0]}, {model.node_enc.irreps_out.dim}])"
)
print(
    f"  edge.shape: {edge.shape} (expected: [{x['edge_type_idx'].shape[0]}, {model.hidden_irreps.dim}])"
)
print()

# Check what EdgeUpdateBlock expects
print("EdgeUpdateBlock EquiConv:")
conv = edge_blk.conv
print(f"  conv.tp type: {type(conv.tp).__name__}")
print(f"  conv.irreps_out: {conv.irreps_out}")
print(f"  conv.irreps_out.dim: {conv.irreps_out.dim}")

# Access TP irreps (these are public attributes)
if hasattr(conv.tp, "irreps_in1"):
    print(f"  TP.irreps_in1: {conv.tp.irreps_in1}")
    print(f"  TP.irreps_in1.dim: {conv.tp.irreps_in1.dim}")
    print(f"  TP.irreps_in2: {conv.tp.irreps_in2}")
    print(f"  TP.irreps_out: {conv.tp.irreps_out}")
print()

# Expected concatenated input
N = x["node_type_idx"].shape[0]
E = x["edge_index"].shape[1]
print("During EdgeUpdateBlock.forward:")
print(f"  N (nodes): {N}")
print(f"  E (edges): {E}")
print(f"  node[src].shape would be: [{E}, {node.shape[1]}]")
print(f"  node[dst].shape would be: [{E}, {node.shape[1]}]")
print(f"  edge.shape: {edge.shape}")
print(
    f"  Concatenated shape would be: [{E}, {2*node.shape[1] + edge.shape[1]}] = [{E}, {2*node.shape[1] + edge.shape[1]}]"
)
print()

# What does TP expect?
if hasattr(conv, "tp"):
    print(f"  But TP expects irreps_in1.dim = {conv.tp.irreps_in1.dim}")
    print(f"  MISMATCH: {2*node.shape[1] + edge.shape[1]} != {conv.tp.irreps_in1.dim}")

print("\n" + "=" * 70)
