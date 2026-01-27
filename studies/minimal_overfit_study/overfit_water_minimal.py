"""
Minimal E(3)-Equivariant Network Study: Overfit Single Water Structure
=======================================================================

This study implements the SIMPLEST possible E(3)-equivariant GNN from scratch
to overfit on a single H2O snapshot. Everything is done explicitly without
importing high-level classes like E3GNN or E3GNNDataset.

Goal: Achieve highest possible accuracy through aggressive overfitting.

Architecture:
- Simple node encoder: element embedding → scalars
- Simple edge encoder: distance + spherical harmonics
- 1-2 message passing layers with basic tensor products
- Simple head: edge features → matrix blocks via BlockIrrepMapper

Verbose logging at every step for educational purposes.
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from e3nn.o3 import Irreps, spherical_harmonics, Linear, FullyConnectedTensorProduct
from e3nn.math import soft_one_hot_linspace
from ase import Atoms
from ase.neighborlist import neighbor_list

# Import only low-level data structures
from data.snapshot import Snapshot
from core.block_irrep_mapper import BlockIrrepMapper

# WandB for logging
import wandb

print("=" * 80)
print("MINIMAL WATER OVERFIT STUDY - EXPLICIT IMPLEMENTATION")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================
print("\n[CONFIG] Setting up hyperparameters...")

CONFIG = {
    # Data
    "data_path": project_root / "data/small/H2O/original/H2O.matrix",
    "info_path": project_root / "data/small/H2O/original/H2O.info.out",
    # Network architecture
    "hidden_dim": 32,  # Small for faster overfitting
    "l_max": 2,  # Up to d orbitals
    "num_layers": 2,  # Minimal depth
    "cutoff_radius": 8.0,  # Angstroms
    "n_radial": 16,  # Radial basis functions
    # Training
    "lr": 1e-3,  # Aggressive learning rate for overfitting
    "num_epochs": 1000,
    "log_interval": 50,
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Target
    "train_target": "matrix",  # Train on matrix blocks, not irrep vectors
}

# Initialize WandB
wandb.init(
    project="mandala-minimal-overfit",
    name="water-single-minimal",
    config=CONFIG,
)

print(f"  Device: {CONFIG['device']}")
print(f"  Hidden dim: {CONFIG['hidden_dim']}")
print(f"  L_max: {CONFIG['l_max']}")
print(f"  Num layers: {CONFIG['num_layers']}")
print(f"  Cutoff radius: {CONFIG['cutoff_radius']} Å")
print(f"  Learning rate: {CONFIG['lr']}")
print(f"  Epochs: {CONFIG['num_epochs']}")

device = torch.device(CONFIG["device"])

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[DATA] Loading single water snapshot...")

# Load snapshot using Snapshot.from_openmx
print(f"  Matrix file: {CONFIG['data_path']}")
print(f"  Info file: {CONFIG['info_path']}")

snapshot = Snapshot.from_openmx(
    matrix_path=CONFIG["data_path"],
    info_path=CONFIG["info_path"],
    convention="e3nn",  # Automatically converts to e3nn basis
    symmetrize_density=True,
    cutoff_radius=None,  # No filtering, we'll use all edges
    dtype=torch.float32,
)

print(f"\n  Snapshot loaded:")
print(f"    Elements: {snapshot.hamiltonian.atoms}")
print(f"    Num atoms: {len(snapshot.hamiltonian.atoms)}")
print(f"    Positions shape: {snapshot.positions.shape}")
print(f"    Box shape: {snapshot.box.shape if snapshot.box is not None else None}")
print(f"    Basis: {snapshot.hamiltonian.basis}")

print(f"\n  Matrices:")
print(f"    Hamiltonian keys: {list(snapshot.hamiltonian.pair_blocks.keys())}")
print(f"    Overlap keys: {list(snapshot.overlap.pair_blocks.keys())}")
print(f"    Density keys: {list(snapshot.density.pair_blocks.keys())}")

print(f"\n  Hamiltonian blocks:")
for key, blocks in snapshot.hamiltonian.pair_blocks.items():
    print(
        f"    {key}: shape {blocks.shape}, edges {snapshot.hamiltonian.pair_edges[key].shape}"
    )

# Extract components
hamiltonian_e3nn = snapshot.hamiltonian.to(device)
overlap_e3nn = snapshot.overlap.to(device)
density_e3nn = snapshot.density.to(device)
orbital_cfg = snapshot.hamiltonian.orbital_cfg
positions = snapshot.positions.to(device)
box = snapshot.box.to(device) if snapshot.box is not None else None
atoms_list = list(snapshot.hamiltonian.atoms)

print(f"\n  Orbital configuration:")
for elem in orbital_cfg.elements():
    irreps = orbital_cfg.element_to_irreps[elem]
    print(f"    {elem}: {irreps} (dim={irreps.dim})")

# Create BlockIrrepMapper
print("\n[MAPPER] Creating BlockIrrepMapper...")
mapper = BlockIrrepMapper(orbital_cfg, device=device, dtype=torch.float32)
print(f"  Mapper edge types: {mapper.edge_types}")
print(f"  Mapper edge_type2idx: {mapper.edge_type2idx}")

# Store target as matrix blocks (train_target = "matrix")
print("\n[TARGETS] Storing target as matrix blocks...")
target_H_matrix = hamiltonian_e3nn

print("  Target matrix blocks:")
for key in target_H_matrix.pair_blocks.keys():
    block_shape = target_H_matrix.pair_blocks[key].shape
    edge_shape = target_H_matrix.pair_edges[key].shape
    pair_irreps = mapper.get_pair_irreps(key)
    print(f"    {key}: blocks {block_shape}, edges {edge_shape}, irreps {pair_irreps}")

# =============================================================================
# BUILD GRAPH
# =============================================================================
print("\n[GRAPH] Constructing molecular graph...")

num_atoms = len(atoms_list)

print(f"  Atoms: {atoms_list}")
print(f"  Positions (e3nn frame):\n{positions}")
if box is not None:
    print(f"  Box (e3nn frame):\n{box}")

# Create ASE atoms object for neighbor list
ase_atoms = Atoms(
    symbols=atoms_list,
    positions=positions.cpu().numpy(),
    cell=box.cpu().numpy() if box is not None else None,
    pbc=box is not None,
)

# Find neighbors using ASE
print(f"\n  Finding neighbors within {CONFIG['cutoff_radius']} Å...")
src, dst, offsets = neighbor_list(
    "ijS", ase_atoms, CONFIG["cutoff_radius"], self_interaction=False
)

print(f"  Found {len(src)} off-diagonal edges")
print(f"  Edge list (first 10):")
for i in range(min(10, len(src))):
    print(f"    {src[i]} → {dst[i]} (offset: {offsets[i]})")

# Add self-edges
self_src = torch.arange(num_atoms, dtype=torch.long, device=device)
self_dst = torch.arange(num_atoms, dtype=torch.long, device=device)
self_offsets = torch.zeros((num_atoms, 3), dtype=torch.long, device=device)

# Combine all edges
all_src = torch.cat([self_src, torch.from_numpy(src).to(device)])
all_dst = torch.cat([self_dst, torch.from_numpy(dst).to(device)])
all_offsets = torch.cat([self_offsets, torch.from_numpy(offsets).to(device).long()])

edge_index = torch.stack([all_src, all_dst], dim=0)  # (2, E)
edge_shift = all_offsets.T  # (3, E)

num_self_edges = num_atoms
print(
    f"  Total edges: {edge_index.shape[1]} ({num_self_edges} self + {edge_index.shape[1] - num_self_edges} off-diagonal)"
)

# Compute edge vectors and distances
print("\n  Computing edge displacements and distances...")
if box is not None:
    shift_float = edge_shift.T.float()
    edge_vec = positions[edge_index[1]] - positions[edge_index[0]] + shift_float @ box
else:
    edge_vec = positions[edge_index[1]] - positions[edge_index[0]]

edge_dist = torch.linalg.norm(edge_vec, dim=1)

print(
    f"  Edge distances (Å): min={edge_dist.min().item():.3f}, max={edge_dist.max().item():.3f}, mean={edge_dist.mean().item():.3f}"
)
print(f"  Self-edge distances: {edge_dist[:num_self_edges]}")

# Compute spherical harmonics
print(f"\n  Computing spherical harmonics (l_max={CONFIG['l_max']})...")
sh_irreps = Irreps.spherical_harmonics(CONFIG["l_max"])
print(f"  SH irreps: {sh_irreps}")

# Normalize edge vectors (avoid division by zero for self-edges)
edge_vec_norm = edge_vec.clone()
non_zero_mask = edge_dist > 1e-6
edge_vec_norm[non_zero_mask] = edge_vec[non_zero_mask] / edge_dist[
    non_zero_mask
].unsqueeze(-1)

edge_sh = spherical_harmonics(sh_irreps, edge_vec_norm, normalize=False)
print(f"  Edge SH shape: {edge_sh.shape}")

# Radial basis functions
print(f"\n  Computing radial embeddings ({CONFIG['n_radial']} basis functions)...")
edge_length_emb = soft_one_hot_linspace(
    edge_dist,
    start=0.0,
    end=CONFIG["cutoff_radius"],
    number=CONFIG["n_radial"],
    basis="gaussian",
    cutoff=False,
)
edge_length_emb = edge_length_emb * CONFIG["n_radial"] ** 0.5  # Normalization
print(f"  Edge length embedding shape: {edge_length_emb.shape}")

# Edge type indices
print("\n  Computing edge type indices...")
element_to_idx = {elem: idx for idx, elem in enumerate(orbital_cfg.elements())}
print(f"  Element to index: {element_to_idx}")

node_type_idx = torch.tensor([element_to_idx[a] for a in atoms_list], device=device)
src_type = node_type_idx[edge_index[0]]
dst_type = node_type_idx[edge_index[1]]

# Create edge type strings and map to indices
edge_type_strs = [
    f"{atoms_list[edge_index[0, i].item()]}-{atoms_list[edge_index[1, i].item()]}"
    for i in range(edge_index.shape[1])
]
edge_type_idx = torch.tensor(
    [mapper.edge_type2idx[et] for et in edge_type_strs], device=device
)

print(f"  Edge types (first 10): {edge_type_strs[:10]}")
print(f"  Edge type indices (first 10): {edge_type_idx[:10].tolist()}")

# =============================================================================
# DEFINE MINIMAL NETWORK
# =============================================================================
print("\n[NETWORK] Defining minimal E(3)-equivariant network...")

hidden_irreps = Irreps(
    f"{CONFIG['hidden_dim']}x0e + {CONFIG['hidden_dim']}x1o + {CONFIG['hidden_dim']}x2e"
)
print(f"  Hidden irreps: {hidden_irreps}")


class MinimalNodeEncoder(nn.Module):
    """Encode element type to scalar features."""

    def __init__(self, num_elements, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_elements, hidden_dim)
        self.irreps_out = Irreps(f"{hidden_dim}x0e")
        print(
            f"    [NodeEncoder] Input: {num_elements} elements → Output: {self.irreps_out}"
        )

    def forward(self, node_type_idx):
        out = self.embedding(node_type_idx)
        print(
            f"      [NodeEncoder.forward] Input shape: {node_type_idx.shape} → Output: {out.shape} (irreps: {self.irreps_out})"
        )
        return out


class MinimalEdgeEncoder(nn.Module):
    """Encode edge distance + edge type + spherical harmonics."""

    def __init__(self, n_radial, num_edge_types, hidden_irreps, sh_irreps):
        super().__init__()
        self.irreps_out = hidden_irreps
        self.num_edge_types = num_edge_types

        # Linear projection from radial basis + edge type one-hot to scalars
        from e3nn.o3 import Irrep

        scalar_dim = sum(mul for mul, ir in hidden_irreps if ir == Irrep("0e"))
        self.radial_proj = nn.Linear(n_radial + num_edge_types, scalar_dim)

        # Tensor product: scalars ⊗ SH → hidden_irreps
        self.tp = FullyConnectedTensorProduct(
            Irreps(f"{scalar_dim}x0e"),
            sh_irreps,
            hidden_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        print(
            f"    [EdgeEncoder] Irreps in: radial({n_radial}) + edge_type({num_edge_types}) → scalars({scalar_dim}x0e)"
        )
        print(
            f"                  TP: {scalar_dim}x0e ⊗ {sh_irreps} → Irreps out: {hidden_irreps}"
        )

    def forward(self, edge_length_emb, edge_type_idx, edge_sh):
        # Create edge type one-hot
        edge_type_onehot = F.one_hot(
            edge_type_idx, num_classes=self.num_edge_types
        ).float()

        # Concatenate radial and edge type features
        combined = torch.cat([edge_length_emb, edge_type_onehot], dim=-1)

        # Project to scalars
        radial_feat = self.radial_proj(combined)  # (E, scalar_dim)

        # Tensor product with spherical harmonics
        edge_feat = self.tp(radial_feat, edge_sh)
        print(
            f"      [EdgeEncoder.forward] Radial: {edge_length_emb.shape}, EdgeType: {edge_type_onehot.shape} → Combined: {combined.shape}"
        )
        print(
            f"                            → Scalars: {radial_feat.shape} (irreps: {self.tp.irreps_in1})"
        )
        print(
            f"                            ⊗ SH: {edge_sh.shape} (irreps: {self.tp.irreps_in2})"
        )
        print(
            f"                            → Output: {edge_feat.shape} (irreps: {self.irreps_out})"
        )
        return edge_feat


class MinimalMessageBlock(nn.Module):
    """Message passing layer with edge update + node update (like DeepH-E3/E3GNN)."""

    def __init__(self, node_irreps, edge_irreps, hidden_irreps, sh_irreps):
        super().__init__()
        self.node_irreps = node_irreps
        self.edge_irreps = edge_irreps
        self.hidden_irreps = hidden_irreps

        # Edge update: concat(src_node, dst_node, edge) ⊗ SH → new edge
        concat_irreps = node_irreps + node_irreps + edge_irreps
        self.edge_update_tp = FullyConnectedTensorProduct(
            concat_irreps,
            sh_irreps,
            hidden_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        # Node update: aggregate messages + self-connection
        self.node_update_lin = Linear(hidden_irreps + node_irreps, hidden_irreps)

        print(f"    [MessageBlock] Irreps:")
        print(
            f"      Edge update: concat({node_irreps}, {node_irreps}, {edge_irreps}) ⊗ {sh_irreps}"
        )
        print(f"                   → Irreps out: {hidden_irreps}")
        print(
            f"      Node update: concat(messages {hidden_irreps}, self {node_irreps})"
        )
        print(f"                   → Irreps out: {hidden_irreps}")

    def forward(self, node_feat, edge_feat, edge_index, edge_sh):
        N = node_feat.shape[0]

        print(
            f"      [MessageBlock.forward] Input: nodes {node_feat.shape} (irreps: {self.node_irreps}), edges {edge_feat.shape} (irreps: {self.edge_irreps})"
        )

        # Edge update: concatenate src node, dst node, and edge features
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        src_node = node_feat[src_idx]
        dst_node = node_feat[dst_idx]

        edge_concat = torch.cat([src_node, dst_node, edge_feat], dim=-1)
        print(
            f"                            Edge concat: src {src_node.shape} + dst {dst_node.shape} + edge {edge_feat.shape} → {edge_concat.shape}"
        )

        # Apply TP with spherical harmonics
        edge_feat_new = self.edge_update_tp(edge_concat, edge_sh)
        print(
            f"                            Edge TP: {edge_concat.shape} ⊗ {edge_sh.shape} → {edge_feat_new.shape} (irreps: {self.hidden_irreps})"
        )

        # Node update: aggregate edge messages + self-connection
        messages = torch.zeros(N, edge_feat_new.shape[1], device=edge_feat_new.device)
        messages.index_add_(0, dst_idx, edge_feat_new)
        print(
            f"                            Messages aggregated: {messages.shape} (irreps: {self.hidden_irreps})"
        )

        # Concatenate with self-connection
        node_concat = torch.cat([messages, node_feat], dim=-1)
        node_feat_new = self.node_update_lin(node_concat)
        print(
            f"                            Node update: {node_concat.shape} → {node_feat_new.shape} (irreps: {self.hidden_irreps})"
        )

        return node_feat_new, edge_feat_new


class MinimalHead(nn.Module):
    """Predict irrep vectors for each edge type."""

    def __init__(self, hidden_irreps, mapper):
        super().__init__()
        self.mapper = mapper

        # One linear projection per edge type
        self.projections = nn.ModuleDict()
        for edge_type in mapper.edge_types:
            pair_irreps = mapper.get_pair_irreps(edge_type)
            self.projections[edge_type] = Linear(hidden_irreps, pair_irreps)
            print(f"    [Head] {edge_type}: {hidden_irreps} → {pair_irreps}")

    def forward(self, edge_feat, edge_type_idx, edge_index, edge_shift):
        """
        Returns: dict[pair_key] → {"vectors": tensor, "edges": tensor}
        """
        outputs = {}

        for type_str, proj in self.projections.items():
            type_idx = self.mapper.edge_type2idx[type_str]
            mask = edge_type_idx == type_idx

            if mask.any():
                selected_feat = edge_feat[mask]
                pred_vectors = proj(selected_feat)

                # Build 5D edge tensor (sx, sy, sz, i, j)
                selected_edges = torch.cat(
                    [
                        edge_shift[:, mask],  # (3, E')
                        edge_index[:, mask],  # (2, E')
                    ],
                    dim=0,
                )  # (5, E')

                outputs[type_str] = {
                    "vectors": pred_vectors,
                    "edges": selected_edges,
                }
                print(
                    f"      [Head.forward] {type_str}: {mask.sum().item()} edges → vectors {pred_vectors.shape}"
                )

        return outputs


class MinimalNetwork(nn.Module):
    """Complete minimal network."""

    def __init__(
        self,
        num_elements,
        n_radial,
        num_edge_types,
        hidden_irreps,
        sh_irreps,
        num_layers,
        mapper,
    ):
        super().__init__()

        # Count scalar irreps correctly
        from e3nn.o3 import Irrep

        scalar_dim = sum(mul for mul, ir in hidden_irreps if ir == Irrep("0e"))

        self.node_enc = MinimalNodeEncoder(num_elements, scalar_dim)
        self.edge_enc = MinimalEdgeEncoder(
            n_radial, num_edge_types, hidden_irreps, sh_irreps
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_layers):
            node_irreps_in = self.node_enc.irreps_out if i == 0 else hidden_irreps
            edge_irreps_in = hidden_irreps
            self.mp_layers.append(
                MinimalMessageBlock(
                    node_irreps_in, edge_irreps_in, hidden_irreps, sh_irreps
                )
            )

        self.head = MinimalHead(hidden_irreps, mapper)

        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        node_type_idx,
        edge_type_idx,
        edge_index,
        edge_shift,
        edge_length_emb,
        edge_sh,
    ):
        print("    [Forward] Starting forward pass...")

        # Encode
        node_feat = self.node_enc(node_type_idx)
        edge_feat = self.edge_enc(edge_length_emb, edge_type_idx, edge_sh)

        # Message passing (both nodes and edges get updated)
        for i, mp_layer in enumerate(self.mp_layers):
            print(f"    [Forward] Message passing layer {i + 1}/{len(self.mp_layers)}")
            node_feat, edge_feat = mp_layer(node_feat, edge_feat, edge_index, edge_sh)

        # Use edge features for head
        head_feat = edge_feat

        # Head
        print(f"    [Forward] Applying head...")
        outputs = self.head(head_feat, edge_type_idx, edge_index, edge_shift)

        return outputs


# Instantiate network
print("\nInstantiating network...")
num_elements = len(orbital_cfg.elements())
num_edge_types = num_elements**2
network = MinimalNetwork(
    num_elements=num_elements,
    n_radial=CONFIG["n_radial"],
    num_edge_types=num_edge_types,
    hidden_irreps=hidden_irreps,
    sh_irreps=sh_irreps,
    num_layers=CONFIG["num_layers"],
    mapper=mapper,
).to(device)

print("\n✓ Network architecture complete!")

# =============================================================================
# TRAINING LOOP
# =============================================================================
print("\n" + "=" * 80)
print("TRAINING TO OVERFIT")
print("=" * 80)

optimizer = Adam(network.parameters(), lr=CONFIG["lr"])

# Training history
history = {
    "loss": [],
    "mse_H": [],
    "mae_H": [],
}

print(f"\nOptimizer: Adam(lr={CONFIG['lr']})")
print(f"Training for {CONFIG['num_epochs']} epochs...\n")

for epoch in range(CONFIG["num_epochs"]):
    network.train()
    optimizer.zero_grad()

    # Forward pass (suppress detailed logging during training)
    if epoch % CONFIG["log_interval"] == 0:
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch + 1}/{CONFIG['num_epochs']}")
        print(f"{'=' * 60}")

    # Temporarily suppress forward pass logging
    verbose = (epoch % CONFIG["log_interval"] == 0) and (
        epoch < 10 or epoch % (CONFIG["log_interval"] * 5) == 0
    )

    if not verbose:
        # Silence print by redirecting to nowhere temporarily
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    pred_raw = network(
        node_type_idx, edge_type_idx, edge_index, edge_shift, edge_length_emb, edge_sh
    )

    if not verbose:
        sys.stdout.close()
        sys.stdout = old_stdout

    # Wrap predictions into IrrepsBlockData then convert to matrix blocks
    from collections import Counter
    from data.block_matrix import IrrepsBlockData

    pair_vec_H = {}
    pair_edges_dict = {}
    lookup_dict = {}

    for key, payload in pred_raw.items():
        pair_vec_H[key] = payload["vectors"]
        pair_edges_dict[key] = payload["edges"]

        for idx, edge_5d in enumerate(payload["edges"].t()):
            sx, sy, sz, i, j = edge_5d.tolist()
            lookup_dict[(int(sx), int(sy), int(sz), int(i), int(j))] = (key, idx)

    pred_H_irreps = IrrepsBlockData(
        atoms=tuple(atoms_list),
        atom_counts=Counter(atoms_list),
        pair_vectors=pair_vec_H,
        pair_edges=pair_edges_dict,
        lookup=lookup_dict,
        orbital_cfg=orbital_cfg,
    )

    # Convert to matrix blocks (train_target = "matrix")
    pred_H_matrix = pred_H_irreps.to_blocks(mapper)

    # Compute loss on matrix blocks
    loss_H = 0.0

    for key in target_H_matrix.pair_blocks.keys():
        if key in pred_H_matrix.pair_blocks:
            # Match sizes (predictions might have more edges due to cutoff)
            pred_blocks = pred_H_matrix.pair_blocks[key]
            targ_blocks = target_H_matrix.pair_blocks[key]
            min_n = min(pred_blocks.shape[0], targ_blocks.shape[0])

            loss_H += F.mse_loss(pred_blocks[:min_n], targ_blocks[:min_n])

    # Total loss (only Hamiltonian)
    loss = loss_H

    # Backward
    loss.backward()
    optimizer.step()

    # Logging
    if epoch % CONFIG["log_interval"] == 0:
        # Compute MAE on matrix blocks
        mae_H = sum(
            torch.mean(
                torch.abs(
                    pred_H_matrix.pair_blocks[k][
                        : min(
                            pred_H_matrix.pair_blocks[k].shape[0],
                            target_H_matrix.pair_blocks[k].shape[0],
                        )
                    ]
                    - target_H_matrix.pair_blocks[k][
                        : min(
                            pred_H_matrix.pair_blocks[k].shape[0],
                            target_H_matrix.pair_blocks[k].shape[0],
                        )
                    ]
                )
            )
            for k in target_H_matrix.pair_blocks.keys()
            if k in pred_H_matrix.pair_blocks
        ).item()

        history["loss"].append(loss.item())
        history["mse_H"].append(loss_H.item())
        history["mae_H"].append(mae_H)

        # Log to console
        print(f"\n[METRICS]")
        print(f"  Loss (MSE): {loss.item():.6e}")
        print(f"  MAE H: {mae_H:.6e}")

        # Log to WandB
        wandb.log(
            {
                "epoch": epoch,
                "loss": loss.item(),
                "mse_H": loss_H.item(),
                "mae_H": mae_H,
            }
        )

        # Check for convergence
        if loss.item() < 1e-8:
            print(f"\n✓ Converged! Loss below 1e-8 at epoch {epoch + 1}")
            break

# =============================================================================
# FINAL EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

network.eval()
with torch.no_grad():
    pred_raw = network(
        node_type_idx, edge_type_idx, edge_index, edge_shift, edge_length_emb, edge_sh
    )

    # Reconstruct full predictions
    pair_vec_H = {}
    pair_edges_dict = {}
    lookup_dict = {}

    for key, payload in pred_raw.items():
        pair_vec_H[key] = payload["vectors"]
        pair_edges_dict[key] = payload["edges"]

        for idx, edge_5d in enumerate(payload["edges"].t()):
            sx, sy, sz, i, j = edge_5d.tolist()
            lookup_dict[(int(sx), int(sy), int(sz), int(i), int(j))] = (key, idx)

    pred_H_irreps = IrrepsBlockData(
        atoms=tuple(atoms_list),
        atom_counts=Counter(atoms_list),
        pair_vectors=pair_vec_H,
        pair_edges=pair_edges_dict,
        lookup=lookup_dict,
        orbital_cfg=orbital_cfg,
    )

    # Convert to blocks
    pred_H_matrix = pred_H_irreps.to_blocks(mapper)

    print("\n[FINAL PREDICTIONS - Hamiltonian]")
    final_metrics = {}
    for key in pred_H_matrix.pair_blocks.keys():
        pred_block = pred_H_matrix.pair_blocks[key]
        true_block = target_H_matrix.pair_blocks[key]
        min_n = min(pred_block.shape[0], true_block.shape[0])

        block_mse = F.mse_loss(pred_block[:min_n], true_block[:min_n]).item()
        block_mae = torch.mean(
            torch.abs(pred_block[:min_n] - true_block[:min_n])
        ).item()

        pair_irreps = mapper.get_pair_irreps(key)

        print(f"\n  {key}:")
        print(f"    Block shape: {pred_block.shape}, Irreps: {pair_irreps}")
        print(f"    MSE: {block_mse:.6e}")
        print(f"    MAE: {block_mae:.6e}")
        print(
            f"    Relative error: {block_mae / (torch.abs(true_block[:min_n]).mean().item() + 1e-10):.6%}"
        )

        final_metrics[f"final/{key}_mse"] = block_mse
        final_metrics[f"final/{key}_mae"] = block_mae

    # Log final metrics to WandB
    wandb.log(final_metrics)

print("\n" + "=" * 80)
print("STUDY COMPLETE")
print("=" * 80)
print(f"\nTotal training epochs: {epoch + 1}")
print(f"Final loss: {history['loss'][-1]:.6e}")
print(f"Best loss: {min(history['loss']):.6e}")
print("\n✓ Minimal overfit study finished successfully!")

# Finish WandB run
wandb.finish()
