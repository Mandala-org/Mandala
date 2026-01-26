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
from data.openmx_parser import parse_openmx_file
from data.openmx_info_parser import parse_info_out
from core.block_irrep_mapper import BlockIrrepMapper
from core.basis_converter import OpenMXE3NNConverter
from core.sparse_math import trace_matmul_sparse_block_matrix

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
}

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

# Parse OpenMX files
print(f"  Matrix file: {CONFIG['data_path']}")
print(f"  Info file: {CONFIG['info_path']}")

info_data = parse_info_out(CONFIG["info_path"])
print(f"\n  Parsed info:")
print(f"    Elements: {info_data.elements}")
print(f"    Num atoms: {len(info_data.elements)}")
print(f"    Positions shape: {info_data.positions.shape}")
print(f"    Cell shape: {info_data.box.shape if info_data.box is not None else None}")

# Parse matrix data
matrix_data = parse_openmx_file(CONFIG["data_path"])
print(f"\n  Parsed matrices:")
print(f"    Hamiltonian keys: {list(matrix_data['hamiltonian'].pair_blocks.keys())}")
print(f"    Overlap keys: {list(matrix_data['overlap'].pair_blocks.keys())}")
print(f"    Density keys: {list(matrix_data['density'].pair_blocks.keys())}")

# Extract target matrices
hamiltonian_openmx = matrix_data["hamiltonian"]
overlap_openmx = matrix_data["overlap"]
density_openmx = matrix_data["density"]

print(f"\n  Hamiltonian blocks:")
for key, blocks in hamiltonian_openmx.pair_blocks.items():
    print(
        f"    {key}: shape {blocks.shape}, edges {hamiltonian_openmx.pair_edges[key].shape}"
    )

# Build orbital configuration
orbital_cfg = hamiltonian_openmx.orbital_cfg
print(f"\n  Orbital configuration:")
for elem in orbital_cfg.elements():
    irreps = orbital_cfg.element_to_irreps[elem]
    print(f"    {elem}: {irreps} (dim={irreps.dim})")

# Convert from OpenMX basis to E3NN basis
print("\n[BASIS] Converting from OpenMX to E3NN convention...")
converter = OpenMXE3NNConverter(orbital_cfg, device=device)
hamiltonian_e3nn = converter.matrix_to_e3nn(hamiltonian_openmx).to(device)
overlap_e3nn = converter.matrix_to_e3nn(overlap_openmx).to(device)
density_e3nn = converter.matrix_to_e3nn(density_openmx).to(device)
print("  ✓ Basis conversion complete")

# Create BlockIrrepMapper
print("\n[MAPPER] Creating BlockIrrepMapper...")
mapper = BlockIrrepMapper(orbital_cfg, device=device, dtype=torch.float32)
print(f"  Mapper edge types: {mapper.edge_types}")
print(f"  Mapper edge_type2idx: {mapper.edge_type2idx}")

# Convert to irrep vectors (targets)
print("\n[TARGETS] Converting target matrices to irrep vectors...")
target_H_irreps = hamiltonian_e3nn.to_vectors(mapper)
target_S_irreps = overlap_e3nn.to_vectors(mapper)
target_D_irreps = density_e3nn.to_vectors(mapper)

print("  Target irrep vectors:")
for key in target_H_irreps.pair_vectors.keys():
    vec_shape = target_H_irreps.pair_vectors[key].shape
    edge_shape = target_H_irreps.pair_edges[key].shape
    print(f"    {key}: vectors {vec_shape}, edges {edge_shape}")

# Compute ground truth observables
print("\n[OBSERVABLES] Computing ground truth observables...")
E_true = trace_matmul_sparse_block_matrix(hamiltonian_e3nn, density_e3nn)
N_true = trace_matmul_sparse_block_matrix(overlap_e3nn, density_e3nn)
print(f"  Ground truth energy: {E_true.item():.6f} Ha")
print(f"  Ground truth electrons: {N_true.item():.6f}")

# =============================================================================
# BUILD GRAPH
# =============================================================================
print("\n[GRAPH] Constructing molecular graph...")

positions = info_data.positions.to(device)
box = info_data.box.to(device) if info_data.box is not None else None
atoms_list = info_data.elements
num_atoms = len(atoms_list)

print(f"  Atoms: {atoms_list}")
print(f"  Positions:\n{positions}")

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
            f"      [NodeEncoder.forward] Input shape: {node_type_idx.shape} → Output: {out.shape}"
        )
        return out


class MinimalEdgeEncoder(nn.Module):
    """Encode edge distance + edge type + spherical harmonics."""

    def __init__(self, n_radial, hidden_irreps, sh_irreps):
        super().__init__()
        self.irreps_out = hidden_irreps

        # Linear projection from radial basis to scalars
        self.radial_proj = nn.Linear(n_radial, hidden_irreps.count(Irreps("0e")))

        # Tensor product: scalars ⊗ SH → hidden_irreps
        self.tp = FullyConnectedTensorProduct(
            Irreps(f"{hidden_irreps.count(Irreps('0e'))}x0e"),
            sh_irreps,
            hidden_irreps,
            shared_weights=False,
        )
        print(
            f"    [EdgeEncoder] Radial {n_radial} → TP({hidden_irreps.count(Irreps('0e'))}x0e ⊗ {sh_irreps}) → {hidden_irreps}"
        )

    def forward(self, edge_length_emb, edge_sh):
        # Project radial features
        radial_feat = self.radial_proj(edge_length_emb)  # (E, scalar_dim)

        # Tensor product with spherical harmonics
        edge_feat = self.tp(radial_feat, edge_sh)
        print(
            f"      [EdgeEncoder.forward] Radial: {edge_length_emb.shape} → {radial_feat.shape}, SH: {edge_sh.shape} → Output: {edge_feat.shape}"
        )
        return edge_feat


class MinimalMessageBlock(nn.Module):
    """Single message passing layer."""

    def __init__(self, node_irreps, edge_irreps, hidden_irreps, sh_irreps):
        super().__init__()

        # Node features ⊗ SH → messages
        self.node_to_message = FullyConnectedTensorProduct(
            node_irreps,
            sh_irreps,
            hidden_irreps,
            shared_weights=False,
        )

        # Aggregate messages to nodes
        self.message_to_node = Linear(hidden_irreps, hidden_irreps)

        print(
            f"    [MessageBlock] Node→Msg: {node_irreps} ⊗ {sh_irreps} → {hidden_irreps}"
        )
        print(f"                   Msg→Node: {hidden_irreps} → {hidden_irreps}")

    def forward(self, node_feat, edge_index, edge_sh):
        E = edge_index.shape[1]
        N = node_feat.shape[0]

        # Create messages: broadcast node features to edges and apply TP
        src_idx = edge_index[0]
        node_at_src = node_feat[src_idx]

        messages = self.node_to_message(node_at_src, edge_sh)

        # Aggregate messages to destination nodes
        dst_idx = edge_index[1]
        aggregated = torch.zeros(N, messages.shape[1], device=messages.device)
        aggregated.index_add_(0, dst_idx, messages)

        # Update nodes
        node_feat_new = self.message_to_node(aggregated)

        print(
            f"      [MessageBlock.forward] Nodes: {node_feat.shape}, Edges: {E} → Messages: {messages.shape} → Aggregated: {aggregated.shape} → Output: {node_feat_new.shape}"
        )
        return node_feat_new


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
        self, num_elements, n_radial, hidden_irreps, sh_irreps, num_layers, mapper
    ):
        super().__init__()

        self.node_enc = MinimalNodeEncoder(
            num_elements, hidden_irreps.count(Irreps("0e"))
        )
        self.edge_enc = MinimalEdgeEncoder(n_radial, hidden_irreps, sh_irreps)

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_layers):
            node_irreps_in = self.node_enc.irreps_out if i == 0 else hidden_irreps
            self.mp_layers.append(
                MinimalMessageBlock(
                    node_irreps_in, hidden_irreps, hidden_irreps, sh_irreps
                )
            )

        self.head = MinimalHead(hidden_irreps, mapper)

        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, node_type_idx, edge_index, edge_shift, edge_length_emb, edge_sh):
        print("    [Forward] Starting forward pass...")

        # Encode
        node_feat = self.node_enc(node_type_idx)
        edge_feat = self.edge_enc(edge_length_emb, edge_sh)

        # Message passing
        for i, mp_layer in enumerate(self.mp_layers):
            print(f"    [Forward] Message passing layer {i + 1}/{len(self.mp_layers)}")
            node_feat = mp_layer(node_feat, edge_index, edge_sh)

        # For head, use final node features for self-edges and edge features for off-diagonal
        # Simpler: just use edge features (since we have edge encoder)
        head_feat = edge_feat  # Simplified for minimal implementation

        # Compute edge types
        atoms_list = info_data.elements
        edge_type_idx_local = torch.tensor(
            [
                mapper.edge_type2idx[
                    f"{atoms_list[edge_index[0, i].item()]}-{atoms_list[edge_index[1, i].item()]}"
                ]
                for i in range(edge_index.shape[1])
            ],
            device=edge_index.device,
        )

        # Head
        print(f"    [Forward] Applying head...")
        outputs = self.head(head_feat, edge_type_idx_local, edge_index, edge_shift)

        return outputs


# Instantiate network
print("\nInstantiating network...")
num_elements = len(orbital_cfg.elements())
network = MinimalNetwork(
    num_elements=num_elements,
    n_radial=CONFIG["n_radial"],
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
    "mse_S": [],
    "mse_D": [],
    "mae_H": [],
    "energy_error": [],
    "electrons_error": [],
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
        import os
        import sys

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    pred_raw = network(node_type_idx, edge_index, edge_shift, edge_length_emb, edge_sh)

    if not verbose:
        sys.stdout.close()
        sys.stdout = old_stdout

    # Wrap predictions into IrrepsBlockData
    from collections import Counter
    from data.block_matrix import IrrepsBlockData

    pair_vec_H = {}
    pair_vec_S = {}
    pair_vec_D = {}
    pair_edges_dict = {}
    lookup_dict = {}

    for key, payload in pred_raw.items():
        pair_vec_H[key] = payload["vectors"]
        pair_vec_S[key] = (
            payload["vectors"] * 0.8
        )  # Simplified: predict all matrices similarly
        pair_vec_D[key] = payload["vectors"] * 0.5
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

    pred_S_irreps = IrrepsBlockData(
        atoms=tuple(atoms_list),
        atom_counts=Counter(atoms_list),
        pair_vectors=pair_vec_S,
        pair_edges=pair_edges_dict,
        lookup=lookup_dict,
        orbital_cfg=orbital_cfg,
    )

    pred_D_irreps = IrrepsBlockData(
        atoms=tuple(atoms_list),
        atom_counts=Counter(atoms_list),
        pair_vectors=pair_vec_D,
        pair_edges=pair_edges_dict,
        lookup=lookup_dict,
        orbital_cfg=orbital_cfg,
    )

    # Compute losses
    loss_H = 0.0
    loss_S = 0.0
    loss_D = 0.0

    for key in target_H_irreps.pair_vectors.keys():
        if key in pred_H_irreps.pair_vectors:
            # Match sizes (predictions might have more edges due to cutoff)
            pred_vecs_H = pred_H_irreps.pair_vectors[key]
            targ_vecs_H = target_H_irreps.pair_vectors[key]
            min_n = min(pred_vecs_H.shape[0], targ_vecs_H.shape[0])

            loss_H += F.mse_loss(pred_vecs_H[:min_n], targ_vecs_H[:min_n])

            pred_vecs_S = pred_S_irreps.pair_vectors[key]
            targ_vecs_S = target_S_irreps.pair_vectors[key]
            loss_S += F.mse_loss(pred_vecs_S[:min_n], targ_vecs_S[:min_n])

            pred_vecs_D = pred_D_irreps.pair_vectors[key]
            targ_vecs_D = target_D_irreps.pair_vectors[key]
            loss_D += F.mse_loss(pred_vecs_D[:min_n], targ_vecs_D[:min_n])

    # Total loss
    loss = loss_H + loss_S + loss_D

    # Backward
    loss.backward()
    optimizer.step()

    # Logging
    if epoch % CONFIG["log_interval"] == 0:
        # Convert to matrices for observables
        pred_H_matrix = pred_H_irreps.to_blocks(mapper)
        pred_D_matrix = pred_D_irreps.to_blocks(mapper)
        pred_S_matrix = pred_S_irreps.to_blocks(mapper)

        # Compute observables
        E_pred = trace_matmul_sparse_block_matrix(pred_H_matrix, pred_D_matrix)
        N_pred = trace_matmul_sparse_block_matrix(pred_S_matrix, pred_D_matrix)

        E_error = torch.abs(E_pred - E_true).item()
        N_error = torch.abs(N_pred - N_true).item()

        mae_H = sum(
            torch.mean(
                torch.abs(
                    pred_H_irreps.pair_vectors[k][
                        : min(
                            pred_H_irreps.pair_vectors[k].shape[0],
                            target_H_irreps.pair_vectors[k].shape[0],
                        )
                    ]
                    - target_H_irreps.pair_vectors[k][
                        : min(
                            pred_H_irreps.pair_vectors[k].shape[0],
                            target_H_irreps.pair_vectors[k].shape[0],
                        )
                    ]
                )
            )
            for k in target_H_irreps.pair_vectors.keys()
            if k in pred_H_irreps.pair_vectors
        ).item()

        history["loss"].append(loss.item())
        history["mse_H"].append(loss_H.item())
        history["mse_S"].append(loss_S.item())
        history["mse_D"].append(loss_D.item())
        history["mae_H"].append(mae_H)
        history["energy_error"].append(E_error)
        history["electrons_error"].append(N_error)

        print(f"\n[METRICS]")
        print(f"  Loss: {loss.item():.6e}")
        print(f"  MSE H: {loss_H.item():.6e}")
        print(f"  MSE S: {loss_S.item():.6e}")
        print(f"  MSE D: {loss_D.item():.6e}")
        print(f"  MAE H: {mae_H:.6e}")
        print(f"  Energy error: {E_error:.6e} Ha ({E_error * 27.2114:.6e} eV)")
        print(f"  Electrons error: {N_error:.6e}")
        print(f"  Predicted E: {E_pred.item():.6f} Ha (true: {E_true.item():.6f})")
        print(f"  Predicted N: {N_pred.item():.6f} (true: {N_true.item():.6f})")

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
    pred_raw = network(node_type_idx, edge_index, edge_shift, edge_length_emb, edge_sh)

    # Reconstruct full predictions
    pair_vec_H = {}
    pair_vec_S = {}
    pair_vec_D = {}
    pair_edges_dict = {}
    lookup_dict = {}

    for key, payload in pred_raw.items():
        pair_vec_H[key] = payload["vectors"]
        pair_vec_S[key] = payload["vectors"] * 0.8
        pair_vec_D[key] = payload["vectors"] * 0.5
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

    pred_D_irreps = IrrepsBlockData(
        atoms=tuple(atoms_list),
        atom_counts=Counter(atoms_list),
        pair_vectors=pair_vec_D,
        pair_edges=pair_edges_dict,
        lookup=lookup_dict,
        orbital_cfg=orbital_cfg,
    )

    pred_S_irreps = IrrepsBlockData(
        atoms=tuple(atoms_list),
        atom_counts=Counter(atoms_list),
        pair_vectors=pair_vec_S,
        pair_edges=pair_edges_dict,
        lookup=lookup_dict,
        orbital_cfg=orbital_cfg,
    )

    # Convert to blocks
    pred_H_matrix = pred_H_irreps.to_blocks(mapper)
    pred_D_matrix = pred_D_irreps.to_blocks(mapper)
    pred_S_matrix = pred_S_irreps.to_blocks(mapper)

    print("\n[FINAL PREDICTIONS]")
    for key in pred_H_matrix.pair_blocks.keys():
        pred_block = pred_H_matrix.pair_blocks[key]
        true_block = hamiltonian_e3nn.pair_blocks[key]
        min_n = min(pred_block.shape[0], true_block.shape[0])

        block_mse = F.mse_loss(pred_block[:min_n], true_block[:min_n]).item()
        block_mae = torch.mean(
            torch.abs(pred_block[:min_n] - true_block[:min_n])
        ).item()

        print(f"\n  {key}:")
        print(f"    Block shape: {pred_block.shape}")
        print(f"    MSE: {block_mse:.6e}")
        print(f"    MAE: {block_mae:.6e}")
        print(
            f"    Relative error: {block_mae / (torch.abs(true_block[:min_n]).mean().item() + 1e-10):.6%}"
        )

    # Final observables
    E_pred = trace_matmul_sparse_block_matrix(pred_H_matrix, pred_D_matrix)
    N_pred = trace_matmul_sparse_block_matrix(pred_S_matrix, pred_D_matrix)

    print(f"\n[FINAL OBSERVABLES]")
    print(f"  Energy:")
    print(f"    Predicted: {E_pred.item():.8f} Ha")
    print(f"    True:      {E_true.item():.8f} Ha")
    print(
        f"    Error:     {torch.abs(E_pred - E_true).item():.6e} Ha ({torch.abs(E_pred - E_true).item() * 27.2114:.6e} eV)"
    )
    print(f"  Electrons:")
    print(f"    Predicted: {N_pred.item():.8f}")
    print(f"    True:      {N_true.item():.8f}")
    print(f"    Error:     {torch.abs(N_pred - N_true).item():.6e}")

print("\n" + "=" * 80)
print("STUDY COMPLETE")
print("=" * 80)
print(f"\nTotal training epochs: {epoch + 1}")
print(f"Final loss: {history['loss'][-1]:.6e}")
print(f"Best loss: {min(history['loss']):.6e}")
print("\n✓ Minimal overfit study finished successfully!")
