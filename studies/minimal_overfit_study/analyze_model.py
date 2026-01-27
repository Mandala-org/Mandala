"""
Analyze Trained Minimal E(3)-Equivariant Model
==============================================

This script loads a trained model and performs detailed analysis:
1. Evaluates on original water structure
2. Evaluates on rotated water structure (to test equivariance)
3. Computes metrics: MAE, MSE, and modified MAE with mu_H correction
4. Visualizes Hamiltonians for different [sx, sy, sz] shifts

The modified MAE uses a correction factor mu_H to account for systematic shifts:
    MAE_mod = |H_pred - H_gt - mu_H * S|
    where mu_H = sum_{ij} (H_pred_ij - H_gt_ij) * S_ij / sum_{ij} S_ij^2
"""

import sys
import os
from pathlib import Path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.math import soft_one_hot_linspace
from ase import Atoms
from ase.neighborlist import neighbor_list

from data.snapshot import Snapshot
from data.block_matrix import IrrepsBlockData
from core.block_irrep_mapper import BlockIrrepMapper

# Import network classes and utilities from common module
from common import (
    MinimalNetwork,
    compute_mu_H,
)

print("=" * 80)
print("MINIMAL MODEL ANALYSIS")
print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze trained minimal E(3)-GNN model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint or run directory",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default="best_model.pt",
        help="Model filename (best_model.pt or final_model.pt)",
    )
    parser.add_argument(
        "--k-range",
        type=int,
        default=1,
        help="Range for [sx, sy, sz] visualization: [-k, k]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for saving analysis outputs (defaults to checkpoint_dir/analysis)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--dynamic-range",
        action="store_true",
        help="Use dynamic color range based on 95th percentile of abs(H) instead of fixed [-1, 1]",
    )
    return parser.parse_args()


def compute_metrics(H_pred, H_gt, S):
    """Compute MAE, MSE, and modified MAE with mu_H correction."""

    # Standard MAE and MSE
    mae = 0.0
    mse = 0.0
    total_elements = 0

    for key in H_gt.pair_blocks.keys():
        if key in H_pred.pair_blocks:
            pred_blocks = H_pred.pair_blocks[key]
            gt_blocks = H_gt.pair_blocks[key]
            min_n = min(pred_blocks.shape[0], gt_blocks.shape[0])

            diff = pred_blocks[:min_n] - gt_blocks[:min_n]
            mae += torch.sum(torch.abs(diff)).item()
            mse += torch.sum(diff**2).item()
            total_elements += diff.numel()

    mae /= total_elements
    mse /= total_elements

    # Modified MAE with mu_H correction
    mu_H = compute_mu_H(H_pred, H_gt, S)

    mae_mod = 0.0
    for key in H_gt.pair_blocks.keys():
        if key in H_pred.pair_blocks and key in S.pair_blocks:
            pred_blocks = H_pred.pair_blocks[key]
            gt_blocks = H_gt.pair_blocks[key]
            s_blocks = S.pair_blocks[key]

            min_n = min(pred_blocks.shape[0], gt_blocks.shape[0], s_blocks.shape[0])

            diff_corrected = (
                pred_blocks[:min_n] - gt_blocks[:min_n] - mu_H * s_blocks[:min_n]
            )
            mae_mod += torch.sum(torch.abs(diff_corrected)).item()

    mae_mod /= total_elements

    return {
        "mae": mae,
        "mse": mse,
        "mae_mod": mae_mod,
        "mu_H": mu_H,
    }


def build_graph_inputs(snapshot, cutoff_radius, device):
    """Build graph inputs from a snapshot (same as in training script)."""

    positions = snapshot.positions.to(device)
    box = snapshot.box.to(device) if snapshot.box is not None else None
    atoms_list = list(snapshot.hamiltonian.atoms)
    num_atoms = len(atoms_list)
    orbital_cfg = snapshot.hamiltonian.orbital_cfg

    # Create ASE atoms for neighbor list
    ase_atoms = Atoms(
        symbols=atoms_list,
        positions=positions.cpu().numpy(),
        cell=box.cpu().numpy() if box is not None else None,
        pbc=box is not None,
    )

    # Find neighbors
    src, dst, offsets = neighbor_list(
        "ijS", ase_atoms, cutoff_radius, self_interaction=False
    )

    # Add self-edges
    self_src = torch.arange(num_atoms, dtype=torch.long, device=device)
    self_dst = torch.arange(num_atoms, dtype=torch.long, device=device)
    self_offsets = torch.zeros((num_atoms, 3), dtype=torch.long, device=device)

    # Combine edges
    all_src = torch.cat([self_src, torch.from_numpy(src).to(device)])
    all_dst = torch.cat([self_dst, torch.from_numpy(dst).to(device)])
    all_offsets = torch.cat([self_offsets, torch.from_numpy(offsets).to(device).long()])

    edge_index = torch.stack([all_src, all_dst], dim=0)
    edge_shift = all_offsets.T

    # Compute edge vectors and distances
    if box is not None:
        shift_float = edge_shift.T.float()
        edge_vec = (
            positions[edge_index[1]] - positions[edge_index[0]] + shift_float @ box
        )
    else:
        edge_vec = positions[edge_index[1]] - positions[edge_index[0]]

    edge_dist = torch.linalg.norm(edge_vec, dim=1)

    # Spherical harmonics
    sh_irreps = Irreps.spherical_harmonics(2)  # l_max=2
    edge_vec_norm = edge_vec.clone()
    non_zero_mask = edge_dist > 1e-6
    edge_vec_norm[non_zero_mask] = edge_vec[non_zero_mask] / edge_dist[
        non_zero_mask
    ].unsqueeze(-1)
    edge_sh = spherical_harmonics(sh_irreps, edge_vec_norm, normalize=False)

    # Radial basis
    edge_length_emb = soft_one_hot_linspace(
        edge_dist,
        start=0.0,
        end=cutoff_radius,
        number=16,  # n_radial
        basis="gaussian",
        cutoff=False,
    )
    edge_length_emb = edge_length_emb * 16**0.5

    # Edge type indices
    element_to_idx = {elem: idx for idx, elem in enumerate(orbital_cfg.elements())}
    node_type_idx = torch.tensor([element_to_idx[a] for a in atoms_list], device=device)

    # Create mapper for edge types
    mapper = BlockIrrepMapper(orbital_cfg, device=device, dtype=torch.float32)
    edge_type_strs = [
        f"{atoms_list[edge_index[0, i].item()]}-{atoms_list[edge_index[1, i].item()]}"
        for i in range(edge_index.shape[1])
    ]
    edge_type_idx = torch.tensor(
        [mapper.edge_type2idx[et] for et in edge_type_strs], device=device
    )

    return {
        "node_type_idx": node_type_idx,
        "edge_type_idx": edge_type_idx,
        "edge_index": edge_index,
        "edge_shift": edge_shift,
        "edge_length_emb": edge_length_emb,
        "edge_sh": edge_sh,
        "atoms_list": atoms_list,
        "orbital_cfg": orbital_cfg,
        "mapper": mapper,
    }


def predict_hamiltonian(network, graph_inputs, device):
    """Run model forward pass and convert to BlockMatrix."""

    network.eval()
    with torch.no_grad():
        # Suppress print statements during forward pass
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        pred_raw = network(
            graph_inputs["node_type_idx"],
            graph_inputs["edge_type_idx"],
            graph_inputs["edge_index"],
            graph_inputs["edge_shift"],
            graph_inputs["edge_length_emb"],
            graph_inputs["edge_sh"],
        )

        sys.stdout.close()
        sys.stdout = old_stdout

    # Convert to IrrepsBlockData
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
        atoms=tuple(graph_inputs["atoms_list"]),
        atom_counts=Counter(graph_inputs["atoms_list"]),
        pair_vectors=pair_vec_H,
        pair_edges=pair_edges_dict,
        lookup=lookup_dict,
        orbital_cfg=graph_inputs["orbital_cfg"],
    )

    # Convert to matrix blocks
    pred_H_matrix = pred_H_irreps.to_blocks(graph_inputs["mapper"])

    return pred_H_matrix


def extract_partial_hamiltonian(H, S, atoms_list, orbital_cfg, sx, sy, sz):
    """
    Extract a partial Hamiltonian matrix for a specific [sx, sy, sz] shift.

    Returns a dense matrix where blocks are filled in based on atom types.
    """

    # Compute total matrix size
    orbital_dims = [orbital_cfg.element_to_irreps[atom].dim for atom in atoms_list]
    total_dim = sum(orbital_dims)

    # Create dense matrices
    H_dense = np.zeros((total_dim, total_dim))
    S_dense = np.zeros((total_dim, total_dim)) if S is not None else None

    # Build index map: atom_idx -> (start_row, end_row)
    atom_ranges = []
    current_idx = 0
    for dim in orbital_dims:
        atom_ranges.append((current_idx, current_idx + dim))
        current_idx += dim

    # Fill in blocks for this specific shift
    for key in H.pair_blocks.keys():
        H_blocks = H.pair_blocks[key]
        H_edges = H.pair_edges[key]

        if S is not None and key in S.pair_blocks:
            S_blocks = S.pair_blocks[key]
        else:
            S_blocks = None

        # Find edges with the specified shift
        for idx in range(H_edges.shape[1]):
            shift_x, shift_y, shift_z, i, j = H_edges[:, idx].tolist()

            if int(shift_x) == sx and int(shift_y) == sy and int(shift_z) == sz:
                # Get block
                block_H = H_blocks[idx].cpu().numpy()

                # Get atom ranges
                i_start, i_end = atom_ranges[int(i)]
                j_start, j_end = atom_ranges[int(j)]

                # Fill in matrix
                H_dense[i_start:i_end, j_start:j_end] = block_H

                # Fill overlap if available
                if S_blocks is not None:
                    block_S = S_blocks[idx].cpu().numpy()
                    S_dense[i_start:i_end, j_start:j_end] = block_S

    return H_dense, S_dense, total_dim


def visualize_hamiltonians(
    H_pred, H_gt, S, atoms_list, orbital_cfg, k_range, output_dir, dynamic_range=False
):
    """
    Visualize Hamiltonians for all [sx, sy, sz] combinations in [-k, k]^3.

    For each shift, show:
    - Ground truth H
    - Predicted H
    - Difference (pred - gt)
    - Difference with mu_H correction

    Args:
        dynamic_range: If True, use 95th percentile of abs(H_gt) for color range.
                      If False, use fixed [-1, 1] range.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute mu_H
    mu_H = compute_mu_H(H_pred, H_gt, S)

    print(f"\nVisualizing Hamiltonians for shifts in [{-k_range}, {k_range}]^3")
    print(f"mu_H correction factor: {mu_H:.6e}\n")

    # Iterate over all shifts
    shifts_to_plot = []
    for sx in range(-k_range, k_range + 1):
        for sy in range(-k_range, k_range + 1):
            for sz in range(-k_range, k_range + 1):
                shifts_to_plot.append((sx, sy, sz))

    for sx, sy, sz in shifts_to_plot:
        print(f"Processing shift [{sx}, {sy}, {sz}]...")

        # Extract partial Hamiltonians
        H_gt_dense, S_dense, dim = extract_partial_hamiltonian(
            H_gt, S, atoms_list, orbital_cfg, sx, sy, sz
        )
        H_pred_dense, _, _ = extract_partial_hamiltonian(
            H_pred, None, atoms_list, orbital_cfg, sx, sy, sz
        )

        # Compute differences
        diff = H_pred_dense - H_gt_dense
        diff_corrected = diff - mu_H * S_dense if S_dense is not None else diff

        # Check if there's any data for this shift
        if np.abs(H_gt_dense).max() < 1e-10 and np.abs(H_pred_dense).max() < 1e-10:
            print(f"  Skipping (no data for this shift)")
            continue

        # Create figure with 2x2 subplots (skip overlap)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"Hamiltonian Analysis: shift = [{sx}, {sy}, {sz}]", fontsize=16)

        # Determine color range
        if dynamic_range:
            # Use 95th percentile of abs(H_gt)
            v = np.percentile(np.abs(H_gt_dense), 95)
            vmin, vmax = -v, v
        else:
            vmin, vmax = -1, 1

        # 1. Ground truth (top-left)
        im0 = axes[0, 0].imshow(H_gt_dense, cmap="bwr", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"Ground Truth H\nMax: {np.abs(H_gt_dense).max():.3f}")
        axes[0, 0].set_xlabel("Orbital j")
        axes[0, 0].set_ylabel("Orbital i")
        plt.colorbar(im0, ax=axes[0, 0])

        # 2. Predicted (top-right)
        im1 = axes[0, 1].imshow(H_pred_dense, cmap="bwr", vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(f"Predicted H\nMax: {np.abs(H_pred_dense).max():.3f}")
        axes[0, 1].set_xlabel("Orbital j")
        axes[0, 1].set_ylabel("Orbital i")
        plt.colorbar(im1, ax=axes[0, 1])

        # 3. Difference (bottom-left)
        im2 = axes[1, 0].imshow(diff, cmap="bwr", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f"Difference (pred - gt)\nMAE: {np.abs(diff).mean():.3e}")
        axes[1, 0].set_xlabel("Orbital j")
        axes[1, 0].set_ylabel("Orbital i")
        plt.colorbar(im2, ax=axes[1, 0])

        # 4. Corrected difference (bottom-right)
        im3 = axes[1, 1].imshow(diff_corrected, cmap="bwr", vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(
            f"Corrected Diff (µ_H={mu_H:.2e})\nMAE: {np.abs(diff_corrected).mean():.3e}"
        )
        axes[1, 1].set_xlabel("Orbital j")
        axes[1, 1].set_ylabel("Orbital i")
        plt.colorbar(im3, ax=axes[1, 1])

        # Save figure
        filename = f"hamiltonian_sx{sx:+d}_sy{sy:+d}_sz{sz:+d}.png"
        filepath = output_dir / filename
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved to {filepath}")


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Handle checkpoint path - can be a directory or a file
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_root / checkpoint_path

    # If it's a directory, look for the model file inside
    if checkpoint_path.is_dir():
        model_path = checkpoint_path / args.model_file
        run_name = checkpoint_path.name
        output_dir_default = checkpoint_path / "analysis"
    else:
        model_path = checkpoint_path
        run_name = checkpoint_path.stem
        output_dir_default = checkpoint_path.parent / "analysis"

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
    else:
        output_dir = output_dir_default

    print(f"\n[CONFIG]")
    print(f"  Checkpoint: {model_path}")
    print(f"  Run name: {run_name}")
    print(f"  Device: {device}")
    print(f"  K-range for visualization: [-{args.k_range}, {args.k_range}]")
    print(f"  Output directory: {output_dir}")

    # Load checkpoint
    print(f"\n[LOADING] Loading checkpoint...")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print(f"  Training loss: {checkpoint['loss']:.6e}")
    print(f"  Training MAE_H: {checkpoint['mae_H']:.6e}")

    # Load original data
    print(f"\n[DATA] Loading original water snapshot...")
    snapshot_orig = Snapshot.from_openmx(
        matrix_path=config["data_path"],
        info_path=config["info_path"],
        convention="e3nn",
        symmetrize_density=True,
        cutoff_radius=None,
        dtype=torch.float32,
    )

    print(f"  Atoms: {snapshot_orig.hamiltonian.atoms}")
    print(f"  Positions shape: {snapshot_orig.positions.shape}")

    # Create rotated snapshot (90 degrees around z-axis in e3nn frame)
    print(f"\n[ROTATION] Creating rotated snapshot...")
    angle = np.pi / 2  # 90 degrees
    R = torch.tensor(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    snapshot_rot = snapshot_orig.rotate(R)
    print(f"  Rotation matrix:\n{R}")

    # Build network
    print(f"\n[NETWORK] Reconstructing network architecture...")
    orbital_cfg = snapshot_orig.hamiltonian.orbital_cfg
    num_elements = len(orbital_cfg.elements())
    num_edge_types = num_elements**2
    hidden_irreps = Irreps(
        f"{config['hidden_dim']}x0e + {config['hidden_dim']}x1o + {config['hidden_dim']}x2e"
    )
    sh_irreps = Irreps.spherical_harmonics(config["l_max"])

    mapper = BlockIrrepMapper(orbital_cfg, device=device, dtype=torch.float32)

    network = MinimalNetwork(
        num_elements=num_elements,
        n_radial=config["n_radial"],
        num_edge_types=num_edge_types,
        hidden_irreps=hidden_irreps,
        sh_irreps=sh_irreps,
        num_layers=config["num_layers"],
        mapper=mapper,
    ).to(device)

    network.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"  Network loaded with {sum(p.numel() for p in network.parameters()):,} parameters"
    )

    # Build graph inputs for original
    print(f"\n[GRAPH] Building graph inputs for original structure...")
    graph_inputs_orig = build_graph_inputs(
        snapshot_orig, config["cutoff_radius"], device
    )
    print(f"  Total edges: {graph_inputs_orig['edge_index'].shape[1]}")

    # Build graph inputs for rotated
    print(f"\n[GRAPH] Building graph inputs for rotated structure...")
    graph_inputs_rot = build_graph_inputs(snapshot_rot, config["cutoff_radius"], device)
    print(f"  Total edges: {graph_inputs_rot['edge_index'].shape[1]}")

    # Predict on original
    print(f"\n[PREDICTION] Running inference on original structure...")
    H_pred_orig = predict_hamiltonian(network, graph_inputs_orig, device)
    H_gt_orig = snapshot_orig.hamiltonian.to(device)
    S_orig = snapshot_orig.overlap.to(device)

    # Predict on rotated
    print(f"\n[PREDICTION] Running inference on rotated structure...")
    H_pred_rot = predict_hamiltonian(network, graph_inputs_rot, device)
    H_gt_rot = snapshot_rot.hamiltonian.to(device)
    S_rot = snapshot_rot.overlap.to(device)

    # Compute metrics for original
    print(f"\n{'='*80}")
    print("ORIGINAL STRUCTURE METRICS")
    print(f"{'='*80}")
    metrics_orig = compute_metrics(H_pred_orig, H_gt_orig, S_orig)
    print(f"  MAE:       {metrics_orig['mae']:.6e}")
    print(f"  MSE:       {metrics_orig['mse']:.6e}")
    print(f"  MAE_mod:   {metrics_orig['mae_mod']:.6e}")
    print(f"  mu_H:      {metrics_orig['mu_H']:.6e}")

    # Compute metrics for rotated
    print(f"\n{'='*80}")
    print("ROTATED STRUCTURE METRICS")
    print(f"{'='*80}")
    metrics_rot = compute_metrics(H_pred_rot, H_gt_rot, S_rot)
    print(f"  MAE:       {metrics_rot['mae']:.6e}")
    print(f"  MSE:       {metrics_rot['mse']:.6e}")
    print(f"  MAE_mod:   {metrics_rot['mae_mod']:.6e}")
    print(f"  mu_H:      {metrics_rot['mu_H']:.6e}")

    # Visualize original
    print(f"\n{'='*80}")
    print("VISUALIZING ORIGINAL STRUCTURE")
    print(f"{'='*80}")
    output_dir_orig = output_dir / "original"
    visualize_hamiltonians(
        H_pred_orig,
        H_gt_orig,
        S_orig,
        list(snapshot_orig.hamiltonian.atoms),
        orbital_cfg,
        args.k_range,
        output_dir_orig,
        dynamic_range=args.dynamic_range,
    )

    # Visualize rotated
    print(f"\n{'='*80}")
    print("VISUALIZING ROTATED STRUCTURE")
    print(f"{'='*80}")
    output_dir_rot = output_dir / "rotated"
    visualize_hamiltonians(
        H_pred_rot,
        H_gt_rot,
        S_rot,
        list(snapshot_rot.hamiltonian.atoms),
        orbital_cfg,
        args.k_range,
        output_dir_rot,
        dynamic_range=args.dynamic_range,
    )

    # Save metrics to file
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("MINIMAL MODEL ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Checkpoint: {model_path}\n")
        f.write(f"Training epoch: {checkpoint['epoch']}\n")
        f.write(f"Training loss: {checkpoint['loss']:.6e}\n\n")
        f.write("ORIGINAL STRUCTURE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"MAE:       {metrics_orig['mae']:.6e}\n")
        f.write(f"MSE:       {metrics_orig['mse']:.6e}\n")
        f.write(f"MAE_mod:   {metrics_orig['mae_mod']:.6e}\n")
        f.write(f"mu_H:      {metrics_orig['mu_H']:.6e}\n\n")
        f.write("ROTATED STRUCTURE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"MAE:       {metrics_rot['mae']:.6e}\n")
        f.write(f"MSE:       {metrics_rot['mse']:.6e}\n")
        f.write(f"MAE_mod:   {metrics_rot['mae_mod']:.6e}\n")
        f.write(f"mu_H:      {metrics_rot['mu_H']:.6e}\n")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Visualizations saved to: {output_dir}")
    print("\n✓ Analysis finished successfully!")


if __name__ == "__main__":
    main()
