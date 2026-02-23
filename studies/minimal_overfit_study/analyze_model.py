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
import json
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
from net.common import build_hidden_irreps

# Import network classes and utilities from common module
from common import (
    MinimalNetwork,
    canonicalize_edge_order,
    compute_mu_H,
    compute_distance_error_curve,
    save_distance_error_curve_plot,
    filter_blocks_by_partial_train,
    split_hamiltonian_by_irrep,
    visualize_hamiltonians,
    permutation_to_matrix,
)

print("=" * 80)
print("MINIMAL MODEL ANALYSIS")
print("=" * 80)


# =============================================================================
# HELPER FUNCTIONS FOR PARTIAL TRAINING
# =============================================================================


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
        help="Use dynamic color range based on percentile of abs(H) instead of fixed [-1, 1]",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Percentile for dynamic color range (default: 99.0)",
    )
    parser.add_argument(
        "--split-by-irrep",
        action="store_true",
        help="Generate separate visualizations for each irrep's contribution to the Hamiltonian",
    )
    return parser.parse_args()


def compute_metrics(H_pred, H_gt, S, partial_train=None):
    """Compute MAE, MSE, and modified MAE with mu_H correction.

    Args:
        H_pred: Predicted Hamiltonian BlockMatrix
        H_gt: Ground truth Hamiltonian BlockMatrix
        S: Overlap BlockMatrix
        partial_train: "diag", "offdiag", or None - filters which blocks to evaluate
    """

    # Filter blocks based on partial_train setting
    filtered_pred = filter_blocks_by_partial_train(H_pred, partial_train)
    filtered_gt = filter_blocks_by_partial_train(H_gt, partial_train)
    filtered_s = filter_blocks_by_partial_train(S, partial_train)

    # Standard MAE and MSE
    mae = 0.0
    mse = 0.0
    total_elements = 0

    for key in H_gt.pair_blocks.keys():
        if key in H_pred.pair_blocks:
            pred_blocks_full = H_pred.pair_blocks[key]
            gt_blocks_full = H_gt.pair_blocks[key]
            _, pred_mask = filtered_pred[key]
            _, gt_mask = filtered_gt[key]

            min_n = min(pred_blocks_full.shape[0], gt_blocks_full.shape[0])
            mask = pred_mask[:min_n] & gt_mask[:min_n]

            if mask.any():
                pred_blocks = pred_blocks_full[:min_n][mask]
                gt_blocks = gt_blocks_full[:min_n][mask]

                diff = pred_blocks - gt_blocks
                mae += torch.sum(torch.abs(diff)).item()
                mse += torch.sum(diff**2).item()
                total_elements += diff.numel()

    if total_elements > 0:
        mae /= total_elements
        mse /= total_elements
    else:
        mae = 0.0
        mse = 0.0

    # Modified MAE with mu_H correction
    # Note: compute_mu_H should also respect filtering
    mu_H = compute_mu_H(H_pred, H_gt, S)  # This uses all blocks for mu_H estimation

    mae_mod = 0.0
    for key in H_gt.pair_blocks.keys():
        if key in H_pred.pair_blocks and key in S.pair_blocks:
            pred_blocks_full = H_pred.pair_blocks[key]
            gt_blocks_full = H_gt.pair_blocks[key]
            s_blocks_full = S.pair_blocks[key]

            _, pred_mask = filtered_pred[key]
            _, gt_mask = filtered_gt[key]
            _, s_mask = filtered_s[key]

            min_n = min(
                pred_blocks_full.shape[0],
                gt_blocks_full.shape[0],
                s_blocks_full.shape[0],
            )
            mask = pred_mask[:min_n] & gt_mask[:min_n] & s_mask[:min_n]

            if mask.any():
                pred_blocks = pred_blocks_full[:min_n][mask]
                gt_blocks = gt_blocks_full[:min_n][mask]
                s_blocks = s_blocks_full[:min_n][mask]

                diff_corrected = pred_blocks - gt_blocks - mu_H * s_blocks
                mae_mod += torch.sum(torch.abs(diff_corrected)).item()

    if total_elements > 0:
        mae_mod /= total_elements
    else:
        mae_mod = 0.0

    return {
        "mae": mae,
        "mse": mse,
        "mae_mod": mae_mod,
        "mu_H": mu_H,
    }


def compute_generalized_eigenvalues(H, S):
    """
    Compute generalized eigenvalues for H x = lambda S x using Cholesky reduction.

    Args:
        H: BlockMatrix Hamiltonian
        S: BlockMatrix overlap

    Returns:
        1D torch.Tensor of sorted eigenvalues
    """
    H_dense = H.to_dense().detach()
    S_dense = S.to_dense().detach()

    # Symmetrize to improve numerical robustness
    H_dense = 0.5 * (H_dense + H_dense.T)
    S_dense = 0.5 * (S_dense + S_dense.T)

    L = torch.linalg.cholesky(S_dense)
    tmp = torch.linalg.solve(L, H_dense)
    A = torch.linalg.solve(L, tmp.T).T  # A = L^{-1} H L^{-T}
    A = 0.5 * (A + A.T)
    return torch.linalg.eigvalsh(A)


def compute_dos_from_eigenvalues(eigenvalues, sigma, bin_width, E_min, E_max):
    """
    Compute DOS from eigenvalues via Gaussian broadening.
    """
    grid = torch.arange(
        E_min,
        E_max + bin_width,
        bin_width,
        dtype=eigenvalues.dtype,
        device=eigenvalues.device,
    )
    dos = torch.sum(
        torch.exp(-((grid[:, None] - eigenvalues[None, :]) ** 2) / (2 * sigma**2)),
        dim=1,
    ) / (
        torch.sqrt(
            torch.tensor(2 * torch.pi, dtype=eigenvalues.dtype, device=grid.device)
        )
        * sigma
    )
    return grid, dos


def evaluate_eigen_and_dos(H_pred, H_gt, S, output_dir, prefix):
    """
    Evaluate generalized-eigenvalue and DOS errors, and save DOS comparison plot.

    Returns:
        Dict with eigen and DOS metrics.
    """
    eig_pred = compute_generalized_eigenvalues(H_pred, S)
    eig_gt = compute_generalized_eigenvalues(H_gt, S)

    abs_err = torch.abs(eig_pred - eig_gt)
    rel_err = abs_err / (torch.abs(eig_gt) + 1e-12)

    eig_metrics = {
        "eig_abs_mean": float(abs_err.mean().item()),
        "eig_abs_max": float(abs_err.max().item()),
        "eig_rel_mean": float(rel_err.mean().item()),
        "eig_rel_max": float(rel_err.max().item()),
    }

    # Build a common DOS grid spanning both spectra with a small margin.
    eig_min = float(torch.min(torch.min(eig_pred), torch.min(eig_gt)).item())
    eig_max = float(torch.max(torch.max(eig_pred), torch.max(eig_gt)).item())
    span = max(eig_max - eig_min, 1e-6)
    margin = 0.1 * span + 0.05
    E_min = eig_min - margin
    E_max = eig_max + margin
    sigma = 0.2
    bin_width = 0.1

    grid, dos_pred = compute_dos_from_eigenvalues(
        eig_pred,
        sigma=sigma,
        bin_width=bin_width,
        E_min=E_min,
        E_max=E_max,
    )
    _, dos_gt = compute_dos_from_eigenvalues(
        eig_gt,
        sigma=sigma,
        bin_width=bin_width,
        E_min=E_min,
        E_max=E_max,
    )

    dos_diff = dos_pred - dos_gt
    dos_metrics = {
        "dos_mae": float(torch.mean(torch.abs(dos_diff)).item()),
        "dos_mse": float(torch.mean(dos_diff**2).item()),
        "dos_max_abs": float(torch.max(torch.abs(dos_diff)).item()),
        "dos_grid_min": float(grid[0].item()),
        "dos_grid_max": float(grid[-1].item()),
        "dos_grid_points": int(grid.shape[0]),
    }

    # Save DOS comparison plot
    dos_plot_path = output_dir / f"dos_comparison_{prefix}.png"
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(
        grid.detach().cpu().numpy(),
        dos_gt.detach().cpu().numpy(),
        label="DOS GT",
        linewidth=2.0,
    )
    ax.plot(
        grid.detach().cpu().numpy(),
        dos_pred.detach().cpu().numpy(),
        label="DOS Pred",
        linewidth=2.0,
        linestyle="--",
    )
    ax.set_xlabel("Energy")
    ax.set_ylabel("DOS")
    ax.set_title(f"DOS Comparison ({prefix})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dos_plot_path, dpi=150)
    plt.close(fig)

    return {
        **eig_metrics,
        **dos_metrics,
        "dos_plot_path": str(dos_plot_path),
    }


def build_graph_inputs(
    snapshot,
    cutoff_radius,
    n_radial,
    l_max,
    device,
    xyz_permutation="012",
    change_box="right",
):
    """Build graph inputs from a snapshot (same as in training script)."""

    positions = snapshot.positions.to(device)
    box = snapshot.box.to(device) if snapshot.box is not None else None
    atoms_list = list(snapshot.hamiltonian.atoms)
    num_atoms = len(atoms_list)
    orbital_cfg = snapshot.hamiltonian.orbital_cfg

    # Match training behavior: optional coordinate permutation on graph geometry only
    if xyz_permutation != "012":
        cob_matrix = permutation_to_matrix(xyz_permutation, device)
        positions = positions @ cob_matrix.T
        if box is not None:
            if change_box == "right":
                box = box @ cob_matrix.T
            elif change_box == "left":
                box = cob_matrix @ box
            elif change_box == "both":
                box = cob_matrix @ box @ cob_matrix.T
            else:
                raise ValueError(f"Invalid change_box option: {change_box}")

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

    # Canonicalize edge order exactly as in training script.
    edge_index, edge_shift, _ = canonicalize_edge_order(
        edge_index=edge_index,
        edge_shift=edge_shift,
        positions=positions,
        box=box,
    )

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
    sh_irreps = Irreps.spherical_harmonics(l_max)
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
        number=n_radial,
        basis="gaussian",
        cutoff=False,
    )
    edge_length_emb = edge_length_emb * n_radial**0.5

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
        "positions_used": positions,
        "box_used": box,
        "atoms_list": atoms_list,
        "orbital_cfg": orbital_cfg,
        "mapper": mapper,
    }


def get_all_irreps_in_hamiltonian(mapper):
    """
    Get a list of all unique irreps present in the Hamiltonian.

    Args:
        mapper: BlockIrrepMapper

    Returns:
        List of unique Irrep objects
    """

    irreps_set = set()
    for edge_type in mapper.edge_types:
        pair_irreps = mapper.get_pair_irreps(edge_type)
        for mul, irrep in pair_irreps:
            if mul > 0:  # Only include irreps that are actually present
                irreps_set.add(irrep)

    # Sort by l, then by parity
    return sorted(irreps_set, key=lambda ir: (ir.l, ir.p))


def predict_hamiltonian(network, graph_inputs, device, magnitude_factorization=False):
    """Run model forward pass and convert to BlockMatrix."""

    network.eval()
    with torch.no_grad():
        # Suppress print statements during forward pass
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

        # Create batch indices (single graph)
        num_nodes = graph_inputs["node_type_idx"].shape[0]
        num_edges = graph_inputs["edge_index"].shape[1]
        batch_node = torch.zeros(num_nodes, dtype=torch.long, device=device)
        batch_edge = torch.zeros(num_edges, dtype=torch.long, device=device)

        pred_raw = network(
            graph_inputs["node_type_idx"],
            graph_inputs["edge_type_idx"],
            graph_inputs["edge_index"],
            graph_inputs["edge_shift"],
            graph_inputs["edge_length_emb"],
            graph_inputs["edge_sh"],
            batch_node,
            batch_edge,
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
    if magnitude_factorization:
        reconstructed = {}
        for key, blocks in pred_H_matrix.pair_blocks.items():
            if key not in pred_raw or "magnitudes" not in pred_raw[key]:
                raise RuntimeError(
                    f"Missing predicted magnitudes for key '{key}' in magnitude-factorization mode."
                )
            mags = pred_raw[key]["magnitudes"]
            if mags.shape[0] != blocks.shape[0]:
                raise RuntimeError(
                    f"Magnitude/prediction edge-count mismatch for key '{key}': "
                    f"{mags.shape[0]} vs {blocks.shape[0]}"
                )
            scale = mags
            while scale.ndim < blocks.ndim:
                scale = scale.unsqueeze(-1)
            reconstructed[key] = blocks * scale

        pred_H_matrix = pred_H_matrix._replace_pair_blocks(
            reconstructed, basis=pred_H_matrix.basis
        )

    return pred_H_matrix


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
    output_dir.mkdir(parents=True, exist_ok=True)

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
    partial_train = config.get("partial_train", None)  # Extract partial_train setting

    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print(f"  Training loss: {checkpoint['loss']:.6e}")
    print(f"  Training MAE_H: {checkpoint['mae_H']:.6e}")
    if partial_train is not None:
        print(f"  Partial training mode: {partial_train}")

    # Load original data
    print(f"\n[DATA] Loading original water snapshot...")
    matrix_path = config.get("data_path", "data/small/H2O/original/H2O.matrix")
    info_path = config.get("info_path", "data/small/H2O/original/H2O.info.out")
    snapshot_orig = Snapshot.from_openmx(
        matrix_path=matrix_path,
        info_path=info_path,
        convention="e3nn",
        symmetrize_density=True,
        cutoff_radius=None,
        dtype=torch.float32,
    )

    # Match training behavior: optional target cutoff filtering
    if bool(config.get("apply_cutoff_to_targets", False)):
        cutoff_radius_cfg = float(config["cutoff_radius"])
        snapshot_orig = snapshot_orig.filter_by_distance(cutoff_radius_cfg)
        print(
            f"  Applied cutoff to targets in analysis: cutoff_radius={cutoff_radius_cfg} Å"
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
    print(f"\n[NETWORK] Reconstructing network architecture from config...")
    orbital_cfg = snapshot_orig.hamiltonian.orbital_cfg
    num_elements = len(orbital_cfg.elements())
    num_edge_types = num_elements**2

    # Reconstruct irreps using the same function as training
    print(
        f"  Config: l_max={config['l_max']}, hidden_dim={config['hidden_dim']}, n_radial={config['n_radial']}, num_layers={config['num_layers']}"
    )

    # Use hidden_irreps from config if available, otherwise build from scratch
    if "hidden_irreps" in config and config["hidden_irreps"] is not None:
        hidden_irreps = Irreps(config["hidden_irreps"])
        print(f"  Using hidden_irreps from config: {hidden_irreps}")
    else:
        hidden_irreps = build_hidden_irreps(
            l_max=config["l_max"], base_dim=config["hidden_dim"], use_odd_features=True
        )
        print(f"  Built hidden_irreps from l_max and hidden_dim: {hidden_irreps}")
    sh_irreps = Irreps.spherical_harmonics(config["l_max"])

    print(f"  SH irreps: {sh_irreps}")

    mapper = BlockIrrepMapper(orbital_cfg, device=device, dtype=torch.float32)

    network = MinimalNetwork(
        num_elements=num_elements,
        n_radial=config["n_radial"],
        num_edge_types=num_edge_types,
        hidden_irreps=hidden_irreps,
        sh_irreps=sh_irreps,
        num_layers=config["num_layers"],
        mapper=mapper,
        magnitude_factorization=config.get("magnitude_factorization", False),
        head_mlp_for_scalars=config.get("head_mlp_for_scalars", False),
        verbose=False,  # Disable verbose output during analysis
    ).to(device)

    network.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"  Network loaded with {sum(p.numel() for p in network.parameters()):,} parameters"
    )

    # Build graph inputs for original
    print(f"\n[GRAPH] Building graph inputs for original structure...")
    graph_inputs_orig = build_graph_inputs(
        snapshot_orig,
        config["cutoff_radius"],
        config["n_radial"],
        config["l_max"],
        device,
        xyz_permutation=config.get("xyz_permutation", "012"),
        change_box=config.get("change_box", "right"),
    )
    print(f"  Total edges: {graph_inputs_orig['edge_index'].shape[1]}")

    # Build graph inputs for rotated
    print(f"\n[GRAPH] Building graph inputs for rotated structure...")
    graph_inputs_rot = build_graph_inputs(
        snapshot_rot,
        config["cutoff_radius"],
        config["n_radial"],
        config["l_max"],
        device,
        xyz_permutation=config.get("xyz_permutation", "012"),
        change_box=config.get("change_box", "right"),
    )
    print(f"  Total edges: {graph_inputs_rot['edge_index'].shape[1]}")

    # Predict on original
    print(f"\n[PREDICTION] Running inference on original structure...")
    H_pred_orig = predict_hamiltonian(
        network,
        graph_inputs_orig,
        device,
        magnitude_factorization=config.get("magnitude_factorization", False),
    )
    H_gt_orig = snapshot_orig.hamiltonian.to(device)
    S_orig = snapshot_orig.overlap.to(device)
    H_pred_orig = (H_pred_orig + H_pred_orig.transpose()) * 0.5
    print("  Symmetrized original prediction: H <- 0.5 * (H + H^T)")

    # Predict on rotated
    print(f"\n[PREDICTION] Running inference on rotated structure...")
    H_pred_rot = predict_hamiltonian(
        network,
        graph_inputs_rot,
        device,
        magnitude_factorization=config.get("magnitude_factorization", False),
    )
    H_gt_rot = snapshot_rot.hamiltonian.to(device)
    S_rot = snapshot_rot.overlap.to(device)
    H_pred_rot = (H_pred_rot + H_pred_rot.transpose()) * 0.5
    print("  Symmetrized rotated prediction: H <- 0.5 * (H + H^T)")

    # Compute metrics for original
    print(f"\n{'='*80}")
    print("ORIGINAL STRUCTURE METRICS")
    if partial_train is not None:
        print(f"(evaluating {partial_train} blocks only)")
    print(f"{'='*80}")
    metrics_orig = compute_metrics(
        H_pred_orig, H_gt_orig, S_orig, partial_train=partial_train
    )
    print(f"  MAE:       {metrics_orig['mae']:.6e}")
    print(f"  MSE:       {metrics_orig['mse']:.6e}")
    print(f"  MAE_mod:   {metrics_orig['mae_mod']:.6e}")
    print(f"  mu_H:      {metrics_orig['mu_H']:.6e}")

    # Compute metrics for rotated
    print(f"\n{'='*80}")
    print("ROTATED STRUCTURE METRICS")
    if partial_train is not None:
        print(f"(evaluating {partial_train} blocks only)")
    print(f"{'='*80}")
    metrics_rot = compute_metrics(
        H_pred_rot, H_gt_rot, S_rot, partial_train=partial_train
    )
    print(f"  MAE:       {metrics_rot['mae']:.6e}")
    print(f"  MSE:       {metrics_rot['mse']:.6e}")
    print(f"  MAE_mod:   {metrics_rot['mae_mod']:.6e}")
    print(f"  mu_H:      {metrics_rot['mu_H']:.6e}")

    # Evaluate eigenvalue and DOS errors
    print(f"\n{'='*80}")
    print("EIGENVALUE & DOS METRICS (ORIGINAL)")
    print(f"{'='*80}")
    eig_dos_orig = evaluate_eigen_and_dos(
        H_pred_orig, H_gt_orig, S_orig, output_dir=output_dir, prefix="original"
    )
    print(f"  eig abs mean: {eig_dos_orig['eig_abs_mean']:.6e}")
    print(f"  eig abs max:  {eig_dos_orig['eig_abs_max']:.6e}")
    print(f"  eig rel mean: {eig_dos_orig['eig_rel_mean']:.6e}")
    print(f"  eig rel max:  {eig_dos_orig['eig_rel_max']:.6e}")
    print(f"  dos MAE:      {eig_dos_orig['dos_mae']:.6e}")
    print(f"  dos MSE:      {eig_dos_orig['dos_mse']:.6e}")
    print(f"  dos max abs:  {eig_dos_orig['dos_max_abs']:.6e}")
    print(f"  DOS plot:     {eig_dos_orig['dos_plot_path']}")

    print(f"\n{'='*80}")
    print("EIGENVALUE & DOS METRICS (ROTATED)")
    print(f"{'='*80}")
    eig_dos_rot = evaluate_eigen_and_dos(
        H_pred_rot, H_gt_rot, S_rot, output_dir=output_dir, prefix="rotated"
    )
    print(f"  eig abs mean: {eig_dos_rot['eig_abs_mean']:.6e}")
    print(f"  eig abs max:  {eig_dos_rot['eig_abs_max']:.6e}")
    print(f"  eig rel mean: {eig_dos_rot['eig_rel_mean']:.6e}")
    print(f"  eig rel max:  {eig_dos_rot['eig_rel_max']:.6e}")
    print(f"  dos MAE:      {eig_dos_rot['dos_mae']:.6e}")
    print(f"  dos MSE:      {eig_dos_rot['dos_mse']:.6e}")
    print(f"  dos max abs:  {eig_dos_rot['dos_max_abs']:.6e}")
    print(f"  DOS plot:     {eig_dos_rot['dos_plot_path']}")

    # Visualize original
    print(f"\n{'='*80}")
    print("VISUALIZING ORIGINAL STRUCTURE")
    print(f"{'='*80}")
    visualize_hamiltonians(
        H_pred_orig,
        H_gt_orig,
        S_orig,
        list(snapshot_orig.hamiltonian.atoms),
        orbital_cfg,
        args.k_range,
        output_dir,
        dynamic_range=args.dynamic_range,
        partial_train=partial_train,
        filename_prefix="hamiltonian",
        percentile=args.percentile,
    )

    # Visualize rotated
    print(f"\n{'='*80}")
    print("VISUALIZING ROTATED STRUCTURE")
    print(f"{'='*80}")
    visualize_hamiltonians(
        H_pred_rot,
        H_gt_rot,
        S_rot,
        list(snapshot_rot.hamiltonian.atoms),
        orbital_cfg,
        args.k_range,
        output_dir,
        dynamic_range=args.dynamic_range,
        partial_train=partial_train,
        filename_prefix="hamiltonian_rotated",
        percentile=args.percentile,
    )

    # Visualize irrep contributions if requested
    if args.split_by_irrep:
        print(f"\n{'='*80}")
        print("VISUALIZING IRREP CONTRIBUTIONS")
        print(f"{'='*80}")

        # Get all irreps present in the Hamiltonian
        all_irreps = get_all_irreps_in_hamiltonian(graph_inputs_orig["mapper"])
        print(
            f"\nFound {len(all_irreps)} unique irreps: {[str(ir) for ir in all_irreps]}"
        )

        for irrep in all_irreps:
            irrep_str = str(irrep)
            print(f"\n  Processing irrep: {irrep_str}")

            # Split original structure by irrep
            H_gt_orig_split = split_hamiltonian_by_irrep(
                H_gt_orig, graph_inputs_orig["mapper"], irrep_str
            )
            H_pred_orig_split = split_hamiltonian_by_irrep(
                H_pred_orig, graph_inputs_orig["mapper"], irrep_str
            )

            # Compute metrics for this irrep
            metrics_irrep = compute_metrics(
                H_pred_orig_split, H_gt_orig_split, S_orig, partial_train=partial_train
            )
            print(
                f"    MAE: {metrics_irrep['mae']:.6e}, MSE: {metrics_irrep['mse']:.6e}"
            )

            # Visualize
            visualize_hamiltonians(
                H_pred_orig_split,
                H_gt_orig_split,
                S_orig,
                list(snapshot_orig.hamiltonian.atoms),
                orbital_cfg,
                args.k_range,
                output_dir,
                dynamic_range=args.dynamic_range,
                partial_train=partial_train,
                filename_prefix=f"hamiltonian_{irrep_str}",
                percentile=args.percentile,
            )
            print(f"    Visualizations saved with prefix: hamiltonian_{irrep_str}_*")

    # Save metrics to file
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("MINIMAL MODEL ANALYSIS RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Checkpoint: {model_path}\n")
        f.write(f"Training epoch: {checkpoint['epoch']}\n")
        f.write(f"Training loss: {checkpoint['loss']:.6e}\n")
        if partial_train is not None:
            f.write(f"Partial training mode: {partial_train}\n")
        f.write("\n")
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
        f.write("\n")
        f.write("ORIGINAL EIGENVALUE & DOS METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"eig_abs_mean: {eig_dos_orig['eig_abs_mean']:.6e}\n")
        f.write(f"eig_abs_max:  {eig_dos_orig['eig_abs_max']:.6e}\n")
        f.write(f"eig_rel_mean: {eig_dos_orig['eig_rel_mean']:.6e}\n")
        f.write(f"eig_rel_max:  {eig_dos_orig['eig_rel_max']:.6e}\n")
        f.write(f"dos_mae:      {eig_dos_orig['dos_mae']:.6e}\n")
        f.write(f"dos_mse:      {eig_dos_orig['dos_mse']:.6e}\n")
        f.write(f"dos_max_abs:  {eig_dos_orig['dos_max_abs']:.6e}\n")
        f.write(f"dos_plot:     {eig_dos_orig['dos_plot_path']}\n")
        f.write("\n")
        f.write("ROTATED EIGENVALUE & DOS METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"eig_abs_mean: {eig_dos_rot['eig_abs_mean']:.6e}\n")
        f.write(f"eig_abs_max:  {eig_dos_rot['eig_abs_max']:.6e}\n")
        f.write(f"eig_rel_mean: {eig_dos_rot['eig_rel_mean']:.6e}\n")
        f.write(f"eig_rel_max:  {eig_dos_rot['eig_rel_max']:.6e}\n")
        f.write(f"dos_mae:      {eig_dos_rot['dos_mae']:.6e}\n")
        f.write(f"dos_mse:      {eig_dos_rot['dos_mse']:.6e}\n")
        f.write(f"dos_max_abs:  {eig_dos_rot['dos_max_abs']:.6e}\n")
        f.write(f"dos_plot:     {eig_dos_rot['dos_plot_path']}\n")

    # Distance-binned error curves (64 bins) for original and rotated
    curve_orig = compute_distance_error_curve(
        H_pred=H_pred_orig,
        H_gt=H_gt_orig,
        positions=graph_inputs_orig["positions_used"],
        box=graph_inputs_orig["box_used"],
        partial_train=partial_train,
        n_bins=64,
    )
    curve_rot = compute_distance_error_curve(
        H_pred=H_pred_rot,
        H_gt=H_gt_rot,
        positions=graph_inputs_rot["positions_used"],
        box=graph_inputs_rot["box_used"],
        partial_train=partial_train,
        n_bins=64,
    )

    if curve_orig is not None:
        curve_orig_json = output_dir / "distance_error_curve_original.json"
        curve_orig_plot = output_dir / "distance_error_curve_original.png"
        with open(curve_orig_json, "w") as f:
            json.dump(curve_orig, f, indent=2)
        save_distance_error_curve_plot(
            curve_orig,
            curve_orig_plot,
            title="Distance Error Curve (Original, 64 bins)",
        )
        print(f"Distance curve saved: {curve_orig_json}")
        print(f"Distance curve plot saved: {curve_orig_plot}")

    if curve_rot is not None:
        curve_rot_json = output_dir / "distance_error_curve_rotated.json"
        curve_rot_plot = output_dir / "distance_error_curve_rotated.png"
        with open(curve_rot_json, "w") as f:
            json.dump(curve_rot, f, indent=2)
        save_distance_error_curve_plot(
            curve_rot, curve_rot_plot, title="Distance Error Curve (Rotated, 64 bins)"
        )
        print(f"Distance curve saved: {curve_rot_json}")
        print(f"Distance curve plot saved: {curve_rot_plot}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Visualizations saved to: {output_dir}")
    print("\n✓ Analysis finished successfully!")


if __name__ == "__main__":
    main()
