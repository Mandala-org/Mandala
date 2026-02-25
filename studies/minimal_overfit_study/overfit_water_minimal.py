"""
Minimal E(3)-Equivariant Network Study: Overfit Single Water Structure
=======================================================================

This study implements the SIMPLEST possible E(3)-equivariant GNN from scratch
to overfit on a single H2O snapshot. Everything is done explicitly without
importing high-level classes like E3GNN or E3GNNDataset.

Goal: Achieve highest possible accuracy through aggressive overfitting.

Architecture:
- Simple node encoder: element embedding -> scalars
- Simple edge encoder: distance + spherical harmonics
- 1-2 message passing layers with basic tensor products
- Simple head: edge features -> matrix blocks via BlockIrrepMapper

Verbose logging at every step for educational purposes.
"""

# Standard library imports
import sys
import os
from pathlib import Path
import argparse
import time
import json
from collections import Counter

# Third-party imports
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.math import soft_one_hot_linspace
from ase import Atoms
from ase.neighborlist import neighbor_list

# Project-specific imports
from net.common import build_hidden_irreps
from data.snapshot import Snapshot
from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import IrrepsBlockData, BlockMatrix
import wandb
from common import (
    MinimalNetwork,
    canonicalize_edge_order,
    compute_detailed_metrics,
    compute_distance_error_curve,
    save_distance_error_curve_plot,
    save_dos_comparison_plot,
    save_hamiltonian_frame_to_disk,
    compile_frames_to_video,
    filter_irreps_block_data_by_irrep,
    compute_irrep_metrics,
    get_all_irreps_in_hamiltonian,
    split_hamiltonian_by_irrep,
    visualize_hamiltonians,
    permutation_to_matrix,
)
from strict_checks import strict_edge_alignment_check, strict_reverse_edge_check
from detailed_logging import (
    build_wandb_detailed_metrics_log,
    build_wandb_per_irrep_metrics_log,
    log_config,
    log_cutoff_application,
    log_detailed_training_metrics,
    log_final_metrics,
    log_graph,
    log_invalid_gradient_failure,
    log_invalid_loss_failure,
    log_mapper_info,
    log_orbital_config,
    log_per_irrep_metrics,
    log_snapshot_info,
    log_strict_checks_passed,
    log_study_complete,
)


if __name__ == "__main__":
    # =============================================================================
    # PARSE ARGUMENTS
    # =============================================================================
    parser = argparse.ArgumentParser(
        description="Minimal E(3)-Equivariant Network: Overfit Single Water Structure"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str("data/small/H2O/original/H2O.matrix"),
        help="Path to the matrix file",
    )
    parser.add_argument(
        "--info-path",
        type=str,
        default=str("data/small/H2O/original/H2O.info.out"),
        help="Path to the info file",
    )
    parser.add_argument(
        "--convention",
        type=str,
        default="e3nn",
        help="Basis convention for data loading ('e3nn' or other options) (default: 'e3nn')",
    )
    parser.add_argument(
        "--orbital-selection",
        type=str,
        default=None,
        help=(
            "Optional orbital reduction spec applied via Snapshot.reduce_orbitals(). "
            'Examples: \'1s1p\' (global) or \'{"O":"2s3p","H":"1s"}\' (per-element JSON).'
        ),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden dimension for features (used if --hidden-irreps not provided)",
    )
    parser.add_argument(
        "--l-max",
        type=int,
        default=2,
        help="Maximum angular momentum (used if --hidden-irreps not provided)",
    )
    parser.add_argument(
        "--hidden-irreps",
        type=str,
        default=None,
        help="Hidden irreps string (e.g., '32x0e+32x1o+32x2e'). If not provided, will be constructed using build_hidden_irreps with --hidden-dim and --l-max",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of message passing layers",
    )
    parser.add_argument(
        "--cutoff-radius",
        type=float,
        default=8.0,
        help="Cutoff radius in Angstroms",
    )
    parser.add_argument(
        "--n-radial",
        type=int,
        default=16,
        help="Number of radial basis functions",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=200,
        help="Interval for logging",
    )
    parser.add_argument(
        "--adaptive-log-interval",
        action="store_true",
        default=False,
        help="Adaptive logging cadence: epochs 1-10 every epoch, 11-100 every 10 epochs, then use --log-interval (default: False)",
    )
    parser.add_argument(
        "--log-data",
        action="store_true",
        default=False,
        help="Enable detailed data/graph console logging (default: False)",
    )
    parser.add_argument(
        "--log-model",
        action="store_true",
        default=False,
        help="Enable detailed model/setup console logging (default: False)",
    )
    parser.add_argument(
        "--log-forward",
        action="store_true",
        default=False,
        help="Enable periodic forward-pass console logging (default: False).",
    )
    parser.add_argument(
        "--log-per-irrep-metrics",
        action="store_true",
        default=False,
        help="Enable periodic per-irrep metric console logging during training (default: False).",
    )
    parser.add_argument(
        "--log-per-irrep-images",
        action="store_true",
        default=False,
        help="Generate per-irrep visualization images (k-range=0) and log to WandB (default: False).",
    )
    parser.add_argument(
        "--log-activations-wandb",
        action="store_true",
        default=False,
        help="Enable logging activation magnitudes to WandB (default: False).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Enable per-stage timing benchmark reporting after [METRICS] (default: False)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str("studies/minimal_overfit_study/checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="WandB run name (default: auto-generated by WandB)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping max norm (default: 1.0, set to 0 to disable)",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor (default: 0.5)",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=600,
        help="ReduceLROnPlateau patience (default: 600)",
    )
    parser.add_argument(
        "--partial-train",
        type=str,
        default=None,
        choices=["diag", "offdiag", "shifted_self", None],
        help="Train on partial data: 'diag' (i==j, zero shift), 'shifted_self' (i==j, non-zero shift), 'offdiag' (i!=j), or None (all blocks)",
    )
    parser.add_argument(
        "--separate-shifted-self",
        action="store_true",
        default=False,
        help="Use a separate prediction head and normalization bucket for shifted-self edges (i==j with non-zero shift). Keeps previous 2-way behavior when disabled.",
    )
    parser.add_argument(
        "--train-on-irrep-parts",
        action="store_true",
        help="Decompose loss into per-irrep contributions (log to partial/* in wandb)",
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        default=False,
        help="Generate video from training frames (default: False)",
    )
    parser.add_argument(
        "--verbose-forward",
        action="store_true",
        default=False,
        help="Show detailed forward pass debug prints (shapes, irreps, etc.) (default: False)",
    )
    parser.add_argument(
        "--xyz-permutation",
        type=str,
        default="012",
        help="XYZ permutation as 3-digit string (default: '012', can be '120', '201', etc.)",
    )
    parser.add_argument(
        "--change-box",
        type=str,
        default="right",
        choices=["left", "right", "both"],
        help="How to apply permutation to box: 'left' (M @ box), 'right' (box @ M), or 'both' (M @ box @ M) (default: 'right')",
    )
    parser.add_argument(
        "--box-convention",
        type=str,
        choices=["rows", "cols"],
        default="rows",
        help="Interpretation convention for loaded box matrix; use 'cols' to transpose after loading (default: 'rows')",
    )
    parser.add_argument(
        "--normalize-blocks",
        action="store_true",
        default=False,
        help="Normalize blocks by average magnitude per edge class before training (diag/offdiag by default; diag/shifted_self/offdiag with --separate-shifted-self) (default: False)",
    )
    parser.add_argument(
        "--magnitude-factorization",
        action="store_true",
        default=False,
        help="Predict normalized blocks and separate per-edge magnitudes (default: False)",
    )
    parser.add_argument(
        "--head-mlp-for-scalars",
        dest="head_mlp_for_scalars",
        action="store_true",
        default=False,
        help="Predict scalar irrep parts with a dedicated MLP head from magnitude-branch input (default: False)",
    )
    parser.add_argument(
        "--head-use-tensor-square",
        dest="head_use_tensor_square",
        action="store_true",
        default=False,
        help="Use TensorSquare(edge embeddings) as input to the main head projections (default: False)",
    )
    parser.add_argument(
        "--magnitude-lambda",
        type=float,
        default=1.0,
        help="Weight of magnitude loss contribution when --magnitude-factorization is enabled (default: 1.0)",
    )
    parser.add_argument(
        "--apply-cutoff-to-targets",
        action="store_true",
        default=False,
        help="Apply cutoff filtering to target matrices (H/S/D) before training and metrics. Keeps default behavior unchanged when omitted.",
    )
    parser.add_argument(
        "--require-exact-edge-match",
        action="store_true",
        default=False,
        help="Require exact edge counts and key sets in strict edge alignment checks (default: False).",
    )

    args = parser.parse_args()
    if args.normalize_blocks and args.magnitude_factorization:
        raise ValueError(
            "--normalize-blocks and --magnitude-factorization are mutually exclusive. "
            "Disable one of them."
        )

    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    print("=" * 80)
    print("MINIMAL WATER OVERFIT STUDY - EXPLICIT IMPLEMENTATION")
    print("=" * 80)

    CONFIG = {
        # Data
        "data_path": Path(args.data_path),
        "info_path": Path(args.info_path),
        "convention": args.convention,
        "orbital_selection": args.orbital_selection,
        "xyz_permutation": args.xyz_permutation,
        "change_box": args.change_box,
        "box_convention": args.box_convention,
        # Network architecture
        "hidden_dim": args.hidden_dim,
        "l_max": args.l_max,
        "hidden_irreps": args.hidden_irreps,  # Can be None
        "num_layers": args.num_layers,
        "cutoff_radius": args.cutoff_radius,
        "n_radial": args.n_radial,
        # Training
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "log_interval": args.log_interval,
        "adaptive_log_interval": args.adaptive_log_interval,
        "log_data": args.log_data,
        "log_model": args.log_model,
        "log_forward": args.log_forward,
        "log_per_irrep_metrics": args.log_per_irrep_metrics,
        "log_per_irrep_images": args.log_per_irrep_images,
        "log_activations_wandb": args.log_activations_wandb,
        "benchmark": args.benchmark,
        "grad_clip": args.grad_clip,
        "partial_train": args.partial_train,
        "separate_shifted_self": args.separate_shifted_self,
        "train_on_irrep_parts": args.train_on_irrep_parts,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "generate_video": args.generate_video,
        "verbose_forward": args.verbose_forward,
        "normalize_blocks": args.normalize_blocks,
        "magnitude_factorization": args.magnitude_factorization,
        "head_mlp_for_scalars": args.head_mlp_for_scalars,
        "head_use_tensor_square": args.head_use_tensor_square,
        "magnitude_lambda": args.magnitude_lambda,
        "apply_cutoff_to_targets": args.apply_cutoff_to_targets,
        "require_exact_edge_match": args.require_exact_edge_match,
        # Device
        "device": args.device,
        # Target
        "train_target": "matrix",
        # Checkpointing
        "checkpoint_dir": Path(args.checkpoint_dir),
    }

    # Initialize WandB
    wandb_kwargs = {
        "project": "mandala-minimal-overfit-study",
        "config": CONFIG,
    }
    if args.run_name is not None:
        wandb_kwargs["name"] = args.run_name
    wandb.init(**wandb_kwargs)

    # Create run-specific checkpoint directory
    run_name = wandb.run.name
    run_checkpoint_dir = CONFIG["checkpoint_dir"] / run_name
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Create frame output directory for video frames
    frame_output_dir = run_checkpoint_dir / "frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    log_config(CONFIG, run_checkpoint_dir, frame_output_dir)

    device = torch.device(CONFIG["device"])

    # =============================================================================
    # LOAD DATA
    # =============================================================================
    if CONFIG["log_data"]:
        print("\n[DATA] Loading single water snapshot...")
        print(f"  Matrix file: {CONFIG['data_path']}")
        print(f"  Info file: {CONFIG['info_path']}")

    snapshot = Snapshot.from_openmx(
        matrix_path=CONFIG["data_path"],
        info_path=CONFIG["info_path"],
        convention=CONFIG["convention"],  # Automatically converts to specified basis
        symmetrize_density=True,
        cutoff_radius=None,  # No filtering, we'll use all edges
        dtype=torch.float32,
    )

    # Optionally filter GT matrices by cutoff so loss/metrics ignore long-range blocks.
    # Default remains unchanged unless --apply-cutoff-to-targets is explicitly set.
    if CONFIG["apply_cutoff_to_targets"]:
        before_edges = sum(v.shape[1] for v in snapshot.hamiltonian.pair_edges.values())
        snapshot = snapshot.filter_by_distance(CONFIG["cutoff_radius"])
        after_edges = sum(v.shape[1] for v in snapshot.hamiltonian.pair_edges.values())
        if CONFIG["log_data"]:
            log_cutoff_application(before_edges, after_edges, CONFIG["cutoff_radius"])

    # Optional orbital reduction for GT matrices/targets.
    # String specs are applied globally (e.g., "1s1p"), while JSON object strings
    # can provide per-element specs (e.g., {"O":"2s3p","H":"1s"}).
    if CONFIG["orbital_selection"] is not None:
        orbital_selection = CONFIG["orbital_selection"]
        selection_obj = orbital_selection
        orbital_selection_stripped = orbital_selection.strip()
        if orbital_selection_stripped.startswith("{"):
            try:
                selection_obj = json.loads(orbital_selection_stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON for --orbital-selection. Example: "
                    '\'{"O":"2s3p","H":"1s"}\''
                ) from exc
        snapshot = snapshot.reduce_orbitals(selection_obj)
        if CONFIG["log_data"]:
            print("\n  Applied orbital reduction:")
            print(f"    Spec: {orbital_selection}")
            print(
                f"    Reduced orbital config: "
                f"{snapshot.hamiltonian.orbital_cfg.to_dict()}"
            )

    if CONFIG["log_data"]:
        log_snapshot_info(snapshot)

    # Extract components
    hamiltonian_e3nn = snapshot.hamiltonian.to(device)
    overlap_e3nn = snapshot.overlap.to(device)
    density_e3nn = snapshot.density.to(device)
    orbital_cfg = snapshot.hamiltonian.orbital_cfg
    positions = snapshot.positions.to(device)
    box = snapshot.box.to(device) if snapshot.box is not None else None
    if box is not None and CONFIG["box_convention"] == "cols":
        box = box.T
        if CONFIG["log_data"]:
            print(
                "  Applied box convention 'cols': transposed loaded box (box = box.T)"
            )
    atoms_list = list(snapshot.hamiltonian.atoms)

    # Apply coordinate permutation if specified
    if CONFIG["xyz_permutation"] != "012":
        cob_matrix = permutation_to_matrix(CONFIG["xyz_permutation"], device)
        if CONFIG["log_data"]:
            print(f"\n  Applying xyz permutation: {CONFIG['xyz_permutation']}")
            print(f"    Matrix:\n{cob_matrix}")
        # Multiply positions from the right: positions @ cob_matrix.T
        positions = positions @ cob_matrix.T
        if CONFIG["change_box"] == "right":
            # box @ cob_matrix.T
            box = box @ cob_matrix.T
        elif CONFIG["change_box"] == "left":
            # cob_matrix @ box
            box = cob_matrix @ box
        elif CONFIG["change_box"] == "both":
            # cob_matrix @ box @ cob_matrix.T
            box = cob_matrix @ box @ cob_matrix.T
        else:
            raise ValueError(f"Invalid change_box option: {CONFIG['change_box']}")
        if CONFIG["log_data"]:
            print(
                f"    Applied to positions and box (change_box={CONFIG['change_box']})"
            )

    if CONFIG["log_model"]:
        log_orbital_config(orbital_cfg)

    # Create BlockIrrepMapper
    mapper = BlockIrrepMapper(orbital_cfg, device=device, dtype=torch.float32)
    if CONFIG["log_model"]:
        log_mapper_info(mapper)

    def canonicalize_block_matrix_edges(block_matrix, positions, box):
        """
        Canonicalize per-key edge order in BlockMatrix using the same rule as graph
        canonicalization (self-edges first, then off-diagonals sorted by
        distance/shift/src/dst).
        """
        order_dict = {}
        for key, edges_5d in block_matrix.pair_edges.items():
            edge_shift_local = edges_5d[:3]  # (3, E)
            edge_index_local = edges_5d[3:]  # (2, E)
            _, _, perm_local = canonicalize_edge_order(
                edge_index=edge_index_local,
                edge_shift=edge_shift_local,
                positions=positions,
                box=box,
            )
            order_dict[key] = perm_local
        return block_matrix.reorder_edges(order_dict)

    # Canonicalize target edges after all coordinate/box transforms so target edge
    # ordering uses exactly the same geometric convention as graph canonicalization.
    hamiltonian_e3nn = canonicalize_block_matrix_edges(hamiltonian_e3nn, positions, box)
    overlap_e3nn = canonicalize_block_matrix_edges(overlap_e3nn, positions, box)
    density_e3nn = canonicalize_block_matrix_edges(density_e3nn, positions, box)

    hamiltonian_e3nn = (hamiltonian_e3nn + hamiltonian_e3nn.transpose()) * 0.5
    if CONFIG["log_data"]:
        print(
            "  Pre-symmetrized Hamiltonian target via "
            "0.5 * (H + H.transpose()) in BlockMatrix form."
        )

    # Store target as matrix blocks (train_target = "matrix")
    if CONFIG["log_model"]:
        print("\n[TARGETS] Storing target as matrix blocks...")
    target_H_matrix = hamiltonian_e3nn

    if CONFIG["log_model"]:
        print("  Target matrix blocks:")
        for key in target_H_matrix.pair_blocks.keys():
            block_shape = target_H_matrix.pair_blocks[key].shape
            edge_shape = target_H_matrix.pair_edges[key].shape
            pair_irreps = mapper.get_pair_irreps(key)
            print(
                f"    {key}: blocks {block_shape}, edges {edge_shape}, irreps {pair_irreps}"
            )

    num_atoms = len(atoms_list)

    # Create ASE atoms object for neighbor list
    ase_atoms = Atoms(
        symbols=atoms_list,
        positions=positions.cpu().numpy(),
        cell=box.cpu().numpy() if box is not None else None,
        pbc=box is not None,
    )

    # Find neighbors using ASE
    src, dst, offsets = neighbor_list(
        "ijS", ase_atoms, CONFIG["cutoff_radius"], self_interaction=False
    )

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

    # Canonicalize edge order to match the main data pipeline.
    # Self-edges first (sorted by src), then off-diagonals sorted by
    # (distance, sx, sy, sz, src, dst).
    edge_index, edge_shift, _ = canonicalize_edge_order(
        edge_index=edge_index,
        edge_shift=edge_shift,
        positions=positions,
        box=box,
    )

    src_sorted = edge_index[0]
    dst_sorted = edge_index[1]
    num_self_edges = int(
        (
            (src_sorted == dst_sorted)
            & (edge_shift[0] == 0)
            & (edge_shift[1] == 0)
            & (edge_shift[2] == 0)
        )
        .sum()
        .item()
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

    if CONFIG["log_data"]:
        log_graph(
            atoms_list=atoms_list,
            positions=positions,
            box=box,
            cutoff_radius=CONFIG["cutoff_radius"],
            src=src,
            dst=dst,
            offsets=offsets,
            edge_index=edge_index,
            num_self_edges=num_self_edges,
            edge_dist=edge_dist,
        )

    # Compute spherical harmonics
    if CONFIG["log_data"]:
        print(f"\n  Computing spherical harmonics (l_max={CONFIG['l_max']})...")
    sh_irreps = Irreps.spherical_harmonics(CONFIG["l_max"])
    if CONFIG["log_data"]:
        print(f"  SH irreps: {sh_irreps}")

    # Normalize edge vectors (avoid division by zero for self-edges)
    edge_vec_norm = edge_vec.clone()
    non_zero_mask = edge_dist > 1e-6
    edge_vec_norm[non_zero_mask] = edge_vec[non_zero_mask] / edge_dist[
        non_zero_mask
    ].unsqueeze(-1)

    edge_sh = spherical_harmonics(sh_irreps, edge_vec_norm, normalize=False)
    if CONFIG["log_data"]:
        print(f"  Edge SH shape: {edge_sh.shape}")

    # Radial basis functions
    if CONFIG["log_data"]:
        print(
            f"\n  Computing radial embeddings ({CONFIG['n_radial']} basis functions)..."
        )
    edge_length_emb = soft_one_hot_linspace(
        edge_dist,
        start=0.0,
        end=CONFIG["cutoff_radius"],
        number=CONFIG["n_radial"],
        basis="gaussian",
        cutoff=False,
    )
    edge_length_emb = edge_length_emb * CONFIG["n_radial"] ** 0.5  # Normalization
    if CONFIG["log_data"]:
        print(f"  Edge length embedding shape: {edge_length_emb.shape}")

    # Edge type indices
    if CONFIG["log_data"]:
        print("\n  Computing edge type indices...")
    element_to_idx = {elem: idx for idx, elem in enumerate(orbital_cfg.elements())}
    if CONFIG["log_data"]:
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

    if CONFIG["log_data"]:
        print(f"  Edge types (first 10): {edge_type_strs[:10]}")
        print(f"  Edge type indices (first 10): {edge_type_idx[:10].tolist()}")

    # Create batch indices (all nodes/edges belong to the same graph)
    batch_node = torch.zeros(num_atoms, dtype=torch.long, device=device)
    batch_edge = torch.zeros(edge_index.shape[1], dtype=torch.long, device=device)
    if CONFIG["log_data"]:
        print(f"\n  Batch indices: nodes {batch_node.shape}, edges {batch_edge.shape}")

    # =============================================================================
    # STRICT EDGE ALIGNMENT CHECKS
    # =============================================================================
    if CONFIG["log_data"]:
        print("\n[STRICT CHECKS] Validating graph/target edge alignment...")
    strict_reverse_edge_check(
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_set_name="graph",
    )
    require_exact_edge_match = bool(CONFIG["require_exact_edge_match"])
    strict_edge_alignment_check(
        target_matrix=target_H_matrix,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="hamiltonian",
        require_exact=require_exact_edge_match,
    )
    strict_edge_alignment_check(
        target_matrix=overlap_e3nn,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="overlap",
        require_exact=require_exact_edge_match,
    )
    strict_edge_alignment_check(
        target_matrix=density_e3nn,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="density",
        require_exact=require_exact_edge_match,
    )
    if CONFIG["log_data"]:
        log_strict_checks_passed()

    # =============================================================================
    # HELPER FUNCTIONS FOR PARTIAL TRAINING
    # =============================================================================
    def get_diagonal_mask(edges_5d):
        """
        Get mask for diagonal blocks (self-interactions: sx=sy=sz=0, i=j).

        Args:
            edges_5d: (5, num_edges) tensor with [sx, sy, sz, i, j]

        Returns:
            Boolean mask of shape (num_edges,) with True for diagonal blocks
        """
        sx, sy, sz, i, j = (
            edges_5d[0],
            edges_5d[1],
            edges_5d[2],
            edges_5d[3],
            edges_5d[4],
        )
        is_self_interaction = (sx == 0) & (sy == 0) & (sz == 0)
        is_same_atom = i == j
        return is_self_interaction & is_same_atom

    def compute_block_normalization_factors(block_matrix):
        """
        Compute per-key, per-diagonal-status normalization factors.
        - Diagonal blocks: sx==sy==sz==0 and i==j
        - Shifted-self blocks: i==j and (sx,sy,sz)!=(0,0,0) [optional separate bucket]
        - Off-diagonal blocks: i!=j
        Uses L2 norm (Frobenius norm) for magnitude computation.

        Args:
            block_matrix: BlockMatrix object

        Returns:
            Dictionary:
              - default mode: {key: {"diag": float, "offdiag": float}, ...}
              - separate-shifted-self mode:
                  {key: {"diag": float, "shifted_self": float, "offdiag": float}, ...}
        """
        norm_factors = {}
        for key in block_matrix.pair_blocks.keys():
            blocks = block_matrix.pair_blocks[key]  # (num_edges, ...)
            edges = block_matrix.pair_edges[key]  # (5, num_edges)

            sx, sy, sz, i, j = edges[0], edges[1], edges[2], edges[3], edges[4]
            is_same_atom = i == j
            is_zero_shift = (sx == 0) & (sy == 0) & (sz == 0)
            diag_mask = is_same_atom & is_zero_shift
            shifted_self_mask = is_same_atom & (~is_zero_shift)
            if CONFIG["separate_shifted_self"]:
                offdiag_mask = ~is_same_atom
            else:
                # Backward-compatible behavior: shifted-self shares offdiag bucket.
                offdiag_mask = ~diag_mask

            if diag_mask.any():
                diag_blocks = blocks[diag_mask]
                diag_norms = torch.linalg.norm(
                    diag_blocks.reshape(diag_blocks.shape[0], -1), dim=1
                )
                diag_mag = diag_norms.mean().item()
            else:
                diag_mag = 1.0

            if shifted_self_mask.any():
                shifted_self_blocks = blocks[shifted_self_mask]
                shifted_self_norms = torch.linalg.norm(
                    shifted_self_blocks.reshape(shifted_self_blocks.shape[0], -1), dim=1
                )
                shifted_self_mag = shifted_self_norms.mean().item()
            else:
                shifted_self_mag = 1.0

            if offdiag_mask.any():
                offdiag_blocks = blocks[offdiag_mask]
                offdiag_norms = torch.linalg.norm(
                    offdiag_blocks.reshape(offdiag_blocks.shape[0], -1), dim=1
                )
                offdiag_mag = offdiag_norms.mean().item()
            else:
                offdiag_mag = 1.0

            diag_mag = max(diag_mag, 1e-8)
            shifted_self_mag = max(shifted_self_mag, 1e-8)
            offdiag_mag = max(offdiag_mag, 1e-8)

            if CONFIG["separate_shifted_self"]:
                norm_factors[key] = {
                    "diag": diag_mag,
                    "shifted_self": shifted_self_mag,
                    "offdiag": offdiag_mag,
                }
            else:
                norm_factors[key] = {"diag": diag_mag, "offdiag": offdiag_mag}

        return norm_factors

    def compute_block_norm_targets_and_normalized_matrix(block_matrix, eps=1e-12):
        """
        Build per-edge Frobenius norm targets and a unit-norm normalized BlockMatrix.

        Returns:
            norm_targets: dict[key] -> (E_key,) tensor of per-edge norms (clamped by eps)
            normalized_matrix: BlockMatrix with each block divided by its own norm
            norm_stats: dict[key] -> {"min", "mean", "max"}
        """
        norm_targets = {}
        norm_stats = {}
        normalized_blocks = {}

        for key, blocks in block_matrix.pair_blocks.items():
            flat = blocks.reshape(blocks.shape[0], -1)
            norms = torch.linalg.norm(flat, dim=1)
            norms_safe = torch.clamp(norms, min=eps)
            scale = norms_safe
            while scale.ndim < blocks.ndim:
                scale = scale.unsqueeze(-1)
            normalized_blocks[key] = blocks / scale
            norm_targets[key] = norms_safe
            norm_stats[key] = {
                "min": float(norms_safe.min().item()),
                "mean": float(norms_safe.mean().item()),
                "max": float(norms_safe.max().item()),
            }

        normalized_matrix = BlockMatrix(
            atoms=block_matrix.atoms,
            atom_counts=block_matrix.atom_counts,
            pair_blocks=normalized_blocks,
            pair_edges=block_matrix.pair_edges,
            lookup=block_matrix.lookup,
            orbital_cfg=block_matrix.orbital_cfg,
            basis=block_matrix.basis,
        )
        return norm_targets, normalized_matrix, norm_stats

    def reconstruct_actual_prediction_from_magnitudes(
        pred_matrix_normalized, pred_raw_outputs, require_magnitudes=False
    ):
        """
        Reconstruct actual predicted blocks: B_pred = m_pred * B_pred_normalized.
        """
        reconstructed_blocks = {}
        for key, blocks in pred_matrix_normalized.pair_blocks.items():
            if key not in pred_raw_outputs or "magnitudes" not in pred_raw_outputs[key]:
                if require_magnitudes:
                    raise RuntimeError(
                        f"Missing predicted magnitudes for key '{key}' in magnitude-factorization mode."
                    )
                reconstructed_blocks[key] = blocks
                continue

            magnitudes = pred_raw_outputs[key]["magnitudes"]
            if magnitudes.shape[0] != blocks.shape[0]:
                raise RuntimeError(
                    f"Magnitude/prediction edge-count mismatch for key '{key}': "
                    f"{magnitudes.shape[0]} vs {blocks.shape[0]}"
                )

            scale = magnitudes
            while scale.ndim < blocks.ndim:
                scale = scale.unsqueeze(-1)
            reconstructed_blocks[key] = blocks * scale

        return BlockMatrix(
            atoms=pred_matrix_normalized.atoms,
            atom_counts=pred_matrix_normalized.atom_counts,
            pair_blocks=reconstructed_blocks,
            pair_edges=pred_matrix_normalized.pair_edges,
            lookup=pred_matrix_normalized.lookup,
            orbital_cfg=pred_matrix_normalized.orbital_cfg,
            basis=pred_matrix_normalized.basis,
        )

    def get_block_status(edges_5d, edge_idx):
        """
        Determine class for a block at edge_idx:
        - "diag": i==j and zero shift
        - "shifted_self": i==j and non-zero shift (only when enabled)
        - "offdiag": i!=j, or all non-diag edges when separate mode is disabled
        """
        sx = edges_5d[0, edge_idx].item()
        sy = edges_5d[1, edge_idx].item()
        sz = edges_5d[2, edge_idx].item()
        i = edges_5d[3, edge_idx].item()
        j = edges_5d[4, edge_idx].item()
        if i == j and sx == 0 and sy == 0 and sz == 0:
            return "diag"
        if i == j and CONFIG["separate_shifted_self"]:
            return "shifted_self"
        return "offdiag"

    def filter_blocks_by_partial_train(block_matrix, partial_train):
        """
        Filter blocks based on partial_train setting.

        Args:
            block_matrix: BlockMatrix object
            partial_train: "diag", "offdiag", "shifted_self", or None

        Returns:
            Dictionary mapping edge_type -> (filtered_blocks, mask)
        """
        if partial_train is None:
            # Return all blocks with full mask
            return {
                key: (
                    blocks,
                    torch.ones(blocks.shape[0], dtype=torch.bool, device=blocks.device),
                )
                for key, blocks in block_matrix.pair_blocks.items()
            }

        filtered_data = {}
        for key in block_matrix.pair_blocks.keys():
            blocks = block_matrix.pair_blocks[key]
            edges = block_matrix.pair_edges[key]  # (5, num_edges)

            diag_mask = get_diagonal_mask(edges)
            sx, sy, sz, i, j = edges[0], edges[1], edges[2], edges[3], edges[4]
            shifted_self_mask = (i == j) & ((sx != 0) | (sy != 0) | (sz != 0))

            if partial_train == "diag":
                mask = diag_mask
            elif partial_train == "shifted_self":
                mask = shifted_self_mask
            elif partial_train == "offdiag":
                if CONFIG["separate_shifted_self"]:
                    mask = i != j
                else:
                    mask = ~diag_mask
            else:
                mask = torch.ones(
                    blocks.shape[0], dtype=torch.bool, device=blocks.device
                )

            filtered_data[key] = (blocks, mask)

        return filtered_data

    # =============================================================================
    # COMPUTE BLOCK NORMALIZATION FACTORS (IF ENABLED)
    # =============================================================================
    norm_factors = None
    if CONFIG["normalize_blocks"]:
        if CONFIG["log_model"]:
            print("\n[NORMALIZATION] Computing block normalization factors...")
        norm_factors = compute_block_normalization_factors(target_H_matrix)
        if CONFIG["log_model"]:
            for key, factors in norm_factors.items():
                if CONFIG["separate_shifted_self"]:
                    print(
                        f"  {key}: diag_mag={factors['diag']:.6e}, "
                        f"shifted_self_mag={factors['shifted_self']:.6e}, "
                        f"offdiag_mag={factors['offdiag']:.6e}"
                    )
                else:
                    print(
                        f"  {key}: diag_mag={factors['diag']:.6e}, offdiag_mag={factors['offdiag']:.6e}"
                    )

    # Magnitude-factorization targets (if enabled): per-block Frobenius norms and
    # normalized target blocks.
    target_block_norms = None
    target_H_matrix_normalized = None
    target_H_irreps_normalized = None
    if CONFIG["magnitude_factorization"]:
        if CONFIG["log_model"]:
            print(
                "\n[MAGNITUDE FACTORIZATION] Precomputing per-block norms and normalized targets..."
            )
        target_block_norms, target_H_matrix_normalized, norm_stats = (
            compute_block_norm_targets_and_normalized_matrix(target_H_matrix, eps=1e-12)
        )
        if CONFIG["log_model"]:
            for key, stats in norm_stats.items():
                print(
                    f"  {key}: norm min={stats['min']:.6e}, mean={stats['mean']:.6e}, max={stats['max']:.6e}"
                )

    # =============================================================================
    # DEFINE MINIMAL NETWORK
    # =============================================================================
    if CONFIG["log_model"]:
        print("\n[NETWORK] Defining minimal E(3)-equivariant network...")

    # Use provided hidden_irreps or build them automatically
    if CONFIG["hidden_irreps"] is not None:
        hidden_irreps = Irreps(CONFIG["hidden_irreps"])
        if CONFIG["log_model"]:
            print(f"  Using provided hidden irreps: {hidden_irreps}")
    else:
        hidden_irreps = build_hidden_irreps(
            l_max=CONFIG["l_max"], base_dim=CONFIG["hidden_dim"], use_odd_features=True
        )
        if CONFIG["log_model"]:
            print(f"  Built hidden irreps: {hidden_irreps}")
        # Store the constructed irreps in config for checkpoint saving
        CONFIG["hidden_irreps"] = str(hidden_irreps)

    # Instantiate network
    if CONFIG["log_model"]:
        print("\nInstantiating network...")
    num_elements = len(orbital_cfg.elements())
    num_edge_types = num_elements**2
    if not CONFIG["log_model"]:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    network = MinimalNetwork(
        num_elements=num_elements,
        n_radial=CONFIG["n_radial"],
        num_edge_types=num_edge_types,
        hidden_irreps=hidden_irreps,
        sh_irreps=sh_irreps,
        num_layers=CONFIG["num_layers"],
        mapper=mapper,
        magnitude_factorization=CONFIG["magnitude_factorization"],
        head_mlp_for_scalars=CONFIG["head_mlp_for_scalars"],
        head_use_tensor_square=CONFIG["head_use_tensor_square"],
        separate_shifted_self=CONFIG["separate_shifted_self"],
    ).to(device)
    if not CONFIG["log_model"]:
        sys.stdout.close()
        sys.stdout = old_stdout

    if CONFIG["log_model"]:
        print("\n[OK] Network architecture complete!")

    # =============================================================================
    # PREPARE TARGET IN IRREPS SPACE
    # =============================================================================
    if CONFIG["log_model"]:
        print("\n[IRREP DECOMPOSITION] Converting target to irreps space...")
    target_H_irreps = target_H_matrix.to_vectors(mapper)
    if CONFIG["magnitude_factorization"]:
        target_H_irreps_normalized = target_H_matrix_normalized.to_vectors(mapper)

    target_H_matrix_for_block_loss = (
        target_H_matrix_normalized
        if CONFIG["magnitude_factorization"]
        else target_H_matrix
    )
    target_H_irreps_for_block_loss = (
        target_H_irreps_normalized
        if CONFIG["magnitude_factorization"]
        else target_H_irreps
    )
    all_irreps = get_all_irreps_in_hamiltonian(mapper)
    if CONFIG["log_model"]:
        print(
            f"  Found {len(all_irreps)} unique irreps: {[str(ir) for ir in all_irreps]}"
        )

    # =============================================================================
    # TRAINING LOOP
    # =============================================================================
    print("")
    print("=" * 80)
    print("TRAINING TO OVERFIT")
    print("=" * 80)

    optimizer = Adam(network.parameters(), lr=CONFIG["lr"])

    # Add ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["lr_factor"],
        patience=CONFIG["lr_patience"],
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=10,
        min_lr=1e-6,
        verbose=True,
    )

    # Training history
    history = {
        "loss": [],
        "mse_H": [],
        "mae_H": [],
    }

    # Track best model
    best_loss = float("inf")
    best_epoch = 0

    # Track timing
    last_log_time = time.time()
    last_logged_epoch = -1

    benchmark_order = [
        "epoch_total",
        "forward_total",
        "pred_pack_raw_outputs",
        "pred_to_blocks_norm",
        "loss_block_total",
        "loss_block_irrep_filter_and_to_blocks",
        "loss_block_norm_scaling_overhead",
        "backward_total",
        "optimizer_step",
        "step_wandb_log",
    ]
    benchmark_accum = {k: 0.0 for k in benchmark_order}
    benchmark_epochs_accum = 0

    def benchmark_add(name: str, dt_seconds: float) -> None:
        if CONFIG["benchmark"] and name in benchmark_accum:
            benchmark_accum[name] += max(float(dt_seconds), 0.0)

    def print_benchmark_report() -> None:
        if not CONFIG["benchmark"] or benchmark_epochs_accum <= 0:
            return
        epoch_ms = (
            benchmark_accum["epoch_total"] / benchmark_epochs_accum
            if benchmark_epochs_accum > 0
            else 0.0
        ) * 1000.0
        print("\n[BENCHMARK]")
        print(
            f"  Averaged over {benchmark_epochs_accum} epoch(s): "
            f"{epoch_ms:.3f} ms/epoch total"
        )
        for name in benchmark_order:
            avg_ms = (benchmark_accum[name] / benchmark_epochs_accum) * 1000.0
            pct = 100.0 * avg_ms / max(epoch_ms, 1e-12)
            print(f"  {name:38s} {avg_ms:10.3f} ms/epoch  ({pct:6.2f}%)")

    def should_log_epoch(epoch_zero_based: int) -> bool:
        """Return True when this epoch should emit periodic logs and save frames."""
        if not CONFIG.get("adaptive_log_interval", False):
            return epoch_zero_based % CONFIG["log_interval"] == 0

        epoch_one_based = epoch_zero_based + 1
        if epoch_one_based <= 10:
            return True
        if epoch_one_based <= 100:
            return epoch_one_based % 10 == 0
        return epoch_zero_based % CONFIG["log_interval"] == 0

    print(f"\nOptimizer: Adam(lr={CONFIG['lr']})")
    print(f"Training for {CONFIG['num_epochs']} epochs...\n")

    try:
        for epoch in range(CONFIG["num_epochs"]):
            epoch_core_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            network.train()
            optimizer.zero_grad()
            mag_log10_abs_sum_total = 0.0
            mag_log10_count_total = 0
            mag_log10_abs_sum_per_key = {}
            mag_log10_count_per_key = {}

            should_log_now = should_log_epoch(epoch)
            epoch_lr = optimizer.param_groups[0]["lr"]

            # Forward pass (suppress detailed logging during training)
            if should_log_now:
                print(f"\n{'=' * 60}")
                print(f"EPOCH {epoch + 1}/{CONFIG['num_epochs']}  |  lr={epoch_lr:.6e}")
                print(f"{'=' * 60}")

            # Temporarily suppress forward pass logging
            verbose = should_log_now and CONFIG["log_forward"]

            log_activations = should_log_now and CONFIG.get(
                "log_activations_wandb", False
            )

            if not verbose and not CONFIG["verbose_forward"]:
                # Silence print by redirecting to nowhere temporarily
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")

            t_forward_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            pred_raw = network(
                node_type_idx,
                edge_type_idx,
                edge_index,
                edge_shift,
                edge_length_emb,
                edge_sh,
                batch_node,
                batch_edge,
                log_to_wandb=log_activations,
            )
            if CONFIG["benchmark"]:
                benchmark_add("forward_total", time.perf_counter() - t_forward_start)

            if not verbose and not CONFIG["verbose_forward"]:
                sys.stdout.close()
                sys.stdout = old_stdout

            # Wrap predictions into IrrepsBlockData then convert to matrix blocks

            t_pack_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            pair_vec_H = {}
            pair_edges_dict = {}
            lookup_dict = {}

            for key, payload in pred_raw.items():
                pair_vec_H[key] = payload["vectors"]
                pair_edges_dict[key] = payload["edges"]

                for idx, edge_5d in enumerate(payload["edges"].t()):
                    sx, sy, sz, i, j = edge_5d.tolist()
                    lookup_dict[(int(sx), int(sy), int(sz), int(i), int(j))] = (
                        key,
                        idx,
                    )
            if CONFIG["benchmark"]:
                benchmark_add(
                    "pred_pack_raw_outputs", time.perf_counter() - t_pack_start
                )

            pred_H_irreps = IrrepsBlockData(
                atoms=tuple(atoms_list),
                atom_counts=Counter(atoms_list),
                pair_vectors=pair_vec_H,
                pair_edges=pair_edges_dict,
                lookup=lookup_dict,
                orbital_cfg=orbital_cfg,
                basis=target_H_matrix.basis,
            )

            # Convert to matrix blocks (train_target = "matrix")
            t_to_blocks_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            pred_H_matrix_norm = pred_H_irreps.to_blocks(mapper)
            if CONFIG["benchmark"]:
                benchmark_add(
                    "pred_to_blocks_norm", time.perf_counter() - t_to_blocks_start
                )
            if CONFIG["magnitude_factorization"]:
                pred_H_matrix_actual = reconstruct_actual_prediction_from_magnitudes(
                    pred_H_matrix_norm,
                    pred_raw,
                    require_magnitudes=True,
                )
            else:
                pred_H_matrix_actual = pred_H_matrix_norm

            # Use symmetrized prediction for all metrics/reporting.
            pred_H_matrix_metrics = (
                pred_H_matrix_actual + pred_H_matrix_actual.transpose()
            ) * 0.5

            # Compute loss - either standard or per-irrep decomposed
            t_loss_block_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            loss_block_irrep_filter_to_blocks_time = 0.0
            loss_block_norm_scaling_time = 0.0
            if CONFIG["train_on_irrep_parts"]:
                # Per-irrep decomposed loss
                loss_block = torch.tensor(0.0, device=device)
                irrep_losses = {}

                for irrep in all_irreps:
                    t_irrep_ftb_start = (
                        time.perf_counter() if CONFIG["benchmark"] else 0.0
                    )
                    # Filter both prediction and target by this irrep
                    pred_irrep_filtered = filter_irreps_block_data_by_irrep(
                        pred_H_irreps, irrep, mapper
                    )
                    target_irrep_filtered = filter_irreps_block_data_by_irrep(
                        target_H_irreps_for_block_loss, irrep, mapper
                    )

                    # Convert to matrix blocks
                    pred_irrep_blocks = pred_irrep_filtered.to_blocks(mapper)
                    target_irrep_blocks = target_irrep_filtered.to_blocks(mapper)
                    if CONFIG["benchmark"]:
                        loss_block_irrep_filter_to_blocks_time += (
                            time.perf_counter() - t_irrep_ftb_start
                        )

                    # Apply partial_train filtering
                    filtered_target_irrep = filter_blocks_by_partial_train(
                        target_irrep_blocks, CONFIG["partial_train"]
                    )

                    # Compute loss for this irrep
                    irrep_loss = torch.tensor(0.0, device=device)

                    for key in target_irrep_blocks.pair_blocks.keys():
                        if key in pred_irrep_blocks.pair_blocks:
                            targ_blocks_full = target_irrep_blocks.pair_blocks[key]
                            _, mask = filtered_target_irrep[key]

                            pred_blocks = pred_irrep_blocks.pair_blocks[key]
                            min_n = min(pred_blocks.shape[0], targ_blocks_full.shape[0])

                            mask = mask[:min_n]
                            if mask.any():
                                pred_blocks_filtered = pred_blocks[:min_n][mask]
                                targ_blocks_filtered = targ_blocks_full[:min_n][mask]

                                # Apply block normalization if enabled
                                if (
                                    CONFIG["normalize_blocks"]
                                    and norm_factors is not None
                                    and key in norm_factors
                                ):
                                    t_norm_scale_start = (
                                        time.perf_counter()
                                        if CONFIG["benchmark"]
                                        else 0.0
                                    )
                                    # For per-irrep case, we need to use the original target's edges
                                    # because the irrep-filtered blocks still refer to the same edges
                                    edges_full = target_H_matrix.pair_edges[
                                        key
                                    ]  # (5, num_edges)
                                    norm_vec = torch.ones(
                                        min_n, device=pred_blocks.device
                                    )
                                    for edge_idx in range(min_n):
                                        status = get_block_status(edges_full, edge_idx)
                                        norm_vec[edge_idx] = norm_factors[key][status]

                                    # Select norm factors corresponding to mask
                                    norm_vec_selected = norm_vec[mask]
                                    # Reshape for broadcasting: (num_selected,) -> (num_selected, 1, 1, ...)
                                    while (
                                        norm_vec_selected.ndim
                                        < pred_blocks_filtered.ndim
                                    ):
                                        norm_vec_selected = norm_vec_selected.unsqueeze(
                                            -1
                                        )

                                    pred_blocks_filtered = (
                                        pred_blocks_filtered / norm_vec_selected
                                    )
                                    targ_blocks_filtered = (
                                        targ_blocks_filtered / norm_vec_selected
                                    )
                                    if CONFIG["benchmark"]:
                                        loss_block_norm_scaling_time += (
                                            time.perf_counter() - t_norm_scale_start
                                        )

                                irrep_loss += F.mse_loss(
                                    pred_blocks_filtered, targ_blocks_filtered
                                )

                    irrep_str = str(irrep)
                    irrep_losses[irrep_str] = irrep_loss
                    loss_block += irrep_loss
            else:
                # Standard loss computation
                loss_block = torch.tensor(0.0, device=device)
                filtered_target = filter_blocks_by_partial_train(
                    target_H_matrix_for_block_loss, CONFIG["partial_train"]
                )

                for key in target_H_matrix_for_block_loss.pair_blocks.keys():
                    if key in pred_H_matrix_norm.pair_blocks:
                        # Get filtered target blocks and mask
                        targ_blocks_full = target_H_matrix_for_block_loss.pair_blocks[
                            key
                        ]
                        _, mask = filtered_target[key]

                        # Match sizes (predictions might have fewer edges due to cutoff)
                        pred_blocks = pred_H_matrix_norm.pair_blocks[key]
                        min_n = min(pred_blocks.shape[0], targ_blocks_full.shape[0])

                        # Apply mask to select only relevant blocks
                        mask = mask[:min_n]
                        if mask.any():
                            pred_blocks_filtered = pred_blocks[:min_n][mask]
                            targ_blocks_filtered = targ_blocks_full[:min_n][mask]

                            # Apply block normalization if enabled
                            if (
                                CONFIG["normalize_blocks"]
                                and norm_factors is not None
                                and key in norm_factors
                            ):
                                t_norm_scale_start = (
                                    time.perf_counter() if CONFIG["benchmark"] else 0.0
                                )
                                # Create normalization factors for each edge
                                edges_full = target_H_matrix.pair_edges[
                                    key
                                ]  # (5, num_edges)
                                norm_vec = torch.ones(min_n, device=pred_blocks.device)
                                for edge_idx in range(min_n):
                                    status = get_block_status(edges_full, edge_idx)
                                    norm_vec[edge_idx] = norm_factors[key][status]

                                # Select norm factors corresponding to mask
                                norm_vec_selected = norm_vec[mask]
                                # Reshape for broadcasting: (num_selected,) -> (num_selected, 1, 1, ...)
                                while (
                                    norm_vec_selected.ndim < pred_blocks_filtered.ndim
                                ):
                                    norm_vec_selected = norm_vec_selected.unsqueeze(-1)

                                pred_blocks_filtered = (
                                    pred_blocks_filtered / norm_vec_selected
                                )
                                targ_blocks_filtered = (
                                    targ_blocks_filtered / norm_vec_selected
                                )
                                if CONFIG["benchmark"]:
                                    loss_block_norm_scaling_time += (
                                        time.perf_counter() - t_norm_scale_start
                                    )

                            loss_block += F.mse_loss(
                                pred_blocks_filtered, targ_blocks_filtered
                            )
            if CONFIG["benchmark"]:
                benchmark_add(
                    "loss_block_total", time.perf_counter() - t_loss_block_start
                )
                benchmark_add(
                    "loss_block_irrep_filter_and_to_blocks",
                    loss_block_irrep_filter_to_blocks_time,
                )
                benchmark_add(
                    "loss_block_norm_scaling_overhead",
                    loss_block_norm_scaling_time,
                )

            loss_magnitude = torch.tensor(0.0, device=device)
            if CONFIG["magnitude_factorization"]:
                filtered_target_for_magnitude = filter_blocks_by_partial_train(
                    target_H_matrix, CONFIG["partial_train"]
                )
                for key in target_H_matrix.pair_blocks.keys():
                    if key not in pred_raw:
                        continue
                    if "magnitudes" not in pred_raw[key]:
                        raise RuntimeError(
                            f"Missing predicted magnitudes for key '{key}' in magnitude-factorization mode."
                        )
                    if target_block_norms is None or key not in target_block_norms:
                        continue

                    pred_magnitudes = pred_raw[key]["magnitudes"]
                    target_magnitudes = target_block_norms[key]
                    _, mask = filtered_target_for_magnitude[key]

                    min_n = min(
                        pred_magnitudes.shape[0],
                        target_magnitudes.shape[0],
                        mask.shape[0],
                    )
                    mask = mask[:min_n]
                    if mask.any():
                        pred_sel = pred_magnitudes[:min_n][mask]
                        target_sel = target_magnitudes[:min_n][mask]
                        loss_magnitude += F.mse_loss(pred_sel, target_sel)

                        log10_diff = torch.abs(
                            torch.log10(torch.clamp(pred_sel, min=1e-12))
                            - torch.log10(torch.clamp(target_sel, min=1e-12))
                        )
                        log10_abs_sum = float(log10_diff.sum().item())
                        log10_count = int(log10_diff.numel())
                        mag_log10_abs_sum_total += log10_abs_sum
                        mag_log10_count_total += log10_count
                        mag_log10_abs_sum_per_key[key] = (
                            mag_log10_abs_sum_per_key.get(key, 0.0) + log10_abs_sum
                        )
                        mag_log10_count_per_key[key] = (
                            mag_log10_count_per_key.get(key, 0) + log10_count
                        )

            # Keep Hamiltonian block loss tracking consistent with previous history key.
            loss_H = loss_block
            if CONFIG["magnitude_factorization"]:
                loss = loss_block + CONFIG["magnitude_lambda"] * loss_magnitude
            else:
                loss = loss_block

            # Step scheduler (ReduceLROnPlateau needs validation loss, so we use training loss here)
            scheduler.step(loss)

            # Log current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"lr": current_lr, "epoch": epoch})
            # Check for NaN or Inf in loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                failure_payload = log_invalid_loss_failure(
                    epoch_zero_based=epoch,
                    loss_value=loss.item(),
                    is_nan=bool(torch.isnan(loss)),
                    last_valid_loss=history["loss"][-1] if history["loss"] else "N/A",
                    best_epoch_zero_based=best_epoch,
                    best_loss=best_loss,
                )
                wandb.log(failure_payload)
                break

            # Backward
            t_backward_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            loss.backward()
            if CONFIG["benchmark"]:
                benchmark_add("backward_total", time.perf_counter() - t_backward_start)

            # Gradient clipping to prevent exploding gradients
            if CONFIG["grad_clip"] > 0:
                grad_norm = clip_grad_norm_(network.parameters(), CONFIG["grad_clip"])
                # Log gradient norm periodically
                if should_log_now:
                    wandb.log({"grad_norm": grad_norm.item(), "epoch": epoch})

            # Check for NaN/Inf in gradients
            has_nan_grad = False
            for name, param in network.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        failure_payload = log_invalid_gradient_failure(
                            epoch_zero_based=epoch,
                            param_name=name,
                            lr=CONFIG["lr"],
                            grad_clip=CONFIG["grad_clip"],
                            last_valid_loss=(
                                history["loss"][-1] if history["loss"] else "N/A"
                            ),
                            best_epoch_zero_based=best_epoch,
                            best_loss=best_loss,
                        )
                        wandb.log(failure_payload)
                        has_nan_grad = True
                        break

            if has_nan_grad:
                break

            t_optimizer_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            optimizer.step()
            if CONFIG["benchmark"]:
                benchmark_add("optimizer_step", time.perf_counter() - t_optimizer_start)

            # Log training loss at every step
            step_log = {
                "train/loss_step": loss.item(),
                "train/loss_total": loss.item(),
                "epoch": epoch,
                "lr": current_lr,
            }
            if CONFIG["magnitude_factorization"]:
                step_log["train/loss_block_norm"] = loss_block.item()
                step_log["train/loss_magnitude_weighted"] = (
                    CONFIG["magnitude_lambda"] * loss_magnitude.item()
                )
                step_log["train/loss_magnitude_raw"] = loss_magnitude.item()
                if mag_log10_count_total > 0:
                    step_log["train/magnitude_log10_mae"] = (
                        mag_log10_abs_sum_total / mag_log10_count_total
                    )

            # Log per-irrep losses if enabled
            if CONFIG["train_on_irrep_parts"]:
                for irrep_str, irrep_loss in irrep_losses.items():
                    step_log[f"partial/{irrep_str}"] = irrep_loss.item()

            t_step_wandb_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            wandb.log(step_log)
            if CONFIG["benchmark"]:
                benchmark_add(
                    "step_wandb_log", time.perf_counter() - t_step_wandb_start
                )
                benchmark_add("epoch_total", time.perf_counter() - epoch_core_start)
                benchmark_epochs_accum += 1

            # Logging
            if should_log_now:
                # Compute detailed metrics with filtering
                # First filter both predictions and targets
                filtered_pred = filter_blocks_by_partial_train(
                    pred_H_matrix_metrics, CONFIG["partial_train"]
                )
                filtered_target = filter_blocks_by_partial_train(
                    target_H_matrix, CONFIG["partial_train"]
                )
                filtered_overlap = filter_blocks_by_partial_train(
                    overlap_e3nn, CONFIG["partial_train"]
                )

                # Create filtered versions for metrics computation

                pred_filtered_blocks = {}
                target_filtered_blocks = {}
                overlap_filtered_blocks = {}

                for key in target_H_matrix.pair_blocks.keys():
                    if key in pred_H_matrix_metrics.pair_blocks:
                        pred_full, pred_mask = filtered_pred[key]
                        targ_full, targ_mask = filtered_target[key]
                        ovlp_full, ovlp_mask = filtered_overlap[key]

                        min_n = min(
                            pred_full.shape[0], targ_full.shape[0], ovlp_full.shape[0]
                        )
                        mask = pred_mask[:min_n] & targ_mask[:min_n] & ovlp_mask[:min_n]

                        if mask.any():
                            pred_filtered_blocks[key] = pred_full[:min_n][mask]
                            target_filtered_blocks[key] = targ_full[:min_n][mask]
                            overlap_filtered_blocks[key] = ovlp_full[:min_n][mask]

                # Create temporary BlockMatrix objects for metrics computation
                pred_filtered = BlockMatrix(
                    atoms=target_H_matrix.atoms,
                    atom_counts=target_H_matrix.atom_counts,
                    pair_blocks=pred_filtered_blocks,
                    pair_edges={
                        key: target_H_matrix.pair_edges[key]
                        for key in pred_filtered_blocks.keys()
                    },
                    lookup=target_H_matrix.lookup,
                    orbital_cfg=target_H_matrix.orbital_cfg,
                    basis=target_H_matrix.basis,
                )
                target_filtered = BlockMatrix(
                    atoms=target_H_matrix.atoms,
                    atom_counts=target_H_matrix.atom_counts,
                    pair_blocks=target_filtered_blocks,
                    pair_edges={
                        key: target_H_matrix.pair_edges[key]
                        for key in target_filtered_blocks.keys()
                    },
                    lookup=target_H_matrix.lookup,
                    orbital_cfg=target_H_matrix.orbital_cfg,
                    basis=target_H_matrix.basis,
                )
                overlap_filtered = BlockMatrix(
                    atoms=overlap_e3nn.atoms,
                    atom_counts=overlap_e3nn.atom_counts,
                    pair_blocks=overlap_filtered_blocks,
                    pair_edges={
                        key: overlap_e3nn.pair_edges[key]
                        for key in overlap_filtered_blocks.keys()
                    },
                    lookup=overlap_e3nn.lookup,
                    orbital_cfg=overlap_e3nn.orbital_cfg,
                    basis=overlap_e3nn.basis,
                )

                detailed_metrics = compute_detailed_metrics(
                    pred_filtered, target_filtered, overlap_filtered
                )

                # mae_H for history
                mae_H = detailed_metrics["mae"]

                history["loss"].append(loss.item())
                history["mse_H"].append(loss_H.item())
                history["mae_H"].append(mae_H)

                # Calculate timing
                current_time = time.time()
                time_elapsed = current_time - last_log_time
                epochs_since_last_log = (
                    (epoch - last_logged_epoch)
                    if last_logged_epoch >= 0
                    else (epoch + 1)
                )
                avg_epoch_time = time_elapsed / epochs_since_last_log
                last_log_time = current_time
                last_logged_epoch = epoch

                log_detailed_training_metrics(
                    avg_epoch_time=avg_epoch_time,
                    epochs_since_last_log=epochs_since_last_log,
                    time_elapsed=time_elapsed,
                    loss_value=loss.item(),
                    detailed_metrics=detailed_metrics,
                    irrep_losses=(
                        irrep_losses if CONFIG["train_on_irrep_parts"] else None
                    ),
                )
                if CONFIG["magnitude_factorization"]:
                    loss_block_val = float(loss_block.item())
                    loss_mag_weighted_val = float(
                        CONFIG["magnitude_lambda"] * loss_magnitude.item()
                    )
                    total_loss_val = max(float(loss.item()), 1e-12)
                    block_pct = 100.0 * loss_block_val / total_loss_val
                    mag_pct = 100.0 * loss_mag_weighted_val / total_loss_val

                    print("\n  Factorization Loss Breakdown:")
                    print(
                        f"    Block loss (normalized blocks): {loss_block_val:.6e} ({block_pct:.2f}%)"
                    )
                    print(
                        f"    Magnitude loss (weighted):      {loss_mag_weighted_val:.6e} ({mag_pct:.2f}%)"
                    )
                    if mag_log10_count_total > 0:
                        total_log10_mae = mag_log10_abs_sum_total / max(
                            mag_log10_count_total, 1
                        )
                        print(
                            f"    log10 magnitude MAE (total):    {total_log10_mae:.6e}"
                        )
                        print("    log10 magnitude MAE (per key):")
                        for key in sorted(mag_log10_abs_sum_per_key.keys()):
                            key_count = max(mag_log10_count_per_key.get(key, 0), 1)
                            key_mae = mag_log10_abs_sum_per_key[key] / key_count
                            print(f"      {key}: {key_mae:.6e}")
                    else:
                        print("    log10 magnitude MAE: N/A (no selected blocks)")

                if CONFIG["benchmark"]:
                    print_benchmark_report()
                    benchmark_accum = {k: 0.0 for k in benchmark_order}
                    benchmark_epochs_accum = 0

                # Log to WandB
                wandb.log(
                    build_wandb_detailed_metrics_log(
                        epoch_zero_based=epoch,
                        loss_value=loss.item(),
                        detailed_metrics=detailed_metrics,
                    )
                )

                # Save best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch
                    best_model_path = run_checkpoint_dir / "best_model.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": network.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss.item(),
                            "mae_H": mae_H,
                            "config": CONFIG,
                            "metrics": detailed_metrics,
                        },
                        best_model_path,
                    )
                    print(
                        f"[OK] Best model saved to {best_model_path.name} (loss: {loss.item():.6e})"
                    )

                pred_H_irreps_metrics = pred_H_matrix_metrics.to_vectors(mapper)
                per_irrep_metrics = compute_irrep_metrics(
                    pred_H_irreps_metrics, target_H_irreps, all_irreps, mapper
                )
                if CONFIG["log_per_irrep_metrics"]:
                    log_per_irrep_metrics(
                        "Per-Irrep Metrics (Element, Block, and Full Matrix):",
                        all_irreps,
                        per_irrep_metrics,
                    )
                wandb.log(
                    build_wandb_per_irrep_metrics_log(
                        epoch_zero_based=epoch,
                        all_irreps=all_irreps,
                        per_irrep_metrics=per_irrep_metrics,
                    )
                )

                # Save frame for video at every log interval
                try:
                    save_hamiltonian_frame_to_disk(
                        pred_H_matrix_metrics,
                        target_H_matrix,
                        overlap_e3nn,
                        list(snapshot.hamiltonian.atoms),
                        orbital_cfg,
                        frame_output_dir,
                        epoch,
                        sx=0,
                        sy=0,
                        sz=0,
                        dynamic_range=False,
                        diff_dynamic_range=True,
                        partial_train=CONFIG["partial_train"],
                        percentile=99.0,
                    )
                except Exception as e:
                    print(f"[WARN] Could not save frame for epoch {epoch}: {e}")

                # Check for convergence
                if loss.item() < 1e-10:
                    print("")
                    print(f"[OK] Converged! Loss below 1e-10 at epoch {epoch + 1}")
                    break
    except KeyboardInterrupt:
        print(
            "\n[INFO] Training interrupted by user (Ctrl-C). Proceeding to final evaluation and saving..."
        )

    # =============================================================================
    # FINAL EVALUATION
    # =============================================================================
    print("")
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    network.eval()
    with torch.no_grad():
        pred_raw = network(
            node_type_idx,
            edge_type_idx,
            edge_index,
            edge_shift,
            edge_length_emb,
            edge_sh,
            batch_node,
            batch_edge,
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
            basis=target_H_matrix.basis,
        )

        # Convert to blocks (normalized block prediction in magnitude-factorization mode)
        pred_H_matrix_norm = pred_H_irreps.to_blocks(mapper)
        if CONFIG["magnitude_factorization"]:
            pred_H_matrix_actual = reconstruct_actual_prediction_from_magnitudes(
                pred_H_matrix_norm,
                pred_raw,
                require_magnitudes=True,
            )
        else:
            pred_H_matrix_actual = pred_H_matrix_norm

        # Use symmetrized prediction for final metrics/reporting.
        pred_H_matrix_metrics = (
            pred_H_matrix_actual + pred_H_matrix_actual.transpose()
        ) * 0.5

        # Filter for partial training if needed
        filtered_pred = filter_blocks_by_partial_train(
            pred_H_matrix_metrics, CONFIG["partial_train"]
        )
        filtered_target = filter_blocks_by_partial_train(
            target_H_matrix, CONFIG["partial_train"]
        )
        filtered_overlap = filter_blocks_by_partial_train(
            overlap_e3nn, CONFIG["partial_train"]
        )

        # Create filtered versions for metrics computation
        # ...existing code...

        pred_filtered_blocks = {}
        target_filtered_blocks = {}
        overlap_filtered_blocks = {}

        for key in target_H_matrix.pair_blocks.keys():
            if key in pred_H_matrix_metrics.pair_blocks:
                pred_full, pred_mask = filtered_pred[key]
                targ_full, targ_mask = filtered_target[key]
                ovlp_full, ovlp_mask = filtered_overlap[key]

                min_n = min(pred_full.shape[0], targ_full.shape[0], ovlp_full.shape[0])
                mask = pred_mask[:min_n] & targ_mask[:min_n] & ovlp_mask[:min_n]

                if mask.any():
                    pred_filtered_blocks[key] = pred_full[:min_n][mask]
                    target_filtered_blocks[key] = targ_full[:min_n][mask]
                    overlap_filtered_blocks[key] = ovlp_full[:min_n][mask]

        # Create temporary BlockMatrix objects for metrics computation
        pred_filtered = BlockMatrix(
            atoms=target_H_matrix.atoms,
            atom_counts=target_H_matrix.atom_counts,
            pair_blocks=pred_filtered_blocks,
            pair_edges={
                key: target_H_matrix.pair_edges[key]
                for key in pred_filtered_blocks.keys()
            },
            lookup=target_H_matrix.lookup,
            orbital_cfg=target_H_matrix.orbital_cfg,
            basis=target_H_matrix.basis,
        )
        target_filtered = BlockMatrix(
            atoms=target_H_matrix.atoms,
            atom_counts=target_H_matrix.atom_counts,
            pair_blocks=target_filtered_blocks,
            pair_edges={
                key: target_H_matrix.pair_edges[key]
                for key in target_filtered_blocks.keys()
            },
            lookup=target_H_matrix.lookup,
            orbital_cfg=target_H_matrix.orbital_cfg,
            basis=target_H_matrix.basis,
        )
        overlap_filtered = BlockMatrix(
            atoms=overlap_e3nn.atoms,
            atom_counts=overlap_e3nn.atom_counts,
            pair_blocks=overlap_filtered_blocks,
            pair_edges={
                key: overlap_e3nn.pair_edges[key]
                for key in overlap_filtered_blocks.keys()
            },
            lookup=overlap_e3nn.lookup,
            orbital_cfg=overlap_e3nn.orbital_cfg,
            basis=overlap_e3nn.basis,
        )

        # Compute detailed metrics for final evaluation
        final_detailed_metrics = compute_detailed_metrics(
            pred_filtered, target_filtered, overlap_filtered
        )

        log_final_metrics(final_detailed_metrics)

        final_metrics = {
            "final/mae_H": final_detailed_metrics["mae"],
            "final/mse_H": final_detailed_metrics["mse"],
            "final/mae_H_mod": final_detailed_metrics["mae_mod"],
            "final/mse_H_mod": final_detailed_metrics["mse_mod"],
            "final/mu_H": final_detailed_metrics["mu_H"],
            "final/correction_mae": final_detailed_metrics["correction_mae"],
            "final/correction_mse": final_detailed_metrics["correction_mse"],
        }

        # DOS comparison
        dos_plot_path = run_checkpoint_dir / "dos_comparison_final.png"
        try:
            dos_metrics = save_dos_comparison_plot(
                H_pred=pred_H_matrix_metrics,
                H_gt=target_H_matrix,
                S=overlap_e3nn,
                output_path=dos_plot_path,
                sigma=0.2,
                bin_width=0.1,
                title="DOS Comparison",
            )
            final_metrics.update(
                {
                    "final/eig_abs_mean": dos_metrics["eig_abs_mean"],
                    "final/eig_abs_max": dos_metrics["eig_abs_max"],
                    "final/eig_rel_mean": dos_metrics["eig_rel_mean"],
                    "final/eig_rel_max": dos_metrics["eig_rel_max"],
                    "final/dos_mae": dos_metrics["dos_mae"],
                    "final/dos_mse": dos_metrics["dos_mse"],
                    "final/dos_max_abs": dos_metrics["dos_max_abs"],
                }
            )
            wandb.log({"final/dos_comparison_plot": wandb.Image(str(dos_plot_path))})
            print(
                f"  DOS plot saved: {dos_plot_path} "
                f"(dos_mae={dos_metrics['dos_mae']:.6e})"
            )
        except Exception as e:
            print(f"[WARN] Could not generate DOS comparison plot: {e}")

        print("\n  Per-block Metrics:")
        for key in pred_filtered_blocks.keys():
            pred_block = pred_filtered_blocks[key]
            true_block = target_filtered_blocks[key]

            block_mse = F.mse_loss(pred_block, true_block).item()
            block_mae = torch.mean(torch.abs(pred_block - true_block)).item()

            pair_irreps = mapper.get_pair_irreps(key)

            # Get mask info for reporting
            _, mask = filtered_pred[key]
            num_selected = (
                mask[: pred_H_matrix_metrics.pair_blocks[key].shape[0]].sum().item()
                if key in pred_H_matrix_metrics.pair_blocks
                else 0
            )
            num_total = (
                pred_H_matrix_metrics.pair_blocks[key].shape[0]
                if key in pred_H_matrix_metrics.pair_blocks
                else 0
            )

            print(f"\n  {key}:")
            print(f"    Block shape: {pred_block.shape}, Irreps: {pair_irreps}")
            if CONFIG["partial_train"] is not None:
                print(
                    f"    Selected blocks: {num_selected}/{num_total} ({CONFIG['partial_train']})"
                )
            print(f"    MSE: {block_mse:.6e}")
            print(f"    MAE: {block_mae:.6e}")
            print(
                f"    Relative error: {block_mae / (torch.abs(true_block).mean().item() + 1e-10):.6%}"
            )

            final_metrics[f"final/{key}_mse"] = block_mse
            final_metrics[f"final/{key}_mae"] = block_mae

        # Log final metrics to WandB
        wandb.log(final_metrics)

        # Distance-binned error curves (64 bins)
        print("\n  Distance-binned error curves (64 bins):")
        distance_curve = compute_distance_error_curve(
            H_pred=pred_H_matrix_metrics,
            H_gt=target_H_matrix,
            positions=positions,
            box=box,
            partial_train=CONFIG["partial_train"],
            n_bins=64,
        )
        if distance_curve is not None:
            curve_json_path = run_checkpoint_dir / "distance_error_curve.json"
            with open(curve_json_path, "w") as f:
                json.dump(distance_curve, f, indent=2)

            curve_plot_path = run_checkpoint_dir / "distance_error_curve.png"
            save_distance_error_curve_plot(
                distance_curve,
                curve_plot_path,
                title="Distance Error Curves (Final, 64 bins)",
            )

            # Print a concise summary
            l1_abs = [x for x in distance_curve["l1_abs"] if x == x]
            l2_abs = [x for x in distance_curve["l2_abs"] if x == x]
            l1_rel = [x for x in distance_curve["l1_rel"] if x == x]
            l2_rel = [x for x in distance_curve["l2_rel"] if x == x]
            if l1_abs and l2_abs and l1_rel and l2_rel:
                print(
                    f"    L1 abs range: {min(l1_abs):.3e} .. {max(l1_abs):.3e}, "
                    f"L2 abs range: {min(l2_abs):.3e} .. {max(l2_abs):.3e}"
                )
                print(
                    f"    L1 rel range: {min(l1_rel):.3e} .. {max(l1_rel):.3e}, "
                    f"L2 rel range: {min(l2_rel):.3e} .. {max(l2_rel):.3e}"
                )

            wandb.log(
                {
                    "distance_curve/plot": wandb.Image(str(curve_plot_path)),
                }
            )
            print(f"    Saved: {curve_json_path}")
            print(f"    Saved: {curve_plot_path}")
        else:
            print("    No matched edges found for distance-curve computation.")

        # Per-irrep visualizations (k-range=0)
        if CONFIG["log_per_irrep_images"]:
            irrep_output_dir = run_checkpoint_dir / "per_irrep_images"
            irrep_output_dir.mkdir(parents=True, exist_ok=True)
            print("\n  Per-irrep visualizations (k-range=0, percentile=99):")

            for irrep in all_irreps:
                irrep_str = str(irrep)
                try:
                    pred_irrep = split_hamiltonian_by_irrep(
                        pred_H_matrix_metrics, mapper, irrep_str
                    )
                    target_irrep = split_hamiltonian_by_irrep(
                        target_H_matrix, mapper, irrep_str
                    )

                    visualize_hamiltonians(
                        pred_irrep,
                        target_irrep,
                        overlap_e3nn,
                        list(snapshot.hamiltonian.atoms),
                        orbital_cfg,
                        k_range=0,
                        output_dir=irrep_output_dir,
                        dynamic_range=True,
                        diff_dynamic_range=True,
                        per_panel_dynamic_range=True,
                        partial_train=CONFIG["partial_train"],
                        filename_prefix=f"hamiltonian_{irrep_str}",
                        percentile=99.0,
                    )

                    image_path = (
                        irrep_output_dir / f"hamiltonian_{irrep_str}_sx+0_sy+0_sz+0.png"
                    )
                    if image_path.exists():
                        wandb.log(
                            {f"irrep_images/{irrep_str}": wandb.Image(str(image_path))}
                        )
                    else:
                        print(
                            f"    [WARN] Missing image for irrep {irrep_str}: {image_path.name}"
                        )
                except Exception as e:
                    print(f"    [WARN] Irrep {irrep_str} visualization failed: {e}")

    # Save final model
    final_model_path = run_checkpoint_dir / "final_model.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": history["loss"][-1],
            "mae_H": history["mae_H"][-1],
            "config": CONFIG,
            "history": history,
            "final_metrics": final_detailed_metrics,
        },
        final_model_path,
    )

    log_study_complete(
        run_name=run_name,
        total_training_epochs=epoch + 1,
        final_loss=history["loss"][-1],
        best_loss=min(history["loss"]),
        best_epoch=best_epoch + 1,
        run_checkpoint_dir=run_checkpoint_dir,
        final_model_path=final_model_path,
    )

    # Compile frames to video and log to WandB
    if CONFIG["generate_video"]:
        print("")
        print("=" * 80)
        print("FINAL VIDEO GENERATION")
        print("=" * 80)
        try:
            video_path = run_checkpoint_dir / "training_progress.mp4"
            if (frame_output_dir).glob("frame_epoch_*.png"):
                compile_frames_to_video(
                    frame_output_dir,
                    video_path,
                    fps=5,
                    pattern="frame_epoch_*.png",
                    format="mp4",
                )
                # Log final video to WandB (same key, overwrites previous)
                wandb.log(
                    {
                        "training_video": wandb.Video(
                            str(video_path), fps=5, format="mp4"
                        )
                    }
                )
                print("[OK] Training video logged to WandB")
            else:
                print("[WARN] No frames found for video generation")
        except Exception as e:
            print(f"[WARN] Could not generate final video: {e}")
    else:
        print("")
        print("=" * 80)
        print("VIDEO GENERATION SKIPPED (--generate-video not specified)")
        print("=" * 80)

    # Finish WandB run
    wandb.finish()
