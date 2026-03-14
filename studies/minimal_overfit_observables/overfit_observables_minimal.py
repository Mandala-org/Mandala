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
from core.sparse_math import trace_matmul_sparse_block_matrix
from data.block_matrix import IrrepsBlockData
import wandb
from common import (
    MinimalNetwork,
    canonicalize_edge_order,
    compute_basic_matrix_metrics,
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

HARTREE_TO_EV = 27.2113845
UNIT_SCALE_FROM_HARTREE = {
    "hartree": 1.0,
    "ev": HARTREE_TO_EV,
    "mev": HARTREE_TO_EV * 1000.0,
    "100mev": HARTREE_TO_EV * 10.0,
}
UNIT_DISPLAY_NAME = {
    "hartree": "Hartree",
    "ev": "eV",
    "mev": "meV",
    "100mev": "100meV",
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value_norm = str(value).strip().lower()
    if value_norm in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_norm in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


def parse_matrix_targets(value):
    targets = [t.strip().lower() for t in str(value).split(",") if t.strip()]
    if not targets:
        raise argparse.ArgumentTypeError(
            "Expected at least one target in --matrix-targets."
        )
    return targets


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
        "--training-unit",
        type=str.lower,
        default="ev",
        choices=["hartree", "ev", "mev", "100mev"],
        help=(
            "Unit used for Hamiltonian training targets and metrics. "
            "Raw OpenMX Hamiltonian is interpreted as Hartree and scaled to this unit."
        ),
    )
    parser.add_argument(
        "--matrix-targets",
        type=parse_matrix_targets,
        default=parse_matrix_targets("hamiltonian,overlap,density"),
        help="Comma-separated matrix targets to predict/train (subset of: hamiltonian,overlap,density).",
    )
    parser.add_argument(
        "--enable-energy",
        type=parse_bool,
        default=True,
        help="Enable energy metric computation from predicted matrices (default: True).",
    )
    parser.add_argument(
        "--enable-num-electrons",
        type=parse_bool,
        default=True,
        help="Enable number-of-electrons metric computation from predicted matrices (default: True).",
    )
    parser.add_argument(
        "--train-on-energy",
        type=parse_bool,
        default=True,
        help="Enable energy loss term (default: True).",
    )
    parser.add_argument(
        "--train-on-num-electrons",
        type=parse_bool,
        default=True,
        help="Enable number-of-electrons loss term (default: True).",
    )
    parser.add_argument(
        "--loss-coef-observables",
        type=float,
        default=1e-5,
        help="Loss coefficient for energy/num-electrons terms (default: 1e-5).",
    )
    parser.add_argument(
        "--enable-forces",
        type=parse_bool,
        default=False,
        help="Enable force-related data/paths.",
    )
    parser.add_argument(
        "--train-on-forces",
        type=parse_bool,
        default=False,
        help="Enable force loss term.",
    )
    parser.add_argument(
        "--loss-coef-forces",
        type=float,
        default=0.0,
        help="Loss coefficient for force term (default: 0.0).",
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
        default=4,
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
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Floating-point dtype for data and model parameters (default: float32)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str("studies/minimal_overfit_observables/checkpoints"),
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
        "--e3layernorm",
        type=parse_bool,
        default=True,
        help="Enable e3LayerNorm in edge encoder and message-passing blocks (default: True)",
    )
    parser.add_argument(
        "--edge-encoder-use-sh-tensor-square",
        action="store_true",
        default=False,
        help=(
            "Use TensorSquare(spherical harmonics) before the edge encoder tensor product "
            "(default: False)."
        ),
    )
    parser.add_argument(
        "--radial-embedding-scale",
        type=str,
        default="none",
        choices=["none", "sqrt_n_radial"],
        help=(
            "Optional extra scaling for radial embeddings. "
            "'none' matches src/DeepH-style; 'sqrt_n_radial' reproduces legacy minimal behavior."
        ),
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
        "--head-use-tensor-square",
        dest="head_use_tensor_square",
        action="store_true",
        default=False,
        help="Use TensorSquare(edge embeddings) as input to the main head projections (default: False)",
    )
    parser.add_argument(
        "--head-e3mlp-layers",
        type=int,
        default=3,
        help="Number of e3nn Linear layers in each head projection E3MLP; Gate is used between layers (default: 3).",
    )
    parser.add_argument(
        "--apply-cutoff-to-targets",
        dest="apply_cutoff_to_targets",
        action="store_true",
        default=True,
        help="Apply cutoff filtering to target matrices (H/S/D) before training and metrics (default: True).",
    )
    parser.add_argument(
        "--no-apply-cutoff-to-targets",
        dest="apply_cutoff_to_targets",
        action="store_false",
        help="Disable cutoff filtering for target matrices.",
    )
    parser.add_argument(
        "--require-exact-edge-match",
        dest="require_exact_edge_match",
        action="store_true",
        default=True,
        help="Require exact edge counts and key sets in strict edge alignment checks (default: True).",
    )
    parser.add_argument(
        "--no-require-exact-edge-match",
        dest="require_exact_edge_match",
        action="store_false",
        help="Disable exact edge matching in strict checks.",
    )

    args = parser.parse_args()
    valid_matrix_targets = {"hamiltonian", "overlap", "density"}
    matrix_targets = list(dict.fromkeys(args.matrix_targets))
    invalid_targets = sorted(set(matrix_targets) - valid_matrix_targets)
    if invalid_targets:
        raise ValueError(
            f"Unsupported --matrix-targets entries: {invalid_targets}. "
            f"Valid targets: {sorted(valid_matrix_targets)}"
        )
    if "hamiltonian" not in matrix_targets:
        raise ValueError(
            "This study currently requires 'hamiltonian' in --matrix-targets "
            "for diagnostics/visualization paths."
        )
    if args.head_e3mlp_layers < 1:
        raise ValueError("--head-e3mlp-layers must be >= 1.")
    if (args.train_on_energy or args.train_on_num_electrons) and (
        args.loss_coef_observables == 0.0
    ):
        raise ValueError(
            "--loss-coef-observables must be non-zero when training on "
            "energy and/or number of electrons."
        )
    if args.train_on_energy and not {"hamiltonian", "density"}.issubset(
        set(matrix_targets)
    ):
        raise ValueError(
            "--train-on-energy requires --matrix-targets to include both "
            "'hamiltonian' and 'density'."
        )
    if args.train_on_num_electrons and not {"overlap", "density"}.issubset(
        set(matrix_targets)
    ):
        raise ValueError(
            "--train-on-num-electrons requires --matrix-targets to include "
            "both 'overlap' and 'density'."
        )
    if args.train_on_forces and not args.enable_forces:
        raise ValueError("--train-on-forces requires --enable-forces True.")
    if args.train_on_forces and args.loss_coef_forces == 0.0:
        raise ValueError(
            "--loss-coef-forces must be non-zero when --train-on-forces is enabled."
        )
    if args.train_on_forces and not {"hamiltonian", "density"}.issubset(
        set(matrix_targets)
    ):
        raise ValueError(
            "--train-on-forces requires --matrix-targets to include both "
            "'hamiltonian' and 'density'."
        )
    torch_dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(torch_dtype)

    # =============================================================================
    # CONFIGURATION
    # =============================================================================
    print("=" * 80)
    print("MINIMAL OBSERVABLES OVERFIT STUDY - EXPLICIT IMPLEMENTATION")
    print("=" * 80)

    CONFIG = {
        # Data
        "data_path": Path(args.data_path),
        "info_path": Path(args.info_path),
        "convention": "e3nn",
        "training_unit": args.training_unit,
        "hamiltonian_scale_from_hartree": UNIT_SCALE_FROM_HARTREE[args.training_unit],
        "matrix_targets": matrix_targets,
        "train_target": "matrix",
        "enable_energy": args.enable_energy,
        "enable_num_electrons": args.enable_num_electrons,
        "train_on_energy": args.train_on_energy,
        "train_on_num_electrons": args.train_on_num_electrons,
        "loss_coef_observables": args.loss_coef_observables,
        "enable_forces": args.enable_forces,
        "train_on_forces": args.train_on_forces,
        "loss_coef_forces": args.loss_coef_forces,
        "orbital_selection": args.orbital_selection,
        "xyz_permutation": "012",
        "change_box": "right",
        "box_convention": "rows",
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
        "separate_shifted_self": args.separate_shifted_self,
        "train_on_irrep_parts": args.train_on_irrep_parts,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "loss_aggregation": "per_key",
        "sh_mode": "aligned",
        "e3layernorm": args.e3layernorm,
        "edge_encoder_use_sh_tensor_square": args.edge_encoder_use_sh_tensor_square,
        "radial_embedding_scale": args.radial_embedding_scale,
        "generate_video": args.generate_video,
        "verbose_forward": args.verbose_forward,
        "head_use_tensor_square": args.head_use_tensor_square,
        "head_e3mlp_layers": args.head_e3mlp_layers,
        "apply_cutoff_to_targets": args.apply_cutoff_to_targets,
        "require_exact_edge_match": args.require_exact_edge_match,
        # Device
        "device": args.device,
        "dtype": args.dtype,
        # Checkpointing
        "checkpoint_dir": Path(args.checkpoint_dir),
    }

    # Initialize WandB
    wandb_project = os.environ.get(
        "WANDB_PROJECT", "mandala-minimal-overfit-observables-study"
    )
    wandb_kwargs = {
        "project": wandb_project,
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
    matrix_frame_output_dirs = {
        "hamiltonian": run_checkpoint_dir / "frames_hamiltonian",
        "overlap": run_checkpoint_dir / "frames_overlap",
        "density": run_checkpoint_dir / "frames_density",
    }
    for d in matrix_frame_output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
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
        dtype=torch_dtype,
    )

    # Filter GT matrices by cutoff so loss/metrics ignore long-range blocks
    # (enabled by default; can be disabled with --no-apply-cutoff-to-targets).
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
    hamiltonian_e3nn = (
        snapshot.hamiltonian.to(device) * CONFIG["hamiltonian_scale_from_hartree"]
    )
    overlap_e3nn = snapshot.overlap.to(device)
    density_e3nn = snapshot.density.to(device)
    if CONFIG["log_data"]:
        print(
            "  Converted Hamiltonian units: "
            f"Hartree -> {UNIT_DISPLAY_NAME[CONFIG['training_unit']]} "
            f"(x{CONFIG['hamiltonian_scale_from_hartree']:.7f})"
        )
    orbital_cfg = snapshot.hamiltonian.orbital_cfg
    positions = snapshot.positions.to(device=device, dtype=torch_dtype)
    box = (
        snapshot.box.to(device=device, dtype=torch_dtype)
        if snapshot.box is not None
        else None
    )
    atoms_list = list(snapshot.hamiltonian.atoms)

    if CONFIG["log_model"]:
        log_orbital_config(orbital_cfg)

    # Create BlockIrrepMapper
    mapper = BlockIrrepMapper(orbital_cfg, device=device, dtype=torch_dtype)
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

    # Store target matrices (train_target is fixed to "matrix" in this study).
    if CONFIG["log_model"]:
        print("\n[TARGETS] Storing target as matrix blocks...")
    target_matrices_all = {
        "hamiltonian": hamiltonian_e3nn,
        "overlap": overlap_e3nn,
        "density": density_e3nn,
    }
    target_matrices = {
        name: target_matrices_all[name] for name in CONFIG["matrix_targets"]
    }
    target_H_matrix = target_matrices["hamiltonian"]
    target_overlap_matrix = target_matrices_all["overlap"]
    target_density_matrix = target_matrices_all["density"]

    def compute_block_matrix_max_distance(block_matrix) -> float:
        max_dist = 0.0
        for edges_5d in block_matrix.pair_edges.values():
            if edges_5d.shape[1] == 0:
                continue
            src = edges_5d[3].long()
            dst = edges_5d[4].long()
            if box is not None:
                shift_float = edges_5d[:3].T.to(dtype=positions.dtype)
                edge_vec_local = positions[dst] - positions[src] + shift_float @ box
            else:
                edge_vec_local = positions[dst] - positions[src]
            edge_dist_local = torch.linalg.norm(edge_vec_local, dim=1)
            if edge_dist_local.numel() > 0:
                max_dist = max(max_dist, float(edge_dist_local.max().item()))
        return max_dist

    # With strict exact edge checks enabled, graph cutoff cannot exceed what is
    # present in target matrices; otherwise graph contains extra edges by design.
    target_max_by_matrix = {
        "hamiltonian": compute_block_matrix_max_distance(target_H_matrix),
        "overlap": compute_block_matrix_max_distance(overlap_e3nn),
        "density": compute_block_matrix_max_distance(density_e3nn),
    }
    if CONFIG["log_data"]:
        print("\n  Target max edge distance by matrix:")
        for name, max_dist in target_max_by_matrix.items():
            print(f"    {name}: {max_dist:.6f} A")

    # Scalar observable targets are derived from the (possibly unit-scaled) matrices.
    energy_target = trace_matmul_sparse_block_matrix(
        target_H_matrix, target_density_matrix
    )
    num_electrons_target = trace_matmul_sparse_block_matrix(
        target_density_matrix, target_overlap_matrix
    )
    BOHR_TO_ANGSTROM = 0.529177210903
    forces_target = None
    if CONFIG["enable_forces"]:
        if snapshot.forces is None:
            raise RuntimeError(
                "--enable-forces was set, but snapshot does not contain force targets."
            )
        # OpenMX forces are parsed in Hartree/Bohr; convert to selected energy-unit/Angstrom.
        force_scale = CONFIG["hamiltonian_scale_from_hartree"] / BOHR_TO_ANGSTROM
        forces_target = (
            snapshot.forces.to(device=device, dtype=torch_dtype) * force_scale
        )
    if CONFIG["log_model"]:
        print("  Target matrix blocks:")
        for matrix_name, target_matrix in target_matrices.items():
            print(f"    [{matrix_name}]")
            for key in target_matrix.pair_blocks.keys():
                block_shape = target_matrix.pair_blocks[key].shape
                edge_shape = target_matrix.pair_edges[key].shape
                pair_irreps = mapper.get_pair_irreps(key)
                print(
                    f"      {key}: blocks {block_shape}, edges {edge_shape}, irreps {pair_irreps}"
                )
        print(f"  Target energy scalar: {float(energy_target.item()):.8e}")
        print(
            "  Target num_electrons scalar: "
            f"{float(num_electrons_target.item()):.8e}"
        )
        if forces_target is not None:
            print(f"  Target forces shape: {tuple(forces_target.shape)}")

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
    # Compute spherical harmonics
    if CONFIG["log_data"]:
        print(f"\n  Computing spherical harmonics (l_max={CONFIG['l_max']})...")
    sh_irreps = Irreps.spherical_harmonics(CONFIG["l_max"])
    if CONFIG["log_data"]:
        print(f"  SH irreps: {sh_irreps}")

    def compute_edge_features(curr_positions: torch.Tensor):
        if box is not None:
            shift_float_local = edge_shift.T.to(dtype=curr_positions.dtype)
            edge_vec_local = (
                curr_positions[edge_index[1]]
                - curr_positions[edge_index[0]]
                + shift_float_local @ box
            )
        else:
            edge_vec_local = (
                curr_positions[edge_index[1]] - curr_positions[edge_index[0]]
            )
        edge_dist_local = torch.linalg.norm(edge_vec_local, dim=1)
        edge_sh_local = spherical_harmonics(
            sh_irreps,
            edge_vec_local,
            normalize=True,
            normalization="component",
        )
        edge_length_emb_local = soft_one_hot_linspace(
            edge_dist_local,
            start=0.0,
            end=CONFIG["cutoff_radius"],
            number=CONFIG["n_radial"],
            basis="gaussian",
            cutoff=False,
        )
        if CONFIG["radial_embedding_scale"] == "sqrt_n_radial":
            edge_length_emb_local = edge_length_emb_local * CONFIG["n_radial"] ** 0.5
        return edge_vec_local, edge_dist_local, edge_sh_local, edge_length_emb_local

    edge_vec, edge_dist, edge_sh, edge_length_emb = compute_edge_features(positions)

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
        print(f"  Edge SH shape: {edge_sh.shape}")
        print(
            f"\n  Computing radial embeddings ({CONFIG['n_radial']} basis functions)..."
        )
        print(f"  Radial embedding scale: {CONFIG['radial_embedding_scale']}")
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
    # HELPER FUNCTIONS FOR LOSS / IRREP PROJECTION
    # =============================================================================
    def build_irrep_projection_cache(irreps_data, all_irreps_list):
        """
        Precompute lightweight projectors to map only one irrep component from
        full irrep vectors directly to matrix blocks.

        Returns:
            dict[irrep_str][key] = (idx, q_subset, dim_i, dim_j)
            where:
              - idx: indices in full irrep vector belonging to irrep_str
              - q_subset: corresponding rows of mapper q matrix
              - dim_i, dim_j: block dimensions for reshape
        """
        cache = {str(ir): {} for ir in all_irreps_list}
        all_irrep_strs = set(cache.keys())

        for key in irreps_data.pair_vectors.keys():
            pair_irreps = mapper.get_pair_irreps(key)
            el_a, el_b = key.split("-", 1)
            map_key = (el_a, el_b)
            itm = mapper._lookup(map_key)
            q_full = mapper._get_q(map_key)

            start = 0
            idx_parts_by_irrep = {}
            for mul, ir in pair_irreps:
                term_dim = mul * ir.dim
                ir_str = str(ir)
                if term_dim > 0 and ir_str in all_irrep_strs:
                    part_idx = torch.arange(
                        start,
                        start + term_dim,
                        device=q_full.device,
                        dtype=torch.long,
                    )
                    idx_parts_by_irrep.setdefault(ir_str, []).append(part_idx)
                start += term_dim

            for ir_str, idx_parts in idx_parts_by_irrep.items():
                idx = (
                    idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=0)
                )
                q_subset = q_full.index_select(0, idx)
                cache[ir_str][key] = (idx, q_subset, itm.dim_i, itm.dim_j)

        return cache

    def project_irrep_vectors_to_blocks(vectors_full, projector):
        """Project selected irrep vector components to block space."""
        idx, q_subset, dim_i, dim_j = projector
        vec_sel = vectors_full.index_select(1, idx)
        if vec_sel.dtype != q_subset.dtype:
            vec_sel = vec_sel.to(dtype=q_subset.dtype)
        block_flat = vec_sel @ q_subset
        return block_flat.view(vec_sel.shape[0], dim_i, dim_j)

    def init_block_loss_accumulator():
        return {
            "loss_sum": torch.tensor(0.0, device=device),
            "sq_sum": torch.tensor(0.0, device=device),
            "count": 0,
        }

    def accumulate_block_loss(acc, pred_blocks_filtered, targ_blocks_filtered):
        if targ_blocks_filtered.dtype != pred_blocks_filtered.dtype:
            targ_blocks_filtered = targ_blocks_filtered.to(pred_blocks_filtered.dtype)
        acc["loss_sum"] = acc["loss_sum"] + F.mse_loss(
            pred_blocks_filtered, targ_blocks_filtered
        )

    def finalize_block_loss(acc):
        return acc["loss_sum"]

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
        matrix_targets=CONFIG["matrix_targets"],
        head_e3mlp_layers=CONFIG["head_e3mlp_layers"],
        edge_encoder_use_sh_tensor_square=CONFIG["edge_encoder_use_sh_tensor_square"],
        use_e3layernorm=CONFIG["e3layernorm"],
        head_use_tensor_square=CONFIG["head_use_tensor_square"],
        separate_shifted_self=CONFIG["separate_shifted_self"],
    ).to(device=device, dtype=torch_dtype)
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
    target_irreps_by_name = {
        matrix_name: target_matrices[matrix_name].to_vectors(mapper)
        for matrix_name in CONFIG["matrix_targets"]
    }
    target_H_irreps = target_irreps_by_name["hamiltonian"]
    all_irreps = get_all_irreps_in_hamiltonian(mapper)
    if CONFIG["log_model"]:
        print(
            f"  Found {len(all_irreps)} unique irreps: {[str(ir) for ir in all_irreps]}"
        )

    # In this study variant we always train against unnormalized, full target blocks.
    target_matrix_for_block_loss_by_name = dict(target_matrices)

    target_irrep_blocks_cache_by_matrix = {}
    irrep_projection_cache = {}
    if CONFIG["train_on_irrep_parts"]:
        irrep_projection_cache = build_irrep_projection_cache(
            target_H_irreps, all_irreps
        )
        for matrix_name in CONFIG["matrix_targets"]:
            cache_for_matrix = {}
            target_irreps = target_irreps_by_name[matrix_name]
            for irrep in all_irreps:
                irrep_str = str(irrep)
                target_irrep_filtered = filter_irreps_block_data_by_irrep(
                    target_irreps, irrep, mapper
                )
                cache_for_matrix[irrep_str] = target_irrep_filtered.to_blocks(mapper)
            target_irrep_blocks_cache_by_matrix[matrix_name] = cache_for_matrix

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
            optimizer.zero_grad(set_to_none=True)

            should_log_now = should_log_epoch(epoch)
            epoch_lr = optimizer.param_groups[0]["lr"]

            # Forward pass (suppress detailed logging during training)
            if should_log_now:
                print(f"\n{'=' * 60}")
                print(f"EPOCH {epoch + 1}/{CONFIG['num_epochs']}  |  lr={epoch_lr:.6e}")
                print(f"{'=' * 60}")

            # Detailed forward logging is enabled either always (--verbose-forward)
            # or periodically (--log-forward at logging epochs).
            verbose = CONFIG["verbose_forward"] or (
                should_log_now and CONFIG["log_forward"]
            )

            log_activations = should_log_now and CONFIG.get(
                "log_activations_wandb", False
            )

            t_forward_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            if CONFIG["enable_forces"]:
                positions_for_forces = positions.detach().clone().requires_grad_(True)
                _, _, edge_sh_epoch, edge_length_emb_epoch = compute_edge_features(
                    positions_for_forces
                )
            else:
                positions_for_forces = None
                edge_sh_epoch = edge_sh
                edge_length_emb_epoch = edge_length_emb

            pred_raw_out = network(
                node_type_idx,
                edge_type_idx,
                edge_index,
                edge_shift,
                edge_length_emb_epoch,
                edge_sh_epoch,
                batch_node,
                batch_edge,
                log_to_wandb=log_activations,
                verbose=verbose,
            )
            if CONFIG["benchmark"]:
                benchmark_add("forward_total", time.perf_counter() - t_forward_start)

            pred_raw_by_matrix = pred_raw_out

            missing_targets = [
                name
                for name in CONFIG["matrix_targets"]
                if name not in pred_raw_by_matrix
            ]
            if len(missing_targets) > 0:
                raise RuntimeError(
                    f"Network output is missing targets: {missing_targets}. "
                    f"Got: {sorted(list(pred_raw_by_matrix.keys()))}"
                )

            # Wrap predictions into IrrepsBlockData then convert to matrix blocks.
            t_pack_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            pair_vectors_by_name = {}
            pred_irreps_by_name = {}
            for matrix_name in CONFIG["matrix_targets"]:
                raw_matrix = pred_raw_by_matrix[matrix_name]
                pair_vec = {}
                pair_edges_dict = {}
                lookup_dict = {}

                for key, payload in raw_matrix.items():
                    pair_vec[key] = payload["vectors"]
                    pair_edges_dict[key] = payload["edges"]

                    if should_log_now:
                        for idx, edge_5d in enumerate(payload["edges"].t()):
                            sx, sy, sz, i, j = edge_5d.tolist()
                            lookup_dict[(int(sx), int(sy), int(sz), int(i), int(j))] = (
                                key,
                                idx,
                            )

                pair_vectors_by_name[matrix_name] = pair_vec
                pred_irreps_by_name[matrix_name] = IrrepsBlockData(
                    atoms=tuple(atoms_list),
                    atom_counts=Counter(atoms_list),
                    pair_vectors=pair_vec,
                    pair_edges=pair_edges_dict,
                    lookup=lookup_dict,
                    orbital_cfg=orbital_cfg,
                    basis=target_matrices[matrix_name].basis,
                )

            if CONFIG["benchmark"]:
                benchmark_add(
                    "pred_pack_raw_outputs", time.perf_counter() - t_pack_start
                )

            # Convert to matrix blocks for all active targets.
            pred_matrix_norm_by_name = {}
            t_to_blocks_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            for matrix_name in CONFIG["matrix_targets"]:
                pred_matrix_norm_by_name[matrix_name] = pred_irreps_by_name[
                    matrix_name
                ].to_blocks(mapper)
            if CONFIG["benchmark"]:
                benchmark_add(
                    "pred_to_blocks_norm", time.perf_counter() - t_to_blocks_start
                )

            # Compute matrix block losses (optionally decomposed per irrep).
            t_loss_block_start = time.perf_counter() if CONFIG["benchmark"] else 0.0
            loss_block_irrep_filter_to_blocks_time = 0.0
            matrix_block_losses = {}
            irrep_losses_by_matrix = {}
            if CONFIG["train_on_irrep_parts"]:
                for matrix_name in CONFIG["matrix_targets"]:
                    loss_block_acc = init_block_loss_accumulator()
                    irrep_losses_curr = {}
                    pair_vec_curr = pair_vectors_by_name[matrix_name]
                    target_irrep_blocks_cache = target_irrep_blocks_cache_by_matrix[
                        matrix_name
                    ]

                    for irrep in all_irreps:
                        irrep_str = str(irrep)
                        t_irrep_ftb_start = (
                            time.perf_counter() if CONFIG["benchmark"] else 0.0
                        )
                        target_irrep_blocks = target_irrep_blocks_cache[irrep_str]
                        irrep_projectors = irrep_projection_cache[irrep_str]
                        if CONFIG["benchmark"]:
                            loss_block_irrep_filter_to_blocks_time += (
                                time.perf_counter() - t_irrep_ftb_start
                            )

                        irrep_loss_acc = init_block_loss_accumulator()
                        for key in target_irrep_blocks.pair_blocks.keys():
                            if key not in pair_vec_curr or key not in irrep_projectors:
                                continue

                            pred_blocks = project_irrep_vectors_to_blocks(
                                pair_vec_curr[key], irrep_projectors[key]
                            )
                            targ_blocks_full = target_irrep_blocks.pair_blocks[key]
                            min_n = min(pred_blocks.shape[0], targ_blocks_full.shape[0])
                            if min_n <= 0:
                                continue

                            pred_blocks_sel = pred_blocks[:min_n]
                            targ_blocks_sel = targ_blocks_full[:min_n]
                            accumulate_block_loss(
                                irrep_loss_acc,
                                pred_blocks_sel,
                                targ_blocks_sel,
                            )
                            accumulate_block_loss(
                                loss_block_acc,
                                pred_blocks_sel,
                                targ_blocks_sel,
                            )

                        irrep_losses_curr[irrep_str] = finalize_block_loss(
                            irrep_loss_acc
                        )

                    matrix_block_losses[matrix_name] = finalize_block_loss(
                        loss_block_acc
                    )
                    irrep_losses_by_matrix[matrix_name] = irrep_losses_curr
            else:
                for matrix_name in CONFIG["matrix_targets"]:
                    loss_block_acc = init_block_loss_accumulator()
                    target_matrix_for_loss = target_matrix_for_block_loss_by_name[
                        matrix_name
                    ]
                    pred_matrix_norm = pred_matrix_norm_by_name[matrix_name]
                    for key in target_matrix_for_loss.pair_blocks.keys():
                        if key not in pred_matrix_norm.pair_blocks:
                            continue
                        targ_blocks_full = target_matrix_for_loss.pair_blocks[key]
                        pred_blocks = pred_matrix_norm.pair_blocks[key]
                        min_n = min(pred_blocks.shape[0], targ_blocks_full.shape[0])
                        if min_n <= 0:
                            continue
                        accumulate_block_loss(
                            loss_block_acc,
                            pred_blocks[:min_n],
                            targ_blocks_full[:min_n],
                        )
                    matrix_block_losses[matrix_name] = finalize_block_loss(
                        loss_block_acc
                    )

            loss_block = (
                sum(matrix_block_losses.values())
                if matrix_block_losses
                else torch.tensor(0.0, device=device)
            )
            if CONFIG["benchmark"]:
                benchmark_add(
                    "loss_block_total", time.perf_counter() - t_loss_block_start
                )
                benchmark_add(
                    "loss_block_irrep_filter_and_to_blocks",
                    loss_block_irrep_filter_to_blocks_time,
                )

            # Symmetrized predictions for reporting and observable/force computations.
            pred_matrix_metrics_by_name = {
                matrix_name: (
                    pred_matrix_norm_by_name[matrix_name]
                    + pred_matrix_norm_by_name[matrix_name].transpose()
                )
                * 0.5
                for matrix_name in CONFIG["matrix_targets"]
            }

            # Observable losses from predicted matrices.
            loss_energy_weighted = torch.tensor(0.0, device=device)
            loss_num_electrons_weighted = torch.tensor(0.0, device=device)
            loss_forces_weighted = torch.tensor(0.0, device=device)
            energy_mae = None
            num_electrons_mae = None
            forces_mae = None
            forces_mse = None

            if (
                CONFIG["enable_energy"]
                and "hamiltonian" in pred_matrix_metrics_by_name
                and "density" in pred_matrix_metrics_by_name
            ):
                energy_pred = trace_matmul_sparse_block_matrix(
                    pred_matrix_metrics_by_name["hamiltonian"],
                    pred_matrix_metrics_by_name["density"],
                )
                energy_mae = torch.abs(energy_pred - energy_target)
                if CONFIG["train_on_energy"]:
                    loss_energy_weighted = CONFIG["loss_coef_observables"] * F.mse_loss(
                        energy_pred, energy_target
                    )

            if (
                CONFIG["enable_num_electrons"]
                and "overlap" in pred_matrix_metrics_by_name
                and "density" in pred_matrix_metrics_by_name
            ):
                num_electrons_pred = trace_matmul_sparse_block_matrix(
                    pred_matrix_metrics_by_name["density"],
                    pred_matrix_metrics_by_name["overlap"],
                )
                num_electrons_mae = torch.abs(num_electrons_pred - num_electrons_target)
                if CONFIG["train_on_num_electrons"]:
                    loss_num_electrons_weighted = CONFIG[
                        "loss_coef_observables"
                    ] * F.mse_loss(num_electrons_pred, num_electrons_target)

            if (
                CONFIG["enable_forces"]
                and positions_for_forces is not None
                and forces_target is not None
                and "hamiltonian" in pred_matrix_metrics_by_name
                and "density" in pred_matrix_metrics_by_name
            ):
                energy_for_forces = trace_matmul_sparse_block_matrix(
                    pred_matrix_metrics_by_name["hamiltonian"],
                    pred_matrix_metrics_by_name["density"],
                )
                grad_pos = torch.autograd.grad(
                    energy_for_forces,
                    positions_for_forces,
                    create_graph=CONFIG["train_on_forces"],
                    retain_graph=True,
                )[0]
                forces_pred = -grad_pos
                forces_err = forces_pred - forces_target
                forces_mae = torch.mean(torch.abs(forces_err))
                forces_mse = torch.mean(forces_err**2)
                if CONFIG["train_on_forces"]:
                    loss_forces_weighted = CONFIG["loss_coef_forces"] * forces_mse

            # Keep Hamiltonian block loss tracking consistent with previous history key.
            loss_H = matrix_block_losses.get(
                "hamiltonian", torch.tensor(0.0, device=device)
            )
            loss = (
                loss_block
                + loss_energy_weighted
                + loss_num_electrons_weighted
                + loss_forces_weighted
            )

            # Step scheduler (ReduceLROnPlateau needs validation loss, so we use training loss here)
            scheduler.step(float(loss.item()))

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
                "train/loss_block_total": loss_block.item(),
                "epoch": epoch,
                "lr": current_lr,
            }
            for matrix_name, matrix_loss in matrix_block_losses.items():
                step_log[f"train/loss_block_{matrix_name}"] = float(matrix_loss.item())

            if CONFIG["train_on_energy"]:
                step_log["train/loss_energy_weighted"] = float(
                    loss_energy_weighted.item()
                )
            if CONFIG["train_on_num_electrons"]:
                step_log["train/loss_num_electrons_weighted"] = float(
                    loss_num_electrons_weighted.item()
                )
            if CONFIG["train_on_forces"]:
                step_log["train/loss_forces_weighted"] = float(
                    loss_forces_weighted.item()
                )
            if energy_mae is not None:
                step_log["train/energy_mae"] = float(energy_mae.item())
            if num_electrons_mae is not None:
                step_log["train/num_electrons_mae"] = float(num_electrons_mae.item())
            if forces_mae is not None and forces_mse is not None:
                step_log["train/forces_mae"] = float(forces_mae.item())
                step_log["train/forces_mse"] = float(forces_mse.item())

            # Log per-irrep losses if enabled
            if CONFIG["train_on_irrep_parts"]:
                matrix_alias = {"hamiltonian": "H", "overlap": "S", "density": "D"}
                for matrix_name, irrep_losses in irrep_losses_by_matrix.items():
                    prefix = matrix_alias.get(matrix_name, matrix_name)
                    for irrep_str, irrep_loss in irrep_losses.items():
                        step_log[f"partial/{prefix}_{irrep_str}"] = float(
                            irrep_loss.item()
                        )

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
                pred_H_matrix_metrics = pred_matrix_metrics_by_name["hamiltonian"]

                detailed_metrics = compute_detailed_metrics(
                    pred_H_matrix_metrics, target_H_matrix, overlap_e3nn
                )
                matrix_basic_metrics = {
                    matrix_name: compute_basic_matrix_metrics(
                        pred_matrix_metrics_by_name[matrix_name],
                        target_matrices[matrix_name],
                    )
                    for matrix_name in CONFIG["matrix_targets"]
                }

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

                irrep_losses_for_print = None
                if CONFIG["train_on_irrep_parts"]:
                    matrix_alias = {"hamiltonian": "H", "overlap": "S", "density": "D"}
                    irrep_losses_for_print = {}
                    for matrix_name, irrep_losses in irrep_losses_by_matrix.items():
                        prefix = matrix_alias.get(matrix_name, matrix_name)
                        for irrep_str, irrep_loss in irrep_losses.items():
                            irrep_losses_for_print[f"{prefix}_{irrep_str}"] = irrep_loss

                log_detailed_training_metrics(
                    avg_epoch_time=avg_epoch_time,
                    epochs_since_last_log=epochs_since_last_log,
                    time_elapsed=time_elapsed,
                    loss_value=loss.item(),
                    detailed_metrics=detailed_metrics,
                    irrep_losses=irrep_losses_for_print,
                )

                if CONFIG["benchmark"]:
                    print_benchmark_report()
                    benchmark_accum = {k: 0.0 for k in benchmark_order}
                    benchmark_epochs_accum = 0

                # Log to WandB
                periodic_metrics_payload = build_wandb_detailed_metrics_log(
                    epoch_zero_based=epoch,
                    loss_value=loss.item(),
                    detailed_metrics=detailed_metrics,
                )
                # Replicate basic matrix metrics for all trained targets.
                if "hamiltonian" in matrix_basic_metrics:
                    periodic_metrics_payload["mae_H"] = matrix_basic_metrics[
                        "hamiltonian"
                    ]["mae"]
                    periodic_metrics_payload["mse_H"] = matrix_basic_metrics[
                        "hamiltonian"
                    ]["mse"]
                if "overlap" in matrix_basic_metrics:
                    periodic_metrics_payload["mae_S"] = matrix_basic_metrics["overlap"][
                        "mae"
                    ]
                    periodic_metrics_payload["mse_S"] = matrix_basic_metrics["overlap"][
                        "mse"
                    ]
                if "density" in matrix_basic_metrics:
                    periodic_metrics_payload["mae_D"] = matrix_basic_metrics["density"][
                        "mae"
                    ]
                    periodic_metrics_payload["mse_D"] = matrix_basic_metrics["density"][
                        "mse"
                    ]
                if forces_mae is not None and forces_mse is not None:
                    periodic_metrics_payload["mae_F"] = float(forces_mae.item())
                    periodic_metrics_payload["mse_F"] = float(forces_mse.item())
                wandb.log(periodic_metrics_payload)

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

                irrep_prefix_by_matrix = {
                    "hamiltonian": "H_",
                    "overlap": "S_",
                    "density": "D_",
                }
                for matrix_name in CONFIG["matrix_targets"]:
                    pred_irreps_metrics = pred_matrix_metrics_by_name[
                        matrix_name
                    ].to_vectors(mapper)
                    per_irrep_metrics = compute_irrep_metrics(
                        pred_irreps_metrics,
                        target_irreps_by_name[matrix_name],
                        all_irreps,
                        mapper,
                    )
                    metric_prefix = irrep_prefix_by_matrix.get(matrix_name, "")
                    if CONFIG["log_per_irrep_metrics"]:
                        log_per_irrep_metrics(
                            f"Per-Irrep Metrics ({matrix_name}):",
                            all_irreps,
                            per_irrep_metrics,
                            metric_prefix=metric_prefix,
                        )
                    wandb.log(
                        build_wandb_per_irrep_metrics_log(
                            epoch_zero_based=epoch,
                            all_irreps=all_irreps,
                            per_irrep_metrics=per_irrep_metrics,
                            metric_prefix=metric_prefix,
                        )
                    )

                # Save frame for video at every log interval
                for matrix_name in CONFIG["matrix_targets"]:
                    try:
                        pred_matrix_curr = pred_matrix_metrics_by_name[matrix_name]
                        target_matrix_curr = target_matrices[matrix_name]
                        correction_overlap = (
                            overlap_e3nn if matrix_name == "hamiltonian" else None
                        )
                        label = (
                            "H"
                            if matrix_name == "hamiltonian"
                            else ("S" if matrix_name == "overlap" else "D")
                        )
                        save_hamiltonian_frame_to_disk(
                            pred_matrix_curr,
                            target_matrix_curr,
                            correction_overlap,
                            list(snapshot.hamiltonian.atoms),
                            orbital_cfg,
                            matrix_frame_output_dirs[matrix_name],
                            epoch,
                            sx=0,
                            sy=0,
                            sz=0,
                            dynamic_range=False,
                            diff_dynamic_range=True,
                            percentile=99.0,
                            matrix_label=label,
                        )
                    except Exception as e:
                        print(
                            f"[WARN] Could not save {matrix_name} frame for epoch {epoch}: {e}"
                        )

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
    if CONFIG["enable_forces"]:
        positions_eval = positions.detach().clone().requires_grad_(True)
        _, _, edge_sh_eval, edge_length_emb_eval = compute_edge_features(positions_eval)
    else:
        positions_eval = None
        edge_sh_eval = edge_sh
        edge_length_emb_eval = edge_length_emb

    pred_raw_by_matrix = network(
        node_type_idx,
        edge_type_idx,
        edge_index,
        edge_shift,
        edge_length_emb_eval,
        edge_sh_eval,
        batch_node,
        batch_edge,
        verbose=CONFIG["verbose_forward"],
    )

    missing_targets = [
        name for name in CONFIG["matrix_targets"] if name not in pred_raw_by_matrix
    ]
    if len(missing_targets) > 0:
        raise RuntimeError(
            f"Network output is missing targets: {missing_targets}. "
            f"Got: {sorted(list(pred_raw_by_matrix.keys()))}"
        )

    pred_irreps_by_name = {}
    pred_matrix_norm_by_name = {}
    pred_matrix_metrics_by_name = {}
    for matrix_name in CONFIG["matrix_targets"]:
        raw_matrix = pred_raw_by_matrix[matrix_name]
        pair_vec = {}
        pair_edges_dict = {}
        lookup_dict = {}
        for key, payload in raw_matrix.items():
            pair_vec[key] = payload["vectors"]
            pair_edges_dict[key] = payload["edges"]
            for idx, edge_5d in enumerate(payload["edges"].t()):
                sx, sy, sz, i, j = edge_5d.tolist()
                lookup_dict[(int(sx), int(sy), int(sz), int(i), int(j))] = (
                    key,
                    idx,
                )

        pred_irreps_by_name[matrix_name] = IrrepsBlockData(
            atoms=tuple(atoms_list),
            atom_counts=Counter(atoms_list),
            pair_vectors=pair_vec,
            pair_edges=pair_edges_dict,
            lookup=lookup_dict,
            orbital_cfg=orbital_cfg,
            basis=target_matrices[matrix_name].basis,
        )
        pred_matrix_norm_by_name[matrix_name] = pred_irreps_by_name[
            matrix_name
        ].to_blocks(mapper)
        pred_matrix_metrics_by_name[matrix_name] = (
            pred_matrix_norm_by_name[matrix_name]
            + pred_matrix_norm_by_name[matrix_name].transpose()
        ) * 0.5

    pred_H_matrix_metrics = pred_matrix_metrics_by_name["hamiltonian"]

    # Compute detailed metrics for final evaluation (Hamiltonian + overlap gauge correction).
    final_detailed_metrics = compute_detailed_metrics(
        pred_H_matrix_metrics, target_H_matrix, overlap_e3nn
    )
    final_basic_metrics_by_name = {
        matrix_name: compute_basic_matrix_metrics(
            pred_matrix_metrics_by_name[matrix_name],
            target_matrices[matrix_name],
        )
        for matrix_name in CONFIG["matrix_targets"]
    }

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
    if "overlap" in final_basic_metrics_by_name:
        final_metrics["final/mae_S"] = final_basic_metrics_by_name["overlap"]["mae"]
        final_metrics["final/mse_S"] = final_basic_metrics_by_name["overlap"]["mse"]
    if "density" in final_basic_metrics_by_name:
        final_metrics["final/mae_D"] = final_basic_metrics_by_name["density"]["mae"]
        final_metrics["final/mse_D"] = final_basic_metrics_by_name["density"]["mse"]

    if (
        CONFIG["enable_energy"]
        and "hamiltonian" in pred_matrix_metrics_by_name
        and "density" in pred_matrix_metrics_by_name
    ):
        energy_pred = trace_matmul_sparse_block_matrix(
            pred_matrix_metrics_by_name["hamiltonian"],
            pred_matrix_metrics_by_name["density"],
        )
        energy_abs_err = torch.abs(energy_pred - energy_target)
        final_metrics["final/energy_pred"] = float(energy_pred.item())
        final_metrics["final/energy_target"] = float(energy_target.item())
        final_metrics["final/energy_mae"] = float(energy_abs_err.item())
        print(
            "  Energy: "
            f"pred={float(energy_pred.item()):.8e}, "
            f"target={float(energy_target.item()):.8e}, "
            f"|err|={float(energy_abs_err.item()):.8e}"
        )

    if (
        CONFIG["enable_num_electrons"]
        and "overlap" in pred_matrix_metrics_by_name
        and "density" in pred_matrix_metrics_by_name
    ):
        num_electrons_pred = trace_matmul_sparse_block_matrix(
            pred_matrix_metrics_by_name["density"],
            pred_matrix_metrics_by_name["overlap"],
        )
        num_electrons_abs_err = torch.abs(num_electrons_pred - num_electrons_target)
        final_metrics["final/num_electrons_pred"] = float(num_electrons_pred.item())
        final_metrics["final/num_electrons_target"] = float(num_electrons_target.item())
        final_metrics["final/num_electrons_mae"] = float(num_electrons_abs_err.item())
        print(
            "  Num electrons: "
            f"pred={float(num_electrons_pred.item()):.8e}, "
            f"target={float(num_electrons_target.item()):.8e}, "
            f"|err|={float(num_electrons_abs_err.item()):.8e}"
        )

    if (
        CONFIG["enable_forces"]
        and positions_eval is not None
        and forces_target is not None
        and "hamiltonian" in pred_matrix_metrics_by_name
        and "density" in pred_matrix_metrics_by_name
    ):
        energy_for_forces = trace_matmul_sparse_block_matrix(
            pred_matrix_metrics_by_name["hamiltonian"],
            pred_matrix_metrics_by_name["density"],
        )
        grad_pos_eval = torch.autograd.grad(
            energy_for_forces,
            positions_eval,
            create_graph=False,
            retain_graph=False,
        )[0]
        forces_pred_eval = -grad_pos_eval
        forces_err_eval = forces_pred_eval - forces_target
        forces_mae_eval = torch.mean(torch.abs(forces_err_eval))
        forces_mse_eval = torch.mean(forces_err_eval**2)
        final_metrics["final/mae_F"] = float(forces_mae_eval.item())
        final_metrics["final/mse_F"] = float(forces_mse_eval.item())
        print(
            "  Forces: "
            f"mae={float(forces_mae_eval.item()):.8e}, "
            f"mse={float(forces_mse_eval.item()):.8e}"
        )

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
    for key in target_H_matrix.pair_blocks.keys():
        if key not in pred_H_matrix_metrics.pair_blocks:
            continue
        pred_block_full = pred_H_matrix_metrics.pair_blocks[key]
        true_block_full = target_H_matrix.pair_blocks[key]
        min_n = min(pred_block_full.shape[0], true_block_full.shape[0])
        if min_n <= 0:
            continue

        pred_block = pred_block_full[:min_n]
        true_block = true_block_full[:min_n]

        block_mse = F.mse_loss(
            pred_block,
            (
                true_block.to(pred_block.dtype)
                if true_block.dtype != pred_block.dtype
                else true_block
            ),
        ).item()
        block_mae = torch.mean(torch.abs(pred_block - true_block)).item()

        pair_irreps = mapper.get_pair_irreps(key)

        num_selected = min_n
        num_total = min_n

        print(f"\n  {key}:")
        print(f"    Block shape: {pred_block.shape}, Irreps: {pair_irreps}")
        print(f"    MSE: {block_mse:.6e}")
        print(f"    MAE: {block_mae:.6e}")
        print(f"    Edges used: {num_selected}/{num_total}")
        print(
            f"    Relative error: {block_mae / (torch.abs(true_block).mean().item() + 1e-10):.6%}"
        )

        final_metrics[f"final/{key}_mse"] = block_mse
        final_metrics[f"final/{key}_mae"] = block_mae

    # Log final metrics to WandB
    wandb.log(final_metrics)

    # Distance-binned error curves (16 bins) for all targets
    print("\n  Distance-binned error curves (16 bins):")
    matrix_alias = {"hamiltonian": "H", "overlap": "S", "density": "D"}
    for matrix_name in CONFIG["matrix_targets"]:
        print(f"    [{matrix_name}]")
        distance_curve = compute_distance_error_curve(
            H_pred=pred_matrix_metrics_by_name[matrix_name],
            H_gt=target_matrices[matrix_name],
            positions=positions,
            box=box,
            n_bins=16,
        )
        if distance_curve is None:
            print("      No matched edges found for distance-curve computation.")
            continue

        curve_json_path = (
            run_checkpoint_dir / f"distance_error_curve_{matrix_name}.json"
        )
        with open(curve_json_path, "w") as f:
            json.dump(distance_curve, f, indent=2)

        curve_plot_path = run_checkpoint_dir / f"distance_error_curve_{matrix_name}.png"
        save_distance_error_curve_plot(
            distance_curve,
            curve_plot_path,
            title=f"Distance Error Curves ({matrix_name}, Final, 16 bins)",
        )

        # Print a concise summary
        l1_abs = [x for x in distance_curve["l1_abs"] if x == x]
        l2_abs = [x for x in distance_curve["l2_abs"] if x == x]
        l1_rel = [x for x in distance_curve["l1_rel"] if x == x]
        l2_rel = [x for x in distance_curve["l2_rel"] if x == x]
        if l1_abs and l2_abs and l1_rel and l2_rel:
            print(
                f"      L1 abs range: {min(l1_abs):.3e} .. {max(l1_abs):.3e}, "
                f"L2 abs range: {min(l2_abs):.3e} .. {max(l2_abs):.3e}"
            )
            print(
                f"      L1 rel range: {min(l1_rel):.3e} .. {max(l1_rel):.3e}, "
                f"L2 rel range: {min(l2_rel):.3e} .. {max(l2_rel):.3e}"
            )

        alias = matrix_alias.get(matrix_name, matrix_name)
        wandb.log(
            {
                f"distance_curve/{alias}_plot": wandb.Image(str(curve_plot_path)),
            }
        )
        print(f"      Saved: {curve_json_path}")
        print(f"      Saved: {curve_plot_path}")

    # Per-irrep visualizations (k-range=0), separated by matrix target.
    if CONFIG["log_per_irrep_images"]:
        irrep_output_dir = run_checkpoint_dir / "per_irrep_images"
        irrep_output_dir.mkdir(parents=True, exist_ok=True)
        print("\n  Per-irrep visualizations (k-range=0, percentile=99):")
        matrix_alias = {"hamiltonian": "H", "overlap": "S", "density": "D"}
        for matrix_name in CONFIG["matrix_targets"]:
            alias = matrix_alias.get(matrix_name, matrix_name)
            pred_matrix_full = pred_matrix_metrics_by_name[matrix_name]
            target_matrix_full = target_matrices[matrix_name]
            correction_overlap = overlap_e3nn if matrix_name == "hamiltonian" else None
            print(f"    [{matrix_name}]")
            for irrep in all_irreps:
                irrep_str = str(irrep)
                try:
                    pred_irrep = split_hamiltonian_by_irrep(
                        pred_matrix_full, mapper, irrep_str
                    )
                    target_irrep = split_hamiltonian_by_irrep(
                        target_matrix_full, mapper, irrep_str
                    )

                    filename_prefix = f"{matrix_name}_{irrep_str}"
                    visualize_hamiltonians(
                        pred_irrep,
                        target_irrep,
                        correction_overlap,
                        list(snapshot.hamiltonian.atoms),
                        orbital_cfg,
                        k_range=0,
                        output_dir=irrep_output_dir,
                        dynamic_range=True,
                        diff_dynamic_range=True,
                        per_panel_dynamic_range=True,
                        filename_prefix=filename_prefix,
                        percentile=99.0,
                    )

                    image_path = (
                        irrep_output_dir / f"{filename_prefix}_sx+0_sy+0_sz+0.png"
                    )
                    if image_path.exists():
                        wandb.log(
                            {
                                f"irrep_images/{alias}/{irrep_str}": wandb.Image(
                                    str(image_path)
                                )
                            }
                        )
                    else:
                        print(
                            f"      [WARN] Missing image for irrep {irrep_str}: {image_path.name}"
                        )
                except Exception as e:
                    print(f"      [WARN] Irrep {irrep_str} visualization failed: {e}")

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
            "final_metrics": final_metrics,
            "final_metrics_hamiltonian_detailed": final_detailed_metrics,
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
        matrix_alias = {"hamiltonian": "H", "overlap": "S", "density": "D"}
        for matrix_name in CONFIG["matrix_targets"]:
            try:
                frame_dir = matrix_frame_output_dirs[matrix_name]
                if not any(frame_dir.glob("frame_epoch_*.png")):
                    print(
                        f"[WARN] No frames found for {matrix_name} video generation in {frame_dir}"
                    )
                    continue

                video_path = run_checkpoint_dir / f"training_progress_{matrix_name}.mp4"
                compile_frames_to_video(
                    frame_dir,
                    video_path,
                    fps=5,
                    pattern="frame_epoch_*.png",
                    format="mp4",
                )
                alias = matrix_alias.get(matrix_name, matrix_name)
                wandb.log(
                    {
                        f"training_video_{alias}": wandb.Video(
                            str(video_path), fps=5, format="mp4"
                        )
                    }
                )
                print(f"[OK] {matrix_name} training video logged to WandB")
            except Exception as e:
                print(f"[WARN] Could not generate final video for {matrix_name}: {e}")
    else:
        print("")
        print("=" * 80)
        print("VIDEO GENERATION SKIPPED (--generate-video not specified)")
        print("=" * 80)

    # Finish WandB run
    wandb.finish()
