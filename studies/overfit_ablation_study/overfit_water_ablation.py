"""
Standalone overfit ablation study for single-water Hamiltonian fitting.

This script is intentionally self-contained under studies/overfit_ablation_study and
does not import code from studies/minimal_overfit_study.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import Irreps, spherical_harmonics
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau

from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix, IrrepsBlockData
from data.pyscf_baseline_parser import load_pyscf_snapshot
from data.snapshot import Snapshot
from net.common import build_hidden_irreps

from common import (
    MinimalNetwork,
    canonicalize_edge_order,
    compile_frames_to_video,
    compute_detailed_metrics,
    compute_distance_error_curve,
    compute_irrep_metrics,
    get_all_irreps_in_hamiltonian,
    save_distance_error_curve_plot,
    save_dos_comparison_plot,
    save_hamiltonian_frame_to_disk,
    split_hamiltonian_by_irrep,
    visualize_hamiltonians,
    filter_irreps_block_data_by_irrep,
)
from strict_checks import strict_edge_alignment_check, strict_reverse_edge_check

HARTREE_TO_EV = 27.2113845
DEFAULT_CUTOFF_RADIUS = 7.5
UNIT_SCALE_FROM_HARTREE = {
    "hartree": 1.0,
    "ev": HARTREE_TO_EV,
    "mev": HARTREE_TO_EV * 1000.0,
}
UNIT_DISPLAY_NAME = {
    "hartree": "Hartree",
    "ev": "eV",
    "mev": "meV",
}
DEFAULT_BASELINE_NPZ = "data/pyscf_baseline/results/h2o_original_rhf_openmx_like.npz"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def should_log_epoch(epoch_zero_based: int, log_interval: int, adaptive: bool) -> bool:
    if not adaptive:
        return epoch_zero_based % log_interval == 0

    epoch_one_based = epoch_zero_based + 1
    if epoch_one_based <= 10:
        return True
    if epoch_one_based <= 100:
        return epoch_one_based % 10 == 0
    return epoch_zero_based % log_interval == 0


def canonicalize_block_matrix_edges(
    block_matrix: BlockMatrix,
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> BlockMatrix:
    order_dict: dict[str, torch.Tensor] = {}
    for key, edges_5d in block_matrix.pair_edges.items():
        edge_shift_local = edges_5d[:3]
        edge_index_local = edges_5d[3:]
        _, _, perm_local = canonicalize_edge_order(
            edge_index=edge_index_local,
            edge_shift=edge_shift_local,
            positions=positions,
            box=box,
        )
        order_dict[key] = perm_local
    return block_matrix.reorder_edges(order_dict)


def align_block_matrix_to_reference(
    source: BlockMatrix,
    reference: BlockMatrix,
) -> BlockMatrix:
    """
    Align source matrix to reference edge sets.

    Missing source edges are filled with zeros, so this is robust when baseline and
    target edge sets differ slightly.
    """
    src_lookup: dict[str, dict[tuple[int, int, int, int, int], int]] = {}
    for key, edges in source.pair_edges.items():
        src_lookup[key] = {
            tuple(map(int, edges[:, i].tolist())): i for i in range(edges.shape[1])
        }

    out_blocks: dict[str, torch.Tensor] = {}
    for key, ref_edges in reference.pair_edges.items():
        ref_blocks = reference.pair_blocks[key]
        out = torch.zeros_like(ref_blocks)
        key_lookup = src_lookup.get(key, {})
        src_blocks = source.pair_blocks.get(key)
        if src_blocks is not None:
            for i in range(ref_edges.shape[1]):
                edge_key = tuple(map(int, ref_edges[:, i].tolist()))
                src_idx = key_lookup.get(edge_key)
                if src_idx is not None and src_idx < src_blocks.shape[0]:
                    out[i] = src_blocks[src_idx]
        out_blocks[key] = out

    return BlockMatrix(
        atoms=reference.atoms,
        atom_counts=reference.atom_counts,
        pair_blocks=out_blocks,
        pair_edges=reference.pair_edges,
        lookup=reference.lookup,
        orbital_cfg=reference.orbital_cfg,
        basis=reference.basis,
    )


def cast_block_matrix_dtype(matrix: BlockMatrix, dtype: torch.dtype) -> BlockMatrix:
    return BlockMatrix(
        atoms=matrix.atoms,
        atom_counts=matrix.atom_counts,
        pair_blocks={k: v.to(dtype=dtype) for k, v in matrix.pair_blocks.items()},
        pair_edges=matrix.pair_edges,
        lookup=matrix.lookup,
        orbital_cfg=matrix.orbital_cfg,
        basis=matrix.basis,
    )


def get_diagonal_mask(edges_5d: torch.Tensor) -> torch.Tensor:
    sx, sy, sz, i, j = edges_5d[0], edges_5d[1], edges_5d[2], edges_5d[3], edges_5d[4]
    return (sx == 0) & (sy == 0) & (sz == 0) & (i == j)


def init_loss_accumulator(device: torch.device) -> dict[str, torch.Tensor | int]:
    return {
        "loss_sum": torch.tensor(0.0, device=device),
        "sq_sum": torch.tensor(0.0, device=device),
        "count": 0,
    }


def accumulate_block_loss(
    acc: dict[str, torch.Tensor | int],
    pred_blocks: torch.Tensor,
    targ_blocks: torch.Tensor,
    aggregation: str,
) -> None:
    if pred_blocks.dtype != targ_blocks.dtype:
        targ_blocks = targ_blocks.to(pred_blocks.dtype)
    if aggregation == "global":
        diff = pred_blocks - targ_blocks
        acc["sq_sum"] = acc["sq_sum"] + (diff**2).sum()  # type: ignore[operator]
        acc["count"] = int(acc["count"]) + int(diff.numel())
    else:
        acc["loss_sum"] = acc["loss_sum"] + F.mse_loss(pred_blocks, targ_blocks)  # type: ignore[operator]


def finalize_block_loss(
    acc: dict[str, torch.Tensor | int],
    aggregation: str,
    device: torch.device,
) -> torch.Tensor:
    if aggregation == "global":
        count = int(acc["count"])
        if count <= 0:
            return torch.tensor(0.0, device=device)
        return acc["sq_sum"] / count  # type: ignore[operator]
    return acc["loss_sum"]  # type: ignore[return-value]


def compute_block_metrics(
    pred: BlockMatrix,
    target: BlockMatrix,
) -> dict[str, float]:
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    max_abs = 0.0

    diag_abs = 0.0
    diag_sq = 0.0
    diag_count = 0
    offdiag_abs = 0.0
    offdiag_sq = 0.0
    offdiag_count = 0

    per_block_rmse: dict[str, float] = {}

    for key in target.pair_blocks.keys():
        if key not in pred.pair_blocks:
            continue

        pred_b = pred.pair_blocks[key]
        targ_b = target.pair_blocks[key]
        edges = target.pair_edges[key]

        min_n = min(pred_b.shape[0], targ_b.shape[0], edges.shape[1])
        if min_n <= 0:
            continue

        pred_sel = pred_b[:min_n]
        targ_sel = targ_b[:min_n]
        if targ_sel.dtype != pred_sel.dtype:
            targ_sel = targ_sel.to(pred_sel.dtype)

        diff = pred_sel - targ_sel
        abs_diff = torch.abs(diff)

        total_abs += float(abs_diff.sum().item())
        total_sq += float((diff**2).sum().item())
        total_count += int(diff.numel())
        max_abs = max(max_abs, float(abs_diff.max().item()))

        per_block_rmse[key] = float(torch.sqrt(torch.mean(diff**2)).item())

        diag_mask = get_diagonal_mask(edges[:, :min_n])
        offdiag_mask = ~diag_mask

        if diag_mask.any():
            d = diff[diag_mask]
            diag_abs += float(torch.abs(d).sum().item())
            diag_sq += float((d**2).sum().item())
            diag_count += int(d.numel())

        if offdiag_mask.any():
            d = diff[offdiag_mask]
            offdiag_abs += float(torch.abs(d).sum().item())
            offdiag_sq += float((d**2).sum().item())
            offdiag_count += int(d.numel())

    def safe_div(x: float, n: int) -> float:
        return x / max(n, 1)

    def safe_rmse(sq: float, n: int) -> float:
        return (sq / max(n, 1)) ** 0.5

    out: dict[str, float] = {
        "max_abs_element_error": max_abs,
        "diag_mae": safe_div(diag_abs, diag_count),
        "diag_mse": safe_div(diag_sq, diag_count),
        "diag_rmse": safe_rmse(diag_sq, diag_count),
        "offdiag_mae": safe_div(offdiag_abs, offdiag_count),
        "offdiag_mse": safe_div(offdiag_sq, offdiag_count),
        "offdiag_rmse": safe_rmse(offdiag_sq, offdiag_count),
        "global_mae": safe_div(total_abs, total_count),
        "global_mse": safe_div(total_sq, total_count),
        "global_rmse": safe_rmse(total_sq, total_count),
    }

    for key, rmse in per_block_rmse.items():
        out[f"per_block_rmse/{key}"] = rmse

    return out


def compute_symmetry_violation_metrics(matrix: BlockMatrix) -> dict[str, float]:
    antisym = matrix - matrix.transpose()
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    max_abs = 0.0

    for key, block in antisym.pair_blocks.items():
        abs_b = torch.abs(block)
        total_abs += float(abs_b.sum().item())
        total_sq += float((block**2).sum().item())
        total_count += int(block.numel())
        max_abs = max(max_abs, float(abs_b.max().item()))

    return {
        "mae": total_abs / max(total_count, 1),
        "mse": total_sq / max(total_count, 1),
        "rmse": (total_sq / max(total_count, 1)) ** 0.5,
        "max_abs": max_abs,
    }


def evaluate_prediction_bundle(
    pred_pre_sym: BlockMatrix,
    pred_post_sym: BlockMatrix,
    target_H_full: BlockMatrix,
    overlap: BlockMatrix,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    detailed = compute_detailed_metrics(pred_post_sym, target_H_full, overlap)
    block_metrics = compute_block_metrics(pred_post_sym, target_H_full)
    sym_pre = compute_symmetry_violation_metrics(pred_pre_sym)
    sym_post = compute_symmetry_violation_metrics(pred_post_sym)

    extra = {
        **block_metrics,
        "symmetry_violation_pre_mae": sym_pre["mae"],
        "symmetry_violation_pre_mse": sym_pre["mse"],
        "symmetry_violation_pre_rmse": sym_pre["rmse"],
        "symmetry_violation_pre_max_abs": sym_pre["max_abs"],
        "symmetry_violation_post_mae": sym_post["mae"],
        "symmetry_violation_post_mse": sym_post["mse"],
        "symmetry_violation_post_rmse": sym_post["rmse"],
        "symmetry_violation_post_max_abs": sym_post["max_abs"],
    }
    return detailed, extra, block_metrics


def build_graph_inputs(
    target_matrix: BlockMatrix,
    atoms_list: list[str],
    positions: torch.Tensor,
    box: torch.Tensor | None,
    mapper: BlockIrrepMapper,
    cutoff_radius: float,
    n_radial: int,
    l_max: int,
    sh_mode: str,
    norm_kind: str,
    device: torch.device,
    log_data: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    # Build graph edges directly from the target edge set so strict exact-edge
    # matching is guaranteed by construction.
    edge_index_chunks: list[torch.Tensor] = []
    edge_shift_chunks: list[torch.Tensor] = []
    edge_type_idx_chunks: list[torch.Tensor] = []

    for key in sorted(target_matrix.pair_edges.keys()):
        edges_5d = target_matrix.pair_edges[key].to(device)
        edge_shift_chunks.append(edges_5d[:3])
        edge_index_chunks.append(edges_5d[3:])
        edge_type_idx_chunks.append(
            torch.full(
                (edges_5d.shape[1],),
                mapper.edge_type2idx[key],
                dtype=torch.long,
                device=device,
            )
        )

    edge_shift = torch.cat(edge_shift_chunks, dim=1)
    edge_index = torch.cat(edge_index_chunks, dim=1)
    edge_type_idx = torch.cat(edge_type_idx_chunks, dim=0)

    if box is not None:
        shift_float = edge_shift.T.to(dtype=positions.dtype)
        edge_vec = (
            positions[edge_index[1]] - positions[edge_index[0]] + shift_float @ box
        )
    else:
        edge_vec = positions[edge_index[1]] - positions[edge_index[0]]

    edge_dist = torch.linalg.norm(edge_vec, dim=1)

    sh_irreps = Irreps.spherical_harmonics(l_max)
    if sh_mode == "legacy":
        edge_vec_norm = edge_vec.clone()
        non_zero_mask = edge_dist > 1e-6
        edge_vec_norm[non_zero_mask] = edge_vec[non_zero_mask] / edge_dist[
            non_zero_mask
        ].unsqueeze(-1)
        if norm_kind == "none":
            edge_sh = spherical_harmonics(sh_irreps, edge_vec_norm, normalize=False)
        else:
            edge_sh = spherical_harmonics(
                sh_irreps,
                edge_vec_norm,
                normalize=False,
                normalization=norm_kind,
            )
    else:
        if norm_kind == "none":
            edge_sh = spherical_harmonics(sh_irreps, edge_vec, normalize=False)
        else:
            edge_sh = spherical_harmonics(
                sh_irreps,
                edge_vec,
                normalize=True,
                normalization=norm_kind,
            )

    edge_length_emb = soft_one_hot_linspace(
        edge_dist,
        start=0.0,
        end=cutoff_radius,
        number=n_radial,
        basis="gaussian",
        cutoff=False,
    )

    element_to_idx = {
        elem: idx for idx, elem in enumerate(mapper.orbital_cfg.elements())
    }
    node_type_idx = torch.tensor([element_to_idx[a] for a in atoms_list], device=device)

    batch_node = torch.zeros(len(atoms_list), dtype=torch.long, device=device)
    batch_edge = torch.zeros(edge_index.shape[1], dtype=torch.long, device=device)

    num_self_edges = int(
        (
            (edge_index[0] == edge_index[1])
            & (edge_shift[0] == 0)
            & (edge_shift[1] == 0)
            & (edge_shift[2] == 0)
        )
        .sum()
        .item()
    )

    if log_data:
        print(f"[GRAPH] edges={edge_index.shape[1]}, self_edges={num_self_edges}")
        print(f"[GRAPH] SH mode={sh_mode}, norm_kind={norm_kind}")

    return (
        node_type_idx,
        edge_type_idx,
        edge_index,
        edge_shift,
        edge_length_emb,
        edge_sh,
        batch_node,
        batch_edge,
        num_self_edges,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone overfit ablation study for H2O",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/small/H2O/original/H2O.matrix",
    )
    parser.add_argument(
        "--info-path",
        type=str,
        default="data/small/H2O/original/H2O.info.out",
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--l-max", type=int, default=4)
    parser.add_argument("--hidden-irreps", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--cutoff-radius", type=float, default=DEFAULT_CUTOFF_RADIUS)
    parser.add_argument("--n-radial", type=int, default=64)
    parser.add_argument(
        "--training-unit",
        type=str.lower,
        choices=["hartree", "ev", "mev"],
        default="ev",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--adaptive-log-interval", action="store_true", default=False)
    parser.add_argument("--log-data", action="store_true", default=False)
    parser.add_argument("--log-model", action="store_true", default=False)
    parser.add_argument("--log-per-irrep-metrics", action="store_true", default=False)
    parser.add_argument("--log-per-irrep-images", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="studies/overfit_ablation_study/checkpoints",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=600)
    parser.add_argument(
        "--loss-aggregation",
        type=str,
        choices=["per_key", "global"],
        default="global",
    )
    parser.add_argument(
        "--sh-mode",
        type=str,
        choices=["aligned", "legacy"],
        default="aligned",
    )
    parser.add_argument("--train-on-irrep-parts", action="store_true", default=False)
    parser.add_argument("--generate-video", action="store_true", default=False)

    parser.add_argument("--e3layernorm", type=parse_bool, default=False)
    parser.add_argument(
        "--norm-kind",
        type=str,
        choices=["component", "norm", "none"],
        default="component",
    )
    parser.add_argument("--skip-connections", type=parse_bool, default=False)

    parser.add_argument("--delta-learning", type=parse_bool, default=False)
    parser.add_argument("--baseline-npz-path", type=str, default=DEFAULT_BASELINE_NPZ)
    parser.add_argument("--baseline-json-path", type=str, default=None)

    parser.add_argument("--lbfgs-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    torch_dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(torch_dtype)
    device = torch.device(args.device)

    config = {
        "data_path": args.data_path,
        "info_path": args.info_path,
        "training_unit": args.training_unit,
        "hamiltonian_scale_from_hartree": UNIT_SCALE_FROM_HARTREE[args.training_unit],
        "hidden_dim": args.hidden_dim,
        "l_max": args.l_max,
        "hidden_irreps": args.hidden_irreps,
        "num_layers": args.num_layers,
        "cutoff_radius": args.cutoff_radius,
        "n_radial": args.n_radial,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "log_interval": args.log_interval,
        "adaptive_log_interval": args.adaptive_log_interval,
        "log_data": args.log_data,
        "log_model": args.log_model,
        "log_per_irrep_metrics": args.log_per_irrep_metrics,
        "log_per_irrep_images": args.log_per_irrep_images,
        "benchmark": args.benchmark,
        "device": args.device,
        "dtype": args.dtype,
        "checkpoint_dir": args.checkpoint_dir,
        "run_name": args.run_name,
        "grad_clip": args.grad_clip,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "loss_aggregation": args.loss_aggregation,
        "sh_mode": args.sh_mode,
        "train_on_irrep_parts": args.train_on_irrep_parts,
        "generate_video": args.generate_video,
        "e3layernorm": args.e3layernorm,
        "norm_kind": args.norm_kind,
        "skip_connections": args.skip_connections,
        "delta_learning": args.delta_learning,
        "baseline_npz_path": args.baseline_npz_path,
        "baseline_json_path": args.baseline_json_path,
        "lbfgs_steps": args.lbfgs_steps,
        "seed": args.seed,
        "apply_cutoff_to_targets": True,
        "require_exact_edge_match": True,
        "weight_decay": 0.0,
    }

    wandb_project = os.environ.get("WANDB_PROJECT", "mandala-overfit-ablation-study")
    wandb_kwargs: dict[str, object] = {
        "project": wandb_project,
        "config": config,
    }
    if args.run_name is not None:
        wandb_kwargs["name"] = args.run_name
    try:
        wandb.init(**wandb_kwargs)
    except Exception as exc:
        print(
            "[WARN] wandb.init failed; falling back to disabled mode. " f"Reason: {exc}"
        )
        wandb.init(**wandb_kwargs, mode="disabled")

    run_name = wandb.run.name
    run_checkpoint_dir = Path(args.checkpoint_dir) / run_name
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frame_output_dir = run_checkpoint_dir / "frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OVERFIT ABLATION STUDY - STANDALONE")
    print("=" * 80)
    print(f"run_name={run_name}")
    print(f"device={device}, dtype={torch_dtype}")
    print(f"cutoff_radius={args.cutoff_radius} A")
    print(
        "training unit: "
        f"{UNIT_DISPLAY_NAME[args.training_unit]} "
        f"(x{config['hamiltonian_scale_from_hartree']:.7g} from Hartree)"
    )
    print(
        f"hardcoded: apply_cutoff_to_targets={config['apply_cutoff_to_targets']}, "
        f"require_exact_edge_match={config['require_exact_edge_match']}"
    )

    snapshot = Snapshot.from_openmx(
        matrix_path=args.data_path,
        info_path=args.info_path,
        convention="e3nn",
        symmetrize_density=True,
        cutoff_radius=None,
        dtype=torch_dtype,
    )

    before_edges = sum(v.shape[1] for v in snapshot.hamiltonian.pair_edges.values())
    snapshot = snapshot.filter_by_distance(args.cutoff_radius)
    after_edges = sum(v.shape[1] for v in snapshot.hamiltonian.pair_edges.values())
    if args.log_data:
        print(
            f"[DATA] cutoff filtering applied to targets: {before_edges} -> {after_edges} edges"
        )

    hamiltonian_e3nn = (
        snapshot.hamiltonian.to(device) * config["hamiltonian_scale_from_hartree"]
    )
    overlap_e3nn = snapshot.overlap.to(device)
    density_e3nn = snapshot.density.to(device)

    positions = snapshot.positions.to(device=device, dtype=torch_dtype)
    box = (
        snapshot.box.to(device=device, dtype=torch_dtype)
        if snapshot.box is not None
        else None
    )
    atoms_list = list(snapshot.hamiltonian.atoms)

    mapper = BlockIrrepMapper(
        snapshot.hamiltonian.orbital_cfg, device=device, dtype=torch_dtype
    )

    hamiltonian_e3nn = canonicalize_block_matrix_edges(hamiltonian_e3nn, positions, box)
    overlap_e3nn = canonicalize_block_matrix_edges(overlap_e3nn, positions, box)
    density_e3nn = canonicalize_block_matrix_edges(density_e3nn, positions, box)
    hamiltonian_e3nn = (hamiltonian_e3nn + hamiltonian_e3nn.transpose()) * 0.5

    baseline_hamiltonian: BlockMatrix | None = None
    if args.delta_learning:
        baseline_snapshot = load_pyscf_snapshot(
            npz_path=args.baseline_npz_path,
            json_path=args.baseline_json_path,
            # Keep parser/converter in float32 for basis-conversion robustness,
            # then cast to requested training dtype below.
            dtype=torch.float32,
            device=device,
            basis="openmx",
        ).to_e3nn()
        baseline_hamiltonian_raw = (
            cast_block_matrix_dtype(
                baseline_snapshot.hamiltonian.to(device), torch_dtype
            )
            * config["hamiltonian_scale_from_hartree"]
        )
        baseline_hamiltonian_raw = canonicalize_block_matrix_edges(
            baseline_hamiltonian_raw, positions, box
        )
        baseline_hamiltonian_raw = (
            baseline_hamiltonian_raw + baseline_hamiltonian_raw.transpose()
        ) * 0.5
        baseline_hamiltonian = align_block_matrix_to_reference(
            baseline_hamiltonian_raw,
            hamiltonian_e3nn,
        )
        if args.log_data:
            print(f"[DELTA] baseline loaded from {args.baseline_npz_path}")

    target_H_full = hamiltonian_e3nn
    target_H_train = (
        target_H_full - baseline_hamiltonian
        if baseline_hamiltonian is not None
        else target_H_full
    )

    (
        node_type_idx,
        edge_type_idx,
        edge_index,
        edge_shift,
        edge_length_emb,
        edge_sh,
        batch_node,
        batch_edge,
        _,
    ) = build_graph_inputs(
        target_matrix=target_H_full,
        atoms_list=atoms_list,
        positions=positions,
        box=box,
        mapper=mapper,
        cutoff_radius=args.cutoff_radius,
        n_radial=args.n_radial,
        l_max=args.l_max,
        sh_mode=args.sh_mode,
        norm_kind=args.norm_kind,
        device=device,
        log_data=args.log_data,
    )

    strict_reverse_edge_check(
        edge_index=edge_index, edge_shift=edge_shift, edge_set_name="graph"
    )
    strict_edge_alignment_check(
        target_matrix=target_H_full,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="hamiltonian",
        require_exact=True,
    )
    strict_edge_alignment_check(
        target_matrix=overlap_e3nn,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="overlap",
        require_exact=True,
    )
    strict_edge_alignment_check(
        target_matrix=density_e3nn,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="density",
        require_exact=True,
    )

    if args.hidden_irreps is not None:
        hidden_irreps = Irreps(args.hidden_irreps)
    else:
        hidden_irreps = build_hidden_irreps(
            l_max=args.l_max,
            base_dim=args.hidden_dim,
            use_odd_features=True,
        )

    sh_irreps = Irreps.spherical_harmonics(args.l_max)
    num_elements = len(snapshot.hamiltonian.orbital_cfg.elements())
    num_edge_types = num_elements**2

    network = MinimalNetwork(
        num_elements=num_elements,
        n_radial=args.n_radial,
        num_edge_types=num_edge_types,
        hidden_irreps=hidden_irreps,
        sh_irreps=sh_irreps,
        num_layers=args.num_layers,
        mapper=mapper,
        magnitude_factorization=False,
        head_mlp_for_scalars=False,
        head_use_tensor_square=False,
        separate_shifted_self=False,
        use_e3layernorm=args.e3layernorm,
        norm_kind=args.norm_kind,
        skip_connections=args.skip_connections,
    ).to(device=device, dtype=torch_dtype)

    target_H_train_irreps = target_H_train.to_vectors(mapper)
    target_H_full_irreps = target_H_full.to_vectors(mapper)
    all_irreps = get_all_irreps_in_hamiltonian(mapper)

    target_irrep_cache: dict[str, BlockMatrix] = {}
    if args.train_on_irrep_parts:
        for irrep in all_irreps:
            irrep_str = str(irrep)
            target_ir = filter_irreps_block_data_by_irrep(
                target_H_train_irreps, irrep, mapper
            )
            target_irrep_cache[irrep_str] = target_ir.to_blocks(mapper)

    optimizer = AdamW(network.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=10,
        min_lr=1e-6,
        verbose=True,
    )

    history = {"loss": [], "mae_H": [], "mse_H": []}
    best_loss = float("inf")
    best_epoch = -1

    def forward_to_irreps(verbose: bool = False) -> IrrepsBlockData:
        pred_raw = network(
            node_type_idx,
            edge_type_idx,
            edge_index,
            edge_shift,
            edge_length_emb,
            edge_sh,
            batch_node,
            batch_edge,
            log_to_wandb=False,
            verbose=verbose,
        )

        pair_vec_H = {}
        pair_edges_dict = {}
        lookup_dict = {}
        for key, payload in pred_raw.items():
            pair_vec_H[key] = payload["vectors"]
            pair_edges_dict[key] = payload["edges"]
            for idx, edge_5d in enumerate(payload["edges"].t()):
                sx, sy, sz, i, j = edge_5d.tolist()
                lookup_dict[(int(sx), int(sy), int(sz), int(i), int(j))] = (key, idx)

        return IrrepsBlockData(
            atoms=tuple(atoms_list),
            atom_counts=Counter(atoms_list),
            pair_vectors=pair_vec_H,
            pair_edges=pair_edges_dict,
            lookup=lookup_dict,
            orbital_cfg=snapshot.hamiltonian.orbital_cfg,
            basis=target_H_full.basis,
        )

    def compute_training_loss() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_irreps = forward_to_irreps(verbose=False)

        irrep_losses: dict[str, torch.Tensor] = {}
        if args.train_on_irrep_parts:
            total_acc = init_loss_accumulator(device)
            for irrep in all_irreps:
                irrep_str = str(irrep)
                pred_ir = filter_irreps_block_data_by_irrep(pred_irreps, irrep, mapper)
                pred_blocks = pred_ir.to_blocks(mapper)
                targ_blocks = target_irrep_cache[irrep_str]
                irrep_acc = init_loss_accumulator(device)

                for key in targ_blocks.pair_blocks.keys():
                    if key not in pred_blocks.pair_blocks:
                        continue
                    pb = pred_blocks.pair_blocks[key]
                    tb = targ_blocks.pair_blocks[key]
                    min_n = min(pb.shape[0], tb.shape[0])
                    if min_n <= 0:
                        continue
                    pb = pb[:min_n]
                    tb = tb[:min_n]
                    accumulate_block_loss(irrep_acc, pb, tb, args.loss_aggregation)
                    accumulate_block_loss(total_acc, pb, tb, args.loss_aggregation)

                irrep_losses[irrep_str] = finalize_block_loss(
                    irrep_acc, args.loss_aggregation, device
                )

            loss = finalize_block_loss(total_acc, args.loss_aggregation, device)
        else:
            pred_blocks = pred_irreps.to_blocks(mapper)
            total_acc = init_loss_accumulator(device)
            for key in target_H_train.pair_blocks.keys():
                if key not in pred_blocks.pair_blocks:
                    continue
                pb = pred_blocks.pair_blocks[key]
                tb = target_H_train.pair_blocks[key]
                min_n = min(pb.shape[0], tb.shape[0])
                if min_n <= 0:
                    continue
                pb = pb[:min_n]
                tb = tb[:min_n]
                accumulate_block_loss(total_acc, pb, tb, args.loss_aggregation)
            loss = finalize_block_loss(total_acc, args.loss_aggregation, device)

        return loss, irrep_losses

    def predict_full_matrices() -> tuple[BlockMatrix, BlockMatrix]:
        pred_irreps = forward_to_irreps(verbose=False)
        pred_train = pred_irreps.to_blocks(mapper)
        pred_pre_sym = (
            pred_train + baseline_hamiltonian
            if baseline_hamiltonian is not None
            else pred_train
        )
        pred_post_sym = (pred_pre_sym + pred_pre_sym.transpose()) * 0.5
        return pred_pre_sym, pred_post_sym

    print("=" * 80)
    print("ADAMW STAGE")
    print("=" * 80)

    last_log_time = time.time()
    for epoch in range(args.num_epochs):
        t_epoch = time.perf_counter() if args.benchmark else 0.0

        network.train()
        optimizer.zero_grad(set_to_none=True)
        loss, irrep_losses = compute_training_loss()

        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(
                f"Invalid loss at epoch {epoch + 1}: {float(loss.item())}"
            )

        loss.backward()
        if args.grad_clip > 0:
            grad_norm = clip_grad_norm_(network.parameters(), args.grad_clip)
        else:
            grad_norm = None
        optimizer.step()
        scheduler.step(float(loss.item()))

        history["loss"].append(float(loss.item()))
        wandb.log(
            {
                "epoch": epoch,
                "train/loss_step": float(loss.item()),
                "lr": optimizer.param_groups[0]["lr"],
                **(
                    {"train/grad_norm": float(grad_norm.item())}
                    if grad_norm is not None
                    else {}
                ),
            }
        )

        if args.train_on_irrep_parts and len(irrep_losses) > 0:
            wandb.log(
                {f"partial/{k}": float(v.item()) for k, v in irrep_losses.items()}
            )

        if float(loss.item()) < best_loss:
            best_loss = float(loss.item())
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": config,
                },
                run_checkpoint_dir / "best_model_adamw.pt",
            )

        do_log = should_log_epoch(epoch, args.log_interval, args.adaptive_log_interval)
        if do_log:
            network.eval()
            with torch.inference_mode():
                pred_pre_sym, pred_post_sym = predict_full_matrices()
                detailed, extra, _ = evaluate_prediction_bundle(
                    pred_pre_sym,
                    pred_post_sym,
                    target_H_full,
                    overlap_e3nn,
                )

                history["mae_H"].append(float(detailed["mae"]))
                history["mse_H"].append(float(detailed["mse"]))

                now = time.time()
                dt = now - last_log_time
                last_log_time = now
                print("-" * 80)
                print(
                    f"epoch={epoch + 1}/{args.num_epochs} "
                    f"loss={float(loss.item()):.6e} mae={detailed['mae']:.6e} "
                    f"mse={detailed['mse']:.6e} dt={dt:.2f}s"
                )
                if args.benchmark:
                    print(
                        f"[BENCHMARK] epoch_time={time.perf_counter() - t_epoch:.4f}s"
                    )

                wandb_payload = {
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "mae_H": float(detailed["mae"]),
                    "mse_H": float(detailed["mse"]),
                    "mae_H_mod": float(detailed["mae_mod"]),
                    "mse_H_mod": float(detailed["mse_mod"]),
                    "mu_H": float(detailed["mu_H"]),
                    "correction_mae": float(detailed["correction_mae"]),
                    "correction_mse": float(detailed["correction_mse"]),
                }
                wandb_payload.update(
                    {f"train_extra/{k}": float(v) for k, v in extra.items()}
                )
                wandb.log(wandb_payload)

                pred_irreps_metrics = pred_post_sym.to_vectors(mapper)
                per_irrep_metrics = compute_irrep_metrics(
                    pred_irreps_metrics,
                    target_H_full_irreps,
                    all_irreps,
                    mapper,
                )
                if args.log_per_irrep_metrics:
                    print("[PER-IRREP]")
                    for ir in sorted(all_irreps, key=str):
                        irs = str(ir)
                        print(
                            f"  {irs}: "
                            f"l1_elem={per_irrep_metrics.get(f'{irs}_l1_elem', 0.0):.3e}, "
                            f"l2_elem={per_irrep_metrics.get(f'{irs}_l2_elem', 0.0):.3e}"
                        )
                wandb.log(
                    {
                        f"irrep_metrics/{str(ir)}_l1_elem": float(
                            per_irrep_metrics.get(f"{str(ir)}_l1_elem", 0.0)
                        )
                        for ir in sorted(all_irreps, key=str)
                    }
                )
                wandb.log(
                    {
                        f"irrep_metrics/{str(ir)}_l2_elem": float(
                            per_irrep_metrics.get(f"{str(ir)}_l2_elem", 0.0)
                        )
                        for ir in sorted(all_irreps, key=str)
                    }
                )

                try:
                    save_hamiltonian_frame_to_disk(
                        pred_post_sym,
                        target_H_full,
                        overlap_e3nn,
                        atoms_list,
                        snapshot.hamiltonian.orbital_cfg,
                        frame_output_dir,
                        epoch,
                        sx=0,
                        sy=0,
                        sz=0,
                        dynamic_range=False,
                        diff_dynamic_range=True,
                        partial_train=None,
                        percentile=99.0,
                    )
                except Exception as exc:
                    print(f"[WARN] frame save failed at epoch {epoch}: {exc}")

    # Final evaluation helper
    def evaluate_and_log(prefix: str) -> dict[str, float]:
        network.eval()
        with torch.inference_mode():
            pred_pre_sym, pred_post_sym = predict_full_matrices()
            detailed, extra, _ = evaluate_prediction_bundle(
                pred_pre_sym,
                pred_post_sym,
                target_H_full,
                overlap_e3nn,
            )

            payload: dict[str, float] = {}
            payload[f"{prefix}/mae_H"] = float(detailed["mae"])
            payload[f"{prefix}/mse_H"] = float(detailed["mse"])
            payload[f"{prefix}/mae_H_mod"] = float(detailed["mae_mod"])
            payload[f"{prefix}/mse_H_mod"] = float(detailed["mse_mod"])
            payload[f"{prefix}/mu_H"] = float(detailed["mu_H"])
            payload[f"{prefix}/correction_mae"] = float(detailed["correction_mae"])
            payload[f"{prefix}/correction_mse"] = float(detailed["correction_mse"])
            for k, v in extra.items():
                payload[f"{prefix}/{k}"] = float(v)

            # distance curve and DOS diagnostics for post-LBFGS only
            if prefix == "final_post_lbfgs":
                dos_plot_path = run_checkpoint_dir / "dos_comparison_final.png"
                try:
                    dos_metrics = save_dos_comparison_plot(
                        H_pred=pred_post_sym,
                        H_gt=target_H_full,
                        S=overlap_e3nn,
                        output_path=dos_plot_path,
                        sigma=0.2,
                        bin_width=0.1,
                        title="DOS Comparison",
                    )
                    for k, v in dos_metrics.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            payload[f"{prefix}/{k}"] = float(v)
                    wandb.log(
                        {
                            f"{prefix}/dos_comparison_plot": wandb.Image(
                                str(dos_plot_path)
                            )
                        }
                    )
                except Exception as exc:
                    print(f"[WARN] DOS plot failed: {exc}")

                distance_curve = compute_distance_error_curve(
                    H_pred=pred_post_sym,
                    H_gt=target_H_full,
                    positions=positions,
                    box=box,
                    partial_train=None,
                    n_bins=16,
                )
                if distance_curve is not None:
                    curve_json_path = run_checkpoint_dir / "distance_error_curve.json"
                    curve_plot_path = run_checkpoint_dir / "distance_error_curve.png"
                    with open(curve_json_path, "w", encoding="utf-8") as f:
                        import json

                        json.dump(distance_curve, f, indent=2)
                    save_distance_error_curve_plot(
                        distance_curve,
                        curve_plot_path,
                        title="Distance Error Curves (Final, 16 bins)",
                    )
                    wandb.log(
                        {
                            f"{prefix}/distance_curve_plot": wandb.Image(
                                str(curve_plot_path)
                            )
                        }
                    )

                if args.log_per_irrep_images:
                    irrep_output_dir = run_checkpoint_dir / "per_irrep_images"
                    irrep_output_dir.mkdir(parents=True, exist_ok=True)
                    for irrep in all_irreps:
                        irrep_str = str(irrep)
                        try:
                            pred_irrep = split_hamiltonian_by_irrep(
                                pred_post_sym, mapper, irrep_str
                            )
                            target_irrep = split_hamiltonian_by_irrep(
                                target_H_full, mapper, irrep_str
                            )
                            visualize_hamiltonians(
                                pred_irrep,
                                target_irrep,
                                overlap_e3nn,
                                atoms_list,
                                snapshot.hamiltonian.orbital_cfg,
                                k_range=0,
                                output_dir=irrep_output_dir,
                                dynamic_range=True,
                                diff_dynamic_range=True,
                                per_panel_dynamic_range=True,
                                partial_train=None,
                                filename_prefix=f"hamiltonian_{irrep_str}",
                                percentile=99.0,
                            )
                            image_path = (
                                irrep_output_dir
                                / f"hamiltonian_{irrep_str}_sx+0_sy+0_sz+0.png"
                            )
                            if image_path.exists():
                                wandb.log(
                                    {
                                        f"{prefix}/irrep_images/{irrep_str}": wandb.Image(
                                            str(image_path)
                                        )
                                    }
                                )
                        except Exception as exc:
                            print(
                                f"[WARN] per-irrep image failed for {irrep_str}: {exc}"
                            )

            wandb.log(payload)

            print(
                f"[{prefix}] mae_H={payload[f'{prefix}/mae_H']:.6e} mse_H={payload[f'{prefix}/mse_H']:.6e}"
            )
            print(
                f"[{prefix}] max_abs={payload[f'{prefix}/max_abs_element_error']:.6e} "
                f"diag_rmse={payload[f'{prefix}/diag_rmse']:.6e} "
                f"offdiag_rmse={payload[f'{prefix}/offdiag_rmse']:.6e}"
            )
            print(
                f"[{prefix}] symmetry_pre_rmse={payload[f'{prefix}/symmetry_violation_pre_rmse']:.6e} "
                f"symmetry_post_rmse={payload[f'{prefix}/symmetry_violation_post_rmse']:.6e}"
            )
            return payload

    print("=" * 80)
    print("PRE-LBFGS EVALUATION")
    print("=" * 80)
    pre_lbfgs_metrics = evaluate_and_log("final_pre_lbfgs")

    print("=" * 80)
    print("LBFGS STAGE")
    print("=" * 80)
    lbfgs = LBFGS(
        network.parameters(),
        lr=1.0,
        max_iter=max(int(args.lbfgs_steps), 1),
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def lbfgs_closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss, _ = compute_training_loss()
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Invalid LBFGS loss: {float(loss.item())}")
        loss.backward()
        return loss

    lbfgs_loss = lbfgs.step(lbfgs_closure)
    wandb.log({"lbfgs/final_objective": float(lbfgs_loss.item())})

    print("=" * 80)
    print("POST-LBFGS EVALUATION")
    print("=" * 80)
    post_lbfgs_metrics = evaluate_and_log("final_post_lbfgs")

    final_model_path = run_checkpoint_dir / "final_model.pt"
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "history": history,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "pre_lbfgs_metrics": pre_lbfgs_metrics,
            "post_lbfgs_metrics": post_lbfgs_metrics,
        },
        final_model_path,
    )

    if args.generate_video:
        try:
            video_path = run_checkpoint_dir / "training_progress.mp4"
            compile_frames_to_video(
                frame_output_dir,
                video_path,
                fps=5,
                pattern="frame_epoch_*.png",
                format="mp4",
            )
            wandb.log(
                {"training_video": wandb.Video(str(video_path), fps=5, format="mp4")}
            )
            print(f"[VIDEO] saved: {video_path}")
        except Exception as exc:
            print(f"[WARN] final video generation failed: {exc}")

    print("=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)
    print(f"checkpoint_dir={run_checkpoint_dir}")
    print(f"best_loss={best_loss:.6e} at epoch={best_epoch + 1}")
    print(f"pre_lbfgs_mae={pre_lbfgs_metrics['final_pre_lbfgs/mae_H']:.6e}")
    print(f"post_lbfgs_mae={post_lbfgs_metrics['final_post_lbfgs/mae_H']:.6e}")

    wandb.finish()


if __name__ == "__main__":
    main()
