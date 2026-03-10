"""
Minimal silicon study (multi-snapshot).

A simplified variant of minimal_overfit_study that:
  - loads multiple silicon snapshots from a directory,
  - builds train/val datasets via DatasetFactory,
  - trains MinimalNetwork on multiple snapshots.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# Add project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse minimal-overfit utilities.
MINIMAL_OVERFIT_DIR = Path(__file__).resolve().parents[1] / "minimal_overfit_study"
sys.path.insert(0, str(MINIMAL_OVERFIT_DIR))

from data.factory import DatasetFactory
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix, IrrepsBlockData
from core.block_irrep_mapper import BlockIrrepMapper
from net.common import Config as NetConfig, build_hidden_irreps
from data.graph_features import compute_graph_features
from e3nn.o3 import Irreps

from common import (
    MinimalNetwork,
    canonicalize_edge_order,
    compile_frames_to_video,
    compute_detailed_metrics,
    compute_distance_error_curve,
    compute_irrep_metrics,
    filter_irreps_block_data_by_irrep,
    get_all_irreps_in_hamiltonian,
    save_distance_error_curve_plot,
    save_dos_comparison_plot,
    save_hamiltonian_frame_to_disk,
    split_hamiltonian_by_irrep,
    visualize_hamiltonians,
)
from strict_checks import strict_edge_alignment_check, strict_reverse_edge_check
from detailed_logging import (
    build_wandb_detailed_metrics_log,
    build_wandb_per_irrep_metrics_log,
    log_config,
    log_detailed_training_metrics,
    log_final_metrics,
    log_mapper_info,
    log_orbital_config,
    log_per_irrep_metrics,
    log_study_complete,
)

DISTANCE_NORM_POWER = 4
DISTANCE_NORM_MIN_ARG = 1e-12
DISTANCE_NORM_MAG_EPS = 1e-300
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal silicon multi-snapshot study")

    # Data split arguments.
    parser.add_argument("--data-path", type=str, required=True)
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
        "--num-train",
        type=int,
        default=None,
        help="Global split mode: number of train pairs from discovered list",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=None,
        help="Global split mode: number of val pairs from discovered list (can be 0)",
    )
    parser.add_argument(
        "--train-temps",
        type=str,
        default=None,
        help=(
            "Temperature split mode: comma-separated training temperatures, e.g. "
            "'900,1200,1500'. If omitted, all detected temperatures except --val-temp are used."
        ),
    )
    parser.add_argument(
        "--val-temp",
        type=int,
        default=None,
        help=(
            "Temperature split mode: validation temperature (e.g., 2700). "
            "Can be one of training temperatures; validation snapshots are then "
            "selected disjointly from training snapshots."
        ),
    )
    parser.add_argument(
        "--n-snapshots-per-temp",
        type=int,
        default=None,
        help="Temperature split mode: number of snapshots sampled per training temperature",
    )
    parser.add_argument(
        "--val-n-snapshots",
        type=int,
        default=None,
        help=(
            "Temperature split mode: validation snapshots at --val-temp "
            "(defaults to --n-snapshots-per-temp, can be 0)"
        ),
    )

    # Selected features.
    parser.add_argument("--convention", type=str, default="e3nn")
    parser.add_argument(
        "--orbital-selection",
        type=str,
        default=None,
        help=(
            "Optional orbital reduction spec for all snapshots. "
            "Examples: '1s1p' or '{\"Si\":\"2s2p\"}'."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--l-max", type=int, default=4)
    parser.add_argument("--hidden-irreps", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--cutoff-radius", type=float, default=8.0)
    parser.add_argument("--n-radial", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=4000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--adaptive-log-interval", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--separate-shifted-self", action="store_true", default=False)
    parser.add_argument("--normalize-blocks", action="store_true", default=False)
    parser.add_argument(
        "--distance-magnitude-normalization",
        action="store_true",
        default=False,
        help=(
            "Pre-normalize off-diagonal block magnitudes using a distance model "
            "(fixed power=4), train in normalized space, and evaluate on de-normalized predictions."
        ),
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
    parser.add_argument("--head-mlp-for-scalars", action="store_true", default=False)
    parser.add_argument("--train-on-irrep-parts", action="store_true", default=False)
    parser.add_argument("--apply-cutoff-to-targets", action="store_true", default=False)
    parser.add_argument(
        "--require-exact-edge-match", action="store_true", default=False
    )

    parser.add_argument("--log-data", action="store_true", default=False)
    parser.add_argument("--log-model", action="store_true", default=False)
    parser.add_argument("--log-forward", action="store_true", default=False)
    parser.add_argument("--verbose-forward", action="store_true", default=False)
    parser.add_argument("--log-per-irrep-metrics", action="store_true", default=False)
    parser.add_argument("--log-per-irrep-images", action="store_true", default=False)
    parser.add_argument("--generate-video", action="store_true", default=False)
    parser.add_argument(
        "--video-max-atoms",
        type=int,
        default=6,
        help=(
            "For training-video frame plots, keep only first N atoms in the displayed "
            "matrix. Set <=0 to disable cropping."
        ),
    )

    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="studies/minimal_silicon_study/checkpoints",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_orbital_selection(selection_raw: str | None) -> Any:
    if selection_raw is None:
        return None
    stripped = selection_raw.strip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    return selection_raw


def discover_snapshot_pairs(root: Path) -> list[tuple[Path, Path]]:
    """
    Discover (matrix, info) pairs under root.
    """
    matrix_candidates = set()
    for pattern in ("Si_DM", "*_DM", "*.matrix", "*.scfout"):
        for p in root.rglob(pattern):
            if p.is_file():
                matrix_candidates.add(p.resolve())

    pairs: list[tuple[Path, Path]] = []
    for mat in sorted(matrix_candidates):
        parent = mat.parent
        info_candidates = [
            parent / "info.dat",
            parent / "info.out",
            parent / "info.txt",
            parent / "Si.info.out",
        ]
        info_path = None
        for c in info_candidates:
            if c.exists():
                info_path = c
                break
        if info_path is None:
            # Fallback: first sibling file starting with "info".
            fallback = sorted(parent.glob("info*"))
            if fallback:
                info_path = fallback[0]
        if info_path is not None:
            pairs.append((mat, info_path.resolve()))
    return pairs


def parse_temperature_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    temps: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if p.endswith("K") or p.endswith("k"):
            p = p[:-1]
        temps.append(int(p))
    if len(temps) == 0:
        raise ValueError("Parsed --train-temps is empty")
    return temps


def discover_temperatures(root: Path) -> list[int]:
    temps: list[int] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.endswith("K"):
            continue
        t = name[:-1]
        if t.isdigit():
            temps.append(int(t))
    return sorted(set(temps))


def discover_snapshot_pairs_for_temp(root: Path, temp: int) -> list[tuple[Path, Path]]:
    temp_dir = root / f"{temp}K"
    if not temp_dir.exists():
        return []
    pairs: list[tuple[Path, Path]] = []
    for mat in sorted(temp_dir.glob("*/Si_DM")):
        if not mat.is_file():
            continue
        info = mat.parent / "info.dat"
        if not info.exists():
            fallback = sorted(mat.parent.glob("info*"))
            if not fallback:
                continue
            info = fallback[0]
        pairs.append((mat.resolve(), info.resolve()))
    return pairs


def get_diagonal_mask(edges_5d: torch.Tensor) -> torch.Tensor:
    sx, sy, sz, i, j = edges_5d[0], edges_5d[1], edges_5d[2], edges_5d[3], edges_5d[4]
    return (sx == 0) & (sy == 0) & (sz == 0) & (i == j)


def get_block_status(
    edges_5d: torch.Tensor,
    edge_idx: int,
    separate_shifted_self: bool,
) -> str:
    sx = int(edges_5d[0, edge_idx].item())
    sy = int(edges_5d[1, edge_idx].item())
    sz = int(edges_5d[2, edge_idx].item())
    i = int(edges_5d[3, edge_idx].item())
    j = int(edges_5d[4, edge_idx].item())
    if i == j and sx == 0 and sy == 0 and sz == 0:
        return "diag"
    if i == j and separate_shifted_self:
        return "shifted_self"
    return "offdiag"


def compute_block_normalization_factors(
    block_matrix: BlockMatrix,
    separate_shifted_self: bool,
) -> dict[str, dict[str, float]]:
    norm_factors: dict[str, dict[str, float]] = {}
    for key in block_matrix.pair_blocks.keys():
        blocks = block_matrix.pair_blocks[key]
        edges = block_matrix.pair_edges[key]
        sx, sy, sz, i, j = edges[0], edges[1], edges[2], edges[3], edges[4]
        is_same_atom = i == j
        is_zero_shift = (sx == 0) & (sy == 0) & (sz == 0)
        diag_mask = is_same_atom & is_zero_shift
        shifted_self_mask = is_same_atom & (~is_zero_shift)
        if separate_shifted_self:
            offdiag_mask = ~is_same_atom
        else:
            offdiag_mask = ~diag_mask

        def avg_norm(mask: torch.Tensor) -> float:
            if mask.any():
                b = blocks[mask]
                v = torch.linalg.norm(b.reshape(b.shape[0], -1), dim=1).mean().item()
                return max(float(v), 1e-8)
            return 1.0

        entry = {
            "diag": avg_norm(diag_mask),
            "offdiag": avg_norm(offdiag_mask),
        }
        if separate_shifted_self:
            entry["shifted_self"] = avg_norm(shifted_self_mask)
        norm_factors[key] = entry
    return norm_factors


def compute_edge_distances_by_key(
    block_matrix: BlockMatrix,
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """
    Compute PBC-aware edge distances for every key in a BlockMatrix.
    """
    out: dict[str, torch.Tensor] = {}
    for key, edges in block_matrix.pair_edges.items():
        sx, sy, sz = edges[0], edges[1], edges[2]
        src, dst = edges[3], edges[4]
        shift = torch.stack([sx, sy, sz], dim=1).to(positions.dtype)
        if box is not None:
            disp = positions[dst] - positions[src] + shift @ box
        else:
            disp = positions[dst] - positions[src]
        out[key] = torch.linalg.norm(disp, dim=1)
    return out


def fit_distance_normalization_coeffs(
    samples: list[dict],
) -> dict[str, dict[str, float | int | bool]]:
    """
    Fit per-key coefficients for:
        z = (log_offset - log(m))^(1/4) - (a*r + b)
    on off-diagonal edges of training data only.
    """
    dist_acc: dict[str, list[np.ndarray]] = {}
    mag_acc: dict[str, list[np.ndarray]] = {}

    for sample in samples:
        H_raw: BlockMatrix = sample["target_H_matrix_raw"]
        dists_by_key: dict[str, torch.Tensor] = sample["edge_distances_by_key"]
        for key, blocks in H_raw.pair_blocks.items():
            if key not in dists_by_key:
                continue
            edges = H_raw.pair_edges[key]
            mask = ~get_diagonal_mask(edges)
            if not mask.any():
                continue

            mags = torch.linalg.norm(blocks[mask].reshape(mask.sum().item(), -1), dim=1)
            d = dists_by_key[key][mask]
            valid = torch.isfinite(d) & torch.isfinite(mags) & (d > 0) & (mags > 0)
            if not valid.any():
                continue

            d_np = d[valid].detach().cpu().numpy()
            m_np = mags[valid].detach().cpu().numpy()
            dist_acc.setdefault(key, []).append(d_np)
            mag_acc.setdefault(key, []).append(m_np)

    coeffs: dict[str, dict[str, float | int | bool]] = {}
    keys = sorted(set(list(dist_acc.keys()) + list(mag_acc.keys())))
    for key in keys:
        d_list = dist_acc.get(key, [])
        m_list = mag_acc.get(key, [])
        if len(d_list) == 0 or len(m_list) == 0:
            coeffs[key] = {
                "identity": True,
                "a": 0.0,
                "b": 0.0,
                "log_offset": 0.0,
                "power": DISTANCE_NORM_POWER,
                "min_arg": DISTANCE_NORM_MIN_ARG,
                "num_points": 0,
            }
            continue

        d = np.concatenate(d_list)
        m = np.concatenate(m_list)
        n = int(d.shape[0])
        if n < 2:
            coeffs[key] = {
                "identity": True,
                "a": 0.0,
                "b": 0.0,
                "log_offset": 0.0,
                "power": DISTANCE_NORM_POWER,
                "min_arg": DISTANCE_NORM_MIN_ARG,
                "num_points": n,
            }
            continue

        log_m = np.log(np.clip(m, DISTANCE_NORM_MAG_EPS, None))
        log_offset = max(0.0, float(np.max(log_m)) + DISTANCE_NORM_MIN_ARG)
        arg = np.maximum(log_offset - log_m, DISTANCE_NORM_MIN_ARG)
        y = np.power(arg, 1.0 / DISTANCE_NORM_POWER)

        if float(np.ptp(d)) <= 1e-12:
            a = 0.0
            b = float(np.mean(y))
        else:
            a, b = np.polyfit(d, y, deg=1)
            a = float(a)
            b = float(b)

        coeffs[key] = {
            "identity": False,
            "a": a,
            "b": b,
            "log_offset": float(log_offset),
            "power": DISTANCE_NORM_POWER,
            "min_arg": DISTANCE_NORM_MIN_ARG,
            "num_points": n,
        }

    return coeffs


def apply_distance_normalization_to_samples(
    samples: list[dict],
    mapper: BlockIrrepMapper,
    separate_shifted_self: bool,
    coeffs_by_key: dict[str, dict[str, float | int | bool]],
) -> None:
    """
    In-place: replace training targets with distance-normalized versions.
    Raw targets stay available in target_H_matrix_raw / target_H_irreps_raw.
    """
    for sample in samples:
        H_raw: BlockMatrix = sample["target_H_matrix_raw"]
        H_norm = normalize_block_matrix_with_distance_model(
            H_raw,
            sample["edge_distances_by_key"],
            coeffs_by_key,
        )
        sample["target_H_matrix"] = H_norm
        sample["target_H_irreps"] = H_norm.to_vectors(mapper)
        sample["norm_factors"] = compute_block_normalization_factors(
            H_norm, separate_shifted_self
        )


def normalize_block_matrix_with_distance_model(
    block_matrix: BlockMatrix,
    edge_distances_by_key: dict[str, torch.Tensor],
    coeffs_by_key: dict[str, dict[str, float | int | bool]],
) -> BlockMatrix:
    """
    Apply pre-training normalization to off-diagonal block magnitudes.
    """
    new_blocks: dict[str, torch.Tensor] = {}

    for key, blocks in block_matrix.pair_blocks.items():
        edges = block_matrix.pair_edges[key]
        dists = edge_distances_by_key[key].to(blocks.device, blocks.dtype)
        coeffs = coeffs_by_key.get(key, {"identity": True})
        identity = bool(coeffs.get("identity", True))
        offdiag_mask = ~get_diagonal_mask(edges)

        if identity or not offdiag_mask.any():
            new_blocks[key] = blocks
            continue

        a = float(coeffs["a"])
        b = float(coeffs["b"])
        log_offset = float(coeffs["log_offset"])

        out = blocks.clone()
        blk = blocks[offdiag_mask]
        dist = dists[offdiag_mask]

        mags = torch.linalg.norm(blk.reshape(blk.shape[0], -1), dim=1)
        mags_safe = torch.clamp(mags, min=DISTANCE_NORM_MAG_EPS)

        log_m = torch.log(mags_safe)
        arg = torch.clamp(log_offset - log_m, min=DISTANCE_NORM_MIN_ARG)
        y = torch.pow(arg, 1.0 / DISTANCE_NORM_POWER)
        z = y - (a * dist + b)
        mags_norm = torch.exp(z)

        scale = mags_safe
        while scale.ndim < blk.ndim:
            scale = scale.unsqueeze(-1)
        unit = blk / scale

        scale_norm = mags_norm
        while scale_norm.ndim < blk.ndim:
            scale_norm = scale_norm.unsqueeze(-1)
        out[offdiag_mask] = unit * scale_norm

        new_blocks[key] = out

    return block_matrix._replace_pair_blocks(new_blocks, basis=block_matrix.basis)


def denormalize_block_matrix_with_distance_model(
    block_matrix: BlockMatrix,
    edge_distances_by_key: dict[str, torch.Tensor],
    coeffs_by_key: dict[str, dict[str, float | int | bool]],
) -> BlockMatrix:
    """
    Invert distance-based normalization on off-diagonal block magnitudes.
    """
    new_blocks: dict[str, torch.Tensor] = {}

    for key, blocks in block_matrix.pair_blocks.items():
        edges = block_matrix.pair_edges[key]
        dists = edge_distances_by_key[key].to(blocks.device, blocks.dtype)
        coeffs = coeffs_by_key.get(key, {"identity": True})
        identity = bool(coeffs.get("identity", True))
        offdiag_mask = ~get_diagonal_mask(edges)

        if identity or not offdiag_mask.any():
            new_blocks[key] = blocks
            continue

        a = float(coeffs["a"])
        b = float(coeffs["b"])
        log_offset = float(coeffs["log_offset"])

        out = blocks.clone()
        blk = blocks[offdiag_mask]
        dist = dists[offdiag_mask]

        mags_norm = torch.linalg.norm(blk.reshape(blk.shape[0], -1), dim=1)
        mags_norm_safe = torch.clamp(mags_norm, min=DISTANCE_NORM_MAG_EPS)

        z = torch.log(mags_norm_safe)
        y = z + (a * dist + b)
        arg = torch.pow(y, DISTANCE_NORM_POWER)
        log_m = log_offset - arg
        mags = torch.exp(log_m)

        scale = mags_norm_safe
        while scale.ndim < blk.ndim:
            scale = scale.unsqueeze(-1)
        unit = blk / scale

        scale_raw = mags
        while scale_raw.ndim < blk.ndim:
            scale_raw = scale_raw.unsqueeze(-1)
        out[offdiag_mask] = unit * scale_raw

        new_blocks[key] = out

    return block_matrix._replace_pair_blocks(new_blocks, basis=block_matrix.basis)


def canonicalize_block_matrix_edges(
    block_matrix: BlockMatrix,
    positions: torch.Tensor,
    box: torch.Tensor | None,
) -> BlockMatrix:
    order_dict: dict[str, torch.Tensor] = {}
    for key, edges_5d in block_matrix.pair_edges.items():
        edge_shift_local = edges_5d[:3]
        edge_index_local = edges_5d[3:]
        _, _, perm = canonicalize_edge_order(
            edge_index=edge_index_local,
            edge_shift=edge_shift_local,
            positions=positions,
            box=box,
        )
        order_dict[key] = perm
    return block_matrix.reorder_edges(order_dict)


def filter_block_matrix_by_cutoff(
    block_matrix: BlockMatrix,
    positions: torch.Tensor,
    box: torch.Tensor | None,
    cutoff_radius: float,
) -> BlockMatrix:
    mask_dict: dict[str, torch.Tensor] = {}
    for key, edges in block_matrix.pair_edges.items():
        sx, sy, sz = edges[0], edges[1], edges[2]
        src, dst = edges[3], edges[4]
        shift = torch.stack([sx, sy, sz], dim=1).to(positions.dtype)
        if box is not None:
            disp = positions[dst] - positions[src] + shift @ box
        else:
            disp = positions[dst] - positions[src]
        dist = torch.linalg.norm(disp, dim=1)
        mask_dict[key] = dist <= float(cutoff_radius)
    return block_matrix._apply_edge_mask(mask_dict, drop_empty=True)


def should_log_epoch(epoch_zero_based: int, log_interval: int, adaptive: bool) -> bool:
    if not adaptive:
        return epoch_zero_based % log_interval == 0
    ep = epoch_zero_based + 1
    if ep <= 10:
        return True
    if ep <= 100:
        return ep % 10 == 0
    return epoch_zero_based % log_interval == 0


def prepare_mapper_from_sample(
    x: dict,
    y: dict,
    orbital_selection: Any,
    device: torch.device,
) -> BlockIrrepMapper:
    snap = Snapshot(
        hamiltonian=y["hamiltonian"],
        overlap=y["overlap"],
        density=y["density"],
        positions=x["positions"],
        box=x["box"],
    )
    if orbital_selection is not None:
        snap = snap.reduce_orbitals(orbital_selection)
    return BlockIrrepMapper(
        snap.hamiltonian.orbital_cfg, device=device, dtype=torch.float32
    )


def preprocess_sample(
    x: dict,
    y: dict,
    mapper: BlockIrrepMapper,
    cfg_graph: NetConfig,
    sh_irreps: Irreps,
    hamiltonian_scale_from_hartree: float,
    cutoff_radius: float,
    apply_cutoff_to_targets: bool,
    orbital_selection: Any,
    separate_shifted_self: bool,
    require_exact_edge_match: bool,
    device: torch.device,
) -> dict:
    snap = Snapshot(
        hamiltonian=y["hamiltonian"],
        overlap=y["overlap"],
        density=y["density"],
        positions=x["positions"],
        box=x["box"],
    )

    if orbital_selection is not None:
        snap = snap.reduce_orbitals(orbital_selection)

    positions = snap.positions.to(device)
    box = snap.box.to(device) if snap.box is not None else None
    atoms_list = list(snap.hamiltonian.atoms)

    H = snap.hamiltonian.to(device) * float(hamiltonian_scale_from_hartree)
    S = snap.overlap.to(device)
    D = snap.density.to(device)

    if apply_cutoff_to_targets:
        H = filter_block_matrix_by_cutoff(H, positions, box, cutoff_radius)
        S = filter_block_matrix_by_cutoff(S, positions, box, cutoff_radius)
        D = filter_block_matrix_by_cutoff(D, positions, box, cutoff_radius)

    H = canonicalize_block_matrix_edges(H, positions, box)
    S = canonicalize_block_matrix_edges(S, positions, box)
    D = canonicalize_block_matrix_edges(D, positions, box)

    H = (H + H.transpose()) * 0.5
    edge_distances_by_key = compute_edge_distances_by_key(H, positions, box)

    (
        edge_index,
        edge_shift,
        edge_type_idx,
        edge_length_emb,
        edge_sh,
        _num_self_edges,
    ) = compute_graph_features(
        positions=positions,
        box=box,
        atoms=tuple(atoms_list),
        cfg=cfg_graph,
        sh_irreps=sh_irreps,
        edge_type2idx=mapper.edge_type2idx,
    )

    # extra canonicalization to match current training script convention exactly
    edge_index, edge_shift, _ = canonicalize_edge_order(
        edge_index=edge_index,
        edge_shift=edge_shift,
        positions=positions,
        box=box,
    )

    strict_reverse_edge_check(
        edge_index=edge_index, edge_shift=edge_shift, edge_set_name="graph"
    )
    strict_edge_alignment_check(
        target_matrix=H,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="hamiltonian",
        require_exact=require_exact_edge_match,
    )
    strict_edge_alignment_check(
        target_matrix=S,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="overlap",
        require_exact=require_exact_edge_match,
    )
    strict_edge_alignment_check(
        target_matrix=D,
        edge_index=edge_index,
        edge_shift=edge_shift,
        edge_type_idx=edge_type_idx,
        atoms_list=atoms_list,
        edge_types=mapper.edge_types,
        matrix_name="density",
        require_exact=require_exact_edge_match,
    )

    element_to_idx = {
        elem: idx for idx, elem in enumerate(mapper.orbital_cfg.elements())
    }
    node_type_idx = torch.tensor(
        [element_to_idx[a] for a in atoms_list], dtype=torch.long, device=device
    )
    batch_node = torch.zeros(len(atoms_list), dtype=torch.long, device=device)
    batch_edge = torch.zeros(edge_index.shape[1], dtype=torch.long, device=device)

    return {
        "node_type_idx": node_type_idx,
        "edge_type_idx": edge_type_idx.to(device),
        "edge_index": edge_index.to(device),
        "edge_shift": edge_shift.to(device),
        "edge_length_emb": edge_length_emb.to(device),
        "edge_sh": edge_sh.to(device),
        "batch_node": batch_node,
        "batch_edge": batch_edge,
        "positions": positions,
        "box": box,
        "atoms_list": atoms_list,
        "edge_distances_by_key": edge_distances_by_key,
        "target_H_matrix_raw": H,
        "target_H_irreps_raw": H.to_vectors(mapper),
        "target_H_matrix": H,
        "target_overlap": S,
        "target_density": D,
        "target_H_irreps": H.to_vectors(mapper),
        "norm_factors": compute_block_normalization_factors(H, separate_shifted_self),
    }


def compute_loss_for_sample(
    pred_irreps: IrrepsBlockData,
    pred_matrix: BlockMatrix,
    sample: dict,
    mapper: BlockIrrepMapper,
    *,
    train_on_irrep_parts: bool,
    normalize_blocks: bool,
    separate_shifted_self: bool,
    all_irreps: list,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_H_matrix = sample["target_H_matrix"]
    target_H_irreps = sample["target_H_irreps"]
    norm_factors = sample["norm_factors"]
    irrep_losses: dict[str, torch.Tensor] = {}

    def mse_with_optional_norm(
        pred_blocks: torch.Tensor,
        targ_blocks: torch.Tensor,
        edges_5d: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        min_n = min(pred_blocks.shape[0], targ_blocks.shape[0], edges_5d.shape[1])
        p = pred_blocks[:min_n]
        t = targ_blocks[:min_n]
        if normalize_blocks:
            norm_vec = torch.ones(min_n, device=p.device, dtype=p.dtype)
            for e in range(min_n):
                status = get_block_status(edges_5d, e, separate_shifted_self)
                norm_vec[e] = norm_factors[key][status]
            while norm_vec.ndim < p.ndim:
                norm_vec = norm_vec.unsqueeze(-1)
            p = p / norm_vec
            t = t / norm_vec
        return F.mse_loss(p, t)

    if train_on_irrep_parts:
        loss_total = torch.tensor(
            0.0,
            device=pred_matrix.pair_blocks[next(iter(pred_matrix.pair_blocks))].device,
        )
        for ir in all_irreps:
            pred_part = filter_irreps_block_data_by_irrep(
                pred_irreps, ir, mapper
            ).to_blocks(mapper)
            targ_part = filter_irreps_block_data_by_irrep(
                target_H_irreps, ir, mapper
            ).to_blocks(mapper)
            l_ir = torch.tensor(0.0, device=loss_total.device)
            for key in targ_part.pair_blocks.keys():
                if key not in pred_part.pair_blocks:
                    continue
                l_ir = l_ir + mse_with_optional_norm(
                    pred_part.pair_blocks[key],
                    targ_part.pair_blocks[key],
                    target_H_matrix.pair_edges[key],
                    key,
                )
            irrep_losses[str(ir)] = l_ir
            loss_total = loss_total + l_ir
        return loss_total, irrep_losses

    loss_total = torch.tensor(
        0.0, device=pred_matrix.pair_blocks[next(iter(pred_matrix.pair_blocks))].device
    )
    for key in target_H_matrix.pair_blocks.keys():
        if key not in pred_matrix.pair_blocks:
            continue
        loss_total = loss_total + mse_with_optional_norm(
            pred_matrix.pair_blocks[key],
            target_H_matrix.pair_blocks[key],
            target_H_matrix.pair_edges[key],
            key,
        )
    return loss_total, irrep_losses


def predict_sample(
    network: MinimalNetwork,
    sample: dict,
    mapper: BlockIrrepMapper,
    *,
    verbose_forward: bool,
    log_forward: bool,
) -> tuple[IrrepsBlockData, BlockMatrix, BlockMatrix]:
    if not verbose_forward and not log_forward:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    raw = network(
        sample["node_type_idx"],
        sample["edge_type_idx"],
        sample["edge_index"],
        sample["edge_shift"],
        sample["edge_length_emb"],
        sample["edge_sh"],
        sample["batch_node"],
        sample["batch_edge"],
        log_to_wandb=False,
    )
    if not verbose_forward and not log_forward:
        sys.stdout.close()
        sys.stdout = old_stdout

    pair_vec = {}
    pair_edges = {}
    lookup = {}
    for key, payload in raw.items():
        pair_vec[key] = payload["vectors"]
        pair_edges[key] = payload["edges"]
        for idx, e5d in enumerate(payload["edges"].t()):
            sx, sy, sz, i, j = map(int, e5d.tolist())
            lookup[(sx, sy, sz, i, j)] = (key, idx)

    pred_irreps = IrrepsBlockData(
        atoms=tuple(sample["atoms_list"]),
        atom_counts=Counter(sample["atoms_list"]),
        pair_vectors=pair_vec,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=mapper.orbital_cfg,
        basis=sample["target_H_matrix"].basis,
    )
    pred_matrix = pred_irreps.to_blocks(mapper)
    pred_metrics = (pred_matrix + pred_matrix.transpose()) * 0.5
    return pred_irreps, pred_matrix, pred_metrics


def evaluate_split(
    network: MinimalNetwork,
    samples: list[dict],
    mapper: BlockIrrepMapper,
    all_irreps: list,
    *,
    train_on_irrep_parts: bool,
    normalize_blocks: bool,
    separate_shifted_self: bool,
    distance_magnitude_normalization: bool,
    distance_norm_coeffs: dict[str, dict[str, float | int | bool]] | None,
) -> dict:
    empty_detailed = {
        "mae": 0.0,
        "mse": 0.0,
        "mae_mod": 0.0,
        "mse_mod": 0.0,
        "mu_H": 0.0,
        "correction_mae": 0.0,
        "correction_mse": 0.0,
    }
    if not samples:
        return {
            "loss": 0.0,
            "detailed": empty_detailed,
            "per_irrep": {},
            "first_pred_metrics": None,
            "first_sample": None,
        }

    network.eval()
    total_loss = 0.0
    detailed_sum: dict[str, float] = {}
    per_irrep_sum: dict[str, float] = {}
    first_pred_metrics = None
    first_sample = None

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            pred_irreps, pred_matrix, pred_metrics = predict_sample(
                network, sample, mapper, verbose_forward=False, log_forward=False
            )
            loss_t, _ = compute_loss_for_sample(
                pred_irreps,
                pred_matrix,
                sample,
                mapper,
                train_on_irrep_parts=train_on_irrep_parts,
                normalize_blocks=normalize_blocks,
                separate_shifted_self=separate_shifted_self,
                all_irreps=all_irreps,
            )
            total_loss += float(loss_t.item())

            if distance_magnitude_normalization:
                if distance_norm_coeffs is None:
                    raise RuntimeError(
                        "distance_magnitude_normalization is enabled but coefficients are missing."
                    )
                pred_eval = denormalize_block_matrix_with_distance_model(
                    pred_metrics,
                    sample["edge_distances_by_key"],
                    distance_norm_coeffs,
                )
                target_eval_matrix = sample["target_H_matrix_raw"]
                target_eval_irreps = sample["target_H_irreps_raw"]
            else:
                pred_eval = pred_metrics
                target_eval_matrix = sample["target_H_matrix"]
                target_eval_irreps = sample["target_H_irreps"]

            detailed = compute_detailed_metrics(
                pred_eval, target_eval_matrix, sample["target_overlap"]
            )
            for k, v in detailed.items():
                detailed_sum[k] = detailed_sum.get(k, 0.0) + float(v)

            pred_ir_metrics = pred_eval.to_vectors(mapper)
            per_irrep = compute_irrep_metrics(
                pred_ir_metrics, target_eval_irreps, all_irreps, mapper
            )
            for k, v in per_irrep.items():
                per_irrep_sum[k] = per_irrep_sum.get(k, 0.0) + float(v)

            if idx == 0:
                first_pred_metrics = pred_eval
                first_sample = sample

    n = float(len(samples))
    return {
        "loss": total_loss / n,
        "detailed": {k: v / n for k, v in detailed_sum.items()},
        "per_irrep": {k: v / n for k, v in per_irrep_sum.items()},
        "first_pred_metrics": first_pred_metrics,
        "first_sample": first_sample,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("MINIMAL SILICON STUDY - MULTI SNAPSHOT")
    print("=" * 80)

    data_root = Path(args.data_path).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data-path does not exist: {data_root}")

    temperature_mode = any(
        [
            args.train_temps is not None,
            args.val_temp is not None,
            args.n_snapshots_per_temp is not None,
            args.val_n_snapshots is not None,
        ]
    )

    all_pairs: list[tuple[Path, Path]] | None = None
    train_pairs: list[tuple[Path, Path]]
    val_pairs: list[tuple[Path, Path]]
    train_temps_effective: list[int] | None = None
    split_mode: str

    if temperature_mode:
        if args.val_temp is None:
            raise ValueError(
                "Temperature split mode requires --val-temp to be specified."
            )
        if args.n_snapshots_per_temp is None:
            raise ValueError(
                "Temperature split mode requires --n-snapshots-per-temp to be specified."
            )
        if args.n_snapshots_per_temp <= 0:
            raise ValueError("--n-snapshots-per-temp must be > 0")
        if args.val_n_snapshots is not None and args.val_n_snapshots < 0:
            raise ValueError("--val-n-snapshots must be >= 0 when provided")

        requested_train_temps = parse_temperature_list(args.train_temps)
        if requested_train_temps is None:
            discovered_temps = discover_temperatures(data_root)
            train_temps_effective = [t for t in discovered_temps if t != args.val_temp]
        else:
            train_temps_effective = sorted(set(requested_train_temps))

        if len(train_temps_effective) == 0:
            raise ValueError("No training temperatures selected.")

        train_pairs = []
        train_pairs_by_temp: dict[int, list[tuple[Path, Path]]] = {}
        for temp in train_temps_effective:
            temp_pairs = discover_snapshot_pairs_for_temp(data_root, temp)
            if len(temp_pairs) == 0:
                raise ValueError(
                    f"No snapshot pairs found for training temperature {temp}K"
                )
            n_pick = min(len(temp_pairs), args.n_snapshots_per_temp)
            picked = random.sample(temp_pairs, n_pick)
            train_pairs_by_temp[temp] = picked
            train_pairs.extend(picked)

        val_n = (
            args.val_n_snapshots
            if args.val_n_snapshots is not None
            else args.n_snapshots_per_temp
        )
        val_n = max(int(val_n), 0)
        if val_n == 0:
            val_pairs = []
        else:
            val_temp_pairs = discover_snapshot_pairs_for_temp(data_root, args.val_temp)
            if len(val_temp_pairs) == 0:
                raise ValueError(
                    f"No snapshot pairs found for validation temperature {args.val_temp}K"
                )
            if args.val_temp in train_pairs_by_temp:
                train_set_this_temp = set(train_pairs_by_temp[args.val_temp])
                val_temp_pairs = [
                    p for p in val_temp_pairs if p not in train_set_this_temp
                ]
                if len(val_temp_pairs) == 0:
                    raise ValueError(
                        "No disjoint validation snapshots available at "
                        f"{args.val_temp}K after removing training snapshots."
                    )
            val_pairs = random.sample(val_temp_pairs, min(len(val_temp_pairs), val_n))
        split_mode = "temperature"
    else:
        if args.num_train is None or args.num_val is None:
            raise ValueError(
                "Provide either temperature split args "
                "(--train-temps/--val-temp/--n-snapshots-per-temp) "
                "or global split args (--num-train and --num-val)."
            )
        if args.num_train <= 0:
            raise ValueError("--num-train must be > 0")
        if args.num_val < 0:
            raise ValueError("--num-val must be >= 0")
        all_pairs = discover_snapshot_pairs(data_root)
        if len(all_pairs) < args.num_train + args.num_val:
            raise ValueError(
                f"Requested train+val={args.num_train + args.num_val} pairs but found only {len(all_pairs)} under {data_root}"
            )
        train_pairs = all_pairs[: args.num_train]
        val_pairs = all_pairs[args.num_train : args.num_train + args.num_val]
        split_mode = "global"

    if len(train_pairs) == 0:
        raise ValueError("Resolved training snapshot list is empty.")

    device = torch.device(args.device)
    orbital_selection_obj = parse_orbital_selection(args.orbital_selection)

    config = {
        "data_path": str(data_root),
        "training_unit": args.training_unit,
        "hamiltonian_scale_from_hartree": UNIT_SCALE_FROM_HARTREE[args.training_unit],
        "split_mode": split_mode,
        "num_train": args.num_train,
        "num_val": args.num_val,
        "train_temps": args.train_temps,
        "val_temp": args.val_temp,
        "n_snapshots_per_temp": args.n_snapshots_per_temp,
        "val_n_snapshots": args.val_n_snapshots,
        "effective_train_temps": train_temps_effective,
        "convention": args.convention,
        "orbital_selection": args.orbital_selection,
        "hidden_dim": args.hidden_dim,
        "l_max": args.l_max,
        "hidden_irreps": args.hidden_irreps,
        "num_layers": args.num_layers,
        "cutoff_radius": args.cutoff_radius,
        "n_radial": args.n_radial,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "grad_clip": args.grad_clip,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "log_interval": args.log_interval,
        "adaptive_log_interval": args.adaptive_log_interval,
        "benchmark": args.benchmark,
        "separate_shifted_self": args.separate_shifted_self,
        "normalize_blocks": args.normalize_blocks,
        "distance_magnitude_normalization": args.distance_magnitude_normalization,
        "edge_encoder_use_sh_tensor_square": args.edge_encoder_use_sh_tensor_square,
        "head_mlp_for_scalars": args.head_mlp_for_scalars,
        "train_on_irrep_parts": args.train_on_irrep_parts,
        "apply_cutoff_to_targets": args.apply_cutoff_to_targets,
        "require_exact_edge_match": args.require_exact_edge_match,
        "log_data": args.log_data,
        "log_model": args.log_model,
        "log_forward": args.log_forward,
        "verbose_forward": args.verbose_forward,
        "log_per_irrep_metrics": args.log_per_irrep_metrics,
        "log_per_irrep_images": args.log_per_irrep_images,
        "generate_video": args.generate_video,
        "video_max_atoms": args.video_max_atoms,
        "device": args.device,
        "checkpoint_dir": args.checkpoint_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        # Logger-compat keys expected by detailed_logging.log_config from minimal_overfit_study.
        "partial_train": None,
        "box_convention": "rows",
        "log_activations_wandb": False,
    }

    wandb_kwargs = {"project": "mandala-minimal-silicon-study", "config": config}
    if args.run_name is not None:
        wandb_kwargs["name"] = args.run_name
    wandb.init(**wandb_kwargs)

    run_name = wandb.run.name
    run_checkpoint_dir = Path(args.checkpoint_dir) / run_name
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frame_output_dir = run_checkpoint_dir / "frames"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "training unit: "
        f"{UNIT_DISPLAY_NAME[args.training_unit]} "
        f"(x{config['hamiltonian_scale_from_hartree']:.7g} from Hartree)"
    )
    log_config(config, run_checkpoint_dir, frame_output_dir)

    # Build datasets via DatasetFactory (requested).
    # Use very large cutoff when targets should stay unfiltered.
    cfg_ds = NetConfig()
    cfg_ds.dtype = torch.float32
    cfg_ds.device = "cpu"
    cfg_ds.verbosity = 0
    cfg_ds.cache_root = None
    cfg_ds.precompute_edge_features = False
    cfg_ds.cutoff_radius = args.cutoff_radius if args.apply_cutoff_to_targets else 1e6

    fac = DatasetFactory(cfg_ds, convention=args.convention)
    for m, i in train_pairs:
        fac.add_snapshot(m, i, purpose="train")
    for m, i in val_pairs:
        fac.add_snapshot(m, i, purpose="val")
    train_ds, val_ds, _mapper_full = fac.create()

    if len(train_ds) == 0:
        raise ValueError(
            "DatasetFactory created an empty train dataset. "
            "Check file integrity (Si_DM/info.dat) and convention/orbital settings."
        )

    if args.log_data:
        if all_pairs is not None:
            print(f"[DATA] discovered pairs: {len(all_pairs)}")
        if split_mode == "temperature":
            print(
                f"[DATA] temperature split: train_temps={train_temps_effective}, val_temp={args.val_temp}"
            )
            print(
                f"[DATA] n_snapshots_per_temp={args.n_snapshots_per_temp}, "
                f"val_n_snapshots={args.val_n_snapshots if args.val_n_snapshots is not None else args.n_snapshots_per_temp}"
            )
        print(f"[DATA] using train={len(train_pairs)}, val={len(val_pairs)}")
        print(f"[DATA] first train pair: {train_pairs[0] if train_pairs else 'N/A'}")

    # Build final mapper from first sample after optional orbital reduction.
    x0, y0 = train_ds[0]
    mapper = prepare_mapper_from_sample(x0, y0, orbital_selection_obj, device)
    if args.log_model:
        log_orbital_config(mapper.orbital_cfg)
        log_mapper_info(mapper)

    # Graph feature config used per sample.
    cfg_graph = NetConfig()
    cfg_graph.cutoff_radius = args.cutoff_radius
    cfg_graph.n_radial = args.n_radial
    cfg_graph.l_max = args.l_max
    cfg_graph.dtype = torch.float32
    cfg_graph.safety_checks = False
    sh_irreps = Irreps.spherical_harmonics(args.l_max)

    if args.log_data:
        print("[DATA] preprocessing train samples...")
    train_samples = [
        preprocess_sample(
            x=x,
            y=y,
            mapper=mapper,
            cfg_graph=cfg_graph,
            sh_irreps=sh_irreps,
            hamiltonian_scale_from_hartree=config["hamiltonian_scale_from_hartree"],
            cutoff_radius=args.cutoff_radius,
            apply_cutoff_to_targets=args.apply_cutoff_to_targets,
            orbital_selection=orbital_selection_obj,
            separate_shifted_self=args.separate_shifted_self,
            require_exact_edge_match=args.require_exact_edge_match,
            device=device,
        )
        for (x, y) in train_ds
    ]

    if args.log_data:
        print("[DATA] preprocessing val samples...")
    val_iterable = val_ds if val_ds is not None else []
    val_samples = [
        preprocess_sample(
            x=x,
            y=y,
            mapper=mapper,
            cfg_graph=cfg_graph,
            sh_irreps=sh_irreps,
            hamiltonian_scale_from_hartree=config["hamiltonian_scale_from_hartree"],
            cutoff_radius=args.cutoff_radius,
            apply_cutoff_to_targets=args.apply_cutoff_to_targets,
            orbital_selection=orbital_selection_obj,
            separate_shifted_self=args.separate_shifted_self,
            require_exact_edge_match=args.require_exact_edge_match,
            device=device,
        )
        for (x, y) in val_iterable
    ]

    distance_norm_coeffs: dict[str, dict[str, float | int | bool]] | None = None
    if args.distance_magnitude_normalization:
        distance_norm_coeffs = fit_distance_normalization_coeffs(train_samples)
        apply_distance_normalization_to_samples(
            train_samples,
            mapper,
            args.separate_shifted_self,
            distance_norm_coeffs,
        )
        apply_distance_normalization_to_samples(
            val_samples,
            mapper,
            args.separate_shifted_self,
            distance_norm_coeffs,
        )
        if args.log_data:
            non_identity = sum(
                1
                for c in distance_norm_coeffs.values()
                if not bool(c.get("identity", True))
            )
            print(
                "[DATA] distance normalization enabled: "
                f"{len(distance_norm_coeffs)} key(s), {non_identity} fitted"
            )

    if args.hidden_irreps is not None:
        hidden_irreps = Irreps(args.hidden_irreps)
    else:
        hidden_irreps = build_hidden_irreps(
            l_max=args.l_max, base_dim=args.hidden_dim, use_odd_features=True
        )
        config["hidden_irreps"] = str(hidden_irreps)

    num_elements = len(mapper.orbital_cfg.elements())
    num_edge_types = num_elements**2

    if not args.log_model:
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
    network = MinimalNetwork(
        num_elements=num_elements,
        n_radial=args.n_radial,
        num_edge_types=num_edge_types,
        hidden_irreps=hidden_irreps,
        sh_irreps=sh_irreps,
        num_layers=args.num_layers,
        mapper=mapper,
        edge_encoder_use_sh_tensor_square=args.edge_encoder_use_sh_tensor_square,
        magnitude_factorization=False,
        head_mlp_for_scalars=args.head_mlp_for_scalars,
        head_use_tensor_square=False,
        separate_shifted_self=args.separate_shifted_self,
    ).to(device)
    if not args.log_model:
        sys.stdout.close()
        sys.stdout = old_stdout
    if args.log_model:
        print("[OK] Network architecture complete.")

    all_irreps = get_all_irreps_in_hamiltonian(mapper)
    optimizer = Adam(network.parameters(), lr=args.lr)
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

    print("")
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"train samples: {len(train_samples)} | val samples: {len(val_samples)}")

    history = {"train_loss": [], "val_loss": [], "val_mae_H": []}
    best_score = float("inf")
    best_epoch = -1
    best_model_path = run_checkpoint_dir / "best_model.pt"
    last_log_time = time.time()
    last_logged_epoch = -1

    benchmark_order = [
        "epoch_total",
        "train_forward",
        "train_loss",
        "backward_total",
        "optimizer_step",
        "eval_total",
        "wandb_log",
    ]
    benchmark_accum = {k: 0.0 for k in benchmark_order}
    benchmark_epochs = 0

    def benchmark_add(name: str, dt: float) -> None:
        if args.benchmark and name in benchmark_accum:
            benchmark_accum[name] += max(float(dt), 0.0)

    def benchmark_report() -> None:
        if not args.benchmark or benchmark_epochs <= 0:
            return
        epoch_ms = benchmark_accum["epoch_total"] * 1000.0 / benchmark_epochs
        print("\n[BENCHMARK]")
        print(f"  averaged over {benchmark_epochs} epoch(s): {epoch_ms:.3f} ms/epoch")
        for k in benchmark_order:
            ms = benchmark_accum[k] * 1000.0 / benchmark_epochs
            pct = 100.0 * ms / max(epoch_ms, 1e-12)
            print(f"  {k:24s} {ms:10.3f} ms  ({pct:6.2f}%)")

    try:
        for epoch in range(args.num_epochs):
            epoch_t0 = time.perf_counter() if args.benchmark else 0.0
            network.train()
            train_loss_total = 0.0
            train_irrep_losses_total: dict[str, float] = {}

            for sample in train_samples:
                optimizer.zero_grad()
                t_fwd = time.perf_counter() if args.benchmark else 0.0
                pred_irreps, pred_matrix, _pred_metrics = predict_sample(
                    network,
                    sample,
                    mapper,
                    verbose_forward=args.verbose_forward,
                    log_forward=args.log_forward
                    and should_log_epoch(
                        epoch, args.log_interval, args.adaptive_log_interval
                    ),
                )
                benchmark_add(
                    "train_forward",
                    time.perf_counter() - t_fwd if args.benchmark else 0.0,
                )

                t_loss = time.perf_counter() if args.benchmark else 0.0
                loss, ir_losses = compute_loss_for_sample(
                    pred_irreps=pred_irreps,
                    pred_matrix=pred_matrix,
                    sample=sample,
                    mapper=mapper,
                    train_on_irrep_parts=args.train_on_irrep_parts,
                    normalize_blocks=args.normalize_blocks,
                    separate_shifted_self=args.separate_shifted_self,
                    all_irreps=all_irreps,
                )
                benchmark_add(
                    "train_loss",
                    time.perf_counter() - t_loss if args.benchmark else 0.0,
                )

                t_bwd = time.perf_counter() if args.benchmark else 0.0
                loss.backward()
                benchmark_add(
                    "backward_total",
                    time.perf_counter() - t_bwd if args.benchmark else 0.0,
                )

                if args.grad_clip > 0:
                    clip_grad_norm_(network.parameters(), args.grad_clip)

                t_opt = time.perf_counter() if args.benchmark else 0.0
                optimizer.step()
                benchmark_add(
                    "optimizer_step",
                    time.perf_counter() - t_opt if args.benchmark else 0.0,
                )

                train_loss_total += float(loss.item())
                for k, v in ir_losses.items():
                    train_irrep_losses_total[k] = train_irrep_losses_total.get(
                        k, 0.0
                    ) + float(v.item())

            train_loss = train_loss_total / max(len(train_samples), 1)

            do_log = should_log_epoch(
                epoch, args.log_interval, args.adaptive_log_interval
            )
            val_eval = {
                "loss": train_loss,
                "detailed": {},
                "per_irrep": {},
                "first_pred_metrics": None,
                "first_sample": None,
            }
            if do_log:
                t_eval = time.perf_counter() if args.benchmark else 0.0
                val_eval = evaluate_split(
                    network=network,
                    samples=val_samples if len(val_samples) > 0 else train_samples,
                    mapper=mapper,
                    all_irreps=all_irreps,
                    train_on_irrep_parts=args.train_on_irrep_parts,
                    normalize_blocks=args.normalize_blocks,
                    separate_shifted_self=args.separate_shifted_self,
                    distance_magnitude_normalization=args.distance_magnitude_normalization,
                    distance_norm_coeffs=distance_norm_coeffs,
                )
                benchmark_add(
                    "eval_total",
                    time.perf_counter() - t_eval if args.benchmark else 0.0,
                )

            target_scheduler_loss = val_eval["loss"] if do_log else train_loss
            scheduler.step(target_scheduler_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(
                float(val_eval["loss"]) if do_log else float("nan")
            )
            history["val_mae_H"].append(
                float(val_eval["detailed"].get("mae", float("nan")))
                if do_log
                else float("nan")
            )

            # Best checkpoint by evaluation loss when available.
            score = float(val_eval["loss"]) if do_log else train_loss
            if score < best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": network.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": score,
                        "config": config,
                        "metrics": val_eval["detailed"],
                    },
                    best_model_path,
                )
                print(
                    f"[OK] Best model saved to {best_model_path.name} (score: {score:.6e})"
                )

            if do_log:
                current_time = time.time()
                time_elapsed = current_time - last_log_time
                epochs_since_last_log = (
                    (epoch - last_logged_epoch)
                    if last_logged_epoch >= 0
                    else (epoch + 1)
                )
                avg_epoch_time = time_elapsed / max(epochs_since_last_log, 1)
                last_log_time = current_time
                last_logged_epoch = epoch

                print(f"\n{'=' * 60}")
                print(f"EPOCH {epoch + 1}/{args.num_epochs}  |  lr={current_lr:.6e}")
                print(f"{'=' * 60}")
                log_detailed_training_metrics(
                    avg_epoch_time=avg_epoch_time,
                    epochs_since_last_log=epochs_since_last_log,
                    time_elapsed=time_elapsed,
                    loss_value=train_loss,
                    detailed_metrics=val_eval["detailed"],
                    irrep_losses=(
                        {
                            k: torch.tensor(v / max(len(train_samples), 1))
                            for k, v in train_irrep_losses_total.items()
                        }
                        if args.train_on_irrep_parts
                        else None
                    ),
                )
                if args.log_per_irrep_metrics and val_eval["per_irrep"]:
                    log_per_irrep_metrics(
                        "Validation Per-Irrep Metrics:",
                        all_irreps,
                        val_eval["per_irrep"],
                    )

                # frame from first eval sample
                if (
                    val_eval["first_pred_metrics"] is not None
                    and val_eval["first_sample"] is not None
                ):
                    fs = val_eval["first_sample"]
                    target_for_eval = (
                        fs["target_H_matrix_raw"]
                        if args.distance_magnitude_normalization
                        else fs["target_H_matrix"]
                    )
                    try:
                        save_hamiltonian_frame_to_disk(
                            val_eval["first_pred_metrics"],
                            target_for_eval,
                            fs["target_overlap"],
                            fs["atoms_list"],
                            mapper.orbital_cfg,
                            frame_output_dir,
                            epoch,
                            sx=0,
                            sy=0,
                            sz=0,
                            dynamic_range=False,
                            diff_dynamic_range=True,
                            partial_train=None,
                            percentile=99.0,
                            max_atoms=(
                                args.video_max_atoms
                                if args.video_max_atoms is not None
                                and args.video_max_atoms > 0
                                else None
                            ),
                        )
                    except Exception as exc:
                        print(f"[WARN] Could not save frame for epoch {epoch}: {exc}")

                t_wandb = time.perf_counter() if args.benchmark else 0.0
                wandb.log(
                    build_wandb_detailed_metrics_log(
                        epoch_zero_based=epoch,
                        loss_value=train_loss,
                        detailed_metrics=val_eval["detailed"],
                    )
                )
                if val_eval["per_irrep"]:
                    wandb.log(
                        build_wandb_per_irrep_metrics_log(
                            epoch_zero_based=epoch,
                            all_irreps=all_irreps,
                            per_irrep_metrics=val_eval["per_irrep"],
                        )
                    )
                wandb.log(
                    {
                        "train/loss_step": train_loss,
                        "val/loss": float(val_eval["loss"]),
                        "lr": current_lr,
                        "epoch": epoch,
                    }
                )
                benchmark_add(
                    "wandb_log",
                    time.perf_counter() - t_wandb if args.benchmark else 0.0,
                )

            if args.benchmark:
                benchmark_add("epoch_total", time.perf_counter() - epoch_t0)
                benchmark_epochs += 1
                if do_log:
                    benchmark_report()
                    benchmark_accum = {k: 0.0 for k in benchmark_order}
                    benchmark_epochs = 0

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user; running final evaluation.")

    print("")
    print("=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    # Final eval on validation split if available, otherwise train.
    final_eval = evaluate_split(
        network=network,
        samples=val_samples if len(val_samples) > 0 else train_samples,
        mapper=mapper,
        all_irreps=all_irreps,
        train_on_irrep_parts=args.train_on_irrep_parts,
        normalize_blocks=args.normalize_blocks,
        separate_shifted_self=args.separate_shifted_self,
        distance_magnitude_normalization=args.distance_magnitude_normalization,
        distance_norm_coeffs=distance_norm_coeffs,
    )

    final_detailed_metrics = (
        final_eval["detailed"]
        if final_eval["detailed"]
        else {
            "mae": 0.0,
            "mse": 0.0,
            "mae_mod": 0.0,
            "mse_mod": 0.0,
            "mu_H": 0.0,
            "correction_mae": 0.0,
            "correction_mse": 0.0,
        }
    )
    log_final_metrics(final_detailed_metrics)

    final_metrics = {f"final/{k}": float(v) for k, v in final_detailed_metrics.items()}

    # Representative sample diagnostics.
    if (
        final_eval["first_sample"] is not None
        and final_eval["first_pred_metrics"] is not None
    ):
        fs = final_eval["first_sample"]
        pred_metrics = final_eval["first_pred_metrics"]
        target_for_eval = (
            fs["target_H_matrix_raw"]
            if args.distance_magnitude_normalization
            else fs["target_H_matrix"]
        )

        # DOS/eigen diagnostics.
        dos_plot_path = run_checkpoint_dir / "dos_comparison_final.png"
        try:
            dos_metrics = save_dos_comparison_plot(
                H_pred=pred_metrics,
                H_gt=target_for_eval,
                S=fs["target_overlap"],
                output_path=dos_plot_path,
                sigma=0.2,
                bin_width=0.1,
                title="DOS Comparison (Representative)",
            )
            for k, v in dos_metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    final_metrics[f"final/{k}"] = float(v)
            wandb.log({"final/dos_comparison_plot": wandb.Image(str(dos_plot_path))})
            print(f"  DOS plot saved: {dos_plot_path}")
        except Exception as exc:
            print(f"[WARN] Could not generate DOS comparison plot: {exc}")

        # Distance curve.
        dist_curve = compute_distance_error_curve(
            H_pred=pred_metrics,
            H_gt=target_for_eval,
            positions=fs["positions"],
            box=fs["box"],
            partial_train=None,
            n_bins=64,
        )
        if dist_curve is not None:
            curve_json_path = run_checkpoint_dir / "distance_error_curve.json"
            curve_plot_path = run_checkpoint_dir / "distance_error_curve.png"
            with open(curve_json_path, "w", encoding="utf-8") as f:
                json.dump(dist_curve, f, indent=2)
            save_distance_error_curve_plot(
                dist_curve,
                curve_plot_path,
                title="Distance Error Curves (Final, representative)",
            )
            wandb.log({"distance_curve/plot": wandb.Image(str(curve_plot_path))})
            print(f"  Saved: {curve_json_path}")
            print(f"  Saved: {curve_plot_path}")

        if args.log_per_irrep_images:
            irrep_output_dir = run_checkpoint_dir / "per_irrep_images"
            irrep_output_dir.mkdir(parents=True, exist_ok=True)
            for irrep in all_irreps:
                ir_str = str(irrep)
                try:
                    pred_ir = split_hamiltonian_by_irrep(pred_metrics, mapper, ir_str)
                    targ_ir = split_hamiltonian_by_irrep(
                        target_for_eval, mapper, ir_str
                    )
                    visualize_hamiltonians(
                        pred_ir,
                        targ_ir,
                        fs["target_overlap"],
                        fs["atoms_list"],
                        mapper.orbital_cfg,
                        k_range=0,
                        output_dir=irrep_output_dir,
                        dynamic_range=True,
                        diff_dynamic_range=True,
                        per_panel_dynamic_range=True,
                        partial_train=None,
                        filename_prefix=f"hamiltonian_{ir_str}",
                        percentile=99.0,
                    )
                    img = irrep_output_dir / f"hamiltonian_{ir_str}_sx+0_sy+0_sz+0.png"
                    if img.exists():
                        wandb.log({f"irrep_images/{ir_str}": wandb.Image(str(img))})
                except Exception as exc:
                    print(f"[WARN] Irrep image failed for {ir_str}: {exc}")

    wandb.log(final_metrics)

    final_model_path = run_checkpoint_dir / "final_model.pt"
    torch.save(
        {
            "epoch": len(history["train_loss"]) - 1,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": (
                history["train_loss"][-1] if history["train_loss"] else float("nan")
            ),
            "config": config,
            "history": history,
            "final_metrics": final_detailed_metrics,
        },
        final_model_path,
    )

    log_study_complete(
        run_name=run_name,
        total_training_epochs=max(len(history["train_loss"]), 0),
        final_loss=history["train_loss"][-1] if history["train_loss"] else float("nan"),
        best_loss=best_score if best_score < float("inf") else float("nan"),
        best_epoch=(best_epoch + 1) if best_epoch >= 0 else -1,
        run_checkpoint_dir=run_checkpoint_dir,
        final_model_path=final_model_path,
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
            print("[OK] Training video logged to WandB")
        except Exception as exc:
            print(f"[WARN] Could not generate final video: {exc}")

    wandb.finish()


if __name__ == "__main__":
    main()
