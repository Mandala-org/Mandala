from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pytorch_lightning as pl
import imageio.v2 as imageio

from core.sparse_math import trace_matmul_sparse_block_matrix_aligned
from data.block_matrix import BlockMatrix, IrrepsBlockData
from net.irrep_tools import (
    compute_irrep_metrics,
    compute_hamiltonian_mae_contributions,
    filter_irreps_block_data_by_irrep,
    get_all_irreps,
)
from net.observable_metrics import (
    build_observable_predictions,
    build_observable_trace_alignment,
)
from net.run_logging import (
    MATRIX_ALIAS,
    IRREP_PREFIX_BY_MATRIX,
    build_wandb_detailed_metrics_log,
    build_wandb_per_irrep_metrics_log,
    log_detailed_training_metrics,
    log_final_metrics,
    log_per_irrep_metrics,
    log_strict_checks_passed,
    log_study_complete,
    should_log_epoch,
)


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _as_dense(matrix: BlockMatrix) -> torch.Tensor:
    dense = matrix.to_dense()
    return dense.detach().cpu().to(torch.float64)


def _compute_mu_h(H_pred: BlockMatrix, H_gt: BlockMatrix, S: BlockMatrix) -> float:
    numerator = 0.0
    denominator = 0.0
    for key in H_gt.pair_blocks.keys():
        if key not in H_pred.pair_blocks or key not in S.pair_blocks:
            continue
        pred_blocks = H_pred.pair_blocks[key]
        gt_blocks = H_gt.pair_blocks[key]
        s_blocks = S.pair_blocks[key]
        target_n = gt_blocks.shape[0]
        if pred_blocks.shape[0] < target_n or s_blocks.shape[0] < target_n:
            raise ValueError(
                f"Aligned mu_H calculation requires prefix length {target_n} for key '{key}', "
                f"got pred_len={pred_blocks.shape[0]} and s_len={s_blocks.shape[0]}"
            )
        diff = pred_blocks[:target_n] - gt_blocks[:target_n]
        s_val = s_blocks[:target_n]
        numerator += torch.sum(diff * s_val).item()
        denominator += torch.sum(s_val * s_val).item()
    return numerator / denominator if denominator > 1e-10 else 0.0


def _crop_dense_to_max_atoms(
    dense: torch.Tensor,
    atoms: tuple[str, ...],
    orbital_cfg,
    max_atoms: int | None,
) -> torch.Tensor:
    if max_atoms is None or int(max_atoms) <= 0:
        return dense
    atom_count = min(int(max_atoms), len(atoms))
    orbital_dims = [orbital_cfg.element_to_irreps[a].dim for a in atoms]
    cut_dim = int(sum(orbital_dims[:atom_count]))
    return dense[:cut_dim, :cut_dim]


def _project_overlap_to_psd(
    overlap_dense: torch.Tensor,
    *,
    eig_floor: float = 1.0e-6,
) -> torch.Tensor:
    overlap_dense = 0.5 * (overlap_dense + overlap_dense.T)
    evals, evecs = torch.linalg.eigh(overlap_dense)
    evals_real = evals.real
    min_eval = float(torch.min(evals_real).item())
    max_eval = float(torch.max(evals_real).item())
    if min_eval < eig_floor or not bool(torch.isfinite(evals_real).all()):
        print(
            f"[OVERLAP] PSD projection needed: shape={tuple(overlap_dense.shape)} "
            f"eig_min={min_eval:.6e} eig_max={max_eval:.6e} floor={eig_floor:.1e}"
        )
    finite = torch.isfinite(evals_real)
    if not bool(finite.all()):
        n_bad = int((~finite).sum().item())
        print(f"[OVERLAP] Replacing {n_bad} non-finite overlap eigenvalues with zero")
        evals_real = torch.where(finite, evals_real, torch.zeros_like(evals_real))
    positive = evals_real[evals_real > eig_floor]
    if positive.numel() > 0:
        upper = float(torch.quantile(positive, 0.995).item()) * 10.0
        upper = max(upper, eig_floor)
    else:
        upper = eig_floor * 10.0
    evals_real = torch.clamp(evals_real, min=eig_floor, max=upper)
    overlap_clean = evecs @ torch.diag(evals_real.to(dtype=evecs.dtype)) @ evecs.T
    return 0.5 * (overlap_clean + overlap_clean.T)


def compute_basic_matrix_metrics_aligned(
    pred: BlockMatrix,
    target: BlockMatrix,
) -> dict[str, float]:
    mae = 0.0
    mse = 0.0
    total_elements = 0

    for key in target.pair_blocks.keys():
        pred_blocks = pred.pair_blocks[key]
        gt_blocks = target.pair_blocks[key]
        if pred_blocks.shape != gt_blocks.shape:
            raise ValueError(
                f"Aligned basic metrics require matching block shapes for key '{key}'"
            )
        diff = pred_blocks - gt_blocks
        mae += torch.sum(torch.abs(diff)).item()
        mse += torch.sum(diff**2).item()
        total_elements += diff.numel()

    if total_elements <= 0:
        return {"mae": 0.0, "mse": 0.0}
    return {"mae": mae / total_elements, "mse": mse / total_elements}


def compute_detailed_metrics_aligned(
    H_pred: BlockMatrix,
    H_gt: BlockMatrix,
    S: BlockMatrix,
) -> dict[str, float]:
    mae = 0.0
    mse = 0.0
    total_elements = 0

    for key in H_gt.pair_blocks.keys():
        pred_blocks = H_pred.pair_blocks[key]
        gt_blocks = H_gt.pair_blocks[key]
        if pred_blocks.shape != gt_blocks.shape:
            raise ValueError(
                f"Aligned detailed metrics require matching block shapes for key '{key}'"
            )
        diff = pred_blocks - gt_blocks
        mae += torch.sum(torch.abs(diff)).item()
        mse += torch.sum(diff**2).item()
        total_elements += diff.numel()

    if total_elements <= 0:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "mae_mod": 0.0,
            "mse_mod": 0.0,
            "mu_H": 0.0,
            "correction_mae": 0.0,
            "correction_mse": 0.0,
        }

    mae /= total_elements
    mse /= total_elements
    mu_h = _compute_mu_h(H_pred, H_gt, S)

    mae_mod = 0.0
    mse_mod = 0.0
    correction_mae = 0.0
    correction_mse = 0.0
    for key in H_gt.pair_blocks.keys():
        pred_blocks = H_pred.pair_blocks[key]
        gt_blocks = H_gt.pair_blocks[key]
        s_blocks = S.pair_blocks[key]
        correction = mu_h * s_blocks
        diff_corrected = pred_blocks - gt_blocks - correction
        mae_mod += torch.sum(torch.abs(diff_corrected)).item()
        mse_mod += torch.sum(diff_corrected**2).item()
        correction_mae += torch.sum(torch.abs(correction)).item()
        correction_mse += torch.sum(correction**2).item()

    return {
        "mae": mae,
        "mse": mse,
        "mae_mod": mae_mod / total_elements,
        "mse_mod": mse_mod / total_elements,
        "mu_H": mu_h,
        "correction_mae": correction_mae / total_elements,
        "correction_mse": correction_mse / total_elements,
    }


def align_pred_block_matrix_to_target_edges(
    pred_matrix: BlockMatrix,
    target_matrix: BlockMatrix,
    *,
    require_exact_prefix: bool = False,
) -> BlockMatrix:
    pair_blocks: dict[str, torch.Tensor] = {}
    pair_edges: dict[str, torch.Tensor] = {}
    lookup: dict[tuple[int, int, int, int, int], tuple[str, int]] = {}

    for key, target_edges in target_matrix.pair_edges.items():
        if key not in pred_matrix.pair_blocks:
            raise ValueError(f"Prediction is missing key '{key}' required by target")
        pred_blocks = pred_matrix.pair_blocks[key]
        target_n = target_edges.shape[1]
        if pred_blocks.shape[0] < target_n:
            raise ValueError(
                f"Prediction is missing the target prefix for key '{key}': "
                f"pred_len={pred_blocks.shape[0]} target_len={target_n}"
            )
        if require_exact_prefix:
            pred_edges = pred_matrix.pair_edges.get(key)
            if pred_edges is None:
                raise ValueError(f"Prediction is missing edge tensor for key '{key}'")
            if pred_edges.shape[1] < target_n:
                raise ValueError(
                    f"Prediction edge tensor for key '{key}' is too short: "
                    f"pred_len={pred_edges.shape[1]} target_len={target_n}"
                )
            if not torch.equal(pred_edges[:, :target_n], target_edges):
                raise ValueError(
                    f"Prediction prefix edge mismatch for key '{key}' while aligning metrics"
                )
        pair_blocks[key] = (
            pred_blocks[:target_n]
            if target_n > 0
            else pred_blocks.new_zeros((0, *pred_blocks.shape[1:]))
        )
        pair_edges[key] = target_edges
        for idx, (sx, sy, sz, i, j) in enumerate(target_edges.t().tolist()):
            lookup[(sx, sy, sz, i, j)] = (key, idx)

    return BlockMatrix(
        atoms=pred_matrix.atoms,
        atom_counts=pred_matrix.atom_counts,
        pair_blocks=pair_blocks,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=pred_matrix.orbital_cfg,
        basis=pred_matrix.basis,
    )


def align_pred_irreps_to_target_edges(
    pred_irreps: IrrepsBlockData,
    target_irreps: IrrepsBlockData,
    *,
    require_exact_prefix: bool = False,
) -> IrrepsBlockData:
    pair_vectors: dict[str, torch.Tensor] = {}
    pair_edges: dict[str, torch.Tensor] = {}
    lookup: dict[tuple[int, int, int, int, int], tuple[str, int]] = {}

    for key, target_edges in target_irreps.pair_edges.items():
        if key not in pred_irreps.pair_vectors:
            raise ValueError(
                f"Prediction is missing irrep key '{key}' required by target"
            )
        pred_vectors = pred_irreps.pair_vectors[key]
        target_n = target_edges.shape[1]
        if pred_vectors.shape[0] < target_n:
            raise ValueError(
                f"Prediction is missing the target irrep prefix for key '{key}': "
                f"pred_len={pred_vectors.shape[0]} target_len={target_n}"
            )
        if require_exact_prefix:
            pred_edges = pred_irreps.pair_edges.get(key)
            if pred_edges is None:
                raise ValueError(
                    f"Prediction is missing edge tensor for irrep key '{key}'"
                )
            if pred_edges.shape[1] < target_n:
                raise ValueError(
                    f"Prediction edge tensor for irrep key '{key}' is too short: "
                    f"pred_len={pred_edges.shape[1]} target_len={target_n}"
                )
            if not torch.equal(pred_edges[:, :target_n], target_edges):
                raise ValueError(
                    f"Prediction prefix edge mismatch for irrep key '{key}' while aligning metrics"
                )
        pair_vectors[key] = (
            pred_vectors[:target_n]
            if target_n > 0
            else pred_vectors.new_zeros((0, pred_vectors.shape[-1]))
        )
        pair_edges[key] = target_edges
        for idx, (sx, sy, sz, i, j) in enumerate(target_edges.t().tolist()):
            lookup[(sx, sy, sz, i, j)] = (key, idx)

    return IrrepsBlockData(
        atoms=pred_irreps.atoms,
        atom_counts=pred_irreps.atom_counts,
        pair_vectors=pair_vectors,
        pair_edges=pair_edges,
        lookup=lookup,
        orbital_cfg=pred_irreps.orbital_cfg,
        basis=pred_irreps.basis,
    )


def compute_generalized_eigenvalues(
    H: BlockMatrix,
    S: BlockMatrix,
    *,
    psd_cleanup: bool = False,
    allow_jitter: bool = False,
) -> torch.Tensor:
    H_dense = _as_dense(H)
    S_dense = _as_dense(S)
    H_dense = 0.5 * (H_dense + H_dense.T)
    if psd_cleanup:
        S_dense = _project_overlap_to_psd(S_dense)
    n = S_dense.shape[-1]
    eye = torch.eye(n, dtype=S_dense.dtype, device=S_dense.device)
    jitters = (
        (0.0, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
        if allow_jitter
        else (0.0,)
    )
    for jitter in jitters:
        try:
            S_reg = S_dense if jitter == 0.0 else S_dense + jitter * eye
            if jitter > 0.0:
                print(f"[OVERLAP] Retrying Cholesky with jitter={jitter:.1e}")
            L = torch.linalg.cholesky(S_reg)
            tmp = torch.linalg.solve(L, H_dense)
            A = torch.linalg.solve(L, tmp.T).T
            A = 0.5 * (A + A.T)
            return torch.linalg.eigvalsh(A)
        except torch.linalg.LinAlgError:
            print(f"[OVERLAP] Cholesky failed at jitter={jitter:.1e}")
            continue

    raise torch.linalg.LinAlgError(
        "Generalized eigensolve failed: overlap Cholesky did not succeed"
        + (" even after jitter retries." if allow_jitter else ".")
    )


def compute_dos_from_eigenvalues(
    eigenvalues: torch.Tensor,
    sigma: float = 0.2,
    bin_width: float = 0.1,
    e_min: float | None = None,
    e_max: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if e_min is None or e_max is None:
        eig_min = float(torch.min(eigenvalues).item())
        eig_max = float(torch.max(eigenvalues).item())
        span = max(eig_max - eig_min, 1e-6)
        margin = 0.1 * span + 0.05
        e_min = eig_min - margin
        e_max = eig_max + margin

    grid = torch.arange(
        e_min,
        e_max + bin_width,
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


def save_dos_comparison_plot(
    H_pred: BlockMatrix,
    H_gt: BlockMatrix,
    S: BlockMatrix,
    output_path: Path | str,
    *,
    sigma: float = 0.2,
    bin_width: float = 0.1,
    title: str = "DOS Comparison",
) -> dict[str, float]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eig_pred = compute_generalized_eigenvalues(H_pred, S)
    eig_gt = compute_generalized_eigenvalues(H_gt, S)
    abs_err = torch.abs(eig_pred - eig_gt)
    rel_err = abs_err / (torch.abs(eig_gt) + 1e-12)

    eig_min = float(torch.min(torch.min(eig_pred), torch.min(eig_gt)).item())
    eig_max = float(torch.max(torch.max(eig_pred), torch.max(eig_gt)).item())
    span = max(eig_max - eig_min, 1e-6)
    margin = 0.1 * span + 0.05
    e_min = eig_min - margin
    e_max = eig_max + margin

    grid, dos_pred = compute_dos_from_eigenvalues(
        eig_pred, sigma=sigma, bin_width=bin_width, e_min=e_min, e_max=e_max
    )
    _, dos_gt = compute_dos_from_eigenvalues(
        eig_gt, sigma=sigma, bin_width=bin_width, e_min=e_min, e_max=e_max
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].plot(eig_gt.cpu().numpy(), label="GT", lw=1.5)
    axes[0].plot(eig_pred.cpu().numpy(), label="Pred", lw=1.2)
    axes[0].set_title("Generalized eigenvalues")
    axes[0].legend()
    axes[1].plot(grid.cpu().numpy(), dos_gt.cpu().numpy(), label="GT", lw=1.5)
    axes[1].plot(grid.cpu().numpy(), dos_pred.cpu().numpy(), label="Pred", lw=1.2)
    axes[1].set_title("DOS")
    axes[1].legend()
    axes[2].plot(abs_err.cpu().numpy())
    axes[2].set_title("Abs eig error")
    axes[3].plot(rel_err.cpu().numpy())
    axes[3].set_title("Rel eig error")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle(title)
    _save_plot(fig, output_path)

    return {
        "eig_abs_mean": float(abs_err.mean().item()),
        "eig_abs_max": float(abs_err.max().item()),
        "eig_rel_mean": float(rel_err.mean().item()),
        "eig_rel_max": float(rel_err.max().item()),
    }


def _filter_blocks_by_partial_train(
    block_matrix: BlockMatrix, partial_train: str | None
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if partial_train is None:
        return {
            key: (
                blocks,
                torch.ones(blocks.shape[0], dtype=torch.bool, device=blocks.device),
            )
            for key, blocks in block_matrix.pair_blocks.items()
        }

    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, blocks in block_matrix.pair_blocks.items():
        edges = block_matrix.pair_edges[key]
        sx, sy, sz, i, j = edges
        is_diag = (i == j) & (sx == 0) & (sy == 0) & (sz == 0)
        is_shifted_self = (i == j) & ~((sx == 0) & (sy == 0) & (sz == 0))
        if partial_train == "diag":
            mask = is_diag
        elif partial_train == "shifted_self":
            mask = is_shifted_self
        elif partial_train == "offdiag":
            mask = i != j
        else:
            raise ValueError(f"Unsupported partial_train={partial_train!r}")
        out[key] = (blocks, mask)
    return out


def compute_distance_error_curve(
    H_pred: BlockMatrix,
    H_gt: BlockMatrix,
    positions: torch.Tensor,
    box: torch.Tensor | None = None,
    partial_train: str | None = None,
    n_bins: int = 16,
) -> dict[str, Any] | None:
    if positions is None:
        raise ValueError("positions must be provided for distance-binned analysis")

    filtered_pred = _filter_blocks_by_partial_train(H_pred, partial_train)
    filtered_gt = _filter_blocks_by_partial_train(H_gt, partial_train)

    dists = []
    abs_l1_sum = []
    abs_l2_sum = []
    gt_l1_sum = []
    gt_l2_sum = []
    elem_count = []
    edge_l1_abs = []
    edge_l2_abs = []
    edge_l1_rel = []
    edge_l2_rel = []

    for key in H_gt.pair_blocks.keys():
        if key not in H_pred.pair_blocks:
            continue
        pred_blocks_full = H_pred.pair_blocks[key]
        gt_blocks_full = H_gt.pair_blocks[key]
        gt_edges_full = H_gt.pair_edges[key]
        _, pred_mask = filtered_pred[key]
        _, gt_mask = filtered_gt[key]

        pred_edge_to_idx = {
            tuple(map(int, edge.tolist())): idx
            for idx, edge in enumerate(H_pred.pair_edges[key].t())
        }

        for gt_idx, edge in enumerate(gt_edges_full.t()):
            if not bool(gt_mask[gt_idx]):
                continue
            edge_key = tuple(map(int, edge.tolist()))
            pred_idx = pred_edge_to_idx.get(edge_key)
            if pred_idx is None:
                continue
            if (
                pred_idx >= pred_blocks_full.shape[0]
                or gt_idx >= gt_blocks_full.shape[0]
            ):
                continue
            if pred_idx < pred_mask.shape[0] and not bool(pred_mask[pred_idx]):
                continue

            pred_block = pred_blocks_full[pred_idx]
            gt_block = gt_blocks_full[gt_idx]
            diff = pred_block - gt_block

            sx, sy, sz, i, j = edge_key
            shift = torch.tensor(
                [sx, sy, sz], dtype=positions.dtype, device=positions.device
            )
            if box is not None:
                disp = positions[j] - positions[i] + shift @ box
            else:
                disp = positions[j] - positions[i]
            dist = torch.linalg.norm(disp).item()

            dists.append(dist)
            abs_l1_sum.append(torch.abs(diff).sum().item())
            abs_l2_sum.append((diff**2).sum().item())
            gt_l1_sum.append(torch.abs(gt_block).sum().item())
            gt_l2_sum.append((gt_block**2).sum().item())
            elem_count.append(diff.numel())

            abs_sum = torch.abs(diff).sum().item()
            sq_sum = (diff**2).sum().item()
            gt_abs_sum = torch.abs(gt_block).sum().item()
            gt_sq_sum = (gt_block**2).sum().item()
            n_el = diff.numel()
            edge_l1_abs.append(abs_sum / max(n_el, 1))
            edge_l2_abs.append((sq_sum / max(n_el, 1)) ** 0.5)
            edge_l1_rel.append(abs_sum / max(gt_abs_sum, 1e-14))
            edge_l2_rel.append((sq_sum / max(gt_sq_sum, 1e-14)) ** 0.5)

    if not dists:
        return None

    dists_t = torch.tensor(dists, dtype=torch.float64)
    d_min = float(dists_t.min().item())
    d_max = float(dists_t.max().item())
    if abs(d_max - d_min) < 1e-12:
        d_max = d_min + 1e-6
    bin_edges = torch.linspace(d_min, d_max, n_bins + 1, dtype=torch.float64)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = torch.bucketize(dists_t, bin_edges[1:], right=False).clamp(max=n_bins - 1)

    l1_abs = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_abs = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_rel = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_rel = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_abs_min = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_abs_max = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_abs_min = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_abs_max = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_rel_min = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l1_rel_max = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_rel_min = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    l2_rel_max = torch.full((n_bins,), float("nan"), dtype=torch.float64)
    n_edges_per_bin = torch.zeros((n_bins,), dtype=torch.int64)
    n_elems_per_bin = torch.zeros((n_bins,), dtype=torch.int64)

    abs_l1_sum_t = torch.tensor(abs_l1_sum, dtype=torch.float64)
    abs_l2_sum_t = torch.tensor(abs_l2_sum, dtype=torch.float64)
    gt_l1_sum_t = torch.tensor(gt_l1_sum, dtype=torch.float64)
    gt_l2_sum_t = torch.tensor(gt_l2_sum, dtype=torch.float64)
    elem_count_t = torch.tensor(elem_count, dtype=torch.int64)
    edge_l1_abs_t = torch.tensor(edge_l1_abs, dtype=torch.float64)
    edge_l2_abs_t = torch.tensor(edge_l2_abs, dtype=torch.float64)
    edge_l1_rel_t = torch.tensor(edge_l1_rel, dtype=torch.float64)
    edge_l2_rel_t = torch.tensor(edge_l2_rel, dtype=torch.float64)

    for b in range(n_bins):
        mask = bin_idx == b
        if not torch.any(mask):
            continue
        sum_abs_l1 = abs_l1_sum_t[mask].sum()
        sum_abs_l2 = abs_l2_sum_t[mask].sum()
        sum_gt_l1 = gt_l1_sum_t[mask].sum()
        sum_gt_l2 = gt_l2_sum_t[mask].sum()
        sum_elems = elem_count_t[mask].sum()
        sum_edges = mask.sum()
        n_edges_per_bin[b] = sum_edges
        n_elems_per_bin[b] = sum_elems
        if sum_elems > 0:
            l1_abs[b] = sum_abs_l1 / sum_elems
            l2_abs[b] = torch.sqrt(sum_abs_l2 / sum_elems)
        if sum_gt_l1 > 1e-14:
            l1_rel[b] = sum_abs_l1 / sum_gt_l1
        if sum_gt_l2 > 1e-14:
            l2_rel[b] = torch.sqrt(sum_abs_l2 / sum_gt_l2)
        l1_abs_min[b] = edge_l1_abs_t[mask].min()
        l1_abs_max[b] = edge_l1_abs_t[mask].max()
        l2_abs_min[b] = edge_l2_abs_t[mask].min()
        l2_abs_max[b] = edge_l2_abs_t[mask].max()
        l1_rel_min[b] = edge_l1_rel_t[mask].min()
        l1_rel_max[b] = edge_l1_rel_t[mask].max()
        l2_rel_min[b] = edge_l2_rel_t[mask].min()
        l2_rel_max[b] = edge_l2_rel_t[mask].max()

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_centers": bin_centers.tolist(),
        "l1_abs": l1_abs.tolist(),
        "l2_abs": l2_abs.tolist(),
        "l1_rel": l1_rel.tolist(),
        "l2_rel": l2_rel.tolist(),
        "l1_abs_min": l1_abs_min.tolist(),
        "l1_abs_max": l1_abs_max.tolist(),
        "l2_abs_min": l2_abs_min.tolist(),
        "l2_abs_max": l2_abs_max.tolist(),
        "l1_rel_min": l1_rel_min.tolist(),
        "l1_rel_max": l1_rel_max.tolist(),
        "l2_rel_min": l2_rel_min.tolist(),
        "l2_rel_max": l2_rel_max.tolist(),
        "n_edges": n_edges_per_bin.tolist(),
        "n_elements": n_elems_per_bin.tolist(),
        "d_min": d_min,
        "d_max": d_max,
        "n_bins": n_bins,
    }


def save_distance_error_curve_plot(
    curve_data: dict[str, Any],
    output_path: Path | str,
    title: str | None = None,
    relative_log_scale: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.array(curve_data["bin_centers"], dtype=float)

    def _interp_nans(x_vals: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
        y = y_vals.astype(float).copy()
        valid = ~np.isnan(y)
        if valid.sum() == 0:
            return y
        if valid.sum() == 1:
            y[~valid] = y[valid][0]
            return y
        y[~valid] = np.interp(x_vals[~valid], x_vals[valid], y[valid])
        return y

    def _interp_nans_log(
        x_vals: np.ndarray, y_vals: np.ndarray, eps: float = 1e-16
    ) -> np.ndarray:
        y = y_vals.astype(float).copy()
        valid = np.isfinite(y) & (y > 0.0)
        if valid.sum() == 0:
            return y
        if valid.sum() == 1:
            y[~np.isfinite(y)] = y[valid][0]
            return y
        y_log = np.log10(y[valid])
        missing = ~np.isfinite(y)
        y_interp_log = np.interp(x_vals[missing], x_vals[valid], y_log)
        y[missing] = np.power(10.0, y_interp_log)
        y[(~np.isfinite(y)) | (y <= 0.0)] = eps
        return y

    def _make_log_safe(y_vals: np.ndarray, eps: float = 1e-16) -> np.ndarray:
        y = y_vals.astype(float).copy()
        finite = np.isfinite(y)
        nonpos = finite & (y <= 0.0)
        y[nonpos] = eps
        return y

    l1_abs = _interp_nans(x, np.array(curve_data["l1_abs"], dtype=float))
    l2_abs = _interp_nans(x, np.array(curve_data["l2_abs"], dtype=float))
    l1_rel = _interp_nans_log(x, np.array(curve_data["l1_rel"], dtype=float))
    l2_rel = _interp_nans_log(x, np.array(curve_data["l2_rel"], dtype=float))
    l1_abs_min = _interp_nans(x, np.array(curve_data["l1_abs_min"], dtype=float))
    l1_abs_max = _interp_nans(x, np.array(curve_data["l1_abs_max"], dtype=float))
    l2_abs_min = _interp_nans(x, np.array(curve_data["l2_abs_min"], dtype=float))
    l2_abs_max = _interp_nans(x, np.array(curve_data["l2_abs_max"], dtype=float))
    l1_rel_min = _interp_nans_log(x, np.array(curve_data["l1_rel_min"], dtype=float))
    l1_rel_max = _interp_nans_log(x, np.array(curve_data["l1_rel_max"], dtype=float))
    l2_rel_min = _interp_nans_log(x, np.array(curve_data["l2_rel_min"], dtype=float))
    l2_rel_max = _interp_nans_log(x, np.array(curve_data["l2_rel_max"], dtype=float))
    if relative_log_scale:
        l1_rel = _make_log_safe(l1_rel)
        l2_rel = _make_log_safe(l2_rel)
        l1_rel_min = _make_log_safe(l1_rel_min)
        l1_rel_max = _make_log_safe(l1_rel_max)
        l2_rel_min = _make_log_safe(l2_rel_min)
        l2_rel_max = _make_log_safe(l2_rel_max)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        title or f"Distance Error Curves ({curve_data['n_bins']} bins)", fontsize=14
    )
    panels = [
        (axes[0, 0], l1_abs, l1_abs_min, l1_abs_max, "Absolute L1"),
        (axes[0, 1], l2_abs, l2_abs_min, l2_abs_max, "Absolute L2 (RMSE)"),
        (axes[1, 0], l1_rel, l1_rel_min, l1_rel_max, "Relative L1"),
        (axes[1, 1], l2_rel, l2_rel_min, l2_rel_max, "Relative L2"),
    ]
    for ax, y, y_min, y_max, ttl in panels:
        ax.plot(x, y, marker="o", linewidth=1.2, markersize=2.2, label="mean")
        ax.plot(x, y_min, linewidth=0.9, linestyle="--", alpha=0.8, label="min")
        ax.plot(x, y_max, linewidth=0.9, linestyle="--", alpha=0.8, label="max")
        ax.fill_between(x, y_min, y_max, alpha=0.15)
        ax.set_title(ttl)
        ax.set_xlabel("Distance (A)")
        ax.set_ylabel("Error")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    if relative_log_scale:
        axes[1, 0].set_yscale("log")
        axes[1, 1].set_yscale("log")
    _save_plot(fig, output_path)


def save_matrix_comparison_plot(
    pred: BlockMatrix,
    target: BlockMatrix,
    output_path: Path | str,
    *,
    reference: BlockMatrix | None = None,
    title: str = "",
    max_atoms: int | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_dense = _crop_dense_to_max_atoms(
        _as_dense(pred), pred.atoms, pred.orbital_cfg, max_atoms
    )
    target_dense = _crop_dense_to_max_atoms(
        _as_dense(target), target.atoms, target.orbital_cfg, max_atoms
    )
    diff = pred_dense - target_dense
    if reference is not None:
        ref_dense = _crop_dense_to_max_atoms(
            _as_dense(reference), reference.atoms, reference.orbital_cfg, max_atoms
        )
        mu = float(
            torch.sum((pred_dense - target_dense) * ref_dense).item()
            / max(float(torch.sum(ref_dense * ref_dense).item()), 1e-12)
        )
        diff_corr = diff - mu * ref_dense
    else:
        diff_corr = diff.abs()

    vmax = max(
        float(torch.quantile(torch.abs(target_dense.flatten()), 0.8).item()),
        float(torch.quantile(torch.abs(pred_dense.flatten()), 0.8).item()),
        1e-12,
    )
    dmax = max(
        float(torch.quantile(torch.abs(diff.flatten()), 0.8).item()),
        float(torch.quantile(torch.abs(diff_corr.flatten()), 0.8).item()),
        1e-12,
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    panels = [
        (target_dense, "GT", vmax),
        (pred_dense, "Pred", vmax),
        (diff, "Diff", dmax),
        (diff_corr, "Corr diff" if reference is not None else "Abs diff", dmax),
    ]
    for ax, (mat, label, lim) in zip(axes.ravel(), panels):
        im = ax.imshow(mat.cpu().numpy(), cmap="bwr", vmin=-lim, vmax=lim)
        ax.set_title(label)
        plt.colorbar(im, ax=ax)
    if title:
        fig.suptitle(title)
    _save_plot(fig, output_path)


def compile_frames_to_gif(
    frame_paths: Iterable[Path], output_path: Path | str, fps: int = 5
) -> None:
    output_path = Path(output_path)
    frames = [Image.open(path).convert("RGB") for path in sorted(frame_paths)]
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(1000 / max(fps, 1)), 1)
    first, rest = frames[0], frames[1:]
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )


def compile_frames_to_video(
    frame_paths: Iterable[Path],
    output_path: Path | str,
    fps: int = 5,
    format: str = "mp4",
) -> None:
    output_path = Path(output_path)
    frames = [imageio.imread(str(path)) for path in sorted(frame_paths)]
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, format=format)


def _get_logger_run(trainer: Any):
    logger = getattr(trainer, "logger", None)
    if logger is None:
        return None
    return getattr(logger, "experiment", None)


def _maybe_log_wandb(run: Any, payload: dict[str, Any]) -> None:
    if run is None:
        return
    try:
        run.log(payload)
    except Exception:
        pass


@dataclass
class ArtifactCheckpointState:
    best_score: float | None = None
    best_epoch: int | None = None


@dataclass
class RevertOnSpikeState:
    best_score: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    revert_count: int = 0


class RevertOnSpikeCallback(pl.Callback):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        monitor: str = "val/loss_total",
        mode: str = "min",
        patience: int = 20,
        decay_rate: float = 0.8,
        spike_factor: float = 2.0,
        restore_optimizer_state: bool = True,
        restore_lr_schedulers: bool = True,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if decay_rate <= 0.0:
            raise ValueError("decay_rate must be > 0")
        if spike_factor <= 1.0:
            raise ValueError("spike_factor must be > 1")
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.decay_rate = float(decay_rate)
        self.spike_factor = float(spike_factor)
        self.restore_optimizer_state = restore_optimizer_state
        self.restore_lr_schedulers = restore_lr_schedulers
        self.best_path = self.output_dir / "best_model.pt"
        self.state = RevertOnSpikeState()

    @property
    def state_key(self) -> str:
        return (
            f"{self.__class__.__qualname__}"
            f"[monitor={self.monitor},best_path={self.best_path}]"
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.state.best_score,
            "best_epoch": self.state.best_epoch,
            "bad_epochs": self.state.bad_epochs,
            "revert_count": self.state.revert_count,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.state = RevertOnSpikeState(
            best_score=state_dict.get("best_score"),
            best_epoch=state_dict.get("best_epoch"),
            bad_epochs=int(state_dict.get("bad_epochs", 0)),
            revert_count=int(state_dict.get("revert_count", 0)),
        )

    def _metric_to_float(self, metric: Any) -> float:
        if torch.is_tensor(metric):
            return float(metric.detach().cpu().item())
        return float(metric)

    def _is_better(self, score: float) -> bool:
        if self.state.best_score is None:
            return True
        if self.mode == "min":
            return score < self.state.best_score
        if self.mode == "max":
            return score > self.state.best_score
        raise ValueError(f"Unsupported mode={self.mode!r}")

    def _is_spike(self, score: float) -> bool:
        if self.state.best_score is None:
            return False
        best_score = float(self.state.best_score)
        if self.mode == "min":
            return score > self.spike_factor * max(best_score, 1e-12)
        if self.mode == "max":
            return score * self.spike_factor < best_score
        raise ValueError(f"Unsupported mode={self.mode!r}")

    def _save_best_checkpoint(self, trainer: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.best_path))

    def _restore_best_checkpoint(self, trainer: Any) -> bool:
        if not self.best_path.exists():
            print(
                f"--- RevertOnSpike: best checkpoint not found at {self.best_path}; "
                "skipping revert. ---"
            )
            return False

        checkpoint = trainer.strategy.load_checkpoint(str(self.best_path), False)
        trainer.strategy.load_model_state_dict(checkpoint, strict=True)
        if self.restore_optimizer_state and "optimizer_states" in checkpoint:
            trainer.strategy.load_optimizer_state_dict(checkpoint)
        if self.restore_lr_schedulers and "lr_schedulers" in checkpoint:
            for config, state in zip(
                getattr(trainer, "lr_scheduler_configs", []),
                checkpoint["lr_schedulers"],
            ):
                config.scheduler.load_state_dict(state)

        print(
            "--- RevertOnSpike: restored "
            f"{self.best_path.name} from epoch {self.state.best_epoch} "
            f"(best {self.monitor}={self.state.best_score:.6g}). ---"
        )
        return True

    def _decay_learning_rates(self, trainer: Any) -> None:
        for optimizer_idx, optimizer in enumerate(getattr(trainer, "optimizers", [])):
            for group_idx, param_group in enumerate(optimizer.param_groups):
                old_lr = float(param_group["lr"])
                new_lr = old_lr * self.decay_rate
                param_group["lr"] = new_lr
                print(
                    "--- RevertOnSpike: decayed "
                    f"optimizer {optimizer_idx} group {group_idx} lr "
                    f"from {old_lr:.6e} to {new_lr:.6e}. ---"
                )

        for config in getattr(trainer, "lr_scheduler_configs", []):
            scheduler = config.scheduler
            if hasattr(scheduler, "cooldown") and hasattr(
                scheduler, "cooldown_counter"
            ):
                scheduler.cooldown_counter = scheduler.cooldown
            if hasattr(scheduler, "_last_lr") and hasattr(scheduler, "optimizer"):
                scheduler._last_lr = [
                    group["lr"] for group in scheduler.optimizer.param_groups
                ]

    def on_fit_start(self, trainer, pl_module) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return

        score = self._metric_to_float(metric)
        if self._is_better(score):
            self.state.best_score = score
            self.state.best_epoch = int(trainer.current_epoch)
            self.state.bad_epochs = 0
            self._save_best_checkpoint(trainer)
            return

        if self._is_spike(score):
            self.state.bad_epochs += 1
        else:
            self.state.bad_epochs = 0

        if self.state.bad_epochs < self.patience:
            return

        print(
            "--- RevertOnSpike: "
            f"{self.monitor}={score:.6g} exceeded {self.spike_factor:.3g}x "
            f"the best score {self.state.best_score:.6g} for "
            f"{self.patience} consecutive validation epochs. ---"
        )
        reverted = self._restore_best_checkpoint(trainer)
        if reverted:
            self._decay_learning_rates(trainer)
            self.state.revert_count += 1
        self.state.bad_epochs = 0


class ArtifactCheckpointCallback(pl.Callback):
    def __init__(
        self,
        output_dir: str | Path,
        *,
        monitor: str = "val/loss_total",
        mode: str = "min",
        save_latest: bool = True,
        save_best: bool = True,
        save_final: bool = True,
        generate_video: bool = False,
        log_per_irrep_images: bool = True,
        distance_bins: int = 16,
        video_fps: int = 5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_latest = save_latest
        self.save_best = save_best
        self.save_final = save_final
        self.generate_video = generate_video
        self.log_per_irrep_images = log_per_irrep_images
        self.distance_bins = distance_bins
        self.video_fps = video_fps
        self.state = ArtifactCheckpointState()
        self.reference_batch = None
        self.frames_dir = self.output_dir / "frames"
        self.per_irrep_dir = self.output_dir / "per_irrep_images"
        self.latest_path = self.output_dir / "latest_checkpoint.pt"
        self.best_path = self.output_dir / "best_model.pt"
        self.final_path = self.output_dir / "final_model.pt"
        self._last_logged_epoch = -1
        self._printed_strict_checks = False

    def _is_better(self, score: float) -> bool:
        if self.state.best_score is None:
            return True
        if self.mode == "min":
            return score < self.state.best_score
        if self.mode == "max":
            return score > self.state.best_score
        raise ValueError(f"Unsupported mode={self.mode!r}")

    def _get_reference_batch(self, trainer: Any):
        loaders = getattr(trainer, "val_dataloaders", None)
        if loaders is None:
            loaders = getattr(trainer, "train_dataloader", None)
            if callable(loaders):
                loaders = loaders()
        if loaders is None:
            return None
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        for loader in loaders:
            try:
                return next(iter(loader))
            except Exception:
                continue
        return None

    def _predict(self, pl_module, batch):
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
        batch = pl_module.on_after_batch_transfer(batch, 0)
        pl_module.eval()
        x, y = batch
        enable_force_eval = bool(
            getattr(pl_module.cfg, "enable_forces", False)
            and y.get("forces") is not None
        )
        if enable_force_eval:
            x = dict(x)
            x["positions"] = x["positions"].detach().clone().requires_grad_(True)
            batch = (x, y)
        with torch.set_grad_enabled(enable_force_eval):
            x, y = batch
            preds = pl_module(x)
        return x, y, preds

    def _iter_eval_batches(self, trainer: Any):
        loaders = getattr(trainer, "val_dataloaders", None)
        if loaders is None:
            loaders = getattr(trainer, "train_dataloader", None)
            if callable(loaders):
                loaders = loaders()
        if loaders is None:
            return
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        for loader in loaders:
            for batch in loader:
                yield batch

    @staticmethod
    def _as_block_matrix(obj, mapper) -> BlockMatrix:
        if isinstance(obj, BlockMatrix):
            return obj
        if isinstance(obj, IrrepsBlockData):
            return obj.to_blocks(mapper)
        raise TypeError(f"Expected BlockMatrix or IrrepsBlockData, got {type(obj)!r}")

    def _maybe_upload_image(self, trainer: Any, key: str, path: Path) -> None:
        run = _get_logger_run(trainer)
        if run is None:
            return
        try:
            import wandb

            _maybe_log_wandb(run, {key: wandb.Image(str(path))})
        except Exception:
            return

    def _maybe_upload_video(self, trainer: Any, key: str, path: Path) -> None:
        run = _get_logger_run(trainer)
        if run is None:
            return
        try:
            import wandb

            _maybe_log_wandb(
                run, {key: wandb.Video(str(path), fps=self.video_fps, format="mp4")}
            )
        except Exception:
            return

    def _align_pred_matrices(
        self,
        preds: dict[str, Any],
        y: dict[str, Any],
        mapper,
        *,
        require_exact_prefix: bool = False,
    ) -> dict[str, BlockMatrix]:
        aligned: dict[str, BlockMatrix] = {}
        for name, pred_obj in preds.items():
            if name not in y:
                continue
            pred_mat = self._as_block_matrix(pred_obj, mapper)
            target_mat = self._as_block_matrix(y[name], mapper)
            aligned[name] = align_pred_block_matrix_to_target_edges(
                pred_mat,
                target_mat,
                require_exact_prefix=require_exact_prefix,
            )
        return aligned

    def _metric_prediction_matrices(
        self,
        aligned_preds: dict[str, BlockMatrix],
        y: dict[str, Any],
        trace_alignment: dict[str, Any],
        pl_module,
    ) -> tuple[dict[str, BlockMatrix], float | None]:
        metrics_preds = dict(aligned_preds)
        if not getattr(pl_module.cfg, "rescale_density_to_num_electrons", False):
            return metrics_preds, None
        if (
            "density" not in metrics_preds
            or "overlap" not in metrics_preds
            or "num_electrons" not in y
        ):
            return metrics_preds, None

        num_electrons_target = y["num_electrons"]
        num_electrons_pred = trace_matmul_sparse_block_matrix_aligned(
            metrics_preds["density"],
            metrics_preds["overlap"],
            trace_alignment,
        )
        num_electrons_mae_pre_correction = float(
            torch.mean(torch.abs(num_electrons_pred - num_electrons_target))
            .detach()
            .cpu()
            .item()
        )
        pred_safe = torch.where(
            torch.abs(num_electrons_pred) > 1e-12,
            num_electrons_pred,
            torch.full_like(num_electrons_pred, 1e-12),
        )
        density_scale = float((num_electrons_target / pred_safe).detach().cpu().item())
        metrics_preds["density"] = metrics_preds["density"] * density_scale
        return metrics_preds, num_electrons_mae_pre_correction

    def _evaluate_epoch_split(self, trainer: Any, pl_module) -> dict[str, Any] | None:
        all_irreps = get_all_irreps(pl_module.mapper)
        detailed_sum = {
            "mae": 0.0,
            "mse": 0.0,
            "mae_mod": 0.0,
            "mse_mod": 0.0,
            "mu_H": 0.0,
            "correction_mae": 0.0,
            "correction_mse": 0.0,
        }
        basic_sum_by_name: dict[str, dict[str, float]] = {}
        per_irrep_sum_by_name: dict[str, dict[str, float]] = {}
        forces_mae_sum = 0.0
        forces_mse_sum = 0.0
        forces_count = 0
        energy_metric_sums = {
            "energy_mae": 0.0,
            "energy_mae_gt_hamiltonian": 0.0,
            "energy_mae_gt_density": 0.0,
        }
        energy_metric_counts = {
            "energy_mae": 0,
            "energy_mae_gt_hamiltonian": 0,
            "energy_mae_gt_density": 0,
        }
        num_metric_sums = {
            "num_electrons_mae": 0.0,
            "num_electrons_mae_gt_overlap": 0.0,
            "num_electrons_mae_gt_density": 0.0,
        }
        num_metric_counts = {
            "num_electrons_mae": 0,
            "num_electrons_mae_gt_overlap": 0,
            "num_electrons_mae_gt_density": 0,
        }
        num_electrons_mae_pre_correction_sum = 0.0
        num_electrons_pre_correction_count = 0
        hamiltonian_contrib_abs_sum = 0.0
        hamiltonian_contrib_count = 0
        hamiltonian_contrib_irrep_sums: dict[str, float] = {}
        hamiltonian_contrib_pair_sums: dict[str, float] = {}
        first_payload = None
        n_batches = 0

        for batch in self._iter_eval_batches(trainer):
            x, y, preds = self._predict(pl_module, batch)
            aligned_preds = self._align_pred_matrices(
                preds,
                y,
                pl_module.mapper,
                require_exact_prefix=bool(
                    getattr(pl_module.cfg, "require_exact_edge_match", False)
                ),
            )
            observable_trace_alignment = {}
            if aligned_preds:
                first_name = next(iter(aligned_preds))
                observable_trace_alignment = build_observable_trace_alignment(
                    self._as_block_matrix(y[first_name], pl_module.mapper)
                )
            metrics_preds, num_electrons_mae_pre_correction = (
                self._metric_prediction_matrices(
                    aligned_preds,
                    y,
                    observable_trace_alignment,
                    pl_module,
                )
            )
            if (
                aligned_preds
                and getattr(pl_module.cfg, "require_exact_edge_match", False)
                and not self._printed_strict_checks
            ):
                log_strict_checks_passed()
                self._printed_strict_checks = True

            if "hamiltonian" in metrics_preds and "hamiltonian" in y and "overlap" in y:
                detail = compute_detailed_metrics_aligned(
                    metrics_preds["hamiltonian"],
                    self._as_block_matrix(y["hamiltonian"], pl_module.mapper),
                    self._as_block_matrix(y["overlap"], pl_module.mapper),
                )
                for key, value in detail.items():
                    detailed_sum[key] += float(value)

            for name, pred_mat in metrics_preds.items():
                target_mat = self._as_block_matrix(y[name], pl_module.mapper)
                basic = compute_basic_matrix_metrics_aligned(pred_mat, target_mat)
                basic_acc = basic_sum_by_name.setdefault(name, {})
                for key, value in basic.items():
                    basic_acc[key] = basic_acc.get(key, 0.0) + float(value)

                pred_ir = align_pred_irreps_to_target_edges(
                    pred_mat.to_vectors(pl_module.mapper),
                    target_mat.to_vectors(pl_module.mapper),
                    require_exact_prefix=bool(
                        getattr(pl_module.cfg, "require_exact_edge_match", False)
                    ),
                )
                targ_ir = target_mat.to_vectors(pl_module.mapper)
                per_irrep = compute_irrep_metrics(
                    pred_ir,
                    targ_ir,
                    all_irreps,
                    pl_module.mapper,
                )
                per_irrep_acc = per_irrep_sum_by_name.setdefault(name, {})
                for key, value in per_irrep.items():
                    per_irrep_acc[key] = per_irrep_acc.get(key, 0.0) + float(value)

                if name == "hamiltonian" and (
                    getattr(
                        pl_module.cfg, "log_hamiltonian_irrep_contrib_metrics", False
                    )
                    or getattr(
                        pl_module.cfg, "log_hamiltonian_pair_contrib_metrics", False
                    )
                ):
                    h_contribs = compute_hamiltonian_mae_contributions(
                        pred_mat.to_vectors(pl_module.mapper),
                        target_mat.to_vectors(pl_module.mapper),
                        pl_module.mapper,
                        all_irreps=all_irreps,
                    )
                    hamiltonian_contrib_abs_sum += float(h_contribs["total_abs_sum"])
                    hamiltonian_contrib_count += int(h_contribs["total_count"])
                    for key, value in h_contribs["irrep_abs_sums"].items():
                        hamiltonian_contrib_irrep_sums[key] = (
                            hamiltonian_contrib_irrep_sums.get(key, 0.0) + float(value)
                        )
                    for key, value in h_contribs["pair_abs_sums"].items():
                        hamiltonian_contrib_pair_sums[key] = (
                            hamiltonian_contrib_pair_sums.get(key, 0.0) + float(value)
                        )

            if (
                getattr(pl_module.cfg, "enable_forces", False)
                and y.get("forces") is not None
                and "hamiltonian" in preds
                and "density" in preds
            ):
                forces_pred = pl_module.get_forces(preds, x["positions"], x["box"])
                forces_true = y["forces"]
                force_diff = forces_pred - forces_true
                forces_mae_sum += float(
                    torch.mean(torch.abs(force_diff)).detach().cpu().item()
                )
                forces_mse_sum += float(torch.mean(force_diff**2).detach().cpu().item())
                forces_count += 1

            observable_values = build_observable_predictions(
                metrics_preds,
                trace_alignment=observable_trace_alignment,
                H_true=self._as_block_matrix(y["hamiltonian"], pl_module.mapper),
                D_true=self._as_block_matrix(y["density"], pl_module.mapper),
                S_true=self._as_block_matrix(y["overlap"], pl_module.mapper),
            )
            if pl_module.cfg.enable_energy and "energy" in y:
                if "energy" in observable_values:
                    energy_metric_sums["energy_mae"] += float(
                        (
                            torch.mean(
                                torch.abs(observable_values["energy"] - y["energy"])
                            )
                            * 27.2113845
                        ).item()
                    )
                    energy_metric_counts["energy_mae"] += 1
                if "energy_gt_hamiltonian" in observable_values:
                    energy_metric_sums["energy_mae_gt_hamiltonian"] += float(
                        (
                            torch.mean(
                                torch.abs(
                                    observable_values["energy_gt_hamiltonian"]
                                    - y["energy"]
                                )
                            )
                            * 27.2113845
                        ).item()
                    )
                    energy_metric_counts["energy_mae_gt_hamiltonian"] += 1
                if "energy_gt_density" in observable_values:
                    energy_metric_sums["energy_mae_gt_density"] += float(
                        (
                            torch.mean(
                                torch.abs(
                                    observable_values["energy_gt_density"] - y["energy"]
                                )
                            )
                            * 27.2113845
                        ).item()
                    )
                    energy_metric_counts["energy_mae_gt_density"] += 1
            if pl_module.cfg.enable_num_electrons and "num_electrons" in y:
                if "num_electrons" in observable_values:
                    num_metric_sums["num_electrons_mae"] += float(
                        torch.mean(
                            torch.abs(
                                observable_values["num_electrons"] - y["num_electrons"]
                            )
                        ).item()
                    )
                    num_metric_counts["num_electrons_mae"] += 1
                if "num_electrons_gt_overlap" in observable_values:
                    num_metric_sums["num_electrons_mae_gt_overlap"] += float(
                        torch.mean(
                            torch.abs(
                                observable_values["num_electrons_gt_overlap"]
                                - y["num_electrons"]
                            )
                        ).item()
                    )
                    num_metric_counts["num_electrons_mae_gt_overlap"] += 1
                if "num_electrons_gt_density" in observable_values:
                    num_metric_sums["num_electrons_mae_gt_density"] += float(
                        torch.mean(
                            torch.abs(
                                observable_values["num_electrons_gt_density"]
                                - y["num_electrons"]
                            )
                        ).item()
                    )
                    num_metric_counts["num_electrons_mae_gt_density"] += 1
                if num_electrons_mae_pre_correction is not None:
                    num_electrons_mae_pre_correction_sum += (
                        num_electrons_mae_pre_correction
                    )
                    num_electrons_pre_correction_count += 1

            if first_payload is None:
                first_payload = {"x": x, "y": y, "preds": metrics_preds}
            n_batches += 1

        if n_batches == 0:
            return None

        return {
            "detailed": {k: v / n_batches for k, v in detailed_sum.items()},
            "basic_by_name": {
                matrix_name: {k: v / n_batches for k, v in metrics.items()}
                for matrix_name, metrics in basic_sum_by_name.items()
            },
            "per_irrep_by_name": {
                matrix_name: {k: v / n_batches for k, v in metrics.items()}
                for matrix_name, metrics in per_irrep_sum_by_name.items()
            },
            "forces_mae": (forces_mae_sum / forces_count) if forces_count > 0 else None,
            "forces_mse": (forces_mse_sum / forces_count) if forces_count > 0 else None,
            "energy_mae": (
                energy_metric_sums["energy_mae"] / energy_metric_counts["energy_mae"]
                if energy_metric_counts["energy_mae"] > 0
                else None
            ),
            "energy_mae_gt_hamiltonian": (
                energy_metric_sums["energy_mae_gt_hamiltonian"]
                / energy_metric_counts["energy_mae_gt_hamiltonian"]
                if energy_metric_counts["energy_mae_gt_hamiltonian"] > 0
                else None
            ),
            "energy_mae_gt_density": (
                energy_metric_sums["energy_mae_gt_density"]
                / energy_metric_counts["energy_mae_gt_density"]
                if energy_metric_counts["energy_mae_gt_density"] > 0
                else None
            ),
            "num_electrons_mae": (
                num_metric_sums["num_electrons_mae"]
                / num_metric_counts["num_electrons_mae"]
                if num_metric_counts["num_electrons_mae"] > 0
                else None
            ),
            "num_electrons_mae_gt_overlap": (
                num_metric_sums["num_electrons_mae_gt_overlap"]
                / num_metric_counts["num_electrons_mae_gt_overlap"]
                if num_metric_counts["num_electrons_mae_gt_overlap"] > 0
                else None
            ),
            "num_electrons_mae_gt_density": (
                num_metric_sums["num_electrons_mae_gt_density"]
                / num_metric_counts["num_electrons_mae_gt_density"]
                if num_metric_counts["num_electrons_mae_gt_density"] > 0
                else None
            ),
            "num_electrons_mae_pre_correction": (
                num_electrons_mae_pre_correction_sum
                / num_electrons_pre_correction_count
                if num_electrons_pre_correction_count > 0
                else None
            ),
            "hamiltonian_contrib_abs_sum": hamiltonian_contrib_abs_sum,
            "hamiltonian_contrib_count": hamiltonian_contrib_count,
            "hamiltonian_contrib_irrep_sums": hamiltonian_contrib_irrep_sums,
            "hamiltonian_contrib_pair_sums": hamiltonian_contrib_pair_sums,
            "log_hamiltonian_irrep_contrib_metrics": bool(
                getattr(pl_module.cfg, "log_hamiltonian_irrep_contrib_metrics", False)
            ),
            "log_hamiltonian_pair_contrib_metrics": bool(
                getattr(pl_module.cfg, "log_hamiltonian_pair_contrib_metrics", False)
            ),
            "first_payload": first_payload,
            "num_batches": n_batches,
        }

    def _build_epoch_wandb_payload(
        self,
        epoch_idx: int,
        eval_result: dict[str, Any],
        train_loss: float,
        val_loss: float,
        all_irreps,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"epoch": epoch_idx}
        payload.update(
            build_wandb_detailed_metrics_log(
                epoch_zero_based=epoch_idx,
                loss_value=train_loss,
                detailed_metrics=eval_result["detailed"],
            )
        )
        basic_by_name = eval_result["basic_by_name"]
        if "hamiltonian" in basic_by_name:
            payload["mae_H"] = basic_by_name["hamiltonian"]["mae"]
            payload["mse_H"] = basic_by_name["hamiltonian"]["mse"]
        if "overlap" in basic_by_name:
            payload["mae_S"] = basic_by_name["overlap"]["mae"]
            payload["mse_S"] = basic_by_name["overlap"]["mse"]
        if "density" in basic_by_name:
            payload["mae_D"] = basic_by_name["density"]["mae"]
            payload["mse_D"] = basic_by_name["density"]["mse"]
        if eval_result["energy_mae"] is not None:
            payload["val/energy_mae_study"] = eval_result["energy_mae"]
        if eval_result["energy_mae_gt_hamiltonian"] is not None:
            payload["val/energy_mae_gt_hamiltonian"] = eval_result[
                "energy_mae_gt_hamiltonian"
            ]
        if eval_result["energy_mae_gt_density"] is not None:
            payload["val/energy_mae_gt_density"] = eval_result["energy_mae_gt_density"]
        if eval_result["num_electrons_mae"] is not None:
            payload["val/num_electrons_mae_study"] = eval_result["num_electrons_mae"]
        if eval_result["num_electrons_mae_gt_overlap"] is not None:
            payload["val/num_electrons_mae_gt_overlap"] = eval_result[
                "num_electrons_mae_gt_overlap"
            ]
        if eval_result["num_electrons_mae_gt_density"] is not None:
            payload["val/num_electrons_mae_gt_density"] = eval_result[
                "num_electrons_mae_gt_density"
            ]
        if eval_result["num_electrons_mae_pre_correction"] is not None:
            payload["val/num_electrons_mae_pre_correction"] = eval_result[
                "num_electrons_mae_pre_correction"
            ]
        if eval_result["hamiltonian_contrib_count"] > 0:
            denom = float(eval_result["hamiltonian_contrib_count"])
            if eval_result["log_hamiltonian_irrep_contrib_metrics"]:
                for irrep_key, abs_sum in eval_result[
                    "hamiltonian_contrib_irrep_sums"
                ].items():
                    payload[f"val/hamiltonian_mae_{irrep_key}"] = (
                        abs_sum / denom * 27.2113845
                    )
            if eval_result["log_hamiltonian_pair_contrib_metrics"]:
                for pair_key, abs_sum in eval_result[
                    "hamiltonian_contrib_pair_sums"
                ].items():
                    payload[f"val/hamiltonian_mae_{pair_key.replace('-', '_')}"] = (
                        abs_sum / denom * 27.2113845
                    )
        if (
            eval_result["forces_mae"] is not None
            and eval_result["forces_mse"] is not None
        ):
            payload["mae_F"] = eval_result["forces_mae"]
            payload["mse_F"] = eval_result["forces_mse"]
        payload["val/loss"] = val_loss
        for matrix_name, per_irrep in eval_result["per_irrep_by_name"].items():
            payload.update(
                build_wandb_per_irrep_metrics_log(
                    epoch_zero_based=epoch_idx,
                    all_irreps=all_irreps,
                    per_irrep_metrics=per_irrep,
                    metric_prefix=IRREP_PREFIX_BY_MATRIX.get(matrix_name, ""),
                )
            )
        return payload

    def _should_log_now(self, epoch_zero_based: int, pl_module) -> bool:
        return should_log_epoch(
            epoch_zero_based,
            pl_module.cfg.log_interval,
            pl_module.cfg.adaptive_log_interval,
        )

    def on_fit_start(self, trainer, pl_module) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.per_irrep_dir.mkdir(parents=True, exist_ok=True)
        self.reference_batch = self._get_reference_batch(trainer)
        run = _get_logger_run(trainer)
        if run is not None:
            try:
                run.summary["checkpoint/latest_path"] = str(self.latest_path.resolve())
                run.summary["checkpoint/best_path"] = str(self.best_path.resolve())
                run.summary["checkpoint/final_path"] = str(self.final_path.resolve())
            except Exception:
                pass
        initial_eval = self._evaluate_epoch_split(trainer, pl_module)
        if initial_eval is not None:
            log_detailed_training_metrics(
                avg_epoch_time=0.0,
                epochs_since_last_log=0,
                time_elapsed=0.0,
                loss_value=(
                    float(
                        trainer.callback_metrics.get(self.monitor, torch.tensor(0.0))
                        .detach()
                        .cpu()
                        .item()
                    )
                    if self.monitor in trainer.callback_metrics
                    else 0.0
                ),
                detailed_metrics=initial_eval["detailed"],
            )
        if initial_eval is not None and pl_module.cfg.print_per_irrep_metrics:
            all_irreps = get_all_irreps(pl_module.mapper)
            for matrix_name, per_irrep in initial_eval["per_irrep_by_name"].items():
                log_per_irrep_metrics(
                    f"Initial Per-Irrep Metrics ({matrix_name})",
                    all_irreps,
                    per_irrep,
                    metric_prefix=IRREP_PREFIX_BY_MATRIX.get(matrix_name, ""),
                )
        if initial_eval is not None:
            all_irreps = get_all_irreps(pl_module.mapper)
            payload = {}
            payload.update(
                {
                    f"initial/{k}": v
                    for k, v in build_wandb_detailed_metrics_log(
                        0, 0.0, initial_eval["detailed"]
                    ).items()
                    if k != "epoch"
                }
            )
            payload["initial/loss_total"] = (
                float(
                    trainer.callback_metrics.get(self.monitor, torch.tensor(0.0))
                    .detach()
                    .cpu()
                    .item()
                )
                if self.monitor in trainer.callback_metrics
                else 0.0
            )
            for matrix_name, basic in initial_eval["basic_by_name"].items():
                alias = MATRIX_ALIAS.get(matrix_name, matrix_name)
                payload[f"initial/mae_{alias}"] = basic["mae"]
                payload[f"initial/mse_{alias}"] = basic["mse"]
                per_irrep_payload = build_wandb_per_irrep_metrics_log(
                    0,
                    all_irreps,
                    initial_eval["per_irrep_by_name"].get(matrix_name, {}),
                    metric_prefix=IRREP_PREFIX_BY_MATRIX.get(matrix_name, ""),
                )
                for key, value in per_irrep_payload.items():
                    if key != "epoch":
                        payload[f"initial/{alias}_{key}"] = value
            if initial_eval["energy_mae"] is not None:
                payload["initial/energy_mae"] = initial_eval["energy_mae"]
            if initial_eval["energy_mae_gt_hamiltonian"] is not None:
                payload["initial/energy_mae_gt_hamiltonian"] = initial_eval[
                    "energy_mae_gt_hamiltonian"
                ]
            if initial_eval["energy_mae_gt_density"] is not None:
                payload["initial/energy_mae_gt_density"] = initial_eval[
                    "energy_mae_gt_density"
                ]
            if initial_eval["num_electrons_mae"] is not None:
                payload["initial/num_electrons_mae"] = initial_eval["num_electrons_mae"]
            if initial_eval["num_electrons_mae_gt_overlap"] is not None:
                payload["initial/num_electrons_mae_gt_overlap"] = initial_eval[
                    "num_electrons_mae_gt_overlap"
                ]
            if initial_eval["num_electrons_mae_gt_density"] is not None:
                payload["initial/num_electrons_mae_gt_density"] = initial_eval[
                    "num_electrons_mae_gt_density"
                ]
            if initial_eval["num_electrons_mae_pre_correction"] is not None:
                payload["initial/num_electrons_mae_pre_correction"] = initial_eval[
                    "num_electrons_mae_pre_correction"
                ]
            if initial_eval["hamiltonian_contrib_count"] > 0:
                denom = float(initial_eval["hamiltonian_contrib_count"])
                if initial_eval["log_hamiltonian_irrep_contrib_metrics"]:
                    for irrep_key, abs_sum in initial_eval[
                        "hamiltonian_contrib_irrep_sums"
                    ].items():
                        payload[f"initial/hamiltonian_mae_{irrep_key}"] = (
                            abs_sum / denom * 27.2113845
                        )
                if initial_eval["log_hamiltonian_pair_contrib_metrics"]:
                    for pair_key, abs_sum in initial_eval[
                        "hamiltonian_contrib_pair_sums"
                    ].items():
                        payload[
                            f"initial/hamiltonian_mae_{pair_key.replace('-', '_')}"
                        ] = (abs_sum / denom * 27.2113845)
            if (
                initial_eval["forces_mae"] is not None
                and initial_eval["forces_mse"] is not None
            ):
                payload["initial/mae_F"] = initial_eval["forces_mae"]
                payload["initial/mse_F"] = initial_eval["forces_mse"]
            _maybe_log_wandb(run, payload)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is not None:
            score = float(metric.detach().cpu().item())
            if self.save_best and self._is_better(score):
                trainer.save_checkpoint(str(self.best_path))
                self.state.best_score = score
                self.state.best_epoch = int(trainer.current_epoch)
        if self.save_latest:
            trainer.save_checkpoint(str(self.latest_path))

        if self.reference_batch is None or not self._should_log_now(
            int(trainer.current_epoch), pl_module
        ):
            return
        eval_result = self._evaluate_epoch_split(trainer, pl_module)
        if eval_result is None:
            return
        train_loss = float(
            trainer.callback_metrics.get("train/loss_total", torch.tensor(0.0))
            .detach()
            .cpu()
            .item()
        )
        val_loss = float(
            trainer.callback_metrics.get(self.monitor, torch.tensor(0.0))
            .detach()
            .cpu()
            .item()
        )
        current_epoch = int(trainer.current_epoch)
        time_elapsed = 0.0
        epochs_since_last_log = (
            current_epoch - self._last_logged_epoch
            if self._last_logged_epoch >= 0
            else current_epoch + 1
        )
        avg_epoch_time = 0.0
        self._last_logged_epoch = current_epoch
        print(f"\n{'=' * 60}")
        print(
            f"EPOCH {current_epoch + 1}  |  lr={trainer.optimizers[0].param_groups[0]['lr']:.6e}"
        )
        print(f"{'=' * 60}")
        log_detailed_training_metrics(
            avg_epoch_time=avg_epoch_time,
            epochs_since_last_log=epochs_since_last_log,
            time_elapsed=time_elapsed,
            loss_value=train_loss,
            detailed_metrics=eval_result["detailed"],
        )
        if pl_module.cfg.print_per_irrep_metrics:
            all_irreps = get_all_irreps(pl_module.mapper)
            for matrix_name, per_irrep in eval_result["per_irrep_by_name"].items():
                log_per_irrep_metrics(
                    f"Validation Per-Irrep Metrics ({matrix_name})",
                    all_irreps,
                    per_irrep,
                    metric_prefix=IRREP_PREFIX_BY_MATRIX.get(matrix_name, ""),
                )
        run = _get_logger_run(trainer)
        payload = self._build_epoch_wandb_payload(
            current_epoch,
            eval_result,
            train_loss,
            val_loss,
            get_all_irreps(pl_module.mapper),
        )
        _maybe_log_wandb(run, payload)
        fp = eval_result["first_payload"]
        if fp is not None:
            self._save_epoch_frame(trainer, pl_module, fp["x"], fp["y"], fp["preds"])

    def _save_epoch_frame(self, trainer, pl_module, x, y, preds) -> None:
        epoch = int(trainer.current_epoch)
        for name, pred_mat in preds.items():
            if name not in y:
                continue
            target_mat = self._as_block_matrix(y[name], pl_module.mapper)
            ref = None
            if name == "hamiltonian" and "overlap" in y:
                ref = self._as_block_matrix(y["overlap"], pl_module.mapper)
            frame_dir = self.frames_dir / name
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / f"frame_epoch_{epoch:06d}.png"
            save_matrix_comparison_plot(
                pred_mat,
                target_mat,
                frame_path,
                reference=ref,
                title=f"{MATRIX_ALIAS.get(name, name)} epoch {epoch}",
                max_atoms=getattr(pl_module.cfg, "video_max_atoms", None),
            )

    def on_fit_end(self, trainer, pl_module) -> None:
        if self.save_final:
            trainer.save_checkpoint(str(self.final_path))

        if self.reference_batch is None:
            return
        eval_result = self._evaluate_epoch_split(trainer, pl_module)
        if eval_result is None:
            return
        fp = eval_result["first_payload"]
        if fp is None:
            return
        x, y, preds = fp["x"], fp["y"], fp["preds"]
        final_detailed = eval_result["detailed"]
        log_final_metrics(final_detailed)
        run = _get_logger_run(trainer)
        final_payload = {
            "final/mae_H": final_detailed["mae"],
            "final/mse_H": final_detailed["mse"],
            "final/mae_H_mod": final_detailed["mae_mod"],
            "final/mse_H_mod": final_detailed["mse_mod"],
            "final/mu_H": final_detailed["mu_H"],
            "final/correction_mae": final_detailed["correction_mae"],
            "final/correction_mse": final_detailed["correction_mse"],
        }
        for matrix_name, basic in eval_result["basic_by_name"].items():
            alias = MATRIX_ALIAS.get(matrix_name, matrix_name)
            final_payload[f"final/mae_{alias}"] = basic["mae"]
            final_payload[f"final/mse_{alias}"] = basic["mse"]
        if eval_result["energy_mae"] is not None:
            final_payload["final/energy_mae"] = eval_result["energy_mae"]
        if eval_result["energy_mae_gt_hamiltonian"] is not None:
            final_payload["final/energy_mae_gt_hamiltonian"] = eval_result[
                "energy_mae_gt_hamiltonian"
            ]
        if eval_result["energy_mae_gt_density"] is not None:
            final_payload["final/energy_mae_gt_density"] = eval_result[
                "energy_mae_gt_density"
            ]
        if eval_result["num_electrons_mae"] is not None:
            final_payload["final/num_electrons_mae"] = eval_result["num_electrons_mae"]
        if eval_result["num_electrons_mae_gt_overlap"] is not None:
            final_payload["final/num_electrons_mae_gt_overlap"] = eval_result[
                "num_electrons_mae_gt_overlap"
            ]
        if eval_result["num_electrons_mae_gt_density"] is not None:
            final_payload["final/num_electrons_mae_gt_density"] = eval_result[
                "num_electrons_mae_gt_density"
            ]
        if eval_result["num_electrons_mae_pre_correction"] is not None:
            final_payload["final/num_electrons_mae_pre_correction"] = eval_result[
                "num_electrons_mae_pre_correction"
            ]
        if eval_result["hamiltonian_contrib_count"] > 0:
            denom = float(eval_result["hamiltonian_contrib_count"])
            if eval_result["log_hamiltonian_irrep_contrib_metrics"]:
                for irrep_key, abs_sum in eval_result[
                    "hamiltonian_contrib_irrep_sums"
                ].items():
                    final_payload[f"final/hamiltonian_mae_{irrep_key}"] = (
                        abs_sum / denom * 27.2113845
                    )
            if eval_result["log_hamiltonian_pair_contrib_metrics"]:
                for pair_key, abs_sum in eval_result[
                    "hamiltonian_contrib_pair_sums"
                ].items():
                    final_payload[
                        f"final/hamiltonian_mae_{pair_key.replace('-', '_')}"
                    ] = (abs_sum / denom * 27.2113845)
        if (
            eval_result["forces_mae"] is not None
            and eval_result["forces_mse"] is not None
        ):
            final_payload["final/mae_F"] = eval_result["forces_mae"]
            final_payload["final/mse_F"] = eval_result["forces_mse"]

        # final plots
        for name in pl_module.cfg.matrix_targets:
            if name not in preds or name not in y:
                continue
            pred_mat = preds[name]
            target_mat = self._as_block_matrix(y[name], pl_module.mapper)

            ref = None
            if name == "hamiltonian" and "overlap" in y:
                ref = self._as_block_matrix(y["overlap"], pl_module.mapper)

            if name == "hamiltonian" and "overlap" in y:
                dos_path = self.output_dir / "dos_comparison_final.png"
                dos_metrics = save_dos_comparison_plot(
                    pred_mat, target_mat, ref, dos_path
                )
                for key, value in dos_metrics.items():
                    final_payload[f"final/{key}"] = value
                self._maybe_upload_image(trainer, "final/dos_comparison_plot", dos_path)

            curve = compute_distance_error_curve(
                pred_mat,
                target_mat,
                x["positions"],
                x.get("box"),
                n_bins=self.distance_bins,
            )
            if curve is not None:
                curve_json_path = self.output_dir / f"distance_error_curve_{name}.json"
                curve_path = self.output_dir / f"distance_error_curve_{name}.png"
                curve_json_path.write_text(
                    json.dumps(curve, indent=2), encoding="utf-8"
                )
                save_distance_error_curve_plot(
                    curve,
                    curve_path,
                    title=f"Distance Error Curves ({name}, Final)",
                )
                self._maybe_upload_image(
                    trainer,
                    f"distance_curve/{MATRIX_ALIAS.get(name, name)}_plot",
                    curve_path,
                )

            if self.log_per_irrep_images:
                per_irrep_subdir = self.per_irrep_dir / name
                per_irrep_subdir.mkdir(parents=True, exist_ok=True)
                target_irreps = target_mat.to_vectors(pl_module.mapper)
                pred_irreps = pred_mat.to_vectors(pl_module.mapper)
                all_irreps = get_all_irreps(pl_module.mapper)
                for irrep in all_irreps:
                    pred_ir = filter_irreps_block_data_by_irrep(
                        pred_irreps, irrep, pl_module.mapper
                    ).to_blocks(pl_module.mapper)
                    tgt_ir = filter_irreps_block_data_by_irrep(
                        target_irreps, irrep, pl_module.mapper
                    ).to_blocks(pl_module.mapper)
                    if not pred_ir.pair_blocks or not tgt_ir.pair_blocks:
                        continue
                    img_path = per_irrep_subdir / f"{irrep}.png"
                    save_matrix_comparison_plot(
                        pred_ir,
                        tgt_ir,
                        img_path,
                        reference=ref,
                        title=f"{MATRIX_ALIAS.get(name, name)} / {irrep}",
                    )
                    self._maybe_upload_image(
                        trainer,
                        f"irrep_images/{MATRIX_ALIAS.get(name, name)}/{irrep}",
                        img_path,
                    )

        if self.generate_video:
            for name in pl_module.cfg.matrix_targets:
                frame_dir = self.frames_dir / name
                if not frame_dir.exists():
                    continue
                video_path = self.output_dir / f"training_progress_{name}.mp4"
                compile_frames_to_video(
                    frame_dir.glob("*.png"),
                    video_path,
                    fps=self.video_fps,
                    format="mp4",
                )
                if video_path.exists():
                    try:
                        import wandb

                        _maybe_log_wandb(
                            run,
                            {
                                f"training_video_{MATRIX_ALIAS.get(name, name)}": wandb.Video(
                                    str(video_path), fps=self.video_fps, format="mp4"
                                )
                            },
                        )
                    except Exception:
                        pass

        _maybe_log_wandb(run, final_payload)
        log_study_complete(
            run_name=getattr(run, "name", "run"),
            total_training_epochs=int(trainer.current_epoch) + 1,
            final_loss=float(
                trainer.callback_metrics.get("train/loss_total", torch.tensor(0.0))
                .detach()
                .cpu()
                .item()
            ),
            best_loss=(
                self.state.best_score
                if self.state.best_score is not None
                else float("nan")
            ),
            best_epoch=(
                (self.state.best_epoch + 1) if self.state.best_epoch is not None else -1
            ),
            run_checkpoint_dir=self.output_dir,
            final_model_path=self.final_path,
        )

    def on_exception(self, trainer, pl_module, exception) -> None:
        if not self.save_latest:
            return
        try:
            trainer.save_checkpoint(str(self.latest_path))
            print(
                f"--- Saved latest checkpoint after interrupt/exception: {self.latest_path} ---"
            )
        except Exception:
            pass
