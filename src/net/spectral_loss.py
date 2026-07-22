from __future__ import annotations

import math
import time
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.periodic_fourier import (
    phase_matrix,
    shiftspace_to_kspace_dense,
    translation_shifts_for_kmesh,
)
from data.block_matrix import BlockMatrix
from data.kspace_snapshot import (
    _fractional_to_cartesian_kpoints,
    _generalized_eigenvalues_from_cholesky,
    _generalized_eigenvalues_kspace,
    _prepare_overlap_cholesky_kspace,
    block_matrix_to_shiftspace_dense_aligned,
    block_matrix_to_shiftspace_dense,
    build_shiftspace_scatter_metadata,
)
from utils.units import HARTREE_TO_EV


def parse_kmesh_spec(spec: str) -> tuple[int, int, int]:
    raw = str(spec).strip().lower().replace(" ", "")
    parts = raw.split("x")
    if len(parts) != 3:
        raise ValueError(f"Expected kmesh like '2x2x2', got {spec!r}")
    kmesh = tuple(int(part) for part in parts)
    if min(kmesh) <= 0:
        raise ValueError(f"kmesh entries must be positive, got {spec!r}")
    return kmesh


def fractional_kmesh_points(
    kmesh: tuple[int, int, int], *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    grids = [torch.arange(n, device=device, dtype=dtype) / float(n) for n in kmesh]
    mesh = torch.meshgrid(*grids, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=1)


def _soft_window_weights(
    gt_eigs_ev: torch.Tensor,
    *,
    fermi_ev: float,
    window_ev: float,
    taper_ev: float,
) -> torch.Tensor:
    rel = torch.abs(gt_eigs_ev - float(fermi_ev))
    if taper_ev <= 0.0 or taper_ev >= window_ev:
        return (rel <= window_ev).to(dtype=gt_eigs_ev.dtype)
    core = window_ev - taper_ev
    weights = torch.zeros_like(rel)
    weights = torch.where(rel <= core, torch.ones_like(weights), weights)
    in_taper = (rel > core) & (rel < window_ev)
    if bool(in_taper.any()):
        phase = (rel[in_taper] - core) / taper_ev
        weights[in_taper] = 0.5 * (1.0 + torch.cos(math.pi * phase))
    return weights


def build_spectral_reference(
    *,
    hamiltonian: BlockMatrix,
    overlap: BlockMatrix,
    box: torch.Tensor,
    fermi_level_hartree: torch.Tensor | float,
    kmesh_spec: str,
    window_ev: float,
    taper_ev: float,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
    progress_label: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    device = box.device
    real_dtype = box.dtype
    kmesh = parse_kmesh_spec(kmesh_spec)
    total_kpoints = kmesh[0] * kmesh[1] * kmesh[2]
    pbar = None
    if verbose:
        label = progress_label or "spectral"
        print(
            f"--- [{label}] Building spectral reference: "
            f"kmesh={kmesh_spec} ({total_kpoints} k-points), "
            f"window=±{window_ev:.2f} eV, taper={taper_ev:.2f} eV ---",
            flush=True,
        )
        pbar = tqdm(total=5, desc=f"Spectral precompute [{label}]", leave=False)

    def _advance(stage: str) -> None:
        if verbose:
            print(
                f"--- [{progress_label or 'spectral'}] {stage} ---",
                flush=True,
            )
        if pbar is not None:
            pbar.update(1)

    shifts = translation_shifts_for_kmesh(kmesh, device=device)
    fractional_kpoints = fractional_kmesh_points(kmesh, device=device, dtype=real_dtype)
    kpoints_abs = _fractional_to_cartesian_kpoints(fractional_kpoints, box)
    _advance("constructed k-mesh and Cartesian k-points")
    ham_shift = block_matrix_to_shiftspace_dense(hamiltonian, shifts=shifts)
    ovl_shift = block_matrix_to_shiftspace_dense(overlap, shifts=shifts)
    scatter_metadata, dense_shape = build_shiftspace_scatter_metadata(
        hamiltonian,
        shifts=shifts,
    )
    _advance("converted Hamiltonian and overlap to shift-space dense tensors")
    spectral_phase = phase_matrix(kpoints_abs, shifts, box)
    ham_k = shiftspace_to_kspace_dense(
        ham_shift,
        kpoints_abs=kpoints_abs,
        shifts=shifts,
        box=box,
        phase=spectral_phase,
    )
    ovl_k = shiftspace_to_kspace_dense(
        ovl_shift,
        kpoints_abs=kpoints_abs,
        shifts=shifts,
        box=box,
        phase=spectral_phase,
    )
    _advance("Fourier transformed shift-space tensors to k-space")
    overlap_cholesky = _prepare_overlap_cholesky_kspace(
        ovl_k,
        psd_cleanup=overlap_psd_cleanup,
        allow_jitter=overlap_jitter,
    )
    gt_eigs_hartree = _generalized_eigenvalues_from_cholesky(
        ham_k,
        overlap_cholesky,
    )
    _advance("solved generalized eigenproblems on the reference k-mesh")
    gt_eigs_ev = gt_eigs_hartree.real * HARTREE_TO_EV
    fermi_ev = (
        float(
            fermi_level_hartree.item()
            if torch.is_tensor(fermi_level_hartree)
            else fermi_level_hartree
        )
        * HARTREE_TO_EV
    )
    weights = _soft_window_weights(
        gt_eigs_ev,
        fermi_ev=fermi_ev,
        window_ev=window_ev,
        taper_ev=taper_ev,
    )
    weight_sum = weights.sum()
    if (not bool(torch.isfinite(weight_sum).item())) or float(weight_sum.item()) <= 0.0:
        raise ValueError("Spectral window produced a non-positive total weight.")
    _advance("built spectral window weights around the Fermi level")
    if pbar is not None:
        pbar.close()
    if verbose:
        elapsed = time.perf_counter() - t0
        print(
            f"--- [{progress_label or 'spectral'}] Spectral reference ready "
            f"in {elapsed:.2f}s; active window weight sum={float(weights.sum().item()):.1f} ---",
            flush=True,
        )
    return {
        "spectral_kpoints_abs": kpoints_abs.detach().cpu(),
        "spectral_shifts": shifts.detach().cpu(),
        "spectral_gt_overlap_k": ovl_k.detach().cpu(),
        "spectral_gt_overlap_cholesky": overlap_cholesky.detach().cpu(),
        "spectral_phase": spectral_phase.detach().cpu(),
        "spectral_scatter_metadata": {
            key: (source.detach().cpu(), destination.detach().cpu())
            for key, (source, destination) in scatter_metadata.items()
        },
        "spectral_dense_shape": dense_shape,
        "spectral_gt_eigs_ev": gt_eigs_ev.detach().cpu(),
        "spectral_gt_fermi_ev": torch.tensor(fermi_ev, dtype=real_dtype).cpu(),
        "spectral_window_weights": weights.detach().cpu(),
        "spectral_weight_sum": weight_sum.detach().cpu(),
    }


def compute_spectral_eigenvalue_loss(
    *,
    pred_hamiltonian: BlockMatrix,
    spectral_payload: dict[str, torch.Tensor],
    box: torch.Tensor,
    loss_kind: str,
    huber_delta_ev: float,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    kpoints_abs = spectral_payload["spectral_kpoints_abs"].to(device=box.device)
    shifts = spectral_payload["spectral_shifts"].to(device=box.device)
    # Deliberately always use the ground-truth overlap. Spectral supervision is
    # intended to isolate Hamiltonian quality even when overlap is also predicted.
    overlap_k = spectral_payload["spectral_gt_overlap_k"].to(device=box.device)
    gt_eigs_ev = spectral_payload["spectral_gt_eigs_ev"].to(device=box.device)
    weights = spectral_payload["spectral_window_weights"].to(device=box.device)
    fermi_ev = spectral_payload["spectral_gt_fermi_ev"].to(device=box.device)

    scatter_metadata = spectral_payload.get("spectral_scatter_metadata")
    dense_shape = spectral_payload.get("spectral_dense_shape")
    if scatter_metadata is not None and dense_shape is not None:
        pred_shift = block_matrix_to_shiftspace_dense_aligned(
            pred_hamiltonian,
            scatter_metadata=scatter_metadata,
            dense_shape=tuple(dense_shape),
        )
    else:
        pred_shift = block_matrix_to_shiftspace_dense(pred_hamiltonian, shifts=shifts)
    phase = spectral_payload.get("spectral_phase")
    if phase is not None:
        phase = phase.to(device=box.device)
    pred_h_k = shiftspace_to_kspace_dense(
        pred_shift,
        kpoints_abs=kpoints_abs,
        shifts=shifts,
        box=box,
        phase=phase,
    )
    overlap_cholesky = spectral_payload.get("spectral_gt_overlap_cholesky")
    if overlap_cholesky is None:
        pred_eigs_h = _generalized_eigenvalues_kspace(
            pred_h_k,
            overlap_k,
            psd_cleanup=overlap_psd_cleanup,
            allow_jitter=overlap_jitter,
        )
    else:
        pred_eigs_h = _generalized_eigenvalues_from_cholesky(
            pred_h_k,
            overlap_cholesky.to(device=box.device),
        )
    pred_eigs_ev = pred_eigs_h.real * HARTREE_TO_EV
    pred_rel = pred_eigs_ev - fermi_ev
    gt_rel = gt_eigs_ev - fermi_ev
    abs_err = torch.abs(pred_rel - gt_rel)
    loss_kind = str(loss_kind).lower()
    if loss_kind == "mae":
        per_level = abs_err
    elif loss_kind == "mse":
        per_level = torch.square(pred_rel - gt_rel)
    elif loss_kind == "huber":
        per_level = F.huber_loss(
            pred_rel,
            gt_rel,
            reduction="none",
            delta=huber_delta_ev,
        )
    else:
        raise ValueError("spectral_loss_kind must be one of 'huber', 'mse', or 'mae'.")
    weight_sum = spectral_payload.get("spectral_weight_sum")
    if weight_sum is None:
        weight_sum = weights.sum()
        if (not bool(torch.isfinite(weight_sum).item())) or float(
            weight_sum.item()
        ) <= 0.0:
            raise ValueError("Spectral window produced a non-positive total weight.")
    else:
        weight_sum = weight_sum.to(device=box.device)
    loss = torch.sum(per_level * weights) / weight_sum
    mae = torch.sum(abs_err * weights) / weight_sum
    return loss, {
        "spectral_mae_ev": mae,
        "spectral_weight_sum": weight_sum,
    }
