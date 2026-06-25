from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from core.periodic_fourier import (
    shiftspace_to_kspace_dense,
    translation_shifts_for_kmesh,
)
from data.block_matrix import BlockMatrix
from data.kspace_snapshot import (
    _fractional_to_cartesian_kpoints,
    _generalized_eigenvalues_kspace,
    block_matrix_to_shiftspace_dense,
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
) -> dict[str, Any]:
    device = box.device
    real_dtype = box.dtype
    kmesh = parse_kmesh_spec(kmesh_spec)
    shifts = translation_shifts_for_kmesh(kmesh, device=device)
    fractional_kpoints = fractional_kmesh_points(kmesh, device=device, dtype=real_dtype)
    kpoints_abs = _fractional_to_cartesian_kpoints(fractional_kpoints, box)
    ham_shift = block_matrix_to_shiftspace_dense(hamiltonian, shifts=shifts)
    ovl_shift = block_matrix_to_shiftspace_dense(overlap, shifts=shifts)
    ham_k = shiftspace_to_kspace_dense(
        ham_shift,
        kpoints_abs=kpoints_abs,
        shifts=shifts,
        box=box,
    )
    ovl_k = shiftspace_to_kspace_dense(
        ovl_shift,
        kpoints_abs=kpoints_abs,
        shifts=shifts,
        box=box,
    )
    gt_eigs_hartree = _generalized_eigenvalues_kspace(
        ham_k,
        ovl_k,
        psd_cleanup=overlap_psd_cleanup,
        allow_jitter=overlap_jitter,
    )
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
    return {
        "spectral_kpoints_abs": kpoints_abs.detach().cpu(),
        "spectral_shifts": shifts.detach().cpu(),
        "spectral_gt_overlap_k": ovl_k.detach().cpu(),
        "spectral_gt_eigs_ev": gt_eigs_ev.detach().cpu(),
        "spectral_gt_fermi_ev": torch.tensor(fermi_ev, dtype=real_dtype).cpu(),
        "spectral_window_weights": weights.detach().cpu(),
    }


def compute_spectral_eigenvalue_loss(
    *,
    pred_hamiltonian: BlockMatrix,
    spectral_payload: dict[str, torch.Tensor],
    box: torch.Tensor,
    huber_delta_ev: float,
    overlap_psd_cleanup: bool,
    overlap_jitter: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    kpoints_abs = spectral_payload["spectral_kpoints_abs"].to(device=box.device)
    shifts = spectral_payload["spectral_shifts"].to(device=box.device)
    overlap_k = spectral_payload["spectral_gt_overlap_k"].to(device=box.device)
    gt_eigs_ev = spectral_payload["spectral_gt_eigs_ev"].to(device=box.device)
    weights = spectral_payload["spectral_window_weights"].to(device=box.device)
    fermi_ev = spectral_payload["spectral_gt_fermi_ev"].to(device=box.device)

    pred_shift = block_matrix_to_shiftspace_dense(pred_hamiltonian, shifts=shifts)
    pred_h_k = shiftspace_to_kspace_dense(
        pred_shift,
        kpoints_abs=kpoints_abs,
        shifts=shifts,
        box=box,
    )
    pred_eigs_h = _generalized_eigenvalues_kspace(
        pred_h_k,
        overlap_k,
        psd_cleanup=overlap_psd_cleanup,
        allow_jitter=overlap_jitter,
    )
    pred_eigs_ev = pred_eigs_h.real * HARTREE_TO_EV
    pred_rel = pred_eigs_ev - fermi_ev
    gt_rel = gt_eigs_ev - fermi_ev
    abs_err = torch.abs(pred_rel - gt_rel)
    per_level = F.huber_loss(
        pred_rel,
        gt_rel,
        reduction="none",
        delta=huber_delta_ev,
    )
    weight_sum = weights.sum()
    if (not bool(torch.isfinite(weight_sum).item())) or float(weight_sum.item()) <= 0.0:
        raise ValueError("Spectral window produced a non-positive total weight.")
    loss = torch.sum(per_level * weights) / weight_sum
    mae = torch.sum(abs_err * weights) / weight_sum
    return loss, {
        "spectral_mae_ev": mae,
        "spectral_weight_sum": weight_sum,
    }
