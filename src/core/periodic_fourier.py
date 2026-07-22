from __future__ import annotations

from typing import Sequence

import torch

__all__ = [
    "translation_shifts_for_kmesh",
    "phase_matrix",
    "kspace_to_shiftspace_dense",
    "shiftspace_to_kspace_dense",
]


def translation_shifts_for_kmesh(
    kmesh: Sequence[int],
    *,
    wrap_around: bool = True,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if len(kmesh) != 3:
        raise ValueError(f"kmesh must have 3 entries, got {kmesh}")
    kx, ky, kz = (int(kmesh[0]), int(kmesh[1]), int(kmesh[2]))
    if min(kx, ky, kz) <= 0:
        raise ValueError(f"kmesh entries must be positive, got {kmesh}")

    ax = torch.arange(kx, dtype=torch.long, device=device)
    ay = torch.arange(ky, dtype=torch.long, device=device)
    az = torch.arange(kz, dtype=torch.long, device=device)
    if wrap_around:
        ax = ax.clone()
        ay = ay.clone()
        az = az.clone()
        ax[(kx + 1) // 2 :] -= kx
        ay[(ky + 1) // 2 :] -= ky
        az[(kz + 1) // 2 :] -= kz

    mx, my, mz = torch.meshgrid(ax, ay, az, indexing="ij")
    return torch.stack([mx.reshape(-1), my.reshape(-1), mz.reshape(-1)], dim=1)


def phase_matrix(
    kpoints_abs: torch.Tensor,
    shifts: torch.Tensor,
    box: torch.Tensor,
) -> torch.Tensor:
    if kpoints_abs.ndim != 2 or kpoints_abs.shape[1] != 3:
        raise ValueError(
            f"kpoints_abs must have shape (Nk,3), got {tuple(kpoints_abs.shape)}"
        )
    if shifts.ndim != 2 or shifts.shape[1] != 3:
        raise ValueError(
            f"shifts must have shape (Nshift,3), got {tuple(shifts.shape)}"
        )
    if box.shape != (3, 3):
        raise ValueError(f"box must have shape (3,3), got {tuple(box.shape)}")

    device = kpoints_abs.device
    dtype = kpoints_abs.dtype
    shifts_cart = shifts.to(device=device, dtype=dtype) @ box.to(
        device=device, dtype=dtype
    )
    phase_arg = shifts_cart @ kpoints_abs.T
    return torch.exp(
        1j
        * phase_arg.to(torch.complex64 if dtype == torch.float32 else torch.complex128)
    )


def kspace_to_shiftspace_dense(
    matrices_k: torch.Tensor,
    *,
    kpoints_abs: torch.Tensor,
    shifts: torch.Tensor,
    box: torch.Tensor,
) -> torch.Tensor:
    if matrices_k.ndim != 3:
        raise ValueError(
            f"matrices_k must have shape (Nk,nao,nao), got {tuple(matrices_k.shape)}"
        )
    nk = matrices_k.shape[0]
    if kpoints_abs.shape[0] != nk:
        raise ValueError(
            f"kpoint count mismatch: matrices have Nk={nk}, kpoints have {kpoints_abs.shape[0]}"
        )
    phase = phase_matrix(kpoints_abs, shifts, box)
    mats = matrices_k.to(phase.dtype)
    out = torch.einsum("rk,kij->rij", phase.conj(), mats) / float(nk)
    return out.real if not torch.is_complex(matrices_k) else out


def shiftspace_to_kspace_dense(
    matrices_shift: torch.Tensor,
    *,
    kpoints_abs: torch.Tensor,
    shifts: torch.Tensor,
    box: torch.Tensor,
    phase: torch.Tensor | None = None,
) -> torch.Tensor:
    if matrices_shift.ndim != 3:
        raise ValueError(
            "matrices_shift must have shape (Nshift,nao,nao), "
            f"got {tuple(matrices_shift.shape)}"
        )
    if shifts.shape[0] != matrices_shift.shape[0]:
        raise ValueError(
            "shift count mismatch: "
            f"matrices have {matrices_shift.shape[0]}, shifts have {shifts.shape[0]}"
        )
    if phase is None:
        phase = phase_matrix(kpoints_abs, shifts, box)
    elif phase.shape != (shifts.shape[0], kpoints_abs.shape[0]):
        raise ValueError(
            "phase must have shape (Nshift,Nk), got "
            f"{tuple(phase.shape)} for Nshift={shifts.shape[0]}, "
            f"Nk={kpoints_abs.shape[0]}"
        )
    mats = matrices_shift.to(phase.dtype)
    out = torch.einsum("rk,rij->kij", phase, mats)
    return out
