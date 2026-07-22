from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.kspace_snapshot import (
    KSpaceMatrix,
    KSpaceSnapshot,
    load_pyscf_kspace_snapshot,
)


def _toy_shift_matrices() -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    box = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32))
    shifts = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.long)
    ham_shift = torch.tensor(
        [
            [[1.0, 0.2], [0.2, 1.5]],
            [[0.4, -0.1], [0.3, 0.6]],
        ],
        dtype=torch.float32,
    )
    kpoints_abs = torch.tensor(
        [[0.0, 0.0, 0.0], [math.pi / 2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    return box, shifts, ham_shift, kpoints_abs


@pytest.mark.unit
def test_kspace_matrix_roundtrip_from_shiftspace():
    box, shifts, ham_shift, kpoints_abs = _toy_shift_matrices()
    cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})

    kmat = KSpaceMatrix.from_shiftspace_dense(
        ham_shift,
        atoms=("H", "H"),
        orbital_cfg=cfg,
        basis="pyscf",
        kpoints_abs=kpoints_abs,
        box=box,
        shifts=shifts,
        kmesh=(2, 1, 1),
    )
    shifts_back, ham_shift_back = kmat.to_shiftspace_dense()

    assert torch.equal(shifts_back, shifts)
    assert torch.allclose(ham_shift_back.imag, torch.zeros_like(ham_shift), atol=1e-6)
    assert torch.allclose(ham_shift_back.real, ham_shift, atol=1e-6)


@pytest.mark.unit
def test_snapshot_to_kspace_roundtrip_preserves_blocks():
    box, shifts, ham_shift, kpoints_abs = _toy_shift_matrices()
    cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})
    ovl_shift = torch.stack([torch.eye(2), torch.eye(2) * 0.25], dim=0)
    den_shift = torch.stack([torch.eye(2) * 1.5, torch.eye(2) * 0.05], dim=0)

    ksnap = KSpaceSnapshot(
        hamiltonian=KSpaceMatrix.from_shiftspace_dense(
            ham_shift,
            atoms=("H", "H"),
            orbital_cfg=cfg,
            basis="pyscf",
            kpoints_abs=kpoints_abs,
            box=box,
            shifts=shifts,
            kmesh=(2, 1, 1),
        ),
        overlap=KSpaceMatrix.from_shiftspace_dense(
            ovl_shift,
            atoms=("H", "H"),
            orbital_cfg=cfg,
            basis="pyscf",
            kpoints_abs=kpoints_abs,
            box=box,
            shifts=shifts,
            kmesh=(2, 1, 1),
        ),
        density=KSpaceMatrix.from_shiftspace_dense(
            den_shift,
            atoms=("H", "H"),
            orbital_cfg=cfg,
            basis="pyscf",
            kpoints_abs=kpoints_abs,
            box=box,
            shifts=shifts,
            kmesh=(2, 1, 1),
        ),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        box=box,
    )

    snap = ksnap.to_shift_space()
    ksnap_back = snap.to_k_space(kpoints_abs=kpoints_abs, shifts=shifts)
    snap_back = ksnap_back.to_shift_space()

    assert torch.allclose(
        snap.hamiltonian[(1, 0, 0, 0, 1)],
        snap_back.hamiltonian[(1, 0, 0, 0, 1)],
        atol=1e-6,
    )
    assert torch.allclose(
        snap.overlap[(0, 0, 0, 1, 1)],
        snap_back.overlap[(0, 0, 0, 1, 1)],
        atol=1e-6,
    )
    assert torch.allclose(
        snap.density[(1, 0, 0, 1, 0)],
        snap_back.density[(1, 0, 0, 1, 0)],
        atol=1e-6,
    )


@pytest.mark.unit
def test_load_pyscf_kspace_snapshot_from_shift_artifacts(tmp_path: Path):
    atoms = [
        {
            "index": 1,
            "element": "H",
            "x_angstrom": 0.0,
            "y_angstrom": 0.0,
            "z_angstrom": 0.0,
        },
        {
            "index": 2,
            "element": "H",
            "x_angstrom": 1.0,
            "y_angstrom": 0.0,
            "z_angstrom": 0.0,
        },
    ]
    box, shifts, ham_shift, kpoints_abs = _toy_shift_matrices()
    overlap_shift = torch.stack([torch.eye(2), torch.eye(2) * 0.5], dim=0)
    density_shift = torch.stack([torch.eye(2) * 1.2, torch.eye(2) * 0.1], dim=0)

    npz_path = tmp_path / "sample.npz"
    json_path = tmp_path / "sample.json"
    np.savez_compressed(
        npz_path,
        hamiltonian_ao=ham_shift[0].numpy(),
        overlap_ao=overlap_shift[0].numpy(),
        dm_ao=density_shift[0].numpy(),
        hamiltonian_shifted=ham_shift.numpy(),
        overlap_shifted=overlap_shift.numpy(),
        density_shifted=density_shift.numpy(),
        shifts=shifts.numpy(),
        kpts_abs=kpoints_abs.numpy(),
        kmesh=np.array([2, 1, 1], dtype=np.int64),
        box_angstrom=box.numpy(),
        positions_angstrom=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
    )
    json_path.write_text(
        json.dumps(
            {
                "snapshot": {"atoms": atoms, "box_angstrom": box.tolist()},
                "settings": {"orbital_set": {"H": "1s"}, "kmesh": [2, 1, 1]},
            }
        )
    )

    ksnap = load_pyscf_kspace_snapshot(npz_path, json_path=json_path)
    snap = ksnap.to_shift_space()

    assert ksnap.kmesh == (2, 1, 1)
    shifted_block = snap.hamiltonian[(1, 0, 0, 0, 1)]
    assert torch.allclose(shifted_block.imag, torch.zeros_like(shifted_block.imag))
    assert torch.allclose(
        shifted_block.real,
        torch.tensor([[-0.1]], dtype=torch.float32),
        atol=1e-6,
    )
