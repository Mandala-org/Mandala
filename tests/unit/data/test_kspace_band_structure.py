from __future__ import annotations

import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from data.kspace_snapshot import KSpaceMatrix, KSpaceSnapshot


def test_kspace_snapshot_get_band_structure_generalized_eigenvalues():
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})
    kpoints_abs = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    box = torch.eye(3, dtype=torch.float32)
    hamiltonian_k = torch.tensor(
        [
            [[1.0]],
            [[2.0]],
        ],
        dtype=torch.float32,
    )
    overlap_k = torch.tensor(
        [
            [[1.0]],
            [[2.0]],
        ],
        dtype=torch.float32,
    )
    density_k = torch.zeros_like(hamiltonian_k)

    ham = KSpaceMatrix(
        atoms=("H",),
        atom_counts={"H": 1},
        matrices_k=hamiltonian_k,
        kpoints_abs=kpoints_abs,
        orbital_cfg=orbital_cfg,
        basis="e3nn",
        box=box,
    )
    ovl = KSpaceMatrix(
        atoms=("H",),
        atom_counts={"H": 1},
        matrices_k=overlap_k,
        kpoints_abs=kpoints_abs,
        orbital_cfg=orbital_cfg,
        basis="e3nn",
        box=box,
    )
    den = KSpaceMatrix(
        atoms=("H",),
        atom_counts={"H": 1},
        matrices_k=density_k,
        kpoints_abs=kpoints_abs,
        orbital_cfg=orbital_cfg,
        basis="e3nn",
        box=box,
    )
    snap = KSpaceSnapshot(ham, ovl, den, box=box)

    payload = snap.get_band_structure(
        tick_labels=["G", "X"],
        tick_positions=torch.tensor([0.0, 0.5], dtype=torch.float32),
        chunk_size=1,
    )

    assert payload.eigenvalues.shape == (2, 1)
    assert torch.allclose(payload.eigenvalues.squeeze(-1), torch.tensor([1.0, 1.0]))
    assert payload.tick_labels == ["G", "X"]
