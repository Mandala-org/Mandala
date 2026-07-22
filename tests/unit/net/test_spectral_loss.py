import pytest
import torch
from collections import Counter

import net.spectral_loss as spectral_loss
from core.orbital_irrep_config import OrbitalIrrepConfig
from core.periodic_fourier import phase_matrix, shiftspace_to_kspace_dense
from data.block_matrix import BlockMatrix
from data.kspace_snapshot import (
    _generalized_eigenvalues_from_cholesky,
    _prepare_overlap_cholesky_kspace,
    block_matrix_to_shiftspace_dense,
    block_matrix_to_shiftspace_dense_aligned,
    build_shiftspace_scatter_metadata,
)
from net.spectral_loss import compute_spectral_eigenvalue_loss
from utils.units import HARTREE_TO_EV


def _payload():
    return {
        "spectral_kpoints_abs": torch.zeros(1, 3),
        "spectral_shifts": torch.zeros(1, 3, dtype=torch.long),
        "spectral_gt_overlap_k": torch.eye(2).unsqueeze(0),
        "spectral_gt_eigs_ev": torch.tensor([[0.0, 2.0]]),
        "spectral_gt_fermi_ev": torch.tensor(0.0),
        "spectral_window_weights": torch.tensor([[1.0, 3.0]]),
    }


@pytest.mark.parametrize(
    ("loss_kind", "expected"),
    [("mae", 1.75), ("mse", 3.25), ("huber", 0.17)],
)
@pytest.mark.unit
def test_spectral_loss_kinds_always_use_ground_truth_overlap(
    monkeypatch, loss_kind, expected
):
    payload = _payload()
    seen = {}
    monkeypatch.setattr(
        spectral_loss,
        "block_matrix_to_shiftspace_dense",
        lambda pred, shifts: torch.zeros(1),
    )
    monkeypatch.setattr(
        spectral_loss,
        "shiftspace_to_kspace_dense",
        lambda shift, **kwargs: torch.zeros(1),
    )

    def fake_eigenvalues(hamiltonian, overlap, **kwargs):
        seen["overlap"] = overlap
        return torch.tensor([[1.0, 4.0]]) / HARTREE_TO_EV

    monkeypatch.setattr(
        spectral_loss, "_generalized_eigenvalues_kspace", fake_eigenvalues
    )

    loss, stats = compute_spectral_eigenvalue_loss(
        pred_hamiltonian=object(),
        spectral_payload=payload,
        box=torch.eye(3),
        loss_kind=loss_kind,
        huber_delta_ev=0.1,
        overlap_psd_cleanup=True,
        overlap_jitter=False,
    )

    assert seen["overlap"] is payload["spectral_gt_overlap_k"]
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert stats["spectral_mae_ev"].item() == pytest.approx(1.75)


@pytest.mark.unit
def test_precomputed_spectral_operators_match_legacy_outputs_and_gradients():
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})
    edges = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=torch.long,
    ).T
    values = torch.tensor([1.0, 0.2, 0.2, 1.4], dtype=torch.float64)

    def _matrix(block_values: torch.Tensor) -> BlockMatrix:
        return BlockMatrix(
            atoms=("H", "H"),
            atom_counts=Counter(("H", "H")),
            pair_blocks={"H-H": block_values.reshape(-1, 1, 1)},
            pair_edges={"H-H": edges},
            lookup={
                tuple(edge): ("H-H", idx) for idx, edge in enumerate(edges.T.tolist())
            },
            orbital_cfg=orbital_cfg,
            basis="e3nn",
        )

    shifts = torch.zeros(1, 3, dtype=torch.long)
    metadata, dense_shape = build_shiftspace_scatter_metadata(
        _matrix(values), shifts=shifts
    )
    legacy_values = values.clone().requires_grad_(True)
    aligned_values = values.clone().requires_grad_(True)
    legacy_dense = block_matrix_to_shiftspace_dense(
        _matrix(legacy_values), shifts=shifts
    )
    aligned_dense = block_matrix_to_shiftspace_dense_aligned(
        _matrix(aligned_values),
        scatter_metadata=metadata,
        dense_shape=dense_shape,
    )
    assert torch.equal(legacy_dense, aligned_dense)
    legacy_grad = torch.autograd.grad(legacy_dense.square().sum(), legacy_values)[0]
    aligned_grad = torch.autograd.grad(aligned_dense.square().sum(), aligned_values)[0]
    assert torch.equal(legacy_grad, aligned_grad)

    box = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64))
    kpoints = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    precomputed_phase = phase_matrix(kpoints, shifts, box)
    legacy_k = shiftspace_to_kspace_dense(
        legacy_dense.detach(), kpoints_abs=kpoints, shifts=shifts, box=box
    )
    cached_k = shiftspace_to_kspace_dense(
        legacy_dense.detach(),
        kpoints_abs=kpoints,
        shifts=shifts,
        box=box,
        phase=precomputed_phase,
    )
    assert torch.equal(legacy_k, cached_k)

    overlap = torch.tensor([[[1.2, 0.1], [0.1, 1.1]]], dtype=torch.complex128)
    hamiltonian_values = legacy_k.clone().requires_grad_(True)
    cached_values = legacy_k.clone().requires_grad_(True)
    cholesky = _prepare_overlap_cholesky_kspace(overlap)
    tmp = torch.linalg.solve(cholesky, hamiltonian_values)
    transformed = (
        torch.linalg.solve(cholesky, tmp.transpose(-1, -2).conj())
        .transpose(-1, -2)
        .conj()
    )
    expected_eigs = torch.linalg.eigvalsh(
        0.5 * (transformed + transformed.transpose(-1, -2).conj())
    )
    actual_eigs = _generalized_eigenvalues_from_cholesky(cached_values, cholesky)
    assert torch.allclose(actual_eigs, expected_eigs, atol=1e-12, rtol=1e-12)
    expected_grad = torch.autograd.grad(expected_eigs.sum(), hamiltonian_values)[0]
    actual_grad = torch.autograd.grad(actual_eigs.sum(), cached_values)[0]
    assert torch.allclose(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)
