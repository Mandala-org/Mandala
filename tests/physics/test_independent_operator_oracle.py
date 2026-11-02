"""Noncommuting periodic operators checked without MANDALA's physics helpers."""

from collections import Counter
import math

import numpy as np
import pytest
from scipy.linalg import eigh
import torch

from core.orbital_irrep_config import OrbitalIrrepConfig
from core.periodic_fourier import shiftspace_to_kspace_dense
from data.block_matrix import BlockMatrix
from data.kspace_snapshot import (
    _generalized_eigenvalues_kspace,
    block_matrix_to_shiftspace_dense,
)
from data.snapshot import Snapshot
from net.spectral_loss import build_spectral_reference, compute_spectral_eigenvalue_loss
from tests.contract_support import double_precision
from utils.units import HARTREE_TO_EV


def _operators():
    # H(+L) is deliberately not self-adjoint: reversing Fourier's sign changes
    # the bands, and transposing without conjugating corrupts Hermiticity.
    h0 = np.array([[-0.7, 0.22 + 0.13j], [0.22 - 0.13j, 0.9]])
    hp = np.array([[0.12 + 0.04j, -0.18 + 0.07j], [0.09 - 0.11j, -0.08 + 0.02j]])
    s0 = np.array([[1.3, 0.17 - 0.06j], [0.17 + 0.06j, 1.1]])
    sp = np.array(
        [[0.025 + 0.005j, -0.013 + 0.018j], [0.009 + 0.004j, -0.017 - 0.003j]]
    )
    return np.stack([h0, hp, hp.conj().T]), np.stack([s0, sp, sp.conj().T])


SHIFTS = torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
BOX = torch.tensor(
    [[2.4, 0.3, -0.2], [0.1, 3.1, 0.4], [0.2, -0.1, 2.8]], dtype=torch.float64
)


def _bloch(values, fractions):
    """Literal scalar Fourier sum; no production phases or dense conversion."""
    return np.stack(
        [
            sum(
                values[r] * np.exp(2j * np.pi * f * shift)
                for r, shift in enumerate([0, 1, -1])
            )
            for f in fractions
        ]
    )


def _matrix(values):
    # Deliberately shuffle all directed records to exercise lookup/scatter.
    records = [(r, i, j) for r in range(3) for i in range(2) for j in range(2)]
    records = [records[k] for k in [7, 0, 11, 3, 8, 1, 4, 10, 2, 6, 9, 5]]
    edges = torch.tensor([[int(SHIFTS[r, 0]), 0, 0, i, j] for r, i, j in records]).T
    tensor = torch.as_tensor(values)
    blocks = torch.stack([tensor[r, i, j] for r, i, j in records]).reshape(-1, 1, 1)
    return BlockMatrix(
        ("H", "H"),
        Counter({"H": 2}),
        {"H-H": blocks},
        {"H-H": edges},
        {tuple(row): ("H-H", n) for n, row in enumerate(edges.T.tolist())},
        OrbitalIrrepConfig.from_dict({"H": "1s"}),
        "e3nn",
    )


def test_periodic_complex_fourier_generalized_bands_density_and_export(tmp_path):
    with double_precision():
        h, s = _operators()
        fractions = np.array([0.0, 1 / 3, 2 / 3])
        hk, sk = _bloch(h, fractions), _bloch(s, fractions)
        assert np.linalg.norm(hk @ sk - sk @ hk) > 0.1
        assert np.max(np.abs(hk.imag)) > 0.1
        assert np.min(np.linalg.eigvalsh(sk)) > 0.7
        eigenpairs = [eigh(a, b) for a, b in zip(hk, sk)]
        expected_bands = np.stack([e for e, _ in eigenpairs])
        occupations = np.array([1.6, 0.3])
        density_k = np.stack([(c * occupations) @ c.conj().T for _, c in eigenpairs])
        density_shift = np.stack(
            [
                sum(
                    density_k[k] * np.exp(-2j * np.pi * f * shift)
                    for k, f in enumerate(fractions)
                )
                / 3
                for shift in [0, 1, -1]
            ]
        )
        matrices = [_matrix(a) for a in [h, s, density_shift]]
        k_abs = torch.tensor(np.c_[fractions, fractions * 0, fractions * 0]) @ (
            2 * math.pi * torch.linalg.inv(BOX).T
        )
        actual_k = []
        for index, (matrix, expected) in enumerate(zip(matrices, [hk, sk, density_k])):
            # This is the physical native export/reload path, including complex
            # entries and all directed periodic blocks (OpenMX export is real-only).
            path = tmp_path / f"operator-{index}.pt"
            matrix.save(path)
            restored = BlockMatrix.load(path)
            shifted = block_matrix_to_shiftspace_dense(restored, shifts=SHIFTS)
            result = shiftspace_to_kspace_dense(
                shifted, kpoints_abs=k_abs, shifts=SHIFTS, box=BOX
            )
            torch.testing.assert_close(
                result, torch.tensor(expected), atol=3e-14, rtol=3e-14
            )
            torch.testing.assert_close(result, result.mH, atol=3e-14, rtol=3e-14)
            actual_k.append(result)
        bands = _generalized_eigenvalues_kspace(actual_k[0], actual_k[1])
        torch.testing.assert_close(
            bands, torch.tensor(expected_bands), atol=2e-14, rtol=2e-14
        )
        snapshot = Snapshot(*matrices, box=BOX)
        expected_energy = np.mean((expected_bands * occupations).sum(-1))
        torch.testing.assert_close(
            snapshot.get_number_of_electrons().real.squeeze(),
            torch.tensor(1.9),
            atol=2e-14,
            rtol=2e-14,
        )
        torch.testing.assert_close(
            snapshot.get_energy().real.squeeze(),
            torch.tensor(expected_energy),
            atol=2e-14,
            rtol=2e-14,
        )
        assert snapshot.get_energy().imag.abs().max() < 2e-14
        # A nonsymmetric raw perturbation must be projected with the reversed
        # periodic block's adjoint, rather than with its own transpose.
        raw = h.copy()
        raw[1, 0, 1] += 0.27 + 0.19j
        raw_matrix = _matrix(raw)
        projected = 0.5 * (raw_matrix + raw_matrix.adjoint())
        expected_projection = 0.5 * (raw + raw[[0, 2, 1]].conj().transpose(0, 2, 1))
        torch.testing.assert_close(
            block_matrix_to_shiftspace_dense(projected, shifts=SHIFTS),
            torch.tensor(expected_projection),
        )


@pytest.mark.parametrize("loss_kind", ["mse", "mae", "huber"])
def test_real_spectral_pipeline_matches_scipy_loss_and_complex_directional_derivative(
    loss_kind,
):
    with double_precision():
        h, s = _operators()
        reference = build_spectral_reference(
            hamiltonian=_matrix(h),
            overlap=_matrix(s),
            box=BOX,
            fermi_level_hartree=0.1,
            kmesh_spec="3x1x1",
            window_ev=100.0,
            taper_ev=0.0,
            overlap_psd_cleanup=False,
            overlap_jitter=False,
        )
        fractions = [0.0, 1 / 3, 2 / 3]
        sk = _bloch(s, fractions)
        ground = (
            np.stack(
                [
                    eigh(a, b, eigvals_only=True)
                    for a, b in zip(_bloch(h, fractions), sk)
                ]
            )
            * HARTREE_TO_EV
        )
        torch.testing.assert_close(
            reference["spectral_gt_eigs_ev"],
            torch.tensor(ground),
            atol=1e-12,
            rtol=1e-13,
        )
        direction = np.array(
            [
                [[0.3, 0.07 + 0.08j], [0.07 - 0.08j, -0.2]],
                [[0.04 + 0.01j, 0.02 - 0.03j], [0.06 + 0.01j, -0.01 + 0.02j]],
                [[0.04 - 0.01j, 0.06 - 0.01j], [0.02 + 0.03j, -0.01 - 0.02j]],
            ]
        )

        def oracle(alpha):
            pred = (
                np.stack(
                    [
                        eigh(a, b, eigvals_only=True)
                        for a, b in zip(_bloch(h + alpha * direction, fractions), sk)
                    ]
                )
                * HARTREE_TO_EV
            )
            err = pred - ground
            if loss_kind == "mse":
                return np.mean(err**2)
            if loss_kind == "mae":
                return np.mean(np.abs(err))
            a = np.abs(err)
            return np.mean(np.where(a <= 0.4, 0.5 * a * a, 0.4 * (a - 0.2)))

        def actual(alpha):
            return compute_spectral_eigenvalue_loss(
                pred_hamiltonian=_matrix(
                    torch.tensor(h) + alpha * torch.tensor(direction)
                ),
                spectral_payload=reference,
                box=BOX,
                loss_kind=loss_kind,
                huber_delta_ev=0.4,
                overlap_psd_cleanup=False,
                overlap_jitter=False,
            )[0]

        alpha = torch.tensor(0.17, requires_grad=True)
        loss = actual(alpha)
        assert loss > 1e-3
        assert loss.item() == pytest.approx(oracle(0.17), rel=2e-12, abs=2e-13)
        gradient = torch.autograd.grad(loss, alpha, create_graph=True)[0]
        step = 1e-5
        numerical = (oracle(0.17 + step) - oracle(0.17 - step)) / (2 * step)
        assert abs(numerical) > 1e-3
        assert gradient.item() == pytest.approx(numerical, rel=2e-7, abs=2e-8)
        curvature = torch.autograd.grad(gradient, alpha)[0]
        fd_curvature = (
            oracle(0.17 + step) - 2 * oracle(0.17) + oracle(0.17 - step)
        ) / step**2
        assert curvature.item() == pytest.approx(fd_curvature, rel=4e-3, abs=3e-4)
        # Cached and uncached execution must BOTH meet the independent oracle.
        uncached = {
            k: v
            for k, v in reference.items()
            if k
            not in {
                "spectral_gt_overlap_cholesky",
                "spectral_phase",
                "spectral_scatter_metadata",
                "spectral_dense_shape",
            }
        }
        plain, _ = compute_spectral_eigenvalue_loss(
            pred_hamiltonian=_matrix(h + 0.17 * direction),
            spectral_payload=uncached,
            box=BOX,
            loss_kind=loss_kind,
            huber_delta_ev=0.4,
            overlap_psd_cleanup=False,
            overlap_jitter=False,
        )
        assert plain.item() == pytest.approx(oracle(0.17), rel=2e-12, abs=2e-13)


def test_indefinite_overlap_is_rejected_and_cleanup_requires_explicit_policy():
    h, _ = _operators()
    hk = torch.tensor(_bloch(h, [0.0]))
    bad = torch.tensor([[[1.0, 0.2j], [-0.2j, -0.1]]], dtype=torch.complex128)
    with pytest.raises(torch.linalg.LinAlgError):
        _generalized_eigenvalues_kspace(hk, bad)
    # Explicit cleanup has a precise floor and a separately calculated spectrum.
    val, vec = np.linalg.eigh(bad.numpy()[0])
    repaired = (vec * np.maximum(val, 1e-6)) @ vec.conj().T
    expected = eigh(hk.numpy()[0], repaired, eigvals_only=True)
    actual = _generalized_eigenvalues_kspace(hk, bad, psd_cleanup=True)
    torch.testing.assert_close(actual[0], torch.tensor(expected), rtol=3e-10, atol=2e-5)
