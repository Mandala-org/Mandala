import math

import numpy as np
import pytest
from scipy.linalg import block_diag
import torch

from analysis.cp2k_projection import (
    _solid_harmonics,
    gaussian_pao_cross_overlap,
    infer_image_layout,
    metric_project_density,
    pack_images,
    unpack_images,
    gaussian_pao_cross_overlap_numpy,
)


def gaussian_source(exponent):
    x, w = np.polynomial.legendre.leggauss(160)
    r = (x + 1) * 6
    weights = w * 6
    radials = []
    for l in range(3):
        normalization = np.sqrt(2 * (2 * exponent) ** (l + 1.5) / math.gamma(l + 1.5))
        radials.append((l, (normalization * r**l * np.exp(-exponent * r * r))[:, None]))
    return r, weights, radials


def test_pao_cross_integrals_against_independent_gaussian_integrals():
    pyscf = pytest.importorskip("pyscf")
    a, b = 0.6, 0.9
    shells = [(l, np.array([a]), np.ones(1)) for l in range(3)]
    d = np.array([[0.0, 0.0, 0.0], [0.4, -0.7, 1.1], [-1.3, 0.2, -0.6]])
    result = gaussian_pao_cross_overlap(d, shells, *gaussian_source(b))
    for k, center in enumerate(d):
        mol = pyscf.gto.M(
            atom=[("H1", (0, 0, 0)), ("H2", center)],
            unit="Bohr",
            basis={
                "H1": [[l, [a, 1.0]] for l in range(3)],
                "H2": [[l, [b, 1.0]] for l in range(3)],
            },
            verbose=0,
        )
        expected = mol.intor("int1e_ovlp")[:9, 9:]
        expected = expected[[0, 2, 3, 1, 4, 5, 6, 7, 8]][:, [0, 1, 2, 3, 6, 8, 4, 7, 5]]
        np.testing.assert_allclose(result[k], expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("parity", [1, -1])
def test_cross_projection_has_full_rotation_and_reflection_covariance(parity):
    rng = np.random.default_rng(73)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation) * parity
    points = rng.normal(size=(40, 3))
    points /= np.linalg.norm(points, axis=1)[:, None]
    original = _solid_harmonics(torch.tensor(points))
    rotated = _solid_harmonics(torch.tensor(points @ rotation.T))
    ds = [
        np.linalg.lstsq(y.numpy(), yr.numpy(), rcond=None)[0].T
        for y, yr in zip(original, rotated)
    ]
    ds = block_diag(*ds)
    permutation = np.array([0, 2, 3, 1, 6, 8, 4, 7, 5])
    dt = ds[permutation][:, permutation]
    shells = [(l, np.array([0.6]), np.ones(1)) for l in range(3)]
    center = np.array([[0.4, -0.7, 1.1]])
    s = gaussian_pao_cross_overlap(center, shells, *gaussian_source(0.9))[0]
    sr = gaussian_pao_cross_overlap(center @ rotation.T, shells, *gaussian_source(0.9))[
        0
    ]
    np.testing.assert_allclose(sr, dt @ s @ ds.T, rtol=5e-10, atol=5e-11)


def test_native_image_layout_preserves_directed_offsite_values():
    cells = [(1, 0, 0), (-1, 0, 0)]
    positive = np.array([[0.2, 0.3], [0.5, 0.1]])
    physical = {cells[0]: positive, cells[1]: positive.T}
    stored = np.array([[[0.2, 0.3], [0.3, 0.1]], [[0.2, 0.5], [0.5, 0.1]]])
    layout, error = infer_image_layout(stored, physical, cells, block_size=1)
    assert error == 0
    assert layout[0, 1, 0] == -1
    restored = unpack_images(stored, cells, layout, block_size=1)
    for cell in cells:
        np.testing.assert_array_equal(restored[cell], physical[cell])
    np.testing.assert_array_equal(
        pack_images(restored, cells, layout, block_size=1), stored
    )
    with pytest.raises(ValueError, match="Uncertified"):
        infer_image_layout(stored + 0.02, physical, cells, block_size=1)


def test_metric_projection_count_pauli_bounds_and_basis_covariance():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(2, 4, 4)) + 1j * rng.normal(size=(2, 4, 4))
    s = a @ a.swapaxes(-1, -2).conj() + np.eye(4)
    p = a + a.swapaxes(-1, -2).conj()
    cross = 0.1 * a
    result, diagnostics = metric_project_density(p, s, cross, 3)
    assert abs(diagnostics["repaired_electrons"] - 3) < 1e-11
    v, u = np.linalg.eigh(s)
    root = (u * np.sqrt(v)[:, None, :]) @ u.swapaxes(-1, -2).conj()
    occ = np.linalg.eigvalsh(root @ result @ root)
    assert occ.min() >= -1e-12 and occ.max() <= 2 + 1e-12
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    transformed, _ = metric_project_density(p, q @ s @ q.conj().T, q @ cross, 3)
    np.testing.assert_allclose(transformed, q @ result @ q.conj().T, atol=1e-11)


def test_native_text_diagonal_triangle_does_not_symmetrize_physical_image():
    cells = [(1, 0, 0), (-1, 0, 0)]
    a = np.array([[0.3, 0.2], [-0.2, 0.1]])
    physical = {cells[0]: a, cells[1]: a.T}
    printed = np.stack([np.triu(x) + np.triu(x, 1).T for x in (a, a.T)])
    with pytest.raises(ValueError, match="Uncertified"):
        infer_image_layout(printed, physical, cells, block_size=2)
    layout, error = infer_image_layout(
        printed, physical, cells, block_size=2, printed_diagonal_triangle=True
    )
    assert error == 0
    restored = unpack_images(np.stack([a, a.T]), cells, layout, block_size=2)
    for cell in cells:
        np.testing.assert_array_equal(restored[cell], physical[cell])


def test_explicit_radial_derivatives_match_ad_integrals():
    shells = [(l, np.array([0.3, 0.9]), np.array([0.7, -0.2])) for l in range(3)]
    d = np.array(
        [[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0], [0.4, -0.7, 1.1], [8.0, 1.0, -3.0]]
    )
    expected = gaussian_pao_cross_overlap(d, shells, *gaussian_source(0.7))
    actual = gaussian_pao_cross_overlap_numpy(d, shells, *gaussian_source(0.7))
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-11)
