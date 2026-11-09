from types import SimpleNamespace
import numpy as np
import pytest
from analysis.cp2k_gamma_density import (
    finite_gamma_density,
    map_density,
    gamma_template_layout,
    pack_gamma,
)
from data.cp2k_restart import CP2KDensityRestart, CP2KImageDensity


def test_source_gamma_uses_any_image_once_and_reports_missing_pairs():
    section = SimpleNamespace(
        label="Density matrix spin=0",
        blocks={
            (1, 1, 2): np.array([[0.4]]),
            (1, 1, 3): np.array([[0.4]]),
            (2, 2, 0): np.array([[0.6]]),
        },
    )
    dump = SimpleNamespace(sections=[section], block_sizes=[1, 1], atom_count=2)
    p, report = finite_gamma_density(dump)
    np.testing.assert_array_equal(p, np.diag([0.8, 1.2]))
    assert report["missing_atom_pairs"] == 2
    section.blocks[(1, 1, 3)][0, 0] = 0.3
    with pytest.raises(ValueError, match="repeated Gamma"):
        finite_gamma_density(dump)


def test_maps_are_covariant_and_do_not_round_fractional_density():
    rng = np.random.default_rng(91)
    p = np.diag([1.4, 0.6])
    s = np.diag([1.0, 2.0])
    b = np.array([[1.0, 0.2], [0.1, 0.7]])
    qs, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    qt, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    original, report = map_density(p, s, cross=b, electrons=2)
    rotated, _ = map_density(
        qs @ p @ qs.T, qt @ s @ qt.T, cross=qt @ b @ qs.T, electrons=2
    )
    np.testing.assert_allclose(rotated, qt @ original @ qt.T, atol=1e-12)
    assert abs(report["final_target_electrons"] - 2) < 1e-12
    direct, _ = map_density(p, np.eye(2), permutation=np.arange(2), electrons=2)
    np.testing.assert_allclose(direct, p, atol=1e-12)
    transformed, _ = map_density(
        qs @ p @ qs.T, qs @ np.eye(2) @ qs.T, permutation=np.arange(2), electrons=2
    )
    np.testing.assert_allclose(transformed, qs @ direct @ qs.T, atol=1e-12)


def test_whole_atom_native_mask_roundtrip_and_rotation_covariance():
    rng = np.random.default_rng(93)
    a = rng.normal(size=(4, 4))
    p = a @ a.T
    masks = np.array([[[1, 1], [1, 1]], [[0, 1], [0, 0]]], bool)

    def template(matrix):
        blocks = matrix.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3)
        values = [
            (blocks * m[:, :, None, None]).transpose(0, 2, 1, 3).reshape(4, 4)
            for m in masks
        ]
        return CP2KDensityRestart(
            2,
            4,
            1,
            1,
            np.ones(2, int),
            np.ones((1, 2), int),
            np.full((1, 1, 2), 2, int),
            [[CP2KImageDensity((i, 0, 0), v) for i, v in enumerate(values)]],
        )

    native = template(p)
    recovered, layout, _ = gamma_template_layout(native, block_size=2)
    np.testing.assert_allclose(recovered, p)
    result = pack_gamma(native, recovered, layout, block_size=2)
    for old, new in zip(native.spins[0], result.spins[0]):
        np.testing.assert_array_equal(old.matrix, new.matrix)
    q = np.zeros((4, 4))
    for i in (0, 2):
        q[i : i + 2, i : i + 2], _ = np.linalg.qr(rng.normal(size=(2, 2)))
    _, rotated_masks, _ = gamma_template_layout(template(q @ p @ q.T), block_size=2)
    np.testing.assert_array_equal(rotated_masks, masks)
    packed = pack_gamma(native, q @ p @ q.T, layout, block_size=2)
    for old, new in zip(native.spins[0], packed.spins[0]):
        np.testing.assert_allclose(new.matrix, q @ old.matrix @ q.T, atol=1e-12)


def test_raw_finite_matrix_is_not_silently_projected_to_a_psd_state():
    p = np.diag([-0.1, 2.1])
    result, report = map_density(p, np.eye(2), permutation=np.arange(2), electrons=2)
    np.testing.assert_allclose(result, p)
    assert report["metric_eigenvalue_min"] < 0


def test_native_storage_can_imply_a_hermitian_reverse():
    p = np.array([[1.0, 0.3], [0.0, 1.0]])
    native = CP2KDensityRestart(
        2,
        2,
        1,
        1,
        np.ones(2, int),
        np.ones((1, 2), int),
        np.ones((1, 1, 2), int),
        [[CP2KImageDensity((0, 0, 0), p)]],
    )
    full, masks, report = gamma_template_layout(native, block_size=1)
    np.testing.assert_array_equal(full, [[1.0, 0.3], [0.3, 1.0]])
    assert report["implied_reverse_pairs"] == 1
    packed = pack_gamma(native, full, masks, block_size=1)
    np.testing.assert_array_equal(packed.spins[0][0].matrix, p)
