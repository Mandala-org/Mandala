import numpy as np
import pytest

from analysis.cp2k_gamma_compare import (
    occupied_from_hamiltonian,
    project_occupied,
    printed_gamma_overlap,
    gaussian_gamma_overlap,
)
from analysis.cp2k_projection import gaussian_pao_cross_overlap_numpy


def test_occupied_projection_retains_count_and_is_basis_covariant():
    rng = np.random.default_rng(14)
    a = rng.normal(size=(6, 6))
    s = a @ a.T + np.eye(6)
    h = rng.normal(size=(6, 6))
    h = (h + h.T) / 2
    c, _ = occupied_from_hamiltonian(h, s, 4)
    x = rng.normal(size=(7, 7))
    t = x @ x.T + np.eye(7)
    b = rng.normal(size=(7, 6))
    projected, report = project_occupied(c, t, b)
    np.testing.assert_allclose(projected.T @ t @ projected, np.eye(2), atol=1e-12)
    assert abs(report["final_electrons"] - 4) < 1e-12
    q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    rotated, _ = project_occupied(c, q @ t @ q.T, q @ b)
    np.testing.assert_allclose(
        rotated @ rotated.T, q @ projected @ projected.T @ q.T, atol=2e-12
    )


def test_projection_handles_discarded_target_directions_and_rank_failure():
    s = np.diag([1.0, 2.0, 1e-8])
    c = np.eye(2)
    result, report = project_occupied(
        c, s, np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    )
    assert report["target_directions_discarded"] == 1
    np.testing.assert_allclose(result.T @ s @ result, np.eye(2), atol=1e-12)
    with pytest.raises(ValueError, match="Too few"):
        project_occupied(np.eye(3), s, np.eye(3))


def test_gamma_print_reader_with_two_column_sections(tmp_path):
    path = tmp_path / "cp2k.out"
    path.write_text(
        "noise\n OVERLAP MATRIX\n 1 2\n 1 1 Si 2s 1.0 0.2\n 2 1 Si 3s 0.2 1.0\n"
    )
    np.testing.assert_array_equal(
        printed_gamma_overlap(path, 2), [[1.0, 0.2], [0.2, 1.0]]
    )


def test_element_specific_aliases_preserve_gaussian_numbers(tmp_path):
    from scripts.prepare_cp2k_method_comparison import label_cp2k_basis
    from analysis.cp2k_projection import read_cp2k_shells

    source = tmp_path / "library"
    source.write_text(
        "Si SHARED other\n1\n2 0 0 1 1\n.3 1.2\n"
        "O SHARED other\n1\n2 0 0 1 1\n.7 2.3\n"
        "X UNRELATED\n1\n6s 0 0 1 1\n.7 2.3\n"
    )
    target = tmp_path / "labelled"
    aliases = label_cp2k_basis(source, target, {"Si": "SHARED", "O": "SHARED"})
    assert aliases["Si"] != aliases["O"]
    assert "UNRELATED" not in target.read_text()
    for element in aliases:
        original = read_cp2k_shells(source, element, "SHARED")
        labelled = read_cp2k_shells(target, element, aliases[element])
        for old, new in zip(original, labelled):
            assert old[0] == new[0]
            np.testing.assert_array_equal(old[1], new[1])
            np.testing.assert_array_equal(old[2], new[2])


def test_periodic_gaussian_contractions_match_numerical_source_integrals():
    pytest.importorskip("pyscf")
    shells = [(l, np.array([0.3, 0.7]), np.array([0.8, 0.2])) for l in [0, 0, 1, 1, 2]]
    # Distinct s/p contractions, independently normalized in both backends.
    shells[1] = (0, np.array([0.3, 0.7]), np.array([-0.2, 0.8]))
    shells[3] = (1, np.array([0.3, 0.7]), np.array([-0.2, 0.8]))
    normalized = []
    for l, a, c in shells:
        g = (2 * np.sqrt(a[:, None] * a[None, :]) / (a[:, None] + a[None, :])) ** (
            l + 1.5
        )
        normalized.append((l, a, c / np.sqrt(c @ g @ c)))
    s = gaussian_gamma_overlap(
        ("Si",), np.zeros((1, 3)), np.eye(3) * 40, {"Si": normalized}
    )
    import math

    x, w = np.polynomial.legendre.leggauss(180)
    r = (x + 1) * 7
    w = w * 7
    radial = []
    for l in range(3):
        values = []
        for ll, a, c in normalized:
            if ll == l:
                n = np.sqrt(2 * (2 * a) ** (l + 1.5) / math.gamma(l + 1.5))
                values.append((np.exp(-r[:, None] ** 2 * a) * r[:, None] ** l * n) @ c)
        radial.append((l, np.stack(values, axis=1)))
    source = gaussian_pao_cross_overlap_numpy(
        np.zeros((1, 3)), normalized, r, w, radial
    )[0]
    source = source[:, [0, 1, 3, 4, 2, 6, 7, 5, 10, 12, 8, 11, 9]]
    np.testing.assert_allclose(s, source, atol=2e-11)
