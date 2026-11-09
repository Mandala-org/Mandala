"""Analytic counterexamples and nonorthogonal/periodic diagnostic contracts."""

from collections import Counter

import pytest
import torch

from analysis.physical_constraints import (
    BlockFourier,
    diagnose_kpoint,
    fermi_consistency,
    uniform_kpoints,
)
from core.orbital_irrep_config import OrbitalIrrepConfig
from data.block_matrix import BlockMatrix


def operators(occupations=(2.0, 0.0)):
    s = torch.tensor([[1.4, 0.3], [0.3, 0.9]], dtype=torch.complex128)
    e, u = torch.linalg.eigh(s)
    root, inv = (u * e.sqrt()) @ u.mH, (u * e.rsqrt()) @ u.mH
    h = root @ torch.diag(torch.tensor([-0.4, 0.7], dtype=s.dtype)) @ root
    d = inv @ torch.diag(torch.tensor(occupations, dtype=s.dtype)) @ inv
    return h, s, d


def test_exact_nonorthogonal_projector_and_commutator():
    result = diagnose_kpoint(*operators())
    assert result["density"]["occupation_min"] == pytest.approx(0, abs=2e-15)
    assert result["density"]["occupation_max"] == pytest.approx(2)
    assert result["density"]["electron_trace"] == pytest.approx(2)
    assert result["density"]["zero_temperature_idempotency_relative"] < 1e-14
    assert result["consistency"]["commutator_relative"] < 1e-14


def test_commutation_and_correct_charge_do_not_imply_correct_occupations():
    result = diagnose_kpoint(*operators((-0.1, 2.1)))
    assert result["density"]["electron_trace"] == pytest.approx(2)
    assert result["density"]["negative_count"] == 1
    assert result["density"]["overoccupied_count"] == 1
    inverted = diagnose_kpoint(*operators((0.0, 2.0)))
    assert inverted["consistency"]["commutator_relative"] < 1e-14
    fd = fermi_consistency([inverted], num_electrons=2, temperature_kelvin=300)
    assert fd["density_relative_frobenius"] == pytest.approx(2**0.5)


def test_fractional_occupations_are_valid_even_when_not_idempotent():
    result = diagnose_kpoint(*operators((1.8, 0.2)))
    assert (
        result["density"]["negative_count"]
        == result["density"]["overoccupied_count"]
        == 0
    )
    assert result["density"]["zero_temperature_idempotency_relative"] > 0.05


def test_raw_asymmetry_is_preserved_before_projection_and_invalid_s_is_not_repaired():
    h, s, d = operators()
    h[0, 1] += 0.1j
    result = diagnose_kpoint(h, s, d)
    assert result["raw_hermiticity"]["hamiltonian"]["max_absolute"] == pytest.approx(
        0.1
    )
    invalid = diagnose_kpoint(h, torch.diag(torch.tensor([1.0, -0.01])), d)
    assert invalid["status"] == "invalid_overlap_metric"
    assert invalid["overlap"]["cholesky_failed_without_jitter"]
    assert "density" not in invalid


def test_covariance_under_complex_nonorthogonal_basis_change_and_energy_shift():
    h, s, d = operators((1.4, 0.3))
    d[0, 1] += 0.07j
    d[1, 0] -= 0.07j
    a = torch.tensor([[1.1, 0.3j], [0.2, 0.8]], dtype=h.dtype)
    ai = torch.linalg.inv(a)
    before = diagnose_kpoint(h, s, d)
    after = diagnose_kpoint(a.mH @ (h + 13 * s) @ a, a.mH @ s @ a, ai @ d @ ai.mH)
    for key in (
        "occupation_min",
        "occupation_max",
        "electron_trace",
        "zero_temperature_idempotency_relative",
    ):
        assert after["density"][key] == pytest.approx(before["density"][key], abs=1e-13)
    assert after["consistency"]["commutator_relative"] == pytest.approx(
        before["consistency"]["commutator_relative"], abs=1e-13
    )


def test_one_global_fermi_level_uses_all_kpoints():
    records = []
    for energies in ([-1.0, 0.1], [-0.1, 1.0]):
        f = torch.tensor(energies, dtype=torch.float64)
        d = torch.diag(2 * torch.sigmoid(-f / (300 * 3.166811563e-6)))
        records.append(diagnose_kpoint(torch.diag(f), torch.eye(2), d))
    result = fermi_consistency(records, num_electrons=2, temperature_kelvin=300)
    assert result["chemical_potential_hartree"] == pytest.approx(0, abs=1e-10)
    assert result["reconstructed_electron_count"] == pytest.approx(2)
    assert result["density_relative_frobenius"] < 2e-8


def test_periodic_bloch_sum_has_complex_reverse_phases():
    cfg = OrbitalIrrepConfig.from_dict({"H": "1s"})
    edges = torch.tensor([[0, 1, -1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    matrix = BlockMatrix(
        atoms=("H",),
        atom_counts=Counter(H=1),
        pair_edges={"H-H": edges},
        pair_blocks={"H-H": torch.tensor([[[2.0]], [[0.3]], [[0.5]]])},
        lookup={tuple(e): ("H-H", i) for i, e in enumerate(edges.T.tolist())},
        orbital_cfg=cfg,
        basis="e3nn",
    )
    fourier = BlockFourier(matrix)
    assert fourier.reverse_missing == 0
    value = fourier.at([0.25, 0, 0])
    assert complex(value[0, 0]) == pytest.approx(2 - 0.2j)
    assert uniform_kpoints((1, 1, 1)) == [[0, 0, 0]]


def test_input_metadata_uses_declared_charge_not_reference_density(tmp_path):
    from scripts.report.evaluate_physical_constraints import input_metadata

    (tmp_path / "conf.dat").write_text(
        """scf.SpinPolarization off
scf.ElectronicTemperature 300
scf.Kgrid 8 8 8
scf.system.charge 1
<Atoms.SpeciesAndCoordinates
1 Si 0 0 0 2 2
Atoms.SpeciesAndCoordinates>
"""
    )
    result = input_metadata(tmp_path / "Si.out")
    assert result["num_electrons"] == 3
    assert result["native_kmesh"] == [8, 8, 8]


def test_paper_split_preserves_the_historical_metadata_whitelist(tmp_path):
    from scripts.report.evaluate_physical_constraints import select_pairs

    root = tmp_path / "scale_1"
    for index in range(81):
        directory = root / f"sample_{index:03d}"
        directory.mkdir(parents=True)
        (directory / "HS.out").write_text("matrix")
        (directory / "Si.out").write_text("metadata")
    failed = root / "failed_empty"
    failed.mkdir()
    for name in ("HS.out", "log.out", "slurm-1.out"):
        (failed / name).touch()
    validation, manifest = select_pairs("silicon", tmp_path)
    assert len(validation) == 1
    assert len(manifest["discovered"]) == 81
    assert manifest["excluded_without_publication_metadata"] == [str(failed)]
