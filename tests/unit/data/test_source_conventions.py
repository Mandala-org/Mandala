"""Independent marker oracles for imported orbital order and Cartesian axes."""

import json

import h5py
import numpy as np
import pytest
import torch

from data.snapshot import Snapshot
from data.deeph_e3_parser import DeepHE3ParseError
from data.sources import resolve_snapshot_source, openmx_info_path
from utils.units import HARTREE_TO_EV


def write_snapshot(root, *, basis="openmx", axes="xyz", block=None, metadata=True):
    root.mkdir(parents=True, exist_ok=True)
    np.savetxt(root / "element.dat", [6], fmt="%d")
    (root / "orbital_types.dat").write_text("0 1 2 3\n")
    np.savetxt(root / "site_positions.dat", [[1], [2], [3]])
    np.savetxt(root / "lat.dat", np.diag([7, 8, 9]))
    info = {"isspinful": False}
    if metadata:
        info.update(orbital_basis=basis, cartesian_axes=axes)
    (root / "info.json").write_text(json.dumps(info))
    if block is None:
        block = np.diag(np.arange(1, 17))
    with h5py.File(root / "hamiltonians.h5", "w") as stream:
        stream["[0, 0, 0, 1, 1]"] = block * HARTREE_TO_EV
    return block


def test_native_marker_permutation_and_geometry_happen_exactly_once(tmp_path):
    write_snapshot(tmp_path)
    source = resolve_snapshot_source(tmp_path)
    assert source.format == "periodic_hdf5"
    raw = Snapshot.from_deeph_e3(tmp_path, convention="openmx", dtype=torch.float64)
    canonical = Snapshot.from_deeph_e3(tmp_path, convention="e3nn", dtype=torch.float64)
    # Independent published OpenMX-to-Wikipedia permutations for s,p,d,f.
    expected = [1, 3, 4, 2, 7, 9, 5, 8, 6, 16, 14, 12, 10, 11, 13, 15]
    torch.testing.assert_close(
        canonical.hamiltonian["C-C"][0].diag(),
        torch.tensor(expected, dtype=torch.float64),
    )
    assert canonical.positions.tolist() == [[2, 3, 1]]
    assert canonical.to_e3nn() is canonical
    torch.testing.assert_close(
        raw.to_e3nn().hamiltonian["C-C"], canonical.hamiltonian["C-C"]
    )
    restored = canonical.to_openmx()
    torch.testing.assert_close(restored.hamiltonian["C-C"], raw.hamiltonian["C-C"])
    torch.testing.assert_close(restored.positions, raw.positions)
    torch.testing.assert_close(restored.box, raw.box)


def test_already_converted_hdf5_does_not_permute_orbitals_again(tmp_path):
    order = [0, 2, 3, 1, 6, 8, 4, 7, 5, 15, 13, 11, 9, 10, 12, 14]
    expected = np.diag(np.arange(1, 17))[order][:, order]
    write_snapshot(tmp_path, basis="e3nn", axes="xyz", block=expected)
    snapshot = Snapshot.from_deeph_e3(tmp_path, dtype=torch.float64)
    torch.testing.assert_close(
        snapshot.hamiltonian["C-C"][0], torch.tensor(expected, dtype=torch.float64)
    )
    assert snapshot.positions.tolist() == [[2, 3, 1]]
    assert snapshot.to_e3nn() is snapshot
    with pytest.raises(DeepHE3ParseError, match="conflicts"):
        Snapshot.from_deeph_e3(tmp_path, source_basis="openmx")


def test_untagged_published_hdf5_defaults_to_raw_openmx(tmp_path):
    write_snapshot(tmp_path, metadata=False)
    snapshot = Snapshot.from_deeph_e3(tmp_path, dtype=torch.float64)
    assert snapshot.positions.tolist() == [[2, 3, 1]]
    assert snapshot.info.source_orbital_basis == "openmx"


def test_raw_openmx_metadata_discovery_supports_material_names(tmp_path):
    (tmp_path / "HS.out").touch()
    info = tmp_path / "graphene.out"
    info.touch()
    assert openmx_info_path(tmp_path) == info
    assert resolve_snapshot_source(tmp_path).format == "openmx"
    (tmp_path / "other.out").touch()
    with pytest.raises(ValueError, match="Ambiguous"):
        resolve_snapshot_source(tmp_path)


def test_pyscf_directory_uses_unambiguous_metadata(tmp_path):
    np.savez(tmp_path / "matrices.npz", matrix=[1])
    (tmp_path / "info.json").write_text("{}")
    source = resolve_snapshot_source(tmp_path)
    assert source.format == "pyscf"
    assert source.info_path.name == "info.json"
