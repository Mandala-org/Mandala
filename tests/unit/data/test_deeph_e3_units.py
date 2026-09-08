import json

import h5py
import numpy as np
import pytest
import torch

from data.deeph_e3_parser import (
    load_deeph_e3_block_matrix,
    load_deeph_e3_metadata,
)
from utils.units import HARTREE_TO_EV
from data.snapshot import Snapshot


def test_deeph_e3_units_are_converted_to_mandala_internal_units(tmp_path):
    np.savetxt(tmp_path / "element.dat", np.array([6]), fmt="%d")
    (tmp_path / "orbital_types.dat").write_text("0\n")
    np.savetxt(tmp_path / "site_positions.dat", np.array([[1.0], [2.0], [3.0]]))
    raw_lattice_columns = np.array([[4.0, 0.1, 0.2], [0.3, 5.0, 0.4], [0.5, 0.6, 6.0]])
    np.savetxt(tmp_path / "lat.dat", raw_lattice_columns)
    (tmp_path / "info.json").write_text(
        json.dumps({"fermi_level": -2.0 * HARTREE_TO_EV, "isspinful": False})
    )
    with h5py.File(tmp_path / "hamiltonians.h5", "w") as handle:
        handle.create_dataset("[0, 0, 0, 1, 1]", data=[[3.0 * HARTREE_TO_EV]])

    metadata = load_deeph_e3_metadata(tmp_path, dtype=torch.float64)
    matrix = load_deeph_e3_block_matrix(
        tmp_path / "hamiltonians.h5",
        metadata,
        matrix_name="hamiltonian",
        dtype=torch.float64,
    )

    assert metadata.positions.tolist() == [[1.0, 2.0, 3.0]]
    assert torch.equal(metadata.box, torch.tensor(raw_lattice_columns.T))
    assert metadata.fermi_level.item() == pytest.approx(-2.0)
    assert matrix.pair_blocks["C-C"][0, 0, 0].item() == pytest.approx(3.0)

    info = metadata.as_snapshot_info()
    assert info.source_energy_unit == "eV"
    assert info.internal_energy_unit == "Hartree"
    assert info.source_length_unit == "Angstrom"
    assert info.internal_length_unit == "Angstrom"

    # Processed OpenMX and text OpenMX must use the same density convention.
    with h5py.File(tmp_path / "density_matrixs.h5", "w") as handle:
        handle.create_dataset("[0, 0, 0, 1, 1]", data=[[0.75]])
    for symmetrize in (False, True):
        snapshot = Snapshot.from_deeph_e3(
            tmp_path,
            convention="e3nn",
            symmetrize_density=symmetrize,
        )
        assert snapshot.density["C-C"].item() == pytest.approx(0.75)
