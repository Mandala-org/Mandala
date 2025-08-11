import pytest
from data.snapshot import Snapshot
from pathlib import Path
import numpy as np

def _load_snapshot():
    base = Path("data/big/silicon/2700K")
    matrix_path = base / "Si_DM"
    info_path = base / "info.txt"
    snap = Snapshot.from_openmx(
            str(matrix_path),
            str(info_path),
            convention="openmx",
        )
    return snap

@pytest.mark.unit
def test_dos_return_shape():
    snap = _load_snapshot()

    sigma = 0.005
    bin_width = 0.001
    E_min = -1.0
    E_max = 1.0
    dos = snap.dos(sigma, bin_width, E_min, E_max)
    expected_length = int((E_max - E_min) / bin_width) + 1

    assert np.shape(dos["energies"]) == (expected_length,)
    assert np.shape(dos["dos"]) == (expected_length,)