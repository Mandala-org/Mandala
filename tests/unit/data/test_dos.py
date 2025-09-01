import pytest
from data.snapshot import Snapshot
from pathlib import Path
import torch


@pytest.fixture(scope="module")
def snapshot():
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
def test_dos_return_shape(snapshot):
    sigma = 0.005
    bin_width = 0.001
    E_min = -1.0
    E_max = 1.0

    grid, dos = snapshot.dos(sigma, bin_width, E_min, E_max)
    expected_length = int((E_max - E_min) / bin_width) + 1

    assert grid.shape == (expected_length,)
    assert dos.shape == (expected_length,)


@pytest.mark.unit
def test_number_states(snapshot):
    sigma = 0.005
    bin_width = 0.001
    # wide energy range to include all states, for the integration to be accurate
    E_min = -100
    E_max = 100

    grid, dos = snapshot.dos(sigma, bin_width, E_min, E_max)
    H = snapshot.hamiltonian.to_dense().detach()

    # integral over the DOS should be equal to number of states (number of eigenvalues => size of Hamiltonian)
    expected_n_states = H.shape[0]
    dos_n_states = torch.trapezoid(dos, grid)
    assert (
        torch.abs(1 - dos_n_states / expected_n_states) < 0.01
    )  # error due to Gaussian broadening, tolerance of 1%
