import pytest
from data.snapshot import Snapshot
from data.block_matrix import BlockMatrix
from core.orbital_irrep_config import OrbitalIrrepConfig
from collections import Counter
import torch


@pytest.fixture(scope="module")
def snapshot():
    orbital_cfg = OrbitalIrrepConfig.from_dict({"H": "2s"})
    atoms = ("H",)
    edges = torch.zeros((5, 1), dtype=torch.long)
    lookup = {(0, 0, 0, 0, 0): ("H-H", 0)}

    def matrix(block):
        return BlockMatrix(
            atoms=atoms,
            atom_counts=Counter(atoms),
            pair_blocks={"H-H": block.unsqueeze(0)},
            pair_edges={"H-H": edges.clone()},
            lookup=dict(lookup),
            orbital_cfg=orbital_cfg,
            basis="e3nn",
        )

    return Snapshot(
        hamiltonian=matrix(torch.diag(torch.tensor([-0.25, 0.35]))),
        overlap=matrix(torch.eye(2)),
        density=matrix(torch.eye(2)),
        positions=torch.zeros(1, 3),
        box=torch.eye(3) * 10.0,
    )


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
