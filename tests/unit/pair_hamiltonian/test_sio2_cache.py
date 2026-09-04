from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from pair_hamiltonian.hamgnn_sio2 import BOHR_TO_ANGSTROM
from pair_hamiltonian.sio2_cache import (
    build_structure_arrays,
    split_hash,
    validate_structure_shard,
    write_structure_shard,
)


def _packed(matrix: torch.Tensor) -> torch.Tensor:
    active = torch.tensor([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    result = torch.zeros(14, 14, dtype=torch.float32)
    result[active[:, None], active[None, :]] = matrix
    return result.reshape(-1)


@pytest.fixture
def two_atom_graph():
    generator = torch.Generator().manual_seed(123)
    onsite_o = torch.randn(13, 13, generator=generator)
    onsite_si = torch.randn(13, 13, generator=generator)
    forward = torch.randn(13, 13, generator=generator)
    return SimpleNamespace(
        z=torch.tensor([8, 14]),
        pos=torch.tensor([[0.0, 0.0, 0.0], [2.0 / BOHR_TO_ANGSTROM, 0.0, 0.0]]),
        cell=torch.eye(3).mul(10.0 / BOHR_TO_ANGSTROM).unsqueeze(0),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        inv_edge_idx=torch.tensor([1, 0]),
        nbr_shift=torch.zeros(2, 3),
        cell_shift=torch.zeros(2, 3, dtype=torch.long),
        hamiltonian=torch.stack(
            [
                _packed(onsite_o),
                _packed(onsite_si),
                _packed(forward),
                _packed(forward.T),
            ]
        ),
    )


@pytest.mark.unit
def test_build_structure_arrays_canonicalizes_pairs_and_targets(two_atom_graph):
    arrays = build_structure_arrays(
        two_atom_graph,
        descriptor_cutoff_angstrom=3.0,
        density_radial_count=1,
        density_l_max=1,
        hamiltonian_cutoff_angstrom=3.0,
    )
    assert arrays["descriptor"].shape == (2, 8)
    assert arrays["onsite_target_irreps_hartree"].shape == (2, 169)
    assert arrays["offsite_target_irreps_hartree"].shape == (1, 169)
    assert arrays["offsite_pair_type"].tolist() == [1]
    assert arrays["offsite_source"].tolist() == [0]
    assert arrays["offsite_target"].tolist() == [1]
    assert arrays["neighbor_center"].size == 2


@pytest.mark.unit
def test_structure_shard_is_atomic_and_metadata_validated(tmp_path, two_atom_graph):
    arrays = build_structure_arrays(
        two_atom_graph,
        descriptor_cutoff_angstrom=3.0,
        density_radial_count=1,
        density_l_max=1,
        hamiltonian_cutoff_angstrom=3.0,
    )
    path = tmp_path / "structure_0000.h5"
    metadata = {"version": "test", "cutoff": 3.0, "pair_names": ("O-O", "O-Si")}
    summary = write_structure_shard(
        path,
        arrays,
        structure_index=0,
        source_key=17,
        split="train",
        metadata=metadata,
    )
    assert summary.offsite_pair_count == 1
    assert validate_structure_shard(path, metadata)
    assert not validate_structure_shard(path, {"version": "different"})
    with h5py.File(path, "r") as handle:
        assert handle.attrs["complete"]


@pytest.mark.unit
def test_split_hash_depends_on_named_partitions():
    split = {
        "train": np.asarray([0, 1]),
        "validation": np.asarray([2]),
        "test": np.asarray([3]),
    }
    assert split_hash(split) == split_hash(
        {key: value.copy() for key, value in split.items()}
    )
    changed = {**split, "train": np.asarray([1, 0])}
    assert split_hash(changed) != split_hash(split)
