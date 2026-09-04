import numpy as np
import pytest
import torch

from pair_hamiltonian.hamgnn_sio2 import (
    published_protocol_split,
    unpack_active_blocks,
)


@pytest.mark.unit
def test_published_protocol_split_is_deterministic_and_complete():
    first = published_protocol_split(630, seed=42)
    second = published_protocol_split(630, seed=42)
    assert np.array_equal(first.train, second.train)
    assert (len(first.train), len(first.validation), len(first.test)) == (504, 63, 63)
    assert sorted(
        np.concatenate([first.train, first.validation, first.test]).tolist()
    ) == list(range(630))


@pytest.mark.unit
def test_unpack_active_blocks_removes_openmx_padding_index_two():
    matrix = torch.arange(14 * 14, dtype=torch.float64).reshape(1, -1)
    actual = unpack_active_blocks(matrix, torch.tensor([8]), torch.tensor([14]))
    active = torch.tensor([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    expected = matrix.reshape(14, 14).index_select(0, active).index_select(1, active)
    assert torch.equal(actual[0], expected)


@pytest.mark.unit
def test_unpack_active_blocks_rejects_mixed_pairs():
    with pytest.raises(ValueError, match="homogeneous"):
        unpack_active_blocks(
            torch.zeros(2, 196), torch.tensor([8, 14]), torch.tensor([14, 14])
        )
