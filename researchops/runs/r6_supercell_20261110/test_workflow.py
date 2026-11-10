"""Analytic contracts for folding, including non-Gamma phases and negative shifts."""
from itertools import product

import numpy as np
import pytest

from fold_reference import folded_keys


@pytest.mark.parametrize('repeat', [(2, 2, 2), (3, 3, 3), (2, 1, 1)])
def test_folding_preserves_cartesian_displacement_and_support(repeat):
    keys = [(x,y,z,i,j) for x,y,z in product(range(-2,3), repeat=3)
            for i,j in product(range(2), repeat=2)]
    tiles = list(product(*(range(n) for n in repeat)))
    folded = list(folded_keys(keys, 2, repeat))
    assert len(folded) == len(keys) * np.prod(repeat)
    assert len(set(key for key, _ in folded)) == len(folded)
    for (x,y,z,i,j), old in folded:
        target = np.array(tiles[j//2]) - tiles[i//2] + np.array((x,y,z))*repeat
        assert tuple(target) == keys[old][:3]
        assert i % 2 == keys[old][3] and j % 2 == keys[old][4]


@pytest.mark.parametrize('repeat', [(2,2,2), (3,3,3), (2,1,1)])
def test_folded_spectrum_is_union_of_primitive_bloch_spectra(repeat):
    keys = [(0,0,0,0,0), (1,0,0,0,0), (-1,0,0,0,0),
            (0,1,0,0,0), (0,-1,0,0,0), (0,0,1,0,0), (0,0,-1,0,0)]
    values = [2., -.3, -.3, -.2, -.2, -.1, -.1]
    k = np.array([.137, -.219, .087])
    dense = np.zeros((np.prod(repeat),)*2, dtype=complex)
    for (x,y,z,i,j), old in folded_keys(keys, 1, repeat):
        dense[i,j] += values[old] * np.exp(2j*np.pi*np.dot(k,(x,y,z)))
    expected = []
    for tile in product(*(range(n) for n in repeat)):
        primitive_k = (k + tile)/repeat
        expected.append(sum(v*np.exp(2j*np.pi*np.dot(primitive_k,key[:3]))
                            for key,v in zip(keys,values)).real)
    np.testing.assert_allclose(np.linalg.eigvalsh(dense), sorted(expected), atol=1e-13)
