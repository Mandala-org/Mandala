"""Analytic contracts for folding, including non-Gamma phases and negative shifts."""
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

from fold_reference import folded_keys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))


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


def test_native_block_folding_roundtrip_and_reverse_edges(tmp_path):
    from collections import Counter
    import torch
    from core.orbital_irrep_config import OrbitalIrrepConfig
    from data.block_matrix import BlockMatrix
    from fold_reference import fold_matrix
    from analysis.physical_constraints import BlockFourier

    cfg = OrbitalIrrepConfig.from_dict({'H': ['1x0e']})
    keys = [(0,0,0,0,0),(1,0,0,0,0),(-1,0,0,0,0)]
    matrix = BlockMatrix(('H',),Counter(H=1),
                         {'H-H':torch.tensor([2.,-.3,-.3],dtype=torch.float64).reshape(3,1,1)},
                         {'H-H':torch.tensor(keys).T},
                         {key:('H-H',i) for i,key in enumerate(keys)},cfg,'e3nn')
    folded = fold_matrix(matrix,(2,1,1))
    assert folded.atoms == ('H','H') and folded.atom_counts == {'H':2}
    folded.save(tmp_path/'folded.pt')
    loaded = BlockMatrix.load(tmp_path/'folded.pt')
    assembler = BlockFourier(loaded)
    assert assembler.reverse_missing == 0
    values = torch.linalg.eigvalsh(assembler.at([.17,0,0]))
    expected = sorted(2-.6*np.cos(2*np.pi*(.17+i)/2) for i in (0,1))
    np.testing.assert_allclose(values.numpy(),expected,atol=1e-14)


def test_parent_categories_include_shifted_images_and_weight_scalars():
    from types import SimpleNamespace
    import torch
    from pair_categories import atom_mapping, category, category_errors

    mapping = atom_mapping(2, (2,1,1))
    assert mapping[2] == dict(source_atom_index=0, tile_index=1, tile=[1,0,0])
    keys = [(0,0,0,0,0), (0,0,0,0,1), (0,0,0,0,2), (1,0,0,0,0)]
    assert [category(k, mapping) for k in keys] == [
        'onsite', 'different_parent', 'same_parent_image', 'same_parent_image']
    # Unequal block sizes distinguish scalar weighting from block averaging.
    blocks = {str(i): torch.full((1,1,n), float(i+1)) for i,n in enumerate((1,2,3,4))}
    lookup = {key:(str(i),0) for i,key in enumerate(keys)}
    p = SimpleNamespace(lookup=lookup, pair_blocks=blocks)
    r = SimpleNamespace(lookup=lookup, pair_blocks={k:torch.zeros_like(v) for k,v in blocks.items()})
    result = category_errors(p,r,mapping,1000.)
    image = result['categories']['same_parent_image']
    assert image['block_fraction'] == .5
    assert image['scalar_entry_fraction'] == .7
    assert image['mae'] == pytest.approx(25/7)
    assert result['excluding_same_parent_image']['mae_meV'] == pytest.approx(5000/3)
    assert sum(v['absolute_error_sum'] for v in result['categories'].values()) == 30
