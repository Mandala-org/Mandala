"""Fold primitive real-space blocks into an ASE tile-major diagonal supercell.

For a source tile t and primitive edge (R,i,j), destination tile is
u=(t+R) mod repetition, and supercell shift is floor((t+R)/repetition).
This changes the unit cell without changing the infinite periodic operator.
"""
from collections import Counter, defaultdict
from itertools import product

import numpy as np


def folded_keys(keys, atom_count, repetition):
    repetition = np.asarray(repetition, dtype=int)
    if repetition.shape != (3,) or min(repetition) < 1:
        raise ValueError('Expected three positive integer repetitions')
    tiles = list(product(*(range(int(n)) for n in repetition)))
    tile_index = {tile: i for i, tile in enumerate(tiles)}
    for tile_id, tile in enumerate(tiles):
        for old_index, (x, y, z, i, j) in enumerate(keys):
            dest = np.array(tile) + (x, y, z)
            target_tile = tuple(dest % repetition)
            shift = tuple(int(v) for v in dest // repetition)
            yield (*shift, tile_id * atom_count + i,
                   tile_index[target_tile] * atom_count + j), old_index


def fold_matrix(matrix, repetition):
    import torch
    from data.block_matrix import BlockMatrix

    groups = defaultdict(list)
    for pair, edges in matrix.pair_edges.items():
        for new_key, index in folded_keys(edges.T.tolist(), len(matrix.atoms), repetition):
            groups[pair].append((new_key, index))
    edges, blocks, coverage, lookup = {}, {}, {}, {}
    for pair, rows in groups.items():
        keys, indices = zip(*rows)
        edges[pair] = torch.tensor(keys, dtype=torch.long).T
        indices = torch.tensor(indices, dtype=torch.long, device=matrix.pair_blocks[pair].device)
        blocks[pair] = matrix.pair_blocks[pair][indices].cpu()
        coverage[pair] = matrix.coverage_masks[pair][indices].cpu()
        for index, key in enumerate(keys):
            if key in lookup:
                raise ValueError('Folding produced a duplicate edge')
            lookup[key] = pair, index
    atoms = matrix.atoms * int(np.prod(repetition))
    return BlockMatrix(atoms, Counter(atoms), blocks, edges, lookup,
                       matrix.orbital_cfg, matrix.basis, _coverage_masks=coverage)
