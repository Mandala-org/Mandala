"""Parent-cell categories, independent of geometry and perturbation size."""
from itertools import product

CATEGORIES = ('onsite', 'different_parent', 'same_parent_image')


def atom_mapping(parent_count, repetition):
    return [dict(source_atom_index=i, tile_index=t, tile=list(tile))
            for t, tile in enumerate(product(*(range(n) for n in repetition)))
            for i in range(parent_count)]


def category(edge, mapping):
    x, y, z, i, j = edge
    if i == j and (x, y, z) == (0, 0, 0):
        return 'onsite'
    if mapping[i]['source_atom_index'] == mapping[j]['source_atom_index']:
        return 'same_parent_image'
    return 'different_parent'


def category_errors(prediction, reference, mapping, hamiltonian_factor=None):
    """Partition directed common-support blocks; weight MAE by scalar entries.

    Hermitian projection belongs to the caller, matching the paper convention.
    Fractions refer to common support, with full support counts also retained.
    """
    totals = {kind: dict(blocks=0, scalar_entries=0, absolute_error_sum=0.)
              for kind in CATEGORIES}
    support = {label: {kind: dict(blocks=0, scalar_entries=0) for kind in CATEGORIES}
               for label in ('prediction', 'reference')}
    for label, matrix in (('prediction', prediction), ('reference', reference)):
        for edge, (pair, index) in matrix.lookup.items():
            counts = support[label][category(edge, mapping)]
            counts['blocks'] += 1
            counts['scalar_entries'] += matrix.pair_blocks[pair][index].numel()
    for edge, (pair, index) in prediction.lookup.items():
        if edge not in reference.lookup:
            continue
        target_pair, target_index = reference.lookup[edge]
        p = prediction.pair_blocks[pair][index].detach().cpu().double()
        r = reference.pair_blocks[target_pair][target_index].detach().cpu().double()
        counts = totals[category(edge, mapping)]
        counts['blocks'] += 1
        counts['scalar_entries'] += p.numel()
        counts['absolute_error_sum'] += float((p-r).abs().sum())
    blocks = sum(v['blocks'] for v in totals.values())
    entries = sum(v['scalar_entries'] for v in totals.values())
    if not entries:
        raise ValueError('No common support')
    for counts in totals.values():
        n = counts['scalar_entries']
        counts.update(block_fraction=counts['blocks']/blocks,
                      scalar_entry_fraction=n/entries,
                      mae=counts['absolute_error_sum']/n if n else None)
        if hamiltonian_factor is not None:
            counts['mae_meV'] = counts['mae']*hamiltonian_factor if n else None
    kept = [totals[k] for k in ('onsite', 'different_parent')]
    n = sum(v['scalar_entries'] for v in kept)
    excluded = dict(scalar_entries=n, mae=sum(v['absolute_error_sum'] for v in kept)/n if n else None)
    if hamiltonian_factor is not None:
        excluded['mae_meV'] = excluded['mae']*hamiltonian_factor if n else None
    for counts_by_kind in support.values():
        nb = sum(v['blocks'] for v in counts_by_kind.values())
        ne = sum(v['scalar_entries'] for v in counts_by_kind.values())
        for counts in counts_by_kind.values():
            counts.update(block_fraction=counts['blocks']/nb,
                          scalar_entry_fraction=counts['scalar_entries']/ne)
    return dict(categories=totals, excluding_same_parent_image=excluded,
                common_blocks=blocks, common_scalar_entries=entries, full_support=support)
