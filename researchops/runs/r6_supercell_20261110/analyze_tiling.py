"""Attribute tiled-control changes by primitive edge class; never alter predictions."""
from collections import defaultdict
from workflow import *
from fold_reference import folded_keys


def main():
    torch.set_num_threads(4)
    results = []
    for structure in manifest()['structures']:
        if not structure['tiled_control']:
            continue
        current = load_predictions(structure['name'])
        if not current:
            continue
        base_directory = ARTIFACTS/'historical_base_control'/structure['material']
        row = dict(name=structure['name'], operators={})
        for name,predicted in current.items():
            original = BlockMatrix.load(base_directory/(name+'.pt'))
            folded = fold_matrix(original,structure['repetition'])
            keys_by_class = defaultdict(set)
            for edges in original.pair_edges.values():
                keys = [tuple(key) for key in edges.T.tolist()]
                for new,idx in folded_keys(keys,len(original.atoms),structure['repetition']):
                    old = keys[idx]
                    kind = 'onsite' if old[3] == old[4] and old[:3] == (0,0,0) else (
                        'primitive_shifted_self' if old[3] == old[4] else 'primitive_different_atoms')
                    keys_by_class[kind].add(new)
            differences = {}
            for kind,keys in keys_by_class.items():
                error = count = max_error = 0
                for key in keys:
                    if key not in predicted.lookup:
                        continue
                    delta = (predicted[key].double()-folded[key].double()).abs()
                    error += float(delta.sum())
                    count += delta.numel()
                    max_error = max(max_error,float(delta.max()))
                if count:
                    differences[kind] = dict(mae=error/count,max_absolute=max_error,
                                              common_scalar_entries=count,
                                              mae_meV=error/count*HARTREE_TO_EV*1000 if name=='hamiltonian' else None)
            row['operators'][name] = differences
        results.append(row)
    write_json(RUN/'results/tiling_attribution.json',results)
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
