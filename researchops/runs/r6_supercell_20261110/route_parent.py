"""Diagnostic only: route pairs with known parent identities through original heads.

Parent labels are construction metadata, unavailable for a general unseen cell.
No training, parameter changes, operator repair, or primary prediction changes.
"""
from workflow import *
from data.structure_inference import build_model_input_from_structure
from net.evaluation import _move_to_device
from pair_categories import category_errors


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    model, cfg, provenance = load_model('silicon', 'cuda')
    report = dict(label='Diagnostic requiring known source-cell parent identities',
                  learned_parameters_changed=False, applied_to_primary_predictions=False,
                  provenance=provenance, structures=[])
    for structure in manifest()['structures']:
        if structure['material'] != 'silicon':
            continue
        atoms, positions, box = load_paper_cif(RUN/'structures'/structure['cif'], dtype=cfg.dtype)
        x = build_model_input_from_structure(atoms=atoms, positions=positions, box=box,
                                             cfg=cfg, mapper=model.mapper)
        x = _move_to_device(x, torch.device('cuda'))
        parents = torch.tensor([a['source_atom_index'] for a in structure['atom_parent_mapping']], device='cuda')
        routed_counts = {}
        for pair, parts in x['edge_partitions'].items():
            edges = x['pred_pair_edges_static'][pair]
            diagonal = (edges[3] == edges[4]) & (edges[:3] == 0).all(0)
            same_parent = parents[edges[3]] == parents[edges[4]]
            parts['shifted_self_local_idx'] = torch.where(same_parent & ~diagonal)[0]
            parts['offdiag_local_idx'] = torch.where(~same_parent)[0]
            routed_counts[pair] = len(parts['shifted_self_local_idx'])
        predicted = {name: a.to('cpu') for name, a in model.predict_matrices(x, physical=True).items()}
        directory = ARTIFACTS/'diagnostic_parent_routed'/structure['name']
        directory.mkdir(parents=True, exist_ok=True)
        for name, matrix in predicted.items():
            matrix.save(directory/(name+'.pt'))
        target = tiled_reference(structure) if structure['tiled_control'] else reference(structure['dft_directory'])[1]
        row = dict(name=structure['name'], shifted_self_blocks=routed_counts,
                   element_errors=errors(predicted, target),
                   predicted=diagnostics(predicted, structure, 'cuda'),
                   categories={name: category_errors((a+a.adjoint())*.5, target[name],
                               structure['atom_parent_mapping'], HARTREE_TO_EV*1000 if name=='hamiltonian' else None)
                               for name, a in predicted.items()},
                   operator_files={name: dict(path=str(directory/(name+'.pt')), sha256=sha256(directory/(name+'.pt')))
                                   for name in predicted})
        report['structures'].append(row)
        write_json(RUN/'results/parent_routing.json', report)
        print(structure['name'], row['element_errors'], flush=True)


if __name__ == '__main__':
    main()
