"""Compare all available finished OpenMX outputs and folded controls to saved predictions.

Run inside a compute allocation, e.g. sbatch researchops/runs/r6_supercell_20261110/compare.sh.
Missing and incomplete outputs are recorded; no model is loaded and no DFT is launched.
"""
import argparse
import traceback

from workflow import *
from pair_categories import category_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--reuse-diagnostics', action='store_true',
                        help='Reuse stored diagnostics only for byte-identical operators and references')
    args = parser.parse_args()
    previous_path = RUN/'results/comparison.json'
    previous = {s['name']: s for s in json.loads(previous_path.read_text())['structures']} if previous_path.exists() else {}
    torch.set_num_threads(4)
    from compare_calibration import main as refresh_calibration
    refresh_calibration()
    calibration = json.loads((RUN/'results/runtime_calibration.json').read_text())
    result = dict(validation_H_MAE_meV=dict(silicon=0.435164,sio2=1.99588),
                  validation_source='docs/data/physical_constraints_20261104/summary_projected.md; Hermitian projection on reference support',
                  sio2_control_caveat=manifest()['sio2_control_caveat'],
                  runtime_calibration=calibration,structures=[])
    lines = ['Element errors use matching periodic support; raw and Hermitian-projected values are in JSON.',
             'The table uses Hermitian projection to match the paper validation values.',
             manifest()['sio2_control_caveat'], '',
             '| Structure | Atoms | H MAE meV | S MAE | D MAE | Predicted usable S fraction | Reference usable S fraction | Reference min eigenvalue | Validation H meV |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    failed = any(row.get('status')=='ok' and not row['passed_H_tolerance'] for row in calibration.values())
    lines.insert(2, 'Public OpenMX runtime source-cell checks: '+', '.join(
        f"{material}: "+(f"H MAE {row['H_MAE_meV']:.6g} meV, pass={row['passed_H_tolerance']}" if row['status']=='ok' else row['status'])
        for material,row in calibration.items()))
    for structure in manifest()['structures']:
        row = dict(name=structure['name'],atoms=structure['atom_count'])
        result['structures'].append(row)
        try:
            predicted = load_predictions(structure['name'])
            if not predicted:
                row['status'] = 'prediction_pending'
                continue
            if structure['tiled_control']:
                target = tiled_reference(structure)
                row['reference_kind'] = 'folded control; conditional k convergence for SiO2'
            else:
                directory = Path(structure['dft_directory'])
                if not (directory/'HS.out').is_file() or not (directory/'COMPLETE').is_file():
                    row['status'] = 'OpenMX_pending'
                    continue
                _,target = reference(directory)
                row['reference_kind'] = 'new self-consistent OpenMX'
                row['reference_HS_sha256'] = sha256(directory/'HS.out')
                row['runtime_source_check'] = calibration[structure['material']]
            row['operator_hashes'] = {name: sha256(ARTIFACTS/'predictions'/structure['name']/(name+'.pt')) for name in predicted}
            row['reference_identity'] = (dict(source_HS_sha256=sha256(Path(structure['source_cell'])/'HS.out'), repetition=structure['repetition'])
                                         if structure['tiled_control'] else row['reference_HS_sha256'])
            old = previous.get(structure['name'], {})
            reuse = (args.reuse_diagnostics and old.get('status') == 'ok'
                     and old.get('operator_hashes') == row['operator_hashes']
                     and old.get('reference_identity') == row['reference_identity']
                     and old.get('predicted', {}).get('mesh') == structure['diagnostic_mesh']
                     and old.get('predicted', {}).get('nominal_electrons') == structure['num_electrons'])
            row['category_errors'] = {name: {
                'raw_common_support': category_errors(a, target[name], structure['atom_parent_mapping'], HARTREE_TO_EV*1000 if name=='hamiltonian' else None),
                'hermitian_common_support': category_errors((a+a.adjoint())*.5, target[name], structure['atom_parent_mapping'], HARTREE_TO_EV*1000 if name=='hamiltonian' else None)}
                for name, a in predicted.items()}
            row.update(status='ok', element_errors=errors(predicted,target),
                       reference=old['reference'] if reuse else diagnostics(target,structure,args.device))
            # Main inference diagnostics are on the full geometry graph. Also report
            # a reference-support control comparable to the paper's diagnostics.
            row['predicted'] = old['predicted'] if reuse else diagnostics(predicted,structure,args.device)
            row['prediction_on_reference_support'] = old['prediction_on_reference_support'] if reuse else diagnostics(
                {name:restrict_support(a,target[name]) for name,a in predicted.items()},structure,args.device)
            e = row['element_errors']
            values = [structure['name'],str(structure['atom_count']),f"{e['hamiltonian']['hermitian_common_support']['mae_meV']:.6g}"]
            values += [f"{e[name]['hermitian_common_support']['mae']:.6g}" if name in e else '—' for name in ('overlap','density')]
            def metric(group,key,which='mean'):
                value = group['summary']['metrics'].get(key,{}).get(which)
                return f'{value:.6g}' if value is not None else '—'
            values += [metric(row['predicted'],'overlap.well_conditioned_positive_definite'),
                       metric(row['reference'],'overlap.well_conditioned_positive_definite'),
                       metric(row['reference'],'overlap.lambda_min','min'),
                       str(result['validation_H_MAE_meV'][structure['material']])]
            lines.append('| '+' | '.join(values)+' |')
        except Exception:
            failed = True
            row.update(status='error',traceback=traceback.format_exc())
        # Save partial progress so a long diagnostic run remains reviewable.
        write_json(RUN/'results/comparison.json',result)
        (RUN/'results/comparison.md').write_text('\n'.join(lines)+'\n')
    write_json(RUN/'results/comparison.json',result)
    (RUN/'results/comparison.md').write_text('\n'.join(lines)+'\n')
    # Also refresh summaries from already-queued launchers (Slurm snapshots scripts).
    from summarize import main as summarize
    summarize()
    print('\n'.join(lines))
    if failed:
        raise SystemExit('Some comparisons failed; see comparison.json')


if __name__ == '__main__':
    main()
