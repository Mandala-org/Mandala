"""Structure-only inference with explicit historical-FiLM replay and a frozen-source control."""
import argparse
import subprocess
import sys
import time

from workflow import *
from data.structure_inference import build_model_input_from_structure, load_structure_from_cif
from net.evaluation import _move_to_device


@torch.no_grad()
def predict(model, atoms, positions, box, device):
    x = build_model_input_from_structure(atoms=atoms, positions=positions, box=box,
                                         cfg=model.cfg, mapper=model.mapper)
    return {name: a.to('cpu') for name,a in model.predict_matrices(
        _move_to_device(x, torch.device(device)), physical=True).items()}


def historical_control(model, material, cfg, device):
    """Check the structure-only path against the paper's frozen dataset inference."""
    source = next(s['source_cell'] for s in manifest()['structures'] if s['material'] == material)
    snapshot, target = reference(source, cfg)
    current = predict(model, snapshot.hamiltonian.atoms, snapshot.positions, snapshot.box, device)
    output = ARTIFACTS / 'historical_base_control' / material
    if not (output / 'export.json').exists():
        command = [sys.executable, str(ROOT / 'scripts/report/export_paper_operators.py'),
                   '--source-root', str(PAPER / 'historical_source'),
                   '--checkpoint', str(BIG / 'models/checkpoints' / CHECKPOINTS[material]),
                   '--matrix-path', str(Path(source) / 'HS.out'), '--info-path', str(snapshot.info_path),
                   '--output-dir', str(output), '--device', device]
        if material == 'sio2':
            command += ['--envelope-path', str(ENVELOPE)]
        subprocess.run(command, check=True)
    frozen = {name: BlockMatrix.load(output / (name+'.pt')) for name in current}
    differences = errors(current, frozen)
    # Require identical support and agreement far below the published element MAE.
    for name,row in differences.items():
        raw = row['raw_common_support']
        if raw['prediction_only_blocks'] or raw['reference_only_blocks'] or raw['mae'] > 1e-7:
            write_json(RUN / 'results' / (material+'_failed_historical_control.json'), differences)
            raise RuntimeError(f'Historical structure-only control failed: {name}: {raw}')
    return dict(source=source, current_vs_frozen=differences,
                current_vs_reference=errors(current,target),
                frozen_source_manifest_sha256=sha256(PAPER / 'historical_source/source_manifest.json'),
                frozen_export=json.loads((output / 'export.json').read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material', required=True, choices=['silicon','sio2'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.monotonic()
    model,cfg,provenance = load_model(args.material,args.device)
    provenance['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance['historical_control'] = historical_control(model,args.material,cfg,args.device)
    write_json(RUN / 'results' / (args.material+'_provenance.json'), provenance)
    for structure in manifest()['structures']:
        if structure['material'] != args.material:
            continue
        print('Predicting', structure['name'], flush=True)
        atoms,pos,box = load_structure_from_cif(RUN / 'structures' / structure['cif'],dtype=cfg.dtype)
        operators = predict(model,atoms,pos,box,args.device)
        output = ARTIFACTS / 'predictions' / structure['name']
        output.mkdir(parents=True,exist_ok=True)
        for name,a in operators.items():
            a.save(output / (name+'.pt'))
        result = dict(structure=structure, predicted_operators=str(output),
                      operator_sha256={name:sha256(output/(name+'.pt')) for name in operators},
                      predicted=diagnostics(operators,structure,args.device))
        if structure['tiled_control']:
            target = tiled_reference(structure,cfg)
            result['tiled_control_errors'] = errors(operators,target)
            # Match the explicitly labelled support control used for paper validation.
            result['prediction_on_reference_support'] = diagnostics(
                {name:restrict_support(a,target[name]) for name,a in operators.items()},structure,args.device)
        result['elapsed_job_seconds'] = time.monotonic()-start
        write_json(RUN / 'results' / (structure['name']+'.json'),result)
        print('Finished', structure['name'], 'seconds', result['elapsed_job_seconds'],flush=True)


if __name__ == '__main__':
    main()
