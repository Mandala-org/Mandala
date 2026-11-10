"""Shared fixed-checkpoint loading, raw operator diagnostics, and reference folding."""
from pathlib import Path
import json
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import torch
from analysis.physical_constraints import (BlockFourier, diagnose_kpoint, hermiticity,
    public_record, summarize_records, uniform_kpoints)
from core.block_irrep_mapper import BlockIrrepMapper
from data.block_matrix import BlockMatrix
from data.snapshot import Snapshot
from data.structure_inference import load_structure_from_cif
from net.checkpoint_compat import set_checkpoint_film_semantics, strip_mapper_keys
from net.e3gnn import E3GNN
from net.evaluation import load_checkpoint, restore_config
from scripts.report.evaluate_physical_constraints import (matrix_error, restrict_support,
    sha256, sparse_trace)
from utils.units import HARTREE_TO_EV
from fold_reference import fold_matrix

BIG = Path('/bigdata/casus/wdm/hamiltonian_learning')
PAPER = BIG / 'physical_constraints/paper_20261104_v1'
ARTIFACTS = BIG / 'physical_constraints/r6_supercell_20261110'
CHECKPOINTS = {
    'silicon': 'silicon_hdo_energy_stage2_47h25/leafy-sweep-15-half_gt-coef3em3-lr3em4-47h25/best_model.pt',
    'sio2': 'SiOx_hamiltonian_mae_stage2_47h25/dutiful-sweep-12-restart47h9-pat60-restart95h8-mae-lr0p0027-reset-lr2em4-47h25/best_model.pt',
}
ENVELOPE = PAPER / 'source/eval_outputs/SiOx_0_50372000_radial_fit_study/slater_exp_quad_soft_wall_envelope.json'


def load_paper_cif(path, dtype):
    """Match Snapshot._change_basis('e3nn'), including its Cartesian axes.

    The paper's OpenMX-to-e3nn orbital conversion also permutes geometry;
    converting orbital blocks alone would compare different physical frames.
    Fractional lattice shifts and atom indices are unchanged.
    """
    atoms,positions,box = load_structure_from_cif(path,dtype=dtype)
    change_of_basis = torch.eye(3,dtype=dtype)[[2,0,1]]
    return atoms,positions @ change_of_basis,box @ change_of_basis


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=str, allow_nan=False)+'\n')
    temporary.replace(path)


def manifest():
    return json.loads((RUN / 'structures/manifest.json').read_text())


def reference(directory, cfg=None):
    directory = Path(directory)
    info = directory / ('Si.out' if (directory / 'Si.out').is_file() else 'SiO2.out')
    snapshot = Snapshot.from_openmx(directory / 'HS.out', info, cfg=cfg)
    operators = {name: getattr(snapshot, name) for name in ('hamiltonian','overlap','density')}
    operators['density'] = operators['density'] * 2.0
    return snapshot, operators


def load_model(material, device):
    checkpoint = BIG / 'models/checkpoints' / CHECKPOINTS[material]
    payload = load_checkpoint(checkpoint)
    cfg = restore_config(payload)
    cfg.dataset_device = 'cpu'
    cfg.device = device
    cfg.snapshot_cache_dir = None
    cfg.spectral_loss_enabled = False
    cfg.spectral_fermi_cache_path = None
    cfg.verbosity = 0
    if material == 'sio2':
        cfg.hamiltonian_envelope_path = str(ENVELOPE)
    source = next(s['source_cell'] for s in manifest()['structures'] if s['material'] == material)
    first, _ = reference(source, cfg)
    mapper = BlockIrrepMapper(first.hamiltonian.orbital_cfg, dtype=cfg.dtype)
    model = E3GNN(mapper=mapper, cfg=cfg).to(device).eval()
    model.load_state_dict(strip_mapper_keys(payload['state_dict']), strict=True)
    semantics = set_checkpoint_film_semantics(model, 'legacy_2026_07')
    provenance = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        film_semantics=semantics, strict_load=True, graph_source='geometry_cutoff',
        cutoff_radius=cfg.cutoff_radius, torch=torch.__version__, cuda=torch.version.cuda,
        density_convention='restricted total; reference OpenMX single spin multiplied by two',
        units=dict(hamiltonian='Hartree', overlap='dimensionless', density='AO kernel'),
        postprocessing='radial envelope only; raw directed predictions; no repair')
    provenance['geometry_convention'] = "Snapshot._change_basis: OpenMX CIF positions and box right-multiplied by eye(3)[[2,0,1]]; blocks in e3nn basis"
    if material == 'sio2':
        provenance.update(envelope=str(ENVELOPE), envelope_sha256=sha256(ENVELOPE))
    return model, cfg, provenance


@torch.no_grad()
def diagnostics(operators, structure, device='cuda'):
    assemblers = {name: BlockFourier(a, device=device) for name,a in operators.items()}
    records = []
    for k in uniform_kpoints(structure['diagnostic_mesh']):
        dense = {name: a.at(k) for name,a in assemblers.items()}
        if 'overlap' in dense:
            row = diagnose_kpoint(dense['hamiltonian'], dense['overlap'], dense.get('density'))
        else:
            row = dict(status='ok', raw_hermiticity={name: hermiticity(a) for name,a in dense.items()})
        row['fractional_k'] = k
        records.append(public_record(row))
    result = dict(mesh=structure['diagnostic_mesh'], summary=summarize_records(records),
                  records=records, nominal_electrons=structure['num_electrons'],
                  missing_reverse_blocks={name: a.reverse_missing for name,a in assemblers.items()})
    if 'density' in operators:
        raw = sparse_trace(operators['density'], operators['overlap'])
        projected = sparse_trace((operators['density']+operators['density'].adjoint())*.5,
                                 (operators['overlap']+operators['overlap'].adjoint())*.5)
        result.update(exact_realspace_electron_trace=raw,
                      electron_count_absolute_error=abs(raw['real']-structure['num_electrons']),
                      hermitian_realspace_electron_trace=projected,
                      hermitian_electron_count_absolute_error=abs(projected['real']-structure['num_electrons']))
    return result


def errors(predicted, target):
    if any(a.atoms != target[name].atoms or a.basis != target[name].basis
           for name,a in predicted.items()):
        raise ValueError('Prediction and reference atom order/basis differ')
    result = {}
    for name,a in predicted.items():
        raw = matrix_error(a, target[name])
        projected = matrix_error((a+a.adjoint())*.5, target[name])
        if name == 'hamiltonian':
            raw['mae_meV'] = raw['mae'] * HARTREE_TO_EV * 1000
            projected['mae_meV'] = projected['mae'] * HARTREE_TO_EV * 1000
        result[name] = dict(raw_common_support=raw, hermitian_common_support=projected)
    return result


def tiled_reference(structure, cfg=None):
    _, source = reference(structure['source_cell'], cfg)
    return {name: fold_matrix(a, structure['repetition']) for name,a in source.items()}


def load_predictions(name):
    directory = ARTIFACTS / 'predictions' / name
    return {p.stem: BlockMatrix.load(p) for p in directory.glob('*.pt')}
