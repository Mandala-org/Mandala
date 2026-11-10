"""Deterministic cells in ASE tile-major order, with complete geometry provenance."""
from pair_categories import atom_mapping

import hashlib
import io
import json
from pathlib import Path

import ase
from ase.build import bulk
from ase.io import read, write
import numpy as np

RUN = Path(__file__).resolve().parent


def main():
    evidence = json.loads((RUN / 'source_evidence.json').read_text())
    out = RUN / 'structures'
    out.mkdir(exist_ok=True)
    sigma = evidence['silicon_metadata'][0]['sigma_angstrom']
    assert sigma == 0.02
    si = read(io.StringIO(evidence['silicon']['files']['Si.cif']['text']), format='cif')
    lattice = float(si.cell.lengths()[0])
    assert np.allclose(si.cell.array, np.eye(3) * lattice, atol=1e-10)
    ideal = bulk('Si', 'diamond', a=lattice, cubic=True)
    records = []

    def save(name, material, base, repetition, sigma, seed, source, control=False):
        atoms = base.repeat(repetition)
        if seed is not None:
            displacements = np.random.default_rng(seed).normal(0, sigma, (len(atoms), 3))
            atoms.positions += displacements
            atoms.wrap(eps=0)
            rms = float(np.sqrt(np.mean(np.sum(displacements**2, axis=1))))
        else:
            rms = 0.0
        path = out / (name + '.cif')
        write(path, atoms, format='cif')
        loaded = read(path)
        assert loaded.get_chemical_symbols() == atoms.get_chemical_symbols()
        assert np.allclose(loaded.positions, atoms.positions, atol=1e-10)
        records.append(dict(name=name, material=material, cif=path.name,
                            source_cell=evidence[source]['directory'],
                            source_geometry='ideal diamond at dataset lattice constant' if material == 'silicon' and not control else 'dataset CIF',
                            repetition=list(repetition), sigma_angstrom=sigma, seed=seed,
                            atom_count=len(atoms), parent_atom_count=len(base),
                            atom_parent_mapping=atom_mapping(len(base), repetition),
                            rms_displacement_angstrom=rms,
                            tiled_control=control, tile_order='ASE repeat: product(range(nx),range(ny),range(nz)), base atom fastest',
                            num_electrons=sum(4 if z == 14 else 6 for z in atoms.numbers),
                            diagnostic_mesh=([4]*3 if repetition[0] == 2 else ([2]*3 if repetition[0] == 5 else [3]*3)) if material == 'silicon' else [1,1,1],
                            dft_directory=f'/bigdata/casus/wdm/hamiltonian_learning/data/r6_supercells_20261110/{material}/{name}',
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for repeat in (2, 3, 5):
        for sample in ((1,) if repeat == 5 else (1, 2, 3)):
            save(f'si-{repeat}x{repeat}x{repeat}-{sample}', 'silicon', ideal, (repeat,)*3, sigma,
                 20261110 + repeat*100 + sample, 'silicon')
        save(f'si-{repeat}x{repeat}x{repeat}-tiled', 'silicon', si, (repeat,)*3, 0, None, 'silicon', True)
    for sample in (1, 2):
        key = f'sio2_{sample}'
        base = read(io.StringIO(evidence[key]['files']['SiO2.cif']['text']), format='cif')
        assert len(base) == 150
        save(f'sio2-2x1x1-{sample}', 'sio2', base, (2,1,1), 0.02, 20262110+sample, key)
        if sample == 1:
            save('sio2-2x1x1-tiled', 'sio2', base, (2,1,1), 0, None, key, True)
    manifest = dict(schema_version=2, silicon_training=dict(scales=[1], num_train_per_scale=80,
                    num_val_per_scale=16, split_seed=43, sigma_angstrom=sigma,
                    dataset_generation_seed=7, lattice_constant_angstrom=lattice),
                    sio2_split_seed=42, ase_version=ase.__version__, numpy_version=np.__version__,
                    reference_density='OpenMX single spin multiplied by 2 to match historical restricted-total heads',
                    silicon_mesh_note='8/3 and 8/5 are not integral; 3x3x3 and 2x2x2 respectively are slightly denser than the original 8x8x8 density.',
                    sio2_control_caveat='Original Gamma-only sampling corresponds to supercell Gamma and X-type points; folded blocks are an exact DFT reference only if the original Gamma calculation was converged with respect to k sampling.',
                    structures=records)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Built {len(records)} structures; silicon sigma={sigma}; a={lattice}')


if __name__ == '__main__':
    main()
