"""Preserve source OpenMX settings, replacing geometry and silicon k mesh only."""
import json
import math
from pathlib import Path
import re

from ase.io import read

RUN = Path(__file__).resolve().parent
REMOTE = Path('/data/home2/brzoza73/casus/mandala_worktrees/r6')
BIN = '/bigdata/nanostructures/DFT/DI/forWiktoria/codebase/openmx3.9_genoa/source'


def replace_block(text, keyword, body):
    pattern = rf'(<{re.escape(keyword)}[^\n]*\n).*?(^{re.escape(keyword)}>[^\n]*)'
    updated,count = re.subn(pattern, lambda m:m[1]+body+'\n'+m[2],text,flags=re.M|re.S)
    assert count == 1
    return updated


def normalized_settings(text):
    for key in ('Atoms.SpeciesAndCoordinates','Atoms.UnitVectors'):
        text = replace_block(text,key,'GEOMETRY')
    return re.sub(r'^(\s*(?:Atoms.Number|scf.Kgrid))\s+[^\n]*',r'\1 REPLACED',text,flags=re.M|re.I)


def main():
    evidence = json.loads((RUN/'source_evidence.json').read_text())
    manifest = json.loads((RUN/'structures/manifest.json').read_text())
    estimates = []
    for structure in manifest['structures']:
        if structure['tiled_control']:
            continue
        material = structure['material']
        key = 'silicon' if material == 'silicon' else 'sio2_'+structure['name'][-1]
        filename = 'conf.dat' if material == 'silicon' else 'SiO2.dat'
        original = evidence[key]['files'][filename]['text']
        assert re.search(r'level.of.fileout\s+1',original,re.I)
        assert re.search(r'HS.fileout\s+on',original,re.I)
        assert re.search(r'Atoms.SpeciesAndCoordinates.Unit\s+Ang',original,re.I)
        assert re.search(r'Atoms.UnitVectors.Unit\s+Ang',original,re.I)
        oldrows = re.search(r'<Atoms.SpeciesAndCoordinates[^\n]*\n(.*?)Atoms.SpeciesAndCoordinates>',original,re.S)[1]
        oldrows = [line.split() for line in oldrows.splitlines() if line.strip()]
        atoms = read(RUN/'structures'/structure['cif'])
        rows = []
        for index,(symbol,pos) in enumerate(zip(atoms.get_chemical_symbols(),atoms.positions)):
            suffix = oldrows[index % len(oldrows)][5:]
            assert symbol == oldrows[index % len(oldrows)][1]
            rows.append(f'{index+1} {symbol} '+ ' '.join(f'{v:.12f}' for v in pos)+' '+' '.join(suffix))
        updated = replace_block(original,'Atoms.SpeciesAndCoordinates','\n'.join(rows))
        updated = replace_block(updated,'Atoms.UnitVectors','\n'.join(' '.join(f'{v:.12f}' for v in row) for row in atoms.cell.array))
        updated,count = re.subn(r'^(\s*Atoms.Number\s+)\d+',lambda m:m[1]+str(len(atoms)),updated,flags=re.M)
        assert count == 1
        if material == 'silicon':
            updated,count = re.subn(r'^(\s*scf.Kgrid\s+)\d+\s+\d+\s+\d+',
                                    lambda m:m[1]+' '.join(map(str,structure['diagnostic_mesh'])),updated,flags=re.M)
            assert count == 1
        assert normalized_settings(original) == normalized_settings(updated)
        directory = RUN/'openmx'/structure['name']
        directory.mkdir(parents=True,exist_ok=True)
        (directory/filename).write_text(updated)
        if material == 'silicon':
            n = structure['repetition'][0]
            ratio = (len(atoms)/8)**3 * (structure['diagnostic_mesh'][0]/8)**3
            # Four ranks in source run.sh; 96 ranks here, assume only 25% of ideal gain.
            seconds = 114.386 * ratio / (96/4*.25)
            hours = 2 if n == 2 else 18
            memory = '96G' if n == 2 else '192G'
            cost = dict(source_seconds=114.386,source_steps=21,source_seconds_per_step=101.54330/21,
                        source_ranks=4,cubic_and_kpoint_factor=ratio,assumed_parallel_speedup=6,
                        estimated_seconds=seconds,safety_factor=3)
        else:
            lines = evidence[key]['timing_lines']
            total = float(re.search(r'Total Computational Time.*?(\d+\.\d+)\s+\d+\s+(\d+\.\d+)', '\n'.join(lines))[2])
            steps = len([line for line in lines if 'SCF=' in line])
            seconds = total * 8
            hours,memory = 48,'256G'
            cost = dict(source_seconds=total,source_steps=steps,
                        source_seconds_per_step=float(re.search(r'DFT in total\s*=\s*([\d.]+)','\n'.join(lines))[1])/steps,
                        source_ranks='unknown; source run script absent',cubic_and_kpoint_factor=8,
                        assumed_parallel_speedup=1,estimated_seconds=seconds,safety_factor=3,
                        walltime_capped_at_48h=True,
                        note='Conservative: no credit for increasing to 96 ranks because original rank count is unknown; 3x estimate capped at requested 48 h.')
        assert hours <= 48
        name = structure['name']
        run_dir = structure['dft_directory']
        system = 'Si' if material == 'silicon' else 'SiO2'
        script = f'''#!/usr/bin/env bash
#SBATCH --job-name=r6-dft-{name}
#SBATCH --partition=cpu-genoa-casus
#SBATCH --account=casus
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}
#SBATCH --time={hours:02d}:00:00
#SBATCH --output={REMOTE}/researchops/runs/r6_supercell_20261110/openmx/{name}/slurm-%j.out
set -euo pipefail
source /etc/profile.d/lmod.sh
ml purge
ml genoa
ml foss/2022a
export OMPI_MCA_pml=ucx
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {run_dir}
test ! -e COMPLETE
test -x {BIN}/openmx
test -x {BIN}/analysis_example
date -u
mpirun -bind-to core --map-by core -np "$SLURM_NTASKS" {BIN}/openmx {filename} > log.out
grep -q 'The calculation was normally finished.' log.out
grep -q 'The SCF was achieved' log.out
test -s {system}.scfout
mpirun -bind-to core --map-by core -np 1 {BIN}/analysis_example {system}.scfout > HS.out.tmp
test -s HS.out.tmp
mv HS.out.tmp HS.out
# Keep scfout on any failed run. Remove bulky intermediates only after extraction.
shopt -s nullglob
intermediates=( *.md* *.cube *.scfout *.input *.dat# *rst )
if (( ${{#intermediates[@]}} )); then rm -rf -- "${{intermediates[@]}}"; fi
date -u > COMPLETE
'''
        (directory/'run.sh').write_text(script)
        estimates.append(dict(name=name,ranks=96,memory=memory,walltime=f'{hours:02d}:00:00',
                              input=filename,run_directory=run_dir,**cost))
    (RUN/'openmx/cost_estimates.json').write_text(json.dumps(estimates,indent=2)+'\n')
    print('Prepared',len(estimates),'OpenMX inputs; all non-geometry settings preserved')


if __name__ == '__main__':
    main()
