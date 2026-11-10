"""Preserve source OpenMX settings, replacing geometry and silicon k mesh only."""
import json
import math
from pathlib import Path
import re

from ase.io import read

RUN = Path(__file__).resolve().parent
REMOTE = Path('/data/home2/brzoza73/casus/mandala_worktrees/r6')
INSTALL = '/bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9'
BIN = INSTALL + '/source'


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
            ranks,threads = 96,1
            updated,count = re.subn(r'^(\s*scf.Kgrid\s+)\d+\s+\d+\s+\d+',
                                    lambda m:m[1]+' '.join(map(str,structure['diagnostic_mesh'])),updated,flags=re.M)
            assert count == 1
        assert normalized_settings(original) == normalized_settings(updated)
        # Original binaries and both original DATA.PATH locations are inaccessible
        # to this account, including inside the first allocated Genoa job.
        # Only the filesystem location changes: use public OpenMX 3.9.9 / DFT_DATA19.
        source_data_path = re.search(r'^DATA.PATH\s+(\S+)',original,re.M)[1]
        runnable,count = re.subn(r'^(DATA.PATH\s+)\S+',
                                lambda m:m[1]+INSTALL+'/DFT_DATA19/',updated,flags=re.M)
        assert count == 1
        directory = RUN/'openmx'/structure['name']
        directory.mkdir(parents=True,exist_ok=True)
        (directory/filename).write_text(runnable)
        (directory/'original_data_path_input.txt').write_text(updated)
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
            # High MPI counts replicate large neighboring-atom orbital grids.
            # The initial 96-rank/256G runs exhausted memory before SCF.
            # Share those grids among OpenMP threads instead.
            ranks,threads = 8,12
            lines = evidence[key]['timing_lines']
            total = float(re.search(r'Total Computational Time.*?(\d+\.\d+)\s+\d+\s+(\d+\.\d+)', '\n'.join(lines))[2])
            steps = len([line for line in lines if 'SCF=' in line])
            seconds = total * 8
            hours,memory = 44,'768G'
            cost = dict(source_seconds=total,source_steps=steps,
                        source_seconds_per_step=float(re.search(r'DFT in total\s*=\s*([\d.]+)','\n'.join(lines))[1])/steps,
                        source_ranks='unknown; source run script absent',cubic_and_kpoint_factor=8,
                        assumed_parallel_speedup=1,estimated_seconds=seconds,safety_factor=3,
                        walltime_cap_hours=44,
                        note='No speedup credit because original rank count is unknown. After pre-SCF OOM at 96 MPI ranks/256G, use 8 MPI x 12 OpenMP threads and 768G. 3x estimate capped at 44 h to fit before maintenance on 2026-11-12 08:00 CET.')
        if material == 'silicon' and structure['atom_count'] == 1000:
            # Measured 216-atom, 27-kpoint runs on 96 Genoa ranks.
            measured = [413, 440, 350]
            seconds = max(measured) * (1000/216)**3 * (8/27)
            hours, memory = math.ceil(3*seconds/3600), '1024G'
            ranks, threads = 8, 12
            cost = dict(source_seconds=measured, source_atoms=216, source_kpoints=27,
                        target_kpoints=8, estimated_seconds=seconds, safety_factor=3,
                        source_peak_rss_K=[125469708, 125593112, 130314552],
                        source_jobs=['837291', '837292', '837293'],
                        note='MaxRSS from sacct batch steps. Hybrid 8 MPI x 12 OMP reduces replicated grids; 1024G allows 8.2x measured peak. Runtime scaling assumes same aggregate 96 CPUs; hybrid efficiency is unverified.')
        if material == 'sio2' and structure['atom_count'] == 600:
            measured = [6094, 4848]  # sacct elapsed, including extraction
            seconds = max(measured) * 8
            hours, memory = min(40, math.ceil(3*seconds/3600)), '1400G'
            ranks, threads = 8, 12
            cost = dict(source_seconds=measured, source_atoms=300,
                        source_jobs=['837300', '837301'], source_peak_rss_KiB=[366021928, 325799980],
                        source_steps=[55, 45], source_dft_seconds=[5619.11041, 4368.50198],
                        source_seconds_per_step=[5619.11041/55, 4368.50198/45],
                        cubic_and_kpoint_factor=8, estimated_seconds=seconds, safety_factor=3,
                        uncapped_walltime_hours=3*seconds/3600, walltime_cap_hours=40,
                        node_memory_MB=1500000,
                        note='Genoa advertises 1500000 MB, below 1.5 TiB; request 1400G on one node. Same 8 MPI x 12 OMP as completed runs; >4x observed peak, >2x requested 768G unavailable. Walltime capped at 40h; maintenance fit depends on queue start.')
        assert hours <= 48
        name = structure['name']
        run_dir = structure['dft_directory']
        system = 'Si' if material == 'silicon' else 'SiO2'
        script = f'''#!/usr/bin/env bash
#SBATCH --job-name=r6-dft-{name}
#SBATCH --partition=cpu-genoa-casus
#SBATCH --account=casus
#SBATCH --nodes=1
#SBATCH --ntasks={ranks}
#SBATCH --cpus-per-task={threads}
#SBATCH --mem={memory}
#SBATCH --time={hours:02d}:00:00
#SBATCH --output={REMOTE}/researchops/runs/r6_supercell_20261110/openmx/{name}/slurm-%j.out
set -euo pipefail
source /etc/profile.d/lmod.sh
ml purge
ml genoa
ml foss/2022a
export OMPI_MCA_pml=ucx
export OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {run_dir}
test ! -e COMPLETE
test -x {BIN}/openmx
test -x {BIN}/analysis_example
date -u
mpirun -bind-to core --map-by slot:PE={threads} -np "$SLURM_NTASKS" {BIN}/openmx {filename} > log.out
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
        estimates.append(dict(name=name,ranks=ranks,omp_threads=threads,total_cpus=ranks*threads,
                              memory=memory,walltime=f'{hours:02d}:00:00',
                              input=filename,run_directory=run_dir,
                              original_data_path=source_data_path,
                              runtime_data_path=INSTALL+'/DFT_DATA19/',
                              binary=BIN+'/openmx',basis_version='public DFT_DATA19',**cost))
    (RUN/'openmx/cost_estimates.json').write_text(json.dumps(estimates,indent=2)+'\n')
    print('Prepared',len(estimates),'OpenMX inputs; all non-geometry settings preserved')


if __name__ == '__main__':
    main()
