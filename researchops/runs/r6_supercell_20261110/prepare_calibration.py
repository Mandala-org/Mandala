"""Reproduce two source cells to validate the replacement OpenMX runtime/data."""
import json
from pathlib import Path
import re

RUN = Path(__file__).resolve().parent
BASE = Path('/bigdata/casus/wdm/hamiltonian_learning/data/r6_supercells_20261110/calibration')
INSTALL = '/bigdata/casus/wdm/hamiltonian_learning/physical_constraints/r6_supercell_20261110/environment/openmx3.9'


def main():
    evidence = json.loads((RUN/'source_evidence.json').read_text())
    records = {}
    for material,key,example,ranks,mem,time in (
        ('silicon','silicon','si-2x2x2-1',4,'32G','00:30:00'),
        ('sio2','sio2_1','sio2-2x1x1-1',32,'96G','04:00:00'),
    ):
        directory = RUN/'openmx'/('calibration-'+material)
        directory.mkdir(parents=True,exist_ok=True)
        filename,system = ('conf.dat','Si') if material=='silicon' else ('SiO2.dat','SiO2')
        original = evidence[key]['files'][filename]['text']
        updated = re.sub(r'^(DATA.PATH\s+)\S+',lambda m:m[1]+INSTALL+'/DFT_DATA19/',original,flags=re.M)
        (directory/filename).write_text(updated)
        (directory/(system+'.cif')).write_text(evidence[key]['files'][system+'.cif']['text'])
        script = (RUN/'openmx'/example/'run.sh').read_text()
        script = script.replace('r6-dft-'+example,'r6-cal-'+material)
        script = re.sub(r'^#SBATCH --ntasks=\d+',f'#SBATCH --ntasks={ranks}',script,flags=re.M)
        script = re.sub(r'^#SBATCH --mem=\S+',f'#SBATCH --mem={mem}',script,flags=re.M)
        script = re.sub(r'^#SBATCH --time=\S+',f'#SBATCH --time={time}',script,flags=re.M)
        script = script.replace('/openmx/'+example+'/', '/openmx/calibration-'+material+'/')
        script = re.sub(r'^cd /bigdata/[^\n]+',f'cd {BASE/material}',script,flags=re.M)
        (directory/'run.sh').write_text(script)
        records[material] = dict(original=evidence[key]['directory'],new=str(BASE/material),
                                ranks=ranks,memory=mem,walltime=time)
    (RUN/'openmx/calibration.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__ == '__main__':
    main()
