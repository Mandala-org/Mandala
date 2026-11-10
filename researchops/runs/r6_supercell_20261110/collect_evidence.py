"""Run on ROSI with system Python; capture small source files and timing evidence."""
import hashlib
import json
from pathlib import Path
import re

DATA = Path('/bigdata/casus/wdm/hamiltonian_learning/data')
sources = {
    'silicon': DATA / 'perturbed_snapshots_Si/scale_1/Si_scale_1_078',
    'sio2_1': DATA / 'SiOx_new/13_67164000',
    'sio2_2': DATA / 'SiOx_new/2_74240000',
}
result = {}
for material, directory in sources.items():
    names = ['conf.dat', 'Si.cif', 'run.sh'] if material == 'silicon' else ['SiO2.dat', 'SiO2.cif']
    files = {}
    for name in names:
        path = directory / name
        files[name] = {'text': path.read_text(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    log = directory / 'log.out'
    if not log.exists():
        log = next(directory.glob('*log*'))
    lines = log.read_text(errors='replace').splitlines()
    patterns = r'SCF=|Elapsed|elapsed|DFT in total|Total Computational Time|normally finished|SCF was achieved|Num_Procs|numprocs|MPI'
    result[material] = {'directory': str(directory), 'files': files, 'log_path': str(log),
                        'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
                        'timing_lines': [line for line in lines if re.search(patterns, line)]}
metadata = json.loads((DATA / 'perturbed_snapshots_Si/snapshot_metadata.json').read_text())
result['silicon_metadata'] = [x for x in metadata if x['scale_name'] == 'scale_1'][:1]
print(json.dumps(result, indent=2))
