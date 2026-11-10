"""Stage and submit selected DFT runs from the ROSI worktree, once per structure."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess

RUN = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names',nargs='+',help='Explicit names; use --remaining after first SCF check')
    args = parser.parse_args()
    manifest = json.loads((RUN/'structures/manifest.json').read_text())
    path = RUN/'submissions.json'
    document = json.loads(path.read_text())
    assert len(document['evaluation']) == 2, 'Submit model evaluation for both materials first'
    for structure in manifest['structures']:
        name = structure['name']
        if structure['tiled_control'] or name not in args.names or name in document['openmx']:
            continue
        staged = RUN/'openmx'/name
        directory = Path(structure['dft_directory'])
        directory.mkdir(parents=True,exist_ok=True)
        input_name,system = ('conf.dat','Si') if structure['material'] == 'silicon' else ('SiO2.dat','SiO2')
        assert not (directory/'log.out').exists(), 'Refuse to overwrite an existing calculation'
        shutil.copy2(staged/input_name,directory/input_name)
        shutil.copy2(RUN/'structures'/structure['cif'],directory/(system+'.cif'))
        shutil.copy2(staged/'run.sh',directory/'run.sh')
        command = ['sbatch','--parsable',str(staged/'run.sh')]
        job = subprocess.check_output(command,text=True).strip().split(';')[0]
        document['openmx'][name] = dict(job_id=job,command=command,run_directory=str(directory),
            submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        path.write_text(json.dumps(document,indent=2)+'\n')
        print(name,job,flush=True)


if __name__ == '__main__':
    main()
