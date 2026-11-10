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
    parser.add_argument('--retry-failed',action='store_true',help='Retry a recorded failure before SCF only')
    args = parser.parse_args()
    manifest = json.loads((RUN/'structures/manifest.json').read_text())
    path = RUN/'submissions.json'
    document = json.loads(path.read_text())
    assert len(document['evaluation']) == 2, 'Submit model evaluation for both materials first'
    for structure in manifest['structures']:
        name = structure['name']
        if structure['tiled_control'] or name not in args.names:
            continue
        previous = document['openmx'].get(name)
        if previous:
            if not args.retry_failed:
                continue
            state = subprocess.check_output(['sacct','-j',previous['job_id'],'-X','-n','-o','State'],text=True).strip()
            assert state.startswith(('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY')), state
        staged = RUN/'openmx'/name
        directory = Path(structure['dft_directory'])
        directory.mkdir(parents=True,exist_ok=True)
        input_name,system = ('conf.dat','Si') if structure['material'] == 'silicon' else ('SiO2.dat','SiO2')
        assert not (directory/'COMPLETE').exists(), 'Refuse to overwrite a completed calculation'
        if (directory/'log.out').exists():
            assert previous and args.retry_failed, 'Refuse to overwrite an existing calculation'
            archive = directory/('failed-'+previous['job_id'])
            archive.mkdir(exist_ok=False)
            for source in list(directory.iterdir()):
                if source == archive or source.name.startswith('failed-') or source.name in (input_name,system+'.cif','run.sh'):
                    continue
                shutil.move(str(source),str(archive/source.name))
            previous['archived_outputs'] = str(archive)
        shutil.copy2(staged/input_name,directory/input_name)
        shutil.copy2(RUN/'structures'/structure['cif'],directory/(system+'.cif'))
        shutil.copy2(staged/'run.sh',directory/'run.sh')
        command = ['sbatch','--parsable',str(staged/'run.sh')]
        job = subprocess.check_output(command,text=True).strip().split(';')[0]
        document['openmx'][name] = dict(job_id=job,command=command,run_directory=str(directory),
            submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if previous:
            document['openmx'][name]['previous_attempts'] = [previous]
        path.write_text(json.dumps(document,indent=2)+'\n')
        print(name,job,flush=True)


if __name__ == '__main__':
    main()
