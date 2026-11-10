"""Stage and submit source-cell runtime checks on ROSI; preserve original data."""
import datetime
import argparse
import json
from pathlib import Path
import shutil
import subprocess

RUN = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--material',choices=['silicon','sio2'])
    args = parser.parse_args()
    records = json.loads((RUN/'openmx/calibration.json').read_text())
    path = RUN/'submissions.json'
    document = json.loads(path.read_text())
    jobs = document.setdefault('calibration',{})
    for material,record in records.items():
        if args.material and material != args.material:
            continue
        previous = jobs.get(material)
        if previous:
            if not args.retry_failed:
                continue
            state = subprocess.check_output(['sacct','-j',previous['job_id'],'-X','-n','-o','State'],text=True).strip()
            assert state.startswith(('FAILED','OUT_OF_MEMORY','CANCELLED','TIMEOUT')),state
        staged = RUN/'openmx'/('calibration-'+material)
        directory = Path(record['new'])
        directory.mkdir(parents=True,exist_ok=True)
        assert not (directory/'COMPLETE').exists()
        if (directory/'log.out').exists():
            assert args.retry_failed and previous
            archive = directory/('failed-'+previous['job_id'])
            archive.mkdir(exist_ok=False)
            for source in list(directory.iterdir()):
                if source == archive or source.name.startswith('failed-') or source.suffix in ('.dat','.cif','.sh'):
                    continue
                shutil.move(str(source),str(archive/source.name))
            previous['archived_outputs'] = str(archive)
        for source in staged.iterdir():
            if source.suffix in ('.dat','.cif','.sh'):
                shutil.copy2(source,directory/source.name)
        command = ['sbatch','--parsable',str(staged/'run.sh')]
        job = subprocess.check_output(command,text=True).strip().split(';')[0]
        jobs[material] = dict(job_id=job,command=command,run_directory=str(directory),
                             submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if previous:
            jobs[material]['previous_attempts'] = [previous]
        path.write_text(json.dumps(document,indent=2)+'\n')
        print(material,job,flush=True)


if __name__ == '__main__':
    main()
