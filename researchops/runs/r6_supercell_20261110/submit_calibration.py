"""Stage and submit source-cell runtime checks on ROSI; preserve original data."""
import datetime
import json
from pathlib import Path
import shutil
import subprocess

RUN = Path(__file__).resolve().parent


def main():
    records = json.loads((RUN/'openmx/calibration.json').read_text())
    path = RUN/'submissions.json'
    document = json.loads(path.read_text())
    jobs = document.setdefault('calibration',{})
    for material,record in records.items():
        if material in jobs:
            continue
        staged = RUN/'openmx'/('calibration-'+material)
        directory = Path(record['new'])
        directory.mkdir(parents=True,exist_ok=True)
        assert not (directory/'log.out').exists()
        for source in staged.iterdir():
            if source.suffix in ('.dat','.cif','.sh'):
                shutil.copy2(source,directory/source.name)
        command = ['sbatch','--parsable',str(staged/'run.sh')]
        job = subprocess.check_output(command,text=True).strip().split(';')[0]
        jobs[material] = dict(job_id=job,command=command,run_directory=str(directory),
                             submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        path.write_text(json.dumps(document,indent=2)+'\n')
        print(material,job,flush=True)


if __name__ == '__main__':
    main()
