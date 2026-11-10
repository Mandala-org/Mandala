"""Capture Slurm accounting for every registered attempt, including failures."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

RUN = Path(__file__).resolve().parent


def main():
    document = json.loads((RUN/'submissions.json').read_text())
    ids = set()

    def visit(value):
        if isinstance(value,dict):
            if 'job_id' in value:
                ids.add(str(value['job_id']))
            for child in value.values():
                visit(child)
        elif isinstance(value,list):
            for child in value:
                visit(child)

    visit(document)
    command = ['sacct','-P','-j',','.join(sorted(ids)),
               '--format=JobIDRaw,JobName%40,State%30,ExitCode,Elapsed,MaxRSS,ReqMem,AllocCPUS,Timelimit,NodeList']
    output = subprocess.check_output(command,text=True)
    lines = output.splitlines()
    fields = lines[0].split('|')
    rows = [dict(zip(fields,line.split('|'))) for line in lines[1:] if line.strip()]
    result = dict(captured_utc=datetime.now(timezone.utc).isoformat(),command=command,records=rows)
    (RUN/'results/job_status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Captured {len(rows)} Slurm job/step records for {len(ids)} job IDs')


if __name__ == '__main__':
    main()
