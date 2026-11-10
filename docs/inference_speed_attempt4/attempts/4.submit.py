import datetime,json,subprocess
from pathlib import Path
root=Path.cwd()
out=root/'workflow/attempt4.submissions.json'
assert not out.exists(), 'Attempt 4 already submitted'
# The original submit.py is a one-shot initial-submission tool; preserve it and
# its workflow/submissions.json, following attempt 3's separate records.
p=subprocess.run(['scancel','836806'],text=True,capture_output=True)
(root/'attempts/4.cancel.json').write_text(json.dumps(dict(job_id='836806',returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
p.check_returncode()
rows=[]
for material in ['graphene','mos2']:
 command=['sbatch','--parsable',f'run_{material}.sbatch']
 p=subprocess.run(command,text=True,capture_output=True)
 row=dict(material=material,command=command,stdout=p.stdout,stderr=p.stderr,returncode=p.returncode,script=f'run_{material}.sbatch',method='sbatch; separate attempt record as in attempt 3')
 if p.returncode==0:
  job_id=p.stdout.strip().splitlines()[-1].split(';')[0]
  assert job_id.isdigit(),p.stdout
  row['job_id']=job_id
 rows.append(row)
 out.write_text(json.dumps(rows,indent=2)+'\n')
 p.check_returncode()
task=json.loads((root/'task.json').read_text())
task.update(attempt=4,status='submitted',jobs={r['material']:r['job_id'] for r in rows},workflow=str(out.relative_to(root)),repair_report='attempts/4.report.txt',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),validation='Syntax/help and CPU final-linear script/eager parity passed; eager e3nn backend avoids observed first-forward scripted Linear stall; GPU completion pending.')
(root/'task.json').write_text(json.dumps(task,indent=2)+'\n')
print(json.dumps(rows,indent=2))
