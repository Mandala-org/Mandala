import torch,e3nn,time
from omegaconf import OmegaConf
from net.common import Config
from e3nn.o3 import Linear
torch.set_num_threads(4)
c=OmegaConf.to_container(OmegaConf.load('configs/mace_h_graphene.yaml'),resolve=False)
r,layout=Config.resolve(c,species_shells={'C':(0,0,1,1,2)},spinful=False)
models=[]
for mode in ['script','eager']:
 e3nn.set_optimization_defaults(jit_mode=mode)
 torch.manual_seed(42)
 models.append(Linear(r['layout']['post_irreps'],r['layout']['coupled_irreps'],biases=True).eval())
print('BACKENDS',type(models[0]._compiled_main).__name__,type(models[1]._compiled_main).__name__)
assert isinstance(models[0]._compiled_main,torch.jit.ScriptModule)
assert not isinstance(models[1]._compiled_main,torch.jit.ScriptModule)
with torch.no_grad():
 for n in [1,72,3312]:
  x=torch.randn(n,models[0].irreps_in.dim)
  a,b=[m(x) for m in models]
  torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
  print('PARITY',n,'max_abs',float((a-b).abs().max()))
print('LINEAR PARITY PASS')
from core.operator_layout import OrbitalFrameRotation
from e3nn.o3 import rand_matrix
for dtype in [torch.float32,torch.float64]:
 rot=OrbitalFrameRotation(dtype=dtype)
 shells=[0,0,1,1,2]; dim=sum(2*l+1 for l in shells)
 blocks=torch.randn(3,dim,dim,dtype=dtype); R=rand_matrix(dtype=dtype)
 rotated=rot.rotate_blocks(blocks,R,shells,shells)
 restored=rot.rotate_blocks(rotated,R.T,shells,shells)
 torch.testing.assert_close(restored,blocks,atol=2e-4,rtol=2e-4)
 print('ROTATION ROUNDTRIP',dtype,'max_abs',float((restored-blocks).abs().max()))
from unittest.mock import patch
from scripts import benchmark_reference_inference as bench
calls=[]
with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'memory_allocated',return_value=0),patch.object(torch.cuda,'max_memory_allocated',return_value=0):
 result=bench.timing(lambda case: calls.append(case),[0,1,2],5,5)
assert len(calls)==20 and len(result['samples_ms'])==15
assert result['warmup_seconds']>=0 and result['median_ms']>=0
print('TIMING PASS: 5 warmup calls excluded, 15 measured samples')
import tempfile,json
from pathlib import Path
from types import SimpleNamespace
with tempfile.TemporaryDirectory() as tmp:
 args=SimpleNamespace(root=Path('.'),material='graphene',repeats=5,output=Path(tmp)/'result.json',variant_timeout=1800,e3nn_jit_mode='eager')
 def worker(command,**kwargs):
  assert command[command.index('--e3nn-jit-mode')+1]=='eager'
  assert kwargs['start_new_session']
  Path(command[command.index('--output')+1]).write_text(json.dumps({'gpu':'probe','variants':{'mace_h':{'probe':True}}}))
  from unittest.mock import MagicMock
  w=MagicMock(); w.__enter__.return_value=w; w.returncode=0
  return w
 with patch.object(bench.subprocess,'Popen',side_effect=worker):
  assert bench.benchmark_with_timeout(args,'mace_h')==({'probe':True},'probe')
 print('WORKER BACKEND PROPAGATION PASS')
