import sys,time,torch,e3nn
from omegaconf import OmegaConf
from net.common import Config
from e3nn.o3 import Linear
mode=sys.argv[1]
torch.set_num_threads(4)
c=OmegaConf.to_container(OmegaConf.load('configs/mace_h_graphene.yaml'),resolve=False)
r,layout=Config.resolve(c,species_shells={'C':(0,0,1,1,2)},spinful=False)
print('IRREPS',r['layout']['post_irreps'],r['layout']['coupled_irreps'],flush=True)
if mode=='eager': e3nn.set_optimization_defaults(jit_mode='eager')
t0=time.perf_counter()
m=Linear(r['layout']['post_irreps'],r['layout']['coupled_irreps'],biases=True).eval()
x=torch.randn(1,m.irreps_in.dim)
print('CONSTRUCT',time.perf_counter()-t0,flush=True)
with torch.no_grad(),torch.jit.optimized_execution(mode=='default'):
 for i in range(3):
  t0=time.perf_counter(); y=m(x); print(mode,i,time.perf_counter()-t0,y.abs().max().item(),flush=True)
