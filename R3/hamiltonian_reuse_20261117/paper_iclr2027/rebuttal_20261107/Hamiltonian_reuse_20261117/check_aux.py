"""Verify independent development readouts and unchanged-model CQS optimization."""
from pathlib import Path
import json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from run_reuse import BASE,ORDERS,data,ProductMomentREDQ,torch
from history import HistoryCache
from engine import PhysicalSampler,wrap,write

def main():
 torch.set_num_threads(2);frozen=json.loads((HERE/'frozen_selection.json').read_text());readouts=0;maxerror=0.
 for path in sorted((HERE/'development').rglob('complete.json')):
  r=json.loads(path.read_text());e=PhysicalSampler(r['kind'],-.5)
  with np.load(path.parent/'development_holdout.npz') as z:
   terms=e.terms(z['counts_z'],z['counts_x']);j=np.repeat(ORDERS['smooth'],20)
   value=((-.5 if r['kind']=='grid' else .5)*terms[:,:e.n].sum(1)+(j if r['kind']=='grid' else -j)*terms[:,e.n:].sum(1)).mean()
   assert int(2*z['shots_per_group'].sum())==400000
  np.testing.assert_allclose(value,r['validation']['mean_independent_measured_energy'],rtol=0,atol=1e-12)
  assert r['validation']['used_for_search'] is False;readouts+=1
 if readouts:assert readouts==16
 for key in ['grid_low','grid_shared','chain_low','chain_shared']:
  r=frozen[key];assert r['config']==min(r['candidates'],key=lambda z:z['mean'])['config']
 count=0
 for path in sorted((HERE/'runs').rglob('cqs/complete.json')):
  r=json.loads(path.read_text());kind,seed=r['kind'],r['seed'];cache=HistoryCache(kind,data(kind,seed),'shared');e=cache.engine
  model=ProductMomentREDQ(e.n,e.edges,2,kind=='grid',32,4 if kind=='grid' else 1).double().eval()
  model.load_state_dict(torch.load(BASE/kind/f'seed{seed}'/'product.pt',map_location='cpu',weights_only=True))
  for parameter in model.parameters():parameter.requires_grad_(False)
  previous=None
  for target,j in enumerate(ORDERS[r['order']]):
   with np.load(path.parent/f'target{target:02d}.npz') as z:stored={k:z[k] for k in z.files}
   center=cache.best(j);starts=np.broadcast_to(center,(20,*cache.shape)).copy();rng=np.random.default_rng(seed+500000+target*100)
   flat=starts.reshape(20,-1)
   if kind=='grid':flat[1:,cache.active]+=rng.normal(0,.05,(19,len(cache.active)))
   else:flat[1:,cache.active]+=rng.uniform(-.2,.2,(19,len(cache.active)))
   starts=wrap(flat).reshape(starts.shape);coeff=torch.tensor(np.r_[np.full(e.n,-.5 if kind=='grid' else .5),np.full(len(e.edges),j if kind=='grid' else -j)])
   with torch.no_grad():
    if previous is not None:
     old=(model(torch.tensor(previous))[0]@coeff).numpy();cold=(model(torch.tensor(starts))[0]@coeff).numpy();starts=np.where((old<cold)[:,None,None],previous,starts)
   np.testing.assert_allclose(starts,stored['starts'],rtol=0,atol=2e-11);np.testing.assert_allclose(center,stored['center'],rtol=0,atol=2e-11)
   # Stored starts avoid amplifying cross-platform ulp differences in Adam.
   x=torch.tensor(stored['starts'],requires_grad=True);opt=torch.optim.Adam([x],lr=.1,foreach=False)
   for _ in range(100):
    opt.zero_grad();(model(x)[0]@coeff).sum().backward();opt.step()
    with torch.no_grad():x.copy_((x+np.pi)%(2*np.pi)-np.pi)
   final=x.detach().numpy();maxerror=max(maxerror,float(np.max(np.abs(final-stored['parameters']))))
   np.testing.assert_allclose(final,stored['parameters'],rtol=0,atol=1e-8)
   with torch.no_grad():np.testing.assert_allclose((model(torch.tensor(stored['parameters']))[0]@coeff).numpy(),stored['prediction'],rtol=0,atol=2e-11)
   previous=stored['parameters'];count+=1
 assert count==400
 write(HERE/'aux_verification.json',dict(passed=True,independent_development_readouts=readouts,cqs_targets_reoptimized=count,cqs_parameter_max_error=maxerror,development_holdout_never_used_for_search=True))
 print('AUXILIARY CHECKS PASSED',count,readouts,flush=True)

if __name__=='__main__':main()
