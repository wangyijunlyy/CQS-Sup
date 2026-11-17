"""Audit all saved measurement records, then score every retained endpoint."""
from pathlib import Path
import argparse,csv,hashlib,json,sys,time
import numpy as np
from scipy.stats import t as student_t
import torch
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE.parent)]
from engine import PhysicalSampler,write,save
from worker import grid_pilot,chain_starts
from run_evidence import TFIMTask
from direct import GRID_BUDGETS,CHAIN_BUDGETS


def describe(values):
 a=np.asarray(values,dtype=float);sd=float(a.std(ddof=1));half=student_t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a))
 return dict(mean=float(a.mean()),sd=sd,n=len(a),values=a.tolist(),ci95=[float(a.mean()-half),float(a.mean()+half)])

def record(path,kind,j):
 with np.load(path) as a:d={k:a[k] for k in a.files}
 e=PhysicalSampler(kind,j);cx=d['counts_x'].astype(float);cz=d['counts_z'].astype(float);n=d['shots_per_group']
 np.testing.assert_array_equal(cx.sum(1),n);np.testing.assert_array_equal(cz.sum(1),n)
 ex=cx@e.vx/n;ez=cz@e.vz/n
 var=(np.maximum(0,cx@e.vx**2/n-ex**2)+np.maximum(0,cz@e.vz**2/n-ez**2))/np.maximum(1,n-1)
 np.testing.assert_allclose(ex+ez,d['energy'],atol=1e-12);np.testing.assert_allclose(var,d['variance'],atol=1e-12)
 d['cumulative']=np.cumsum(2*n);return d

def acceptance(d,cand,old,z):
 take=d['energy'][cand]+z*np.sqrt(d['variance'][cand]+d['variance'][old])<d['energy'][old]
 return np.where(np.asarray(take).reshape(-1,1,1),d['parameters'][cand],d['parameters'][old]),take

def audit_spsa(folder,kind,j,starts,center,config):
 meta=json.loads((folder/'complete.json').read_text());assert meta['config']==config
 d=record(folder/'measurements.npz',kind,j);prefix=40000 if kind=='grid' else 0
 budgets=GRID_BUDGETS if kind=='grid' else CHAIN_BUDGETS
 with np.load(folder/'trajectory.npz') as a:
  np.testing.assert_array_equal(a['starts'],starts);np.testing.assert_array_equal(a['center'],center)
 previous=np.broadcast_to(center,starts.shape).copy()
 for b in budgets:
  end=int(np.searchsorted(d['cumulative'],b-prefix))+1;assert d['cumulative'][end-1]==b-prefix
  cand=slice(end-40,end-20);old=slice(end-20,end)
  np.testing.assert_array_equal(d['parameters'][old],previous)
  kept,take=acceptance(d,cand,old,0 if kind=='grid' else 2)
  with np.load(folder/f'budget{b}.npz') as a:
   np.testing.assert_array_equal(a['parameters'],kept);np.testing.assert_array_equal(a['accepted'],take)
   np.testing.assert_array_equal(a['candidate_parameters'],d['parameters'][cand])
  previous=kept
 assert d['cumulative'][-1]+prefix==budgets[-1]
 return len(d['energy'])

def audit_cobyla(folder,starts,center,config):
 meta=json.loads((folder/'complete.json').read_text());assert meta['config']==config
 with np.load(folder/'starts.npz') as a:np.testing.assert_array_equal(a['starts'],starts)
 d=record(folder/'measurements.npz','grid',-.5);quotas=[(b-40000)//20 for b in GRID_BUDGETS];out={}
 for restart in range(20):
  previous=center[None].copy()
  for b,q in zip(GRID_BUDGETS,quotas):
   end=int(np.searchsorted(d['cumulative'],restart*quotas[-1]+q))+1
   assert d['cumulative'][end-1]==restart*quotas[-1]+q
   np.testing.assert_array_equal(d['parameters'][end-1:end],previous)
   kept,take=acceptance(d,slice(end-2,end-1),slice(end-1,end),0)
   with np.load(folder/f'budget{b}.npz') as a:np.testing.assert_array_equal(a['parameters'][restart],kept[0])
   previous=kept
 assert d['cumulative'][-1]+40000==4000000
 return len(d['energy'])

def audit_local(folder,j,method,starts,center):
 d=record(folder/'measurements.npz','chain',j);prefix=24000 if method=='product' else 0
 previous=center.copy()
 for k,b in enumerate([578000,1132000,1686000]):
  start=k*277;end=start+277
  assert d['cumulative'][end-1]+prefix==b
  np.testing.assert_array_equal(d['parameters'][start+256],previous)
  cand=slice(start+257,end);old=slice(start+256,start+257)
  kept,take=acceptance(d,cand,old,2)
  with np.load(folder/f'budget{b}.npz') as a:
   np.testing.assert_array_equal(a['parameters'],kept);np.testing.assert_array_equal(a['accepted'],take)
   np.testing.assert_array_equal(a['center'],previous)
   if k==0:
    expected=(starts.reshape(20,22)[:,6:]-center.reshape(1,22)[:,6:])/.2
    np.testing.assert_allclose(a['initial_standardized'],expected,atol=1e-14)
   training=previous.reshape(1,22)+np.c_[np.zeros((256,6)),a['training_coordinates']*([.8,.6,.4][k]/4)]
   np.testing.assert_array_equal(training.reshape(256,2,11),d['parameters'][start:start+256])
  estimated=np.where(take,d['energy'][cand],d['energy'][old][0]);previous=kept[int(np.argmin(estimated))].copy()
 assert d['cumulative'][-1]+prefix==1686000
 return len(d['energy'])

def main(replay=False):
 torch.set_num_threads(2)
 assert json.loads((HERE/'all_saved.json').read_text())['complete']
 frozen=json.loads((HERE/'frozen_selection.json').read_text());proto=json.loads((HERE/'protocol.json').read_text())
 for name,digest in frozen['source_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
 validation=[];models=[];measurement_rows=0
 # All data/control, fit identity, and measured acceptance checks precede scoring.
 for seed in proto['grid_confirmation_seeds']:
  base=HERE/'grid'/f'seed{seed}';assert (base/'complete.json').exists()
  if not replay:assert (base/'dataset.npz').stat().st_mtime>=frozen['frozen_utc']
  with np.load(base/'dataset.npz') as a:d={k:a[k] for k in a.files}
  assert d['X'].shape==(400,2,2) and d['measured_terms'].shape==(200,21)
  assert int(d['shots_used'])==4000000
  np.testing.assert_array_equal(d['counts_x'].sum(1),np.full(200,10000));np.testing.assert_array_equal(d['counts_z'].sum(1),np.full(200,10000))
  assert int(d['pilot_counts_x'].sum()+d['pilot_counts_z'].sum())==40000
  assert np.all(d['counts_x'][:20]>=d['pilot_counts_x']) and np.all(d['counts_z'][:20]>=d['pilot_counts_z'])
  sampler=PhysicalSampler('grid',-.5);np.testing.assert_allclose(sampler.terms(d['counts_z'],d['counts_x']),d['measured_terms'],atol=1e-14)
  measured=sampler.terms(d['pilot_counts_z'],d['pilot_counts_x'])@np.r_[np.full(9,-.5),np.full(12,-.5)]
  np.testing.assert_array_equal(d['center'],d['X'][np.argmin(measured)])
  center,starts=grid_pilot(seed)
  # NumPy versions can differ by one ulp when scaling uniform draws.
  np.testing.assert_allclose(center,d['center'],rtol=0,atol=1e-14)
  np.testing.assert_allclose(starts,d['starts'],rtol=0,atol=1e-14)
  center,starts=d['center'],d['starts']
  for method in ['product','fourier']:
   with np.load(base/f'{method}_search.npz') as a:np.testing.assert_array_equal(a['starts'],starts);assert a['parameters'].shape==(20,2,2)
  info=json.loads((base/'fit.json').read_text());assert info['parameters']==24885 and 500<=info['epochs_run']<=3000
  models.append(hashlib.sha256((base/'product.pt').read_bytes()).hexdigest())
  measurement_rows+=audit_spsa(base/'spsa','grid',-.5,starts,center,frozen['grid_spsa']['config'])
  measurement_rows+=audit_cobyla(base/'cobyla',starts,center,frozen['grid_cobyla']['config'])
  validation.append(dict(task='grid',seed=seed,global_shots=4000000,pilot_shots_included=40000))
 for seed in proto['chain_confirmation_seeds']:
  base=HERE/'chain'/f'seed{seed}';assert (base/'complete.json').exists()
  if not replay:assert (base/'dataset.npz').stat().st_mtime>=frozen['frozen_utc']
  with np.load(base/'dataset.npz') as a:x,y,bits,recipes,cost=a['X'],a['measured_terms'],a['bits'],a['recipes'],int(a['shots_used'])
  assert x.shape==(2800,2,11) and y.shape==(2400,11) and cost==24000
  assert bits.shape==recipes.shape==(2400,10,6) and np.isin(bits,[0,1]).all() and np.isin(recipes,[0,1,2]).all()
  signs=1.-2*bits;e=PhysicalSampler('chain',-.5)
  rebuilt=np.c_[(3*signs*(recipes==0)).mean(1),np.stack([(9*signs[:,:,i]*signs[:,:,j]*(recipes[:,:,i]==2)*(recipes[:,:,j]==2)).mean(1) for i,j in e.edges],axis=1)]
  np.testing.assert_array_equal(rebuilt,y)
  info=json.loads((base/'fit.json').read_text());assert info['parameters']==23595 and info['epochs_run']<=300
  models.append(hashlib.sha256((base/'product.pt').read_bytes()).hexdigest())
  expected_center,expected_starts=chain_starts(seed)
  with np.load(base/'starts.npz') as a:center,starts=a['center'],a['starts']
  np.testing.assert_allclose(center,expected_center,rtol=0,atol=1e-14)
  np.testing.assert_allclose(starts,expected_starts,rtol=0,atol=1e-14)
  for j in [-.5,.5]:
   for method in ['product','local_only']:measurement_rows+=audit_local(base/method/f'J{j:+.1f}',j,method,starts,center)
   measurement_rows+=audit_spsa(base/'spsa'/f'J{j:+.1f}','chain',j,starts,center,frozen['chain_spsa']['config'])
  validation.append(dict(task='chain',seed=seed,global_shots=24000,full_independent_fit=True))
 assert len(set(models))==20
 write(HERE/'preflight.json',dict(passed=True,independent_grid_datasets=10,independent_chain_datasets=10,unique_product_checkpoints=20,measurement_controls_audited=measurement_rows,source_freeze_preserved=True,pilot_and_full_budgets_rebuilt=True,all_acceptance_decisions_rebuilt=True,all_shared_starts_verified=True,completed_before_exact_scoring=True,record_replay_mode=replay,generation_file_order_checked=not replay,details=validation))
 write(HERE/'scoring_started.json',dict(time_utc=time.time(),all_preflight_passed=True))
 # Exact values enter only here, after all complete runs have passed the audit.
 rows=[];max_error=0.
 def score(kind,j,method,seed,budget,parameters,artifact):
  nonlocal max_error
  sampler=PhysicalSampler(kind,j);pz,px=sampler.probabilities(parameters);energy=px@sampler.vx+pz@sampler.vz
  task=TFIMTask(3,3,2,True,1) if kind=='grid' else TFIMTask(1,6,2,False,-1)
  for i in [0,19]:max_error=max(max_error,abs(float(energy[i])-task.energy(parameters[i],j,-.5)))
  save(artifact,energy=energy)
  rows.append(dict(task=kind,J=j,method=method,seed=seed,total_shots=budget,mean_energy=float(energy.mean()),endpoints=20))
 for seed in proto['grid_confirmation_seeds']:
  base=HERE/'grid'/f'seed{seed}'
  for method in ['product','fourier']:
   with np.load(base/f'{method}_search.npz') as a:params=a['parameters']
   score('grid',-.5,method,seed,4000000,params,base/f'{method}_score.npz')
  for method in ['spsa','cobyla']:
   for b in GRID_BUDGETS:
    with np.load(base/method/f'budget{b}.npz') as a:params=a['parameters']
    score('grid',-.5,method,seed,b,params,base/method/f'budget{b}_score.npz')
 for seed in proto['chain_confirmation_seeds']:
  base=HERE/'chain'/f'seed{seed}'
  for j in [-.5,.5]:
   for method in ['product','local_only','spsa']:
    for b in [578000,1132000,1686000]:
     folder=base/method/f'J{j:+.1f}'
     with np.load(folder/f'budget{b}.npz') as a:params=a['parameters']
     score('chain',j,method,seed,b,params,folder/f'budget{b}_score.npz')
 assert max_error<1e-11
 summary=[];contrasts=[]
 for key in sorted(set((r['task'],r['J'],r['total_shots']) for r in rows)):
  task,j,budget=key;methods=sorted(set(r['method'] for r in rows if (r['task'],r['J'],r['total_shots'])==key))
  for method in methods:
   group=sorted((r for r in rows if (r['task'],r['J'],r['total_shots'],r['method'])==(*key,method)),key=lambda r:r['seed']);assert len(group)==10
   summary.append(dict(task=task,J=j,total_shots=budget,method=method,**describe([r['mean_energy'] for r in group])))
  if 'product' in methods:
   vals={method:[r['mean_energy'] for r in sorted((r for r in rows if (r['task'],r['J'],r['total_shots'],r['method'])==(*key,method)),key=lambda r:r['seed'])] for method in methods}
   for other in methods:
    if other=='product':continue
    delta=np.array(vals['product'])-vals[other]
    contrasts.append(dict(task=task,J=j,total_shots=budget,comparison='product-minus-'+other,paired_wins=int((delta<0).sum()),**describe(delta)))
 write(HERE/'rows.json',rows);write(HERE/'summary.json',summary);write(HERE/'contrasts.json',contrasts)
 with (HERE/'rows.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write(HERE/'verification.json',dict(passed=True,grid_datasets=10,chain_datasets=10,full_product_fits=20,scored_method_budget_cells=len(rows),retained_endpoints_scored=len(rows)*20,independent_qiskit_energy_max_error=max_error,measurements_audited=measurement_rows,exact_scores_after_complete_preflight=True,hardware_shots=0,confidence_intervals='paired t95 across10 independent dataset means; unadjusted descriptive intervals'))
 write(HERE/'status.json',dict(stage='complete',all_audits_passed=True))
 for r in summary:
  if r['task']=='grid' or r['total_shots']==1686000:print(r['task'],r['J'],r['total_shots'],r['method'],r['mean'],r['sd'],flush=True)
 print('ALL AUDITS AND EXACT SCORES COMPLETE',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true',help='Replay distributed records; archive timestamps do not encode original generation order')
 main(ap.parse_args().replay)
