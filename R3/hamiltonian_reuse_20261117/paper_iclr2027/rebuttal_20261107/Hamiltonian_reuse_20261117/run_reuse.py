"""Sequential Hamiltonian workloads with method-local historical measurement reuse."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;WORK=HERE.parents[2];BASE=EXP/'JJuz_followup_20261117'
sys.path[:0]=[str(HERE),str(BASE),str(EXP),str(WORK/'world_model_surrogate'),str(WORK/'world_model_surrogate/scripts')]
from history import HistoryCache
from engine import PhysicalSampler,grid_dataset,shadow_dataset,save,write,wrap
from src.wmps.models.periodic_redq import ProductMomentREDQ
ORDERS={'smooth':[-.05,-.10,-.15,-.20,-.25,-.30,-.35,-.40,-.45,-.50],
        'jumping':[-.05,-.50,-.10,-.45,-.15,-.40,-.20,-.35,-.25,-.30]}
CAPS=[0,160000,640000,1280000]

def data(kind,seed,dev=False):
 path=(HERE/'development_data'/f'{kind}{seed}.npz') if dev else BASE/kind/f'seed{seed}'/'dataset.npz'
 if dev and not path.exists():save(path,**(grid_dataset(seed) if kind=='grid' else shadow_dataset(seed)))
 with np.load(path) as a:return {k:a[k] for k in a.files}

def warm(cache,j,previous,rng):
 center=cache.best(j);inc=np.broadcast_to(center,(20,*cache.shape)).copy()
 if previous is not None:
  old,ov=cache.evaluate(previous,j);base,bv=cache.evaluate(inc,j);take=old+2*np.sqrt(ov)<base+2*np.sqrt(bv)
  inc[take]=previous[take]
 x=inc.copy()
 flat=x.reshape(20,-1)
 if cache.kind=='grid':flat[1:,cache.active]+=rng.normal(0,.05,(19,len(cache.active)))
 else:flat[1:,cache.active]+=rng.uniform(-.2,.2,(19,len(cache.active)))
 x=wrap(flat).reshape(x.shape)
 return x,inc,center

def spsa_sequence(kind,seed,order,regime,cap,config,dest,dev=False):
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 started=time.perf_counter();d=data(kind,seed,dev);cache=HistoryCache(kind,d,regime)
 rng=np.random.default_rng(seed+400000+(0 if order=='smooth' else 10000)+(0 if regime=='low' else 1000)+cap)
 previous=None;rows=[];all_final=[];gradient_records=[]
 for target,j in enumerate(ORDERS[order]):
  cache.current_j=j;cache.target_index=target
  init_rng=np.random.default_rng(seed+500000+target*100)
  x,inc,center=warm(cache,j,previous,init_rng);initial=x.copy();before=cache.online_shots;iteration=0
  no_query=inc.copy();directions=np.random.default_rng(seed+600000+target*100)
  n=config['shots'];reserve=40*n
  if cap:
   while iteration<2000:
    delta=directions.choice([-1.,1.],(20,len(cache.active)));c=config['c']/(iteration+1)**.101
    xp=x.reshape(20,-1).copy();xm=xp.copy();xp[:,cache.active]+=c*delta;xm[:,cache.active]-=c*delta
    controls=wrap(np.r_[xp,xm]).reshape(-1,*cache.shape)
    needed=cache.required_shots(controls,n)
    if cache.online_shots-before+needed+reserve>cap:break
    mean,var=cache.request(controls,n,rng,'gradient');iteration+=1
    grad=(mean[:20]-mean[20:])[:,None]*delta/(2*c);gain=config['gain']*(11/(iteration+10))**.602
    flat=x.reshape(20,-1).copy();flat[:,cache.active]-=gain*grad;x=wrap(flat).reshape(x.shape)
    gradient_records.append(dict(target=target,event=len(cache.events)-1,c=c,gain=gain,delta=delta.copy(),after=x.copy()))
   read=np.r_[x,inc];required=cache.required_shots(read,n)
   assert cache.online_shots-before+required<=cap
   mean,var=cache.request(read,n,rng,'retention');z=0. if kind=='grid' else 2.
   take=mean[:20]+z*np.sqrt(var[:20]+var[20:])<mean[20:]
   inc=np.where(take[:,None,None],x,inc)
  counters=cache.counters();assert cache.online_shots-before<=cap
  save(dest/f'target{target:02d}.npz',parameters=inc,initial_iterates=initial,initial_incumbents=no_query,final_iterates=x,center=center)
  rows.append(dict(target=target,J=j,online_cap=cap,online_used_for_target=cache.online_shots-before,iterations=iteration,event_end=len(cache.events),**counters))
  all_final.append(inc.copy());previous=inc.copy()
 cache.save(dest)
 if gradient_records:
  save(dest/'updates.npz',**{k:np.array([r[k] for r in gradient_records]) for k in gradient_records[0]})
 else:save(dest/'updates.npz',target=np.zeros(0,int),event=np.zeros(0,int),c=np.zeros(0),gain=np.zeros(0),delta=np.zeros((0,20,len(cache.active))),after=np.zeros((0,20,*cache.shape)))
 validation=None
 if dev:
  # A third, independent readout evaluates selected endpoints after the entire
  # sequence. These labels never feed retention or later Hamiltonians.
  evaluator=PhysicalSampler(kind,-.5);parameters=np.concatenate(all_final)
  record=evaluator.measure(parameters,1000,np.random.default_rng(seed+900000),'development_holdout')
  evaluator.save(dest/'development_holdout.npz');u=evaluator.terms(record['counts_z'],record['counts_x']);hx=-.5 if kind=='grid' else .5
  target_j=np.repeat(ORDERS[order],20);hz=target_j if kind=='grid' else -target_j
  energies=hx*u[:,:evaluator.n].sum(1)+hz*u[:,evaluator.n:].sum(1)
  validation=dict(mean_independent_measured_energy=float(energies.mean()),extra_development_shots=evaluator.shots,used_for_search=False)
 write(dest/'complete.json',dict(kind=kind,seed=seed,order=order,regime=regime,online_cap_per_target=cap,config=config,rows=rows,validation=validation,seconds=time.perf_counter()-started,exact_scoring_started=False))

def cqs_sequence(kind,seed,order,dest):
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 started=time.perf_counter();d=data(kind,seed);cache=HistoryCache(kind,d,'shared');engine=cache.engine
 path=BASE/kind/f'seed{seed}'/'product.pt';before=hashlib.sha256(path.read_bytes()).hexdigest()
 model=ProductMomentREDQ(engine.n,engine.edges,2,kind=='grid',32,4 if kind=='grid' else 1).double().eval()
 model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
 for p in model.parameters():p.requires_grad_(False)
 previous=None;rows=[]
 for target,j in enumerate(ORDERS[order]):
  center=cache.best(j);starts=np.broadcast_to(center,(20,*cache.shape)).copy();rng=np.random.default_rng(seed+500000+target*100)
  flat=starts.reshape(20,-1)
  if kind=='grid':flat[1:,cache.active]+=rng.normal(0,.05,(19,len(cache.active)))
  else:flat[1:,cache.active]+=rng.uniform(-.2,.2,(19,len(cache.active)))
  starts=wrap(flat).reshape(starts.shape)
  coeff=np.r_[np.full(engine.n,-.5 if kind=='grid' else .5),np.full(len(engine.edges),j if kind=='grid' else -j)]
  ct=torch.tensor(coeff)
  with torch.no_grad():
   if previous is not None:
    old=(model(torch.tensor(previous))[0]@ct).numpy();cold=(model(torch.tensor(starts))[0]@ct).numpy();starts=np.where((old<cold)[:,None,None],previous,starts)
  x=torch.tensor(starts,dtype=torch.float64,requires_grad=True);opt=torch.optim.Adam([x],lr=.1,foreach=False)
  for step in range(101):
   opt.zero_grad();pred=model(x)[0]@ct
   if step==100:break
   pred.sum().backward();opt.step()
   with torch.no_grad():x.copy_((x+np.pi)%(2*np.pi)-np.pi)
  final=x.detach().numpy();previous=final.copy()
  save(dest/f'target{target:02d}.npz',parameters=final,prediction=pred.detach().numpy(),starts=starts,center=center)
  rows.append(dict(target=target,J=j,**cache.counters()))
 after=hashlib.sha256(path.read_bytes()).hexdigest();assert before==after
 write(dest/'complete.json',dict(kind=kind,seed=seed,order=order,method='fixed_cqs',rows=rows,global_model_hash_before=before,global_model_hash_after=after,refits=0,seconds=time.perf_counter()-started,exact_scoring_started=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['grid','chain'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--order',choices=list(ORDERS),default='smooth');ap.add_argument('--regime',choices=['low','shared'],default='low');ap.add_argument('--cap',type=int,default=640000);ap.add_argument('--dev',action='store_true');ap.add_argument('--shots',type=int,default=100);ap.add_argument('--cqs',action='store_true');a=ap.parse_args();torch.set_num_threads(2)
 if a.cqs:cqs_sequence(a.kind,a.seed,a.order,HERE/'runs'/a.kind/f'seed{a.seed}'/a.order/'cqs')
 else:
  if a.dev:
   conf=dict(gain=.05 if a.kind=='grid' else .2,c=.05 if a.kind=='grid' else .2,shots=a.shots)
   dest=HERE/'development'/a.kind/f'seed{a.seed}'/f'{a.regime}_shots{a.shots}'
  else:
   conf=json.loads((HERE/'frozen_selection.json').read_text())[a.kind+'_'+a.regime]['config']
   dest=HERE/'runs'/a.kind/f'seed{a.seed}'/a.order/f'{a.regime}_cap{a.cap}'
  spsa_sequence(a.kind,a.seed,a.order,a.regime,a.cap,conf,dest,a.dev)
