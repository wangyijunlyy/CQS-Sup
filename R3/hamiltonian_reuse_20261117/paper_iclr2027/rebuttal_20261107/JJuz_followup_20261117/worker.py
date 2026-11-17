"""Full-model grid and independent chain runs; final exact scoring is separate."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;WORK=HERE.parents[2];WORLD=WORK/'world_model_surrogate'
sys.path[:0]=[str(EXP),str(WORLD),str(WORLD/'scripts')]
from engine import PhysicalSampler,grid_dataset,shadow_dataset,save,write,wrap
from direct import spsa,cobyla,SPSA_CONFIGS,COBYLA_CONFIGS
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ
from run_fspt8_fair import fit
from run_vqe_fourier_controls import FourierBasis,ridge_path,RIDGE_GRID
from local_refinement import optimize
from chain_local_calibration import fit_correction,torch_features

def grid_pilot(seed):
 e=PhysicalSampler('grid',-.5);x=np.random.default_rng(seed).uniform(-np.pi,np.pi,(400,2,2))
 r=e.measure(x[:20],1000,np.random.default_rng(seed+2000),'pilot')
 center=x[int(np.argmin(r['energy']))].copy();starts=np.broadcast_to(center,(20,2,2)).copy().reshape(20,4)
 starts[:,e.active]+=np.random.default_rng(seed+9000).normal(0,.05,(20,3))
 return center,wrap(starts).reshape(20,2,2)

def dev(task,seed,method,index):
 dest=HERE/'development'/task/f'seed{seed}'/f'{method}{index}'
 if task=='grid':
  center,starts=grid_pilot(seed);config=(SPSA_CONFIGS if method=='spsa' else COBYLA_CONFIGS)[index]
  (spsa('grid',-.5,starts,center,seed+700000,config,dest) if method=='spsa' else cobyla(-.5,starts,center,seed+700000,config,dest))
 else:
  center,starts=chain_starts(seed)
  for j in [-.5,.5]:spsa('chain',j,starts,center,seed+710000+int(j>0),SPSA_CONFIGS[index],dest/f'J{j:+.1f}')

def grid(seed):
 base=HERE/'grid'/f'seed{seed}';base.mkdir(parents=True,exist_ok=True)
 if not(base/'dataset.npz').exists():save(base/'dataset.npz',**grid_dataset(seed))
 with np.load(base/'dataset.npz') as a:d={k:a[k] for k in a.files}
 center,starts=grid_pilot(seed);np.testing.assert_array_equal(center,d['center']);np.testing.assert_array_equal(starts,d['starts'])
 settings=json.loads((HERE/'frozen_selection.json').read_text());e=PhysicalSampler('grid',-.5)
 c=np.r_[np.full(9,-.5),np.full(12,-.5)];x,y=d['X'],d['measured_terms']
 if not(base/'surrogates_saved.json').exists():
  if not(base/'fit.json').exists():
   _,detail=fit(lambda:ProductMomentREDQ(9,e.edges,2,True,32,4),x[:160],y[:160],x[160:200],y[160:200],seed+1000,'cpu',base,'product',3000,min_epochs=500,patience=100)
   write(base/'fit.json',detail)
  model=ProductMomentREDQ(9,e.edges,2,True,32,4).double().eval();model.load_state_dict(torch.load(base/'product.pt',map_location='cpu',weights_only=True))
  for p in model.parameters():p.requires_grad_(False)
  ct=torch.tensor(c);params,pred=optimize(lambda z:model(z)[0]@ct,starts,center,None)
  save(base/'product_search.npz',parameters=params,prediction=pred,starts=starts)
  best=None;rows=[]
  for q in (1,2,3,4):
   basis=FourierBasis(2,q)
   with torch.no_grad():a=basis(torch.tensor(x[:160])).numpy();b=basis(torch.tensor(x[160:200])).numpy()
   weights=ridge_path(a,y[:160],RIDGE_GRID)
   for reg,w in zip(RIDGE_GRID,weights):
    ew=w@c;score=float(np.mean((b@ew-y[160:200]@c)**2));rows.append(dict(q=q,regularizer=float(reg),validation_mse=score))
    if best is None or score<best[0]:best=(score,q,ew)
  _,q,w=best;basis=FourierBasis(2,q);wt=torch.tensor(w)
  params,pred=optimize(lambda z:basis(z)@wt,starts,center,None)
  save(base/'fourier_search.npz',parameters=params,prediction=pred,starts=starts)
  save(base/'fourier_fit.npz',weights=w,q=np.array(q));write(base/'fourier_selection.json',dict(candidates=rows,selected_q=q,selected_validation_mse=best[0]))
  write(base/'surrogates_saved.json',dict(full_product_fit=True,train=160,validation=40,shots=4000000,all_endpoints_retained=True,exact_scoring_started=False))
 spsa('grid',-.5,starts,center,seed+700000,settings['grid_spsa']['config'],base/'spsa')
 cobyla(-.5,starts,center,seed+700000,settings['grid_cobyla']['config'],base/'cobyla')
 write(base/'complete.json',dict(seed=seed,shots=4000000,pilot_shots_included=40000,exact_scoring_started=False))

def chain_starts(seed):
 center=np.zeros((2,11));center[0,6:9]=np.pi
 starts=np.broadcast_to(center,(20,2,11)).copy().reshape(20,22)
 starts[1:,6:]+=np.random.default_rng(seed+9000).uniform(-.2,.2,(19,16))
 return center,starts.reshape(20,2,11)

def local_propose(prior,center,radius,w,rng,initial=None):
 active=np.arange(6,22);z0=rng.uniform(-1,1,(20,16));z0[0]=0
 if initial is not None:z0=(initial.reshape(20,22)[:,active]-center.reshape(1,22)[:,active])/(radius/4)
 z=torch.tensor(z0,dtype=torch.float64,requires_grad=True);ct=torch.tensor(center.reshape(1,-1));weights=torch.tensor(w)
 opt=torch.optim.Adam([z],lr=.03,foreach=False)
 for _ in range(100):
  opt.zero_grad();x=ct.expand(20,-1).clone();x[:,active]=ct[:,active]+z*(radius/4)
  value=prior(x.reshape(20,2,11))+torch_features(z)@weights;value.sum().backward();opt.step()
  with torch.no_grad():z.clamp_(-1,1)
 with torch.no_grad():
  x=ct.expand(20,-1).clone();x[:,active]=ct[:,active]+z*(radius/4)
  pred=prior(x.reshape(20,2,11))+torch_features(z)@weights
 return x.reshape(20,2,11).numpy(),pred.numpy(),z0

def local(seed,j,method,model,base):
 dest=base/method/f'J{j:+.1f}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 e=PhysicalSampler('chain',j);rng=np.random.default_rng(seed+800000+int(j>0));center,starts=chain_starts(seed)
 coeff=torch.tensor(np.r_[np.full(6,.5),np.full(5,-j)])
 prior=(lambda x:model(x)[0]@coeff) if method=='product' else (lambda x:torch.zeros(len(x),dtype=x.dtype))
 points=[];prefix=24000 if method=='product' else 0
 for round_,radius in enumerate([.8,.6,.4],1):
  z=rng.uniform(-1,1,(256,16));controls=np.broadcast_to(center,(256,2,11)).copy().reshape(256,22);controls[:,e.active]+=z*(radius/4);controls=controls.reshape(256,2,11)
  shots=np.full(256,1000)
  # The no-prior method spends the saved acquisition budget on extra local labels.
  if method=='local_only' and round_==1:shots[:12]+=1000
  train=e.measure(controls,shots,rng,'local_labels');inc=e.measure(center[None],1000,rng,'local_incumbent')
  with torch.no_grad():pp=prior(torch.tensor(controls)).numpy()
  w,detail=fit_correction(z,train['energy'],pp)
  candidates,pred,zstart=local_propose(prior,center,radius,w,rng,starts if round_==1 else None)
  read=e.measure(candidates,1000,rng,'local_candidate');take=read['energy']+2*np.sqrt(read['variance']+inc['variance'][0])<inc['energy'][0]
  kept=np.where(take[:,None,None],candidates,center[None]);measured=np.where(take,read['energy'],inc['energy'][0])
  budget=prefix+e.shots;assert budget==24000+554000*round_
  save(dest/f'budget{budget}.npz',parameters=kept,candidate_parameters=candidates,prediction=pred,accepted=take,estimated_energy=measured,training_coordinates=z,local_weights=w,center=center,initial_standardized=zstart)
  write(dest/f'round{round_}.json',detail);center=kept[int(np.argmin(measured))].copy()
  points.append(dict(budget=budget,acquisition_prefix=prefix,online_shots=e.shots,total_shots=budget,measured_score=float(measured.mean())))
 e.save(dest/'measurements.npz');write(dest/'complete.json',dict(seed=seed,J=j,method=method,points=points,exact_scoring_started=False))

def chain(seed):
 base=HERE/'chain'/f'seed{seed}';base.mkdir(parents=True,exist_ok=True)
 if not(base/'dataset.npz').exists():save(base/'dataset.npz',**shadow_dataset(seed))
 with np.load(base/'dataset.npz') as a:x,y=a['X'],a['measured_terms']
 task=TFIMTask(1,6,2,False,-1)
 if not(base/'fit.json').exists():
  _,detail=fit(lambda:ProductMomentREDQ(6,task.edges,2,False,32,1),x[:2000],y[:2000],x[2000:2400],y[2000:2400],seed,'cpu',base,'product',300,patience=30)
  write(base/'fit.json',detail)
 model=ProductMomentREDQ(6,task.edges,2,False,32,1).double().eval();model.load_state_dict(torch.load(base/'product.pt',map_location='cpu',weights_only=True))
 for p in model.parameters():p.requires_grad_(False)
 settings=json.loads((HERE/'frozen_selection.json').read_text());center,starts=chain_starts(seed);save(base/'starts.npz',center=center,starts=starts)
 for j in [-.5,.5]:
  for method in ['product','local_only']:local(seed,j,method,model,base)
  spsa('chain',j,starts,center,seed+710000+int(j>0),settings['chain_spsa']['config'],base/'spsa'/f'J{j:+.1f}')
 write(base/'complete.json',dict(seed=seed,global_acquisition_shots=24000,independent_full_fit=True,exact_scoring_started=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['grid','chain','dev-grid','dev-chain'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--method',default='spsa');ap.add_argument('--index',type=int,default=0);args=ap.parse_args();torch.set_num_threads(2)
 if args.task.startswith('dev-'):dev(args.task[4:],args.seed,args.method,args.index)
 else:globals()[args.task](args.seed)
