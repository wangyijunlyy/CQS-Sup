"""Batched exact-circuit probabilities; only finite-shot records reach optimizers."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import json
import numpy as np

def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.json.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def save(path,**values):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.npz.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**values)
 tmp.replace(p)
def wrap(x):return (np.asarray(x)+np.pi)%(2*np.pi)-np.pi

class PhysicalSampler:
 def __init__(self,kind,J):
  self.kind=kind;self.J=J;self.n=9 if kind=='grid' else 6
  self.shape=(2,2) if kind=='grid' else (2,11)
  self.active=np.arange(1,4) if kind=='grid' else np.arange(6,22)
  self.edges=([(3*r+c,3*r+c+1) for r in range(3) for c in range(2)]+[(3*r+c,3*r+c+3) for r in range(2) for c in range(3)]) if kind=='grid' else [(i,i+1) for parity in (0,1) for i in range(parity,5,2)]
  # Grid edge order follows TFIMTask's row-major right/down construction.
  if kind=='grid':self.edges=[edge for r in range(3) for c in range(3) for edge in ([(3*r+c,3*r+c+1)] if c<2 else [])+([(3*r+c,3*r+c+3)] if r<2 else [])]
  self.idx=np.arange(2**self.n);self.signs=1-2*((self.idx[:,None]>>np.arange(self.n))&1)
  self.zz=np.stack([self.signs[:,i]*self.signs[:,j] for i,j in self.edges],axis=1)
  self.vx=(-.5 if kind=='grid' else .5)*self.signs.sum(1)
  self.vz=(J if kind=='grid' else -J)*self.zz.sum(1)
  self.shots=0;self.records=[]
 def state(self,controls):
  x=np.asarray(controls,dtype=float).reshape(-1,*self.shape)
  state=np.ones((len(x),len(self.idx)),complex)/np.sqrt(len(self.idx))
  # First RX gates act on |+>, giving a global phase only.
  for layer in range(2):
   if layer:
    for i in range(self.n):
     angle=x[:,layer,0 if self.kind=='grid' else i,None]/2
     state=np.cos(angle)*state-1j*np.sin(angle)*state[:,self.idx^(1<<i)]
   phase=x[:,layer,1,None]*self.zz.sum(1) if self.kind=='grid' else x[:,layer,self.n:]@self.zz.T
   state*=np.exp(-.5j*phase)
  return state
 def rotate(self,state,recipe):
  state=state.copy()
  for i,basis in enumerate(recipe):
   if basis==2:continue
   if basis==1:state*=np.where((self.idx>>i)&1,-1j,1.)
   state=(state[:,self.idx&~(1<<i)]+self.signs[:,i]*state[:,self.idx|(1<<i)])/np.sqrt(2)
  return state
 def probabilities(self,controls):
  state=self.state(controls);px=abs(self.rotate(state,np.zeros(self.n,int)))**2;pz=abs(state)**2
  return pz,px
 def terms(self,cz,cx):
  cz=np.atleast_2d(cz);cx=np.atleast_2d(cx)
  return np.c_[cx@self.signs/cx.sum(1)[:,None],cz@self.zz/cz.sum(1)[:,None]]
 def measure(self,controls,shots,rng,stage='query'):
  x=np.asarray(controls,dtype=float).reshape(-1,*self.shape);pz,px=self.probabilities(x)
  ns=np.broadcast_to(np.asarray(shots,dtype=int),(len(x),)).copy()
  cz=np.stack([rng.multinomial(int(n),p/p.sum()) for n,p in zip(ns,pz)]).astype(np.uint32)
  cx=np.stack([rng.multinomial(int(n),p/p.sum()) for n,p in zip(ns,px)]).astype(np.uint32)
  mx=cx@self.vx/ns;mz=cz@self.vz/ns
  variance=(np.maximum(0,cx@(self.vx**2)/ns-mx**2)+np.maximum(0,cz@(self.vz**2)/ns-mz**2))/np.maximum(1,ns-1)
  row=dict(parameters=x.copy(),counts_x=cx,counts_z=cz,shots_per_group=ns,energy=mx+mz,variance=variance,stage=np.full(len(x),stage))
  self.records.append(row);self.shots+=int(2*ns.sum());return row
 def save(self,path):save(path,**{k:np.concatenate([r[k] for r in self.records]) for k in self.records[0]})

def grid_dataset(seed):
 e=PhysicalSampler('grid',-.5);rng=np.random.default_rng(seed+2000)
 x=np.random.default_rng(seed).uniform(-np.pi,np.pi,(400,2,2))
 pilot=e.measure(x[:20],1000,rng,'pilot')
 extra=e.measure(x[:20],9000,rng,'training_topup');rest=e.measure(x[20:200],10000,rng,'training')
 cx=np.r_[pilot['counts_x']+extra['counts_x'],rest['counts_x']]
 cz=np.r_[pilot['counts_z']+extra['counts_z'],rest['counts_z']]
 center=x[int(np.argmin(pilot['energy']))].copy();starts=np.broadcast_to(center,(20,2,2)).copy().reshape(20,4)
 starts[:,e.active]+=np.random.default_rng(seed+9000).normal(0,.05,(20,3));starts=wrap(starts).reshape(20,2,2)
 return dict(X=x,counts_x=cx,counts_z=cz,measured_terms=e.terms(cz,cx),pilot_counts_x=pilot['counts_x'],pilot_counts_z=pilot['counts_z'],pilot_energy=pilot['energy'],center=center,starts=starts,shots_used=np.array(e.shots))

def shadow_dataset(seed,controls=None,shots=10):
 count=2400 if controls is None else controls;total=2800 if controls is None else controls
 e=PhysicalSampler('chain',-.5);x=np.random.default_rng(seed).uniform(-np.pi,np.pi,(total,2,11));states=e.state(x[:count])
 rng=np.random.default_rng(seed+2000);recipes=rng.integers(0,3,(count,shots,6),dtype=np.int8);bits=np.empty_like(recipes)
 flat=recipes.reshape(-1,6);codes=(flat*(3**np.arange(6))).sum(1)
 for code in np.unique(codes):
  ids=np.flatnonzero(codes==code);recipe=flat[ids[0]];rot=e.rotate(states[ids//shots],recipe);probs=abs(rot)**2;probs/=probs.sum(1)[:,None]
  cdf=np.cumsum(probs,axis=1);cdf[:,-1]=1.;out=(rng.random(len(ids))[:,None]>cdf).sum(1)
  bits.reshape(-1,6)[ids]=((out[:,None]>>np.arange(6))&1)
 signs=1.-2*bits
 y=np.c_[(3*signs*(recipes==0)).mean(1),np.stack([(9*signs[:,:,i]*signs[:,:,j]*(recipes[:,:,i]==2)*(recipes[:,:,j]==2)).mean(1) for i,j in e.edges],axis=1)]
 return dict(X=x,measured_terms=y,bits=bits,recipes=recipes,shots_used=np.array(count*shots))
