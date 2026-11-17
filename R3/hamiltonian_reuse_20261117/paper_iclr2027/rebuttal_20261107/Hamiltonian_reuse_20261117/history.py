"""Historical energy recombination and exact-control finite-shot query cache."""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'JJuz_followup_20261117'))
from engine import PhysicalSampler,save

class HistoryCache:
 def __init__(self,kind,data=None,regime='low'):
  self.kind=kind;self.engine=PhysicalSampler(kind,-.5);self.active=self.engine.active;self.shape=self.engine.shape
  self.x_bins=np.arange(-self.engine.n,self.engine.n+1,2);self.z_bins=np.arange(-len(self.engine.edges),len(self.engine.edges)+1,2)
  self.x_index=((self.engine.signs.sum(1)+self.engine.n)//2).astype(int);self.z_index=((self.engine.zz.sum(1)+len(self.engine.edges))//2).astype(int)
  self.points=[];self.keys={};self.raw_records=[];self.events=[];self.draw_count=0;self.online_shots=0;self.feedback_rounds=0;self.new_points=0;self.hits=0;self.deduplications=0
  self.initial_shots=0;self.initial_queries=0;self.target_index=0;self.current_j=-.5
  if data is not None:
   if kind=='grid':
    count=20 if regime=='low' else 200
    cx=data['pilot_counts_x'] if regime=='low' else data['counts_x'];cz=data['pilot_counts_z'] if regime=='low' else data['counts_z']
    self.initial_shots=int(cx.sum()+cz.sum());self.initial_queries=count
    for x,a,b in zip(data['X'][:count],cx,cz):self._add_group(self._ensure(x),self.compress(a[None],True)[0],self.compress(b[None],False)[0])
   elif regime=='shared':
    bits,recipes=data['bits'],data['recipes'];s=1.-2*bits
    u=(3*s*(recipes==0)).sum(2);v=np.stack([9*s[:,:,i]*s[:,:,j]*(recipes[:,:,i]==2)*(recipes[:,:,j]==2) for i,j in self.engine.edges],axis=2).sum(2)
    self.initial_shots=int(np.prod(bits.shape[:2]));self.initial_queries=len(bits)
    for i,x in enumerate(data['X'][:len(bits)]):
     index=self._ensure(x);mu,mv=u[i].mean(),v[i].mean();du=u[i]-mu;dv=v[i]-mv;n=len(du)
     self.points[index]['shadow']=np.array([mu,mv,(du@du)/(n*(n-1)),(dv@dv)/(n*(n-1)),(du@dv)/(n*(n-1))])
  ref=np.zeros(self.shape)
  if kind=='chain':ref[0,6:9]=np.pi
  self.reference=ref;idx=self._ensure(ref);self.points[idx]['analytic']=np.array([9.,0.]) if kind=='grid' else np.array([-6.,0.])
  self.initial_count=len(self.points)
 def compress(self,counts,x_group):
  index=self.x_index if x_group else self.z_index;size=len(self.x_bins if x_group else self.z_bins)
  return np.stack([np.bincount(index,weights=row,minlength=size) for row in counts]).astype(np.uint32)
 def key(self,x):return np.asarray(x,dtype=np.float64).reshape(-1)[self.active].tobytes()
 def _ensure(self,x):
  key=self.key(x)
  if key not in self.keys:
   self.keys[key]=len(self.points);self.points.append(dict(parameters=np.asarray(x).reshape(self.shape).copy(),cx=np.zeros(len(self.x_bins),np.uint64),cz=np.zeros(len(self.z_bins),np.uint64),analytic=None,shadow=None))
  return self.keys[key]
 def _add_group(self,index,cx,cz):
  self.points[index]['cx']+=cx.astype(np.uint64);self.points[index]['cz']+=cz.astype(np.uint64)
 def _missing(self,x,n):
  index=self.keys.get(self.key(x))
  if index is None:return int(n)
  row=self.points[index]
  if row['analytic'] is not None:return 0
  return max(0,int(n)-int(row['cx'].sum()))
 def required_shots(self,x,n):
  unique={self.key(t):t for t in np.asarray(x).reshape(-1,*self.shape)}
  return sum(2*self._missing(t,n) for t in unique.values())
 def request(self,x,n,rng,stage):
  x=np.asarray(x).reshape(-1,*self.shape);unique={self.key(t):t for t in x}
  self.hits+=sum(self._missing(t,n)==0 for t in x);self.deduplications+=len(x)-len(unique)
  ids=np.array([self._ensure(t) for t in x]);missing=[(self._ensure(t),t,self._missing(t,n)) for t in unique.values() if self._missing(t,n)>0]
  start=self.draw_count
  if missing:
   indices,controls,shots=zip(*missing);r=self.engine.measure(np.array(controls),np.array(shots),rng,stage);self.engine.records.pop()
   cx=self.compress(r['counts_x'],True);cz=self.compress(r['counts_z'],False)
   for index,a,b in zip(indices,cx,cz):self._add_group(index,a,b)
   self.feedback_rounds+=1;self.new_points+=len(indices);self.online_shots+=int(2*np.sum(shots))
   self.draw_count+=len(indices)
   self.raw_records.append(dict(point_indices=np.array(indices),parameters=np.array(controls),counts_x=cx,counts_z=cz,shots_per_group=np.array(shots),round_id=np.full(len(indices),self.feedback_rounds),target_index=np.full(len(indices),self.target_index)))
  end=self.draw_count
  mean,var=self.evaluate(x,self.current_j)
  self.events.append(dict(point_indices=ids,j=self.current_j,min_shots=n,stage=stage,target_index=self.target_index,draw_start=start,draw_end=end,round_id=self.feedback_rounds,online_shots=self.online_shots,energy=mean.copy(),variance=var.copy()))
  return mean,var
 def evaluate(self,x,j):
  hx=-.5 if self.kind=='grid' else .5;hz=j if self.kind=='grid' else -j;means=[];vars=[]
  for t in np.asarray(x).reshape(-1,*self.shape):
   r=self.points[self.keys[self.key(t)]];n=int(r['cx'].sum())
   if r['analytic'] is not None:mu,mv=r['analytic'];var=0.
   elif n:
    mu=float(r['cx']@self.x_bins/n);mv=float(r['cz']@self.z_bins/n)
    var=(hx*hx*max(0,float(r['cx']@(self.x_bins**2)/n-mu*mu))+hz*hz*max(0,float(r['cz']@(self.z_bins**2)/n-mv*mv)))/max(1,n-1)
   else:
    assert r['shadow'] is not None
    mu,mv,vu,vv,cov=r['shadow'];var=max(0,hx*hx*vu+hz*hz*vv+2*hx*hz*cov)
   means.append(hx*mu+hz*mv);vars.append(var)
  return np.array(means),np.array(vars)
 def best(self,j):
  valid=[r['parameters'] for r in self.points if r['analytic'] is not None or r['shadow'] is not None or r['cx'].sum()>0]
  x=np.array(valid);mu,var=self.evaluate(x,j);return x[int(np.argmin(mu+2*np.sqrt(var)))].copy()
 def save(self,dest):
  save(dest/'history_draws.npz',**({k:np.concatenate([r[k] for r in self.raw_records]) for k in self.raw_records[0]} if self.raw_records else dict(point_indices=np.zeros(0,int),parameters=np.zeros((0,*self.shape)),counts_x=np.zeros((0,len(self.x_bins)),np.uint32),counts_z=np.zeros((0,len(self.z_bins)),np.uint32),shots_per_group=np.zeros(0,int),round_id=np.zeros(0,int),target_index=np.zeros(0,int))))
  offsets=np.r_[0,np.cumsum([len(r['point_indices']) for r in self.events])]
  values=dict(point_parameters=np.array([r['parameters'] for r in self.points]),offsets=offsets)
  for k in ['point_indices','energy','variance']:values[k]=np.concatenate([r[k] for r in self.events]) if self.events else np.zeros(0)
  for k in ['j','min_shots','stage','target_index','draw_start','draw_end','round_id','online_shots']:values[k]=np.array([r[k] for r in self.events])
  save(dest/'history_events.npz',**values)
 def counters(self):
  return dict(initial_shots=self.initial_shots,online_shots=self.online_shots,cumulative_shots=self.initial_shots+self.online_shots,initial_queries=self.initial_queries,new_query_controls=self.new_points,cumulative_query_controls=self.initial_queries+self.new_points,feedback_rounds=self.feedback_rounds+(1 if self.initial_shots else 0),online_feedback_rounds=self.feedback_rounds,cache_hits=self.hits,deduplicated_evaluations=self.deduplications)
