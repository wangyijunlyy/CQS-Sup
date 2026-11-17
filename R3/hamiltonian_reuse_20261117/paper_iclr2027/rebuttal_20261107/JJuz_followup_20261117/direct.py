"""Finite-shot SPSA/COBYLA; all candidate and incumbent readouts are charged."""
import time
import numpy as np
from scipy.optimize import minimize
from engine import PhysicalSampler,wrap,save,write

GRID_BUDGETS=(400000,800000,2000000,4000000)
CHAIN_BUDGETS=(24000,578000,1132000,1686000)
SPSA_CONFIGS=[dict(gain=.05,c=.05,shots=100),dict(gain=.1,c=.1,shots=100),dict(gain=.2,c=.2,shots=100),dict(gain=.1,c=.1,shots=500),dict(gain=.1,c=.1,shots=1000)]
COBYLA_CONFIGS=[dict(rhobeg=.1,shots=100),dict(rhobeg=.3,shots=100),dict(rhobeg=.3,shots=500),dict(rhobeg=.3,shots=1000)]

def retain(engine,candidate,incumbent,J,rng,shots,z=2.):
 a=engine.measure(candidate,shots,rng,'candidate_readout');b=engine.measure(incumbent,shots,rng,'incumbent_readout')
 take=a['energy']+z*np.sqrt(a['variance']+b['variance']) < b['energy']
 out=np.where(take[:,None,None],candidate,incumbent)
 estimated=np.where(take,a['energy'],b['energy'])
 return out,estimated,take

def spsa(kind,J,starts,center,seed,config,dest):
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 started=time.perf_counter();engine=PhysicalSampler(kind,J);rng=np.random.default_rng(seed)
 prefix=40000 if kind=='grid' else 0;budgets=GRID_BUDGETS if kind=='grid' else CHAIN_BUDGETS
 x=starts.copy();inc=np.broadcast_to(center,x.shape).copy();iteration=0;trace=[];points=[]
 n=config['shots'];active=engine.active
 for budget in budgets:
  limit=budget-prefix;reserve=80*n
  while engine.shots+80*n+reserve<=limit:
   iteration+=1;delta=rng.choice([-1.,1.],(20,len(active)));c=config['c']/iteration**.101
   xp=x.reshape(20,-1).copy();xm=xp.copy();xp[:,active]+=c*delta;xm[:,active]-=c*delta
   r=engine.measure(wrap(np.r_[xp,xm]).reshape(-1,*engine.shape),n,rng,'spsa_gradient')
   grad=(r['energy'][:20]-r['energy'][20:])[:,None]*delta/(2*c)
   gain=config['gain']*(11/(iteration+10))**.602
   flat=x.reshape(20,-1).copy();flat[:,active]-=gain*grad;x=wrap(flat).reshape(-1,*engine.shape)
   trace.append(x.copy())
  verification=(limit-engine.shots)//80
  assert verification>=100 and (limit-engine.shots)%80==0
  inc,estimated,take=retain(engine,x,inc,J,rng,verification,z=0. if kind=='grid' else 2.)
  save(dest/f'budget{budget}.npz',parameters=inc,candidate_parameters=x,estimated_energy=estimated,accepted=take)
  points.append(dict(budget=budget,acquisition_prefix=prefix,online_shots=engine.shots,total_shots=engine.shots+prefix,iterations=iteration,measured_score=float(estimated.mean())))
  assert engine.shots+prefix==budget
  # Keep the exploratory optimizer trajectory after a rejected endpoint.
 engine.save(dest/'measurements.npz');save(dest/'trajectory.npz',parameters=np.array(trace),starts=starts,center=center)
 write(dest/'complete.json',dict(method='spsa',kind=kind,J=J,seconds=time.perf_counter()-started,acceptance_z=0 if kind=='grid' else 2,config=config,points=points,exact_scoring_started=False))

def cobyla(J,starts,center,seed,config,dest):
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 started=time.perf_counter();engine=PhysicalSampler('grid',J);rng=np.random.default_rng(seed);n=config['shots'];active=engine.active
 quotas=[(b-40000)//20 for b in GRID_BUDGETS]
 outputs={b:[] for b in GRID_BUDGETS};estimates={b:[] for b in GRID_BUDGETS};hist=[]
 for restart in range(20):
  offset=engine.shots;state=dict(k=0,best=starts[restart].copy(),best_value=np.inf,inc=center.copy(),calls=0)
  def checkpoint():
   budget=GRID_BUDGETS[state['k']];remaining=quotas[state['k']]-(engine.shots-offset)
   assert remaining%4==0 and remaining>=4*n
   inc,est,take=retain(engine,state['best'][None],state['inc'][None],J,rng,remaining//4,z=0.)
   state['inc']=inc[0];outputs[budget].append(inc[0].copy());estimates[budget].append(float(est[0]))
   hist.append(dict(restart=restart,budget=budget,online_shots=engine.shots-offset,calls=state['calls']))
   assert engine.shots-offset==quotas[state['k']]
   state['k']+=1;state['best_value']=np.inf
  def objective(v):
   while state['k']<len(quotas) and engine.shots-offset+2*n+4*max(n,1000)>quotas[state['k']]:checkpoint()
   if state['k']==len(quotas):raise StopIteration
   full=center.reshape(-1).copy();full[active]=v;full=wrap(full).reshape(2,2)
   r=engine.measure(full[None],n,rng,'cobyla_objective');value=float(r['energy'][0]);state['calls']+=1
   if value<state['best_value']:state['best_value']=value;state['best']=full.copy()
   return value
  start=starts[restart].reshape(-1)[active]
  while state['k']<len(quotas):
   try:
    result=minimize(objective,start,method='COBYLA',options=dict(rhobeg=config['rhobeg'],tol=.001,maxiter=100000))
    start=result.x+rng.uniform(-config['rhobeg'],config['rhobeg'],3)
   except StopIteration:break
  assert engine.shots-offset==quotas[-1]
 assert engine.shots+40000==4000000
 for budget in GRID_BUDGETS:save(dest/f'budget{budget}.npz',parameters=np.array(outputs[budget]),estimated_energy=np.array(estimates[budget]))
 engine.save(dest/'measurements.npz');save(dest/'starts.npz',starts=starts,center=center)
 points=[dict(budget=b,total_shots=b,acquisition_prefix=40000,measured_score=float(np.mean(estimates[b]))) for b in GRID_BUDGETS]
 write(dest/'complete.json',dict(method='cobyla',kind='grid',J=J,seconds=time.perf_counter()-started,acceptance_z=0,config=config,points=points,per_restart_checkpoints=hist,exact_scoring_started=False))
