"""Measured SPSA and parameter-shift Adam on the full two-layer chain."""
from pathlib import Path
import os,sys,json,argparse,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'evidence/resource_controls_20261108/chain'
ACTIVE=np.arange(6,22)
BUDGETS=(24000,578000,1132000,1686000)

def shot_cost(controls,shots):return int(2*controls*shots)

class ChainMeasurement:
    def __init__(self,shots=100):
        self.default_shots=shots;self.shots=0;self.records=[]
        self.indices=np.arange(64);self.signs=1-2*((self.indices[:,None]>>np.arange(6))&1)
        self.edges=[(i,i+1) for parity in (0,1) for i in range(parity,5,2)]
        self.zz=np.stack([self.signs[:,i]*self.signs[:,j] for i,j in self.edges],axis=1)
    def probabilities(self,x):
        x=np.asarray(x).reshape(-1,2,11);state=np.ones((len(x),64),complex)/8
        for layer in range(2):
            for i in range(6):
                t=x[:,layer,i,None]/2
                state=np.cos(t)*state-1j*np.sin(t)*state[:,self.indices^(1<<i)]
            for e in range(5):state*=np.exp(-.5j*x[:,layer,6+e,None]*self.zz[:,e])
        pz=abs(state)**2
        for i in range(6):state=(state[:,self.indices&~(1<<i)]+self.signs[:,i]*state[:,self.indices|(1<<i)])/np.sqrt(2)
        return pz,abs(state)**2
    def x_values(self,J):return .5*self.signs.sum(1)
    def z_values(self,J):return -J*self.zz.sum(1)
    def measure(self,x,J,rng,shots=None):
        shots=self.default_shots if shots is None else int(shots)
        pz,px=self.probabilities(x)
        cz=np.stack([rng.multinomial(shots,p/p.sum()) for p in pz]);cx=np.stack([rng.multinomial(shots,p/p.sum()) for p in px])
        vx,vz=self.x_values(J),self.z_values(J)
        energy=(cx@vx+cz@vz)/shots
        variance=(np.maximum(0,cx@vx**2/shots-(cx@vx/shots)**2)+np.maximum(0,cz@vz**2/shots-(cz@vz/shots)**2))/(shots-1)
        record={'parameters':np.asarray(x).copy(),'counts_x':cx.astype(np.uint16),'counts_z':cz.astype(np.uint16),
            'energy':energy,'variance':variance,'shots_per_group':np.full(len(x),shots,dtype=int)}
        self.records.append(record);self.shots+=shot_cost(len(x),shots)
        return record
    def save(self,path):
        np.savez_compressed(path,**{key:np.concatenate([r[key] for r in self.records]) for key in self.records[0]})

def shifted_gradients(engine,x,J,rng):
    x=np.asarray(x);shift=np.zeros((16,22));shift[np.arange(16),ACTIVE]=np.pi/2
    controls=x.reshape(-1,1,22)+np.stack([shift,-shift],axis=1).reshape(1,32,22)
    energies=engine.measure(controls.reshape(-1,2,11),J,rng)['energy'].reshape(len(x),16,2)
    return .5*(energies[:,:,0]-energies[:,:,1])

def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi

def run(seed,J,method):
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/f'seed{seed}_{method}_J{J:+.1f}';dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():return
    if (dest/'started.json').exists():raise RuntimeError('Incomplete run preserved; use a fresh run path')
    (dest/'started.json').write_text(json.dumps({'seed':seed,'J':J,'method':method}))
    engine=ChainMeasurement(100);rng=np.random.default_rng(800000+seed*100+int(round((J+.5)*10)))
    x=np.zeros((20,2,11));x[:,0,6:9]=np.pi;incumbent=x.copy();m=np.zeros((20,16));v=m.copy();iteration=0
    points=[];trace=[];start=time.perf_counter()
    for budget in BUDGETS:
        step_controls=40 if method=='spsa' else 640
        reserve=shot_cost(40,100)
        while engine.shots+shot_cost(step_controls,100)+reserve<=budget:
            iteration+=1
            if method=='spsa':
                delta=rng.choice([-1.,1.],size=(20,16));c=.1/iteration**.101
                xp=x.copy().reshape(20,22);xm=x.copy().reshape(20,22)
                xp[:,ACTIVE]+=c*delta;xm[:,ACTIVE]-=c*delta
                record=engine.measure(wrap(np.concatenate([xp,xm])).reshape(-1,2,11),J,rng)
                grad=(record['energy'][:20]-record['energy'][20:])[:,None]*delta/(2*c)
                gain=.1*(11/(iteration+10))**.602
                flat=x.reshape(20,22).copy();flat[:,ACTIVE]-=gain*grad;x=wrap(flat).reshape(20,2,11)
            else:
                grad=shifted_gradients(engine,x,J,rng);m=.9*m+.1*grad;v=.999*v+.001*grad**2
                flat=x.reshape(20,22).copy();flat[:,ACTIVE]-=.03*(m/(1-.9**iteration))/(np.sqrt(v/(1-.999**iteration))+1e-8)
                x=wrap(flat).reshape(20,2,11)
            trace.append(x.copy())
        # Use the residual budget for fresh independent candidate/incumbent readout.
        verification_shots=(budget-engine.shots)//80
        assert verification_shots>=100
        candidate=engine.measure(x,J,rng,verification_shots);old=engine.measure(incumbent,J,rng,verification_shots)
        accepted=candidate['energy']+2*np.sqrt(candidate['variance']+old['variance'])<old['energy']
        incumbent=np.where(accepted[:,None,None],x,incumbent);x=incumbent.copy()
        np.savez_compressed(dest/f'budget{budget}.npz',parameters=incumbent,accepted=accepted,
            candidate_energy=candidate['energy'],candidate_variance=candidate['variance'],
            incumbent_energy=old['energy'],incumbent_variance=old['variance'])
        points.append(dict(budget=budget,actual_shots=engine.shots,iterations=iteration,accepted=int(accepted.sum()),
            final_verification_shots_per_group=verification_shots))
    engine.save(dest/'measurements.npz');np.save(dest/'iterate_parameters.npy',np.array(trace))
    (dest/'complete.json').write_text(json.dumps(dict(seed=seed,J=J,method=method,points=points,
        seconds=time.perf_counter()-start,hardware_shots=0,exact_scoring_started=False),indent=2)+'\n')
    print('DIRECT SAVED',seed,J,method,'seconds',round(time.perf_counter()-start,1),'updates',iteration,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);args=ap.parse_args()
    for J in (-.5,-.3,-.1,.1,.3,.5):
        for method in ('spsa','parameter_shift_adam'):run(args.seed,J,method)

if __name__=='__main__':main()
