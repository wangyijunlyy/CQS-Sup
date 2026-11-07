"""Repeat the exact32-start ansatz diagnostic; this is not a shot-budgeted learner."""
from pathlib import Path
import os,json,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'paper_iclr2027/rebuttal_20261107')]
import numpy as np
import torch
from scipy.optimize import minimize
from run_evidence import TFIMTask,ChainExact
def main():
    torch.set_num_threads(2);task=TFIMTask(1,6,2,False,-1);oracle=ChainExact(task)
    reference=np.zeros((2,11));reference[0,6:9]=np.pi
    assert abs(task.energy(reference,-.1,-.5)+3)<1e-12
    def objective(flat):
        x=torch.tensor(flat.reshape(1,2,11),requires_grad=True);y=oracle(x,-.1,-1).sum()
        gradient=torch.autograd.grad(y,x)[0]
        return float(y.detach()),gradient.detach().numpy().ravel()
    starts=np.random.default_rng(20261107).uniform(-np.pi,np.pi,(32,2,11));starts[0]=reference
    results=[minimize(objective,x.ravel(),method='BFGS',jac=True,options=dict(maxiter=1000,gtol=1e-9)) for x in starts]
    best=min(results,key=lambda r:r.fun);energy=task.energy(best.x.reshape(2,11),-.1,-.5)
    assert abs(energy-best.fun)<1e-10
    lower,upper=task.spectrum_bounds(-.1,-.5)
    out=ROOT/'outputs/reference';out.mkdir(parents=True,exist_ok=True)
    np.save(out/'best_found_ansatz_controls.npy',best.x.reshape(2,11))
    result=dict(seed=20261107,starts=32,BFGS_maxiter=1000,BFGS_gtol=1e-9,ground_lower_bound=lower,
        best_found_ansatz_energy=energy,ansatz_bracket_width=energy-lower,global_ansatz_certificate=False,
        restart_energies=[float(r.fun) for r in results])
    (out/'reference.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
