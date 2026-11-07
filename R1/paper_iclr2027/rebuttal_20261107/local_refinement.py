"""Direct local refinement after measured-data initialization, with matched arms.

The global continuation schedule need not be appropriate for already good
measured starts. This tests that mechanism without adding measurements.
No search configuration is selected using exact endpoint energy.
"""
from pathlib import Path
import sys
import os
import json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_evidence import ROOT,OUT,TFIMTask
from search_improvements import summarize
from src.wmps.models.periodic_redq import ProductMomentREDQ
from run_vqe_fourier_controls import FourierBasis,ridge_path,RIDGE_GRID


def optimize(predict,starts,center,radius):
    x=torch.tensor(starts,dtype=torch.float64,requires_grad=True)
    c=torch.tensor(center,dtype=torch.float64)
    opt=torch.optim.Adam([x],lr=.1,foreach=False)
    for step in range(101):
        opt.zero_grad()
        y=predict(x)
        if step==100:break
        y.sum().backward();opt.step()
        with torch.no_grad():
            if radius is not None:
                d=torch.remainder(x-c+np.pi,2*np.pi)-np.pi
                x.copy_(c+d.clamp(-radius,radius))
            x.copy_(torch.remainder(x+np.pi,2*np.pi)-np.pi)
    return x.detach().numpy(),y.detach().numpy()


def main():
    torch.set_num_threads(2)
    folder=OUT/'search_improvements'
    rows=[]
    protocol=dict(couplings=[-.1,-.3,-.5],seeds=list(range(1001,1006)),
                  start_rules=['lowest_measured_train20','best_measured_training_plus_jitter'],
                  local_radius=[None,.25],jitter_std=.05,search='100 Adam steps, lr.1, no smoothing, torus wrapping',
                  rationale='Local refinement starts from measured low-energy controls; large initial smoothing may erase useful initialization.',
                  selection='full predefined intervention grid; no exact-energy selection',extra_measurements=0)
    (folder/'local_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for case,rc in [('grid2x3_l2',(2,3)),('grid3x3_l2',(3,3))]:
        task=TFIMTask(*rc,2,True,1)
        for seed in range(1001,1006):
            src=ROOT/f'results_supplementary/e3/confirmation/{case}_seed{seed}'
            dest=folder/f'{case}_seed{seed}'
            with np.load(src/'dataset.npz') as a:data={k:a[k] for k in a.files}
            model=ProductMomentREDQ(task.n,task.edges,2,True,32,4).double().eval()
            model.load_state_dict(torch.load(src/'product.pt',map_location='cpu',weights_only=True))
            for p in model.parameters():p.requires_grad_(False)
            paths=[]
            for q in (1,2,3,4):
                basis=FourierBasis(2,q)
                with torch.no_grad():
                    a=basis(torch.tensor(data['X'][:160],dtype=torch.float64)).numpy()
                    b=basis(torch.tensor(data['X'][160:200],dtype=torch.float64)).numpy()
                paths.append((basis,b,ridge_path(a,data['measured_terms'][:160],RIDGE_GRID)))
            for J in protocol['couplings']:
                c=torch.tensor(task.coefficients(J,-.5),dtype=torch.float64)
                measured=data['measured_terms'][:160]@c.numpy()
                target=data['measured_terms'][160:200]@c.numpy()
                best=None
                for basis,b,weights in paths:
                    for w in weights:
                        energyw=w@c.numpy()
                        val=np.mean((b@energyw-target)**2)
                        if best is None or val<best[0]:best=(val,basis,energyw)
                _,basis,fw=best
                wt=torch.tensor(fw,dtype=torch.float64)
                predictors=dict(product=lambda z:model(z)[0]@c,
                                fourier_energy_tuned=lambda z:basis(z)@wt)
                ranked=np.argsort(measured,kind='stable')
                center=data['X'][ranked[0]]
                rng=np.random.default_rng(seed+9000)
                jittered=center+rng.normal(0,.05,(20,2,2))
                jittered=np.remainder(jittered+np.pi,2*np.pi)-np.pi
                starts=dict(lowest_measured_train20=data['X'][ranked[:20]],best_measured_training_plus_jitter=jittered)
                for initialization,initial in starts.items():
                    radii=[None] if initialization=='lowest_measured_train20' else protocol['local_radius']
                    for radius in radii:
                        for method,predict in predictors.items():
                            final,pred=optimize(predict,initial,center,radius)
                            true=np.array([task.energy(z,J,-.5) for z in final])
                            initial_true=np.array([task.energy(z,J,-.5) for z in initial])
                            row=dict(case=case,seed=seed,J=J,method=method,initialization=initialization,
                                     boundary='direct_wrap' if radius is None else 'direct_wrap_radius.25',
                                     mean_energy=float(true.mean()),mean_prediction=float(pred.mean()),
                                     single_training_candidate=float(task.energy(center,J,-.5)),
                                     top20_training_mean=float(initial_true.mean()),reference=-task.n/2,
                                     beyond_reference_fraction=float(np.mean(true < -task.n/2-.01)),seconds=0.,restarts=20)
                            rows.append(row)
                            name=f'local_{method}_{initialization}_{radius}_J{J:+.1f}'
                            np.savez_compressed(dest/f'{name}.npz',parameters=final,exact_final=true,prediction=pred,starts=initial)
                print('LOCAL',case,seed,J,[r['mean_energy'] for r in rows[-6:]],flush=True)
                (folder/'local_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    summarize(rows,folder/'local_summary.json')


if __name__=='__main__':main()
