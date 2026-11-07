"""Hamiltonian-specific validation-tuned controls for the full coupling sweep.

Product weights remain frozen. Fourier frequency/regularization and original
NC scalar coefficients are selected from the same measured validation records.
All baseline search arms match the product's starts and boundary rules.
"""
from pathlib import Path
import sys
import json
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_evidence import ROOT,OUT,TFIMTask,GridExact
from search_improvements import search,summarize
from tune_nc import NormL2Path,ALPHAS,stable_solve
from run_vqe_fourier_controls import FourierBasis,ridge_path,RIDGE_GRID
from src.wmps.models.nc_baselines import TrigFeatureMap,frequency_vectors
from src.wmps.models.fourier_continuation import smoothed_nc_features


def main():
    torch.set_num_threads(2)
    folder=OUT/'search_improvements'
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case,rc in [('grid2x3_l2',(2,3)),('grid3x3_l2',(3,3))]:
        task=TFIMTask(*rc,2,True,1)
        oracles={J:GridExact(task,J) for J in (-.1,-.3,-.5)}
        for seed in range(1001,1006):
            src=ROOT/f'results_supplementary/e3/confirmation/{case}_seed{seed}'
            dest=folder/f'{case}_seed{seed}'
            dest.mkdir(exist_ok=True)
            with np.load(src/'dataset.npz') as a:data={k:a[k] for k in a.files}
            x=data['X'];measured=data['measured_terms']
            fourier_paths=[]
            for q in (1,2,3,4):
                basis=FourierBasis(2,q)
                with torch.no_grad():
                    a=basis(torch.tensor(x[:160],dtype=torch.float64)).numpy()
                    b=basis(torch.tensor(x[160:200],dtype=torch.float64)).numpy()
                weights=ridge_path(a,measured[:160],RIDGE_GRID)
                fourier_paths.append((basis,b,weights))
            fmap=TrigFeatureMap(frequency_vectors(len(task.gate_to_control),2,1000,seed),task.gate_to_control).double()
            with torch.no_grad():phi=fmap(torch.tensor(x[:200].reshape(200,-1),dtype=torch.float64)).numpy()
            path=NormL2Path(phi[:160],measured[:160]@task.coefficients(-.1,-.5))
            for J in (-.1,-.3,-.5):
                c=task.coefficients(J,-.5)
                yt=measured[:160]@c;yv=measured[160:200]@c
                candidates=[];best=None
                for basis,b,weights in fourier_paths:
                    for ridge,w in zip(RIDGE_GRID,weights):
                        energyweights=w@c
                        val=float(np.mean((b@energyweights-yv)**2))
                        candidate=dict(q=basis.q,ridge=ridge,validation_energy_mse=val)
                        candidates.append(candidate)
                        if best is None or val<best[0]:best=(val,basis,energyweights,candidate)
                _,fbasis,fw,fselect=best
                path.y=yt;path.b=phi[:160].T@yt;path.zero_threshold=2*np.linalg.norm(path.b)
                path.projected=path.vectors.T@(path.b if path.space=='primal' else yt)
                path.norm_weights=path.projected**2
                if path.space=='dual':path.norm_weights*=path.eigenvalues
                nc_candidates=[];nw=[]
                for alpha in ALPHAS:
                    weight,info=stable_solve(path,float(alpha))
                    info['validation_energy_mse']=float(np.mean((phi[160:]@weight-yv)**2))
                    nc_candidates.append(info);nw.append(weight)
                index=min(range(len(nw)),key=lambda i:nc_candidates[i]['validation_energy_mse'])
                (dest/f'hamiltonian_tuned_selection_J{J:+.1f}.json').write_text(json.dumps(dict(
                    fourier_selected=fselect,fourier_candidates=candidates,NC_selected=nc_candidates[index],
                    NC_candidates=nc_candidates,selection='measured validation energy MSE; no validation refit; no exact selection'),indent=2)+'\n')
                fw=torch.tensor(fw,dtype=torch.float64);ncw=torch.tensor(nw[index],dtype=torch.float64)
                predictors=dict(fourier_energy_tuned=lambda z,s:fbasis(z,sigma=s)@fw,
                                nc_energy_tuned=lambda z,s:smoothed_nc_features(fmap,z.flatten(1),s)@ncw,
                                exact_oracle=lambda z,s:oracles[J](z,s))
                ranked=np.argsort(yt,kind='stable')
                initializations=dict(original_uniform20=data['starts'],lowest_measured_train20=x[ranked[:20]])
                traces={}
                for initialization,starts in initializations.items():
                    for boundary in ('clip','wrap'):
                        for method,predict in predictors.items():
                            trace=search(predict,starts,boundary)
                            key=f'{method}_{initialization}_{boundary}_J{J:+.1f}'
                            np.savez_compressed(dest/f'{key}.npz',**trace,starts=starts)
                            traces[key]=trace
                single=task.energy(x[ranked[0]],J,-.5)
                top20=np.array([task.energy(z,J,-.5) for z in initializations['lowest_measured_train20']])
                for initialization in initializations:
                    for boundary in ('clip','wrap'):
                        for method in predictors:
                            key=f'{method}_{initialization}_{boundary}_J{J:+.1f}'
                            trace=traces[key]
                            true=np.array([task.energy(z,J,-.5) for z in trace['parameters']])
                            np.save(dest/f'{key}_exact.npy',true)
                            rows.append(dict(case=case,seed=seed,J=J,method=method,initialization=initialization,boundary=boundary,
                                             mean_energy=float(true.mean()),mean_prediction=float(trace['predictions'][-1].mean()),
                                             single_training_candidate=single,top20_training_mean=float(top20.mean()),reference=-task.n/2,
                                             beyond_reference_fraction=float(np.mean(true < -task.n/2-.01)),seconds=trace['seconds'],restarts=20))
                print('STRONG BASELINES',case,seed,J,'Fourier Q/ridge',fselect,'NC alpha',float(ALPHAS[index]),flush=True)
                (folder/'strong_baselines_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    summarize(rows,folder/'strong_baselines_summary.json')


if __name__=='__main__':main()
