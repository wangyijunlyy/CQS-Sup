"""NC and exact-energy controls for the identical measured warm-start policy."""
from pathlib import Path
import sys
import json
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_evidence import ROOT,OUT,TFIMTask,GridExact
from tune_nc import NormL2Path,ALPHAS,stable_solve
from local_refinement import optimize
from search_improvements import summarize
from src.wmps.models.nc_baselines import TrigFeatureMap,frequency_vectors


def main():
    torch.set_num_threads(2)
    folder=OUT/'search_improvements'
    rows=[]
    for case,rc in [('grid2x3_l2',(2,3)),('grid3x3_l2',(3,3))]:
        task=TFIMTask(*rc,2,True,1)
        oracles={j:GridExact(task,j) for j in (-.1,-.3,-.5)}
        for seed in range(1001,1006):
            src=ROOT/f'results_supplementary/e3/confirmation/{case}_seed{seed}'
            dest=folder/f'{case}_seed{seed}'
            with np.load(src/'dataset.npz') as a:data={k:a[k] for k in a.files}
            x=data['X'];y=data['measured_terms']
            fmap=TrigFeatureMap(frequency_vectors(len(task.gate_to_control),2,1000,seed),task.gate_to_control).double()
            with torch.no_grad():phi=fmap(torch.tensor(x[:200].reshape(200,-1),dtype=torch.float64)).numpy()
            path=NormL2Path(phi[:160],y[:160]@task.coefficients(-.1,-.5))
            for J in (-.1,-.3,-.5):
                c=task.coefficients(J,-.5)
                yt=y[:160]@c;yv=y[160:200]@c
                path.y=yt;path.b=phi[:160].T@yt;path.zero_threshold=2*np.linalg.norm(path.b)
                path.projected=path.vectors.T@(path.b if path.space=='primal' else yt)
                path.norm_weights=path.projected**2
                if path.space=='dual':path.norm_weights*=path.eigenvalues
                candidates=[];weights=[]
                for alpha in ALPHAS:
                    w,info=stable_solve(path,float(alpha))
                    info['validation_energy_mse']=float(np.mean((phi[160:]@w-yv)**2))
                    weights.append(w);candidates.append(info)
                best=min(range(len(weights)),key=lambda i:candidates[i]['validation_energy_mse'])
                weight=torch.tensor(weights[best],dtype=torch.float64)
                np.save(dest/f'NC_hamiltonian_tuned_weights_J{J:+.1f}.npy',weights[best])
                (dest/f'NC_local_selection_J{J:+.1f}.json').write_text(json.dumps(dict(selected=candidates[best],candidates=candidates),indent=2)+'\n')
                predictors=dict(nc_energy_tuned=lambda z:fmap(z.flatten(1))@weight,
                                exact_oracle=lambda z:oracles[J](z,0.))
                center=x[np.argmin(yt)]
                rng=np.random.default_rng(seed+9000)
                starts=center+rng.normal(0,.05,(20,2,2))
                starts=np.remainder(starts+np.pi,2*np.pi)-np.pi
                traces={}
                for radius in (None,.25):
                    for method,predict in predictors.items():
                        final,pred=optimize(predict,starts,center,radius)
                        key=f'local_{method}_best_measured_training_plus_jitter_{radius}_J{J:+.1f}'
                        traces[key]=(final,pred,radius,method)
                single=task.energy(center,J,-.5)
                initial_true=np.array([task.energy(z,J,-.5) for z in starts])
                for key,(final,pred,radius,method) in traces.items():
                    true=np.array([task.energy(z,J,-.5) for z in final])
                    np.savez_compressed(dest/f'{key}.npz',parameters=final,exact_final=true,prediction=pred,starts=starts)
                    rows.append(dict(case=case,seed=seed,J=J,method=method,initialization='best_measured_training_plus_jitter',
                                     boundary='direct_wrap' if radius is None else 'direct_wrap_radius.25',
                                     mean_energy=float(true.mean()),mean_prediction=float(pred.mean()),single_training_candidate=single,
                                     top20_training_mean=float(initial_true.mean()),reference=-task.n/2,
                                     beyond_reference_fraction=float(np.mean(true < -task.n/2-.01)),seconds=0.,restarts=20))
                (folder/'local_strong_results.json').write_text(json.dumps(rows,indent=2)+'\n')
                print('LOCAL STRONG',case,seed,J,'alpha',float(ALPHAS[best]),flush=True)
    summarize(rows,folder/'local_strong_summary.json')


if __name__=='__main__':main()
