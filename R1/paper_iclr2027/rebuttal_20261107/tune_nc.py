"""Validation-tuned NC objective on cached records; no new quantum labels.

Uses the original unsquared L2 objective and feature coordinates, solved to
convergence. It is an additional control, not the original Adam/EMA algorithm.
"""
from pathlib import Path
import os
import sys
import json
import argparse
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

WORK=Path(__file__).resolve().parents[2]
ROOT=WORK/'world_model_surrogate'
OUT=Path(__file__).resolve().parent/'evidence'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from run_table1_tuned_control import NormL2Path, nc_features
from run_redq_continuation import continuation
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.nc_baselines import TrigFeatureMap
from src.wmps.models.fourier_continuation import smoothed_nc_features

ALPHAS=np.array([.001,.003,.01,.03,.1,.3,1.,3.,10.,30.,100.,300.,1000.,3000.,10000.,30000.,100000.])


def stable_solve(path,alpha):
    try:
        return path.solve(alpha)
    except FloatingPointError:
        # Repeated shared controls make the Gram matrix rank deficient. Work
        # directly with A's SVD to avoid cancellation in A.T @ dual_solution.
        # Keep the same objective and the original certificate tolerances.
        if not hasattr(path,'direct_svd'):
            path.direct_svd=np.linalg.svd(path.a,full_matrices=False)
        u,s,vt=path.direct_svd
        q=s*(u.T@path.y)
        if alpha>=2*np.linalg.norm(path.a.T@path.y):
            theta=np.zeros(path.a.shape[1])
            t=None
        else:
            def equation(log_t):
                t=np.exp(log_t)
                return 2*np.sqrt(np.sum(q*q*(t/(s*s+t))**2))-alpha
            t=float(np.exp(brentq(equation,-60,60,xtol=1e-12,rtol=1e-14)))
            theta=vt.T@(q/(s*s+t))
        residual=path.a@theta-path.y
        norm=np.linalg.norm(theta)
        grad=2*path.a.T@residual
        kkt=np.linalg.norm(grad+alpha*theta/norm) if norm else max(0.,np.linalg.norm(grad)-alpha)
        dualvar=2*residual*min(1.,alpha/max(np.linalg.norm(grad),np.finfo(float).tiny))
        dual=-path.y@dualvar-dualvar@dualvar/4
        obj=float(residual@residual+alpha*norm)
        gap=float(max(0.,obj-dual)/max(1.,abs(obj)))
        info=dict(alpha=float(alpha),effective_ridge=t,coefficient_norm=float(norm),objective=obj,
                  dual_lower_bound=float(dual),relative_duality_gap=gap,
                  relative_kkt_residual=float(kkt/max(1.,alpha)),train_mse=float(np.mean(residual**2)),
                  solver='direct thin SVD after Gram-space precision failure')
        if gap>1e-7 or info['relative_kkt_residual']>1e-5:
            raise FloatingPointError(('SVD certificate failed',info))
        print('SVD SOLVER CERTIFICATE',info,flush=True)
        return theta,info


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def vqe():
    config=json.loads((ROOT/'configs/nc_tfim_vqe_independent.json').read_text())
    results=json.loads((OUT/'nc_vqe_tuned.json').read_text()) if (OUT/'nc_vqe_tuned.json').exists() else []
    for case in config['cases']:
        task=TFIMTask(case['rows'],case['cols'],case['layers'],case['shared'],1 if case['shared'] else -1)
        for seed in config['seeds']:
            name=f"{case['id']}_seed{seed}"
            existing=[r for r in results if r['case']==case['id'] and r['seed']==seed]
            if len(existing)==len(case['couplings'])*4:
                print('REUSE completed',name,flush=True)
                continue
            folder=OUT/'nc_tuned'/name
            folder.mkdir(parents=True,exist_ok=True)
            audit=ROOT/'results_nc_tfim_vqe_independent/code'/name
            with np.load(ROOT/'results_nc_tfim_vqe_independent/datasets'/name/'dataset.npz') as a:
                x,measured,exact=a['X'],a['measured_terms'],a['exact_terms']
            w=np.load(audit/'frequencies.npy')
            fmap=TrigFeatureMap(w,task.gate_to_control).double()
            with torch.no_grad():
                phi=fmap(torch.tensor(x.reshape(len(x),-1),dtype=torch.float64)).numpy()
            nt,nv=case['train'],case['val']
            a,b,c=phi[:nt],phi[nt:nt+nv],phi[nt+nv:]
            starts=np.load(audit/'initial_parameters.npy')
            path=NormL2Path(a,measured[:nt]@task.coefficients(case['couplings'][0],-.5))
            for j in case['couplings']:
                coef=task.coefficients(j,-.5)
                yt=measured[:nt]@coef
                yv=measured[nt:nt+nv]@coef
                # Reuse the feature eigendecomposition for each scalar target.
                path.y=yt
                path.b=a.T@yt
                path.zero_threshold=2*np.linalg.norm(path.b)
                path.projected=path.vectors.T@(path.b if path.space=='primal' else yt)
                path.norm_weights=path.projected**2
                if path.space=='dual':
                    path.norm_weights*=path.eigenvalues
                candidates,weights=[],[]
                for alpha in ALPHAS:
                    theta,info=stable_solve(path,float(alpha))
                    info['val_mse']=float(np.mean((b@theta-yv)**2))
                    candidates.append(info)
                    weights.append(theta)
                best=int(np.argmin([r['val_mse'] for r in candidates]))
                write(folder/f'selection_J{j:+.1f}.json',dict(selected_alpha=float(ALPHAS[best]),
                      objective='sum squared residuals + alpha * unsquared L2 coefficient norm; no free intercept',
                      no_validation_refit=True,candidates=candidates,selection_labels='stored measured validation energy'))
                for label,index in [('fixed_converged',int(np.flatnonzero(ALPHAS==(0.1 if task.shared else 10.))[0])),('validation_tuned',best)]:
                    theta=weights[index]
                    np.save(folder/f'{label}_weights_J{j:+.1f}.npy',theta)
                    target=exact[nt+nv:]@coef
                    mae=float(np.abs(c@theta-target).mean())
                    wt=torch.tensor(theta,dtype=torch.float64)
                    predict=lambda z,s:smoothed_nc_features(fmap,z.flatten(1),s)@wt
                    for search,sigma in [('direct',0.),('continuation',1.)]:
                        trace=continuation(predict,starts,config['vqe'],sigma,.6)
                        final=np.array([task.energy(z,j,-.5) for z in trace['parameters'][-1]])
                        np.savez_compressed(folder/f'{label}_{search}_J{j:+.1f}.npz',**trace,exact_final=final)
                        row=dict(case=case['id'],seed=seed,J=j,fit=label,search=search,alpha=float(ALPHAS[index]),
                                 test_mae=mae,mean_energy=float(final.mean()),reference=-task.n/2,
                                 reference_attainment=float(np.mean(final<=-task.n/2+.01)))
                        results.append(row)
                        print('TUNED VQE',row,flush=True)
            write(OUT/'nc_vqe_tuned.json',results)


def counts_features(angles,counts):
    result=np.ones((len(angles),len(counts)),dtype=np.float64)
    for d in range(angles.shape[1]):
        result*=np.cos(angles[:,d,None])**counts[None,:,d,0]
        result*=np.sin(angles[:,d,None])**counts[None,:,d,1]
    return result


def multi_norm_path(a,y,b,yv):
    """Shared spectral solve, independent per-output original NC objectives."""
    gram=a.T@a
    vals,vec=eigh(gram,driver='evd')
    vals=np.maximum(vals,0.)
    at_y=a.T@y
    proj=vec.T@at_y
    paths=[]
    val_mse=[]
    max_kkt=0.
    for alpha in ALPHAS:
        sol=np.zeros_like(at_y)
        for o in range(y.shape[1]):
            zero=2*np.linalg.norm(at_y[:,o])
            if alpha<zero:
                weights=proj[:,o]**2
                def equation(log_t):
                    t=np.exp(log_t)
                    return 2*np.sqrt(np.sum(weights*(t/(vals+t))**2))-alpha
                t=np.exp(brentq(equation,-60,60,xtol=1e-12,rtol=1e-14))
                sol[:,o]=vec@(proj[:,o]/(vals+t))
                residual=2*(gram@sol[:,o]-at_y[:,o])+alpha*sol[:,o]/np.linalg.norm(sol[:,o])
                kkt=float(np.linalg.norm(residual)/max(1.,alpha))
                if kkt>1e-5:
                    raise ValueError(('KKT failure',kkt))
                max_kkt=max(max_kkt,kkt)
        paths.append(sol)
        val_mse.append(np.mean((b@sol-yv)**2,axis=0))
    return np.stack(paths),np.stack(val_mse),max_kkt


def fspt():
    # Targeted hardest-relevant existing N8 settings, all original paired splits.
    # Both a single global alpha and per-output alpha selection are reported.
    results=[]
    with np.load(ROOT/'results_fspt_scaling/datasets/dataset_n8.npz') as a:
        x,y=a['X'],a['y']
    angles=np.concatenate([np.pi-2*x[:,:1],-2*x[:,1:]],axis=1)
    for k in (12,24):
        for seed in (42,43,44):
            folder=OUT/'nc_tuned'/f'fspt_n8_k{k}_seed{seed}'
            folder.mkdir(parents=True,exist_ok=True)
            old=ROOT/f'results_remaining_comparisons/fspt_scaling_n8_k{k}_seed{seed}_nc_hqs'
            neural=ROOT/f'results_fspt_scaling/n8_k{k}_seed{seed}'
            with np.load(neural/'split.npz') as a:
                tr,va,te=a['train'],a['val'],a['test']
            all_weights,curves,counts_list=[],[],[]
            tic=time.perf_counter()
            max_kkt=0.
            for period in range(k):
                counts=np.load(old/f'counts_k{period+1}.npy')
                phi=counts_features(angles,counts)
                weights,scores,check=multi_norm_path(phi[tr],y[tr,period],phi[va],y[va,period])
                max_kkt=max(max_kkt,check)
                all_weights.append(weights)
                curves.append(scores)
                counts_list.append(counts)
            curves=np.stack(curves,axis=1)
            global_idx=int(np.argmin(curves.mean((1,2))))
            per_output_idx=curves.argmin(0)
            write(folder/'selection.json',dict(alphas=ALPHAS.tolist(),global_alpha=float(ALPHAS[global_idx]),
                  per_output_alpha=ALPHAS[per_output_idx].tolist(),validation_mse=curves.tolist(),
                  max_relative_kkt=max_kkt,no_validation_refit=True,test_used_for_selection=False))
            for selection in ('global_alpha','per_output_alpha'):
                pred=np.empty((len(te),k,8))
                for period,(weights,counts) in enumerate(zip(all_weights,counts_list)):
                    idx=np.full(8,global_idx) if selection=='global_alpha' else per_output_idx[period]
                    coeff=np.column_stack([weights[idx[o],:,o] for o in range(8)])
                    pred[:,period]=counts_features(angles[te],counts)@coeff
                    np.save(folder/f'{selection}_weights_k{period+1}.npy',coeff)
                np.save(folder/f'pred_{selection}.npy',pred)
                rows=dict(N=8,K=k,seed=seed,method='NC_'+selection,
                          mae=float(np.abs(pred-y[te,:k]).mean()),
                          first6_mae=float(np.abs(pred[:,:6]-y[te,:6]).mean()),seconds=time.perf_counter()-tic)
                results.append(rows)
                print('TUNED FSPT',rows,flush=True)
            for label,pred in [('NC_original_Adam_alpha1',np.load(old/'pred.npy')),('CQS_R',np.load(neural/'pred_redq.npy'))]:
                results.append(dict(N=8,K=k,seed=seed,method=label,mae=float(np.abs(pred-y[te,:k]).mean()),
                                    first6_mae=float(np.abs(pred[:,:6]-y[te,:6]).mean()),seconds=None))
            write(OUT/'nc_fspt_tuned.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--part',choices=['vqe','fspt'],required=True)
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        (vqe if args.part=='vqe' else fspt)()
