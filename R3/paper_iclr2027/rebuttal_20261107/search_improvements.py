"""Matched search interventions and Hamiltonian reuse on frozen moment fits.

Couplings and search arms are fixed in this file before endpoint evaluation.
Every arm is retained. No exact value selects a checkpoint, start, or search arm.
"""
from pathlib import Path
import sys
import os
import json
import argparse
import time
from collections import defaultdict
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import numpy as np
import torch
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_evidence import ROOT,OUT,TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ
from src.wmps.models.nc_baselines import TrigFeatureMap
from src.wmps.models.fourier_continuation import smoothed_nc_features
from run_vqe_fourier_controls import FourierBasis


def search(predict,starts,boundary):
    x=torch.tensor(starts,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([x],lr=.1,foreach=False)
    predictions=[]
    tic=time.perf_counter()
    for step in range(101):
        sigma=max(0.,1-step/60)
        opt.zero_grad()
        pred=predict(x,sigma).reshape(20)
        predictions.append(pred.detach().numpy())
        assert torch.isfinite(pred).all()
        if step==100:break
        pred.sum().backward()
        opt.step()
        with torch.no_grad():
            if boundary=='clip':x.clamp_(-np.pi,np.pi)
            elif boundary=='wrap':x.copy_(torch.remainder(x+np.pi,2*np.pi)-np.pi)
            else:raise ValueError(boundary)
    return dict(parameters=x.detach().numpy(),predictions=np.stack(predictions),seconds=time.perf_counter()-tic)


def grid():
    rows=[]
    protocol=dict(couplings=[-.1,-.3,-.5],seeds=list(range(1001,1006)),
                  cases=['grid2x3_l2','grid3x3_l2'],initialization=['original_uniform20','lowest_measured_train20'],
                  boundaries=['clip','wrap'],search='Adam100 lr0.1; sigma=max(0,1-step/60)',
                  extra_training=False,extra_measurements=0,
                  control='same starts and boundary rule for product and validation-selected complete Fourier',
                  exact_values='only endpoint evaluation after all searches for a coupling are saved',
                  interpretation='retrospective matched interventions and Hamiltonian-reuse extension; report entire grid, not test-selected best arms')
    folder=OUT/'search_improvements'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for case,rc in [('grid2x3_l2',(2,3)),('grid3x3_l2',(3,3))]:
        task=TFIMTask(*rc,2,True,1)
        for seed in range(1001,1006):
            src=ROOT/f'results_supplementary/e3/confirmation/{case}_seed{seed}'
            target=folder/f'{case}_seed{seed}'
            target.mkdir(exist_ok=True)
            with np.load(src/'dataset.npz') as a:
                data={k:a[k] for k in a.files}
            model=ProductMomentREDQ(task.n,task.edges,2,True,32,4).double().eval()
            model.load_state_dict(torch.load(src/'product.pt',map_location='cpu',weights_only=True))
            for p in model.parameters():p.requires_grad_(False)
            with np.load(src/'fourier.npz') as a:
                q=int(a['q'])
                weights=torch.tensor(a['weights'],dtype=torch.float64)
            basis=FourierBasis(2,q)
            # Same learned fits for every Hamiltonian. Neither fit is refit.
            for J in protocol['couplings']:
                c=torch.tensor(task.coefficients(J,-.5),dtype=torch.float64)
                measured_energy=data['measured_terms'][:160]@c.numpy()
                ranked=np.argsort(measured_energy,kind='stable')
                starts=dict(original_uniform20=data['starts'],lowest_measured_train20=data['X'][ranked[:20]])
                predictors=dict(product=lambda z,s:model(z,smoothing=s)[0]@c,
                                fourier=lambda z,s:basis(z,sigma=s)@weights@c)
                traces={}
                for initialization,initial in starts.items():
                    for boundary in protocol['boundaries']:
                        for method,predict in predictors.items():
                            key=f'{method}_{initialization}_{boundary}_J{J:+.1f}'
                            trace=search(predict,initial,boundary)
                            traces[key]=trace
                            np.savez_compressed(target/f'{key}.npz',**trace,starts=initial)
                # Selection of the single candidate uses measured energy only.
                single=task.energy(data['X'][ranked[0]],J,-.5)
                top20=np.array([task.energy(z,J,-.5) for z in starts['lowest_measured_train20']])
                for initialization in starts:
                    for boundary in protocol['boundaries']:
                        for method in predictors:
                            key=f'{method}_{initialization}_{boundary}_J{J:+.1f}'
                            trace=traces[key]
                            true=np.array([task.energy(z,J,-.5) for z in trace['parameters']])
                            np.save(target/f'{key}_exact.npy',true)
                            rows.append(dict(case=case,seed=seed,J=J,method=method,initialization=initialization,boundary=boundary,
                                             mean_energy=float(true.mean()),mean_prediction=float(trace['predictions'][-1].mean()),
                                             single_training_candidate=single,top20_training_mean=float(top20.mean()),reference=-task.n/2,
                                             beyond_reference_fraction=float(np.mean(true < -task.n/2-.01)),
                                             seconds=trace['seconds'],restarts=20))
                print('GRID SEARCH',case,seed,J,{k:float(np.mean(np.load(target/f'{k}_exact.npy'))) for k in traces},'single',single,flush=True)
                (folder/'grid_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    summarize(rows,folder/'grid_summary.json')


def chain():
    folder=OUT/'search_improvements'
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    task=TFIMTask(1,6,2,False,-1)
    for seed in (45,46,47):
        src=ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}'
        with np.load(ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz') as a:
            data={k:a[k] for k in a.files}
        starts=np.load(src/'initial_parameters.npy')
        model=ProductMomentREDQ(6,task.edges,2,False,32,1).double().eval()
        model.load_state_dict(torch.load(ROOT/f'results_product_redq_independent/chain6_l2_seed{seed}/periodic_redq.pt',map_location='cpu',weights_only=True))
        for p in model.parameters():p.requires_grad_(False)
        fmap=TrigFeatureMap(np.load(src/'frequencies.npy'),task.gate_to_control).double()
        for J in (-.5,-.3,-.1,.1,.3,.5):
            c=torch.tensor(task.coefficients(J,-.5),dtype=torch.float64)
            ncw=torch.tensor(np.load(OUT/f'nc_tuned/chain6_l2_seed{seed}/validation_tuned_weights_J{J:+.1f}.npy'),dtype=torch.float64)
            predictors=dict(product=lambda z,s:model(z,smoothing=s)[0]@c,
                            tuned_nc=lambda z,s:smoothed_nc_features(fmap,z.flatten(1),s)@ncw)
            rank=np.argsort(data['measured_terms'][:2000]@c.numpy(),kind='stable')
            variants=dict(original_uniform20=starts,lowest_measured_train20=data['X'][rank[:20]])
            for initialization,initial in variants.items():
                for boundary in ('clip','wrap'):
                    for method,predict in predictors.items():
                        trace=search(predict,initial,boundary)
                        true=np.array([task.energy(z,J,-.5) for z in trace['parameters']])
                        filename=folder/f'chain_seed{seed}_J{J:+.1f}_{initialization}_{boundary}_{method}.npz'
                        np.savez_compressed(filename,**trace,starts=initial,exact_final=true)
                        row=dict(case='chain6_l2',seed=seed,J=J,method=method,initialization=initialization,boundary=boundary,
                                 mean_energy=float(true.mean()),mean_prediction=float(trace['predictions'][-1].mean()),
                                 single_training_candidate=float(task.energy(data['X'][rank[0]],J,-.5)),
                                 top20_training_mean=float(np.mean([task.energy(z,J,-.5) for z in variants['lowest_measured_train20']])),
                                 reference=-3.,beyond_reference_fraction=float(np.mean(true < -3.01)),seconds=trace['seconds'],restarts=20)
                        rows.append(row)
            print('CHAIN SEARCH',seed,J,flush=True)
            (folder/'chain_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    summarize(rows,folder/'chain_summary.json')


def summarize(rows,path):
    groups=defaultdict(list)
    for r in rows:
        groups[r['case'],r['J'],r['method'],r['initialization'],r['boundary']].append(r)
    summary=[]
    for key,group in groups.items():
        values=np.array([r['mean_energy'] for r in group])
        diff=np.array([r['single_training_candidate']-r['mean_energy'] for r in group])
        half=student_t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff))
        summary.append(dict(case=key[0],J=key[1],method=key[2],initialization=key[3],boundary=key[4],
                            mean_energy=float(values.mean()),seed_sd=float(values.std(ddof=1)),n=len(group),seed_values=values.tolist(),
                            improvement_over_single_candidate=float(diff.mean()),paired_ci95=[float(diff.mean()-half),float(diff.mean()+half)],
                            reference=group[0]['reference'],mean_beyond_reference_fraction=float(np.mean([r['beyond_reference_fraction'] for r in group]))))
    path.write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--part',choices=['grid','chain'],required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    (grid if args.part=='grid' else chain)()
