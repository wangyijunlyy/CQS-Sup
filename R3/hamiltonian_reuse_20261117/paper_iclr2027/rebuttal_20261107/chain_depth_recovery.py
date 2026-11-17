"""Matched, offline chain-depth search and local-calibration experiments.

Original fits/data are read only. Exact values score saved candidates only.
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
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_evidence import ROOT, TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ
from src.wmps.models.nc_baselines import TrigFeatureMap
from src.wmps.models.fourier_continuation import smoothed_nc_features

OUT=HERE/'evidence/chain_depth_recovery'
SEEDS=(45,46,47)
COUPLINGS=(-.5,-.3,-.1,.1,.3,.5)


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def embed_shallow(x):
    x=np.asarray(x)
    assert x.shape[1:]==(1,11)
    return np.concatenate((x,np.zeros_like(x)),axis=1)


def optimize(predict,starts,staged=False,lr=.03,steps=100):
    x=torch.tensor(starts,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([x],lr=lr,foreach=False)
    trace=[starts.copy()]
    for step in range(steps):
        sigma=max(0.,1-step/(steps*.6))
        opt.zero_grad()
        value=predict(x,sigma)
        value.sum().backward()
        if staged and step<steps//2:x.grad[:,0]=0
        opt.step()
        with torch.no_grad():x.copy_(torch.remainder(x+np.pi,2*np.pi)-np.pi)
        trace.append(x.detach().numpy().copy())
    with torch.no_grad():pred=predict(x,0.).numpy()
    return dict(parameters=np.array(trace),prediction=pred)


def load_models(seed,task):
    result={}
    for label,rank in [('product',32),('product_valrank',None)]:
        if rank is None:
            rank=json.loads((HERE/f'evidence/chain_rank_tuning/seed{seed}_selection.json').read_text())['rank']
            path=HERE/f'evidence/chain_rank_tuning/seed{seed}_rank{rank}/product.pt'
        else:path=ROOT/f'results_product_redq_independent/chain6_l2_seed{seed}/periodic_redq.pt'
        m=ProductMomentREDQ(6,task.edges,2,False,rank,1).double().eval()
        m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        for p in m.parameters():p.requires_grad_(False)
        result[label]=m
    audit=ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}'
    result['nc_features']=TrigFeatureMap(np.load(audit/'frequencies.npy'),task.gate_to_control).double()
    return result


def predictors(models,task,seed,J):
    c=torch.tensor(task.coefficients(J,-.5))
    nw=torch.tensor(np.load(HERE/f'evidence/nc_tuned/chain6_l2_seed{seed}/validation_tuned_weights_J{J:+.1f}.npy'))
    result={name:(lambda z,s,m=m:m(z,smoothing=s)[0]@c) for name,m in models.items() if name.startswith('product')}
    fmap=models['nc_features']
    result['tuned_nc']=lambda z,s:smoothed_nc_features(fmap,z.flatten(1),s)@nw
    return result


def check_physics():
    a,b=TFIMTask(1,6,1,False,-1),TFIMTask(1,6,2,False,-1)
    starts=np.random.default_rng(20261108).uniform(-np.pi,np.pi,(4,1,11))
    embedded=embed_shallow(starts)
    error=max(abs(a.energy(x,-.1,-.5)-b.energy(y,-.1,-.5)) for x,y in zip(starts,embedded))
    assert error<1e-12,error
    reference=np.zeros((2,11));reference[0,6:9]=np.pi
    assert abs(b.energy(reference,-.1,-.5)+3)<1e-12
    # A real smooth objective must not move the frozen layer, but must move the new one.
    out=optimize(lambda x,s:((x-.4)**2).flatten(1).sum(1),embedded,True,steps=10)
    frozen=max(float(np.max(abs(row[:,0]-embedded[:,0]))) for row in out['parameters'][:6])
    assert frozen<1e-12
    assert np.max(abs(out['parameters'][5,:,1]))>0
    return dict(identity_embedding_error=error,reference_energy=b.energy(reference,-.1,-.5),frozen_layer_max_change=frozen)


def phase1():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(seeds=SEEDS,couplings=COUPLINGS,qubits=6,layers=2,
        models=['product rank32','product validation-selected rank','validation-tuned NC'],
        arms=['uniform_joint','shallow_joint','shallow_staged'],restarts=20,updates=100,
        staged='50 updates new layer only, then 50 all layers',lr=.03,boundary='periodic wrap',
        smoothing='sigma 1 to 0 by update60, same all arms',
        shallow_source='all 20 original saved CQS-P+C one-layer endpoints; shared by every method',
        extra_measurements=0,additional_shallow_fit=False,
        scope='uses pre-existing shallow dataset and checkpoint in addition to two-layer fit; total workflow cost not identical to original single-depth experiment',
        exact_values='all search candidates saved before their post-search scoring; no exact selection')
    save(OUT/'phase1_protocol.json',protocol)
    save(OUT/'engineering_checks.json',check_physics())
    task=TFIMTask(1,6,2,False,-1)
    rows=[]
    for seed in SEEDS:
        folder=OUT/f'seed{seed}';folder.mkdir(exist_ok=True)
        models=load_models(seed,task)
        with np.load(ROOT/f'results_redq_continuation_independent/chain6_l1_seed{seed}/redq_J-0.1.npz') as a:shallow=a['parameters'][-1]
        starts=embed_shallow(shallow)
        uniform=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
        np.save(folder/'shared_shallow_starts.npy',starts)
        pending=[]
        for J in COUPLINGS:
            for method,predict in predictors(models,task,seed,J).items():
                for arm in protocol['arms']:
                    path=folder/f'{method}_{arm}_J{J:+.1f}.npz'
                    if not path.exists():
                        tic=time.perf_counter()
                        trace=optimize(predict,uniform if arm=='uniform_joint' else starts,arm=='shallow_staged')
                        np.savez_compressed(path,**trace,seconds=time.perf_counter()-tic)
                    pending.append((J,method,arm,path))
            print('SAVED',seed,J,flush=True)
        save(folder/'phase1_searches_saved.json',dict(complete=True,candidates=len(pending),before_exact_scoring=True))
        for J,method,arm,path in pending:
            with np.load(path) as a:x=a['parameters'][-1];pred=a['prediction']
            true=np.array([task.energy(z,J,-.5) for z in x])
            start_true=np.array([task.energy(z,J,-.5) for z in (uniform if arm=='uniform_joint' else starts)])
            np.save(path.with_suffix('.exact.npy'),true)
            rows.append(dict(seed=seed,J=J,method=method,arm=arm,mean_energy=float(true.mean()),
                mean_prediction=float(pred.mean()),prediction_gap=float(np.mean(true-pred)),
                fraction_below_minus3=float(np.mean(true < -3)),initial_mean=float(start_true.mean()),
                fraction_improved=float(np.mean(true<start_true)),extra_shots=0))
        save(OUT/'phase1_results.json',rows)
        print('SCORED',seed,[(r['method'],r['arm'],round(r['mean_energy'],4)) for r in rows if r['seed']==seed and r['J']==-.1],flush=True)
    summary=[]
    for J in COUPLINGS:
        for method in ('product','product_valrank','tuned_nc'):
            for arm in protocol['arms']:
                r=[x for x in rows if x['J']==J and x['method']==method and x['arm']==arm]
                v=[x['mean_energy'] for x in r]
                summary.append(dict(J=J,method=method,arm=arm,mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),
                    n=len(v),below_minus3_fraction=float(np.mean([x['fraction_below_minus3'] for x in r])),
                    mean_prediction_gap=float(np.mean([x['prediction_gap'] for x in r]))))
    save(OUT/'phase1_summary.json',summary)
    print('PHASE1 COMPLETE',json.dumps([r for r in summary if r['J']==-.1]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    if args.check:print(json.dumps(check_physics(),indent=2))
    else:phase1()
