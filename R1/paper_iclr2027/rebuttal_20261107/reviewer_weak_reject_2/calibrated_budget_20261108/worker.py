"""Full new-data fits and measured local searches, with no exact scoring."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
PROJECT = WORK/'world_model_surrogate'
EXT = HERE.parent/'extensions_20261108'
sys.path[:0] = [str(PROJECT/'scripts'),str(PROJECT),str(EXT),str(HERE.parents[1])]
from calibration import select_correction, search, ACTIVE
from development import write
from complete_fourier import SharedFeatureFourier
from resource_chain import ChainMeasurement
from src.wmps.data.nc_tfim import TFIMTask, shadow_terms
from src.wmps.models.periodic_redq import ProductMomentREDQ, causal_parameter_masks
from run_supplement_e1 import fit_fourier
from run_fspt8_fair import fit

SEEDS = (2101,2102,2103,2104,2105)
COUPLINGS = (-.5,-.3,-.1,.1,.3,.5)
ARMS = ('fixed_original','fixed_calibrated','time_calibrated')


def dataset(seed, folder):
    folder.mkdir(parents=True,exist_ok=True)
    file = folder/'dataset.npz'
    if file.exists():
        return
    task = TFIMTask(1,6,2,False,-1)
    x = np.random.default_rng(seed).uniform(-np.pi,np.pi,(2800,2,11))
    rng = np.random.default_rng(seed+2000)
    bits = np.empty((2400,10,6),dtype=np.int8)
    recipes = np.empty_like(bits)
    measured = np.empty((2400,11))
    for i in range(2400):
        bits[i],recipes[i] = task.sample_shadow(x[i],10,rng)
        measured[i] = shadow_terms(bits[i],recipes[i],task.edges)
    np.savez_compressed(folder/'shadow_records.npz',bits=bits,recipes=recipes)
    np.savez_compressed(file,X=x,measured_terms=measured)
    write(folder/'complete.json',dict(seed=seed,measurement_seed=seed+2000,
        controls=2800,measured_controls=2400,shots_per_control=10,original_shots=24000,
        exact_targets_generated=False,reserved_test_controls=400))
    print('DATA SAVED',seed,flush=True)


def models(seed, base):
    task = TFIMTask(1,6,2,False,-1)
    dataset(seed,base/'data')
    with np.load(base/'data/dataset.npz') as a:
        x,y = a['X'],a['measured_terms']
    pdir,fdir = base/'product',base/'fourier'
    pdir.mkdir(exist_ok=True)
    fdir.mkdir(exist_ok=True)
    if not (pdir/'fit.json').exists():
        model,meta = fit(lambda:ProductMomentREDQ(6,task.edges,2,False,32,1),
            x[:2000],y[:2000],x[2000:2400],y[2000:2400],seed,'cpu',pdir,'product',300,
            patience=30,min_epochs=0)
        write(pdir/'fit.json',dict(seed=seed,rank=32,Q=1,train=2000,validation=400,
            selection='measured validation moment MSE',details=meta,exact_targets_used=False))
    if not (fdir/'fit.json').exists():
        started=time.perf_counter()
        fit_fourier(x[:2000].reshape(2000,22),y[:2000],
            x[2000:2400].reshape(400,22),y[2000:2400],
            causal_parameter_masks(6,task.edges,2,False).reshape(11,22),fdir)
        write(fdir/'fit.json',dict(seed=seed,Q=1,train=2000,validation=400,
            seconds=time.perf_counter()-started,selection='original8-ridge measured validation MSE',
            exact_targets_used=False))
    p = ProductMomentREDQ(6,task.edges,2,False,32,1).double().eval()
    p.load_state_dict(torch.load(pdir/'product.pt',map_location='cpu',weights_only=True))
    for parameter in p.parameters():
        parameter.requires_grad_(False)
    f = SharedFeatureFourier(fdir)
    print('FITS SAVED',seed,flush=True)
    return task,p,f


def local(seed,j,model,prior,arm,base,cap):
    out=base/f'{arm}_{model}_J{j:+.1f}'
    out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():
        return
    ji=COUPLINGS.index(j)
    design_rng=np.random.default_rng(930000+seed*100+ji)
    measurement_rng=np.random.default_rng(950000+seed*100+ji)
    search_rng=np.random.default_rng(940000+seed*100+ji)
    engine=ChainMeasurement(1000)
    center=np.zeros((2,11));center[0,6:9]=np.pi
    rounds=[]
    started=time.perf_counter()
    for r,radius in enumerate((.8,.6,.4)):
        z=design_rng.uniform(-1,1,(256,16))
        design=np.repeat(center[None],256,axis=0).reshape(256,22)
        design[:,ACTIVE]+=z*(radius/4)
        design=design.reshape(256,2,11)
        measured=engine.measure(design,j,measurement_rng)
        tick=time.perf_counter()
        with torch.no_grad():
            prediction=np.concatenate([prior(torch.tensor(b)).numpy() for b in np.array_split(design,8)])
        weights,choice=select_correction(z,measured['energy'],prediction,
            alphas=(1.,) if arm=='fixed_original' else (0.,.5,1.))
        calibration_seconds=time.perf_counter()-tick
        candidates,predicted,meta=search(prior,center,radius,weights,choice['alpha'],search_rng,
                                       budget=cap if arm=='time_calibrated' else None)
        block_records=meta.pop('eligible_block_records')
        np.savez_compressed(out/f'search_round{r}.npz',
            parameters=np.asarray([b['parameters'] for b in block_records]),
            predicted=np.asarray([b['predicted'] for b in block_records]),
            eligible_seconds=np.asarray([b['seconds'] for b in block_records]))
        inc=engine.measure(center[None],j,measurement_rng)
        result=engine.measure(candidates,j,measurement_rng)
        accepted=result['energy']+2*np.sqrt(result['variance']+inc['variance'][0]) < inc['energy'][0]
        retained=np.where(accepted[:,None,None],candidates,center[None])
        np.savez_compressed(out/f'round{r}.npz',center=center,design=design,
            design_energy=measured['energy'],design_variance=measured['variance'],
            prior_prediction=prediction,correction_weights=weights,
            candidate_parameters=candidates,candidate_prediction=predicted,
            candidate_energy=result['energy'],candidate_variance=result['variance'],
            incumbent_energy=inc['energy'][0],incumbent_variance=inc['variance'][0],
            accepted=accepted,retained_parameters=retained)
        rounds.append(dict(round=r,radius=radius,choice=choice,
            calibration_seconds=calibration_seconds,search=meta,accepted=int(accepted.sum())))
        if accepted.any():
            index=int(np.argmin(np.where(accepted,result['energy'],np.inf)))
            center=candidates[index].copy()
    np.save(out/'final_parameters.npy',retained)
    engine.save(out/'measurements.npz')
    assert engine.shots==1662000
    write(out/'complete.json',dict(seed=seed,J=j,model=model,arm=arm,rounds=rounds,
        added_shots=engine.shots,original_shots=24000,total_shots=engine.shots+24000,
        seconds=time.perf_counter()-started,exact_scoring_started=False,hardware_shots=0))
    print('SEARCH SAVED',seed,j,model,arm,'alpha',[r['choice']['alpha'] for r in rounds],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=SEEDS,required=True)
    args=ap.parse_args();seed=args.seed
    torch.set_num_threads(2)
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert protocol['confirmation_seeds']==list(SEEDS)
    cap=json.loads((HERE/'timing_cap.json').read_text())['search_seconds_per_round']
    base=HERE/'confirmation'/f'seed{seed}'
    base.mkdir(parents=True,exist_ok=True)
    write(base/'started.json',dict(seed=seed,cpu_affinity=(sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))),
        threads=2,torch=torch.__version__,protocol=protocol,exact_scoring_started=False))
    task,p,f=models(seed,base)
    for ji,j in enumerate(COUPLINGS):
        coefficient=torch.tensor(task.coefficients(j,-.5))
        funcs=dict(product=lambda x:p(x)[0]@coefficient,fourier=lambda x:f(x)@coefficient)
        order=('product','fourier') if (seed+ji)%2==0 else ('fourier','product')
        for arm in ARMS:
            for model in order:
                local(seed,j,model,funcs[model],arm,base,cap)
    write(base/'all_saved.json',dict(seed=seed,searches=36,exact_scoring_started=False,
        original_acquisition_shots=24000,new_search_shots=36*1662000))


if __name__=='__main__':
    main()
