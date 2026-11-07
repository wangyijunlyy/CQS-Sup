"""Timing pilot and pure fixed-data, time-matched continuation workers."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from run_tuned import models,identity,ROOT,HERE
from search import search,COUPLINGS
from fitting import write
from timed import timed_search
from artifacts import save_npz,valid_timed
from complete_fourier import SharedFeatureFourier

METHODS=('product','fourier','fourier_original')


def timed_identity(seed,method):
    return f'seed{seed}:fourier_original:historical_moment_validation_Q1' if method=='fourier_original' else identity(seed,method)


def original_fourier(seed):
    return SharedFeatureFourier(HERE.parent/'extensions_20261108/chain'/f'seed{seed}'/'global')


def pilot():
    torch.set_num_threads(2)
    target=HERE/'time_protocol.json'
    if target.exists():return
    records=[]
    for seed in (45,46,47):
        base=HERE/'tuned'/f'seed{seed}'
        assert (base/'all_searches_saved.json').exists()
        task,product,fourier=models(seed)
        old_fourier=original_fourier(seed)
        first=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
        coeff=torch.tensor(task.coefficients(-.5,-.5))
        for method in METHODS:
            fn=(lambda z,s:product(z,smoothing=s)[0]@coeff) if method=='product' else (lambda z,s:(old_fourier if method=='fourier_original' else fourier)(z,s)@coeff)
            search(fn,first,steps=2,boundary='wrap')
            for repeat in range(2):
                tic=time.perf_counter();search(fn,first,steps=100,boundary='wrap')
                records.append(dict(seed=seed,method=method,repeat=repeat,seconds=time.perf_counter()-tic))
    cap=max(10,int(np.ceil(1.5*max(r['seconds'] for r in records))))
    write(target,dict(cap_seconds=cap,pilot=records,
        rule='ceil(1.5*slowest full100-update/20-start block), minimum10seconds',
        CPU_threads=2,dtype='float64',steps_per_block=100,continuation=dict(sigma=1.,fraction=.6),
        boundary='periodic wrap',selection='lowest predicted energy with periodic active-coordinate RMS diversity0.2; fill remaining by predicted score',
        eligible='complete block, pooled selection and copied outputs within cap; late blocks discarded',
        original_acquisition_shots=24000,added_shots=0,exact_targets_used=False))
    print('CAP FROZEN',cap,flush=True)


def worker(seed):
    torch.set_num_threads(2)
    cap=json.loads((HERE/'time_protocol.json').read_text())['cap_seconds']
    task,product,fourier=models(seed)
    old_fourier=original_fourier(seed)
    first=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
    dest=HERE/'timed'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    for ji,j in enumerate(COUPLINGS):
        coeff=torch.tensor(task.coefficients(j,-.5))
        offset=(seed+ji)%3
        for method in METHODS[offset:]+METHODS[:offset]:
            path=dest/f'{method}_J{j:+.1f}.npz'
            name=timed_identity(seed,method)
            if valid_timed(path,cap,name):continue
            fn=(lambda z,s:product(z,smoothing=s)[0]@coeff) if method=='product' else (lambda z,s:(old_fourier if method=='fourier_original' else fourier)(z,s)@coeff)
            # Warmup consumes no search RNG and is outside the search window.
            search(fn,first,steps=2,boundary='wrap')
            result=timed_search(fn,first,seed=980000+seed*100+ji,cap=cap)
            save_npz(path,**result,model_identity=name)
            print('TIMED',seed,j,method,result['completed_blocks'],result['actual_seconds'],flush=True)
    assert all(valid_timed(dest/f'{method}_J{j:+.1f}.npz',cap,timed_identity(seed,method))
        for method in METHODS for j in COUPLINGS)
    write(dest/'all_saved.json',dict(searches=18,exact_scoring_started=False,added_shots=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--seed',type=int,choices=(45,46,47))
    args=ap.parse_args()
    if args.pilot:pilot()
    else:worker(args.seed)
