"""All four paired searches on original complete checkpoints, without truth."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[3]
ROOT=WORK/'world_model_surrogate'
EXT=HERE.parent/'extensions_20261108'
sys.path[:0]=[str(ROOT),str(EXT)]
from src.wmps.models.periodic_redq import ProductMomentREDQ
from src.wmps.data.nc_tfim import TFIMTask
from complete_fourier import SharedFeatureFourier,CompleteFourier
from search import search,reference_starts,ARMS,COUPLINGS
from artifacts import save_npz,valid_search


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def frozen_models(seed):
    task=TFIMTask(1,6,2,False,-1)
    product=ProductMomentREDQ(6,task.edges,2,False,32,1).double().eval()
    product.load_state_dict(torch.load(ROOT/f'results_product_redq_independent/chain6_l2_seed{seed}/periodic_redq.pt',map_location='cpu',weights_only=True))
    for p in product.parameters():p.requires_grad_(False)
    folder=EXT/'chain'/f'seed{seed}'/'global'
    fourier=SharedFeatureFourier(folder)
    original=CompleteFourier(folder)
    x=torch.tensor(reference_starts(seed)[:3],requires_grad=True)
    a,b=fourier(x,.6),original(x,.6)
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    ga=torch.autograd.grad(a.sum(),x)[0];gb=torch.autograd.grad(b.sum(),x)[0]
    torch.testing.assert_close(ga,gb,atol=1e-9,rtol=1e-9)
    for model in (lambda z,s:product(z,smoothing=s)[0],lambda z,s:fourier(z,s)):
        a=model(x,.6);b=model(x+2*np.pi,.6)
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    return task,product,fourier


def run(seed):
    torch.set_num_threads(2)
    task,product,fourier=frozen_models(seed)
    dest=HERE/'frozen'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    uniform=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
    starts={'uniform':uniform,'reference':reference_starts(seed)}
    write(dest/'protocol.json',dict(seed=seed,arms=ARMS,couplings=COUPLINGS,
        lr=.1,sigma=1.,fraction=.6,frozen_checkpoints=True,
        original_acquisition_shots=24000,added_shots=0,truth_used=False))
    for arm,(boundary,initial,steps) in ARMS.items():
        for j in COUPLINGS:
            coeff=torch.tensor(task.coefficients(j,-.5),dtype=torch.float64)
            for method in ('product','fourier') if seed%2 else ('fourier','product'):
                path=dest/arm/f'{method}_J{j:+.1f}.npz'
                if valid_search(path,steps):continue
                path.parent.mkdir(parents=True,exist_ok=True)
                fn=(lambda z,s:product(z,smoothing=s)[0]@coeff) if method=='product' else (lambda z,s:fourier(z,s)@coeff)
                started=time.perf_counter()
                trace=search(fn,starts[initial],steps=steps,boundary=boundary)
                elapsed=time.perf_counter()-started
                save_npz(path,**trace,seconds=elapsed)
                print(seed,arm,j,method,round(elapsed,3),flush=True)
    assert all(valid_search(dest/arm/f'{method}_J{j:+.1f}.npz',steps)
        for arm,(_,_,steps) in ARMS.items() for method in ('product','fourier') for j in COUPLINGS)
    write(dest/'all_saved.json',dict(searches=48,exact_scoring_started=False,added_shots=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(45,46,47),required=True)
    run(ap.parse_args().seed)
