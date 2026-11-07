"""Full development fitting/search. No exact targets or endpoint scores are read."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[3];ROOT=WORK/'world_model_surrogate'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(HERE.parents[1]),str(HERE.parent/'extensions_20261108')]
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ,causal_parameter_masks
from complete_fourier import SharedFeatureFourier
from fitting import write,fit_product,fit_fourier_candidates,choose,energy_validation,RANKS,PRODUCT_REGULARIZERS,RAW_GRID,SCALED_GRID
from search import search,reference_starts,ARMS,COUPLINGS
from artifacts import save_npz,valid_search


def fit(seed):
    torch.set_num_threads(2)
    task=TFIMTask(1,6,2,False,-1)
    base=HERE/'tuned'/f'seed{seed}';base.mkdir(parents=True,exist_ok=True)
    if (base/'all_fits_saved.json').exists():return
    file=ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz'
    with np.load(file) as a:x,y=a['X'],a['measured_terms']
    coeff=np.stack([task.coefficients(j,-.5) for j in COUPLINGS])
    write(base/'protocol.json',dict(seed=seed,ranks=RANKS,regularizers=PRODUCT_REGULARIZERS,
        raw_grid=RAW_GRID,scaled_grid=SCALED_GRID,epochs=300,patience=30,train=2000,validation=400,
        criterion='six-coupling measured validation energy MSE; one shared fit per representation',
        original_acquisition_shots=24000,added_shots=0,exact_targets_used=False,
        torch=torch.__version__,numpy=np.__version__,device='cpu'))
    candidates=[]
    for rank in RANKS:
        for regularizer,strength in PRODUCT_REGULARIZERS:
            name=f'r{rank}_{regularizer}{strength:g}'
            record=fit_product(x[:2000],y[:2000],x[2000:2400],y[2000:2400],coeff,
                seed,rank,regularizer,strength,base/'products'/name)
            candidates.append(dict(**record,candidate=name))
    original=ProductMomentREDQ(6,task.edges,2,False,32,1).eval()
    original.load_state_dict(torch.load(ROOT/f'results_product_redq_independent/chain6_l2_seed{seed}/periodic_redq.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():pred=original(torch.tensor(x[2000:2400],dtype=torch.float32))[0].numpy()
    candidates.append(dict(candidate='historical',rank=32,
        validation_energy_mse=energy_validation(pred,y[2000:2400],coeff),
        checkpoint='periodic_redq.pt',training='original moment MSE and moment-validation checkpoint'))
    selected=choose(candidates)
    write(base/'product_selection.json',dict(selected=selected,candidates=candidates,exact_targets_used=False))
    print('PRODUCT SELECTED',seed,selected['candidate'],flush=True)
    if not (base/'fourier_selection.json').exists():
        masks=causal_parameter_masks(6,task.edges,2,False).reshape(11,22)
        tic=time.perf_counter()
        fc,groups=fit_fourier_candidates(x[:2000].reshape(2000,22),y[:2000],
            x[2000:2400].reshape(400,22),y[2000:2400],masks,coeff)
        best=min(range(len(fc)),key=lambda i:fc[i]['validation_energy_mse'])
        folder=base/'fourier';folder.mkdir(exist_ok=True)
        for g,group in enumerate(groups):
            weights=group.pop('weights')[best]
            np.savez_compressed(folder/f'kernel_group{g}.npz',**group,weights=weights[None])
        np.savez_compressed(base/'fourier_validation_predictions.npz',predictions=np.stack([r['prediction'] for r in fc]))
        for r in fc:del r['prediction']
        write(base/'fourier_selection.json',dict(selected_index=best,selected=fc[best],candidates=fc,
            seconds=time.perf_counter()-tic,exact_targets_used=False))
        print('FOURIER SELECTED',seed,fc[best],flush=True)
    write(base/'all_fits_saved.json',dict(product_full_fits=30,product_historical_candidate=True,
        fourier_converged_candidates=16,selection='measured validation energy only',exact_targets_used=False))


def models(seed):
    task=TFIMTask(1,6,2,False,-1);base=HERE/'tuned'/f'seed{seed}'
    meta=json.loads((base/'product_selection.json').read_text())['selected']
    product=ProductMomentREDQ(6,task.edges,2,False,meta['rank'],1).double().eval()
    path=(ROOT/f'results_product_redq_independent/chain6_l2_seed{seed}/periodic_redq.pt') if meta['candidate']=='historical' else base/'products'/meta['candidate']/'model.pt'
    product.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    for p in product.parameters():p.requires_grad_(False)
    return task,product,SharedFeatureFourier(base/'fourier')


def identity(seed,method):
    base=HERE/'tuned'/f'seed{seed}'
    selected=json.loads((base/f'{method}_selection.json').read_text())['selected']
    return f"seed{seed}:{method}:"+(selected['candidate'] if method=='product' else f"{selected['family']}:{selected['lambda_value']:g}")


def searches(seed):
    torch.set_num_threads(2)
    base=HERE/'tuned'/f'seed{seed}'
    assert (base/'all_fits_saved.json').exists()
    task,product,fourier=models(seed)
    uniform=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
    starts={'uniform':uniform,'reference':reference_starts(seed)}
    for arm,(boundary,initial,steps) in ARMS.items():
        for j in COUPLINGS:
            coeff=torch.tensor(task.coefficients(j,-.5),dtype=torch.float64)
            for method in ('product','fourier') if seed%2 else ('fourier','product'):
                path=base/'search'/arm/f'{method}_J{j:+.1f}.npz'
                name=identity(seed,method)
                if valid_search(path,steps,name):continue
                path.parent.mkdir(parents=True,exist_ok=True)
                fn=(lambda z,s:product(z,smoothing=s)[0]@coeff) if method=='product' else (lambda z,s:fourier(z,s)@coeff)
                tic=time.perf_counter();trace=search(fn,starts[initial],steps=steps,boundary=boundary)
                save_npz(path,**trace,seconds=time.perf_counter()-tic,model_identity=name)
                print('SEARCH',seed,arm,j,method,flush=True)
    assert all(valid_search(base/'search'/arm/f'{method}_J{j:+.1f}.npz',steps,identity(seed,method))
        for arm,(_,_,steps) in ARMS.items() for method in ('product','fourier') for j in COUPLINGS)
    write(base/'all_searches_saved.json',dict(searches=48,exact_scoring_started=False,added_shots=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(45,46,47),required=True)
    ap.add_argument('--phase',choices=('fit','search','all'),default='all');args=ap.parse_args()
    if args.phase in ('fit','all'):fit(args.seed)
    if args.phase in ('search','all'):searches(args.seed)
