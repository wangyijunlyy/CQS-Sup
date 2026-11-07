"""Fresh fixed-data confirmation; measured records only, no terminal truth."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
from run_tuned import HERE,ROOT
from src.wmps.data.nc_tfim import TFIMTask,shadow_terms
from src.wmps.models.periodic_redq import ProductMomentREDQ,causal_parameter_masks
from run_fspt8_fair import fit as original_fit
from complete_fourier import SharedFeatureFourier
from search import search,reference_starts,ARMS,COUPLINGS
from fitting import write,fit_product,fit_fourier_candidates,choose,energy_validation,RANKS,PRODUCT_REGULARIZERS
from timed import timed_search
from artifacts import save_npz,valid_search,valid_timed


def data(seed,folder):
    if (folder/'complete.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    task=TFIMTask(1,6,2,False,-1)
    x=np.random.default_rng(seed).uniform(-np.pi,np.pi,(2800,2,11))
    rng=np.random.default_rng(seed+2000)
    bits=np.empty((2400,10,6),np.int8);recipes=np.empty_like(bits);measured=np.empty((2400,11))
    for i in range(2400):
        bits[i],recipes[i]=task.sample_shadow(x[i],10,rng)
        measured[i]=shadow_terms(bits[i],recipes[i],task.edges)
    save_npz(folder/'shadow_records.npz',bits=bits,recipes=recipes)
    save_npz(folder/'dataset.npz',X=x,measured_terms=measured)
    write(folder/'complete.json',dict(seed=seed,measurement_seed=seed+2000,measured_controls=2400,
        shots_per_control=10,simulated_acquisition_shots=24000,hardware_shots=0,exact_labels_generated=False))
    print('NEW DATA',seed,flush=True)


def fit(seed,base):
    if (base/'all_fits_saved.json').exists():return
    task=TFIMTask(1,6,2,False,-1)
    data(seed,base/'data')
    with np.load(base/'data/dataset.npz') as a:x,y=a['X'],a['measured_terms']
    coeff=np.stack([task.coefficients(j,-.5) for j in COUPLINGS])
    old=base/'original_product';old.mkdir(exist_ok=True)
    if not (old/'fit.json').exists():
        _,info=original_fit(lambda:ProductMomentREDQ(6,task.edges,2,False,32,1),
            x[:2000],y[:2000],x[2000:2400],y[2000:2400],seed,'cpu',old,'model',300,patience=30)
        write(old/'fit.json',dict(**info,criterion='original measured moment validation'))
    candidates=[]
    for rank in RANKS:
        for regularizer,strength in PRODUCT_REGULARIZERS:
            name=f'r{rank}_{regularizer}{strength:g}'
            record=fit_product(x[:2000],y[:2000],x[2000:2400],y[2000:2400],coeff,
                seed,rank,regularizer,strength,base/'products'/name)
            candidates.append(dict(**record,candidate=name))
    original=ProductMomentREDQ(6,task.edges,2,False,32,1).eval()
    original.load_state_dict(torch.load(old/'model.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():pred=original(torch.tensor(x[2000:2400],dtype=torch.float32))[0].numpy()
    candidates.append(dict(candidate='original',rank=32,validation_energy_mse=energy_validation(pred,y[2000:2400],coeff)))
    write(base/'product_selection.json',dict(selected=choose(candidates),candidates=candidates,exact_targets_used=False))
    masks=causal_parameter_masks(6,task.edges,2,False).reshape(11,22)
    tic=time.perf_counter()
    fc,groups=fit_fourier_candidates(x[:2000].reshape(2000,22),y[:2000],x[2000:2400].reshape(400,22),y[2000:2400],masks,coeff)
    best=min(range(len(fc)),key=lambda i:fc[i]['validation_energy_mse'])
    original_index=min((i for i,row in enumerate(fc) if row['family']=='scaled_ridge'),key=lambda i:fc[i]['validation_moment_mse'])
    for method,index in [('fourier',best),('fourier_original',original_index)]:
        folder=base/method;folder.mkdir(exist_ok=True)
        for g,group in enumerate(groups):
            values={key:value for key,value in group.items() if key!='weights'}
            save_npz(folder/f'kernel_group{g}.npz',**values,weights=group['weights'][index][None])
    save_npz(base/'fourier_validation_predictions.npz',predictions=np.stack([r['prediction'] for r in fc]))
    for row in fc:del row['prediction']
    write(base/'fourier_selection.json',dict(selected_index=best,selected=fc[best],original_index=original_index,
        original=fc[original_index],candidates=fc,seconds=time.perf_counter()-tic,exact_targets_used=False))
    write(base/'all_fits_saved.json',dict(new_product_fits=31,fourier_candidates=16,exact_targets_used=False))
    print('ALL NEW FITS',seed,choose(candidates)['candidate'],flush=True)


def models(seed,base):
    task=TFIMTask(1,6,2,False,-1)
    selected=json.loads((base/'product_selection.json').read_text())['selected']
    product=ProductMomentREDQ(6,task.edges,2,False,selected['rank'],1).double().eval()
    path=base/'original_product/model.pt' if selected['candidate']=='original' else base/'products'/selected['candidate']/'model.pt'
    product.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    for p in product.parameters():p.requires_grad_(False)
    identities={'product':f"seed{seed}:product:{selected['candidate']}"}
    fs=json.loads((base/'fourier_selection.json').read_text())
    for method,key in [('fourier','selected'),('fourier_original','original')]:
        row=fs[key];identities[method]=f"seed{seed}:{method}:{row['family']}:{row['lambda_value']:g}"
    return task,product,SharedFeatureFourier(base/'fourier'),SharedFeatureFourier(base/'fourier_original'),identities


def run(seed):
    torch.set_num_threads(2)
    protocol=json.loads((HERE/'confirmation_protocol.json').read_text())
    assert seed in protocol['seeds']
    base=HERE/'confirmation'/f'seed{seed}';base.mkdir(parents=True,exist_ok=True)
    fit(seed,base)
    task,product,fourier,old_fourier,identities=models(seed,base)
    first=np.random.default_rng(seed+10000).uniform(-np.pi,np.pi,(20,2,11))
    save_npz(base/'starts.npz',uniform=first,reference=reference_starts(seed))
    names=('product','fourier','fourier_original')
    for arm in protocol['fixed_arms']:
        boundary,initial,steps=ARMS[arm]
        starts=first if initial=='uniform' else reference_starts(seed)
        for ji,j in enumerate(COUPLINGS):
            coeff=torch.tensor(task.coefficients(j,-.5))
            for method in names[(seed+ji)%3:]+names[:(seed+ji)%3]:
                path=base/'search'/arm/f'{method}_J{j:+.1f}.npz'
                if valid_search(path,steps,identities[method]):continue
                fn=(lambda z,s:product(z,smoothing=s)[0]@coeff) if method=='product' else (lambda z,s:(old_fourier if method=='fourier_original' else fourier)(z,s)@coeff)
                tic=time.perf_counter();trace=search(fn,starts,steps=steps,boundary=boundary)
                save_npz(path,**trace,seconds=time.perf_counter()-tic,model_identity=identities[method])
    if protocol['primary_arm']=='time_matched':
        cap=protocol['time_cap_seconds']
        for ji,j in enumerate(COUPLINGS):
            coeff=torch.tensor(task.coefficients(j,-.5))
            for method in names[(seed+ji)%3:]+names[:(seed+ji)%3]:
                path=base/'timed'/f'{method}_J{j:+.1f}.npz'
                if valid_timed(path,cap,identities[method]):continue
                fn=(lambda z,s:product(z,smoothing=s)[0]@coeff) if method=='product' else (lambda z,s:(old_fourier if method=='fourier_original' else fourier)(z,s)@coeff)
                search(fn,first,steps=2,boundary='wrap')
                result=timed_search(fn,first,seed=980000+seed*100+ji,cap=cap)
                save_npz(path,**result,model_identity=identities[method])
    expected=0
    for arm in protocol['fixed_arms']:
        steps=ARMS[arm][2]
        for method in names:
            for j in COUPLINGS:
                assert valid_search(base/'search'/arm/f'{method}_J{j:+.1f}.npz',steps,identities[method]);expected+=1
    if protocol['primary_arm']=='time_matched':
        for method in names:
            for j in COUPLINGS:
                assert valid_timed(base/'timed'/f'{method}_J{j:+.1f}.npz',protocol['time_cap_seconds'],identities[method]);expected+=1
    write(base/'all_saved.json',dict(seed=seed,searches=expected,original_acquisition_shots=24000,
        added_search_shots=0,hardware_shots=0,exact_scoring_started=False))
    print('CONFIRMATION SAVED',seed,expected,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);run(ap.parse_args().seed)
