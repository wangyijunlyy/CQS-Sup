"""Fresh-data confirmation of the fixed measured-warm-start refinement policy.

This run fixes its protocol before generating five new pools. All methods share
measurement histograms, splits and 20 starts. Product rank/training is unchanged
from the paper's full E3 model. No test result selects hyperparameters or seeds.
"""
from pathlib import Path
import os
import sys
import json
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import numpy as np
import torch
from scipy.stats import t as student_t
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_evidence import ROOT,OUT,TFIMTask,GridExact
from run_supplement_e3 import generate_data
from run_fspt8_fair import fit
from run_redq_continuation import continuation
from local_refinement import optimize
from tune_nc import NormL2Path,ALPHAS,stable_solve
from run_vqe_fourier_controls import FourierBasis,ridge_path,RIDGE_GRID
from src.wmps.models.periodic_redq import ProductMomentREDQ
from src.wmps.models.nc_baselines import TrigFeatureMap,frequency_vectors

FOLDER=OUT/'warmstart_confirmation'
PROTOCOL=dict(version='measured-warmstart-confirmation-v1',case='grid3x3_l2',
    seeds=list(range(5101,5106)),primary_J=-.5,all_couplings=[-.1,-.3,-.5],
    train=160,val=40,test=200,shots=10000,
    acquisition='joint all-Z/all-X, 10000 multinomial draws per group per train/validation control',
    product=dict(rank=32,q=4,epochs=3000,min_epochs=500,patience=100,lr=.001,weight_decay=.0001,batch=32),
    training_selection='minimum measured validation moment MSE',
    fourier_selection='Hamiltonian-specific measured validation energy MSE over Q1..4 and original ridge grid',
    NC_selection='same sampled expanded-gate support2 family; converged unsquared-L2 objective; measured validation energy MSE',
    warm_start='lowest measured training-energy control plus 20 fixed Gaussian perturbations std0.05; torus wrapping',
    search='100 Adam updates, lr0.1, no smoothing, torus wrapping; all 20 endpoints retained',
    original_control='same product, original uniform20 starts, sigma1->0 by step60, clip[-pi,pi]',
    acquisition_shots_per_dataset=4000000,total_new_simulated_shots=20000000,hardware_shots=0,
    additional_quantum_measurements_for_warm_start_or_other_couplings=0,
    primary_contrast='Fourier-tuned minus product true endpoint energy at J=-0.5',
    secondary_contrasts=['tuned NC minus product','single measured training candidate minus product','explicit feasible reference minus product'],
    statistics='five independent dataset means; nominal paired t95 intervals, df4; no restart-pooled confidence intervals',
    exact_use='after checkpoints and searches for each dataset are saved; oracle is diagnostic only')


def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def describe(v):
    a=np.asarray(v,dtype=float);half=student_t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a))
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),ci95=[float(a.mean()-half),float(a.mean()+half)],values=a.tolist(),n=len(a))


def main():
    torch.set_num_threads(2)
    FOLDER.mkdir(exist_ok=True)
    path=FOLDER/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==PROTOCOL
    else:save(path,PROTOCOL)
    task=TFIMTask(3,3,2,True,1)
    oracles={j:GridExact(task,j) for j in PROTOCOL['all_couplings']}
    rows=[]
    for seed in PROTOCOL['seeds']:
        dest=FOLDER/f'seed{seed}';dest.mkdir(exist_ok=True)
        if (dest/'results.json').exists():
            rows.extend(json.loads((dest/'results.json').read_text()));continue
        spec=dict(train=160,val=40,test=200,shots=10000,seeds=dict(data=seed,initialization=seed+1000,measurement=seed+2000,starts=seed+3000),vqe=dict(restarts=20))
        if not(dest/'dataset.npz').exists():np.savez_compressed(dest/'dataset.npz',**generate_data(task,spec))
        with np.load(dest/'dataset.npz') as a:data={k:a[k] for k in a.files}
        x,y=data['X'],data['measured_terms']
        if (dest/'product.pt').exists() and (dest/'fit.json').exists():
            model=ProductMomentREDQ(9,task.edges,2,True,32,4)
            model.load_state_dict(torch.load(dest/'product.pt',map_location='cpu',weights_only=True))
        else:
            model,detail=fit(lambda:ProductMomentREDQ(9,task.edges,2,True,32,4),
                x[:160],y[:160],x[160:200],y[160:200],seed+1000,'cpu',dest,'product',3000,min_epochs=500,patience=100)
            save(dest/'fit.json',detail)
        model=model.double().eval()
        for p in model.parameters():p.requires_grad_(False)
        fpaths=[]
        for q in (1,2,3,4):
            basis=FourierBasis(2,q)
            with torch.no_grad():a=basis(torch.tensor(x[:160],dtype=torch.float64)).numpy();b=basis(torch.tensor(x[160:200],dtype=torch.float64)).numpy()
            fpaths.append((basis,b,ridge_path(a,y[:160],RIDGE_GRID)))
        fmap=TrigFeatureMap(frequency_vectors(len(task.gate_to_control),2,1000,seed),task.gate_to_control).double()
        with torch.no_grad():phi=fmap(torch.tensor(x[:200].reshape(200,-1),dtype=torch.float64)).numpy()
        path=NormL2Path(phi[:160],y[:160]@task.coefficients(-.1,-.5))
        pending=[]
        for J in PROTOCOL['all_couplings']:
            c=task.coefficients(J,-.5);ct=torch.tensor(c,dtype=torch.float64)
            yt=y[:160]@c;yv=y[160:200]@c
            fcandidates=[];best=None
            for basis,b,weights in fpaths:
                for ridge,w in zip(RIDGE_GRID,weights):
                    ew=w@c;val=float(np.mean((b@ew-yv)**2))
                    info=dict(q=basis.q,ridge=ridge,validation_energy_mse=val);fcandidates.append(info)
                    if best is None or val<best[0]:best=(val,basis,ew,info)
            _,basis,fw,fsel=best;fw=torch.tensor(fw,dtype=torch.float64)
            path.y=yt;path.b=phi[:160].T@yt;path.zero_threshold=2*np.linalg.norm(path.b)
            path.projected=path.vectors.T@(path.b if path.space=='primal' else yt)
            path.norm_weights=path.projected**2
            if path.space=='dual':path.norm_weights*=path.eigenvalues
            candidates=[];weights=[]
            for alpha in ALPHAS:
                w,info=stable_solve(path,float(alpha));info['validation_energy_mse']=float(np.mean((phi[160:]@w-yv)**2))
                candidates.append(info);weights.append(w)
            selected=min(range(len(weights)),key=lambda i:candidates[i]['validation_energy_mse'])
            nw=torch.tensor(weights[selected],dtype=torch.float64)
            save(dest/f'selection_J{J:+.1f}.json',dict(Fourier_selected=fsel,Fourier_candidates=fcandidates,NC_selected=candidates[selected],NC_candidates=candidates))
            predictors=dict(product=lambda z:model(z)[0]@ct,fourier_energy_tuned=lambda z:basis(z)@fw,
                            nc_energy_tuned=lambda z:fmap(z.flatten(1))@nw,exact_oracle=lambda z:oracles[J](z,0.))
            index=int(np.argmin(yt));center=x[index]
            rng=np.random.default_rng(seed+9000)
            initial=np.remainder(center+rng.normal(0,.05,(20,2,2))+np.pi,2*np.pi)-np.pi
            for method,predict in predictors.items():
                final,pred=optimize(predict,initial,center,None)
                np.savez_compressed(dest/f'{method}_J{J:+.1f}.npz',parameters=final,prediction=pred,starts=initial)
                pending.append((J,method,final,pred,index))
            old=continuation(lambda z,s:model(z,smoothing=s)[0]@ct,data['starts'],dict(steps=100,lr=.1,clip=float(np.pi)),1.,.6)
            np.savez_compressed(dest/f'product_original_J{J:+.1f}.npz',**old)
            pending.append((J,'product_original',old['parameters'][-1],old['predictions'][-1],index))
        save(dest/'searches_saved.json',dict(all_searches_saved=True,exact_endpoint_evaluation_has_started=False))
        local=[]
        for J,method,final,pred,index in pending:
            true=np.array([task.energy(z,J,-.5) for z in final])
            single=task.energy(x[index],J,-.5)
            np.save(dest/f'{method}_J{J:+.1f}_exact.npy',true)
            local.append(dict(seed=seed,J=J,method=method,mean_energy=float(true.mean()),predicted_mean=float(pred.mean()),
                              single_training_candidate=single,reference=-4.5,below_reference_fraction=float(np.mean(true < -4.51)),
                              initialization='original uniform' if method=='product_original' else 'measured minimum + same jitter20',restarts=20))
        save(dest/'results.json',local);rows.extend(local);save(FOLDER/'results.json',rows)
        print('FRESH CONFIRM',seed,[(r['method'],r['mean_energy']) for r in local if r['J']==-.5],flush=True)
    summary={}
    for J in PROTOCOL['all_couplings']:
        chosen=[r for r in rows if r['J']==J];summary[J]={}
        for method in ('product','fourier_energy_tuned','nc_energy_tuned','exact_oracle','product_original'):
            summary[J][method]=describe([r['mean_energy'] for r in chosen if r['method']==method])
        ps={r['seed']:r for r in chosen if r['method']=='product'}
        for method in ('fourier_energy_tuned','nc_energy_tuned'):
            summary[J][method+'_minus_product']=describe([r['mean_energy']-ps[r['seed']]['mean_energy'] for r in chosen if r['method']==method])
        summary[J]['single_minus_product']=describe([r['single_training_candidate']-r['mean_energy'] for r in ps.values()])
        summary[J]['reference_minus_product']=describe([-4.5-r['mean_energy'] for r in ps.values()])
    save(FOLDER/'summary.json',summary)
    print('FRESH COMPLETE',json.dumps(summary[-.5]),flush=True)


if __name__=='__main__':main()
