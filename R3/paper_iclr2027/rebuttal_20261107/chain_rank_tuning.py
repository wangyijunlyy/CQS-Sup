"""Measured-validation rank selection on the original finite-shot chain data.

The rank-6 capacity construction motivates checking overparameterization.
Ranks 6/8/16 use the original full training cap, labels and fitting rules;
the stored rank-32 fit remains a candidate. No exact value selects rank.
"""
from pathlib import Path
import os
import sys
import json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[k]='2'
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_evidence import ROOT,OUT,TFIMTask
from run_fspt8_fair import fit
from run_redq_continuation import continuation
from src.wmps.models.periodic_redq import ProductMomentREDQ


def main():
    torch.set_num_threads(2)
    task=TFIMTask(1,6,2,False,-1)
    folder=OUT/'chain_rank_tuning'
    folder.mkdir(exist_ok=True)
    (folder/'protocol.json').write_text(json.dumps(dict(ranks=[6,8,16,32],seeds=[45,46,47],
        new_fit_ranks=[6,8,16],rank32='original stored fit',epochs=300,patience=30,
        selection='minimum measured validation moment MSE',initialization_seed='original seed',
        original_labels=True,extra_measurements=0,exact_energy_selection=False),indent=2)+'\n')
    records=[]
    for seed in (45,46,47):
        with np.load(ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz') as a:
            x,measured=a['X'],a['measured_terms']
        candidates=[]
        for rank in (6,8,16):
            dest=folder/f'seed{seed}_rank{rank}'
            dest.mkdir(exist_ok=True)
            model,detail=fit(lambda:ProductMomentREDQ(6,task.edges,2,False,rank,1),
                x[:2000],measured[:2000],x[2000:2400],measured[2000:2400],seed,'cpu',dest,'product',300,min_epochs=0,patience=30)
            model=model.eval().double()
            candidates.append((detail['best_val_mse'],rank,model,detail))
            (dest/'fit.json').write_text(json.dumps(detail,indent=2)+'\n')
        original=ROOT/f'results_product_redq_independent/chain6_l2_seed{seed}'
        old=ProductMomentREDQ(6,task.edges,2,False,32,1).double().eval()
        old.load_state_dict(torch.load(original/'periodic_redq.pt',map_location='cpu',weights_only=True))
        with torch.no_grad():oldval=float((old(torch.tensor(x[2000:2400],dtype=torch.float64))[0]-torch.tensor(measured[2000:2400])).square().mean())
        candidates.append((oldval,32,old,dict(source='original stored model')))
        val,rank,model,detail=min(candidates,key=lambda t:t[0])
        for p in model.parameters():p.requires_grad_(False)
        (folder/f'seed{seed}_selection.json').write_text(json.dumps(dict(rank=rank,measured_validation_mse=val,
            candidates=[dict(rank=r,validation_mse=v) for v,r,_,_ in candidates]),indent=2)+'\n')
        with np.load(ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz') as a:exact=a['exact_terms'][2400:]
        with torch.no_grad():pred=model(torch.tensor(x[2400:],dtype=torch.float64))[0].numpy()
        mse=float(np.mean((pred-exact)**2))
        starts=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
        settings=dict(steps=100,lr=.1,clip=float(np.pi))
        for J in (-.5,-.3,-.1,.1,.3,.5):
            coeff=torch.tensor(task.coefficients(J,-.5),dtype=torch.float64)
            predict=lambda z,s:model(z,smoothing=s)[0]@coeff
            trace=continuation(predict,starts,settings,1.,.6)
            truth=np.array([task.energy(z,J,-.5) for z in trace['parameters'][-1]])
            np.savez_compressed(folder/f'seed{seed}_selected_J{J:+.1f}.npz',**trace,exact_final=truth)
            records.append(dict(seed=seed,J=J,selected_rank=rank,validation_mse=val,test_moment_mse=mse,mean_energy=float(truth.mean())))
        print('RANK SELECTED',seed,rank,val,mse,[r['mean_energy'] for r in records[-6:]],flush=True)
        (folder/'results.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
