"""Post-search prediction errors at the same retained controls."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import csv
import json
import numpy as np
import torch
from analyze import HERE,SEEDS,COUPLINGS,ARMS,TFIMTask,ProductMomentREDQ,SharedFeatureFourier,predict_batched,stats,write
from calibration import ACTIVE,features


def main():
    torch.set_num_threads(2)
    assert json.loads((HERE/'verification.json').read_text())['passed']
    task=TFIMTask(1,6,2,False,-1)
    rows=[]
    for seed in SEEDS:
        base=HERE/'confirmation'/f'seed{seed}'
        p=ProductMomentREDQ(6,task.edges,2,False,32,1).double().eval()
        p.load_state_dict(torch.load(base/'product/product.pt',map_location='cpu',weights_only=True))
        for parameter in p.parameters():parameter.requires_grad_(False)
        f=SharedFeatureFourier(base/'fourier')
        for j in COUPLINGS:
            coefficient=torch.tensor(task.coefficients(j,-.5))
            funcs=dict(product=lambda x:p(x)[0]@coefficient,fourier=lambda x:f(x)@coefficient)
            for arm in ARMS:
                for model in ('product','fourier'):
                    out=base/f'{arm}_{model}_J{j:+.1f}'
                    x=np.load(out/'final_parameters.npy')
                    true=np.load(out/'exact_final.npy')
                    with np.load(out/'round2.npz') as a:
                        z=(x.reshape(20,22)[:,ACTIVE]-a['center'].reshape(-1)[ACTIVE])/.1
                        q=features(z)@a['correction_weights']
                        zc=(a['candidate_parameters'].reshape(20,22)[:,ACTIVE]-a['center'].reshape(-1)[ACTIVE])/.1
                        distances=np.sqrt(((zc[:,None]-zc[None,:])**2).mean(-1))
                        diversity=float(np.median(distances[np.triu_indices(20,1)]))
                        accepted=int(a['accepted'].sum())
                    alpha=json.loads((out/'complete.json').read_text())['rounds'][2]['choice']['alpha']
                    prior=predict_batched(funcs[model],x)
                    corrected=alpha*prior+q
                    rows.append(dict(seed=seed,J=j,arm=arm,model=model,alpha=alpha,
                        true_energy=float(true.mean()),prior_prediction=float(prior.mean()),
                        corrected_prediction=float(corrected.mean()),
                        prior_mae=float(abs(prior-true).mean()),
                        corrected_mae=float(abs(corrected-true).mean()),
                        candidate_standardized_pair_rms_median=diversity,
                        accepted_final_candidates=accepted))
    with (HERE/'endpoint_diagnostics.csv').open('w') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for arm in ARMS:
        for model in ('product','fourier'):
            values={key:[np.mean([r[key] for r in rows if r['seed']==seed and r['arm']==arm and r['model']==model]) for seed in SEEDS] for key in ('prior_mae','corrected_mae','candidate_standardized_pair_rms_median','accepted_final_candidates')}
            summary.append(dict(arm=arm,model=model,**{key:stats(v) for key,v in values.items()}))
    write(HERE/'endpoint_diagnostics.json',dict(by_arm=summary,
        scope='post-search errors at identical retained controls; five dataset means over all six couplings',
        selection_used_exact_energy=False))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
