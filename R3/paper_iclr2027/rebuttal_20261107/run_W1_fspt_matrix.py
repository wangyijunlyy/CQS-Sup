"""Complete the original FSPT matrix with validation-tuned NC objectives."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '2'
import csv
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from tune_nc import ROOT, OUT, ALPHAS, counts_features, multi_norm_path

DEST = OUT/'W1_execution/fspt_matrix'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    write(DEST/'protocol.json', dict(N=[4,6,8,10], K=[6,12,24], seeds=[42,43,44],
        alphas=ALPHAS.tolist(), selection='measured/exact validation MSE, no validation refit',
        objective='sum squared residuals + alpha * unsquared coefficient L2 norm',
        fixed_features='original stored counts for each setting/seed/period',
        neural='unchanged original cap300 predictions', independent_pools=False))
    original_rows = json.loads((OUT/'nc_fspt_tuned.json').read_text())
    rows = []
    for n in (4,6,8,10):
        with np.load(ROOT/f'results_fspt_scaling/datasets/dataset_n{n}.npz') as data:
            x, y = data['X'], data['y']
        angles = np.concatenate([np.pi-2*x[:,:1], -2*x[:,1:]],axis=1)
        for seed in (42,43,44):
            cache = {}
            for k in (6,12,24):
                name=f'n{n}_k{k}_seed{seed}'
                folder=DEST/name
                folder.mkdir(exist_ok=True)
                if (folder/'results.json').exists():
                    rows.extend(json.loads((folder/'results.json').read_text()))
                    print('REUSE',name,flush=True)
                    continue
                if n==8 and k in (12,24):
                    selected=[r for r in original_rows if r['N']==n and r['K']==k and r['seed']==seed]
                    assert len(selected)==4
                    selected=[dict(r,source='existing audited focused NC comparison') for r in selected]
                    write(folder/'results.json',selected)
                    rows.extend(selected)
                    print('REUSE FOCUSED',name,flush=True)
                    continue
                neural=ROOT/f'results_fspt_scaling/n{n}_k{k}_seed{seed}'
                old=ROOT/f'results_remaining_comparisons/fspt_scaling_n{n}_k{k}_seed{seed}_nc_hqs'
                with np.load(neural/'split.npz') as split:
                    tr,va,te=split['train'],split['val'],split['test']
                tic=time.perf_counter()
                paths, curves, counts_list=[],[],[]
                max_kkt=0.
                for period in range(k):
                    counts=np.load(old/f'counts_k{period+1}.npy')
                    if period in cache and np.array_equal(counts,cache[period][0]):
                        stored,weights,scores,check,old_tr,old_va=cache[period]
                        assert np.array_equal(tr,old_tr) and np.array_equal(va,old_va)
                    else:
                        phi=counts_features(angles,counts)
                        weights,scores,check=multi_norm_path(phi[tr],y[tr,period],phi[va],y[va,period])
                        cache[period]=(counts,weights,scores,check,tr.copy(),va.copy())
                    max_kkt=max(max_kkt,check)
                    paths.append(weights);curves.append(scores);counts_list.append(counts)
                curves=np.stack(curves,axis=1)
                global_idx=int(np.argmin(curves.mean((1,2))))
                per_output_idx=curves.argmin(0)
                write(folder/'selection.json',dict(alphas=ALPHAS.tolist(),
                    global_alpha=float(ALPHAS[global_idx]),
                    per_output_alpha=ALPHAS[per_output_idx].tolist(),
                    validation_mse=curves.tolist(),max_relative_kkt=max_kkt,
                    no_test_selection=True,upper_boundary_outputs=int((per_output_idx==len(ALPHAS)-1).sum()),
                    lower_boundary_outputs=int((per_output_idx==0).sum())))
                current=[]
                for selection in ('global_alpha','per_output_alpha'):
                    pred=np.empty((len(te),k,n))
                    for period,(weights,counts) in enumerate(zip(paths,counts_list)):
                        idx=np.full(n,global_idx) if selection=='global_alpha' else per_output_idx[period]
                        coef=np.column_stack([weights[idx[o],:,o] for o in range(n)])
                        pred[:,period]=counts_features(angles[te],counts)@coef
                        np.save(folder/f'{selection}_weights_k{period+1}.npy',coef)
                    np.save(folder/f'pred_{selection}.npy',pred)
                    current.append(dict(N=n,K=k,seed=seed,method='NC_'+selection,
                        mae=float(np.abs(pred-y[te,:k]).mean()),
                        first6_mae=float(np.abs(pred[:,:6]-y[te,:6]).mean()),
                        seconds=time.perf_counter()-tic,source='completed full matrix'))
                for label,path in [('CQS_R',neural/'pred_redq.npy'),('NC_original_Adam_alpha1',old/'pred.npy')]:
                    pred=np.load(path)
                    current.append(dict(N=n,K=k,seed=seed,method=label,
                        mae=float(np.abs(pred-y[te,:k]).mean()),
                        first6_mae=float(np.abs(pred[:,:6]-y[te,:6]).mean()),seconds=None,
                        source='unchanged original saved predictions'))
                write(folder/'results.json',current)
                rows.extend(current)
                print('DONE',name,[(r['method'],round(r['mae'],6)) for r in current],flush=True)
                write(DEST/'all_rows.json',rows)
    write(DEST/'all_rows.json',rows)
    with (DEST/'all_rows.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    summary=[]
    for n in (4,6,8,10):
        for k in (6,12,24):
            for method in ('CQS_R','NC_global_alpha','NC_per_output_alpha','NC_original_Adam_alpha1'):
                a=np.array([r['mae'] for r in rows if (r['N'],r['K'],r['method'])==(n,k,method)])
                assert len(a)==3
                summary.append(dict(N=n,K=k,method=method,mean=float(a.mean()),sample_sd=float(a.std(ddof=1))))
    write(DEST/'summary.json',summary)
    print('COMPLETE 36 paired settings, 144 result rows',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
