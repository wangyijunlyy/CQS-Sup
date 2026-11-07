"""Recompute W1 constants, paired mask contrasts, energy errors and TFIM results."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='2'
import sys
import json
import csv
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'world_model_surrogate'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from src.wmps.data.nc_tfim import TFIMTask

OUT=HERE/'evidence/W1_execution/existing_analysis'


def write(name, rows):
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    if isinstance(rows,list) and rows:
        with (OUT/(name+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(rows, keys, metrics):
    groups={}
    for row in rows:groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    result=[]
    for values,group in groups.items():
        record=dict(zip(keys,values));record['runs']=len(group)
        for metric in metrics:
            a=np.array([r[metric] for r in group]);record[metric+'_mean']=float(a.mean())
            record[metric+'_sd']=float(a.std(ddof=1)) if len(a)>1 else 0.
        result.append(record)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    constants=json.loads((HERE/'evidence/constant_baselines.json').read_text())
    comparisons=[]
    for row in constants:
        base=next(r for r in constants if r['family']==row['family'] and r['seed']==row['seed'] and r['method']=='training_measured_mean')
        comparisons.append(dict(row,relative_mse_reduction_vs_train_mean=1-row['moment_mse']/base['moment_mse']))
    write('constant_comparisons',comparisons)
    write('constant_summary',summarize(comparisons,['family','method'],['moment_mse','relative_mse_reduction_vs_train_mean']))
    energy=[];paired=[];max_identity_error=0.
    task=TFIMTask(1,6,2,False,-1)
    for seed in range(1001,1006):
        folder=ROOT/f'results_supplementary/e1/confirmation/seed{seed}'
        by_method={}
        for method in ('P-local','P-full','F-local','F-full'):
            with np.load(folder/method/'test_predictions.npz') as d:pred,target=d['prediction'],d['exact']
            error=pred-target
            second=error.T@error/len(error)
            by_method[method]=float(np.mean(error**2))
            for coupling in (-.1,-.3,-.5):
                c=task.coefficients(coupling,-.5)
                mse=float(np.mean((error@c)**2));identity=float(c@second@c)
                max_identity_error=max(max_identity_error,abs(mse-identity))
                diagonal=float(np.sum(c*c*np.diag(second)))
                energy.append(dict(seed=seed,method=method,J=coupling,moment_mse=float(np.mean(error**2)),
                    x_mse=float(np.mean(error[:,:6]**2)),zz_mse=float(np.mean(error[:,6:]**2)),
                    energy_mae=float(np.abs(error@c).mean()),energy_mse=mse,
                    diagonal_term=diagonal,cross_term=identity-diagonal,
                    weighted_x_diagonal=float(np.sum(c[:6]**2*np.diag(second)[:6])),
                    weighted_zz_diagonal=float(np.sum(c[6:]**2*np.diag(second)[6:]))))
        paired.append(dict(seed=seed,product_mask_gain=by_method['P-full']-by_method['P-local'],
            fourier_mask_gain=by_method['F-full']-by_method['F-local'],
            local_fourier_minus_product=by_method['F-local']-by_method['P-local'],
            full_fourier_minus_product=by_method['F-full']-by_method['P-full']))
    assert max_identity_error<1e-10
    write('energy_error_decomposition',energy)
    write('energy_error_summary',summarize(energy,['method','J'],
        ['moment_mse','x_mse','zz_mse','energy_mae','energy_mse','diagonal_term','cross_term','weighted_x_diagonal','weighted_zz_diagonal']))
    write('mask_paired_differences',paired)
    tfim=[]
    for n in (8,10,12,14):
        for seed in (42,43,44):
            folder=ROOT/f'results_review_table1/n{n}_seed{seed}'
            saved=json.loads((folder/'results.json').read_text())
            with np.load(folder/'predictions.npz') as d:
                target=d['exact_test'];denom=float(np.sum((target-target.mean())**2))
                for method,key in [('CQS-R','redq_original'),('CQS-P','redq_product_original'),
                                   ('NC-tuned','nc_unsquared_convex_validation_tuned')]:
                    pred=d[key];error=pred-target
                    mae=float(np.abs(error).mean());r2=1-float(np.sum(error**2))/denom
                    np.testing.assert_allclose(mae,saved['models'][key]['mae'],atol=1e-12)
                    np.testing.assert_allclose(r2,saved['models'][key]['r2'],atol=1e-12)
                    tfim.append(dict(N=n,seed=seed,method=method,mae=mae,r2=r2))
    write('tfim_per_seed',tfim);write('tfim_summary',summarize(tfim,['N','method'],['mae','r2']))
    downstream=json.loads((HERE/'evidence/warmstart_confirmation/results.json').read_text())
    grid=TFIMTask(3,3,2,True,1)
    physical=[]
    for coupling in (-.1,-.3,-.5):
        lower,upper=grid.spectrum_bounds(coupling,-.5)
        for row in downstream:
            if row['J']!=coupling:continue
            value=row['mean_energy'];reference=-4.5
            assert value>=lower-1e-9
            physical.append(dict(seed=row['seed'],J=coupling,method=row['method'],energy=value,
                exact_ground_energy=lower,reference=reference,
                reference_gap_fraction_closed=(reference-value)/(reference-lower),
                energy_improvement_per_qubit=(reference-value)/9,
                measured_candidate_improvement=row['single_training_candidate']-value))
    write('downstream_physical_scale',physical)
    write('downstream_summary',summarize(physical,['J','method'],
        ['energy','reference_gap_fraction_closed','energy_improvement_per_qubit','measured_candidate_improvement']))
    write('verification',dict(energy_mse_identity_max_error=max_identity_error,
        tfim_exact_prediction_replays=36,constants_rows=len(constants),independent_E1_pools=5,
        independent_stronger_grid_pools=5,new_training=False))
    for method in ('P-local','F-local'):
        r=next(r for r in summarize(energy,['method','J'],['moment_mse','x_mse','zz_mse','energy_mae','energy_mse','diagonal_term','cross_term'])
               if r['method']==method and r['J']==-.1)
        print('ENERGY ANALYSIS',r)
    print('PHYSICAL SCALE',next(r for r in json.loads((OUT/'downstream_summary.json').read_text()) if r['J']==-.5 and r['method']=='product'))
    print('EXISTING ANALYSIS COMPLETE')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
