"""Score only complete saved search matrices; datasets are statistical units."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
import torch
from scipy.stats import t
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[3];ROOT=WORK/'world_model_surrogate'
sys.path[:0]=[str(HERE.parents[1]),str(ROOT/'scripts'),str(ROOT)]
from run_evidence import ChainExact
from src.wmps.data.nc_tfim import TFIMTask
from search import ARMS,COUPLINGS,reference_starts
from fitting import write
from artifacts import preflight_searches


def describe(values):
    values=np.asarray(values,dtype=float);n=len(values)
    se=values.std(ddof=1)/np.sqrt(n)
    width=t.ppf(.975,n-1)*se
    return dict(mean=float(values.mean()),sd=float(values.std(ddof=1)),values=values.tolist(),ci95=[float(values.mean()-width),float(values.mean()+width)])


def analyze(phase):
    torch.set_num_threads(2)
    prefix=HERE/phase
    for seed in (45,46,47):
        marker='all_saved.json' if phase=='frozen' else 'all_searches_saved.json'
        assert (prefix/f'seed{seed}'/marker).exists(),(phase,seed,'search not finished')
    records=[]
    for seed in (45,46,47):
        folder=prefix/f'seed{seed}'
        if phase=='tuned':folder=folder/'search'
        for arm,(_,_,steps) in ARMS.items():
            for method in ('product','fourier'):
                if phase=='tuned':
                    from run_tuned import identity
                    name=identity(seed,method)
                else:name=None
                for j in COUPLINGS:records.append((folder/arm/f'{method}_J{j:+.1f}.npz',steps,name))
    preflight_searches(records)
    # Validate the entire schedule and paired starts before any physical score.
    for seed in (45,46,47):
        folder=prefix/f'seed{seed}'
        if phase=='tuned':folder=folder/'search'
        uniform=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
        for arm,(_,initial,steps) in ARMS.items():
            for method in ('product','fourier'):
                for j in COUPLINGS:
                    with np.load(folder/arm/f'{method}_J{j:+.1f}.npz') as a:
                        np.testing.assert_array_equal(a['parameters'][0],uniform if initial=='uniform' else reference_starts(seed))
                        np.testing.assert_allclose(a['sigma'],[max(0,1-k/(steps*.6)) for k in range(steps+1)],atol=1e-15)
    write(prefix/'scoring_started.json',dict(all_searches_saved_before_scoring=True,expected_searches=144))
    task=TFIMTask(1,6,2,False,-1);oracle=ChainExact(task)
    rows=[];replay_errors=[];max_statevector_error=0.
    for seed in (45,46,47):
        folder=prefix/f'seed{seed}'
        if phase=='tuned':folder=folder/'search'
        uniform=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
        for arm,(boundary,initial,steps) in ARMS.items():
            for method in ('product','fourier'):
                for j in COUPLINGS:
                    path=folder/arm/f'{method}_J{j:+.1f}.npz'
                    with np.load(path) as a:
                        parameters=a['parameters'];prediction=a['predictions'];sigma=a['sigma']
                        np.testing.assert_array_equal(parameters[0],uniform if initial=='uniform' else reference_starts(seed))
                        assert parameters.shape==(steps+1,20,2,11)
                        np.testing.assert_allclose(sigma,[max(0,1-k/(steps*.6)) for k in range(steps+1)],atol=1e-15)
                        with torch.no_grad():energy=oracle(torch.tensor(parameters[-1]),j,-1).numpy()
                        error=abs(float(energy[0])-task.energy(parameters[-1,0],j,-.5))
                        max_statevector_error=max(max_statevector_error,error)
                        if phase=='frozen' and arm=='clamp_uniform100':
                            old=(ROOT/f'results_redq_continuation_independent/chain6_l2_seed{seed}'/f'redq_J{j:+.1f}.npz') if method=='product' else HERE.parent/'extensions_20261108/chain'/f'seed{seed}'/'offline'/f'continued_J{j:+.1f}.npz'
                            with np.load(old) as b:
                                replay_errors.append(float(abs(parameters-b['parameters']).max()))
                        np.savez_compressed(path.with_suffix('.score.npz'),energy=energy,
                            prediction=prediction[-1],difference=energy-prediction[-1])
                        rows.append(dict(seed=seed,arm=arm,method=method,J=j,
                            energy=float(energy.mean()),below_reference_fraction=float(np.mean(energy < -3.-1e-6)),
                            prediction_mae=float(np.mean(abs(energy-prediction[-1]))),
                            seconds=float(a['seconds']),original_shots=24000,added_shots=0))
    summary=[];overall=[]
    for arm in ARMS:
        for j in COUPLINGS:
            group={method:[r['energy'] for r in rows if r['arm']==arm and r['method']==method and r['J']==j] for method in ('product','fourier')}
            difference=np.array(group['product'])-group['fourier']
            summary.append(dict(arm=arm,J=j,product=describe(group['product']),fourier=describe(group['fourier']),
                product_minus_fourier=describe(difference),product_lower_datasets=int(np.sum(difference<0))))
        group={method:[np.mean([r['energy'] for r in rows if r['seed']==seed and r['arm']==arm and r['method']==method]) for seed in (45,46,47)] for method in ('product','fourier')}
        difference=np.array(group['product'])-group['fourier']
        overall.append(dict(arm=arm,product=describe(group['product']),fourier=describe(group['fourier']),
            product_minus_fourier=describe(difference),product_lower_datasets=int(np.sum(difference<0))))
    write(prefix/'rows.json',rows);write(prefix/'summary.json',dict(by_coupling=summary,overall=overall))
    with (prefix/'rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert max_statevector_error<1e-10
    if replay_errors:assert max(replay_errors)<1e-5,max(replay_errors)
    write(prefix/'verification.json',dict(passed=True,searches=len(rows),datasets=3,
        all_saved_before_scoring=True,full_matrix_preflight=True,max_independent_statevector_error=max_statevector_error,
        original_trace_max_difference=max(replay_errors) if replay_errors else None,
        every_start_and_schedule_verified=True,added_shots=0,hardware_shots=0))
    for record in overall:print(phase,record['arm'],record['product']['mean'],record['fourier']['mean'],record['product_minus_fourier']['mean'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('frozen','tuned'),required=True)
    analyze(ap.parse_args().phase)
