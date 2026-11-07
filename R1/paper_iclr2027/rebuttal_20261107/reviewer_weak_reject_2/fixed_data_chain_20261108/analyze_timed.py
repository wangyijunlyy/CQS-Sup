"""Reconstruct the time-matched selection and score retained controls only."""
from pathlib import Path
import json
import numpy as np
import torch
from analyze import HERE,ROOT,COUPLINGS,ChainExact,TFIMTask,describe
from fitting import write
from timed import select_candidates
from artifacts import preflight_timed
from run_timed import METHODS,timed_identity


def main():
    torch.set_num_threads(2)
    folder=HERE/'timed'
    for seed in (45,46,47):assert (folder/f'seed{seed}'/'all_saved.json').exists()
    protocol=json.loads((HERE/'time_protocol.json').read_text());cap=protocol['cap_seconds']
    records=[(folder/f'seed{seed}'/f'{method}_J{j:+.1f}.npz',timed_identity(seed,method))
        for seed in (45,46,47) for method in METHODS for j in COUPLINGS]
    preflight_timed(records,cap)
    for seed in (45,46,47):
        first=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
        for method in METHODS:
            for j in COUPLINGS:
                with np.load(folder/f'seed{seed}'/f'{method}_J{j:+.1f}.npz') as a:
                    np.testing.assert_array_equal(a['block_starts'][0],first)
                    pool=a['block_parameters'].reshape(-1,2,11);pred=a['block_predictions'].reshape(-1)
                    chosen=select_candidates(pool,pred)
                    np.testing.assert_array_equal(pool[chosen],a['retained_parameters'])
                    np.testing.assert_array_equal(pred[chosen],a['retained_prediction'])
    write(folder/'scoring_started.json',dict(all54searches_saved=True))
    task=TFIMTask(1,6,2,False,-1);oracle=ChainExact(task)
    rows=[];max_statevector_error=0.;max_overrun=0.
    for seed in (45,46,47):
        first=np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
        for j in COUPLINGS:
            for method in METHODS:
                path=folder/f'seed{seed}'/f'{method}_J{j:+.1f}.npz'
                with np.load(path) as a:
                    assert float(a['cap'])==cap
                    assert np.all(a['eligible_seconds']<=cap)
                    max_overrun=max(max_overrun,float(a['actual_seconds'])-cap)
                    np.testing.assert_array_equal(a['block_starts'][0],first)
                    x=a['block_parameters'].reshape(-1,2,11);p=a['block_predictions'].reshape(-1)
                    indices=select_candidates(x,p)
                    np.testing.assert_array_equal(x[indices],a['retained_parameters'])
                    np.testing.assert_array_equal(p[indices],a['retained_prediction'])
                    with torch.no_grad():energy=oracle(torch.tensor(a['retained_parameters']),j,-1).numpy()
                    max_statevector_error=max(max_statevector_error,abs(float(energy[0])-task.energy(a['retained_parameters'][0],j,-.5)))
                    np.savez_compressed(path.with_suffix('.score.npz'),energy=energy)
                    rows.append(dict(seed=seed,J=j,method=method,energy=float(energy.mean()),
                        below_reference_fraction=float(np.mean(energy < -3.-1e-6)),
                        prediction_mae=float(abs(energy-a['retained_prediction']).mean()),
                        blocks=int(a['completed_blocks']),actual_seconds=float(a['actual_seconds'])))
    by_j=[]
    for j in COUPLINGS:
        values={m:[r['energy'] for r in rows if r['J']==j and r['method']==m] for m in METHODS}
        by_j.append(dict(J=j,product=describe(values['product']),fourier=describe(values['fourier']),
            fourier_original=describe(values['fourier_original']),
            product_minus_fourier=describe(np.array(values['product'])-values['fourier']),
            product_minus_original_fourier=describe(np.array(values['product'])-values['fourier_original'])))
    values={m:[np.mean([r['energy'] for r in rows if r['seed']==seed and r['method']==m]) for seed in (45,46,47)] for m in METHODS}
    overall=dict(product=describe(values['product']),fourier=describe(values['fourier']),
        fourier_original=describe(values['fourier_original']),
        product_minus_fourier=describe(np.array(values['product'])-values['fourier']),
        product_minus_original_fourier=describe(np.array(values['product'])-values['fourier_original']))
    write(folder/'rows.json',rows);write(folder/'summary.json',dict(by_coupling=by_j,overall=overall))
    assert max_statevector_error<1e-10
    write(folder/'verification.json',dict(passed=True,searches=54,selection_reconstructed=True,
        full_matrix_preflight=True,complete_pool_copies_within_cap=True,max_overrun_seconds=max_overrun,
        max_independent_statevector_error=max_statevector_error,added_shots=0,hardware_shots=0))
    print('TIMED OVERALL',overall,flush=True)


if __name__=='__main__':main()
