"""Replay the complete chain extension and compare all saved resource arms."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import json
import sys
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
ROOT = WORK/'world_model_surrogate'
OLD = HERE.parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT), str(OLD)]
from complete_fourier import CompleteFourier
from run_evidence import ChainExact, TFIMTask
from chain_local_calibration import fit_correction
from run_fourier_chain import write, COUPLINGS


def describe(values):
    values = [float(v) for v in values]
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)), values=values)


def main():
    torch.set_num_threads(2)
    out = HERE/'chain'
    for seed in (45, 46, 47):
        assert (out/f'seed{seed}'/'all_saved.json').exists()
    task = TFIMTask(1, 6, 2, False, -1)
    oracle = ChainExact(task)
    rows, curves = [], []
    max_replay = 0.
    max_fit_error = 0.
    first_designs_match = 0
    old_summary = json.loads((OLD/'evidence/chain_depth_recovery/local_calibration/summary.json').read_text())
    old_nc = json.loads((OLD/'evidence/nc_vqe_tuned.json').read_text())
    for seed in (45, 46, 47):
        base = out/f'seed{seed}'
        model = CompleteFourier(base/'global')
        with np.load(ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz') as a:
            test_x, test_exact = a['X'][2400:], a['exact_terms'][2400:]
        with torch.no_grad():
            test_pred = np.concatenate([model(torch.tensor(b)).numpy() for b in np.array_split(test_x, 16)])
        np.savez_compressed(base/'global/test_predictions.npz', prediction=test_pred, exact=test_exact)
        write(base/'global/test_metrics.json', dict(moment_mse=float(np.mean((test_pred-test_exact)**2))))
        starts = np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
        for j in COUPLINGS:
            for mode in ('direct', 'continued'):
                path = base/'offline'/f'{mode}_J{j:+.1f}.npz'
                with np.load(path) as a:
                    np.testing.assert_array_equal(a['parameters'][0], starts)
                    with torch.no_grad():
                        truth = oracle(torch.tensor(a['parameters'][-1]), j, -1).numpy()
                    max_replay = max(max_replay, abs(float(truth[0])-task.energy(a['parameters'][-1, 0], j, -.5)))
                np.save(path.with_suffix('.exact.npy'), truth)
                rows.append(dict(seed=seed, J=j, method='complete_fourier', workflow=mode,
                    mean_energy=float(truth.mean()), total_shots=24000, added_shots=0))
            dest = base/f'local_J{j:+.1f}'
            meta = json.loads((dest/'complete.json').read_text())
            assert meta['simulated_shots'] == 1662000 and meta['total_shots'] == 1686000
            with np.load(dest/'round0.npz') as a:
                original = OLD/f'evidence/chain_depth_recovery/local_calibration/seed{seed}_product_J{j:+.1f}/round0.npz'
                with np.load(original) as old:
                    np.testing.assert_array_equal(a['design'], old['design'])
                    np.testing.assert_array_equal(a['design_moments'], old['design_moments'])
                first_designs_match += 1
            for iteration, radius in enumerate((.8, .6, .4)):
                with np.load(dest/f'round{iteration}.npz') as a:
                    accepted = a['candidate_energy']+2*np.sqrt(a['candidate_variance']+a['incumbent_variance']) < a['incumbent_energy']
                    np.testing.assert_array_equal(accepted, a['accepted'])
                    retained = np.where(accepted[:, None, None], a['candidate_parameters'], a['center'][None])
                    np.testing.assert_array_equal(retained, a['retained_parameters'])
                    z = (a['design'].reshape(256, 22)[:, 6:]-a['center'].reshape(1, 22)[:, 6:])*4/radius
                    weights, selection = fit_correction(z, a['design_energy'], a['prior_prediction'])
                    error = float(np.max(np.abs(weights-a['correction_weights'])))
                    max_fit_error = max(max_fit_error, error)
                    assert error < 1e-8, error
                    assert selection['alpha'] == meta['rounds'][iteration]['selection']['alpha']
                    with torch.no_grad():
                        truth = oracle(torch.tensor(retained), j, -1).numpy()
                np.save(dest/f'round{iteration}.exact.npy', truth)
                curves.append(dict(seed=seed, J=j, method='complete_fourier',
                    total_shots=24000+(iteration+1)*554000,
                    mean_energy=float(truth.mean())))
            final = np.load(dest/'final_parameters.npy')
            np.testing.assert_array_equal(final, retained)
            max_replay = max(max_replay, abs(float(truth[0])-task.energy(final[0], j, -.5)))
            rows.append(dict(seed=seed, J=j, method='complete_fourier', workflow='local',
                mean_energy=float(truth.mean()), total_shots=1686000, added_shots=1662000))
            for method in ('product', 'tuned_nc', 'local_only'):
                old_folder = OLD/f'evidence/chain_depth_recovery/local_calibration/seed{seed}_{method}_J{j:+.1f}'
                for iteration in range(3):
                    with np.load(old_folder/f'round{iteration}.npz') as a:
                        with torch.no_grad():
                            old_truth = oracle(torch.tensor(a['retained_parameters']), j, -1).numpy()
                    curves.append(dict(seed=seed, J=j, method=method,
                        total_shots=(0 if method == 'local_only' else 24000)+(iteration+1)*554000,
                        mean_energy=float(old_truth.mean())))
    assert first_designs_match == 18 and max_replay < 1e-10
    write(out/'rows.json', rows)
    summary = []
    for j in COUPLINGS:
        for mode in ('direct', 'continued', 'local'):
            group = [r for r in rows if r['J'] == j and r['workflow'] == mode]
            summary.append(dict(J=j, method='complete_fourier', workflow=mode,
                energy=describe([r['mean_energy'] for r in group]),
                total_shots=group[0]['total_shots']))
        f = next(r for r in summary if r['J'] == j and r['workflow'] == 'local')['energy']['values']
        p = next(r for r in old_summary if r['J'] == j and r['method'] == 'product')['values']
        summary.append(dict(J=j, contrast='Fourier_local_minus_product_local',
            difference=describe(np.array(f)-np.array(p)), product_lower_runs=int(np.sum(np.array(p)<f))))
    resources = OLD/'evidence/resource_controls_20261108'
    if (resources/'chain_verification.json').exists():
        check = json.loads((resources/'chain_verification.json').read_text())
        assert check['passed']
        direct = json.loads((resources/'chain_summary.json').read_text())
        for r in direct:
            if r['method'] in ('spsa', 'parameter_shift_adam'):
                for index, seed in enumerate((45, 46, 47)):
                    curves.append(dict(seed=seed, J=r['J'], method=r['method'],
                        total_shots=r['total_shots'], mean_energy=r['values'][index]))
    curve_summary = []
    for j in COUPLINGS:
        for method in sorted(set(r['method'] for r in curves)):
            group = [r for r in curves if r['J'] == j and r['method'] == method]
            for budget in sorted(set(r['total_shots'] for r in group)):
                values = [r['mean_energy'] for r in group if r['total_shots'] == budget]
                assert len(values) == 3
                curve_summary.append(dict(J=j, method=method, total_shots=budget, energy=describe(values)))
    write(out/'summary.json', summary)
    write(out/'resource_curves.json', curve_summary)
    write(out/'verification.json', dict(passed=True, global_fits=3, offline_searches=36,
        local_runs=18, local_rounds=54, first_round_measurements_match_original=first_designs_match,
        acceptance_reconstructed=True, local_validation_ridge_replayed=True,
        correction_weights_max_difference=max_fit_error, independent_energy_replay_error=max_replay,
        added_simulated_shots=18*1662000, hardware_shots=0,
        test_scoring_after_all_searches_saved=True))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
