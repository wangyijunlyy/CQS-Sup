"""Reconstruct direct-search histograms, acceptance and exact resource curves."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import json
import sys
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from resource_chain import ChainMeasurement, BUDGETS
from run_evidence import ChainExact, TFIMTask
from direct_jitter import initial_controls
from run_fourier_chain import write, COUPLINGS


def main():
    root = HERE/'direct_jitter'
    for seed in (45, 46, 47):
        for j in COUPLINGS:
            for method in ('spsa', 'parameter_shift_adam'):
                assert (root/f'seed{seed}_{method}_J{j:+.1f}'/'complete.json').exists()
    torch.set_num_threads(2)
    task = TFIMTask(1, 6, 2, False, -1)
    oracle, engine = ChainExact(task), ChainMeasurement()
    rows = []
    maximum_error, physical_error, total_shots = 0., 0., 0
    for seed in (45, 46, 47):
        for j in COUPLINGS:
            for method in ('spsa', 'parameter_shift_adam'):
                folder = root/f'seed{seed}_{method}_J{j:+.1f}'
                with np.load(folder/'measurements.npz') as a:
                    n = a['shots_per_group']
                    cx, cz = a['counts_x'].astype(float), a['counts_z'].astype(float)
                    np.testing.assert_array_equal(cx.sum(1), n)
                    np.testing.assert_array_equal(cz.sum(1), n)
                    vx, vz = engine.x_values(j), engine.z_values(j)
                    energy = (cx@vx+cz@vz)/n
                    variance = (np.maximum(0, cx@vx**2/n-(cx@vx/n)**2)
                        +np.maximum(0, cz@vz**2/n-(cz@vz/n)**2))/(n-1)
                    maximum_error = max(maximum_error,
                        float(np.max(abs(energy-a['energy']))),
                        float(np.max(abs(variance-a['variance']))))
                    cumulative = np.cumsum(2*n)
                    assert cumulative[-1] == 1686000
                    total_shots += int(cumulative[-1])
                    _, previous = initial_controls(seed, j)
                    for budget in BUDGETS:
                        end = int(np.searchsorted(cumulative, budget))+1
                        assert cumulative[end-1] == budget
                        candidates = a['parameters'][end-40:end-20]
                        incumbents = a['parameters'][end-20:end]
                        np.testing.assert_array_equal(incumbents, previous)
                        accepted = (energy[end-40:end-20]+2*np.sqrt(
                            variance[end-40:end-20]+variance[end-20:end]) < energy[end-20:end])
                        retained = np.where(accepted[:, None, None], candidates, incumbents)
                        with np.load(folder/f'budget{budget}.npz') as b:
                            np.testing.assert_array_equal(retained, b['parameters'])
                            np.testing.assert_array_equal(accepted, b['accepted'])
                        with torch.no_grad():
                            truth = oracle(torch.tensor(retained), j, -1).numpy()
                        physical_error = max(physical_error,
                            abs(float(truth[0])-task.energy(retained[0], j, -.5)))
                        np.save(folder/f'budget{budget}.exact.npy', truth)
                        rows.append(dict(seed=seed, J=j, method=method+'_jitter',
                            total_shots=budget, mean_energy=float(truth.mean()),
                            accepted=int(accepted.sum())))
                        previous = retained
    assert maximum_error < 1e-12 and physical_error < 1e-10
    summary = []
    for j in COUPLINGS:
        for method in ('spsa_jitter', 'parameter_shift_adam_jitter'):
            for budget in BUDGETS:
                values = [r['mean_energy'] for r in rows if r['J'] == j
                          and r['method'] == method and r['total_shots'] == budget]
                assert len(values) == 3
                summary.append(dict(J=j, method=method, total_shots=budget,
                    mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)), values=values))
    write(root/'rows.json', rows)
    write(root/'summary.json', summary)
    write(root/'verification.json', dict(passed=True, full_runs=36, checkpoints=144,
        histogram_energy_and_variance_max_error=maximum_error,
        independent_qiskit_energy_replay_error=physical_error,
        measured_acceptance_reconstructed=True, final_total_shots_per_run=1686000,
        new_simulated_shots_total=total_shots, hardware_shots=0,
        exact_scoring_after_all36_runs_saved=True))
    print('Verified direct exploratory-start runs:', len(rows)//4, flush=True)


if __name__ == '__main__':
    main()
