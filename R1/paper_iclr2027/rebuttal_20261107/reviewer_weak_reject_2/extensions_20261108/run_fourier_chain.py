"""Complete Fourier on the original chain pools, with matched local correction.

No historical fit changes. Exact energies only score saved search outputs.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
ROOT = WORK/'world_model_surrogate'
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT), str(HERE.parents[1])]
from complete_fourier import CompleteFourier
from run_supplement_e1 import fit_fourier, fourier_kernel, kernel_predict
from run_redq_continuation import continuation
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import causal_parameter_masks
from chain_local_calibration import measure, fit_correction, propose, ACTIVE

COUPLINGS = (-.5, -.3, -.1, .1, .3, .5)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def fit_global(seed, out):
    dest = out/f'seed{seed}'/'global'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'fit.json').exists():
        return CompleteFourier(dest)
    task = TFIMTask(1, 6, 2, False, -1)
    source = ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz'
    with np.load(source) as a:
        x, y = a['X'], a['measured_terms']
    masks = causal_parameter_masks(6, task.edges, 2, False).reshape(11, 22)
    flat = x.reshape(len(x), -1)
    started = time.perf_counter()
    models = fit_fourier(flat[:2000], y[:2000], flat[2000:2400], y[2000:2400],
                         masks, dest)
    # Replay the persisted model against the independently evaluated kernel path.
    model = CompleteFourier(dest)
    replay = np.empty((8, 11))
    for mask, outputs, fitted in models:
        replay[:, outputs] = kernel_predict(fitted,
            fourier_kernel(flat[2000:2008], flat[:2000], mask))[0]
    with torch.no_grad():
        restored = model(torch.tensor(x[2000:2008])).numpy()
    error = float(np.max(np.abs(restored-replay)))
    assert error < 1e-9, error
    write(dest/'fit.json', dict(seed=seed, source=str(source), train=2000, validation=400,
        Q=1, outputs=11, mask='original observable dependencies',
        solver='complete tensor-kernel centered scaled ridge, converged eigensolve',
        regularization='original eight-value Fourier grid, validation mean moment MSE',
        replay_max_error=error, seconds=time.perf_counter()-started,
        test_evaluated=False, extra_global_measurements=0))
    print('GLOBAL FIT SAVED', seed, 'replay', error, flush=True)
    return model


def offline(model, task, seed, out):
    dest = out/f'seed{seed}'/'offline'
    dest.mkdir(exist_ok=True)
    starts = np.load(ROOT/f'results_nc_tfim_vqe_independent/code/chain6_l2_seed{seed}/initial_parameters.npy')
    settings = dict(steps=100, lr=.1, clip=float(np.pi))
    for j in COUPLINGS:
        c = torch.tensor(task.coefficients(j, -.5))
        for mode, sigma in [('direct', 0.), ('continued', 1.)]:
            path = dest/f'{mode}_J{j:+.1f}.npz'
            if path.exists():
                continue
            trace = continuation(lambda z, s: model(z, s)@c, starts, settings, sigma, .6)
            np.savez_compressed(path, **trace)
            print('OFFLINE SAVED', seed, j, mode, flush=True)
    write(dest/'all_saved.json', dict(searches=12, exact_scoring_started=False))


def local(model, task, seed, out):
    for ji, j in enumerate(COUPLINGS):
        dest = out/f'seed{seed}'/f'local_J{j:+.1f}'
        dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            continue
        c = torch.tensor(task.coefficients(j, -.5))
        prior = lambda z: model(z)@c
        rng = np.random.default_rng(900000+seed*100+ji)
        center = np.zeros((2, 11))
        center[0, 6:9] = np.pi
        logs = []
        start = time.perf_counter()
        for iteration, radius in enumerate((.8, .6, .4)):
            z = rng.uniform(-1, 1, (256, 16))
            x = np.repeat(center[None], 256, axis=0).reshape(256, -1)
            x[:, ACTIVE] += z*(radius/4)
            x = x.reshape(256, 2, 11)
            records = [measure(task, p, j, rng) for p in x]
            measured = np.array([p[1] for p in records])
            variance = np.array([p[2] for p in records])
            with torch.no_grad():
                # Bound temporary kernel memory on complete data, without sampling.
                prediction = np.concatenate([prior(torch.tensor(b)).numpy()
                                              for b in np.array_split(x, 8)])
            weights, selection = fit_correction(z, measured, prediction)
            candidates, predicted = propose(prior, center, radius, weights, rng)
            _, incumbent, incumbent_variance = measure(task, center, j, rng)
            results = [measure(task, p, j, rng) for p in candidates]
            energies = np.array([p[1] for p in results])
            variances = np.array([p[2] for p in results])
            accepted = energies+2*np.sqrt(variances+incumbent_variance) < incumbent
            retained = np.where(accepted[:, None, None], candidates, center[None])
            index = int(np.argmin(np.where(accepted, energies, np.inf))) if accepted.any() else None
            np.savez_compressed(dest/f'round{iteration}.npz', center=center, design=x,
                design_moments=np.stack([p[0] for p in records]), design_energy=measured,
                design_variance=variance, prior_prediction=prediction,
                correction_weights=weights, candidate_parameters=candidates,
                candidate_prediction=predicted, candidate_energy=energies,
                candidate_variance=variances, incumbent_energy=incumbent,
                incumbent_variance=incumbent_variance, accepted=accepted,
                retained_parameters=retained)
            logs.append(dict(round=iteration, accepted=int(accepted.sum()),
                             selection=selection, radius=radius))
            if index is not None:
                center = candidates[index].copy()
        np.save(dest/'final_parameters.npy', retained)
        write(dest/'complete.json', dict(seed=seed, J=j, method='complete_fourier',
            rounds=logs, simulated_shots=1662000, original_fit_shots=24000,
            total_shots=1686000, seconds=time.perf_counter()-start,
            exact_scoring_started=False, hardware_shots=0))
        print('LOCAL SAVED', seed, j, [r['accepted'] for r in logs], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True, choices=(45, 46, 47))
    ap.add_argument('--out', type=Path, default=HERE/'chain')
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out/f'protocol_seed{args.seed}.json', dict(seed=args.seed,
        source='original independent VQE dataset, unchanged measured moments',
        couplings=COUPLINGS, global_model='complete masked Q1 tensor-Fourier',
        local_protocol='unchanged chain_local_calibration.py measure/fit/propose/acceptance',
        local_runs=6, added_shots_per_run=1662000, total_shots_per_run=1686000,
        offline='original saved starts, direct/+C,100Adam lr.1,clip pi',
        evaluation='only after all original three-seed searches are saved'))
    task = TFIMTask(1, 6, 2, False, -1)
    model = fit_global(args.seed, args.out)
    offline(model, task, args.seed, args.out)
    local(model, task, args.seed, args.out)
    write(args.out/f'seed{args.seed}'/'all_saved.json', dict(global_fit=True,
        offline_searches=12, local_runs=6, exact_scoring_started=False))


if __name__ == '__main__':
    main()
