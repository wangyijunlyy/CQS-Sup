"""Frozen E1 representation-by-mask comparison on independent measured datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '2'

import numpy as np
from scipy.linalg import eigh
import torch

from run_fspt8_fair import fit, seed_all
from run_nc_tfim_vqe import initialize_run, write_json
from run_product_mask_control import build_model, initial_state, make_masks, score_terms
from src.wmps.data.nc_tfim import TFIMTask, shadow_terms

RIDGES = (1e-8, 1e-6, 1e-4, .001, .01, .1, 1., 10.)
ROOT = Path(__file__).resolve().parents[1]


def fourier_kernel(x, z, mask):
    x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    result = np.ones((len(x), len(z)))
    for d in np.flatnonzero(mask):
        result *= 1 + 2*np.cos(x[:, d, None]-z[None, :, d])
    return result


def kernel_path(kernel, targets, ridges):
    kernel, targets = np.asarray(kernel, float), np.asarray(targets, float)
    mean = kernel.mean(0)
    overall = float(mean.mean())
    centered = kernel-mean[None, :]-mean[:, None]+overall
    scale = float(np.trace(centered)/len(kernel))
    target_mean = targets.mean(0)
    if scale <= np.finfo(float).eps * max(1., float(np.max(np.abs(kernel)))) * 32:
        return dict(mean=mean, overall=overall, scale=1., target_mean=target_mean,
                    weights=np.zeros((len(ridges), len(kernel), targets.shape[1])))
    values, vectors = eigh(centered/scale, check_finite=True, driver='evd')
    tolerance = np.finfo(float).eps * len(kernel) * max(1., abs(values).max()) * 64
    if values.min() < -tolerance:
        raise ValueError('Fourier kernel is not positive semidefinite within tolerance')
    values = np.maximum(values, 0)
    projected = vectors.T @ (targets-target_mean)
    weights = np.stack([vectors @ (projected/(values[:, None]+len(kernel)*ridge))
                        for ridge in ridges])
    return dict(mean=mean, overall=overall, scale=scale, target_mean=target_mean, weights=weights)


def kernel_predict(fitted, cross):
    centered = (cross-cross.mean(1, keepdims=True)-fitted['mean'][None, :]
                +fitted['overall'])/fitted['scale']
    return np.einsum('vn,lnm->lvm', centered, fitted['weights'])+fitted['target_mean']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data(task, folder, seed, mode):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'dataset.npz'
    spec = dict(mode=mode, data_seed=seed, measurement_seed=seed+2000,
                splits=[2000, 400, 400], shots=10, n=6, layers=2,
                labels='joint random-Pauli bitstrings; exact test moments')
    initialize_run(folder, spec)
    if path.exists():
        meta = json.loads((folder/'hashes.json').read_text())
        if digest(path) != meta['dataset'] or digest(folder/'shadow_records.npz') != meta['records']:
            raise ValueError('Dataset provenance changed')
        return path
    if mode == 'development':
        source = ROOT/'results_nc_tfim_vqe_independent/datasets'/f'chain6_l2_seed{seed}'
        # Reuse the old measured dataset without changing its records.
        import shutil
        shutil.copyfile(source/'dataset.npz', path)
        shutil.copyfile(source/'shadow_records.npz', folder/'shadow_records.npz')
        spec['development_source'] = str(source)
    else:
        controls_rng = np.random.default_rng(seed)
        measure_rng = np.random.default_rng(seed+2000)
        x = controls_rng.uniform(-np.pi, np.pi, size=(2800, 2, 11))
        exact = np.stack([task.exact_terms(t) for t in x])
        bits = np.empty((2400, 10, 6), dtype=np.int8)
        recipes = np.empty_like(bits)
        measured = np.empty((2400, 11))
        for i in range(2400):
            bits[i], recipes[i] = task.sample_shadow(x[i], 10, measure_rng)
            measured[i] = shadow_terms(bits[i], recipes[i], task.edges)
        np.savez_compressed(folder/'shadow_records.npz', bits=bits, recipes=recipes)
        np.savez_compressed(path, X=x, measured_terms=measured, exact_terms=exact)
    write_json(folder/'hashes.json', dict(dataset=digest(path), records=digest(folder/'shadow_records.npz'),
                                        source=spec.get('development_source')))
    return path


def fit_fourier(xtr, ytr, xval, yval, masks, folder):
    groups = {}
    for o, mask in enumerate(masks):
        groups.setdefault(tuple(mask.astype(bool)), []).append(o)
    candidates = np.zeros((len(RIDGES), len(xval), ytr.shape[1]))
    models = []
    for g, (mask, outputs) in enumerate(groups.items()):
        mask = np.array(mask, dtype=bool)
        kernel = fourier_kernel(xtr, xtr, mask)
        fitted = kernel_path(kernel, ytr[:, outputs], RIDGES)
        candidates[:, :, outputs] = kernel_predict(fitted, fourier_kernel(xval, xtr, mask))
        models.append((mask, outputs, fitted))
    errors = np.mean((candidates-yval)**2, axis=(1, 2))
    best = int(np.argmin(errors))
    write_json(folder/'selection.json', dict(ridges=RIDGES, validation_mse=errors.tolist(),
                                            selected=RIDGES[best], shared_lambda=True))
    for g, (mask, outputs, fitted) in enumerate(models):
        fitted['weights'] = fitted['weights'][best:best+1]
        np.savez_compressed(folder/f'kernel_group{g}.npz', train_x=xtr, mask=mask,
                            outputs=outputs, **fitted)
    return models


def run(seed, out, device, mode):
    if mode == 'confirmation' and seed not in range(1001, 1006):
        raise ValueError('Confirmation seeds must be 1001..1005')
    if mode == 'development' and seed not in (45, 46, 47):
        raise ValueError('Development must reuse old seeds')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    task = TFIMTask(1, 6, 2, False, -1)
    base = out/mode/f'seed{seed}'
    path = data(task, base/'data', seed, mode)
    with np.load(path) as a:
        x, measured = a['X'].copy(), a['measured_terms'].copy()
    if x.shape != (2800, 2, 11) or measured.shape != (2400, 11):
        raise ValueError('Unexpected full experiment dataset shape')
    initialization = seed+1000 if mode == 'confirmation' else seed
    masks = make_masks(initialization)
    initial = initial_state(initialization)
    sources = {str(p.relative_to(ROOT)): digest(p) for p in
               (Path(__file__), ROOT/'scripts/run_fspt8_fair.py', ROOT/'scripts/run_product_mask_control.py',
                ROOT/'src/wmps/models/periodic_redq.py', ROOT/'src/wmps/data/nc_tfim.py')}
    for name, arm in (('P-local', 'actual'), ('P-full', 'full'), ('F-local', 'actual'), ('F-full', 'full')):
        folder = base/name
        folder.mkdir(parents=True, exist_ok=True)
        spec = dict(experiment='E1', arm=name, seed=seed, initialization_seed=initialization,
                    mode=mode, dataset_sha256=digest(path), rank=32, q=1, ridges=RIDGES,
                    epoch_cap=300, patience=30, device=device, torch=torch.__version__,
                    numpy=np.__version__, sources=sources, test_selection=False,
                    mask_sha256=hashlib.sha256(masks[arm].tobytes()).hexdigest())
        initialize_run(folder, spec)
        if (folder/'results.json').exists():
            print(f'SKIP completed {mode} {seed} {name}', flush=True)
            continue
        print(f'START {mode} {seed} {name}', flush=True)
        started = time.perf_counter()
        np.save(folder/'mask.npy', masks[arm])
        if name.startswith('P'):
            model, training = fit(lambda: build_model(initial, masks[arm]),
                                  x[:2000], measured[:2000], x[2000:2400], measured[2000:2400],
                                  initialization, device, folder, 'product', 300)
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(x[2400:], dtype=torch.float32, device=device))[0].cpu().numpy()
            cost = dict(training=training, allocated_parameters=sum(p.numel() for p in model.parameters()))
        else:
            flat = x.reshape(len(x), -1)
            models = fit_fourier(flat[:2000], measured[:2000], flat[2000:2400], measured[2000:2400],
                                masks[arm].reshape(11, -1), folder)
            pred = np.empty((400, 11))
            for mask, outputs, fitted in models:
                pred[:, outputs] = kernel_predict(fitted, fourier_kernel(flat[2400:], flat[:2000], mask))[0]
            cost = dict(groups=len(models), candidate_ridges=len(RIDGES), gram_bytes=2000**2*8)
        with np.load(path) as a:
            target = a['exact_terms'][2400:]
        np.savez_compressed(folder/'test_predictions.npz', prediction=pred, exact=target, x=x[2400:])
        result = dict(completed=True, protocol=spec, metrics=score_terms(target, pred),
                      seconds=time.perf_counter()-started, cost=cost)
        write_json(folder/'results.json', result)
        print(json.dumps(dict(seed=seed, arm=name, metrics=result['metrics']['moments'],
                              seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--mode', default='confirmation', choices=('confirmation', 'development'))
    args = parser.parse_args()
    run(args.seed, args.out, args.device, args.mode)
