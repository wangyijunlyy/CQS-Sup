"""Review-only full Fourier moment controls on stored independent grid datasets.

No data acquisition, simulator training labels, or changes to historical results.
Run from the repository with .venv_ibm/bin/python and no arguments for all 12 cases.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '2'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_nc_tfim_vqe import initialize_run, write_json
from run_redq_continuation import continuation
from src.wmps.data.nc_tfim import TFIMTask

OUTPUT = ROOT / 'results_review_controls'
SOURCE = ROOT / 'results_nc_tfim_vqe_independent'
Q_GRID = (1, 2, 3, 4)
RIDGE_GRID = (0., 1e-8, 1e-6, 1e-4, 1e-3, .01, .1, 1., 10.)


class FourierBasis(torch.nn.Module):
    """Complete real orthonormal basis for |k_j| <= Q, not a total-degree cut."""

    def __init__(self, layers, q):
        super().__init__()
        if layers not in (1, 2) or q < 1:
            raise ValueError('Require shared L1/L2 controls and positive Q')
        self.layers, self.q = layers, q
        self.active_indices = tuple(range(1, 2 * layers))
        frequencies = [k for k in itertools.product(range(-q, q+1),
                       repeat=len(self.active_indices))
                       if any(k) and next(v for v in k if v) > 0]
        self.register_buffer('frequencies', torch.tensor(frequencies, dtype=torch.float64))
        self.feature_count = 1 + 2 * len(frequencies)

    def forward(self, controls, sigma=0.):
        if controls.ndim != 3 or tuple(controls.shape[1:]) != (self.layers, 2):
            raise ValueError('Expected batch x layers x 2 shared controls')
        if sigma < 0:
            raise ValueError('Gaussian sigma must be nonnegative')
        active = controls.flatten(1)[:, self.active_indices]
        phase = active @ self.frequencies.T
        damping = torch.exp(-.5 * sigma**2 * self.frequencies.square().sum(1))
        return torch.cat((torch.ones_like(phase[:, :1]),
                          np.sqrt(2) * torch.cos(phase) * damping,
                          np.sqrt(2) * torch.sin(phase) * damping), dim=1)


def ridge_path(features, targets, ridges):
    """Minimize ||Phi B-Y||_F^2/n + ridge*||B_nonconstant||_F^2.

    Centering leaves the intercept unpenalized. Thin SVD gives a stable minimum
    coefficient-norm solution at zero ridge even in the underdetermined L2 fit.
    """
    if any(r < 0 for r in ridges):
        raise ValueError('Ridge coefficients must be nonnegative')
    mean_x, mean_y = features[:, 1:].mean(0), targets.mean(0)
    design, y = features[:, 1:] - mean_x, targets - mean_y
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    projected = u.T @ y
    threshold = np.finfo(float).eps * max(design.shape) * singular[0]
    solutions = []
    for ridge in ridges:
        factors = np.zeros_like(singular)
        if ridge == 0:
            np.divide(1., singular, out=factors, where=singular > threshold)
        else:
            factors = singular / (singular**2 + len(design) * ridge)
        coefficients = vt.T @ (factors[:, None] * projected)
        solutions.append(np.vstack((mean_y - mean_x @ coefficients, coefficients)))
    return solutions


def select_model(train_x, train_measured, val_x, val_measured, qs=Q_GRID, ridges=RIDGE_GRID):
    """No exact targets accepted by this fitting/selection API; no validation refit."""
    candidates, best = [], None
    for q in qs:
        basis = FourierBasis(train_x.shape[1], q)
        with torch.no_grad():
            train_features = basis(torch.as_tensor(train_x, dtype=torch.float64)).numpy()
            val_features = basis(torch.as_tensor(val_x, dtype=torch.float64)).numpy()
        for ridge, weights in zip(ridges, ridge_path(train_features, train_measured, ridges)):
            score = float(np.mean((val_features @ weights - val_measured)**2))
            record = dict(q=q, ridge=ridge, features=basis.feature_count, validation_mse=score,
                          train_mse=float(np.mean((train_features @ weights-train_measured)**2)))
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite validation score')
            candidates.append(record)
            # Grid order breaks exact ties: smaller Q, then smaller ridge.
            if best is None or score < best['selected']['validation_mse']:
                best = dict(basis=basis, weights=weights, selected=record)
    best['candidates'] = candidates
    return best


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(case, seed):
    name = f"{case['id']}_seed{seed}"
    audit = SOURCE / 'code' / name
    product = ROOT / 'results_product_redq_independent' / name
    smooth = ROOT / 'results_redq_continuation_independent' / name
    paths = dict(dataset=SOURCE / 'datasets' / name / 'dataset.npz',
                 dataset_protocol=SOURCE / 'datasets' / name / 'protocol.json',
                 starts=audit / 'initial_parameters.npy', audit_protocol=audit / 'protocol.json',
                 product_results=product / 'results.json', smooth_results=smooth / 'results.json')
    for j in case['couplings']:
        tag = f'J{j:+.1f}'
        paths[f'{tag}_results'] = audit / tag / 'results.json'
        for method in ('h_qs', 'redq', 'exact_vqe'):
            paths[f'{tag}_original_{method}'] = audit / tag / f'vqe_{method}.npz'
        paths[f'{tag}_product_direct'] = product / f'vqe_{tag}.npz'
        for method in ('nc', 'redq'):
            paths[f'{tag}_{method}_continuation'] = smooth / f'{method}_{tag}.npz'
    return paths


def validate_sources(case, seed, config, paths):
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Missing local input; no regeneration or remote fallback: {path}')
    p = json.loads(paths['audit_protocol'].read_text())
    expected = {k: case[k] for k in ('rows', 'cols', 'layers', 'shared', 'train', 'val', 'test', 'shots')}
    if json.loads(paths['dataset_protocol'].read_text()) != dict(seed=seed, **expected):
        raise ValueError('Dataset protocol mismatch')
    if p['case'] != case or p['seed'] != seed or p['vqe'] != config['vqe']:
        raise ValueError('Original experiment protocol mismatch')
    if p['hamiltonian_sign'] != 1 or p['h'] != -.5 or not case['shared']:
        raise ValueError('This control is for the +J ZZ -0.5 X shared grids only')
    product = json.loads(paths['product_results'].read_text())['protocol']
    smooth = json.loads(paths['smooth_results'].read_text())
    if product['dataset_sha256'] != sha256(paths['dataset']) or product['vqe'] != config['vqe']:
        raise ValueError('Product comparator dataset/schedule mismatch')
    if smooth['sigma'] != 1. or smooth['fraction'] != .6 or smooth['training_protocol'] != product:
        raise ValueError('Stored continuation comparator protocol mismatch')


def endpoint_summary(energies, product_reference, spectral_lower):
    energies = np.asarray(energies)
    return dict(final_mean=float(energies.mean()), restart_std=float(energies.std(ddof=1)),
                mean_gap_to_product_reference=float((energies-product_reference).mean()),
                mean_gap_to_spectral_lower=float((energies-spectral_lower).mean()),
                restarts=len(energies))


def run_one(case, seed, config, output):
    start_time = time.perf_counter()
    paths = source_paths(case, seed)
    validate_sources(case, seed, config, paths)
    sources = {key: dict(path=str(path), sha256=sha256(path)) for key, path in paths.items()}
    task = TFIMTask(case['rows'], case['cols'], case['layers'], True, 1)
    name = f"{case['id']}_seed{seed}"
    out = output / name
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(version='review-full-fourier-controls-v1', case=case, seed=seed,
        sources=sources, implementation_sha256=sha256(Path(__file__)),
        basis='complete real orthonormal Fourier cube |k_j|<=Q; 1,sqrt(2)cos,sqrt(2)sin',
        active_control_indices=list(range(1, 2*task.layers)),
        inactive_control='flattened index 0: initial RX on plus state is a global phase',
        q_grid=list(Q_GRID), ridge_grid=list(RIDGE_GRID),
        ridge_objective='||Phi B-Y||_F^2/n_train + ridge*||B_nonconstant||_F^2; free intercept',
        training_labels='stored measured_terms[0:160], all X and edge-ZZ moments',
        selection='minimum measured validation moment MSE across Q and ridge; exact ties follow grid order',
        validation_refit=False, selection_and_training_exact_labels=False,
        exact_values='only post-selection test metrics, endpoint evaluation and reference checks',
        gaussian='independent N(0,sigma^2) perturbations of active shared control coordinates; '
                 'each Fourier mode damped by exp(-sigma^2*||k||^2/2)',
        vqe=config['vqe'], continuation=dict(sigma=1., fraction=.6),
        endpoint_rule='step 100; all 20 original starts retained; no best-restart or exact selection',
        h=-.5, hamiltonian_sign=1, product_reference='zero RZZ, arbitrary RX: E=-N/2; not a ground-state claim',
        extra_measurements=0, quantum_hardware_access=False, device='cpu', threads=2,
        environment=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__))
    initialize_run(out, protocol)
    if (out / 'results.json').exists():
        print(f'Already complete {name}', flush=True)
        return json.loads((out / 'results.json').read_text())
    with np.load(paths['dataset']) as data:
        x, measured = data['X'], data['measured_terms']
    nt, nv, ne = case['train'], case['val'], case['test']
    if x.shape != (nt+nv+ne, task.layers, 2) or measured.shape != (nt+nv, task.block):
        raise ValueError('Unexpected input data shape')
    if not np.isfinite(x).all() or not np.isfinite(measured).all():
        raise ValueError('Nonfinite stored data')
    fit_start = time.perf_counter()
    fit = select_model(x[:nt], measured[:nt], x[nt:nt+nv], measured[nt:nt+nv])
    fitting_seconds = time.perf_counter() - fit_start
    basis, weights = fit['basis'], fit['weights']
    write_json(out / 'selection.json', dict(selected=fit['selected'], candidates=fit['candidates'],
                                            seconds=fitting_seconds))
    np.savez_compressed(out / 'model.npz', weights=weights, frequencies=basis.frequencies.numpy(),
                        active_indices=basis.active_indices, q=basis.q,
                        ridge=fit['selected']['ridge'], train_indices=np.arange(nt),
                        validation_indices=np.arange(nt, nt+nv))
    with torch.no_grad():
        predictions = basis(torch.tensor(x, dtype=torch.float64)).numpy() @ weights
    # Exact arrays are first opened after the measured-only selection is persisted.
    with np.load(paths['dataset']) as data:
        exact_test = data['exact_terms'][nt+nv:]
    np.savez_compressed(out / 'predictions.npz', test_x=x[nt+nv:], test_terms=predictions[nt+nv:],
                        exact_test_terms=exact_test, validation_terms=predictions[nt:nt+nv],
                        measured_validation_terms=measured[nt:nt+nv])
    starts = np.load(paths['starts'])
    if starts.shape != (config['vqe']['restarts'], task.layers, 2):
        raise ValueError('Unexpected start shape')
    records = []
    for j in case['couplings']:
        tag = f'J{j:+.1f}'
        coeff = task.coefficients(j, -.5)
        energy_weights = torch.tensor(weights @ coeff, dtype=torch.float64)
        predict = lambda z, sigma: basis(z, sigma) @ energy_weights
        original = json.loads(paths[f'{tag}_results'].read_text())
        spectral_lower = original['spectrum'][0]
        reference = -task.n / 2
        reference_controls = starts.copy()
        reference_controls[:, :, 1] = 0
        reference_exact = np.array([task.energy(z, j, -.5) for z in reference_controls])
        np.testing.assert_allclose(reference_exact, reference, atol=1e-12, rtol=0)
        methods, endpoints, timings = {}, {}, {}
        for method, sigma in (('fourier_direct', 0.), ('fourier_continuation', 1.)):
            t = time.perf_counter()
            trace = continuation(predict, starts, config['vqe'], sigma, .6)
            optimizer_seconds = time.perf_counter() - t
            t = time.perf_counter()
            exact = np.array([task.energy(z, j, -.5) for z in trace['parameters'][-1]])
            endpoint_seconds = time.perf_counter() - t
            timings[method] = dict(optimizer_seconds=optimizer_seconds,
                                   endpoint_evaluation_seconds=endpoint_seconds)
            np.savez_compressed(out / f'{method}_{tag}.npz', **trace, exact_final=exact,
                                gap_to_product_reference=exact-reference,
                                gap_to_spectral_lower=exact-spectral_lower,
                                optimizer_seconds=optimizer_seconds,
                                endpoint_evaluation_seconds=endpoint_seconds)
            endpoints[method] = exact
        stored = dict(nc_direct='original_h_qs', original_redq_direct='original_redq',
                      exact_vqe='original_exact_vqe', product_direct='product_direct',
                      nc_continuation='nc_continuation', product_continuation='redq_continuation')
        for method, source_key in stored.items():
            with np.load(paths[f'{tag}_{source_key}']) as data:
                np.testing.assert_array_equal(data['parameters'][0], starts)
                if data['parameters'].shape[0] != config['vqe']['steps']+1:
                    raise ValueError('Historical comparator update count differs')
                endpoints[method] = data['exact_final'].copy() if 'exact_final' in data else data['exact_energy'][-1].copy()
        for method, exact in endpoints.items():
            methods[method] = endpoint_summary(exact, reference, spectral_lower)
            methods[method]['mean_gap_to_exact_vqe_endpoint'] = float((exact-endpoints['exact_vqe']).mean())
        paired = endpoints['fourier_direct'] - endpoints['fourier_continuation']
        np.savez_compressed(out / f'endpoints_{tag}.npz', **endpoints, starts=starts,
                            product_reference_controls=reference_controls,
                            product_reference_exact=reference_exact,
                            paired_direct_minus_continuation=paired,
                            spectral_lower=spectral_lower)
        record = dict(J=j, h=-.5, N=task.n, product_reference=reference,
            spectral_lower=spectral_lower, methods=methods, timings=timings,
            test_moment_mse=float(np.mean((predictions[nt+nv:]-exact_test)**2)),
            test_energy_mse=float(np.mean(((predictions[nt+nv:]-exact_test) @ coeff)**2)),
            paired_direct_minus_continuation_mean=float(paired.mean()),
            paired_direct_minus_continuation_restart_std=float(paired.std(ddof=1)))
        records.append(record)
    for key, path in paths.items():
        if sha256(path) != sources[key]['sha256']:
            raise RuntimeError(f'Source changed during evaluation: {path}')
    result = dict(case=case['id'], seed=seed, selected=fit['selected'],
                  fitting_seconds=fitting_seconds, total_seconds=time.perf_counter()-start_time,
                  records=records, input_hashes_verified_unchanged=True)
    write_json(out / 'results.json', result)
    print(json.dumps(result), flush=True)
    return result


def aggregate(results, output, wall_seconds):
    rows = []
    for result in results:
        for record in result['records']:
            for method, metrics in record['methods'].items():
                is_fourier = method.startswith('fourier_')
                rows.append(dict(case=result['case'], seed=result['seed'], J=record['J'], method=method,
                    selected_q=result['selected']['q'] if is_fourier else None,
                    selected_ridge=result['selected']['ridge'] if is_fourier else None,
                    validation_moment_mse=result['selected']['validation_mse'] if is_fourier else None,
                    test_moment_mse=record['test_moment_mse'] if is_fourier else None,
                    test_energy_mse=record['test_energy_mse'] if is_fourier else None,
                    fitting_seconds=result['fitting_seconds'] if is_fourier else None,
                    review_run_seconds=result['total_seconds'], **metrics))
    with (output / 'per_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for case, j, method in sorted({(r['case'], r['J'], r['method']) for r in rows}):
        group = [r for r in rows if (r['case'], r['J'], r['method']) == (case, j, method)]
        item = dict(case=case, J=j, method=method, seeds=[r['seed'] for r in group], n_seeds=len(group))
        for key in ('final_mean', 'mean_gap_to_product_reference', 'mean_gap_to_spectral_lower',
                    'mean_gap_to_exact_vqe_endpoint', 'test_moment_mse', 'test_energy_mse'):
            values = [r[key] for r in group]
            if all(v is None for v in values):
                continue
            item[key] = dict(mean=float(np.mean(values)),
                             seed_std=float(np.std(values, ddof=1)) if len(values)>1 else None)
        summary.append(item)
    paired = []
    for case in sorted({r['case'] for r in results}):
        group = [record['paired_direct_minus_continuation_mean']
                 for result in results if result['case'] == case for record in result['records']]
        paired.append(dict(case=case, direct_minus_continuation_mean=float(np.mean(group)),
                           seed_std=float(np.std(group, ddof=1)) if len(group)>1 else None))
    write_json(output / 'aggregate.json', dict(summary=summary, paired_fourier=paired,
        aggregation='equal-weight seed means; sample SD across seeds, not pooled restart error bars',
        n_runs=len(results), invocation_wall_seconds=wall_seconds,
        sum_run_seconds=sum(r['total_seconds'] for r in results),
        sum_fitting_seconds=sum(r['fitting_seconds'] for r in results),
        caveat='Exact endpoint energies and spectral gaps are evaluations, not ground-state claims. '
               'Q and ridge use measured validation only; no exact-informed tuning.'))
    lines = ['# Review Fourier Controls', '',
        'Stored independent grid data, seeds 45/46/47; 160 train, 40 measured validation, 200 exact test.',
        'Full Fourier cubes Q=1,2,3,4, jointly selected with coefficient ridge by measured moment MSE.',
        'Initial RX excluded; independent shared-coordinate Gaussian damping, sigma=1 to 0 by update 60.',
        'Same 20 starts, 100 Adam updates, lr=0.1 and [-pi,pi] clipping for both Fourier variants.',
        'All endpoints retained. Historical NC/product/exact-VQE arrays are read-only comparators.', '',
        '| Case | Method | Exact endpoint energy (seed mean +/- SD) | Gap to E=-N/2 |',
        '|---|---|---:|---:|']
    for item in summary:
        e = item['final_mean']
        sd = f"{e['seed_std']:.6f}" if e['seed_std'] is not None else 'NA'
        lines.append(f"| {item['case']} | {item['method']} | {e['mean']:.6f} +/- {sd} | "
                     f"{item['mean_gap_to_product_reference']['mean']:.6f} |")
    lines += ['', 'The zero-RZZ plus-product state has E=-N/2 and is checked with arbitrary RX values.',
        'This is a feasible reference, not a ground-state claim. Spectral-lower-bound and paired',
        'exact-VQE endpoint gaps are in aggregate.json and per_seed.csv. Exact-VQE is itself a',
        '100-step local optimizer, not a ground-state certificate.', '',
        f"Total recorded run time: {sum(r['total_seconds'] for r in results):.3f} s; "
        f"fitting/selection: {sum(r['fitting_seconds'] for r in results):.3f} s.",
        'Optimizer and simulator endpoint timings are separated in each per-seed results.json.',
        'Per-seed protocol.json records source hashes, complete selection rules and environment.',
        'No additional quantum measurements, hardware access, or edits to historical results.', '']
    (output / 'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('grid2x3_l1', 'grid2x3_l2', 'grid3x3_l1', 'grid3x3_l2'))
    parser.add_argument('--seed', type=int, choices=(45, 46, 47))
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(OUTPUT.resolve()):
        parser.error('Output must stay inside world_model_surrogate/results_review_controls')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    config = json.loads((ROOT / 'configs/nc_tfim_vqe_independent.json').read_text())
    cases = [c for c in config['cases'] if c['shared'] and (args.case is None or c['id'] == args.case)]
    seeds = [s for s in config['seeds'] if args.seed is None or s == args.seed]
    # Fail before any fitting if even one required local dataset/comparator is absent.
    for case in cases:
        for seed in seeds:
            validate_sources(case, seed, config, source_paths(case, seed))
    t = time.perf_counter()
    results = [run_one(case, seed, config, args.out) for case in cases for seed in seeds]
    aggregate(results, args.out, time.perf_counter()-t)
    print(f'Completed {len(results)} runs in {time.perf_counter()-t:.3f} s: {args.out}', flush=True)


if __name__ == '__main__':
    main()
