"""Measured-validation-tuned convex NC-objective control on cached Table1 T32 data.

The feature family and UNSQUARED L2 objective match the public profile. The
solver and alpha selection deliberately differ from the historical Adam rows.
Only results_review_table1 is writable; no simulator or hardware labels created.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.optimize import brentq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wmps.models.nc_baselines import TrigFeatureMap, frequency_vectors, nc_objective
from run_nc_tfim_vqe import initialize_run, write_json
from run_fspt_controls import split_indices

OUTPUT = ROOT / 'results_review_table1'
NS, SEEDS = (8, 10, 12, 14), (42, 43, 44)
ALPHAS = (.01, .03, .1, .3, 1., 3., 10., 30., 100., 300., 1000., 3000., 10000., 30000., 100000.)
TUNED = 'nc_unsquared_convex_validation_tuned'

PROOF = """# Objective and Convex Solver

Let A contain the original unnormalized NC support-two product features, including
the constant feature. No feature scaling, centering, target scaling, free
intercept, or division by sample count is used. The public objective is

    F(theta) = ||A theta - y||_2^2 + alpha ||theta||_2.

For a nonzero optimum, stationarity is

    (A^T A + t I) theta = A^T y,  t = alpha / (2 ||theta||_2).

Thus solve the ridge *linear system* theta(t), and find the positive root
2 t ||theta(t)||_2 = alpha. This is an exact solution to the UNSQUARED objective,
not a replacement by a fixed squared-ridge penalty. With eigenvalues d_i of
A^T A and projected b_i=(V^T A^T y)_i, the root function is
2 sqrt(sum_i b_i^2 [t/(d_i+t)]^2). It increases from zero to 2||A^T y||.
For the dual eigensystem A A^T=U diag(d) U^T, use b_i^2=d_i(U^T y)_i^2.
When alpha >= 2||A^T y||, theta=0 satisfies the norm subgradient condition.
Convexity makes these conditions sufficient for global optimality.

Every candidate is checked against the original, unscaled primal objective.
The dual is max_u -y^T u - ||u||^2/4 subject to ||A^T u||<=alpha. Scaling
u=2(A theta-y) into this ball gives a feasible lower bound and a primal-dual gap.
We require relative duality gap <=1e-7 and KKT residual/max(1,alpha) <=1e-5.
The identity-design group-shrinkage solution, independent BFGS optimization,
rank-deficient designs, and sample-replication scaling are unit tested.

Alpha is selected from a prespecified grid by noisy held-out validation MSE only.
Train=2000, validation=400, test=400, from each original cached split. No refit on
validation and no test-based model selection. Exact labels are loaded only after
the chosen alpha/coefficients have been persisted. Fixed-alpha convex optima at
0.1 and 10 are also reported to separate optimization from regularization effects.
These are new control rows; they are NOT unchanged public-code Adam/EMA runs.

Labels are the EXISTING cached scalar T32 estimates produced by the repository's
termwise marginal-based sampler, not new joint bitstring measurements or QPU data.
Seeds 42/43/44 reuse one fixed input/noise pool per N with different partitions.
Reported sample SDs therefore describe split variation, not independent pools.
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def nc_features(x, w):
    """Same ordered monomials as TrigFeatureMap, evaluating only active factors."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or w.shape[1] != x.shape[1] or not np.isin(w, [-1, 0, 1]).all():
        raise ValueError('Invalid NC feature coordinates')
    cos, sin = np.cos(x), np.sin(x)
    result = np.ones((len(x), len(w)), dtype=np.float64, order='F')
    for column, frequency in enumerate(w):
        for coordinate in np.flatnonzero(frequency):
            result[:, column] *= (cos if frequency[coordinate] == 1 else sin)[:, coordinate]
    return result


class NormL2Path:
    """Exact positive-alpha unsquared-L2 path via a shared spectral factorization."""

    def __init__(self, features, targets):
        self.a = np.asarray(features, dtype=np.float64)
        self.y = np.asarray(targets, dtype=np.float64).reshape(-1)
        if self.a.ndim != 2 or len(self.a) != len(self.y):
            raise ValueError('Feature/target shape mismatch')
        if not np.isfinite(self.a).all() or not np.isfinite(self.y).all():
            raise ValueError('Nonfinite training inputs')
        t = time.perf_counter()
        self.b = self.a.T @ self.y
        self.zero_threshold = 2 * np.linalg.norm(self.b)
        self.space = 'primal' if self.a.shape[1] <= len(self.a) else 'dual'
        gram = self.a.T @ self.a if self.space == 'primal' else self.a @ self.a.T
        self.gram_seconds = time.perf_counter()-t
        t = time.perf_counter()
        eigenvalues, self.vectors = eigh(gram, overwrite_a=True, check_finite=False, driver='evd')
        tolerance = np.finfo(float).eps * len(eigenvalues) * max(1., eigenvalues[-1])
        if eigenvalues[0] < -10*tolerance:
            raise FloatingPointError('Gram matrix is not numerically positive semidefinite')
        self.eigenvalues = np.maximum(eigenvalues, 0.)
        self.projected = self.vectors.T @ (self.b if self.space == 'primal' else self.y)
        self.norm_weights = self.projected**2
        if self.space == 'dual':
            self.norm_weights = self.norm_weights * self.eigenvalues
        self.eigh_seconds = time.perf_counter()-t

    def solve(self, alpha):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('Require finite positive alpha')
        if alpha >= self.zero_threshold:
            theta = np.zeros(self.a.shape[1])
            effective_ridge, branch = None, 'zero_subgradient'
        else:
            def root(log_t):
                t = np.exp(log_t)
                return 2*np.sqrt(np.sum(self.norm_weights*(t/(self.eigenvalues+t))**2))-alpha
            log_t = brentq(root, -60., 60., xtol=1e-12, rtol=1e-14)
            effective_ridge, branch = float(np.exp(log_t)), 'nonzero_stationary'
            spectral = self.vectors @ (self.projected/(self.eigenvalues+effective_ridge))
            theta = spectral if self.space == 'primal' else self.a.T @ spectral
        residual = self.a @ theta-self.y
        norm = np.linalg.norm(theta)
        objective = float(residual@residual + alpha*norm)
        raw_gradient = 2*self.a.T@residual
        kkt = np.linalg.norm(raw_gradient+alpha*theta/norm) if norm else max(0., np.linalg.norm(raw_gradient)-alpha)
        dual_variable = 2*residual * min(1., alpha/max(np.linalg.norm(raw_gradient), np.finfo(float).tiny))
        dual = float(-self.y@dual_variable - dual_variable@dual_variable/4)
        gap = objective-dual
        info = dict(alpha=float(alpha), effective_ridge=effective_ridge, branch=branch,
                    coefficient_norm=float(norm), objective=objective, dual_lower_bound=dual,
                    relative_duality_gap=float(max(0., gap)/max(1., abs(objective))),
                    relative_kkt_residual=float(kkt/max(1., alpha)),
                    train_mse=float(np.mean(residual**2)), zero_threshold=float(self.zero_threshold))
        if gap < -1e-7*max(1., abs(objective)) or info['relative_duality_gap'] > 1e-7 or info['relative_kkt_residual'] > 1e-5:
            raise FloatingPointError(f'Uncertified convex optimum: {info}')
        return theta, info


def select_alpha(train_features, train_noisy, val_features, val_noisy, alphas=ALPHAS):
    """The selection API deliberately has no exact or test-label argument."""
    t = time.perf_counter()
    path = NormL2Path(train_features, train_noisy)
    candidates, coefficients = [], []
    for alpha in alphas:
        theta, info = path.solve(alpha)
        info['validation_mse'] = float(np.mean((val_features@theta-np.asarray(val_noisy).ravel())**2))
        if not np.isfinite(info['validation_mse']):
            raise FloatingPointError('Nonfinite validation metric')
        candidates.append(info)
        coefficients.append(theta)
    index = min(range(len(candidates)), key=lambda i: candidates[i]['validation_mse'])
    return dict(selected=candidates[index], selected_index=index, candidates=candidates,
                coefficients=np.column_stack(coefficients), seconds=time.perf_counter()-t,
                gram_seconds=path.gram_seconds, eigh_seconds=path.eigh_seconds, solver_space=path.space)


def score(y, pred):
    y, pred = np.asarray(y).ravel(), np.asarray(pred).ravel()
    error = y-pred
    return dict(mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                r2=float(1-error@error/np.sum((y-y.mean())**2)), mse=float(np.mean(error**2)))


def source_paths(n, seed):
    original = ROOT / 'results_table1_hcs' / f'n{n}_seed{seed}'
    public = ROOT / 'results_remaining_comparisons' / f'table1_n{n}_seed{seed}_nc_hqs'
    product = ROOT / 'results_product_table1' / f'n{n}_seed{seed}'
    return dict(dataset=original.parent/'datasets'/f'table1_hcs_n{n}_T32.npz',
        split=original/'split.npz', original_results=original/'results.json',
        original_protocol=original/'protocol.json', original_nc10_pred=original/'pred_hcs.npy',
        original_nc10_coef=original/'coef_hcs.npy', original_redq_pred=original/'pred_redq_hcs.npy',
        public_results=public/'results.json', public_protocol=public/'protocol.json',
        public_split=public/'split.npz', public_pred=public/'pred.npy',
        public_coef=public/'coef.npy', public_counts=public/'counts.npy',
        product_results=product/'results.json', product_split=product/'split.npz',
        product_pred=product/'pred_product.npy',
        nc_model_source=ROOT/'src/wmps/models/nc_baselines.py',
        public_port_source=ROOT/'scripts/nc_hqs_expanded.py',
        public_source=ROOT.parent/'tmp/nc_predictive_surrogates_upstream/lib/surrogate.py',
        dataset_generation_source=ROOT/'scripts/run_table1_hcs.py')


def load_inputs(n, seed):
    paths = source_paths(n, seed)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'Local source missing; no regeneration/fallback: {path}')
    with np.load(paths['dataset']) as data:
        x, noisy = data['X'], data['y_hcs']
    if x.shape != (2800, 2, 2*n-1) or noisy.shape != (2800,):
        raise ValueError('Unexpected cached data dimensions')
    with np.load(paths['split']) as data:
        splits = [data[k] for k in ('train', 'val', 'test')]
    for a, b in zip(splits, split_indices(2800, 2000, 400, seed)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(2800))
    for key in ('public_split', 'product_split'):
        with np.load(paths[key]) as data:
            for name, ids in zip(('train', 'val', 'test'), splits):
                np.testing.assert_array_equal(data[name], ids)
    public = json.loads(paths['public_results'].read_text())
    product = json.loads(paths['product_results'].read_text())
    original = json.loads(paths['original_results'].read_text())
    p = public['protocol']
    if not all(r['completed'] for r in (public, product, original)):
        raise ValueError('Incomplete historical comparator')
    if p['n'] != n or p['seed'] != seed or public['training']['alpha'] != .1:
        raise ValueError('Unexpected original public-code profile')
    if p['dataset_sha256'] != sha(paths['dataset']) or product['protocol']['dataset_sha256'] != p['dataset_sha256']:
        raise ValueError('Historical dataset hash mismatch')
    for name, ids in zip(('train', 'val', 'test'), splits):
        if p['split_sha256'][name] != digest(ids):
            raise ValueError('Original split hash mismatch')
    for name, ids in zip(('train', 'val'), splits[:2]):
        if p[f'{name}_labels_sha256'] != digest(noisy[ids, None]):
            raise ValueError('Noisy training/validation labels differ from original')
    return paths, x, noisy, splits, public, original, product


def run_one(n, seed, out):
    started = time.perf_counter()
    paths, x, noisy, (tr, va, te), public, original, product = load_inputs(n, seed)
    sources = {key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()}
    folder = out / f'n{n}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    w = frequency_vectors(2*(2*n-1), 2)
    counts = np.load(paths['public_counts'])
    np.testing.assert_array_equal(counts[:, :, 0]-counts[:, :, 1], w)
    protocol = dict(version='table1-unsquared-convex-validation-v1', N=n, seed=seed,
        sources=sources, implementation_sha256=sha(Path(__file__)),
        objective='sum squared residual + alpha * UNSQUARED L2 coefficient norm, including intercept',
        feature_family='original ordered support<=2 unnormalized sin/cos product monomials',
        features=len(w), feature_scaling=False, centering=False, free_intercept=False,
        alpha_grid=list(ALPHAS), selection='noisy cached y_hcs validation MSE only; first grid entry breaks ties',
        training_labels='cached scalar T32 y_hcs[train]', train=2000, validation=400, test=400,
        exact_labels='test evaluation only after saved selection', validation_refit=False,
        solver='symmetric Gram eigendecomposition + monotone log-ridge scalar root; convex KKT and dual-gap certificates',
        tolerances=dict(relative_duality_gap=1e-7, kkt_over_max_1_alpha=1e-5),
        naming='New optimized/tuned NC-objective control, not unchanged original NC-code Adam rows',
        uncertainty='sample SD across three partitions of one fixed cached input/noise pool per N',
        no_extra_measurements=True, hardware=False, cpu_threads=2,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         torch=torch.__version__))
    initialize_run(folder, protocol)
    if (folder/'results.json').exists():
        return json.loads((folder/'results.json').read_text())
    t = time.perf_counter()
    # Only training/validation inputs are transformed until alpha has been selected.
    train_val_features = nc_features(x[np.r_[tr, va]].reshape(2400, -1), w)
    feature_seconds = time.perf_counter()-t
    fit = select_alpha(train_val_features[:2000], noisy[tr], train_val_features[2000:], noisy[va])
    selected, coef = fit['selected'], fit['coefficients']
    write_json(folder/'selection.json', {k:v for k,v in fit.items() if k != 'coefficients'})
    np.savez_compressed(folder/'model.npz', coefficients=coef, frequencies=w, alphas=ALPHAS,
                        selected_index=fit['selected_index'])
    np.savez_compressed(folder/'split.npz', train=tr, val=va, test=te)
    np.savez_compressed(folder/'validation_predictions.npz',
                        predictions=train_val_features[2000:]@coef, noisy_validation=noisy[va])
    # First exact-label access. This cannot influence candidates or the selected fit.
    with np.load(paths['dataset']) as data:
        exact_test = data['y_exact'][te]
    if public['protocol']['test_targets_sha256'] != digest(exact_test[:, None]):
        raise ValueError('Original exact test labels differ')
    test_features = nc_features(x[te].reshape(400, -1), w)
    fixed_indices = {f'nc_unsquared_convex_alpha{a:g}': list(ALPHAS).index(a) for a in (.1, 10.)}
    indices = dict(fixed_indices, **{TUNED: fit['selected_index']})
    predictions = {name: test_features@coef[:, index] for name,index in indices.items()}
    references = dict(nc_code_original_alpha0p1=('public_pred', public['metrics']),
        nc_adam_original_alpha10=('original_nc10_pred', original['models']['hcs']['metrics']),
        redq_original=('original_redq_pred', original['models']['redq_hcs']['metrics']),
        redq_product_original=('product_pred', product['models']['product']))
    for method, (key, historical) in references.items():
        predictions[method] = np.load(paths[key]).ravel()
        for metric,value in score(exact_test, predictions[method]).items():
            np.testing.assert_allclose(value, historical['R(h)' if metric=='mse' else metric], atol=1e-10)
    # The original coefficients independently verify feature ordering/normalization.
    np.testing.assert_allclose(test_features@np.load(paths['public_coef']).ravel(),
                               predictions['nc_code_original_alpha0p1'], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(test_features@np.load(paths['original_nc10_coef']).ravel(),
                               predictions['nc_adam_original_alpha10'], atol=1e-9, rtol=1e-9)
    np.savez_compressed(folder/'predictions.npz', **predictions, exact_test=exact_test, test_indices=te)
    result = dict(N=n, seed=seed, completed=True, selected=selected,
        models={method:score(exact_test, pred) for method,pred in predictions.items()},
        training_seconds=feature_seconds+fit['seconds'], feature_seconds=feature_seconds,
        convex_fit_and_selection_seconds=fit['seconds'], gram_seconds=fit['gram_seconds'],
        eigendecomposition_seconds=fit['eigh_seconds'], total_seconds=time.perf_counter()-started,
        historical_times=dict(nc_code_total_seconds=public['total_seconds'],
            redq_training_seconds=original['models']['redq_hcs']['training']['training_seconds'],
            redq_product_training_seconds=product['training']['training_seconds']),
        sources_verified_unchanged=True)
    for key,path in paths.items():
        if sha(path) != sources[key]['sha256']:
            raise RuntimeError(f'Source changed during run: {path}')
    write_json(folder/'results.json', result)
    print(json.dumps(dict(N=n, seed=seed, alpha=selected['alpha'], training_seconds=result['training_seconds'],
                          selected_metrics=result['models'][TUNED],
                          original_nc_metrics=result['models']['nc_code_original_alpha0p1'])), flush=True)
    return result


def audit_fourier_selection(out):
    """Reproduce the previous Q/ridge decisions without reading any exact targets."""
    import run_vqe_fourier_controls as fourier
    config = json.loads((ROOT/'configs/nc_tfim_vqe_independent.json').read_text())
    runs = []
    for case in (c for c in config['cases'] if c['shared']):
        for seed in (45, 46, 47):
            folder = ROOT/'results_review_controls'/f"{case['id']}_seed{seed}"
            protocol = json.loads((folder/'protocol.json').read_text())
            saved = json.loads((folder/'selection.json').read_text())
            data_path = Path(protocol['sources']['dataset']['path'])
            if sha(data_path) != protocol['sources']['dataset']['sha256']:
                raise ValueError('Fourier data provenance mismatch')
            if sha(Path(fourier.__file__)) != protocol['implementation_sha256']:
                raise ValueError('Fourier implementation changed since the original sweep')
            with np.load(data_path) as data:
                x, measured = data['X'], data['measured_terms']
            nt, nv = case['train'], case['val']
            def replay(controls):
                return fourier.select_model(controls[:nt], measured[:nt], controls[nt:nt+nv],
                    measured[nt:nt+nv], protocol['q_grid'], protocol['ridge_grid'])
            fit = replay(x)
            self_match = fit['selected'] == saved['selected']
            if not self_match:
                raise AssertionError('Fourier selection not reproducible from measured-only validation')
            with np.load(folder/'model.npz') as model:
                np.testing.assert_allclose(fit['weights'], model['weights'], atol=1e-12)
            poisoned = x.copy()
            poisoned[nt+nv:] = np.nan
            other = replay(poisoned)
            np.testing.assert_array_equal(other['weights'], fit['weights'])
            if other['selected'] != fit['selected']:
                raise AssertionError('Fourier selection responds to held-out test controls')
            for source in protocol['sources'].values():
                if sha(Path(source['path'])) != source['sha256']:
                    raise ValueError('Fourier historical source changed')
            runs.append(dict(case=case['id'], seed=seed, selected=saved['selected'],
                selected_reproduced_without_exact_labels=self_match, test_poisoning_invariant=True,
                selection_sha256=sha(folder/'selection.json'), protocol_sha256=sha(folder/'protocol.json')))
    old = ROOT/'results_review_controls/aggregate.json'
    summary = json.loads(old.read_text())
    write_json(out/'fourier_selection_audit.json', dict(runs=runs,
        scope='Implementation-level audit: selection reproduced using only train/validation measured labels; '
              'all test controls poisoned with NaN without changing any selected coefficient. '
              'No exact target array was accessed. This does not certify historical human grid-design decisions.',
        original_aggregate_sha256=sha(old), summary_with_full_seed_sds=summary['summary'],
        paired_with_full_seed_sds=summary['paired_fourier']))
    return summary


def aggregate(results, out, fourier, wall_seconds):
    cells, rows = [], []
    for n in NS:
        group = [r for r in results if r['N']==n]
        models = {}
        for method in group[0]['models']:
            models[method] = {}
            for metric in ('mae', 'rmse', 'r2', 'mse'):
                values = [r['models'][method][metric] for r in group]
                models[method][metric] = dict(mean=float(np.mean(values)), seed_std=float(np.std(values, ddof=1)))
        training = [r['training_seconds'] for r in group]
        cells.append(dict(N=n, models=models, selected_alphas=[r['selected']['alpha'] for r in group],
            training_seconds=dict(mean=float(np.mean(training)), seed_std=float(np.std(training, ddof=1)))))
        for r in group:
            for method, metrics in r['models'].items():
                rows.append(dict(N=n, seed=r['seed'], method=method, **metrics,
                    selected_alpha=r['selected']['alpha'] if method==TUNED else None,
                    tuning_training_seconds=r['training_seconds'] if method==TUNED else None))
    with (out/'per_seed.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(out/'aggregate.json', dict(n_runs=len(results), cells=cells, seeds=SEEDS,
        uncertainty='sample SD across three partitions of one fixed cached pool per N, not independent label pools',
        invocation_wall_seconds=wall_seconds, sum_training_seconds=sum(r['training_seconds'] for r in results),
        sum_run_seconds=sum(r['total_seconds'] for r in results)))
    def formatted(stats):
        return f"{stats['mean']:.6f} +/- {stats['seed_std']:.6f}"
    lines = ['# Table1 Validation-Tuned NC-Objective Control', '',
        'All 12 N/seed combinations; cached scalar T32 labels and original 2000/400/400 splits.',
        'No QPU jobs, new labels, manuscript edits, or edits to original result rows.',
        'Same NC features and unsquared-L2 objective; alpha chosen solely by noisy validation MSE.',
        'This new convex-solver control is distinct from the unchanged public-code Adam/EMA rows.', '',
        '| N | Method | Exact-test MAE +/- SD | Exact-test R2 +/- SD |',
        '|---:|---|---:|---:|']
    for cell in cells:
        for method, metrics in cell['models'].items():
            lines.append(f"| {cell['N']} | {method} | {formatted(metrics['mae'])} | {formatted(metrics['r2'])} |")
    lines += ['', '## Selection and Training Time', '',
        'Times include train/validation features, factorization, all alpha solves/certificates and validation selection;',
        'test prediction/evaluation is excluded. One factorization is reused across the entire alpha grid.', '',
        '| N | Selected alphas, seeds 42/43/44 | Training seconds +/- SD |', '|---:|---|---:|']
    for cell in cells:
        lines.append(f"| {cell['N']} | {cell['selected_alphas']} | {formatted(cell['training_seconds'])} |")
    lines += ['', f'Total training/selection time: {sum(r["training_seconds"] for r in results):.3f} s.',
        'The split SDs are not independent-data-pool uncertainties. All original prediction metrics are recomputed',
        'and checked; saved original coefficients reproduce the feature normalization and ordering.',
        'See OBJECTIVE.md for the exact unsquared objective equivalence and optimality certificates.', '',
        '## Fourier Selection Audit', '',
        'All 12 Q/ridge choices and coefficients reproduced from measured train/validation labels only.',
        'Replacing every held-out test control by NaN changes no choice or coefficient; exact target arrays',
        'are not read by this audit. This checks code selection, not historical human grid-design decisions.', '',
        '| Grid | Fourier direct +/- SD | Fourier continuation +/- SD | Paired improvement +/- SD |',
        '|---|---:|---:|---:|']
    for pair in fourier['paired_fourier']:
        methods = {r['method']:r for r in fourier['summary'] if r['case']==pair['case']}
        gain = dict(mean=pair['direct_minus_continuation_mean'], seed_std=pair['seed_std'])
        lines.append(f"| {pair['case']} | {formatted(methods['fourier_direct']['final_mean'])} | "
                     f"{formatted(methods['fourier_continuation']['final_mean'])} | {formatted(gain)} |")
    lines += ['', 'All comparator SDs and exact gap SDs are retained in fourier_selection_audit.json.',
              'Fourier energies are exact endpoint evaluations, not ground-state claims.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(OUTPUT.resolve()):
        parser.error('Output must remain inside results_review_table1')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    # Preflight all 12 runs before starting a substantial factorization.
    for n in NS:
        for seed in SEEDS:
            load_inputs(n, seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'OBJECTIVE.md').write_text(PROOF)
    start = time.perf_counter()
    results = [run_one(n, seed, args.out) for n in NS for seed in SEEDS]
    fourier = audit_fourier_selection(args.out)
    aggregate(results, args.out, fourier, time.perf_counter()-start)
    print(f'Completed 12 Table1 runs and Fourier audit in {time.perf_counter()-start:.3f} seconds', flush=True)


if __name__ == '__main__':
    main()
