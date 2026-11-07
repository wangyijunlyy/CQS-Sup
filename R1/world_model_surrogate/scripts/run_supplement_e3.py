"""E3 independent L2 VQE confirmation runner (no QPU access).

Launch confirmation only after the parent coordinates protocol/source freeze.
--development-smoke uses old development seeds and reduced budgets, never evidence.
Resume reuses only hashed, completed stages; an interrupted stage is recomputed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_vqe_fourier_controls import Q_GRID, RIDGE_GRID, FourierBasis, select_model
from run_fspt8_fair import fit
from run_nc_tfim_vqe import initialize_run, write_json
from run_redq_continuation import continuation
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ
import numpy as np
import torch

PLAN = ROOT.parent / 'paper_iclr2027/experiments/supplementary-experiment-plan.md'
EXECUTION_LOCK = PLAN.with_name('execution-lock.md')
CASES = {'grid2x3_l2': (2, 3), 'grid3x3_l2': (3, 3)}
METHODS = ('product_direct', 'product_continuation',
           'fourier_direct', 'fourier_continuation')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    # Include local transitive model/training dependencies, not just this wrapper.
    paths = [Path(__file__), Path(__file__).with_name('test_supplement_e3.py'), PLAN, EXECUTION_LOCK]
    paths += [ROOT / 'scripts' / name for name in (
        'run_vqe_fourier_controls.py', 'run_fspt8_fair.py',
        'run_nc_tfim_vqe.py', 'run_redq_continuation.py')]
    paths += sorted((ROOT / 'src').rglob('*.py'))
    return {str(p.relative_to(ROOT.parent)): sha256(p) for p in paths}


def make_spec(case, seed, device, development_smoke=False):
    if case not in CASES or device not in ('cpu', 'cuda'):
        raise ValueError('Require registered L2 grid and cpu/cuda')
    if seed not in ((45, 46, 47) if development_smoke else range(1001, 1006)):
        raise ValueError('Development uses 45..47; confirmation uses 1001..1005')
    rows, cols = CASES[case]
    return dict(version='supplement-e3-v1', case=case, rows=rows, cols=cols,
        layers=2, shared=True, J=-.1, h=-.5, hamiltonian_sign=1,
        scope='development_only_not_evidence' if development_smoke else 'minimum60_confirmation',
        seeds=dict(data=seed, initialization=seed+1000, measurement=seed+2000, starts=seed+3000),
        streams='separate NumPy PCG64 generators; initialization also seeds torch/batch shuffle',
        train=8 if development_smoke else 160, val=4 if development_smoke else 40,
        test=5 if development_smoke else 200, shots=64 if development_smoke else 10000,
        measurement='joint all-Z then all-X multinomial histograms; little-endian wire bits',
        product=dict(rank=32, q=4, epochs=2 if development_smoke else 3000,
                     min_epochs=0 if development_smoke else 500, patience=100,
                     batch=32, lr=.001, weight_decay=.0001, fit_dtype='float32'),
        fourier=dict(q_grid=list(Q_GRID), ridge_grid=list(RIDGE_GRID), dtype='float64'),
        vqe=dict(restarts=2 if development_smoke else 20,
                 steps=3 if development_smoke else 100, lr=.1, clip=float(np.pi)),
        continuation=dict(sigma=1., fraction=.6), optimization_device='cpu',
        optimization_dtype='float64', training_device=device,
        selection='measured validation moment MSE only; no validation refit',
        exact_policy='only after both checkpoints and all surrogate searches are frozen',
        deployment='argmin measured training energy; single candidate, no validation candidates',
        fixedref='all RZZ=0; RX=0; feasible E=-N/2', quantum_hardware_access=False)


class MeasurementRecorder:
    """Record the actual draws made by TFIMTask, preserving its sampler exactly."""
    def __init__(self, rng):
        self.rng = rng
        self.counts = []

    def multinomial(self, *args, **kwargs):
        counts = self.rng.multinomial(*args, **kwargs)
        self.counts.append(counts.copy())
        return counts


def generate_data(task, spec):
    nt, nv, ne = (spec[k] for k in ('train', 'val', 'test'))
    seeds = spec['seeds']
    x = np.random.default_rng(seeds['data']).uniform(
        -np.pi, np.pi, size=(nt+nv+ne, *task.control_shape))
    starts = np.random.default_rng(seeds['starts']).uniform(
        -np.pi, np.pi, size=(spec['vqe']['restarts'], *task.control_shape))
    recorder = MeasurementRecorder(np.random.default_rng(seeds['measurement']))
    measured, zs, xs = [], [], []
    for control in x[:nt+nv]:
        recorder.counts.clear()
        measured.append(task.sample_grouped_terms(control, spec['shots'], recorder))
        if len(recorder.counts) != 2:
            raise RuntimeError('Grouped sampler no longer makes two multinomial draws')
        z, xx = recorder.counts
        zs.append(z)
        xs.append(xx)
    return dict(X=x, measured_terms=np.asarray(measured), counts_z=np.asarray(zs),
                counts_x=np.asarray(xs), starts=starts,
                train_indices=np.arange(nt), validation_indices=np.arange(nt, nt+nv),
                test_indices=np.arange(nt+nv, nt+nv+ne))


def fixed_reference(task):
    return np.zeros(task.control_shape, dtype=np.float64)


def select_deployment(x, measured, train_count, coefficients):
    energies = measured[:train_count] @ coefficients
    index = int(np.argmin(energies))
    return index, x[index].copy(), float(energies[index])


def save_arrays(path, **arrays):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def freeze_stage(out, name, files, **details):
    value = dict(artifacts={f: sha256(out / f) for f in files}, **details)
    write_json(out / name, value)
    return value


def read_stage(out, name):
    path = out / name
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    for filename, digest in value['artifacts'].items():
        artifact = out / filename
        if not artifact.is_file() or sha256(artifact) != digest:
            raise ValueError(f'Changed or missing frozen artifact: {artifact}')
    return value


def energy_summary(energies, reference, n, initial=None):
    energies = np.asarray(energies)
    density = (energies-reference)/n
    result = dict(final_mean=float(energies.mean()), median=float(np.median(energies)),
        p90=float(np.quantile(energies, .9)), restarts=len(energies),
        restart_std=float(energies.std(ddof=1)) if len(energies) > 1 else None,
        mean_gap_to_fixedref=float((energies-reference).mean()),
        thresholds=[dict(epsilon_per_spin=eps,
                         within_or_below=float(np.mean(density <= eps)),
                         strictly_below=float(np.mean(density < -eps)))
                    for eps in (.001, .005, .01)],
        attainment_curve=dict(energy_gap_per_spin=np.sort(density).tolist(),
                              cumulative_fraction=(np.arange(1, len(energies)+1)/len(energies)).tolist()))
    if initial is not None:
        result['mean_final_minus_initial'] = float(np.mean(energies-initial))
    return result


def run_one(spec, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / '.running.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f'Run locked: {lock}; verify old process stopped before removing stale lock') from error
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(str(os.getpid()))
        return _run_locked(spec, out)
    finally:
        lock.unlink()


def _run_locked(spec, out):
    started = time.perf_counter()
    device = spec['training_device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    protocol = dict(spec=spec, sources=source_hashes(), environment=dict(
        python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
        device=torch.cuda.get_device_name() if device == 'cuda' else platform.machine(),
        qiskit=__import__('qiskit').__version__, threads=2))
    if not (out / 'protocol.json').exists() and any(p.name != '.running.lock' for p in out.iterdir()):
        raise ValueError('Nonempty output without a protocol; choose a fresh run directory')
    initialize_run(out, protocol)
    stages = {name: read_stage(out, name) for name in
              ('data.json', 'fit.json', 'search.json', 'completed.json')}
    if stages['completed.json']:
        return json.loads((out / 'results.json').read_text())
    task = TFIMTask(spec['rows'], spec['cols'], 2, True, 1)
    nt, nv = spec['train'], spec['val']
    coefficients = task.coefficients(spec['J'], spec['h'])
    if not stages['data.json']:
        print(f"{spec['case']} seed={spec['seeds']['data']} generating {spec['scope']} data", flush=True)
        tick = time.perf_counter()
        data = generate_data(task, spec)
        save_arrays(out / 'dataset.npz', **data)
        freeze_stage(out, 'data.json', ['dataset.npz'], seconds=time.perf_counter()-tick,
                     measured_controls=nt+nv, simulated_shots=(nt+nv)*2*spec['shots'])
    with np.load(out / 'dataset.npz') as arrays:
        x, measured, starts = (arrays[k].copy() for k in ('X', 'measured_terms', 'starts'))
    factory = lambda: ProductMomentREDQ(task.n, task.edges, 2, True,
                                       spec['product']['rank'], spec['product']['q'])
    fitting = stages['fit.json']
    if not fitting:
        print('Fitting Product and selecting full Fourier using measured labels', flush=True)
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        model, training = fit(factory, x[:nt], measured[:nt], x[nt:nt+nv], measured[nt:nt+nv],
            spec['seeds']['initialization'], device, out, 'product', spec['product']['epochs'],
            min_epochs=spec['product']['min_epochs'], patience=spec['product']['patience'])
        training.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                        fit_dtype='float32')
        tick = time.perf_counter()
        selected = select_model(x[:nt], measured[:nt], x[nt:nt+nv], measured[nt:nt+nv],
                                qs=spec['fourier']['q_grid'], ridges=spec['fourier']['ridge_grid'])
        fourier_seconds = time.perf_counter()-tick
        save_arrays(out / 'fourier.npz', weights=selected['weights'],
                    frequencies=selected['basis'].frequencies.numpy(),
                    active_indices=selected['basis'].active_indices,
                    q=selected['selected']['q'], ridge=selected['selected']['ridge'])
        index, control, measured_energy = select_deployment(x, measured, nt, coefficients)
        save_arrays(out / 'references.npz', fixedref=fixed_reference(task),
                    deployment=control, deployment_index=index, measured_energy=measured_energy,
                    masks=model.active.detach().cpu().numpy(), coefficients=coefficients)
        fitting = freeze_stage(out, 'fit.json',
            ['data.json', 'product.pt', 'product_history.json', 'fourier.npz', 'references.npz'],
            product=training, fourier=dict(selected=selected['selected'],
                candidates=selected['candidates'], seconds=fourier_seconds,
                candidate_solves=len(selected['candidates'])),
            deployment=dict(training_index=index, measured_energy=measured_energy),
            exact_labels_used=False)
        del model
    if not stages['search.json']:
        model = factory()
        model.load_state_dict(torch.load(out / 'product.pt', map_location='cpu', weights_only=True))
        model.eval().double()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with np.load(out / 'fourier.npz') as arrays:
            basis = FourierBasis(2, int(arrays['q']))
            weights = torch.tensor(arrays['weights'], dtype=torch.float64)
        coeff = torch.tensor(coefficients, dtype=torch.float64)
        predictors = dict(product=lambda z, s: model(z, smoothing=s)[0] @ coeff,
                          fourier=lambda z, s: basis(z, sigma=s) @ weights @ coeff)
        timings = {}
        print('Freezing all four paired surrogate searches before oracle evaluation', flush=True)
        for method in METHODS:
            family, mode = method.split('_')
            tick = time.perf_counter()
            trace = continuation(predictors[family], starts, spec['vqe'],
                spec['continuation']['sigma'] if mode == 'continuation' else 0.,
                spec['continuation']['fraction'])
            timings[method] = time.perf_counter()-tick
            save_arrays(out / f'{method}.npz', **trace)
        freeze_stage(out, 'search.json', ['fit.json'] + [f'{m}.npz' for m in METHODS],
                     optimization_seconds=timings, exact_queries=0)

    # This boundary is intentionally after both model selection and all searches.
    print('Post-freeze exact evaluation (research-only oracle)', flush=True)
    tick = time.perf_counter()
    exact_test = np.asarray([task.exact_terms(z) for z in x[nt+nv:]])
    model = factory()
    model.load_state_dict(torch.load(out / 'product.pt', map_location='cpu', weights_only=True))
    model.eval().double()
    with np.load(out / 'fourier.npz') as arrays:
        basis = FourierBasis(2, int(arrays['q']))
        weights = torch.tensor(arrays['weights'], dtype=torch.float64)
    prediction_seconds, predictions = {}, {}
    with torch.no_grad():
        controls = torch.tensor(x, dtype=torch.float64)
        for name, predictor in (('product', lambda z: model(z)[0]),
                                ('fourier', lambda z: basis(z) @ weights)):
            pred_start = time.perf_counter()
            predictions[name] = predictor(controls).numpy()
            prediction_seconds[name] = time.perf_counter()-pred_start
    save_arrays(out / 'predictions.npz', exact_test=exact_test, **predictions)
    with np.load(out / 'references.npz') as refs:
        reference = task.energy(refs['fixedref'], spec['J'], spec['h'])
        deployment_energy = task.energy(refs['deployment'], spec['J'], spec['h'])
    np.testing.assert_allclose(reference, -task.n/2, atol=1e-10, rtol=0)
    endpoints = dict(starts=starts, fixedref=np.array(reference),
                     deployment=np.array(deployment_energy))
    method_stats = {}
    oracle_files = []
    for method in METHODS:
        with np.load(out / f'{method}.npz') as trace:
            exact = np.array([[task.energy(z, spec['J'], spec['h']) for z in step]
                              for step in trace['parameters']])
        filename = f'{method}_exact.npz'
        save_arrays(out / filename, exact_energy=exact, exact_steps=np.arange(len(exact)))
        oracle_files.append(filename)
        endpoints[method] = exact[-1]
        endpoints['exact_initial'] = exact[0]
        method_stats[method] = energy_summary(exact[-1], reference, task.n, exact[0])
    save_arrays(out / 'endpoints.npz', **endpoints)
    prediction_stats = {}
    for name, pred in predictions.items():
        residual = pred[nt+nv:]-exact_test
        prediction_stats[name] = dict(test_moment_mse=float(np.mean(residual**2)),
            test_energy_mae=float(np.mean(np.abs(residual @ coefficients))),
            test_per_observable_mse=np.mean(residual**2, axis=0).tolist(),
            measured_train_mse=float(np.mean((pred[:nt]-measured[:nt])**2)),
            measured_validation_mse=float(np.mean((pred[nt:nt+nv]-measured[nt:])**2)),
            prediction_seconds=prediction_seconds[name], prediction_controls=len(x))
    result = dict(completed=True, scope=spec['scope'], case=spec['case'], seeds=spec['seeds'],
        seed=spec['seeds']['data'], output=str(out.resolve()), protocol=spec,
        protocol_sha256=sha256(out / 'protocol.json'),
        dataset_sha256=sha256(out / 'dataset.npz'), training=fitting,
        prediction=prediction_stats, methods=method_stats,
        paired_fourierC_minus_productC=float(np.mean(endpoints['fourier_continuation']-
                                                   endpoints['product_continuation'])),
        fixedref=dict(exact_energy=reference, **energy_summary([reference], reference, task.n)),
        deployment=dict(**fitting['deployment'], exact_energy=deployment_energy,
            scope='single-candidate deployment reference, not a 20-restart average',
            **energy_summary([deployment_energy], reference, task.n)),
        cost=dict(simulated_training_validation_shots=(nt+nv)*2*spec['shots'],
            shared_across_models=True, hardware_shots=0, final_measurement_validation_shots=0,
            final_validation_status='not performed; all reported terminal values are exact research evaluation',
            prospective_shots_per_deployed_candidate=2*spec['shots'],
            prospective_shots_all_method_endpoints=4*len(starts)*2*spec['shots'],
            exact_test_controls=spec['test'],
            exact_trajectory_controls=4*(spec['vqe']['steps']+1)*len(starts),
            exact_reference_controls=2,
            surrogate_objective_control_evaluations=4*(spec['vqe']['steps']+1)*len(starts),
            postfit_evaluation_seconds=time.perf_counter()-tick),
        resources=dict(invocation_seconds=time.perf_counter()-started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                (1 if sys.platform == 'darwin' else 1024),
            peak_rss_scope='whole process lifetime including prior calls; not model-specific'),
        statistical_unit='one independent dataset; aggregate seeds equally; restarts are paired within seed')
    write_json(out / 'results.json', result)
    freeze_stage(out, 'completed.json', ['protocol.json', 'data.json', 'fit.json', 'search.json',
        'predictions.npz', 'endpoints.npz', 'results.json'] + oracle_files)
    return result


def run(case, seed, out, device='cpu', development_smoke=False):
    """Run one case/dataset pair; return its completed results, including output path."""
    spec = make_spec(case, seed, device, development_smoke)
    root = Path(out) / ('development_smoke_not_evidence' if development_smoke else 'confirmation')
    return run_one(spec, root / f'{case}_seed{seed}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, choices=tuple(CASES))
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path, help='Root; case/seed subdirectory is added')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--development-smoke', action='store_true')
    args = parser.parse_args()
    try:
        make_spec(args.case, args.seed, args.device, args.development_smoke)
    except ValueError as error:
        parser.error(str(error))
    result = run(args.case, args.seed, args.out, args.device, args.development_smoke)
    print(json.dumps(dict(completed=result['completed'], scope=result['scope'],
                         case=result['case'], seed=result['seed'], output=result['output'])))


if __name__ == '__main__':
    main()
