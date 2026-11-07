"""Paired N6 L2 product-mask intervention on cached measured moments only.

All three arms retain the dense ProductMomentREDQ allocation and execution.
This runner performs prediction evaluation only, with no VQE or new labels.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import mean, stdev
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_fspt8_fair import fit, seed_all
from run_nc_tfim_vqe import initialize_run, write_json
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ, causal_parameter_masks

SEEDS = (45, 46, 47)
ARMS = ('actual', 'full', 'random')
TASK = TFIMTask(1, 6, 2, False, -1)
COUPLINGS = (-0.5, -0.3, -0.1, 0.1, 0.3, 0.5)


def make_masks(seed):
    actual = causal_parameter_masks(6, TASK.edges, 2, False)
    full = np.ones_like(actual)
    full[:, 0, :6] = 0
    random = np.zeros_like(actual)
    rng = np.random.default_rng(100000 + seed)
    for observable in range(11):
        # Conditional uniform draws preserve each block count and change each row.
        while True:
            random[observable] = 0
            for layer in (0, 1):
                for start, stop in ((0, 6), (6, 11)):
                    count = int(actual[observable, layer, start:stop].sum())
                    chosen = rng.choice(np.arange(start, stop), size=count, replace=False)
                    random[observable, layer, chosen] = 1
            if not np.array_equal(random[observable], actual[observable]):
                break
    return dict(actual=actual, full=full, random=random)


def initial_state(seed):
    seed_all(seed)
    model = ProductMomentREDQ(6, TASK.edges, 2, False, width=32, harmonics=1)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def build_model(initial, mask):
    model = ProductMomentREDQ(6, TASK.edges, 2, False, width=32, harmonics=1)
    model.load_state_dict(initial)
    model.active.copy_(torch.tensor(mask.reshape(11, 22)))
    return model


def parameter_hash(state):
    digest = hashlib.sha256()
    for key in ('factors', 'readout', 'bias'):
        digest.update(key.encode('ascii'))
        digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def score_terms(target, prediction):
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 11:
        raise ValueError('Expected matching (samples, 11) moment arrays')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite evaluation arrays')

    def error(truth, pred):
        residual = np.asarray(pred, dtype=np.float64) - truth
        mse = float(np.mean(residual ** 2))
        return dict(mse=mse, rmse=mse ** .5, mae=float(np.mean(np.abs(residual))))

    return dict(
        moments=error(target, prediction),
        x=error(target[:, :6], prediction[:, :6]),
        zz=error(target[:, 6:], prediction[:, 6:]),
        per_observable=[error(target[:, o], prediction[:, o]) for o in range(11)],
        energy={f'{j:+.1f}': error(target @ TASK.coefficients(j, -.5),
                                 prediction @ TASK.coefficients(j, -.5)) for j in COUPLINGS})


def run_seed(seed, dataset_root, out):
    name = f'chain6_l2_seed{seed}'
    data_path = dataset_root / name / 'dataset.npz'
    with np.load(data_path) as data:
        x, measured = data['X'], data['measured_terms']
    if x.shape != (2800, 2, 11) or measured.shape != (2400, 11):
        raise ValueError('Unexpected cached dataset shape')
    initial = initial_state(seed)
    masks = make_masks(seed)
    for arm, mask in masks.items():
        folder = out / f'{name}_{arm}'
        folder.mkdir(parents=True, exist_ok=True)
        protocol = dict(
            case='chain6_l2', seed=seed, arm=arm, rank=32, harmonics=1,
            n=6, layers=2, shared=False, edges=TASK.edges,
            epochs=300, patience=30, min_epochs=0, batch_size=32,
            optimizer='Adam', lr=.001, weight_decay=.0001,
            dataset_path=str(data_path.resolve()),
            dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
            train=[0, 2000], validation=[2000, 2400], test=[2400, 2800],
            training_labels='stored measured_terms only',
            checkpoint='minimum measured validation moment MSE',
            exact_values='held-out post-fit evaluation only', extra_measurements=0,
            vqe=False, smoothing=0., initial_rx_excluded_in_all_arms=True,
            random_mask_seed=100000 + seed,
            random_rule='uniform within observable/layer/gate-family blocks; reject unchanged rows',
            initial_parameter_sha256=parameter_hash(initial),
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            runtime=dict(device='cpu', threads=2, python=platform.python_version(),
                         torch=torch.__version__, numpy=np.__version__, platform=platform.platform()),
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), ROOT / 'scripts/run_fspt8_fair.py',
                                     ROOT / 'src/wmps/models/periodic_redq.py')},
            scope='revision intervention on existing independent-seed datasets, not newly unseen seeds')
        initialize_run(folder, protocol)
        if (folder / 'results.json').exists():
            print(f'Already complete {folder.name}', flush=True)
            continue
        torch.save(initial, folder / 'common_initial_state.pt')
        np.save(folder / 'mask.npy', mask)
        np.savez_compressed(folder / 'split.npz', train=np.arange(2000),
                            val=np.arange(2000, 2400), test=np.arange(2400, 2800))
        print(f'START {folder.name} active_entries={int(mask.sum())}', flush=True)
        started = time.perf_counter()
        model, training = fit(lambda: build_model(initial, mask),
                              x[:2000], measured[:2000], x[2000:2400], measured[2000:2400],
                              seed, 'cpu', folder, 'product', 300, min_epochs=0, patience=30)
        model.eval()
        evaluation_started = time.perf_counter()
        with torch.no_grad():
            prediction = model(torch.tensor(x[2400:], dtype=torch.float32))[0].numpy()
        inference_seconds = time.perf_counter() - evaluation_started
        with np.load(data_path) as data:
            target = data['exact_terms'][2400:]
        scores = score_terms(target, prediction)
        np.savez_compressed(folder / 'test_predictions.npz', prediction=prediction, exact=target)
        active_factors = int(mask.sum())
        allocation = dict(allocated_parameters=sum(p.numel() for p in model.parameters()),
                          parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
                          active_factor_entries=active_factors,
                          effective_parameter_slots=active_factors * 32 * 3 + 11 * 32 + 11,
                          per_observable_active_dims=mask.sum((1, 2)).astype(int).tolist(),
                          factor_tensor_shape=list(model.factors.shape),
                          dense_observable_coordinate_entries=11 * 22,
                          sparse_execution=False,
                          note='Effective slots are not identifiable degrees of freedom; dense allocation/execution unchanged')
        result = dict(protocol=protocol, training=training, metrics=scores, allocation=allocation,
                      inference_seconds_400_controls=inference_seconds,
                      total_seconds=time.perf_counter() - started, completed=True)
        write_json(folder / 'results.json', result)
        print(json.dumps(dict(run=folder.name, epochs=training['epochs_run'],
                              seconds=training['training_seconds'], moments=scores['moments'],
                              energy=scores['energy']['-0.1'])), flush=True)


def aggregate(out):
    runs = []
    for seed in SEEDS:
        for arm in ARMS:
            path = out / f'chain6_l2_seed{seed}_{arm}/results.json'
            if not path.exists():
                raise ValueError(f'Missing full fit: {path}')
            runs.append(json.loads(path.read_text()))
    groups = {}
    for arm in ARMS:
        selected = [r for r in runs if r['protocol']['arm'] == arm]
        values = dict(moment_mse=[r['metrics']['moments']['mse'] for r in selected],
                      moment_mae=[r['metrics']['moments']['mae'] for r in selected],
                      x_mse=[r['metrics']['x']['mse'] for r in selected],
                      zz_mse=[r['metrics']['zz']['mse'] for r in selected],
                      energy_mae=[r['metrics']['energy']['-0.1']['mae'] for r in selected])
        groups[arm] = {key: dict(mean=mean(v), std=stdev(v)) for key, v in values.items()}
    paired = {}
    for arm in ('full', 'random'):
        differences = {metric: [] for metric in ('moment_mse', 'energy_mae')}
        for seed in SEEDS:
            a = next(r for r in runs if r['protocol']['seed'] == seed and r['protocol']['arm'] == 'actual')
            b = next(r for r in runs if r['protocol']['seed'] == seed and r['protocol']['arm'] == arm)
            assert a['protocol']['initial_parameter_sha256'] == b['protocol']['initial_parameter_sha256']
            assert a['protocol']['dataset_sha256'] == b['protocol']['dataset_sha256']
            differences['moment_mse'].append(b['metrics']['moments']['mse'] - a['metrics']['moments']['mse'])
            differences['energy_mae'].append(b['metrics']['energy']['-0.1']['mae'] - a['metrics']['energy']['-0.1']['mae'])
        paired[arm + '_minus_actual'] = {key: dict(values=v, mean=mean(v), std=stdev(v))
                                          for key, v in differences.items()}
    summary = dict(completed=9, seeds=list(SEEDS), groups=groups, paired=paired, runs=runs,
                   training_seconds_total=sum(r['training']['training_seconds'] for r in runs),
                   scope='Full scheduled fits with validation early stopping; no smoke or VQE results')
    write_json(out / 'summary.json', summary)
    lines = ['# N6 L2 measured-moment mask intervention', '',
             '| Arm | Moment MSE | Moment MAE | Energy MAE (J=-0.1) |',
             '|---|---:|---:|---:|']
    for arm, group in groups.items():
        lines.append('| ' + arm + ' | ' + ' | '.join(
            f"{group[key]['mean']:.6f} +/- {group[key]['std']:.6f}"
            for key in ('moment_mse', 'moment_mae', 'energy_mae')) + ' |')
    lines += ['', 'Three paired seeds 45/46/47; sample standard deviations across seeds.',
              'Rank32 Q1, cap300/patience30/min0, Adam lr0.001/wd0.0001, batch32.',
              'All arms exclude initial RX and use identical initial learnable tensors.',
              'Random masks preserve per-observable, layer, and gate-family counts.',
              'Dense allocation/execution is unchanged; this is not a sparse-speed comparison.',
              f"Total recorded fit time: {summary['training_seconds_total']:.2f} seconds."]
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'results_nc_tfim_vqe_independent/datasets')
    parser.add_argument('--out', type=Path, default=ROOT / 'results_review_masks')
    parser.add_argument('--aggregate-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not args.aggregate_only:
        for seed in SEEDS:
            run_seed(seed, args.dataset_root, args.out)
    aggregate(args.out)
