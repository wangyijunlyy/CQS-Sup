"""Paired landscape continuation; same schedule and steps for both surrogates."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_nc_tfim_vqe import initialize_run, write_json
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.nc_baselines import TrigFeatureMap
from src.wmps.models.periodic_redq import ProductMomentREDQ
from src.wmps.models.fourier_continuation import smoothed_nc_features


def continuation(predict, starts, settings, sigma, fraction):
    x = torch.tensor(starts, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([x], lr=settings['lr'], foreach=False)
    parameters, values, scales = [], [], []
    for step in range(settings['steps'] + 1):
        scale = sigma * max(0., 1 - step / (settings['steps'] * fraction))
        opt.zero_grad()
        pred = predict(x, scale)
        if not torch.isfinite(pred).all():
            raise FloatingPointError('Nonfinite objective')
        parameters.append(x.detach().numpy().copy())
        values.append(pred.detach().numpy())
        scales.append(scale)
        if step == settings['steps']:
            break
        pred.sum().backward()
        if not torch.isfinite(x.grad).all():
            raise FloatingPointError('Nonfinite gradient')
        opt.step()
        with torch.no_grad():
            x.clamp_(-settings['clip'], settings['clip'])
    return dict(parameters=np.array(parameters), predictions=np.array(values), sigma=np.array(scales))


def run(args):
    torch.set_num_threads(2)
    source = json.loads((args.model / 'results.json').read_text())
    p = source['protocol']
    if p['method'] != 'product':
        raise ValueError('Continuation requires the multiplicative Fourier model')
    case, seed = p['case'], p['seed']
    task = TFIMTask(case['rows'], case['cols'], case['layers'], case['shared'], 1 if case['shared'] else -1)
    audit = args.audit_root / f"{case['id']}_seed{seed}"
    out = args.out / f"{case['id']}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    initialize_run(out, dict(model_protocol=p, sigma=args.sigma, fraction=args.fraction,
                             matched_baseline=True, extra_measurements=0, exact_selection=False))
    if (out / 'results.json').exists():
        print(f'Already complete {out}', flush=True)
        return
    model = ProductMomentREDQ(task.n, task.edges, task.layers, task.shared, p['width'], p['harmonics'])
    model.load_state_dict(torch.load(args.model / 'periodic_redq.pt', weights_only=True))
    model.eval().double()
    for param in model.parameters():
        param.requires_grad_(False)
    fmap = TrigFeatureMap(np.load(audit / 'frequencies.npy'), task.gate_to_control)
    weights = torch.tensor(np.load(audit / 'baseline_weights.npy'), dtype=torch.float64)
    starts = np.load(audit / 'initial_parameters.npy')
    records = []
    for index, j in enumerate(case['couplings']):
        coeff = torch.tensor(task.coefficients(j, -.5), dtype=torch.float64)
        methods = dict(redq=lambda z, s: model(z, smoothing=s)[0] @ coeff,
                       nc=lambda z, s: smoothed_nc_features(fmap, z.flatten(1), s) @ weights[:, index])
        record = dict(case=case['id'], seed=seed, J=j, methods={})
        for name, predict in methods.items():
            trace = continuation(predict, starts, p['vqe'], args.sigma, args.fraction)
            final = np.array([task.energy(z, j, -.5) for z in trace['parameters'][-1]])
            np.savez_compressed(out / f'{name}_J{j:+.1f}.npz', **trace, exact_final=final)
            record['methods'][name] = dict(final_mean=float(final.mean()), restart_std=float(final.std(ddof=1)))
        original = next(r for r in source['records'] if r['J'] == j)
        record['original_baseline'] = original['baseline_final_mean']
        record['unsmoothed_product'] = original['final_mean']
        record['paired_improvement'] = record['methods']['nc']['final_mean'] - record['methods']['redq']['final_mean']
        records.append(record)
        print(json.dumps(record), flush=True)
    write_json(out / 'results.json', dict(records=records, sigma=args.sigma, fraction=args.fraction,
                                          source=str(args.model), training_protocol=p))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=ROOT / 'results_redq_continuation_v1')
    ap.add_argument('--audit-root', type=Path, default=ROOT / 'results_nc_tfim_vqe/code')
    ap.add_argument('--sigma', type=float, default=1.)
    ap.add_argument('--fraction', type=float, default=.6)
    args = ap.parse_args()
    if args.sigma < 0 or not 0 < args.fraction <= 1:
        ap.error('Require sigma >= 0 and 0 < fraction <= 1')
    run(args)
