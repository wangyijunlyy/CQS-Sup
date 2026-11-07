"""Same-data periodic RED-Q development audit; exact values are evaluation-only."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_fspt8_fair import fit
from run_nc_tfim_vqe import initialize_run, optimize, write_json
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import PeriodicMomentREDQ, CausalMomentREDQ, ProductMomentREDQ
from src.wmps.train import metrics


def run(args):
    torch.set_num_threads(2)
    config = json.loads(args.config.read_text())
    case = next(c for c in config['cases'] if c['id'] == args.case)
    task = TFIMTask(case['rows'], case['cols'], case['layers'], case['shared'],
                    1 if case['shared'] else -1)
    data_path = args.dataset_root / f'{args.case}_seed{args.seed}' / 'dataset.npz'
    audit = args.audit_root / f'{args.case}_seed{args.seed}'
    out = args.out / f'{args.case}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(method=args.architecture, case=case, seed=args.seed,
                    width=args.width, harmonics=args.harmonics, epochs=args.epochs,
                    dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    training_labels='stored measured_terms only', checkpoint='measured validation MSE',
                    extra_measurements=0, exact_values='post-fit evaluation only',
                    vqe=config['vqe'], scope='development audit; seeds used in earlier development')
    if args.min_epochs or args.patience != 30:
        protocol.update(min_epochs=args.min_epochs, patience=args.patience)
    initialize_run(out, protocol)
    if (out / 'results.json').exists():
        print(f'Already complete {out}', flush=True)
        return
    with np.load(data_path) as a:
        x, measured = a['X'], a['measured_terms']
    nt, nv = case['train'], case['val']
    factory = (lambda: CausalMomentREDQ(task.n, task.edges, task.layers, task.shared, args.width, args.harmonics)) if args.architecture == 'causal' else (lambda: PeriodicMomentREDQ(task.n, task.edges, task.shared, args.width, args.harmonics))
    if args.architecture == 'product':
        factory = lambda: ProductMomentREDQ(task.n, task.edges, task.layers, task.shared, args.width, args.harmonics)
    model, detail = fit(factory,
                        x[:nt], measured[:nt], x[nt:nt+nv], measured[nt:nt+nv],
                        args.seed, 'cpu', out, 'periodic_redq', args.epochs,
                        min_epochs=args.min_epochs, patience=args.patience)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    starts = np.load(audit / 'initial_parameters.npy')
    # Exact test targets are loaded only after checkpoint selection is finished.
    with np.load(data_path) as a:
        exact_test = a['exact_terms'][nt+nv:]
    with torch.no_grad():
        pred_terms = model(torch.tensor(x[nt+nv:], dtype=torch.float32))[0].numpy()
    np.save(out / 'pred_terms.npy', pred_terms)
    records = []
    for j in case['couplings']:
        coeff = torch.tensor(task.coefficients(j, -.5), dtype=torch.float64)
        predict = lambda z: model(z.float())[0].double() @ coeff
        trace = optimize(predict, starts, config['vqe'], lambda z: task.energy(z, j, -.5))
        np.savez_compressed(out / f'vqe_J{j:+.1f}.npz', **trace)
        baseline_name = 'h_qs' if task.shared else 'h_cs'
        old = json.loads((audit / f'J{j:+.1f}' / 'results.json').read_text())
        record = dict(J=j, final_mean=float(trace['exact_energy'][-1].mean()),
                      restart_std=float(trace['exact_energy'][-1].std(ddof=1)),
                      baseline_final_mean=old['vqe'][baseline_name]['final_mean'],
                      original_redq_final_mean=old['vqe']['redq']['final_mean'],
                      prediction=metrics(exact_test @ coeff.numpy(), pred_terms @ coeff.numpy()))
        record['improvement'] = record['baseline_final_mean'] - record['final_mean']
        records.append(record)
        print(json.dumps(dict(case=args.case, seed=args.seed, **record)), flush=True)
    write_json(out / 'results.json', dict(protocol=protocol, training=detail, records=records))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/nc_tfim_vqe.json')
    ap.add_argument('--case', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--harmonics', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--min-epochs', type=int, default=0)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--architecture', choices=('periodic', 'causal', 'product'), default='periodic')
    ap.add_argument('--out', type=Path, default=ROOT / 'results_periodic_redq_v1')
    ap.add_argument('--dataset-root', type=Path, default=ROOT / 'results_nc_tfim_vqe/datasets')
    ap.add_argument('--audit-root', type=Path, default=ROOT / 'results_nc_tfim_vqe/code')
    run(ap.parse_args())
