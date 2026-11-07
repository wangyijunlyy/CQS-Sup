"""Matched chain Fourier+C check using frozen E1 models, without refitting.

This is a post-review extension of the existing five E1 datasets, not a new
holdout study. It does not replace the original seeds45--47 VQE benchmark.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import sys
import json
import time
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import t

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
ROOT = WORK / 'world_model_surrogate'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT), str(HERE.parent)]
from run_redq_continuation import continuation
from run_evidence import ChainExact
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ


class SavedCompleteFourier(torch.nn.Module):
    def __init__(self, folder):
        super().__init__()
        self.groups = []
        for path in sorted(folder.glob('kernel_group*.npz')):
            with np.load(path) as a:
                active = np.flatnonzero(a['mask'])
                self.groups.append(dict(
                    active=active,
                    train=torch.tensor(a['train_x'][:, active]),
                    outputs=a['outputs'].tolist(),
                    mean=torch.tensor(a['mean']),
                    overall=float(a['overall']), scale=float(a['scale']),
                    target_mean=torch.tensor(a['target_mean']),
                    weights=torch.tensor(a['weights'][0])))
        assert sum(len(g['outputs']) for g in self.groups) == 11

    def forward(self, controls, sigma=0.):
        flat = controls.flatten(1)
        result = flat.new_zeros((len(flat), 11))
        damping = np.exp(-.5 * sigma**2)
        for g in self.groups:
            x = flat[:, g['active']]
            kernel = (1 + 2*damping*(x[:, None, :] - g['train'][None]).cos()).prod(-1)
            centered = (kernel-kernel.mean(1, keepdim=True)-g['mean'][None]
                        +g['overall']) / g['scale']
            result[:, g['outputs']] = centered @ g['weights'] + g['target_mean']
        return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    torch.set_num_threads(2)
    out = HERE / 'chain_evidence'
    out.mkdir(exist_ok=True)
    task = TFIMTask(1, 6, 2, False, -1)
    settings = dict(steps=100, lr=.1, clip=float(np.pi))
    save(out/'protocol.json', dict(
        datasets=list(range(1001, 1006)), J=-.1, h=-.5, masks='same E1 local masks',
        fits='frozen E1 product and complete Fourier; no refitting or selection',
        starts='20 uniform controls; RNG seed=data seed+3000',
        settings=settings, sigma=1., fraction=.6,
        score='mean exact energy of all 20 starts, then equal dataset weights',
        scope='post-review search on existing E1 datasets; distinct from original VQE seeds'))
    coeff = torch.tensor(task.coefficients(-.1, -.5))
    rows = []
    for seed in range(1001, 1006):
        base = ROOT / 'results_supplementary/e1/confirmation' / f'seed{seed}'
        f = SavedCompleteFourier(base/'F-local')
        with np.load(base/'F-local/test_predictions.npz') as a:
            check_x = torch.tensor(a['x'][:8])
            saved = a['prediction'][:8]
        replay = float(np.max(np.abs(f(check_x).detach().numpy()-saved)))
        assert replay < 1e-8, replay
        p = ProductMomentREDQ(6, task.edges, 2, False, 32, 1).double().eval()
        p.load_state_dict(torch.load(base/'P-local/product.pt', map_location='cpu', weights_only=True))
        for param in p.parameters():
            param.requires_grad_(False)
        # Check the smoothed kernel's derivative at a nonzero smoothing scale.
        z = check_x[:1].clone().requires_grad_(True)
        val = (f(z, .6) @ coeff).sum()
        grad = torch.autograd.grad(val, z)[0]
        eps = 1e-5
        zp, zm = z.detach().clone(), z.detach().clone()
        zp[0, 1, 0] += eps
        zm[0, 1, 0] -= eps
        fd = float(((f(zp, .6)-f(zm, .6)) @ coeff).item()/(2*eps))
        derivative_error = abs(fd-float(grad[0, 1, 0]))
        assert derivative_error < 1e-7, derivative_error
        starts = np.random.default_rng(seed+3000).uniform(-np.pi, np.pi, (20, 2, 11))
        np.save(out/f'seed{seed}_starts.npy', starts)
        for name, predictor in (
            ('Fourier_local+C', lambda x, s: f(x, s) @ coeff),
            ('CQS_P_local+C', lambda x, s: p(x, smoothing=s)[0] @ coeff)):
            start_time = time.perf_counter()
            trace = continuation(predictor, starts, settings, 1., .6)
            # All endpoints are retained; exact values never select candidates.
            truth = np.array([task.energy(x, -.1, -.5) for x in trace['parameters'][-1]])
            np.savez_compressed(out/f'seed{seed}_{name}.npz', **trace, exact_final=truth)
            row = dict(seed=seed, method=name, mean_energy=float(truth.mean()),
                       within_reference_fraction=float(np.mean(truth <= -3+.006)),
                       seconds=time.perf_counter()-start_time,
                       unsmoothed_replay_max_error=replay,
                       derivative_error=derivative_error)
            rows.append(row)
            save(out/'rows.json', rows)
            print(json.dumps(row), flush=True)
    summary = {}
    for name in ('Fourier_local+C', 'CQS_P_local+C'):
        vals = [r['mean_energy'] for r in rows if r['method'] == name]
        summary[name] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)),
                             dataset_energies=vals)
    paired = np.array(summary['Fourier_local+C']['dataset_energies'])-np.array(summary['CQS_P_local+C']['dataset_energies'])
    summary['Fourier_minus_product'] = dict(mean=float(paired.mean()), sd=float(paired.std(ddof=1)),
        values=paired.tolist(), fourier_lower_datasets=int((paired < 0).sum()),
        ci95=(paired.mean()+np.array([-1, 1])*t.ppf(.975, 4)*paired.std(ddof=1)/np.sqrt(5)).tolist())
    # Exact spectrum bounds certify the physical lower bound, not attainability.
    oracle = ChainExact(task)
    spectrum_low, spectrum_high = task.spectrum_bounds(-.1, -.5)
    reference = np.zeros((2, 11))
    for e, edge in enumerate(task.edges):
        if edge in ((0, 1), (2, 3), (4, 5)):
            reference[0, 6+e] = np.pi
    reference_energy = task.energy(reference, -.1, -.5)
    assert abs(reference_energy+3) < 1e-12
    def objective(flat):
        x = torch.tensor(flat.reshape(1, 2, 11), requires_grad=True)
        y = oracle(x, -.1, -1).sum()
        gradient = torch.autograd.grad(y, x)[0]
        return float(y.detach()), gradient.detach().numpy().ravel()
    controls = np.random.default_rng(20261107).uniform(-np.pi, np.pi, (32, 2, 11))
    controls[0] = reference
    results = [minimize(objective, x.ravel(), method='BFGS', jac=True,
                        options=dict(maxiter=1000, gtol=1e-9)) for x in controls]
    best = min(results, key=lambda r: r.fun)
    exact_check = task.energy(best.x.reshape(2, 11), -.1, -.5)
    assert abs(exact_check-best.fun) < 1e-10
    np.save(out/'best_found_ansatz_controls.npy', best.x.reshape(2, 11))
    summary['reference'] = dict(feasible_energy=reference_energy,
        spectral_lower=spectrum_low, spectral_upper=spectrum_high,
        best_found_ansatz_energy=float(best.fun),
        ansatz_bracket_width=float(best.fun)-spectrum_low,
        global_ansatz_certificate=False, oracle_restarts=32,
        successful_bfgs=int(sum(r.success for r in results)),
        restart_energies=[float(r.fun) for r in results], simulator_replay_error=abs(exact_check-best.fun))
    save(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
