"""Same-CPU full moment prediction and scalar-input gradient cost."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]/'world_model_surrogate'
sys.path.insert(0, str(ROOT))
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ
from complete_fourier import CompleteFourier, SharedFeatureFourier
from run_fourier_chain import write


def main():
    torch.set_num_threads(2)
    task = TFIMTask(1, 6, 2, False, -1)
    coefficient = torch.tensor(task.coefficients(-.5, -.5))
    rows = []
    for seed in (45, 46, 47):
        f = CompleteFourier(HERE/'chain'/f'seed{seed}'/'global')
        shared = SharedFeatureFourier(HERE/'chain'/f'seed{seed}'/'global')
        p = ProductMomentREDQ(6, task.edges, 2, False, 32, 1).double().eval()
        p.load_state_dict(torch.load(ROOT/f'results_product_redq_independent/chain6_l2_seed{seed}/periodic_redq.pt',
                                    map_location='cpu', weights_only=True))
        for parameter in p.parameters():
            parameter.requires_grad_(False)
        with np.load(ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz') as a:
            x = a['X'][2400:].copy()
        z = torch.tensor(x[:8], requires_grad=True)
        a, b = f(z, .6), shared(z, .6)
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
        ga, gb = torch.autograd.grad(a.square().sum(), z)[0], torch.autograd.grad(b.square().sum(), z)[0]
        torch.testing.assert_close(ga, gb, atol=1e-9, rtol=1e-9)
        def prediction(model, is_product):
            with torch.inference_mode():
                out = model(torch.tensor(x))
                return (out[0] if is_product else out).numpy()
        def gradient(model, is_product):
            z = torch.tensor(x[:20], requires_grad=True)
            out = model(z)
            moments = out[0] if is_product else out
            grad = torch.autograd.grad((moments@coefficient).sum(), z)[0]
            return grad.numpy()
        for operation, function in [('moments400', prediction), ('energy_gradient20', gradient)]:
            functions = dict(product=lambda: function(p, True), complete_fourier=lambda: function(f, False),
                             fourier_shared_features=lambda: function(shared, False))
            for call in functions.values():
                for _ in range(2):
                    call()
            times = {name: [] for name in functions}
            for repeat in range(10):
                order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
                for name in order:
                    start = time.perf_counter()
                    functions[name]()
                    times[name].append((time.perf_counter()-start)*1000)
            row = dict(seed=seed, operation=operation,
                milliseconds={name: float(np.median(v)) for name, v in times.items()},
                raw_milliseconds=times)
            rows.append(row)
            print(row['seed'], operation, row['milliseconds'], flush=True)
    summary = {}
    for operation in ('moments400', 'energy_gradient20'):
        group = [r for r in rows if r['operation'] == operation]
        summary[operation] = {}
        for method in ('product', 'complete_fourier', 'fourier_shared_features'):
            values = [r['milliseconds'][method] for r in group]
            summary[operation][method] = dict(mean=float(np.mean(values)),
                sd=float(np.std(values, ddof=1)), dataset_medians=values)
        fastest = min(('complete_fourier', 'fourier_shared_features'),
                      key=lambda k: summary[operation][k]['mean'])
        summary[operation]['faster_fourier_evaluator'] = fastest
        summary[operation]['ratio_of_mean_medians'] = (
            summary[operation][fastest]['mean']/summary[operation]['product']['mean'])
    write(HERE/'chain'/'timing.json', dict(rows=rows, summary=summary,
        threads=2, dtype='float64', device='cpu', platform=platform.platform(),
        warmup=2, repetitions=10, timing_boundary='numpy controls through numpy moments or input gradients',
        scope='stored rank32 product versus exact complete-kernel implementation, excluding fitting and acquisition'))


if __name__ == '__main__':
    main()
