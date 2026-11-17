"""Offline reviewer diagnostics. Historical inputs are read only; output is local.

Run with workspace .venv_ibm/bin/python. No QPU or network access.
Exact-energy continuation is an oracle diagnostic, not a learned baseline.
"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '2'
import numpy as np
import torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

WORK = Path(__file__).resolve().parents[2]
ROOT = WORK / 'world_model_surrogate'
OUT = Path(__file__).resolve().parent / 'evidence'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT)]
from src.wmps.data.nc_tfim import TFIMTask
import run_hardware_full_prefix_control as hw
from run_redq_continuation import continuation


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(name, rows):
    if rows:
        with (OUT / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def hardware():
    # Search expansion is fixed before evaluating any new test predictions.
    # It keeps original controls, feature map, output labels, and early validation.
    alphas = np.array([.001, .01, .1, 1., 10., 100., 300., 1000., 3000.,
                       10000., 30000., 100000., 1e6, 1e8, 1e10])
    frozen, rows = [], []
    with threadpool_limits(limits=1):
        for name in hw.DATASETS:
            X, y, split, info = hw.load_dataset(name)
            pending = {}
            for mode in ('full', 'current'):
                a = hw.trig_features(hw.control_rows(X, split['train'], range(4), mode))
                b = hw.trig_features(hw.control_rows(X, split['val'], range(4), mode))
                yt = y[split['train'], :4].reshape(-1, y.shape[-1])
                yv = y[split['val'], :4].reshape(-1, y.shape[-1])
                models = [Ridge(alpha=float(alpha)).fit(a, yt) for alpha in alphas]
                vals = [float(np.mean((m.predict(b)-yv)**2)) for m in models]
                best = int(np.argmin(vals))
                frozen.append(dict(dataset=name, mode=mode, alpha=float(alphas[best]),
                                   validation_mse=vals, candidates=alphas.tolist()))
                pending[mode] = (models, best)
            save('hardware_selection_frozen.json', frozen)
            for mode, (models, best) in pending.items():
                features = hw.trig_features(hw.control_rows(X, range(len(X)), range(10), mode))
                for label, idx in [('fixed_alpha1', 3), ('old_tuned_to100', int(np.argmin(frozen[-2 if mode == 'full' else -1]['validation_mse'][:6]))), ('expanded_tuned', best)]:
                    pred = models[idx].predict(features).reshape(y.shape)
                    scores = hw.evaluate(y, pred, split['test'])
                    np.save(OUT / f'hardware_{name}_{mode}_{label}.npy', pred)
                    rows.append(dict(dataset=name, N=info['N'], seed=info['seed'], method=mode+'_'+label,
                                     alpha=float(alphas[idx]), early_mae=scores['early_test']['mae'],
                                     late_mae=scores['late_test']['mae'], late_mse=scores['late_test']['R(h)']))
            pred = np.load(ROOT / info['source'] / 'pred_latent_gru.npy')
            scores = hw.evaluate(y, pred, split['test'])
            rows.append(dict(dataset=name, N=info['N'], seed=info['seed'], method='CQS-R', alpha='',
                             early_mae=scores['early_test']['mae'], late_mae=scores['late_test']['mae'],
                             late_mse=scores['late_test']['R(h)']))
            print('HARDWARE', name, 'expanded alpha', frozen[-2]['alpha'], frozen[-1]['alpha'], flush=True)
    save('hardware.json', dict(selection=frozen, metrics=rows, selection_uses='k1..4 validation MSE only'))
    write_csv('hardware.csv', rows)


def constants():
    task = TFIMTask(1, 6, 2, False, -1)
    coef = task.coefficients(-.1, -.5)
    rows = []
    for family, seeds in [('mask', (45, 46, 47)), ('E1', range(1001, 1006))]:
        for seed in seeds:
            base = ROOT / (f'results_supplementary/e1/confirmation/seed{seed}' if family == 'E1'
                           else f'results_review_masks/chain6_l2_seed{seed}_actual')
            datap = (base/'data/dataset.npz' if family == 'E1' else ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz')
            with np.load(datap) as a:
                target = a['exact_terms'][2400:]
                measured = a['measured_terms'][:2000]
            arms = ({name: base/name/'test_predictions.npz' for name in ('P-local', 'P-full', 'F-local', 'F-full')}
                    if family == 'E1' else {name: ROOT/f'results_review_masks/chain6_l2_seed{seed}_{name}/test_predictions.npz' for name in ('actual', 'full', 'random')})
            for name, path in arms.items():
                with np.load(path) as a:
                    np.testing.assert_allclose(a['exact'], target, atol=1e-12)
                    pred = a['prediction']
                    rows.append(dict(family=family, seed=seed, method=name,
                                     moment_mse=float(np.mean((pred-target)**2)),
                                     energy_mae=float(np.mean(np.abs((pred-target)@coef)))))
            for name, pred in [('zero', np.zeros_like(target)), ('training_measured_mean', np.broadcast_to(measured.mean(0), target.shape))]:
                rows.append(dict(family=family, seed=seed, method=name,
                                 moment_mse=float(np.mean((pred-target)**2)),
                                 energy_mae=float(np.mean(np.abs((pred-target)@coef)))))
    save('constant_baselines.json', rows)
    write_csv('constant_baselines.csv', rows)


def endpoint_diagnostics():
    config = json.loads((ROOT/'configs/nc_tfim_vqe_independent.json').read_text())
    rows, choices = [], []
    for case in config['cases']:
        task = TFIMTask(case['rows'], case['cols'], case['layers'], case['shared'], 1 if case['shared'] else -1)
        for seed in config['seeds']:
            name = f"{case['id']}_seed{seed}"
            data = np.load(ROOT/'results_nc_tfim_vqe_independent/datasets'/name/'dataset.npz')
            audit = ROOT/'results_nc_tfim_vqe_independent/code'/name
            for j in case['couplings']:
                tag = f'J{j:+.1f}'
                coef = task.coefficients(j, -.5)
                selected = int(np.argmin(data['measured_terms'][:case['train']]@coef))
                true_selected = float(data['exact_terms'][selected]@coef)
                choices.append(dict(case=case['id'], seed=seed, J=j, selected_training_index=selected,
                                    measured_energy=float(data['measured_terms'][selected]@coef),
                                    true_energy=true_selected, reference=-task.n/2))
                paths = dict(CQS_R=audit/tag/'vqe_redq.npz', NC=audit/tag/f"vqe_{'h_qs' if case['shared'] else 'h_cs'}.npz",
                             exact_direct=audit/tag/'vqe_exact_vqe.npz',
                             CQS_P=ROOT/'results_product_redq_independent'/name/f'vqe_{tag}.npz',
                             CQS_P_C=ROOT/'results_redq_continuation_independent'/name/f'redq_{tag}.npz',
                             NC_C=ROOT/'results_redq_continuation_independent'/name/f'nc_{tag}.npz')
                for method, path in paths.items():
                    with np.load(path) as a:
                        truth = a['exact_final'] if 'exact_final' in a else a['exact_energy'][-1]
                        pred = a['predictions'][-1]
                        x = a['parameters'][-1]
                        active = x.reshape(20, -1)[:, (1 if case['shared'] else task.n):]
                        near = np.abs(active) >= np.pi-1e-6
                        positive = truth > 0
                        for r in range(20):
                            rows.append(dict(case=case['id'], seed=seed, J=j, method=method, restart=r,
                                             true_energy=float(truth[r]), predicted_energy=float(pred[r]),
                                             true_minus_predicted=float(truth[r]-pred[r]),
                                             near_boundary_fraction=float(near[r].mean()), any_boundary=bool(near[r].any()),
                                             positive=bool(positive[r]), reaches_reference=bool(truth[r] <= -task.n/2+.01),
                                             improves_reference=bool(truth[r] < -task.n/2-.01),
                                             selected_training_true_energy=true_selected))
            data.close()
    save('endpoint_diagnostics.json', rows)
    save('single_candidate.json', choices)
    write_csv('endpoint_diagnostics.csv', rows)
    write_csv('single_candidate.csv', choices)


class ChainExact(torch.nn.Module):
    """Exact independent-angle Pauli expansion; Gaussian damping is analytic.

    Each gate coordinate appears at most once, so a monomial with t sin/cos
    factors is damped by exp(-t*sigma^2/2). Initial RX on plus is inactive.
    """
    def __init__(self, task):
        super().__init__()
        gates = []
        for layer in range(task.layers):
            if layer:
                gates += [(layer*task.block+i, 1 << i, 0) for i in range(task.n)]
            gates += [(layer*task.block+task.n+e, 0, (1 << i)|(1 << j)) for e, (i,j) in enumerate(task.edges)]
        observables = [(1 << i, 0) for i in range(task.n)] + [(0, (1 << i)|(1 << j)) for i,j in task.edges]
        terms = []
        for output, (px, pz) in enumerate(observables):
            branches = {(px, pz, (), ()): 1.}
            for idx, gx, gz in reversed(gates):
                following = defaultdict(float)
                for (x, z, cs, sn), coeff in branches.items():
                    if ((x & gz).bit_count()+(z & gx).bit_count()) % 2 == 0:
                        following[x,z,cs,sn] += coeff
                    else:
                        qx, qz = x ^ gx, z ^ gz
                        phase = (1j)**((gx & gz).bit_count()+(x & z).bit_count()-(qx & qz).bit_count())
                        phase *= (-1)**((gz & x).bit_count())
                        sign = (1j*phase).real
                        assert abs(abs(sign)-1.) < 1e-12
                        following[x,z,cs+(idx,),sn] += coeff
                        following[qx,qz,cs,sn+(idx,)] += coeff*sign
                branches = {k:v for k,v in following.items() if abs(v)>1e-14}
            reduced = defaultdict(float)
            for (x,z,cs,sn), value in branches.items():
                if z == 0:
                    reduced[tuple(sorted(cs)),tuple(sorted(sn))] += value
            for (cs,sn), value in reduced.items():
                if abs(value)>1e-14:
                    terms.append((output,cs,sn,value))
        self.terms = terms
        self.block = task.block
        self.dims = task.layers*task.block

    def forward(self, controls, J, sign, sigma=0.):
        x = controls.flatten(1)
        cs, sn = x.cos(), x.sin()
        # Group monomials by observable through a fixed coefficient vector.
        result = torch.zeros(len(x), dtype=x.dtype)
        for o, ci, si, value in self.terms:
            coefficient = sign*(-.5 if o < (self.block+1)//2 else J)
            term = x.new_full((len(x),), value*coefficient*np.exp(-.5*sigma*sigma*(len(ci)+len(si))))
            if ci:
                term = term*cs[:,ci].prod(1)
            if si:
                term = term*sn[:,si].prod(1)
            result = result+term
        return result


def grid_state_energy(x, task, J):
    """Batched little-endian statevector, same RX/RZZ gates as TFIMTask."""
    n = task.n
    idx = torch.arange(2**n)
    state = torch.ones((len(x), 2**n), dtype=torch.complex128)/np.sqrt(2**n)
    zz = [torch.tensor(1-2*((((np.arange(2**n)>>i)^(np.arange(2**n)>>j))&1)), dtype=torch.float64) for i,j in task.edges]
    for l in range(task.layers):
        for i in range(n):
            t = x[:,l,0,None]/2
            state = t.cos()*state-1j*t.sin()*state[:,idx ^ (1 << i)]
        for signs in zz:
            state = state*torch.exp(-.5j*x[:,l,1,None]*signs)
    xs = sum((state.conj()*state[:,idx ^ (1 << i)]).real.sum(1) for i in range(n))
    zzs = sum((state.abs().square()*signs).sum(1) for signs in zz)
    return task.hamiltonian_sign*(-.5*xs+J*zzs)


class GridExact(torch.nn.Module):
    """Alias-free Fourier transform of the exact energy on its full frequency grid.

    A shared RX coordinate has |frequency|<=N and RZZ <= number of edges.
    Initial shared RX is a global phase. Retain modes down to 1e-13 and record
    omitted L1 norm. This is an exact simulator oracle to numerical precision.
    """
    def __init__(self, task, J):
        super().__init__()
        bounds = [len(task.edges)] if task.layers == 1 else [len(task.edges), task.n, len(task.edges)]
        shape = [2*q+1 for q in bounds]
        mesh = np.meshgrid(*[2*np.pi*np.arange(v)/v for v in shape], indexing='ij')
        active = np.stack(mesh, axis=-1).reshape(-1,len(shape))
        controls = np.zeros((len(active),task.layers,2))
        controls.reshape(len(active),-1)[:,1:] = active
        energy = []
        with torch.no_grad():
            for batch in np.array_split(controls, max(1, int(np.ceil(len(active)/512)))):
                energy.append(grid_state_energy(torch.tensor(batch), task, J).numpy())
        coefficients = np.fft.fftn(np.concatenate(energy).reshape(shape))/len(active)
        frequencies = np.stack(np.meshgrid(*[np.fft.fftfreq(s)*s for s in shape], indexing='ij'),axis=-1)
        keep = np.abs(coefficients)>1e-13
        self.omitted_l1 = float(np.abs(coefficients[~keep]).sum())
        self.register_buffer('frequencies',torch.tensor(frequencies[keep], dtype=torch.float64))
        self.register_buffer('coef',torch.tensor(coefficients[keep],dtype=torch.complex128))
        self.grid_shape = shape

    def forward(self, controls, sigma=0.):
        phase = controls.flatten(1)[:,1:]@self.frequencies.T
        damp = torch.exp(-.5*sigma*sigma*self.frequencies.square().sum(1))
        return (torch.exp(1j*phase)*(self.coef*damp)).sum(1).real


def exact_continuation():
    config = json.loads((ROOT/'configs/nc_tfim_vqe_independent.json').read_text())
    results, checks = [], []
    for case in config['cases']:
        task = TFIMTask(case['rows'],case['cols'],case['layers'],case['shared'],1 if case['shared'] else -1)
        chain = None if task.shared else ChainExact(task)
        for J in case['couplings']:
            model = GridExact(task,J) if task.shared else chain
            predict = (lambda z,s: model(z,s)) if task.shared else (lambda z,s: model(z,J,-1,s))
            rng = np.random.default_rng(912)
            check_x = rng.uniform(-np.pi,np.pi,(8,*task.control_shape))
            tensor = torch.tensor(check_x,dtype=torch.float64,requires_grad=True)
            predicted = predict(tensor,0.)
            truth = np.array([task.energy(z,J,-.5) for z in check_x])
            error = float(np.max(np.abs(predicted.detach().numpy()-truth)))
            if error > 1e-10:
                raise ValueError(('Exact objective verification failed',case['id'],J,error))
            grad = torch.autograd.grad(predicted.sum(),tensor)[0].numpy()
            eps = 1e-5
            xp,xm=check_x.copy(),check_x.copy()
            xp.reshape(8,-1)[:,-1]+=eps
            xm.reshape(8,-1)[:,-1]-=eps
            finite=(np.array([task.energy(z,J,-.5) for z in xp])-np.array([task.energy(z,J,-.5) for z in xm]))/(2*eps)
            grad_error=float(np.max(np.abs(finite-grad.reshape(8,-1)[:,-1])))
            if grad_error>1e-8:
                raise ValueError(('Gradient verification failed',grad_error))
            # Check analytic Gaussian smoothing against a deterministic Gaussian
            # Monte Carlo diagnostic only; MC is never used for optimization.
            z=np.zeros(task.control_shape)
            if not task.shared:
                for e,(i,j) in enumerate(task.edges):
                    if i%2==0:
                        z[0,task.n+e]=np.pi
            ref=task.energy(z,J,-.5)
            np.testing.assert_allclose(ref,-task.n/2,atol=1e-12)
            check=dict(case=case['id'],J=J,random_qiskit_max_error=error,gradient_max_error=grad_error,
                       feasible_reference_energy=ref,oracle='Fourier FFT' if task.shared else 'Pauli monomials',
                       terms=int(len(model.coef)) if task.shared else len(model.terms),
                       omitted_l1=model.omitted_l1 if task.shared else 0.)
            checks.append(check)
            print('EXACT VERIFIED',check,flush=True)
            for seed in config['seeds']:
                name=f"{case['id']}_seed{seed}"
                folder=OUT/'exact_continuation'/name
                folder.mkdir(parents=True,exist_ok=True)
                audit=ROOT/'results_nc_tfim_vqe_independent/code'/name
                starts=np.load(audit/'initial_parameters.npy')
                old=np.load(audit/f'J{J:+.1f}'/'vqe_exact_vqe.npz')
                for mode,sigma in [('direct_replay',0.),('exact_continuation',1.)]:
                    tic=time.perf_counter()
                    trace=continuation(predict,starts,config['vqe'],sigma,.6)
                    end=trace['parameters'][-1]
                    final=np.array([task.energy(z,J,-.5) for z in end])
                    np.testing.assert_allclose(predict(torch.tensor(end),0.).detach().numpy(),final,atol=1e-10)
                    np.savez_compressed(folder/f'{mode}_J{J:+.1f}.npz',**trace,exact_final=final)
                    delta=float(np.max(np.abs(final-old['exact_energy'][-1]))) if mode=='direct_replay' else None
                    row=dict(case=case['id'],seed=seed,J=J,method=mode,mean_energy=float(final.mean()),
                             sd_restarts=float(final.std(ddof=1)),reference=-task.n/2,
                             attain_reference_fraction=float(np.mean(final<=-task.n/2+.01)),
                             below_reference_fraction=float(np.mean(final< -task.n/2-.01)),
                             max_difference_from_saved_direct=delta,seconds=time.perf_counter()-tic)
                    results.append(row)
                    print('EXACT RESULT',row,flush=True)
            save('exact_oracle_verification.json',checks)
            save('exact_continuation.json',results)
            write_csv('exact_continuation.csv',results)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--part',choices=('audits','exact','all'),default='all')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    save('execution_protocol.json',dict(date='2026-11-07',historical_results_read_only=True,
         exact_oracle_is='diagnostic, not measurement-matched competitor',
         continuation=dict(sigma=1.,zero_at_update=60,steps=100,Adam_lr=.1,clip=float(np.pi),restarts=20),
         units='Independent seed/dataset is statistical unit. All 20 starts retained.',
         environment=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version)))
    if args.part in ('audits','all'):
        hardware()
        constants()
        endpoint_diagnostics()
    if args.part in ('exact','all'):
        exact_continuation()


if __name__=='__main__':
    main()
