"""Finite-shot direct optimizers with local exploratory starts and retention.

Optimizer iterates continue across reporting checkpoints, while the independently
measured incumbent is retained for final output. No exact values guide updates.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import sys
import time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from resource_chain import ChainMeasurement, shifted_gradients, shot_cost, ACTIVE, wrap, BUDGETS
from run_fourier_chain import write, COUPLINGS


def initial_controls(seed, j):
    incumbent = np.zeros((20, 2, 11))
    incumbent[:, 0, 6:9] = np.pi
    x = incumbent.copy().reshape(20, 22)
    rng = np.random.default_rng(810000+seed*100+COUPLINGS.index(j))
    x[1:, ACTIVE] += rng.uniform(-.2, .2, (19, 16))
    return x.reshape(20, 2, 11), incumbent


def run(seed, j, method):
    out = HERE/'direct_jitter'/f'seed{seed}_{method}_J{j:+.1f}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        return
    engine = ChainMeasurement(100)
    rng = np.random.default_rng(820000+seed*100+COUPLINGS.index(j))
    x, incumbent = initial_controls(seed, j)
    m = np.zeros((20, 16))
    v = m.copy()
    iteration = 0
    trace, points = [], []
    start = time.perf_counter()
    for budget in BUDGETS:
        control_cost = 40 if method == 'spsa' else 640
        while engine.shots+shot_cost(control_cost, 100)+shot_cost(40, 100) <= budget:
            iteration += 1
            if method == 'spsa':
                delta = rng.choice([-1., 1.], size=(20, 16))
                c = .1/iteration**.101
                xp, xm = x.reshape(20, 22).copy(), x.reshape(20, 22).copy()
                xp[:, ACTIVE] += c*delta
                xm[:, ACTIVE] -= c*delta
                r = engine.measure(wrap(np.concatenate([xp, xm])).reshape(-1, 2, 11), j, rng)
                gradient = (r['energy'][:20]-r['energy'][20:])[:, None]*delta/(2*c)
                gain = .1*(11/(iteration+10))**.602
                flat = x.reshape(20, 22).copy()
                flat[:, ACTIVE] -= gain*gradient
            else:
                gradient = shifted_gradients(engine, x, j, rng)
                m = .9*m+.1*gradient
                v = .999*v+.001*gradient**2
                flat = x.reshape(20, 22).copy()
                flat[:, ACTIVE] -= .03*(m/(1-.9**iteration))/(np.sqrt(v/(1-.999**iteration))+1e-8)
            x = wrap(flat).reshape(20, 2, 11)
            trace.append(x.copy())
        verification_shots = (budget-engine.shots)//80
        assert verification_shots >= 100
        candidate = engine.measure(x, j, rng, verification_shots)
        old = engine.measure(incumbent, j, rng, verification_shots)
        accepted = candidate['energy']+2*np.sqrt(candidate['variance']+old['variance']) < old['energy']
        incumbent = np.where(accepted[:, None, None], x, incumbent)
        # Continue exploration from x rather than resetting a rejected iterate.
        np.savez_compressed(out/f'budget{budget}.npz', parameters=incumbent,
            candidate_parameters=x, accepted=accepted,
            candidate_energy=candidate['energy'], candidate_variance=candidate['variance'],
            incumbent_energy=old['energy'], incumbent_variance=old['variance'])
        points.append(dict(budget=budget, actual_shots=engine.shots, iterations=iteration,
            final_verification_shots_per_group=verification_shots))
    engine.save(out/'measurements.npz')
    np.save(out/'iterates.npy', np.array(trace))
    write(out/'complete.json', dict(seed=seed, J=j, method=method, points=points,
        seconds=time.perf_counter()-start, total_shots=engine.shots,
        initialization='reference plus19 uniform perturbations within±.2 in16 active angles',
        retention='independent2SE measured incumbent acceptance; optimizer trajectory continues',
        exact_scoring_started=False, hardware_shots=0))
    print('DIRECT JITTER SAVED', seed, j, method, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=(45, 46, 47), required=True)
    args = ap.parse_args()
    write(HERE/f'direct_jitter_protocol_seed{args.seed}.json', dict(seed=args.seed,
        couplings=COUPLINGS, methods=['spsa', 'parameter_shift_adam'],
        total_budgets=BUDGETS, gradient_shots_per_group=100,
        SPSA='a0.1,A10,alpha.602,c0.1,gamma.101', Adam='lr.03,beta.9/.999',
        initialization='reference plus19 uniform perturbations within±.2, same range as first local search box',
        candidate_selection='fresh measured2SE rule, not exact energy',
        checkpoints='incumbent retained without resetting optimizer iterates',
        final_evaluation='only after all36 search records are complete'))
    for j in COUPLINGS:
        for method in ('spsa', 'parameter_shift_adam'):
            run(args.seed, j, method)


if __name__ == '__main__':
    main()
