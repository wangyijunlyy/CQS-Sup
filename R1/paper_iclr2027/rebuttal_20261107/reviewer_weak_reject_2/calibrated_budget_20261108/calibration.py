"""Measured-validation prior calibration and fully timed candidate search."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve

HERE = Path(__file__).resolve().parent
ACTIVE = np.arange(6,22)
ALPHAS = (.001,.01,.1,1.,10.,100.)
UPPER = np.triu_indices(16)
PRIOR_WEIGHTS = (0., .5, 1.)


def features(z):
    return np.concatenate((np.ones((len(z),1)),z,z[:,UPPER[0]]*z[:,UPPER[1]]),axis=1)


def torch_features(z):
    return torch.cat((torch.ones_like(z[:,:1]),z,z[:,UPPER[0]]*z[:,UPPER[1]]),dim=1)


def select_correction(z, measured, prior, alphas=PRIOR_WEIGHTS):
    train, val = features(z[:192]), features(z[192:])
    residual = measured[:, None]-prior[:, None]*np.asarray(alphas)[None]
    center = residual[:192].mean(0)
    mean = train[:, 1:].mean(0)
    a, b = train[:, 1:]-mean, val[:, 1:]-mean
    gram, rhs = a.T@a, a.T@(residual[:192]-center)
    records = []
    for ridge in ALPHAS:
        coefficients = cho_solve(cho_factor(gram+ridge*np.eye(152)), rhs)
        prediction = prior[192:, None]*np.asarray(alphas)[None]+center+b@coefficients
        errors = ((prediction-measured[192:, None])**2).mean(0)
        for index, alpha in enumerate(alphas):
            w = np.r_[center[index]-mean@coefficients[:, index], coefficients[:, index]]
            records.append((float(errors[index]), float(alpha), float(ridge), w))
    error, alpha, ridge, w = min(records, key=lambda r: r[:3])
    return w, dict(alpha=alpha, ridge=ridge, validation_mse=error,
        candidates=[dict(validation_mse=e, alpha=a, ridge=r) for e,a,r,_ in records])


def local_controls(z, center, radius):
    ct = torch.as_tensor(center.reshape(1, -1), dtype=z.dtype, device=z.device)
    x = ct.expand(len(z), -1).clone()
    x[:, ACTIVE] = ct[:, ACTIVE]+z*(radius/4)
    return x.reshape(-1, 2, 11)


def local_objective(z, prior, center, radius, weights, alpha):
    w = torch.as_tensor(weights, dtype=z.dtype, device=z.device)
    value = torch_features(z)@w
    if alpha != 0:
        value = value+alpha*prior(local_controls(z, center, radius))
    return value


def select_eligible(records, budget, count=20):
    eligible = [r for r in records if r['seconds'] <= budget]
    joined = [(p, x) for r in eligible for p,x in zip(r['predicted'], r['parameters'])]
    joined.sort(key=lambda item: item[0])
    selected = joined[:count]
    return dict(predicted=[p for p,_ in selected], parameters=[x for _,x in selected],
                eligible_blocks=len(eligible))


def search(prior, center, radius, weights, alpha, rng, budget=None):
    """Run one fixed block, or admit complete blocks within a wall-clock cap."""
    # Warm up model/autograd/Adam initialization without consuming search RNG.
    warm = torch.zeros((20,16), dtype=torch.float64, requires_grad=True)
    warm_opt = torch.optim.Adam([warm], lr=.03, foreach=False)
    for _ in range(2):
        warm_opt.zero_grad()
        local_objective(warm, prior, center, radius, weights, alpha).sum().backward()
        warm_opt.step()
    records, pool, updates, attempted, copied_seconds = [], [], 0, 0, 0.
    started = time.perf_counter()
    deadline = float('inf') if budget is None else started+budget
    while time.perf_counter() < deadline:
        attempted += 1
        z = torch.tensor(rng.uniform(-1,1,(20,16)), dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            z[0] = 0
        opt = torch.optim.Adam([z], lr=.03, foreach=False)
        finished = True
        for _ in range(100):
            if time.perf_counter() >= deadline:
                finished = False
                break
            opt.zero_grad()
            local_objective(z, prior, center, radius, weights, alpha).sum().backward()
            opt.step()
            with torch.no_grad():
                z.clamp_(-1,1)
            updates += 1
        if not finished:
            break
        with torch.no_grad():
            prediction = local_objective(z, prior, center, radius, weights, alpha).numpy().copy()
            controls = local_controls(z, center, radius).numpy().copy()
        joined = pool+[(float(v), x.copy()) for v,x in zip(prediction, controls)]
        joined.sort(key=lambda r: r[0])
        proposed_pool = [(v,x.copy()) for v,x in joined[:20]]
        elapsed = time.perf_counter()-started
        if budget is not None and elapsed > budget:
            break
        pool = proposed_pool
        copied_seconds = elapsed
        records.append(dict(seconds=elapsed, predicted=prediction.tolist(),
                            parameters=controls.tolist()))
        if budget is None:
            break
    actual = time.perf_counter()-started
    if not pool:
        # No incomplete or late block supplies a candidate.
        pool = [(float('nan'), center.copy()) for _ in range(20)]
    return np.stack([x for _,x in pool]), np.asarray([v for v,_ in pool]), dict(
        budget=budget, eligible_seconds=copied_seconds, actual_seconds=actual,
        overrun_seconds=max(0., actual-budget) if budget else 0.,
        completed_blocks=len(records), attempted_blocks=attempted, executed_updates=updates,
        eligible_block_records=records)
