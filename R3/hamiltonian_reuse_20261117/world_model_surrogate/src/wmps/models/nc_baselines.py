"""NC experimental trigonometric regression and the separate Eq. 5 estimator.

Experimental objective/optimizer: Liao et al., SI H.2.b; public
module/model.py::Ridgejax. The published penalty is an UNSQUARED L2 norm.
Ported optimizer predictions are checked against the original JAX implementation.
"""
from itertools import combinations, product
from math import comb
import numpy as np
import torch
from torch import nn


def frequency_vectors(d, support=2, count=None, seed=42):
    """Full lexicographic C(Lambda), or uniform distinct samples from that set."""
    support = min(d, support)
    sizes = [comb(d, k)*2**k for k in range(support+1)]
    total = sum(sizes)
    if count is None or count >= total:
        if total > 2_000_000:
            raise ValueError("Full feature set too large; specify a sampled feature budget")
        rows = []
        for k in range(support+1):
            for ids in combinations(range(d), k):
                for signs in product((-1, 1), repeat=k):
                    row = np.zeros(d, dtype=np.int8)
                    row[list(ids)] = signs
                    rows.append(row)
        return np.array(sorted(rows, key=lambda row: tuple(row)), dtype=np.int8)
    rng = np.random.default_rng(seed)
    probabilities = np.array([float(s/total) for s in sizes])
    rows, seen = [], set()
    while len(rows) < count:
        k = int(rng.choice(support+1, p=probabilities))
        ids = rng.choice(d, k, replace=False)
        row = np.zeros(d, dtype=np.int8)
        row[ids] = rng.choice([-1, 1], k)
        token = row.tobytes()
        if token not in seen:
            seen.add(token)
            rows.append(row)
    return np.array(rows)


class TrigFeatureMap(nn.Module):
    """Gate-coordinate monomials; repeated gate inputs retain multiplicities."""
    def __init__(self, frequencies, gate_to_control=None):
        super().__init__()
        w = np.asarray(frequencies)
        if w.ndim != 2 or not np.isin(w, [-1, 0, 1]).all():
            raise ValueError("Frequencies must be a {-1,0,1} matrix")
        mapping = np.arange(w.shape[1]) if gate_to_control is None else np.asarray(gate_to_control)
        if mapping.shape != (w.shape[1],) or (mapping < 0).any():
            raise ValueError("One nonnegative control index is required per gate")
        d = int(mapping.max())+1
        c, s = np.zeros((len(w), d), dtype=int), np.zeros((len(w), d), dtype=int)
        for gate, coordinate in enumerate(mapping):
            c[:, coordinate] += w[:, gate] == 1
            s[:, coordinate] += w[:, gate] == -1
        self.register_buffer("cos_counts", torch.tensor(c))
        self.register_buffer("sin_counts", torch.tensor(s))

    def forward(self, x):
        if x.shape[-1] != self.cos_counts.shape[1]:
            raise ValueError("Control dimension does not match feature map")
        # Only nonzero exponents are evaluated: avoids 0**0 derivatives.
        result = torch.ones((len(x), len(self.cos_counts)), dtype=x.dtype, device=x.device)
        for j in range(x.shape[1]):
            c = self.cos_counts[:, j]
            s = self.sin_counts[:, j]
            result = result * x[:, j:j+1].cos().pow(c) * x[:, j:j+1].sin().pow(s)
        return result


def nc_objective(features, y, theta, alpha):
    return ((features @ theta-y)**2).sum(dim=0) + alpha*theta.norm(dim=0)


def fit_nc_regression(features, y, initial, alpha=1., epochs=5000, lr=.01,
                      stop_threshold=1e-4):
    """Independent scalar heads with original initialization and per-head EMA stops.

    `initial` must come from JAX key(1234), uniform[-.1,.1], in the same feature
    ordering. Batching heads does not change the independent scalar objectives.
    """
    features = torch.as_tensor(features, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64, device=features.device)
    if y.ndim == 1:
        y = y[:, None]
    if len(y) != len(features) or not torch.isfinite(y).all():
        raise ValueError("Invalid regression targets")
    t0 = torch.as_tensor(initial, dtype=torch.float64, device=features.device)
    if t0.ndim == 1:
        t0 = t0[:, None].repeat(1, y.shape[1])
    theta = nn.Parameter(t0.clone())
    optimizer = torch.optim.Adam([theta], lr=lr, betas=(.9,.999), eps=1e-8, foreach=False)
    active = torch.ones(y.shape[1], dtype=torch.bool, device=features.device)
    avg = torch.zeros(y.shape[1], dtype=torch.float64, device=features.device)
    previous = None
    saved = torch.zeros_like(theta)
    details = [None]*y.shape[1]
    for step in range(epochs):
        optimizer.zero_grad()
        loss = nc_objective(features, y, theta, alpha)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Nonfinite NC loss")
        avg = .9*avg+.1*loss.detach()
        if previous is not None:
            stopped = active & ((avg-previous).abs()/(previous.abs()+1e-8) < stop_threshold)
            for head in stopped.nonzero().flatten().tolist():
                saved[:, head] = theta.detach()[:, head]
                details[head] = dict(updates=step, loss=float(loss[head].detach()), stop="ema")
            active = active & ~stopped
        if not active.any():
            break
        previous = avg.clone()
        loss[active].sum().backward()
        optimizer.step()
    final = nc_objective(features, y, theta, alpha).detach()
    for head in active.nonzero().flatten().tolist():
        saved[:, head] = theta.detach()[:, head]
        details[head] = dict(updates=epochs, loss=float(final[head]), stop="cap")
    return saved.detach().cpu().numpy(), details


def theoretical_hcs(train_x, shadow_labels, test_x, support=2):
    """Article Eq. 5, not the fitted experimental implementation in SI H.2.b."""
    w = frequency_vectors(np.asarray(train_x).shape[1], support)
    fmap = TrigFeatureMap(w)
    with torch.no_grad():
        a = fmap(torch.as_tensor(train_x, dtype=torch.float64)).numpy()
        b = fmap(torch.as_tensor(test_x, dtype=torch.float64)).numpy()
    weights = 2.**np.count_nonzero(w, axis=1)
    return ((b*weights) @ a.T) @ np.asarray(shadow_labels) / len(train_x)
