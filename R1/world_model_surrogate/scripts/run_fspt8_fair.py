#!/usr/bin/env python3
"""Matched-information FSPT prediction using the original cached split."""
import argparse
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.wmps.models.world_model import ObsWorldModel
from src.wmps.models.kernel_surrogate import build_trig_features
from src.wmps.train import metrics


def parameter_steps(controls, horizon=12, normalizer=None):
    normalizer = horizon if normalizer is None else normalizer
    if horizon < 1 or normalizer < horizon:
        raise ValueError("Normalizer must be at least the positive horizon")
    steps = np.arange(1, horizon+1, dtype=np.float32)/normalizer
    return np.concatenate([np.broadcast_to(controls[:, None, :], (len(controls), horizon, controls.shape[1])),
                           np.broadcast_to(steps[None, :, None], (len(controls), horizon, 1))], axis=-1).astype(np.float32)


class PeriodMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 256), nn.SiLU(), nn.Linear(256, 256),
                                 nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 8))

    def forward(self, x):
        return self.net(x), None


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(factory, x_train, y_train, x_val, y_val, seed, device, out, name, epochs, checkpoints=(), *, min_epochs=0, patience=30):
    if not 0 <= min_epochs <= epochs or patience < 1:
        raise ValueError("Require 0 <= min_epochs <= epochs and positive patience")
    if any(b < 1 or b > epochs for b in checkpoints):
        raise ValueError("Checkpoint budgets must lie within the training cap")
    seed_all(seed)
    model = factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt, xv, yv = [torch.as_tensor(a, dtype=torch.float32, device=device)
                      for a in (x_train, y_train, x_val, y_val)]
    loader = DataLoader(TensorDataset(xt, yt), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, state, wait, best_epoch = float("inf"), None, 0, 0
    history = []
    start = time.perf_counter()

    def save_budget(budget, epoch):
        torch.save(state, out/f"{name}_budget{budget}.pt")
        info = dict(budget=budget, best_val_mse=best, best_epoch=best_epoch,
                    epochs_run=epoch, parameters=sum(p.numel() for p in model.parameters()),
                    training_seconds=time.perf_counter()-start)
        (out/f"{name}_budget{budget}.json").write_text(json.dumps(info, indent=2))

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(x)[0]
            loss = torch.mean((pred-y)**2)
            loss.backward()
            opt.step()
            total += loss.item()*len(x)
        model.eval()
        with torch.no_grad():
            val = torch.mean((model(xv)[0]-yv)**2).item()
        history.append(dict(epoch=epoch, train_mse=total/len(xt), val_mse=val))
        if val < best:
            best, best_epoch, wait = val, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch in checkpoints:
            save_budget(epoch, epoch)
        if epoch % 25 == 0:
            print(f"{name} seed={seed} epoch={epoch} val_MSE={val:.6f} best={best:.6f}", flush=True)
        if wait >= patience and epoch >= min_epochs:
            break
    if state is None:
        raise RuntimeError("No finite validation checkpoint")
    for budget in checkpoints:
        if budget > epoch:
            save_budget(budget, epoch)
    model.load_state_dict(state)
    torch.save(state, out/f"{name}.pt")
    (out/f"{name}_history.json").write_text(json.dumps(history, indent=2))
    return model, dict(best_val_mse=best, best_epoch=best_epoch, epochs_run=epoch,
                        parameters=sum(p.numel() for p in model.parameters()),
                        training_seconds=time.perf_counter()-start)


def score(y, pred):
    assert pred.shape == y.shape and np.isfinite(pred).all()
    return dict(overall=metrics(y, pred), boundary=metrics(y[:, :, [0, 7]], pred[:, :, [0, 7]]),
                bulk=metrics(y[:, :, 1:7], pred[:, :, 1:7]),
                per_period_mae=np.mean(np.abs(y-pred), axis=(0, 2)).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"results.json").exists():
        raise FileExistsError("Use a fresh output directory; previous results are preserved")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(args.dataset) as data:
        # Deliberately never read obs: measurements cannot enter features or labels.
        controls = data["X"].copy()
        y = data["y"].copy()
    assert controls.shape == (2800, 9) and y.shape == (2800, 12, 8)
    assert np.isfinite(controls).all() and np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 1+1e-6
    assert len(np.unique(controls, axis=0)) == len(controls)
    tr, va, te = slice(0, 2000), slice(2000, 2400), slice(2400, 2800)
    x = parameter_steps(controls)
    np.savez_compressed(args.out/"test_targets.npz", X=controls[te], y=y[te])
    protocol = dict(N=8, K=12, dataset_seed=42, initialization_seed=args.seed,
                    split=dict(train=[0, 2000], val=[2000, 2400], test=[2400, 2800]),
                    inputs="controls and period identity only; no test measurements",
                    labels="exact expectations for all models", selection="validation MSE only",
                    neural=dict(epochs=args.epochs, batch_size=32, patience=30, lr=1e-3, weight_decay=1e-4),
                    kernel=dict(feature_support=2, ridge_grid=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
                    split_scope="fixed dataset; neural initialization/shuffle repeats, not independent data seeds")
    (args.out/"protocol.json").write_text(json.dumps(protocol, indent=2))
    results = dict(protocol=protocol, environment=dict(python=platform.python_version(), torch=torch.__version__,
                  numpy=np.__version__, device=torch.cuda.get_device_name() if args.device=="cuda" else args.device), models={})
    factories = dict(redq=lambda: ObsWorldModel(obs_dim=10, out_dim=8, encoder_hidden=128,
                                               rssm_hidden=128, predictor_hidden=128, stochastic=False),
                     static_mlp=PeriodMLP)
    for name, factory in factories.items():
        model, details = fit(factory, x[tr], y[tr], x[va], y[va], args.seed, args.device, args.out, name, args.epochs)
        model.eval()
        with torch.no_grad():
            pred = model(torch.as_tensor(x[te], dtype=torch.float32, device=args.device))[0].cpu().numpy()
        np.save(args.out/f"pred_{name}.npy", pred)
        results["models"][name] = dict(**score(y[te], pred), training=details)
        print(name, results["models"][name]["overall"], flush=True)
        del model
    # Exact multi-output ridge, equivalent to separate per-(period,observable) fits.
    from scipy.linalg import cho_factor, cho_solve
    phi_tr, phi_va, phi_te = [build_trig_features(controls[s], 2) for s in (tr, va, te)]
    gram, rhs = phi_tr.T@phi_tr, phi_tr.T@y[tr].reshape(2000, -1)
    selected = None
    candidates = []
    for lam in protocol["kernel"]["ridge_grid"]:
        coef = cho_solve(cho_factor(gram+lam*np.eye(gram.shape[0]), lower=True), rhs)
        val = float(np.mean((phi_va@coef-y[va].reshape(400, -1))**2))
        candidates.append(dict(ridge=lam, val_mse=val))
        if lam == 1:
            fixed_coef = coef.copy()
        if selected is None or val < selected[0]:
            selected = (val, lam, coef.copy())
    for name, coef, lam in [("kernel_hqs", selected[2], selected[1]), ("kernel_hqs_fixed", fixed_coef, 1.0)]:
        pred = (phi_te@coef).reshape(y[te].shape)
        np.save(args.out/f"pred_{name}.npy", pred)
        np.save(args.out/f"coef_{name}.npy", coef)
        results["models"][name] = dict(**score(y[te], pred), ridge=lam, parameters=int(coef.size), validation_candidates=candidates)
        print(name, results["models"][name]["overall"], flush=True)
    results["completed"] = True
    (args.out/"results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
