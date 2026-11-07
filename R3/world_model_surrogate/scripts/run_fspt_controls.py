#!/usr/bin/env python3
"""Recurrence ablations and early-label-only forecasting on repeated splits."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn
from scipy.linalg import cho_factor, cho_solve

from run_fspt8_fair import ROOT, fit, parameter_steps
from src.wmps.models.world_model import ObsWorldModel
from src.wmps.models.kernel_surrogate import build_trig_features
from src.wmps.train import metrics

RIDGES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


class ResetGRU(ObsWorldModel):
    def forward(self, x):
        emb = self.encoder(x)
        zero = emb.new_zeros(len(x), self.rssm.hidden)
        hidden = torch.stack([self.rssm.gru(e, zero) for e in emb.unbind(1)], 1)
        return self.predictor(hidden), None


class PeriodMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.SiLU(), nn.Linear(256, 256),
                                 nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, output_dim))

    def forward(self, x):
        return self.net(x), None


def factory(name, n):
    if name == "period_mlp":
        return PeriodMLP(n+2, n)
    cls = ResetGRU if name == "reset_gru" else ObsWorldModel
    return cls(obs_dim=n+2, out_dim=n, encoder_hidden=128, rssm_hidden=128, predictor_hidden=128, stochastic=False)


def split_indices(total, train, val, seed):
    # Split 42 preserves the original cached ordering; 43/44 resample the same pool.
    order = np.arange(total) if seed == 42 else np.random.default_rng(seed).permutation(total)
    return order[:train], order[train:train+val], order[train+val:]


def evaluate(y, pred, train_k):
    assert pred.shape == y.shape and np.isfinite(pred).all()
    result = {"overall": metrics(y, pred), "early": metrics(y[:, :train_k], pred[:, :train_k]),
              "per_period_mae": np.mean(np.abs(y-pred), axis=(0, 2)).tolist(),
              "boundary": metrics(y[:, :, [0, -1]], pred[:, :, [0, -1]]),
              "bulk": metrics(y[:, :, 1:-1], pred[:, :, 1:-1])}
    if train_k < y.shape[1]:
        result["late"] = metrics(y[:, train_k:], pred[:, train_k:])
    return result


def ridge_fit(phi_tr, y_tr, phi_val, y_val):
    gram, rhs = phi_tr.T@phi_tr, phi_tr.T@y_tr
    best, candidates = None, []
    for lam in RIDGES:
        coef = cho_solve(cho_factor(gram+lam*np.eye(gram.shape[0]), lower=True), rhs)
        val = float(np.mean((phi_val@coef-y_val)**2))
        candidates.append({"lambda": lam, "val_mse": val})
        if best is None or val < best[0]:
            best = val, lam, coef
    return best[2], {"ridge": best[1], "best_val_mse": best[0], "validation_candidates": candidates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, help="Use only this prefix of the cached trajectory")
    ap.add_argument("--normalizer", type=int, help="Fixed denominator for the period feature")
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--n-val", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--train-k", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError("Choose a fresh output directory")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(args.dataset) as d:
        controls, y = d["X"].copy(), d["y"].copy()
    n, source_k = args.n, y.shape[1]
    k = source_k if args.k is None else args.k
    normalizer = k if args.normalizer is None else args.normalizer
    nt, nv, epochs = (800, 200, 400) if n == 20 else (2000, 400, 300)
    nt = nt if args.n_train is None else args.n_train
    nv = nv if args.n_val is None else args.n_val
    epochs = epochs if args.epochs is None else args.epochs
    if n < 3 or min(nt, nv, epochs) < 1 or not 0 < k <= source_k or normalizer < k:
        raise ValueError("Invalid dimensions, budget, prefix or normalizer")
    y = y[:, :k].copy()
    assert y.shape == (nt+2*nv, k, n) and controls.shape == (len(y), n+1)
    assert len(np.unique(controls, axis=0)) == len(y) and 0 < args.train_k <= k
    assert np.isfinite(y).all() and np.isfinite(controls).all()
    tr, va, te = split_indices(len(y), nt, nv, args.seed)
    np.savez_compressed(args.out/"split.npz", train=tr, val=va, test=te)
    np.savez_compressed(args.out/"test_targets.npz", X=controls[te], y=y[te])
    x = parameter_steps(controls, k, normalizer=normalizer)
    cutoff = args.train_k
    protocol = dict(N=n, K=k, K_source=source_k, period_normalizer=normalizer,
                    dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    K_train=cutoff, dataset_seed=42, split_seed=args.seed, initialization_seed=args.seed,
                    split_sizes=[nt, nv, nv], split_scope="three partitions of one cached pool; 42 original order, 43/44 shuffled",
                    inputs=f"controls plus k/{normalizer}; no test observations",
                    labels="exact, all selected periods" if cutoff == k else "exact, early periods only for forecasting",
                    selection="MSE on validation trajectories at k<=K_train only",
                    optimizer=dict(name="Adam", lr=0.001, weight_decay=0.0001, batch_size=32, epochs=epochs, patience=30),
                    ridge_grid=RIDGES, hypotheses_fixed_before_test=True)
    result = dict(protocol=protocol, environment=dict(python=platform.python_version(), numpy=np.__version__,
                  torch=torch.__version__, device=torch.cuda.get_device_name() if args.device=="cuda" else args.device), models={})
    (args.out/"protocol.json").write_text(json.dumps(protocol, indent=2))

    def retain(name, pred, **detail):
        np.save(args.out/f"pred_{name}.npy", pred)
        result["models"][name] = dict(**evaluate(y[te], pred, cutoff), **detail)
        print(name, result["models"][name].get("late", result["models"][name]["overall"]), flush=True)

    for name in ("redq", "reset_gru", "period_mlp"):
        model, training = fit(lambda: factory(name, n), x[tr, :cutoff], y[tr, :cutoff],
                              x[va, :cutoff], y[va, :cutoff], args.seed, args.device, args.out, name, epochs)
        model.eval()
        xt = torch.as_tensor(x[te], device=args.device)
        with torch.no_grad():
            retain(name, model(xt)[0].cpu().numpy(), training=training)
            if name == "redq":
                reset = factory("reset_gru", n).to(args.device).eval()
                reset.load_state_dict(model.state_dict())
                retain("test_reset", reset(xt)[0].cpu().numpy(), intervention="reset state at inference, original trained weights")
                del reset
        del model
    if cutoff == k:
        phi_tr, phi_va, phi_te = [build_trig_features(controls[s], 2) for s in (tr, va, te)]
        coef, detail = ridge_fit(phi_tr, y[tr].reshape(nt, -1), phi_va, y[va].reshape(nv, -1))
        np.save(args.out/"coef_kernel.npy", coef)
        retain("kernel", (phi_te@coef).reshape(y[te].shape), **detail)
    else:
        # The same trig family now takes period as an input, so it can forecast.
        phi_tr = build_trig_features(x[tr, :cutoff].reshape(-1, n+2), 2)
        phi_va = build_trig_features(x[va, :cutoff].reshape(-1, n+2), 2)
        coef, detail = ridge_fit(phi_tr, y[tr, :cutoff].reshape(-1, n), phi_va, y[va, :cutoff].reshape(-1, n))
        del phi_tr, phi_va
        np.save(args.out/"coef_time_kernel.npy", coef)
        retain("time_kernel", (build_trig_features(x[te].reshape(-1, n+2), 2)@coef).reshape(y[te].shape), **detail)
        train_targets = y[tr, :cutoff]
        for name, center in (("train_mean", np.mean(train_targets, axis=(0, 1))),
                             ("train_median", np.median(train_targets, axis=(0, 1))),
                             ("zero", np.zeros(n))):
            retain(name, np.broadcast_to(center, y[te].shape).copy())
        parity = np.stack([np.mean(train_targets[:, j::2], axis=(0, 1)) for j in (0, 1)])
        retain("parity_mean", np.broadcast_to(parity[np.arange(k)%2], y[te].shape).copy())
    result["completed"] = True
    temporary = args.out/"results.json.tmp"
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.out/"results.json")


if __name__ == "__main__":
    main()
