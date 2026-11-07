"""Train the world model and the kernel baseline on quantum expectation data."""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data.quantum_data import load_or_generate
from .models.kernel_surrogate import KernelSurrogate
from .models.world_model import WorldModel


DEFAULTS = dict(
    N=6, L=2, J=-0.1, h=-0.5, noise_level=0.0,
    pauli_weights=[0.45, 0.45, 0.1],
    x_range=np.pi,
    n_train=256, n_val=64, n_test=128, seed=42,
    encoder_hidden=128, rssm_hidden=128, predictor_hidden=128,
    stochastic=False, beta=0.001, dropout=0.0,
    epochs=300, batch_size=32, lr=0.001, weight_decay=1e-4, patience=30,
    device="auto", kappa_lambda=2, ridge_lam=1.0,
    eval_eps=0.05, shots_per_sample=1000,
    cache_dir="cache", out_dir="results", ckpt_dir="checkpoints",
)


def load_config(path: str) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        try:
            import yaml
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
                cfg.update(loaded)
        except Exception as e:  # yaml missing or unreadable -> defaults
            print(f"[config] warning: could not load {path} ({e}); using defaults")
    # coerce numeric-looking strings (PyYAML parses "1e-4" as str)
    def _coerce(v):
        if isinstance(v, str):
            s = v.strip()
            try:
                if re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", s):
                    return float(s) if ("." in s or "e" in s or "E" in s) else int(s)
            except ValueError:
                pass
        if isinstance(v, dict):
            return {k: _coerce(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_coerce(x) for x in v]
        return v
    cfg = {k: _coerce(v) for k, v in cfg.items()}
    return cfg


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson, "R(h)": rmse ** 2}


def train_world_model(
    X_train, y_train, X_val, y_val, cfg, device, tag="wm", verbose=True
):
    L, d_layer = X_train.shape[1], X_train.shape[2]
    out_dim = y_train.shape[1]
    model = WorldModel(
        in_dim=d_layer,
        out_dim=out_dim,
        encoder_hidden=cfg["encoder_hidden"],
        rssm_hidden=cfg["rssm_hidden"],
        predictor_hidden=cfg["predictor_hidden"],
        stochastic=cfg["stochastic"],
        beta=cfg["beta"],
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(Xt, yt), batch_size=cfg["batch_size"], shuffle=True
    )

    best_val = float("inf")
    best_state = None
    patience = cfg["patience"]
    wait = 0
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tot = 0.0
        tot_kl = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss, mse, kl = model.loss(xb, yb)
            loss.backward()
            optimizer.step()
            tot += mse * xb.size(0)
            if kl is not None:
                tot_kl += kl * xb.size(0)
        train_mse = tot / len(loader.dataset)
        model.eval()
        with torch.no_grad():
            yv_hat, _ = model(Xv)
            val_mse = float(torch.nn.functional.mse_loss(yv_hat, yv).item())
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[wm] early stop @ epoch {epoch} (best val MSE {best_val:.6f})")
                break
        if verbose and epoch % 50 == 0:
            kl_str = f"  train_kl {tot_kl / len(loader.dataset):.5f}" if cfg.get("stochastic") else ""
            print(f"[wm] epoch {epoch:4d}  train_mse {train_mse:.6f}  val_mse {val_mse:.6f}{kl_str}")
    model.load_state_dict(best_state)
    if verbose:
        print(f"[wm] trained in {time.time()-t0:.1f}s, best val MSE {best_val:.6f}")
    return model, best_val


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tfim.yaml")
    ap.add_argument("--quick", action="store_true", help="small smoke-test config")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--noise-level", type=float, default=None)
    ap.add_argument("--x-range", type=float, default=None)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stochastic", action="store_true", help="enable stochastic latent branch")
    ap.add_argument("--beta", type=float, default=None, help="KL weight for stochastic branch")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.quick:
        cfg.update(N=4, L=2, x_range=1.0, n_train=512, n_val=128, n_test=128, epochs=150,
                   patience=20, scaling_N=[4], scaling_n=[64], noise_sweep=[0.0])
    for key, val in [("N", args.N), ("L", args.L), ("noise_level", args.noise_level),
                     ("x_range", args.x_range),
                     ("n_train", args.n_train), ("epochs", args.epochs),
                     ("device", args.device), ("beta", args.beta)]:
        if val is not None:
            cfg[key] = val
    if args.stochastic:
        cfg["stochastic"] = True

    device = get_device(cfg["device"])
    print(f"[wmps] device={device}, N={cfg['N']}, L={cfg['L']}, "
          f"noise={cfg['noise_level']}, n_train={cfg['n_train']}")

    n_total = cfg["n_train"] + cfg["n_val"] + cfg["n_test"]
    X, y, y_shadow = load_or_generate(
        n_samples=n_total, N=cfg["N"], L=cfg["L"], J=cfg["J"], h=cfg["h"],
        noise_level=cfg["noise_level"], pauli_weights=cfg["pauli_weights"],
        x_range=cfg["x_range"],
        seed=cfg["seed"], shadow_T=cfg.get("shadow_T", 32),
        cache_dir=cfg["cache_dir"], verbose=True,
    )
    n_train, n_val, n_test = cfg["n_train"], cfg["n_val"], cfg["n_test"]
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    Xte, yte = X[n_train + n_val:], y[n_train + n_val:]
    ys_te = y_shadow[n_train + n_val:]
    print(f"[data] train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    # ---- world model ----
    model, _ = train_world_model(Xtr, ytr, Xva, yva, cfg, device, tag=args.tag)
    model.eval()
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        yhat_wm, _ = model(Xte_t)
        yhat_wm = yhat_wm.cpu().numpy()
    m_wm = metrics(yte, yhat_wm)

    # ---- kernel baselines: h_qs (clean labels) and h_cs (shadow labels) ----
    Xtr_flat = Xtr.reshape(n_train, -1)
    Xte_flat = Xte.reshape(n_test, -1)
    ker_qs = KernelSurrogate(Lambda=cfg["kappa_lambda"], lam=cfg["ridge_lam"], seed=cfg["seed"])
    ker_qs.fit(Xtr_flat, ytr)
    m_qs = metrics(yte, ker_qs.predict(Xte_flat))
    ker_cs = KernelSurrogate(Lambda=cfg["kappa_lambda"], lam=cfg["ridge_lam"], seed=cfg["seed"])
    ker_cs.fit(Xtr_flat, y_shadow[:n_train])
    m_cs = metrics(yte, ker_cs.predict(Xte_flat))
    m_raw = metrics(yte, ys_te)

    print("\n=== Test metrics ===")
    print(f"{'model':<18}{'MAE':>10}{'RMSE':>10}{'Pearson':>10}{'R2':>10}{'R(h)':>12}")
    for name, m in [("world_model", m_wm), ("kernel h_cs", m_cs),
                    ("kernel h_qs", m_qs), ("raw shadow", m_raw)]:
        print(f"{name:<18}{m['mae']:>10.5f}{m['rmse']:>10.5f}{m['pearson']:>10.4f}"
              f"{m['r2']:>10.4f}{m['R(h)']:>12.6f}")

    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["ckpt_dir"], f"wm_{args.tag}.pt"))
    record = {
        "tag": args.tag, "config": {k: v for k, v in cfg.items() if isinstance(v, (int, float, str, list, bool))},
        "world_model": m_wm, "kernel_hcs": m_cs, "kernel_hqs": m_qs, "raw_shadow": m_raw,
    }
    with open(os.path.join(cfg["out_dir"], f"metrics_{args.tag}.json"), "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[save] {cfg['ckpt_dir']}/wm_{args.tag}.pt ; {cfg['out_dir']}/metrics_{args.tag}.json")


if __name__ == "__main__":
    main()
