#!/usr/bin/env python3
"""Table-1 TFIM prediction with matched h_cs-style finite-shot labels."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_fspt8_fair import fit
from run_fspt_controls import split_indices
from src.wmps.data.quantum_data import _marginals_from_amplitudes, build_circuit, build_tfim_hamiltonian
from src.wmps.models.nc_baselines import TrigFeatureMap, fit_nc_regression, frequency_vectors
from src.wmps.models.world_model import WorldModel
from src.wmps.train import metrics


NS = (8, 10, 12, 14)
SEEDS = (42, 43, 44)
J = -0.1
H = -0.5
SHOTS = 32


def build_model(n: int) -> WorldModel:
    return WorldModel(
        in_dim=2 * n - 1,
        out_dim=1,
        encoder_hidden=128,
        rssm_hidden=128,
        predictor_hidden=128,
        stochastic=False,
        dropout=0.0,
    )


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def score(y: np.ndarray, pred: np.ndarray) -> dict:
    return metrics(y.reshape(-1, 1), pred.reshape(-1, 1))


def load_exact(n: int) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / "results_tfim_repeats" / "datasets" / f"dataset_n{n}.npz"
    with np.load(path) as data:
        x = data["X"].astype(np.float64)
        y = data["y"].astype(np.float64).reshape(-1)
    if x.shape != (2800, 2, 2 * n - 1) or y.shape != (2800,):
        raise ValueError(f"Unexpected shape for N={n}: {x.shape}, {y.shape}")
    return x, y


def scalar_shadow_energy(z: np.ndarray, zz: np.ndarray, xobs: np.ndarray, n: int, shots: int, rng) -> float:
    basis = rng.integers(0, 3, size=(shots, n), dtype=np.int8)
    per_shot = np.zeros(shots, dtype=np.float64)
    x_mask = basis == 0
    x_sign = np.where(rng.random((shots, n)) < (1.0 + xobs[None, :]) / 2.0, 1.0, -1.0)
    per_shot += (-H) * np.sum(3.0 * x_sign * x_mask, axis=1)
    for i in range(n - 1):
        zz_mask = (basis[:, i] == 2) & (basis[:, i + 1] == 2)
        zz_sign = np.where(rng.random(shots) < (1.0 + zz[i]) / 2.0, 1.0, -1.0)
        per_shot += (-J) * 9.0 * zz_sign * zz_mask
    return float(np.mean(per_shot))


def prepare(args: argparse.Namespace) -> None:
    (args.out / "datasets").mkdir(parents=True, exist_ok=True)
    for n in NS:
        target = args.out / "datasets" / f"table1_hcs_n{n}_T{SHOTS}.npz"
        if target.exists() and not args.force:
            print(f"[prepare] keep {target}", flush=True)
            continue
        x, y = load_exact(n)
        ham = build_tfim_hamiltonian(n, J, H)
        rng = np.random.default_rng(20260908 + n)
        y_shadow = np.zeros(len(x), dtype=np.float64)
        t0 = time.perf_counter()
        from qiskit.quantum_info import Statevector

        for idx in range(len(x)):
            circuit = build_circuit(x[idx, 0], n).compose(build_circuit(x[idx, 1], n))
            sv = Statevector(circuit)
            exact = float(np.real(sv.expectation_value(ham)))
            np.testing.assert_allclose(exact, y[idx], rtol=1e-10, atol=1e-10)
            z, zz, xobs = _marginals_from_amplitudes(sv.data, n)
            y_shadow[idx] = scalar_shadow_energy(z, zz, xobs, n, SHOTS, rng)
            if (idx + 1) % 200 == 0:
                print(f"[prepare] N={n} {idx+1}/2800 in {time.perf_counter()-t0:.1f}s", flush=True)
        np.savez_compressed(target, X=x, y_exact=y, y_hcs=y_shadow)
        print(f"[prepare] wrote {target}", flush=True)


def fit_hcs(x: np.ndarray, y_label: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, y_exact: np.ndarray, out: Path) -> dict:
    flat = x.reshape(len(x), -1)
    frequencies = frequency_vectors(flat.shape[1], support=2)
    fmap = TrigFeatureMap(frequencies)
    with torch.no_grad():
        phi_train = fmap(torch.as_tensor(flat[train_idx], dtype=torch.float64)).numpy()
        phi_test = fmap(torch.as_tensor(flat[test_idx], dtype=torch.float64)).numpy()
    init_path = ROOT / "results_nc_reference" / f"initial_f{phi_train.shape[1]}.npy"
    initial = np.load(init_path).reshape(-1, 1) if init_path.exists() else np.random.default_rng(1234).uniform(-0.1, 0.1, (phi_train.shape[1], 1))
    coef, details = fit_nc_regression(phi_train, y_label[train_idx, None], initial=initial, alpha=10.0, epochs=5000, lr=0.01)
    pred = (phi_test @ coef).reshape(-1)
    np.save(out / "coef_hcs.npy", coef)
    np.save(out / "pred_hcs.npy", pred)
    return {
        "metrics": score(y_exact[test_idx], pred),
        "features": int(phi_train.shape[1]),
        "training": details[0],
        "alpha": 10.0,
        "initialization": str(init_path) if init_path.exists() else "numpy default_rng(1234)",
    }


def run(args: argparse.Namespace) -> None:
    out = args.out / f"n{args.n}_seed{args.seed}"
    if (out / "results.json").exists() and not args.force:
        print(f"[run] keep {out}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(args.out / "datasets" / f"table1_hcs_n{args.n}_T{SHOTS}.npz") as data:
        x = data["X"].astype(np.float64)
        y_exact = data["y_exact"].astype(np.float64)
        y_hcs = data["y_hcs"].astype(np.float64)
    tr, va, te = split_indices(len(x), 2000, 400, args.seed)
    np.savez_compressed(out / "split.npz", train=tr, val=va, test=te)
    protocol = {
        "version": "table1-tfim-hcs-v1",
        "N": args.n,
        "L": 2,
        "seed": args.seed,
        "train": 2000,
        "val": 400,
        "test": 400,
        "shots_per_input": SHOTS,
        "labels": "same scalar h_cs-style finite-shot energy labels for RED-Q and trig regression",
        "test_labels": "exact noiseless final-layer energy",
        "baseline": "NC-style support-two trigonometric regression, unsquared-L2 Adam, alpha=10",
        "redq_epochs": args.epochs,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    model, detail = fit(
        lambda: build_model(args.n),
        x[tr],
        y_hcs[tr, None],
        x[va],
        y_hcs[va, None],
        args.seed,
        args.device,
        out,
        "redq_hcs",
        args.epochs,
    )
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(x[te], dtype=torch.float32, device=args.device))[0].cpu().numpy().reshape(-1)
    np.save(out / "pred_redq_hcs.npy", pred)
    result = {
        "completed": True,
        "protocol": protocol,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name() if args.device == "cuda" and torch.cuda.is_available() else args.device,
        },
        "label_noise": {
            "train_mse": float(np.mean((y_hcs[tr] - y_exact[tr]) ** 2)),
            "val_mse": float(np.mean((y_hcs[va] - y_exact[va]) ** 2)),
        },
        "models": {
            "redq_hcs": {"metrics": score(y_exact[te], pred), "training": detail},
            "hcs": fit_hcs(x, y_hcs, tr, te, y_exact, out),
        },
    }
    tmp = out / "results.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(out / "results.json")
    print(json.dumps({"N": args.n, "seed": args.seed, "models": result["models"], "label_noise": result["label_noise"]}), flush=True)


def stat(values: list[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.4f}\\pm{arr.std(ddof=1):.4f}"


def aggregate(args: argparse.Namespace) -> None:
    rows = []
    groups = {}
    for n in NS:
        records = []
        for seed in SEEDS:
            path = args.out / f"n{n}_seed{seed}" / "results.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            if not rec.get("completed"):
                raise RuntimeError(path)
            records.append(rec)
            for method in ("redq_hcs", "hcs"):
                for metric, value in rec["models"][method]["metrics"].items():
                    rows.append({"N": n, "seed": seed, "method": method, "metric": metric, "value": value})
        groups[str(n)] = {
            method: {
                metric: stat([r["models"][method]["metrics"][metric] for r in records])
                for metric in ("mae", "R(h)", "r2")
            }
            for method in ("redq_hcs", "hcs")
        }
        groups[str(n)]["label_noise_train_mse"] = stat([r["label_noise"]["train_mse"] for r in records])
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["N", "seed", "method", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "version": "table1-tfim-hcs-v1",
        "shots_per_input": SHOTS,
        "groups": groups,
        "scope": "Table-1 TFIM N=8/10/12/14 with same h_cs-style labels for RED-Q and trigonometric regression; exact held-out scoring.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Table 1 TFIM h_cs rerun",
        "",
        "Both RED-Q and Trig. ridge train on the same scalar h_cs-style finite-shot energy labels with T=32 shots/input. Test MAE/MSE use exact held-out final-layer energies.",
        "",
        "| N | RED-Q MAE | h_cs MAE | RED-Q MSE | h_cs MSE | train label MSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for n in NS:
        g = groups[str(n)]
        lines.append(
            f"| {n} | ${g['redq_hcs']['mae']}$ | ${g['hcs']['mae']}$ | "
            f"${g['redq_hcs']['R(h)']}$ | ${g['hcs']['R(h)']}$ | ${g['label_noise_train_mse']}$ |"
        )
    (args.out / "REPORT_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print("[aggregate] wrote summary.json and REPORT_zh.md", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--force", action="store_true")
    runp = sub.add_parser("run")
    runp.add_argument("--out", type=Path, required=True)
    runp.add_argument("--n", type=int, required=True, choices=NS)
    runp.add_argument("--seed", type=int, required=True, choices=SEEDS)
    runp.add_argument("--epochs", type=int, default=250)
    runp.add_argument("--device", default="cuda")
    runp.add_argument("--force", action="store_true")
    plan = sub.add_parser("run-plan")
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--epochs", type=int, default=250)
    plan.add_argument("--device", default="cuda")
    plan.add_argument("--force", action="store_true")
    agg = sub.add_parser("aggregate")
    agg.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "run":
        run(args)
    elif args.mode == "run-plan":
        for n in NS:
            for seed in SEEDS:
                ns = argparse.Namespace(**vars(args))
                ns.n = n
                ns.seed = seed
                run(ns)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
