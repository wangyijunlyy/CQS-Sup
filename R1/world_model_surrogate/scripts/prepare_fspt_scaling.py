"""Exact control-only pools using the established Aer FSPT circuit builder."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from run_fspt8_fair import ROOT
from src.wmps.data.fspt_data import generate_fspt_trajectories_gpu


def exact_targets(controls, n, horizon):
    if controls.shape[1] != n+1 or n < 3 or horizon < 1:
        raise ValueError("Invalid controls or dimensions")
    _, y = generate_fspt_trajectories_gpu(controls, n, horizon, T_shadow=0,
                                         device="CPU", verbose=True)
    if not np.isfinite(y).all() or np.max(np.abs(y)) > 1.00001:
        raise ValueError("Invalid exact expectation values")
    return y


def prepare(folder, n, total=2800, horizon=24):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f"dataset_n{n}.npz"
    meta_path = path.with_suffix(".json")
    spec = dict(N=n, K=horizon, total=total, dataset_seed=42,
                delta_range=[0.0, 2.0], J_range=[0.0, 1.0],
                labels="exact statevector local Z, physical qubit order, no observations")
    if path.exists() or meta_path.exists():
        if not path.exists() or not meta_path.exists():
            raise ValueError("Incomplete cached dataset; preserve and investigate")
        meta = json.loads(meta_path.read_text())
        if meta["spec"] != spec or meta["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Cached dataset differs from requested protocol")
        return path
    rng = np.random.default_rng(42)
    x = np.empty((total, n+1), dtype=np.float32)
    x[:, 0] = rng.uniform(0, 2, size=total)
    x[:, 1:] = rng.uniform(0, 1, size=(total, n))
    y = exact_targets(x, n, horizon)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, X=x, y=y)
    temporary.replace(path)
    meta_path.write_text(json.dumps(dict(spec=spec, sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, choices=[4, 6, 8, 10], required=True)
    args = ap.parse_args()
    print(prepare(args.out, args.n), flush=True)
