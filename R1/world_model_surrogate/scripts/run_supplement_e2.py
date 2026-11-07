#!/usr/bin/env python3
"""Independent minimum-package E2 runner. Development outputs are not evidence.

Confirmation mode is for execution only after the coordinating parent freezes
the protocol. --print-spec inspects the protocol without generating any data.
Incomplete directories are preserved and refused; only verified complete runs
can be reused. No optimizer trajectory is silently restarted or overwritten.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from run_fspt8_fair import ROOT, parameter_steps, seed_all
from run_fspt_controls import PeriodMLP, ResetGRU
from src.wmps.models.world_model import ObsWorldModel

ARMS = ("cqs_r", "reset_gru", "period_mlp", "query_gru")


class QueryGRU(ObsWorldModel):
    """For output k, repeat its fixed query k times from an independent zero."""

    def forward(self, x):
        emb = self.encoder(x)
        outputs = []
        batch, horizon, width = emb.shape
        h = emb.new_zeros(batch, horizon, self.rssm.hidden)
        # Active queries k >= j share a batched call, never hidden state.
        # Removing the completed query avoids computing masked inactive cells.
        for j in range(horizon):
            query = emb[:, j:]
            h = self.rssm.gru(query.reshape(-1, width), h.reshape(-1, self.rssm.hidden))
            h = h.reshape(batch, horizon-j, self.rssm.hidden)
            outputs.append(h[:, 0])
            h = h[:, 1:]
        return self.predictor(torch.stack(outputs, 1)), None


def build_model(name):
    if name == "period_mlp":
        return PeriodMLP(10, 8)
    cls = {"cqs_r": ObsWorldModel, "reset_gru": ResetGRU, "query_gru": QueryGRU}[name]
    return cls(obs_dim=10, out_dim=8, encoder_hidden=128, rssm_hidden=128,
               predictor_hidden=128, stochastic=False)


def cell_cost(name, k, samples):
    if k < 1 or samples < 0 or name not in ARMS:
        raise ValueError("Invalid cell-count inputs")
    steps = 0 if name == "period_mlp" else k*(k+1)//2 if name == "query_gru" else k
    return samples * steps


def eligible_prefix(train_cells, budget):
    return 0 <= train_cells <= budget


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def make_spec(seed, mode):
    if mode not in ("confirmation", "development"):
        raise ValueError("Unknown mode")
    confirm = mode == "confirmation"
    if seed not in (range(1001, 1006) if confirm else range(42, 48)):
        raise ValueError("Confirmation seeds are 1001..1005; development seeds are 42..47")
    return dict(version=1, experiment="E2-minimum60", mode=mode, paper_evidence=confirm,
                N=8, K=12, period_normalizer=24, dataset_seed=seed,
                initialization_seed=seed+1000 if confirm else seed,
                batch_order_seed=seed+1000 if confirm else seed,
                split_sizes=[2000, 400, 400] if confirm else [40, 8, 8],
                epochs=300 if confirm else 2, patience=30, min_epochs=0,
                optimizer=dict(name="Adam", lr=0.001, weight_decay=0.0001, batch_size=32),
                recurrent_width=128, period_mlp_width=256, dtype="float32",
                arms=list(ARMS), delta_range=[0, 2], J_range=[0, 1],
                labels="exact local Z, physical qubit order, all 12 periods equally weighted",
                selection="validation MSE, strict improvement; no test selection",
                partition="contiguous disjoint train/validation/test from newly drawn controls",
                query="independent zero per k; repeat encoder([x,k/24]) exactly k times",
                query_execution="batch active queries; drop completed queries; K Python calls, no inactive cell rows",
                cell_budget="sample-weighted forward GRUCell updates during actual training batches; backward not counted",
                prefix_selection="epoch validations and final whole batch <= CQS-R actual total cells; initial only if no batch fits",
                main_comparison="CQS-R versus full-schedule Query-GRU test trajectory MAE",
                interpretation="state sharing and period conditioning; not proof of physical-order necessity")


def source_metadata():
    paths = [Path(__file__), ROOT/"scripts/run_fspt8_fair.py",
             ROOT/"scripts/run_fspt_controls.py", ROOT/"scripts/prepare_fspt_scaling.py",
             ROOT/"src/wmps/models/world_model.py", ROOT/"src/wmps/data/fspt_data.py"]
    versions = {}
    for package in ("numpy", "torch", "scipy", "qiskit", "qiskit-aer"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    documents = [ROOT.parent/"paper_iclr2027/experiments"/name for name in
                 ("supplementary-experiment-plan.md", "execution-lock.md")]
    return dict(source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in paths},
                protocol_document_hashes={p.name: sha256(p) for p in documents},
                packages=versions, python=platform.python_version())


def verify_complete(out, spec_hash):
    result_path = out/"results.json"
    if not result_path.exists():
        raise ValueError("Incomplete output directory; preserve it and use a fresh path")
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete" or result.get("spec_hash") != spec_hash:
        raise ValueError("Run is incomplete or its protocol/source/environment differs")
    if not result.get("artifacts"):
        raise ValueError("Missing artifact manifest")
    for name, expected in result["artifacts"].items():
        path = out/name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    return True


def generate_data(spec, dataset=None):
    total = sum(spec["split_sizes"])
    if dataset is not None:
        if spec["paper_evidence"]:
            raise ValueError("Old data are development-only")
        with np.load(dataset, allow_pickle=False) as data:
            x, y = data["X"][:total].copy(), data["y"][:total, :12].copy()
    else:
        rng = np.random.default_rng(spec["dataset_seed"])
        x = np.empty((total, 9), dtype=np.float32)
        x[:, 0] = rng.uniform(0, 2, total)
        x[:, 1:] = rng.uniform(0, 1, (total, 8))
        if spec["paper_evidence"]:
            from prepare_fspt_scaling import exact_targets
            y = exact_targets(x, 8, 12)
        else:
            # Synthetic development exercises I/O and training without new physics data.
            y = np.sin(x[:, None, 1:] + np.arange(1, 13)[None, :, None]/24).astype(np.float32)
    x, y = x.astype(np.float32), y.astype(np.float32)
    if x.shape != (total, 9) or y.shape != (total, 12, 8):
        raise ValueError("Unexpected data shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.max(np.abs(y)) > 1.00001:
        raise ValueError("Invalid data values")
    if len(np.unique(x, axis=0)) != total:
        raise ValueError("Duplicated controls")
    return x, y


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def fit_arm(name, x_train, y_train, x_val, y_val, seed, device, out,
            epochs=300, budget=None):
    seed_all(seed)
    model = build_model(name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt, xv, yv = [torch.as_tensor(a, dtype=torch.float32, device=device)
                      for a in (x_train, y_train, x_val, y_val)]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=32, shuffle=True, generator=generator)
    torch.save(model.state_dict(), out/f"{name}_initial.pt")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cells = steps = validation_cells = validation_passes = wait = 0
    train_seconds = validation_seconds = 0.0
    best = prefix = None
    history = []
    start = time.perf_counter()
    last_eligible = 0

    def validate(epoch, kind):
        nonlocal best, prefix, validation_cells, validation_passes, validation_seconds, wait, last_eligible
        model.eval()
        sync(device)
        tick = time.perf_counter()
        with torch.no_grad():
            val = ((model(xv)[0]-yv)**2).mean().item()
        sync(device)
        validation_seconds += time.perf_counter()-tick
        validation_cells += cell_cost(name, xv.shape[1], len(xv))
        validation_passes += 1
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        record = dict(epoch=epoch, kind=kind, optimizer_steps=steps, train_cells=cells, val_mse=val)
        history.append(record)

        def checkpoint(suffix):
            torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            selection=record), out/f"{name}{suffix}.pt")

        # Extra boundary evaluation must not affect full-schedule early stopping.
        if kind == "epoch":
            if best is None or val < best["val_mse"]:
                best, wait = record.copy(), 0
                checkpoint("")
            else:
                wait += 1
        if budget is not None and eligible_prefix(cells, budget):
            last_eligible = cells
            if prefix is None or val < prefix["val_mse"]:
                prefix = record.copy()
                checkpoint("_cell_budget")
        write_json(out/f"{name}_history.json", history)

    if budget is not None and budget < cell_cost(name, xt.shape[1], min(32, len(xt))):
        validate(0, "budget_initial")
    for epoch in range(1, epochs+1):
        total = 0.0
        for batch_index, (x, y) in enumerate(loader):
            model.train()
            sync(device)
            tick = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = ((model(x)[0]-y)**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            opt.step()
            sync(device)
            train_seconds += time.perf_counter()-tick
            steps += 1
            cells += cell_cost(name, x.shape[1], len(x))
            total += loss.item()*len(x)
            remaining = len(xt)-(batch_index+1)*32
            if budget is not None and remaining > 0 and cells <= budget:
                next_cost = cell_cost(name, x.shape[1], min(32, remaining))
                if cells+next_cost > budget:
                    validate(epoch, "budget_boundary")
        validate(epoch, "epoch")
        history[-1]["train_mse"] = total/len(xt)
        write_json(out/f"{name}_history.json", history)
        print(f"{name} epoch={epoch} val_mse={history[-1]['val_mse']:.7g} cells={cells}", flush=True)
        if wait >= 30:
            break
    torch.save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(),
                    epoch=epoch, train_cells=cells, optimizer_steps=steps), out/f"{name}_last.pt")
    info = dict(best=best, epochs_run=epoch, stop_reason="patience" if wait >= 30 else "cap",
                optimizer_steps=steps, train_cells=cells, validation_cells=validation_cells,
                training_python_grucell_calls=steps*xt.shape[1] if name != "period_mlp" else 0,
                validation_python_grucell_calls=validation_passes*xv.shape[1] if name != "period_mlp" else 0,
                inactive_cell_updates=0,
                validation_passes=validation_passes, train_seconds=train_seconds,
                validation_seconds=validation_seconds, fit_wall_seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in model.parameters()),
                peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                           (1 if sys.platform == "darwin" else 1024)),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None)
    if budget is not None:
        if prefix is None:
            raise RuntimeError("No eligible cell-budget checkpoint")
        info["cell_budget_prefix"] = dict(budget=budget, selected_train_cells=prefix["train_cells"],
                                          last_eligible_train_cells=last_eligible, selection=prefix,
                                          reached_budget=cells >= budget)
    return info


def score(y, pred):
    if y.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid predictions")
    def summary(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        variance = np.sum((a-a.mean())**2)
        return dict(mae=float(np.mean(np.abs(a-b))), mse=float(np.mean((a-b)**2)),
                    r2=float(1-np.sum((a-b)**2)/variance) if variance > 0 else None)
    return dict(overall=summary(y, pred), first_six=summary(y[:, :6], pred[:, :6]),
                last_six=summary(y[:, 6:], pred[:, 6:]),
                boundary=summary(y[:, :, [0, -1]], pred[:, :, [0, -1]]),
                bulk=summary(y[:, :, 1:-1], pred[:, :, 1:-1]),
                per_period_mae=np.abs(y-pred).mean(axis=(0, 2)).tolist())


def evaluate_checkpoint(name, checkpoint, x, y, splits, device, out, label):
    model = build_model(name).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    result = {}
    for split, indices in splits.items():
        sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            predictions = [model(torch.as_tensor(x[batch], device=device))[0].cpu().numpy()
                           for batch in np.array_split(indices, max(1, (len(indices)+31)//32))]
        sync(device)
        elapsed = time.perf_counter()-start
        pred = np.concatenate(predictions)
        np.save(out/f"pred_{label}_{split}.npy", pred)
        result[split] = dict(**score(y[indices], pred), prediction_seconds=elapsed,
                             inference_cells=cell_cost(name, 12, len(indices)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("confirmation", "development"), default="confirmation")
    parser.add_argument("--dataset", type=Path, help="Optional old pool, development only")
    parser.add_argument("--print-spec", action="store_true")
    args = parser.parse_args()
    spec = make_spec(args.seed, args.mode)
    if args.dataset and spec["paper_evidence"]:
        parser.error("--dataset is development-only")
    spec["data_source"] = "fresh_exact" if spec["paper_evidence"] else "old_pool" if args.dataset else "synthetic_smoke"
    if not spec["paper_evidence"]:
        spec["labels"] = "development only: " + spec["data_source"]
    spec["old_dataset_sha256"] = sha256(args.dataset) if args.dataset else None
    spec["source_metadata"] = source_metadata()
    spec["device"] = args.device
    spec_hash = json_hash(spec)
    if args.print_spec:
        print(json.dumps(dict(spec=spec, spec_hash=spec_hash), indent=2))
        return
    # Mode directories keep even mistakenly reused output roots visibly separate.
    out = args.out/args.mode/f"seed{args.seed}"
    if out.exists():
        verify_complete(out, spec_hash)
        print(f"Verified complete run: {out}")
        return
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    status = dict(status="running", spec_hash=spec_hash, paper_evidence=spec["paper_evidence"],
                  arms={name: "planned" for name in ARMS})
    write_json(out/"protocol.json", dict(spec=spec, spec_hash=spec_hash))
    write_json(out/"status.json", status)
    active = None
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        start = time.perf_counter()
        controls, y = generate_data(spec, args.dataset)
        np.savez_compressed(out/"dataset.npz", X=controls, y=y)
        nt, nv, ne = spec["split_sizes"]
        splits = dict(train=np.arange(nt), val=np.arange(nt, nt+nv), test=np.arange(nt+nv, nt+nv+ne))
        np.savez_compressed(out/"split.npz", **splits)
        np.save(out/"control_visibility_mask.npy", np.ones((8, 9), dtype=bool))
        write_json(out/"protocol.json", dict(spec=spec, spec_hash=spec_hash,
                   dataset_sha256=sha256(out/"dataset.npz"),
                   split_sha256=sha256(out/"split.npz"),
                   control_visibility_mask_sha256=sha256(out/"control_visibility_mask.npy")))
        write_json(out/"data_metadata.json", dict(dataset_sha256=sha256(out/"dataset.npz"),
                   split_sha256=sha256(out/"split.npz"), spec_hash=spec_hash,
                   rng="numpy.default_rng(PCG64); delta then J; independent of model/batch generators",
                   data_seed=args.seed, generation_seconds=time.perf_counter()-start,
                   source=spec["data_source"], old_path=str(args.dataset.resolve()) if args.dataset else None))
        x = parameter_steps(controls, 12, 24)
        models = {}
        # Finish all validation-based selections before any test evaluation.
        for active in ARMS:
            status["arms"][active] = "running"
            write_json(out/"status.json", status)
            budget = models["cqs_r"]["train_cells"] if active == "query_gru" else None
            models[active] = fit_arm(active, x[splits["train"]], y[splits["train"]],
                                     x[splits["val"]], y[splits["val"]], spec["initialization_seed"],
                                     args.device, out, spec["epochs"], budget)
            write_json(out/f"{active}_training.json", dict(spec_hash=spec_hash, **models[active]))
            status["arms"][active] = "trained"
            write_json(out/"status.json", status)
        for active in ARMS:
            models[active]["metrics"] = evaluate_checkpoint(active, out/f"{active}.pt", x, y,
                                                            splits, args.device, out, active)
            if active == "query_gru":
                models[active]["cell_budget_prefix"]["metrics"] = evaluate_checkpoint(
                    active, out/"query_gru_cell_budget.pt", x, y, splits, args.device, out, "query_gru_cell_budget")
            write_json(out/f"{active}_metrics.json", models[active])
            status["arms"][active] = "complete"
            write_json(out/"status.json", status)
        center = y[splits["train"]].mean(axis=0)
        baseline = {}
        for split, indices in splits.items():
            pred = np.broadcast_to(center, y[indices].shape).copy()
            np.save(out/f"pred_train_mean_{split}.npy", pred)
            baseline[split] = score(y[indices], pred)
        status["status"] = "complete"
        write_json(out/"status.json", status)
        artifacts = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
        result = dict(status="complete", spec_hash=spec_hash, protocol=spec, models=models,
                      train_mean=baseline, artifacts=artifacts, environment=dict(platform=platform.platform(),
                      device=torch.cuda.get_device_name() if args.device == "cuda" else platform.processor(),
                      torch_threads=torch.get_num_threads(), argv=sys.argv),
                      caveats=["Cell counts are sample-weighted forward updates, not FLOPs or wall time.",
                               "Process peak RSS is cumulative; CUDA peak is per fitted arm.",
                               "Query has different period conditioning and shares no cross-period state."])
        write_json(out/"results.json", result)
        print(f"Complete: {out}; paper_evidence={spec['paper_evidence']}")
    except BaseException as exc:
        status["status"] = "failed"
        if active is not None:
            status["arms"][active] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        write_json(out/"status.json", status)
        raise


if __name__ == "__main__":
    main()
