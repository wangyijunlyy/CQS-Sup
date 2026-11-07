"""Train all frozen parallel configurations without inspecting test labels."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import random
import time
import numpy as np
import torch
from parallel_models import CONFIGS, build_model


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def fit(name, seed, dataset, output, device):
    dest = output/f'seed{seed}'/name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'fit.json').exists():
        print('COMPLETE ALREADY', seed, name, flush=True)
        return
    if (dest/'started.json').exists():
        raise RuntimeError(f'Preserve incomplete fit at {dest}, use a new output root')
    with np.load(dataset) as a:
        # Fitting and checkpoint selection never read the held-out target block.
        x = np.asarray(a['X'][:2400], dtype=np.float32)
        y = np.asarray(a['y'][:2400], dtype=np.float32)
    assert x.shape == (2400, 9) and y.shape == (2400, 12, 8)
    write(dest/'started.json', dict(seed=seed, name=name, configuration=CONFIGS[name]))
    random.seed(seed+1000)
    np.random.seed(seed+1000)
    torch.manual_seed(seed+1000)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed+1000)
    model = build_model(name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    xt, yt, xv, yv = [torch.tensor(v, device=device) for v in
                     (x[:2000], y[:2000], x[2000:], y[2000:])]
    generator = torch.Generator().manual_seed(seed+1000)
    best, selected, state, patience = float('inf'), None, None, 0
    history = []
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(1, 301):
        model.train()
        permutation = torch.randperm(len(xt), generator=generator).to(device)
        total = 0.
        for indices in permutation.split(32):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(xt[indices])-yt[indices]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite full-trajectory training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(indices)
        model.eval()
        with torch.no_grad():
            validation = float((model(xv)-yv).square().mean())
        row = dict(epoch=epoch, train_mse=total/len(xt), val_mse=validation)
        history.append(row)
        if validation < best:
            best, selected, patience = validation, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if epoch % 25 == 0:
            print('FIT', seed, name, epoch, 'val', validation, 'best', best, flush=True)
        if patience >= 30:
            break
    if state is None:
        raise RuntimeError('No finite validation checkpoint')
    if device == 'cuda':
        torch.cuda.synchronize()
    torch.save(state, dest/'selected.pt')
    write(dest/'history.json', history)
    write(dest/'fit.json', dict(seed=seed, name=name, configuration=CONFIGS[name],
        selected_epoch=selected, epochs_run=epoch, best_val_mse=best,
        seconds=time.perf_counter()-start,
        parameters=sum(p.numel() for p in model.parameters()),
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None,
        device=torch.cuda.get_device_name() if device == 'cuda' else device,
        torch=torch.__version__, training_labels='first2000 exact trajectories',
        selection_labels='next400 exact validation trajectories',
        test_evaluated=False, cap=300, patience=30))
    print('SAVED', seed, name, epoch, 'val', best, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tasks = [(seed, name) for seed in range(1001, 1006) for name in CONFIGS]
    for index, (seed, name) in enumerate(tasks):
        if index % args.shards == args.shard:
            fit(name, seed, args.data/f'seed{seed}.npz', args.out, args.device)
    print('SHARD COMPLETE', args.shard, flush=True)


if __name__ == '__main__':
    main()
