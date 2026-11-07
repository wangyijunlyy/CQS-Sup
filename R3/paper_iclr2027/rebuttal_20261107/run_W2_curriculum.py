"""Full-model 12-to-24-period continuation on the original paired FSPT splits.

All candidates are selected by full-24-period validation MSE. Test predictions
are produced only after all fits finish. Existing checkpoints are preserved.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
import json
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'world_model_surrogate'
sys.path.insert(0, str(ROOT/'scripts'))
from run_fspt_controls import factory, split_indices
from run_fspt8_fair import parameter_steps, seed_all

DEST = HERE/'evidence/W2_curriculum'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def fit(n, seed):
    torch.set_num_threads(2)
    folder = DEST/f'n{n}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'training.json').exists():
        print('REUSE', n, seed, flush=True)
        return
    source = ROOT/f'results_fspt_scaling/n{n}_k12_seed{seed}'
    data = np.load(ROOT/f'results_fspt_scaling/datasets/dataset_n{n}.npz')
    tr, va, te = split_indices(2800, 2000, 400, seed)
    with np.load(source/'split.npz') as split:
        for name, ids in zip(('train', 'val', 'test'), (tr, va, te)):
            np.testing.assert_array_equal(ids, split[name])
    # No test targets enter the fitting or checkpoint-selection path.
    x = parameter_steps(data['X'], 24, 24)
    xt, xv = torch.from_numpy(x[tr]), torch.from_numpy(x[va])
    yt, yv = torch.from_numpy(data['y'][tr]), torch.from_numpy(data['y'][va])
    seed_all(seed)
    model = factory('redq', n)
    model.load_state_dict(torch.load(source/'redq.pt', map_location='cpu', weights_only=True))
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt, yt),
        batch_size=32, shuffle=True, generator=generator)
    protocol = dict(N=n, K=24, seed=seed, width=128, source_checkpoint=str(source/'redq.pt'),
        stages=[dict(K=12, epoch_cap=300, selection='original validation MSE on periods 1:12'),
                dict(K=24, epoch_cap=300, selection='validation MSE on all periods 1:24')],
        optimizer='Adam lr=.001 weight_decay=.0001 batch32; fresh optimizer at stage transition',
        patience=30, loss='uniform MSE across controls, all 24 periods and all observables',
        period_encoding='j/24 unchanged', split_sizes=[2000,400,400],
        selection='minimum full-24 validation MSE including stage-2 epoch 0',
        scope='three paired split/initialization runs on one original pool per N',
        no_new_quantum_labels=True)
    write(folder/'protocol.json', protocol)
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        initial = float((model(xv)[0]-yv).square().mean())
    best, best_epoch, wait = initial, 0, 0
    state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    history = [dict(epoch=0, val_mse=initial, seconds=0.)]
    for epoch in range(1,301):
        model.train()
        total = 0.
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = (model(xb)[0]-yb).square().mean()
            loss.backward()
            opt.step()
            total += float(loss.detach())*len(xb)
        model.eval()
        with torch.no_grad():
            val = float((model(xv)[0]-yv).square().mean())
        history.append(dict(epoch=epoch, train_mse=total/len(xt), val_mse=val,
                            seconds=time.perf_counter()-start))
        if val < best:
            best, best_epoch, wait = val, epoch, 0
            state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            wait += 1
        if epoch % 25 == 0:
            write(folder/'progress.json', dict(epoch=epoch, best_epoch=best_epoch,
                best_val_mse=best, seconds=time.perf_counter()-start))
            print(n, seed, epoch, 'best_val', best, flush=True)
        if wait >= 30:
            break
    torch.save(state, folder/'selected.pt')
    write(folder/'history.json', history)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        for part, ids in (('train',tr), ('val',va)):
            pred = torch.cat([model(torch.from_numpy(x[ids[i:i+128]]))[0]
                              for i in range(0,len(ids),128)]).numpy()
            np.save(folder/f'pred_{part}.npy', pred)
    write(folder/'training.json', dict(initial_val_mse=initial, best_val_mse=best,
        best_epoch=best_epoch, epochs_run=epoch, stage2_seconds=time.perf_counter()-start,
        parameters=sum(p.numel() for p in model.parameters()),
        source_training=json.loads((source/'results.json').read_text())['models']['redq']['training']))
    print('COMPLETE TRAINING', n, seed, best, flush=True)


def queue():
    DEST.mkdir(parents=True, exist_ok=True)
    write(DEST/'protocol.json', dict(N=[8,10], seeds=[42,43,44], stage2_cap=300,
        stage2_patience=30, selection='full-24 validation MSE, including warm start',
        comparison='original cap300/cap600 CQS-R and per-output validation-tuned NC',
        test_policy='no test scoring until all six fits complete', state='running'))
    def worker(n):
        for seed in (42,43,44):
            with (DEST/f'n{n}_seed{seed}.log').open('a') as stream:
                subprocess.run([sys.executable,str(Path(__file__)), '--n',str(n),'--seed',str(seed)],
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker,n) for n in (8,10)]
        for future in futures:
            future.result()
    write(DEST/'queue_status.json', dict(state='all_training_complete_pending_test_evaluation',fits=6))


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--n',type=int,choices=(8,10))
    ap.add_argument('--seed',type=int,choices=(42,43,44))
    args=ap.parse_args()
    if args.n is None:
        queue()
    else:
        fit(args.n,args.seed)
