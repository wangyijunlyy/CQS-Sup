"""Same-GPU host-to-host latency selection, followed by held-out evaluation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from parallel_models import CONFIGS, build_model, select_under_budget
from run_parallel import write
from historical_world_model import ObsWorldModel


class QueryGRU(ObsWorldModel):
    def forward(self, x):
        emb = self.encoder(x)
        batch, horizon, width = emb.shape
        h = emb.new_zeros(batch, horizon, self.rssm.hidden)
        outputs = []
        for j in range(horizon):
            query = emb[:, j:]
            h = self.rssm.gru(query.reshape(-1, width), h.reshape(-1, self.rssm.hidden))
            h = h.reshape(batch, horizon-j, self.rssm.hidden)
            outputs.append(h[:, 0])
            h = h[:, 1:]
        return self.predictor(torch.stack(outputs, 1)), None


class PeriodMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 256), nn.SiLU(), nn.Linear(256, 256),
            nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 8))

    def forward(self, x):
        return self.net(x), None


class Historical(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.base = (PeriodMLP() if name == 'period_mlp' else
            (QueryGRU if name == 'query_gru' else ObsWorldModel)(
                obs_dim=10, out_dim=8, encoder_hidden=128, rssm_hidden=128,
                predictor_hidden=128, stochastic=False))
        self.register_buffer('periods', torch.arange(1, 13, dtype=torch.float32)/24)

    def forward(self, raw):
        x = torch.cat((raw[:, None].expand(-1, 12, -1),
            self.periods[None, :, None].expand(len(raw), -1, -1)), -1)
        return self.base(x)[0]


def load(name, seed, root, device):
    if name in CONFIGS:
        model = build_model(name)
        state = torch.load(root/'output'/f'seed{seed}'/name/'selected.pt',
                           map_location='cpu', weights_only=True)
        model.load_state_dict(state)
    else:
        model = Historical(name)
        state = torch.load(root/'historical'/f'seed{seed}'/f'{name}.pt',
                           map_location='cpu', weights_only=True)
        model.base.load_state_dict(state.get('state_dict', state))
    return model.to(device).eval()


def prediction(model, raw, device):
    return model(torch.as_tensor(raw, dtype=torch.float32, device=device)).cpu().numpy()


def timing(model, raw, device):
    def one():
        return prediction(model, raw, device)
    with torch.inference_mode():
        for _ in range(10):
            one()
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = []
        for _ in range(40):
            tick = time.perf_counter()
            one()
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed.append((time.perf_counter()-tick)*1000)
    return dict(median_ms=float(np.median(elapsed)), p90_ms=float(np.quantile(elapsed, .9)),
                repetitions=40, raw_ms=elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    root = args.root
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    names = ['cqs_r', 'query_gru', 'period_mlp']+list(CONFIGS)
    for seed in range(1001, 1006):
        for name in CONFIGS:
            assert (root/'output'/f'seed{seed}'/name/'fit.json').exists(), (seed, name)
    output = root/'evaluation'
    output.mkdir(exist_ok=True)
    timings, selections = [], []
    # Phase one: validation labels and label-free timings only.
    for seed in range(1001, 1006):
        with np.load(root/'data'/f'seed{seed}.npz') as a:
            xv = a['X'][2000:2400].astype(np.float32)
            yv = a['y'][2000:2400]
        rotated = names[(seed-1001) % len(names):]+names[:(seed-1001) % len(names)]
        for name in rotated:
            model = load(name, seed, root, args.device)
            with torch.inference_mode():
                pred = prediction(model, xv, args.device)
            val_mse = float(np.mean((pred-yv)**2))
            if name in CONFIGS:
                old = json.loads((root/'output'/f'seed{seed}'/name/'fit.json').read_text())
                assert abs(val_mse-old['best_val_mse']) < 1e-5
            row = dict(seed=seed, name=name, val_mse=val_mse,
                       parameters=sum(p.numel() for p in model.parameters()))
            for batch in (1, 400):
                row[f'batch{batch}'] = timing(model, xv[:batch], args.device)
            timings.append(row)
            del model
        group = [r for r in timings if r['seed'] == seed]
        for batch in (1, 400):
            reference = next(r for r in group if r['name'] == 'cqs_r')[f'batch{batch}']['median_ms']
            for family in ('joint_mlp', 'transformer'):
                candidates = [dict(name=r['name'], val_mse=r['val_mse'],
                    latency_ms=r[f'batch{batch}']['median_ms'])
                    for r in group if r['name'].startswith(family)]
                selected = select_under_budget(candidates, reference)
                unconstrained = min(candidates, key=lambda r: r['val_mse'])
                selections.append(dict(seed=seed, batch=batch, family=family,
                    reference_ms=reference, constrained=selected, unconstrained=unconstrained))
        write(output/'timings.json', timings)
        print('TIMED', seed, flush=True)
    write(output/'selection_frozen.json', dict(selections=selections,
        selection_inputs='validation MSE and inference timings only', test_scoring_started=False,
        device=torch.cuda.get_device_name() if args.device == 'cuda' else args.device,
        torch=torch.__version__, timing_boundary='host raw controls to host full12x8 predictions'))
    # Phase two: all configuration and latency-cap choices are frozen first.
    rows = []
    for seed in range(1001, 1006):
        with np.load(root/'data'/f'seed{seed}.npz') as a:
            xt = a['X'][2400:].astype(np.float32)
            yt = a['y'][2400:]
        for name in names:
            model = load(name, seed, root, args.device)
            with torch.inference_mode():
                pred = prediction(model, xt, args.device)
            error = pred-yt
            row = dict(seed=seed, name=name, test_mae=float(np.mean(abs(error))),
                test_mse=float(np.mean(error**2)),
                per_period_mae=np.mean(abs(error), axis=(0, 2)).tolist())
            rows.append(row)
            np.savez_compressed(output/f'seed{seed}_{name}.npz', prediction=pred, exact=yt)
            del model
        print('SCORED', seed, flush=True)
    write(output/'test_rows.json', rows)
    summary = {}
    def stats(values):
        return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)), values=values)
    for name in names:
        group = [r for r in rows if r['name'] == name]
        summary[name] = stats([r['test_mae'] for r in group])
    constrained = []
    for batch in (1, 400):
        for family in ('joint_mlp', 'transformer'):
            for mode in ('constrained', 'unconstrained'):
                group = [s for s in selections if s['batch'] == batch and s['family'] == family]
                values, choices = [], []
                for s in group:
                    choice = s[mode]
                    choices.append(choice['name'] if choice else None)
                    if choice:
                        values.append(next(r['test_mae'] for r in rows
                                           if r['seed'] == s['seed'] and r['name'] == choice['name']))
                constrained.append(dict(batch=batch, family=family, mode=mode,
                    eligible_datasets=len(values), selected=choices,
                    test_mae=stats(values) if len(values) == 5 else None))
    write(output/'summary.json', dict(all_configurations=summary, selected=constrained,
        timing_device=torch.cuda.get_device_name() if args.device == 'cuda' else args.device,
        independent_datasets=5, full_fits=30, hardware_shots=0))
    print(json.dumps(constrained, indent=2), flush=True)


if __name__ == '__main__':
    main()
