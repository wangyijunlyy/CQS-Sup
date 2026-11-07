"""Replay all stored test predictions in the original CUDA environment."""
import json
from pathlib import Path
import numpy as np
import torch
from benchmark_parallel import load
from parallel_models import CONFIGS

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
root = Path('.')
rows = []
for seed in range(1001, 1006):
    x = np.load(root/f'data/seed{seed}.npz')['X'][2400:]
    for name in ['cqs_r', 'query_gru', 'period_mlp']+list(CONFIGS):
        saved = np.load(root/f'evaluation/seed{seed}_{name}.npz')['prediction']
        model = load(name, seed, root, 'cuda')
        with torch.inference_mode():
            replay = model(torch.tensor(x, dtype=torch.float32, device='cuda')).cpu().numpy()
        error = float(np.max(abs(saved-replay)))
        assert error < 2e-5, (seed, name, error)
        rows.append(dict(seed=seed, name=name, max_error=error))
        print(seed, name, error, flush=True)
result = dict(passed=True, checkpoint_replays=len(rows),
    max_prediction_error=max(r['max_error'] for r in rows), rows=rows,
    device=torch.cuda.get_device_name(), torch=torch.__version__,
    batch_size=400, tf32=False)
(root/'evaluation/gpu_replay.json').write_text(json.dumps(result, indent=2)+'\n')
