"""Measured-label development replay and label-free search-window timing."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
EXT = HERE.parent/'extensions_20261108'
sys.path[:0] = [str(WORK/'world_model_surrogate'), str(EXT)]
from calibration import select_correction, search, ACTIVE
from src.wmps.data.nc_tfim import TFIMTask
from src.wmps.models.periodic_redq import ProductMomentREDQ
from complete_fourier import SharedFeatureFourier


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def replay():
    out = HERE/'development'
    out.mkdir(exist_ok=True)
    rows = []
    for seed in (45,46,47):
        for j in (-.5,-.3,-.1,.1,.3,.5):
            for model in ('product','fourier'):
                source = (HERE.parents[1]/'evidence/chain_depth_recovery/local_calibration'/
                    f'seed{seed}_product_J{j:+.1f}' if model=='product' else
                    EXT/'chain'/f'seed{seed}'/f'local_J{j:+.1f}')
                for r, radius in enumerate((.8,.6,.4)):
                    with np.load(source/f'round{r}.npz') as a:
                        center, design = a['center'], a['design']
                        z = (design.reshape(256,22)[:,ACTIVE]-center.reshape(-1)[ACTIVE])/(radius/4)
                        y, p = a['design_energy'], a['prior_prediction']
                        old = a['correction_weights']
                    weights, selected = select_correction(z,y,p)
                    original, fixed = select_correction(z,y,p,alphas=(1.,))
                    np.testing.assert_allclose(original,old,rtol=1e-10,atol=1e-10)
                    rows.append(dict(seed=seed,J=j,model=model,round=r,
                        original_validation_mse=fixed['validation_mse'],
                        selected=selected, selection_uses_exact_energy=False))
                    if seed==45 and j==-.5 and r==0:
                        np.savez_compressed(out/f'pilot_{model}.npz',center=center,
                                            weights=original,radius=radius)
    write(out/'calibration_replay.json', dict(rows=rows,rounds=len(rows),
        alpha_counts={str(a):sum(r['selected']['alpha']==a for r in rows) for a in (0.,.5,1.)},
        added_shots=0, exact_targets_used=False))
    print('REPLAY',len(rows),'rounds',flush=True)


def pilot():
    torch.set_num_threads(2)
    task = TFIMTask(1,6,2,False,-1)
    p = ProductMomentREDQ(6,task.edges,2,False,32,1).double().eval()
    p.load_state_dict(torch.load(WORK/'world_model_surrogate/results_product_redq_independent/chain6_l2_seed45/periodic_redq.pt',map_location='cpu',weights_only=True))
    for parameter in p.parameters():
        parameter.requires_grad_(False)
    f = SharedFeatureFourier(EXT/'chain/seed45/global')
    coeff = torch.tensor(task.coefficients(-.5,-.5))
    funcs = dict(product=lambda x:p(x)[0]@coeff, fourier=lambda x:f(x)@coeff)
    rows = []
    for repeat in range(3):
        for name in (('product','fourier') if repeat%2==0 else ('fourier','product')):
            with np.load(HERE/f'development/pilot_{name}.npz') as a:
                center, weights, radius = a['center'],a['weights'],float(a['radius'])
            _,_,meta = search(funcs[name],center,radius,weights,1.,np.random.default_rng(770+repeat))
            rows.append(dict(model=name,repeat=repeat,seconds=meta['actual_seconds']))
            print('TIMING',rows[-1],flush=True)
    cap = max(10,int(np.ceil(1.5*max(r['seconds'] for r in rows))))
    write(HERE/'timing_cap.json',dict(search_seconds_per_round=cap,rows=rows,
        rule='ceil(1.5*slowest development alpha1 fixed100 block), minimum10seconds',
        threads=2,device='cpu',dtype='float64',cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        new_confirmation_data_used=False,exact_energy_used=False))
    print('FROZEN CAP',cap,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=('replay','pilot'))
    args=ap.parse_args()
    replay() if args.stage=='replay' else pilot()
