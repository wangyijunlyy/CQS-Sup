"""Full-budget exact-label mask ablation on the existing three N6 L2 pools.

Compare to stored measured-label arms with identical initialization and fitting
rules. Exact-label results are an oracle supervision ablation, not new measured
evidence or a replacement for the paper's finite-shot comparison.
"""
from pathlib import Path
import sys
import os
import json
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import numpy as np
import torch
WORK=Path(__file__).resolve().parents[2]
ROOT=WORK/'world_model_surrogate'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from run_fspt8_fair import fit
from run_product_mask_control import make_masks,initial_state,build_model,score_terms
OUT=Path(__file__).resolve().parent/'evidence'


def main():
    torch.set_num_threads(2)
    rows=[]
    for seed in (45,46,47):
        with np.load(ROOT/f'results_nc_tfim_vqe_independent/datasets/chain6_l2_seed{seed}/dataset.npz') as a:
            x,exact=a['X'],a['exact_terms']
        initial=initial_state(seed)
        masks=make_masks(seed)
        for arm in ('actual','full'):
            folder=OUT/'label_precision'/f'seed{seed}_{arm}_exact'
            folder.mkdir(parents=True,exist_ok=True)
            model,training=fit(lambda:build_model(initial,masks[arm]),x[:2000],exact[:2000],
                  x[2000:2400],exact[2000:2400],seed,'cpu',folder,'product',300,min_epochs=0,patience=30)
            model.eval()
            with torch.no_grad():
                pred=model(torch.tensor(x[2400:],dtype=torch.float32))[0].numpy()
            np.savez_compressed(folder/'test_predictions.npz',prediction=pred,exact=exact[2400:])
            score=score_terms(exact[2400:],pred)
            for regime,metrics in [('exact',score),('measured_T10',json.loads((ROOT/f'results_review_masks/chain6_l2_seed{seed}_{arm}/results.json').read_text())['metrics'])]:
                row=dict(seed=seed,arm=arm,training_labels=regime,moment_mse=metrics['moments']['mse'],
                         energy_mae=metrics['energy']['-0.1']['mae'])
                rows.append(row)
                print('PRECISION',row,flush=True)
            (folder/'results.json').write_text(json.dumps(dict(seed=seed,arm=arm,labels='exact',training=training,metrics=score),indent=2))
            (OUT/'label_precision.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
