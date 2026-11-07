"""Current dense implementation microbenchmarks; no accuracy scaling claims."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_evidence import ROOT,OUT,TFIMTask
from src.wmps.models.periodic_redq import causal_parameter_masks,ProductMomentREDQ


def main():
    torch.set_num_threads(2)
    rows=[]
    for n in (6,10,14,20,30):
        task=TFIMTask(1,n,2,False,-1)
        construction=[]
        for repeat in range(6):
            t=time.perf_counter()
            mask=causal_parameter_masks(n,task.edges,2,False)
            elapsed=time.perf_counter()-t
            if repeat:
                construction.append(elapsed)
        model=ProductMomentREDQ(n,task.edges,2,False,32,1).eval()
        x=torch.tensor(np.random.default_rng(90).uniform(-np.pi,np.pi,(32,2,task.block)),dtype=torch.float32)
        seconds=[]
        with torch.inference_mode():
            for _ in range(3):model(x)
            for _ in range(10):
                t=time.perf_counter()
                pred=model(x)[0]
                seconds.append(time.perf_counter()-t)
                assert pred.shape==(32,task.block)
        active=int(mask.sum())
        rows.append(dict(n=n,layers=2,rank=32,Q=1,batch=32,device='cpu',dtype='float32',threads=2,
                         mask_median_seconds=float(np.median(construction)),
                         inference_median_seconds=float(np.median(seconds)),
                         actual_allocated_parameters=sum(p.numel() for p in model.parameters()),
                         hypothetical_active_only_slots=active*32*3+task.block*32+task.block,
                         active_coordinates=active,dense_coordinates=mask.size,
                         scope='full existing model implementation; random parameters for cost only; no fitting or accuracy result at larger N'))
    fit=[]
    for seed in (45,46,47):
        for arm in ('actual','full','random'):
            a=json.loads((ROOT/f'results_review_masks/chain6_l2_seed{seed}_{arm}/results.json').read_text())
            fit.append(dict(seed=seed,arm=arm,recorded_fit_seconds=a['training']['training_seconds'],
                            recorded_inference_seconds_400=a['inference_seconds_400_controls'],
                            epochs_run=a['training']['epochs_run'],allocated_parameters=a['allocation']['allocated_parameters']))
    (OUT/'cost_audit.json').write_text(json.dumps(dict(dense_microbenchmarks=rows,original_N6_fits=fit,
                    interpretation='Masks restrict the hypothesis class but do not reduce dense allocation or contraction work. Historical fit timings are not cross-host speed evidence.'),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
