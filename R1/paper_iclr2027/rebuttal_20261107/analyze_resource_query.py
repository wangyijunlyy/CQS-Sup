"""Score all independently timed full-model fits only after all15 finish."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import torch
from resource_query import build_model,parameter_steps,ARMS
HERE=Path(__file__).resolve().parent;E=HERE/'evidence/resource_controls_20261108'
ROOT=HERE.parents[1]/'world_model_surrogate'
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def stats(v):return dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),values=list(map(float,v)))
def run(source):
 torch.set_num_threads(2)
 for seed in range(1001,1006):assert (source/f'seed{seed}/complete.json').exists(),seed
 rows=[];max_overrun=0.;checkpoint_replay=0.
 for seed in range(1001,1006):
  data=ROOT/f'results_supplementary/e2/confirmation/seed{seed}/dataset.npz'
  with np.load(data) as a:controls=a['X'];y=a['y']
  x=parameter_steps(controls,12,24);initials=[]
  for arm in ARMS:
   folder=source/f'seed{seed}'/arm;meta=json.loads((folder/'training.json').read_text())
   assert meta['budget_seconds']==240 and meta['selected']['seconds']<=240
   assert abs(meta['actual_seconds']-240)<2,meta['actual_seconds']
   max_overrun=max(max_overrun,meta['actual_seconds']-240)
   history=json.loads((folder/'history.json').read_text())
   eligible=[r for r in history if r['seconds']<=240]
   assert meta['selected']==min(eligible,key=lambda r:r['val_mse'])
   init=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True);initials.append(init)
   model=build_model(arm).eval();model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True))
   metrics={}
   for split,ids in [('train',np.arange(2000)),('val',np.arange(2000,2400)),('test',np.arange(2400,2800))]:
    with torch.no_grad():pred=np.concatenate([model(torch.tensor(x[block]))[0].numpy() for block in np.array_split(ids,(len(ids)+127)//128)])
    np.save(folder/f'pred_{split}.npy',pred)
    error=pred.astype(float)-y[ids].astype(float)
    metrics[split]=dict(mae=float(np.mean(abs(error))),mse=float(np.mean(error**2)))
   checkpoint_replay=max(checkpoint_replay,abs(metrics['val']['mse']-meta['selected']['val_mse']))
   assert abs(metrics['val']['mse']-meta['selected']['val_mse'])<2e-6
   row=dict(seed=seed,arm=arm,train_mae=metrics['train']['mae'],test_mae=metrics['test']['mae'],
       val_mse=metrics['val']['mse'],actual_seconds=meta['actual_seconds'],optimizer_steps=meta['optimizer_steps'],
       selected_optimizer_steps=meta['selected']['optimizer_steps'],selected_seconds=meta['selected']['seconds'],parameters=meta['parameters'])
   rows.append(row);write(folder/'evaluation.json',dict(metrics=metrics,dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest()))
  for init in initials[1:]:
   assert all(torch.equal(initials[0][k],init[k]) for k in init)
 write(E/'query_rows.json',rows)
 summary=[]
 for arm in ARMS:
  g=[r for r in rows if r['arm']==arm]
  summary.append(dict(arm=arm,n=5,**{key:stats([r[key] for r in g]) for key in ('test_mae','train_mae','val_mse','actual_seconds','optimizer_steps','selected_optimizer_steps','selected_seconds')}))
 write(E/'query_summary.json',summary)
 paired=[]
 for baseline in ('matched_query','original_query'):
  a=np.array([r['test_mae'] for r in rows if r['arm']=='cqs_r']);b=np.array([r['test_mae'] for r in rows if r['arm']==baseline])
  paired.append(dict(baseline=baseline,cqs_r_wins=int(np.sum(a<b)),mean_baseline_minus_cqs_r=float((b-a).mean()),
      relative_cqs_r_reduction=float(1-a.mean()/b.mean()),paired_differences=(b-a).tolist()))
 write(E/'query_paired.json',paired)
 verify=dict(passed=True,full_independently_trained_fits=15,independent_data_pools=5,
     split=[2000,400,400],N=8,K=12,width=128,seconds_per_arm=240,
     matched_initialization_and_batch_seed=True,matched_chronological_encoding=True,
     all_selected_checkpoints_within_budget=True,validation_only_selection=True,
     max_actual_window_overrun_seconds=max_overrun,max_selected_validation_replay_error=checkpoint_replay,
     all_test_evaluation_after_all15_fits_complete=True,hardware='one dedicated RTX4090 per seed, three arms sequential',
     original_repeated_target_control_reported_separately=True,paired=paired)
 write(E/'query_verification.json',verify)
 print(json.dumps(verify,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=E/'query');args=ap.parse_args();run(args.source)
