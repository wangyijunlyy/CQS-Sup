"""Full main-task joint fits. No new-model test scoring in this runner."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import sys,json,argparse,time,platform
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[2];PROJECT=WORK/'world_model_surrogate'
sys.path[:0]=[str(PROJECT),str(PROJECT/'scripts')]
from main_models import FSPTJoint,TFIMJoint,VQEJoint
from src.wmps.data.nc_tfim import TFIMTask
from run_fspt8_fair import fit


def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def cells():
 out=[]
 for n in (4,6,8,10):
  for k in (6,12,24):
   for seed in (42,43,44):
    out.append(dict(name=f'fspt_n{n}_k{k}_seed{seed}',task='fspt',n=n,k=k,seed=seed,epochs=300,patience=30,min_epochs=0,
     data=f'results_fspt_scaling/datasets/dataset_n{n}.npz',source=f'results_fspt_scaling/n{n}_k{k}_seed{seed}',
     product=f'results_remaining_comparisons/fspt_scaling_n{n}_k{k}_seed{seed}_redq_m'))
 for n in (8,10,12,14):
  for seed in (42,43,44):
   out.append(dict(name=f'tfim_n{n}_seed{seed}',task='tfim',n=n,seed=seed,epochs=250,patience=30,min_epochs=0,
    data=f'results_table1_hcs/datasets/table1_hcs_n{n}_T32.npz',source=f'results_table1_hcs/n{n}_seed{seed}',product=f'results_product_table1/n{n}_seed{seed}'))
 config=json.loads((PROJECT/'configs/nc_tfim_vqe_independent.json').read_text())
 for case in config['cases']:
  for seed in config['seeds']:
   name=f"{case['id']}_seed{seed}";shared=case['shared']
   out.append(dict(name=f'vqe_{name}',task='vqe',n=case['rows']*case['cols'],case=case,seed=seed,
    epochs=3000 if shared else 300,patience=100 if shared else 30,min_epochs=500 if shared else 0,
    data=f'results_nc_tfim_vqe_independent/datasets/{name}/dataset.npz',source=f'results_nc_tfim_vqe_independent/code/{name}',
    product=f'results_product_redq_independent/{name}',continuation=f'results_redq_continuation_independent/{name}',moment_gru=f'results_vqe_moment_gru_independent/{name}'))
 return out


def task_object(cell):
 c=cell['case'];return TFIMTask(c['rows'],c['cols'],c['layers'],c['shared'],1 if c['shared'] else -1)


def load_data(c):
 with np.load(PROJECT/c['data']) as a:
  x=a['X'].astype(np.float32)
  if c['task']=='tfim':y=a['y_hcs'][:,None].astype(np.float32)
  elif c['task']=='fspt':y=a['y'][:,:c['k']].astype(np.float32)
  else:y=a['measured_terms'].astype(np.float32)
 if c['task']=='vqe':
  nt,nv=c['case']['train'],c['case']['val'];tr=np.arange(nt);va=np.arange(nt,nt+nv);te=np.arange(nt+nv,len(x))
 else:
  with np.load(PROJECT/c['source']/'split.npz') as split:tr,va,te=[split[k] for k in ('train','val','test')]
 if len(set(np.concatenate((tr,va,te)).tolist()))!=len(x) or not np.array_equal(np.sort(np.concatenate((tr,va,te))),np.arange(len(x))):raise ValueError('Source split invalid')
 if (len(tr),len(va),len(te))!=((c['case']['train'],c['case']['val'],c['case']['test']) if c['task']=='vqe' else (2000,400,400)):raise ValueError('Source split sizes changed')
 if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Source data nonfinite')
 return x,y,tr,va,te


def model(c):
 if c['task']=='tfim':return TFIMJoint(c['n'])
 if c['task']=='fspt':return FSPTJoint(c['n'],c['k'])
 return VQEJoint(task_object(c))


def freeze():
 value=dict(version='fullmain66-v1',cells=cells(),rank=32,hidden=64,factor_initialization='constant1,normalizedFouriercoefficientsN(0,.02^2)',
  time_encoding='ell/24',Adam=dict(lr=.001,weight_decay=.0001,batch_size=32,gradient_clipping=None),
  checkpoint='minimum measured/exact validationMSE matchingtask;all66fitssavedbeforetest',
  VQE='direct100Adamupdates,lr.1,original20starts,clamp[-pi,pi],float64;allsearchessavedbeforeexactscore',
  scope='originalmainpools andsplits;notnewindependentconfirmation;epochsnotwalltimematched',extra_measurements=0,hardware_shots=0)
 path=HERE/'protocol.json'
 if path.exists():
  if json.loads(path.read_text())!=value:raise ValueError('Frozen main protocol changed')
 else:write(path,value)
 return value


def train(c):
 out=HERE/'fits'/c['name'];out.mkdir(parents=True,exist_ok=True)
 if (out/'training.json').exists():return
 if (out/'started.json').exists():raise RuntimeError('Preserve interrupted fit; no automatic overwrite')
 write(out/'started.json',dict(cell=c,started_at=time.time()))
 x,y,tr,va,te=load_data(c);np.savez_compressed(out/'split.npz',train=tr,val=va,test=te)
 def factory():
  m=model(c);torch.save(m.state_dict(),out/'initial.pt');return m
 start=time.perf_counter();m,detail=fit(factory,x[tr],y[tr],x[va],y[va],c['seed'],'cuda',out,'joint',c['epochs'],min_epochs=c['min_epochs'],patience=c['patience'])
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 with torch.no_grad():prediction=m(torch.tensor(x[va],device='cuda'))[0].cpu().numpy()
 np.save(out/'validation_prediction.npy',prediction)
 np.testing.assert_allclose(np.mean((prediction-y[va])**2),detail['best_val_mse'],rtol=2e-6,atol=2e-7)
 write(out/'training.json',dict(cell=c,training=detail,total_fit_seconds=seconds,test_evaluated=False,
  torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),device=torch.cuda.get_device_name()))
 print('FULL MAIN FIT SAVED',c['name'],detail,flush=True)


def preflight():
 manifest=cells()
 for c in manifest:
  out=HERE/'fits'/c['name'];meta=json.loads((out/'training.json').read_text());history=json.loads((out/'joint_history.json').read_text())
  if meta['cell']!=c or meta['test_evaluated'] or not(out/'joint.pt').is_file():raise ValueError('Fit identity mismatch')
  best=min(history,key=lambda r:r['val_mse'])
  if (best['epoch'],best['val_mse'])!=(meta['training']['best_epoch'],meta['training']['best_val_mse']):raise ValueError('Not validation selected')
  if len(history)>c['epochs'] or len(history)<c['min_epochs']:raise ValueError('Incorrect training cap')
 return len(manifest)


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;freeze()
 if a.freeze:write(HERE/'all_fits_saved.json',dict(fits=preflight(),before_test=True));return
 # Round-robin full matrix; long FSPT cells start first while all GPUs are idle.
 for c in cells()[a.worker::a.workers]:train(c)
 write(HERE/f'worker{a.worker}_complete.json',dict(completed=True))


if __name__=='__main__':main()
