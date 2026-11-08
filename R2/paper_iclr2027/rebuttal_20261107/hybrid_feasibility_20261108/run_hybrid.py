"""Frozen actual-time feasibility fits; test predictions only after all20 fits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
from pathlib import Path
import sys,json,time,argparse,platform
import numpy as np
import torch
from models import build,ARMS
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[2]
SOURCE=WORK/'world_model_surrogate/results_supplementary/e2/confirmation'
OUT=HERE/'pools';SEEDS=tuple(range(1001,1006));BUDGET=120.


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def protocol():
    return dict(stage='architecture-level product recurrent fusion feasibility on reused independent pools',n=8,horizon=12,
        seeds=SEEDS,data_source='world_model_surrogate/results_supplementary/e2/confirmation/seed*/dataset.npz',
        shapes=dict(X=[2800,9],y=[2800,12,8]),split=[2000,400,400],labels='exact physical-qubit localZ expectations',
        arms=ARMS,hybrid=dict(rank=32,hidden=64,parameters=108872,update='q=maskedproduct;z=z*q;h=GRU(q-1,z-1,t/24);y=linear(z-1,h)',
             masks='fixed union over all12prefix support masks;period-specific factors;GRU shared across time andobservables'),
        recurrent=dict(width=115,parameters=108913),product=dict(original_ProductFSPT_rank=32,parameters=542304),
        additive='same hybrid parameters andinitialweights;spatial1+sum(g-1),temporal additive accumulation',
        factor_initialization='constant1;cos/sin normalstd.02 for both product architectures',
        Adam_lr=.001,weight_decay=.0001,gradient_clip_norm=1.,batch=32,loss='uniform momentMSE',
        seconds_per_arm=BUDGET,selection='best validationMSE checkpoint completed within120s;initialeligible at0',
        time_includes='training,validation,andeligibleCPUcheckpointcopy',warmup='3batches+validation,restore initial model/optimizer/batchRNG',
        schedule='cyclicarmorder bydataset',truth_gate='all20fits saved and audited before any newmodel testprediction',
        primary_metrics=['testMAE','testMSE','first6MAE','last6MAE'],statistics='poolmeans andpaired95%tCI over5pools',
        feasibility='finite stable fit;bothpaths active;exactprefixreuse;errorbelowtrainmean baseline',
        new_data=0,new_measurements=0,hardware_shots=0,scope='feasibilitydevelopment;not newindependentconfirmation')


def freeze():
    path=HERE/'protocol.json';value=json.loads(json.dumps(protocol()))
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Frozen hybrid protocol changed')
    else:write(path,value)


def data(seed):
    with np.load(SOURCE/f'seed{seed}/dataset.npz') as a:x,y=a['X'],a['y']
    if x.shape!=(2800,9) or y.shape!=(2800,12,8) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Wrong physical control pool')
    if len(np.unique(x,axis=0))!=2800 or abs(y).max()>1.00001:raise ValueError('Invalid unique bounded labels')
    return x,y


def sync():torch.cuda.synchronize()


def fit(arm,seed,x,y):
    folder=OUT/f'seed{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'training.json').exists():return
    if (folder/'started.json').exists():raise ValueError('Preserve incomplete fit')
    write(folder/'started.json',dict(seed=seed,arm=arm,budget=BUDGET))
    model=build(arm,seed).cuda();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(initial,folder/'initial.pt')
    xt=torch.tensor(x[:2000],device='cuda');yt=torch.tensor(y[:2000],device='cuda')
    xv=torch.tensor(x[2000:2400],device='cuda');yv=torch.tensor(y[2000:2400],device='cuda')
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);tick=time.perf_counter()
    for _ in range(3):
        opt.zero_grad(set_to_none=True);loss=(model(xt[:32])[0]-yt[:32]).square().mean();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    with torch.no_grad():model(xv)
    sync();warmup=time.perf_counter()-tick
    model.load_state_dict(initial);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed+1000);model.eval()
    with torch.no_grad():initial_val=float((model(xv)[0]-yv).square().mean())
    sync();best=initial_val;state=initial;selected=dict(epoch=0,seconds=0.,steps=0,val_mse=initial_val)
    history=[selected.copy()];steps=0;epochs=0;sync();start=time.perf_counter()
    while time.perf_counter()-start<BUDGET:
        epochs+=1;order=torch.randperm(2000,generator=generator);total=0.;seen=0
        for i in range(0,2000,32):
            if time.perf_counter()-start>=BUDGET:break
            ids=order[i:i+32].to('cuda');model.train();opt.zero_grad(set_to_none=True)
            loss=(model(xt[ids])[0]-yt[ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError((arm,seed,'nonfinite loss'))
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError((arm,seed,'nonfinite gradient'))
            opt.step();sync();total+=float(loss)*len(ids);seen+=len(ids);steps+=1
        if seen<2000:break
        model.eval()
        with torch.no_grad():prediction=model(xv)[0];val=float((prediction-yv).square().mean())
        if not np.isfinite(val):raise FloatingPointError('Nonfinite validation')
        sync();candidate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if val<best else None
        elapsed=time.perf_counter()-start;row=dict(epoch=epochs,seconds=elapsed,steps=steps,train_mse=total/seen,val_mse=val)
        history.append(row)
        if elapsed<=BUDGET and val<best:best=val;state=candidate;selected=row.copy()
    sync();elapsed=time.perf_counter()-start
    model.load_state_dict(state);model.eval()
    with torch.no_grad():prediction=model(xv)[0].cpu().numpy()
    torch.save(state,folder/'selected.pt');np.save(folder/'validation_prediction.npy',prediction)
    write(folder/'history.json',history)
    write(folder/'training.json',dict(arm=arm,seed=seed,budget_seconds=BUDGET,actual_seconds=elapsed,selected=selected,
          epochs_started=epochs,optimizer_steps=steps,parameters=sum(p.numel() for p in model.parameters()),
          initial_validation_mse=initial_val,warmup_seconds_excluded=warmup,setup_excluded=True,test_evaluated=False,
          device=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,python=platform.python_version()))
    print('HYBRID FIT SAVED',seed,arm,selected['epoch'],best,elapsed,flush=True)


def preflight():
    count=0
    for seed in SEEDS:
        if not (OUT/f'seed{seed}/complete.json').exists():raise ValueError('Incomplete pool fits')
        for arm in ARMS:
            b=OUT/f'seed{seed}'/arm;meta=json.loads((b/'training.json').read_text())
            if meta['selected']['seconds']>BUDGET or meta['test_evaluated'] or (meta['seed'],meta['arm'])!=(seed,arm):raise ValueError('Invalid checkpoint eligibility/identity')
            if not (b/'selected.pt').exists() or np.load(b/'validation_prediction.npy').shape!=(400,12,8):raise ValueError('Incomplete fitted model')
            count+=1
    return count


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=SEEDS);ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;freeze()
    if a.freeze:write(HERE/'all_fits_saved.json',dict(fits=preflight(),before_new_test_scoring=True));return
    x,y=data(a.seed);offset=(a.seed-1001)%4;order=ARMS[offset:]+ARMS[:offset]
    write(OUT/f'seed{a.seed}/pool.json',dict(seed=a.seed,arm_order=order,original_split=[2000,400,400],new_labels=0))
    for arm in order:fit(arm,a.seed,x,y)
    write(OUT/f'seed{a.seed}/complete.json',dict(fits=4,test_evaluated=False))


if __name__=='__main__':main()
