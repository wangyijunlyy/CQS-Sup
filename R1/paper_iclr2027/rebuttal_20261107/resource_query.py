"""Independent full-model FSPT fits under synchronized actual-time budgets."""
from pathlib import Path
import os,sys,json,time,argparse,platform,random
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('CQS_PROJECT_ROOT',str(HERE.parents[1]/'world_model_surrogate')))
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from src.wmps.models.world_model import ObsWorldModel
def parameter_steps(controls,horizon,normalizer):
    steps=np.arange(1,horizon+1,dtype=np.float32)/normalizer
    return np.concatenate([np.broadcast_to(controls[:,None,:],(len(controls),horizon,controls.shape[1])),
        np.broadcast_to(steps[None,:,None],(len(controls),horizon,1))],axis=-1).astype(np.float32)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

class QueryGRU(ObsWorldModel):
    # Original repeated-target forward from run_supplement_e2.QueryGRU.
    def forward(self,x):
        emb=self.encoder(x);outputs=[];batch,horizon,width=emb.shape
        h=emb.new_zeros(batch,horizon,self.rssm.hidden)
        for j in range(horizon):
            query=emb[:,j:]
            h=self.rssm.gru(query.reshape(-1,width),h.reshape(-1,self.rssm.hidden))
            h=h.reshape(batch,horizon-j,self.rssm.hidden)
            outputs.append(h[:,0]);h=h[:,1:]
        return self.predictor(torch.stack(outputs,1)),None

ARMS=('cqs_r','matched_query','original_query')

class MatchedQueryGRU(ObsWorldModel):
    """Each requested k independently consumes the same chronological prefix."""
    def forward(self,x):
        emb=self.encoder(x);batch,k,width=emb.shape
        h=emb.new_zeros(batch,k,self.rssm.hidden);outputs=[]
        for j in range(k):
            e=emb[:,j:j+1].expand(-1,k-j,-1)
            h=self.rssm.gru(e.reshape(-1,width),h.reshape(-1,self.rssm.hidden))
            h=h.reshape(batch,k-j,self.rssm.hidden)
            outputs.append(h[:,0]);h=h[:,1:]
        return self.predictor(torch.stack(outputs,1)),None

def build_model(name):
    cls={'cqs_r':ObsWorldModel,'matched_query':MatchedQueryGRU,'original_query':QueryGRU}[name]
    return cls(obs_dim=10,out_dim=8,encoder_hidden=128,rssm_hidden=128,predictor_hidden=128,stochastic=False)

def eligible_checkpoint(seconds,budget):return seconds<=budget

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

def sync(device):
    if device=='cuda':torch.cuda.synchronize()

def fit_arm(arm,x,y,seed,out,budget,device):
    dest=out/arm;dest.mkdir(exist_ok=True)
    if (dest/'training.json').exists():return
    if (dest/'started.json').exists():raise RuntimeError('Preserve incomplete run and choose a fresh directory')
    write(dest/'started.json',{'seed':seed,'arm':arm,'budget_seconds':budget})
    seed_all(seed+1000);model=build_model(arm).to(device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    xt=torch.tensor(x[:2000],device=device);yt=torch.tensor(y[:2000],device=device)
    xv=torch.tensor(x[2000:2400],device=device);yv=torch.tensor(y[2000:2400],device=device)
    # Warm all full-model kernels, then restore weights, optimizer and batch RNG.
    tick=time.perf_counter()
    for _ in range(10):
        opt.zero_grad(set_to_none=True);loss=(model(xt[:32])[0]-yt[:32]).square().mean();loss.backward();opt.step()
    with torch.no_grad():model(xv)
    sync(device);warmup=time.perf_counter()-tick
    model.load_state_dict(initial);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed+1000)
    torch.save(initial,dest/'initial.pt')
    best=float('inf');selected=None;state=None;history=[];steps=0;examples=0;epoch=0
    sync(device);start=time.perf_counter();last_val_time=0.
    while time.perf_counter()-start<budget:
        epoch+=1;order=torch.randperm(2000,generator=generator);total=0.;n=0
        for i in range(0,2000,32):
            if time.perf_counter()-start>=budget:break
            ids=order[i:i+32].to(device);model.train();opt.zero_grad(set_to_none=True)
            loss=(model(xt[ids])[0]-yt[ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();opt.step();sync(device)
            total+=float(loss)*len(ids);n+=len(ids);steps+=1;examples+=len(ids)
        if n<2000 or budget-(time.perf_counter()-start)<last_val_time*1.2:break
        model.eval();vstart=time.perf_counter()
        with torch.no_grad():val=float((model(xv)[0]-yv).square().mean())
        sync(device)
        candidate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()} if val<best else None
        elapsed=time.perf_counter()-start;last_val_time=time.perf_counter()-vstart
        row={'epoch':epoch,'optimizer_steps':steps,'examples':examples,'train_mse':total/n,'val_mse':val,'seconds':elapsed}
        history.append(row)
        if eligible_checkpoint(elapsed,budget) and val<best:
            best=val;state=candidate;selected=row.copy()
        if epoch%50==0:print(seed,arm,epoch,'seconds',round(elapsed,1),'validation',val,flush=True)
    sync(device);actual=time.perf_counter()-start
    assert state is not None and selected['seconds']<=budget
    torch.save(state,dest/'selected.pt');write(dest/'history.json',history)
    write(dest/'training.json',dict(seed=seed,arm=arm,budget_seconds=budget,actual_seconds=actual,
        selected=selected,optimizer_steps=steps,examples_seen=examples,epochs_started=epoch,
        parameters=sum(p.numel() for p in model.parameters()),warmup_seconds_excluded=warmup,
        setup_excluded=True,time_includes='training, validation and checkpoint copying',
        device=torch.cuda.get_device_name() if device=='cuda' else device,
        torch=torch.__version__,python=platform.python_version()))
    print('TRAINED',seed,arm,'elapsed',actual,'best',best,flush=True)

def run(args):
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.out/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    with np.load(args.data if args.data.suffix=='.npz' else args.data/'dataset.npz') as a:controls=a['X'];targets=a['y']
    assert controls.shape==(2800,9) and targets.shape==(2800,12,8)
    x=parameter_steps(controls,12,24)
    order=[ARMS[(i+args.seed-1001)%3] for i in range(3)]
    write(out/'protocol.json',dict(seed=args.seed,N=8,K=12,width=128,split=[2000,400,400],
        seconds_per_arm=args.seconds,arm_order=order,primary_encoding='chronological j/24 in both CQS-R and matched Query',
        original_query='separate repeated-target k/24 conditioning control',optimizer='Adam .001 weightdecay .0001 batch32',
        selection='validation MSE among checkpoints completed within budget',test_policy='score after all15 fits complete'))
    for arm in order:fit_arm(arm,x,targets,args.seed,out,args.seconds,args.device)
    write(out/'complete.json',{'all_three_fits_finished':True,'test_evaluated':False})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seconds',type=float,default=240)
    ap.add_argument('--device',default='cuda');run(ap.parse_args())
