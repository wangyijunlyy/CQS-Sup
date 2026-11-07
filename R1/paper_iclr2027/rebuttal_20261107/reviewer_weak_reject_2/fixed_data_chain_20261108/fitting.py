"""Moment-supervised fits selected by six-Hamiltonian measured validation error."""
from pathlib import Path
import json,time
import numpy as np
import torch
from scipy.linalg import eigh

RANKS=(2,4,6,8,16,32)
PRODUCT_REGULARIZERS=(('weight_decay',1e-4),('coefficient',0.),
                      ('coefficient',.1),('coefficient',1.),('coefficient',10.))
RAW_GRID=(0.,1e-6,1e-4,.01,.1,1.,10.,100.)
SCALED_GRID=(1e-8,1e-6,1e-4,.001,.01,.1,1.,10.)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def energy_validation(prediction,measured,coefficients):
    return float(np.mean(((np.asarray(prediction)-np.asarray(measured))@np.asarray(coefficients).T)**2))


def choose(records):
    return min(records,key=lambda row:row['validation_energy_mse'])


def should_confirm(overall_records):
    """Negative product-minus-Fourier energy means a development advantage."""
    differences=[r['product_minus_fourier']['mean'] for r in overall_records]
    if not np.isfinite(differences).all():raise ValueError('Incomplete development comparison')
    return any(value<0. for value in differences)


def coefficient_norm(model):
    """Same nonconstant orthonormal Fourier penalty as the existing control."""
    factors=model.factors.double();active=model.active.bool()
    gram=torch.einsum('odrf,odsf->odrs',factors,factors)
    gram=torch.where(active[:,:,None,None],gram,torch.ones_like(gram)).prod(1)
    w=model.readout.double();total=torch.einsum('or,ors,os->o',w,gram,w)
    constants=torch.where(active[:,:,None],factors[...,0],torch.ones_like(factors[...,0])).prod(1)
    constant=(constants*w).sum(1)
    penalty=total-constant.square()
    if penalty.detach().min() < -1e-9*max(1.,float(total.detach().abs().max())):
        raise FloatingPointError('Negative coefficient norm')
    return penalty.clamp_min(0.)


def kernel(x,z,mask):
    result=np.ones((len(x),len(z)))
    for d in np.flatnonzero(mask):result*=1+2*np.cos(x[:,d,None]-z[None,:,d])
    return result


def fit_fourier_candidates(x,y,xv,yv,masks,coefficients,raw_grid=RAW_GRID,scaled_grid=SCALED_GRID):
    groups={}
    for o,mask in enumerate(masks):groups.setdefault(tuple(mask.astype(bool)),[]).append(o)
    configs=[('coefficient',v) for v in raw_grid]+[('scaled_ridge',v) for v in scaled_grid]
    candidates=[dict(family=family,lambda_value=lam,prediction=np.empty_like(yv,dtype=float)) for family,lam in configs]
    fitted=[]
    for mask,outputs in groups.items():
        mask=np.array(mask,bool)
        gram=kernel(x,x,mask);mean=gram.mean(0);overall=float(mean.mean())
        gram=gram-mean[:,None]-mean[None,:]+overall
        scale=float(np.trace(gram)/len(gram))
        eigen,u=eigh(gram,driver='evd',check_finite=True)
        tolerance=np.finfo(float).eps*len(gram)*max(1.,abs(eigen).max())*64
        if eigen.min() < -tolerance:raise FloatingPointError('Fourier PSD')
        eigen=np.maximum(eigen,0.)
        ym=y[:,outputs].mean(0);projection=u.T@(y[:,outputs]-ym)
        cross=kernel(xv,x,mask);cross=cross-cross.mean(1,keepdims=True)-mean[None,:]+overall
        weights=[]
        for record in candidates:
            effective=record['lambda_value']*(scale if record['family']=='scaled_ridge' else 1.)
            denominator=eigen+len(x)*effective
            inv=np.divide(1.,denominator,out=np.zeros_like(denominator),
                          where=denominator>tolerance if effective==0 else denominator>0)
            w=u@(projection*inv[:,None]);weights.append(w)
            record['prediction'][:,outputs]=cross@w+ym
        fitted.append(dict(train_x=x,mask=mask,outputs=np.array(outputs),mean=mean,
                           overall=overall,scale=1.,target_mean=ym,weights=weights))
    for record in candidates:
        record['validation_energy_mse']=energy_validation(record['prediction'],yv,coefficients)
        record['validation_moment_mse']=float(np.mean((record['prediction']-yv)**2))
    return candidates,fitted


def fit_product(x,y,xv,yv,coefficients,seed,rank,regularizer,strength,folder,device='cpu'):
    from src.wmps.models.periodic_redq import ProductMomentREDQ
    from src.wmps.data.nc_tfim import TFIMTask
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'fit.json').exists():return json.loads((folder/'fit.json').read_text())
    torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    task=TFIMTask(1,6,2,False,-1)
    model=ProductMomentREDQ(6,task.edges,2,False,rank,1).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001,
                         weight_decay=strength if regularizer=='weight_decay' else 0.)
    xt,yt,xvt=[torch.tensor(v,dtype=torch.float32,device=device) for v in (x,y,xv)]
    yvt=torch.tensor(yv,dtype=torch.float64,device=device)
    coeff=torch.tensor(coefficients.T,dtype=torch.float64,device=device)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt,yt),batch_size=32,
        shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');state=None;wait=0;history=[];tic=time.perf_counter()
    for epoch in range(1,301):
        model.train();total=0.
        for a,b in loader:
            opt.zero_grad(set_to_none=True);pred=model(a)[0]
            mse=(pred-b).square().mean()
            loss=mse
            if regularizer=='coefficient' and strength:
                loss=loss+strength*coefficient_norm(model).mean()
            if not torch.isfinite(loss):raise FloatingPointError((seed,rank,regularizer,strength,epoch))
            loss.backward();opt.step();total+=float(mse.detach())*len(a)
        model.eval()
        with torch.no_grad():
            residual=model(xvt)[0].double()-yvt.double()
            val=float((residual@coeff).square().mean())
            moment=float(residual.square().mean())
        history.append(dict(epoch=epoch,train_moment_mse=total/len(x),validation_energy_mse=val,validation_moment_mse=moment))
        if val<best:
            best=val;wait=0;best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        if wait>=30:break
    if state is None:raise FloatingPointError('No finite checkpoint')
    torch.save(state,folder/'model.pt')
    record=dict(rank=rank,regularizer=regularizer,strength=strength,
        validation_energy_mse=best,best_epoch=best_epoch,epochs=epoch,
        seconds=time.perf_counter()-tic,path=str(folder),seed=seed,
        checkpoint='model.pt',training='moment MSE',validation='six-coupling mean energy MSE',
        exact_targets_used=False,device=device)
    write(folder/'history.json',history);write(folder/'fit.json',record)
    print('FIT',seed,rank,regularizer,strength,epoch,round(record['seconds'],2),flush=True)
    return record
