"""Full-size E1 control with a common nonconstant Fourier-coefficient penalty."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='2'
import argparse
import itertools
import json
import time
import sys
import numpy as np
import torch
from scipy.linalg import eigh
from scipy.sparse.linalg import cg, LinearOperator
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'world_model_surrogate'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from run_product_mask_control import initial_state, build_model, make_masks
from run_fspt8_fair import seed_all
from run_supplement_e1 import fourier_kernel

DEST=HERE/'evidence/W1_execution/common_penalty'
GRID=[0.,1e-6,1e-4,1e-2,1.]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def coefficient_norm(model):
    """Squared orthonormal Fourier norm, excluding the free constant output."""
    f=model.factors.double()
    active=model.active.bool()
    gram=torch.einsum('odrf,odsf->odrs',f,f)
    gram=torch.where(active[:,:,None,None],gram,torch.ones_like(gram)).prod(1)
    w=model.readout.double()
    total=torch.einsum('or,ors,os->o',w,gram,w)
    constant_factors=torch.where(active[:,:,None],f[...,0],torch.ones_like(f[...,0])).prod(1)
    constant=(constant_factors*w).sum(1)
    penalty=total-constant.square()
    if penalty.detach().min() < -1e-9*max(1.,float(total.detach().abs().max())):
        raise FloatingPointError('negative coefficient norm')
    return penalty.clamp_min(0.)


def verify_math():
    # Critical implementation check only; this small fixture is never a paper result.
    model=build_model(initial_state(2001),make_masks(2001)['actual']).double()
    with torch.no_grad():
        model.active.zero_();model.active[:,:2]=1
    explicit=[]
    for o in range(11):
        coeff=torch.stack([(model.readout[o]*model.factors[o,0,:,a]*model.factors[o,1,:,b]).sum()
                           for a,b in itertools.product(range(3),repeat=2)])
        explicit.append(coeff[1:].square().sum())
    explicit=torch.stack(explicit)
    analytic=coefficient_norm(model)
    torch.testing.assert_close(analytic,explicit,rtol=1e-10,atol=1e-12)
    g1=torch.autograd.grad(analytic.sum(),model.factors,retain_graph=True)[0]
    g2=torch.autograd.grad(explicit.sum(),model.factors)[0]
    torch.testing.assert_close(g1,g2,rtol=1e-9,atol=1e-12)
    print('PASS exact coefficient norm and gradients',flush=True)


def fit_product(x,y,seed,lam,folder):
    folder.mkdir(exist_ok=True)
    if (folder/'fit.json').exists():return json.loads((folder/'fit.json').read_text())
    initialization=seed+1000
    seed_all(initialization)
    model=build_model(initial_state(initialization),make_masks(initialization)['actual'])
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    xt,yt,xv,yv=[torch.tensor(a,dtype=torch.float32) for a in (x[:2000],y[:2000],x[2000:2400],y[2000:2400])]
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt,yt),batch_size=32,
        shuffle=True,generator=torch.Generator().manual_seed(initialization))
    best=float('inf');wait=0;best_epoch=0;state=None;history=[]
    tic=time.perf_counter()
    for epoch in range(1,301):
        model.train();train=0.
        for a,b in loader:
            opt.zero_grad(set_to_none=True)
            mse=(model(a)[0]-b).square().mean()
            loss=mse+lam*coefficient_norm(model).mean() if lam else mse
            if not torch.isfinite(loss):raise FloatingPointError((seed,lam,epoch))
            loss.backward();opt.step();train+=float(mse.detach())*len(a)
        model.eval()
        with torch.no_grad():val=float((model(xv)[0]-yv).square().mean())
        history.append(dict(epoch=epoch,train_mse=train/len(xt),validation_mse=val))
        if val<best:
            best=val;best_epoch=epoch;wait=0
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:wait+=1
        if epoch%100==0:print('TRAIN',seed,lam,epoch,best,flush=True)
        if wait>=30:break
    torch.save(state,folder/'model.pt')
    result=dict(lambda_value=lam,validation_mse=best,best_epoch=best_epoch,epochs_run=epoch,
        seconds=time.perf_counter()-tic,weight_decay=0.,penalty='nonconstant Fourier coefficient squared norm')
    write(folder/'history.json',history);write(folder/'fit.json',result)
    print('PRODUCT DONE',seed,result,flush=True)
    return result


def fourier_paths(x,y,mask,grid):
    groups={}
    for o,m in enumerate(mask):groups.setdefault(tuple(m.astype(bool)),[]).append(o)
    fitted=[]
    for active,outputs in groups.items():
        active=np.array(active,bool)
        kernel=fourier_kernel(x[:2000],x[:2000],active)
        mean=kernel.mean(0);grand=float(mean.mean())
        kernel-=mean[:,None];kernel-=mean[None,:];kernel+=grand
        values,u=eigh(kernel,driver='evd',overwrite_a=True,check_finite=False)
        tolerance=np.finfo(float).eps*len(kernel)*max(1.,abs(values).max())*64
        assert values.min()>-tolerance
        values=np.maximum(values,0.)
        ym=y[:2000,outputs].mean(0);projected=u.T@(y[:2000,outputs]-ym)
        cross=fourier_kernel(x[2000:2400],x[:2000],active)
        cross=cross-cross.mean(1,keepdims=True)-mean+grand
        fitted.append(dict(mask=active,outputs=outputs,values=values,u=u,projected=projected,
            mean=mean,grand=grand,ym=ym,cross=cross,tolerance=tolerance))
    return fitted


def dual_weights(fit,lam):
    values=fit['values'];denom=values+2000*lam
    inv=np.divide(1.,denom,out=np.zeros_like(denom),where=denom>fit['tolerance'] if lam==0 else denom>0)
    return fit['u']@(fit['projected']*inv[:,None])


def fourier_validation(fitted,yv,lam):
    pred=np.zeros_like(yv)
    for fit in fitted:pred[:,fit['outputs']]=fit['cross']@dual_weights(fit,lam)+fit['ym']
    return float(np.mean((pred-yv)**2))


def run(seed):
    folder=DEST/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'results.json').exists():
        print('REUSE seed',seed,flush=True);return
    source=ROOT/f'results_supplementary/e1/confirmation/seed{seed}'
    with np.load(source/'data/dataset.npz') as d:x,y=d['X'],d['measured_terms']
    masks=make_masks(seed+1000)['actual'].reshape(11,-1)
    candidates=[];grid=GRID.copy()
    for lam in grid:candidates.append(fit_product(x,y,seed,lam,folder/f'product_lambda{lam:g}'))
    while min(candidates,key=lambda r:r['validation_mse'])['lambda_value']==grid[-1] and grid[-1]<1e6:
        lam=grid[-1]*100;grid.append(lam)
        candidates.append(fit_product(x,y,seed,lam,folder/f'product_lambda{lam:g}'))
    chosen=min(candidates,key=lambda r:r['validation_mse'])
    write(folder/'product_selection.json',dict(selected=chosen,candidates=candidates,test_selection=False))
    print('FOURIER START',seed,flush=True)
    fitted=fourier_paths(x.reshape(len(x),-1),y,masks,GRID)
    fgrid=GRID.copy();fc=[dict(lambda_value=l,validation_mse=fourier_validation(fitted,y[2000:2400],l)) for l in fgrid]
    while min(fc,key=lambda r:r['validation_mse'])['lambda_value']==fgrid[-1] and fgrid[-1]<1e6:
        lam=fgrid[-1]*100;fgrid.append(lam)
        fc.append(dict(lambda_value=lam,validation_mse=fourier_validation(fitted,y[2000:2400],lam)))
    selected=min(fc,key=lambda r:r['validation_mse'])
    write(folder/'fourier_selection.json',dict(selected=selected,candidates=fc,test_selection=False))
    # Selection persisted before loading exact test labels.
    with np.load(source/'data/dataset.npz') as d:exact=d['exact_terms'][2400:]
    model=build_model(initial_state(seed+1000),make_masks(seed+1000)['actual'])
    model.load_state_dict(torch.load(folder/f"product_lambda{chosen['lambda_value']:g}/model.pt",weights_only=True))
    model.eval()
    with torch.no_grad():p=model(torch.tensor(x[2400:],dtype=torch.float32))[0].numpy()
    f=np.zeros_like(exact);iteration_checks=[]
    for g,fit in enumerate(fitted):
        lam=selected['lambda_value'];weights=dual_weights(fit,lam)
        cross=fourier_kernel(x[2400:].reshape(400,-1),x[:2000].reshape(2000,-1),fit['mask'])
        cross=cross-cross.mean(1,keepdims=True)-fit['mean']+fit['grand']
        f[:,fit['outputs']]=cross@weights+fit['ym']
        values=fit['values'];keep=values>fit['tolerance']
        eig=values[keep];rhs=np.sqrt(eig)[:,None]*fit['projected'][keep]/2000
        diag=eig/2000+lam
        for o in range(rhs.shape[1]):
            operator=LinearOperator((len(diag),len(diag)),matvec=lambda z:diag*z)
            beta,info=cg(operator,rhs[:,o],rtol=1e-10,atol=1e-12,maxiter=20000)
            if info:raise FloatingPointError(('Fourier CG not converged',seed,g,o,info))
            alpha=fit['u'][:,keep]@(beta/np.sqrt(eig))
            gap=float(np.max(np.abs(fit['cross']@(alpha-weights[:,o]))))
            iteration_checks.append(gap)
            if gap>2e-5:raise FloatingPointError(('Fourier iterative mismatch',gap))
        np.savez_compressed(folder/f'fourier_group{g}.npz',mask=fit['mask'],outputs=fit['outputs'],
            weights=weights,train_x=x[:2000].reshape(2000,-1),mean=fit['mean'],grand=fit['grand'],target_mean=fit['ym'])
    np.savez_compressed(folder/'predictions.npz',product=p,fourier=f,exact=exact)
    result=dict(seed=seed,product_lambda=chosen['lambda_value'],fourier_lambda=selected['lambda_value'],
        product_mse=float(np.mean((p-exact)**2)),fourier_mse=float(np.mean((f-exact)**2)),
        max_iterative_fourier_validation_difference=max(iteration_checks),
        scope='common objective and coefficient norm; low-rank parameterization changes optimization geometry')
    write(folder/'results.json',result)
    print('SEED COMPLETE',result,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(2)
    verify_math()
    if args.check_only:return
    DEST.mkdir(parents=True,exist_ok=True)
    write(DEST/'protocol.json',dict(seeds=list(range(1001,1006)),initial_grid=GRID,
        expansion='multiply upper lambda by100 while validation-best is upper edge, up to1e6',
        objective='mean squared moment error + lambda * mean nonconstant orthonormal coefficient squared norm',
        cap=300,patience=30,rank=32,harmonics=1,weight_decay=0,intercept_penalized=False,
        selection='validation moment MSE only',data='original full E1 measured pools',
        iterative_fourier='conjugate gradient on the same spectral coefficient normal equations'))
    for seed in range(1001,1006):run(seed)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
