"""Finite-shot local residual correction with a measured incumbent.

All adaptations use fresh simulated all-X/all-Z measurements. Simulator exact
energies are used only after every saved run has finished. No QPU calls.
"""
from pathlib import Path
import os
import sys
import json
import argparse
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
import numpy as np
import torch
from scipy.linalg import cho_factor,cho_solve
from qiskit import QuantumCircuit
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from chain_depth_recovery import ROOT,OUT,SEEDS,COUPLINGS,save,load_models,predictors,TFIMTask

METHODS=('product','tuned_nc','local_only')
SHOTS=1000
ACTIVE=np.arange(6,22)
UPPER=np.triu_indices(16)
ALPHAS=(.001,.01,.1,1.,10.,100.)


def measure(task,x,J,rng,shots=SHOTS):
    """Simulate joint histograms, retaining within-group energy covariance."""
    state=task.state(x)
    q=QuantumCircuit(task.n);q.h(range(task.n))
    probz=state.probabilities();probx=state.evolve(q).probabilities()
    counts_z=rng.multinomial(shots,probz/probz.sum())
    counts_x=rng.multinomial(shots,probx/probx.sum())
    signs=1-2*((np.arange(2**task.n)[:,None]>>np.arange(task.n))&1)
    zz=np.stack([signs[:,i]*signs[:,j] for i,j in task.edges],axis=1)
    mx=counts_x@signs/shots;mz=counts_z@zz/shots
    terms=np.r_[mx,mz];coeff=task.coefficients(J,-.5)
    vx=signs@coeff[:6];vz=zz@coeff[6:]
    variance=sum(max(0.,float(counts@values**2/shots-(counts@values/shots)**2))/(shots-1)
                 for counts,values in ((counts_x,vx),(counts_z,vz)))
    return terms,float(terms@coeff),variance


def features(z):
    z=np.asarray(z)
    return np.concatenate([np.ones((len(z),1)),z,z[:,UPPER[0]]*z[:,UPPER[1]]],axis=1)


def torch_features(z):
    return torch.cat([torch.ones_like(z[:,:1]),z,z[:,UPPER[0]]*z[:,UPPER[1]]],dim=1)


def fit_correction(z,measured,prior):
    a,b=features(z[:192]),features(z[192:])
    residual=measured-prior
    center=float(residual[:192].mean())
    aa=a[:,1:];bb=b[:,1:];mean=aa.mean(0);aa=aa-mean;bb=bb-mean
    gram=aa.T@aa;rhs=aa.T@(residual[:192]-center)
    candidates=[]
    for alpha in ALPHAS:
        w=cho_solve(cho_factor(gram+alpha*np.eye(gram.shape[0])),rhs)
        val=float(np.mean((prior[192:]+center+bb@w-measured[192:])**2))
        candidates.append((val,alpha,np.r_[center-mean@w,w]))
    val,alpha,w=min(candidates,key=lambda r:r[0])
    return w,dict(alpha=alpha,validation_mse=val,candidates=[dict(alpha=a,validation_mse=v) for v,a,_ in candidates])


def propose(prior,center,radius,w,rng):
    # z in [-1,1]^16 corresponds to a local angular box of radius/4 per control.
    z=torch.tensor(rng.uniform(-1,1,(20,16)),dtype=torch.float64,requires_grad=True)
    with torch.no_grad():z[0]=0
    ct=torch.tensor(center.reshape(1,-1),dtype=torch.float64)
    weights=torch.tensor(w)
    opt=torch.optim.Adam([z],lr=.03,foreach=False)
    for _ in range(100):
        opt.zero_grad()
        x=ct.expand(20,-1).clone();x[:,ACTIVE]=ct[:,ACTIVE]+z*(radius/4)
        pred=prior(x.reshape(-1,2,11))+torch_features(z)@weights
        pred.sum().backward();opt.step()
        with torch.no_grad():z.clamp_(-1,1)
    with torch.no_grad():
        x=ct.expand(20,-1).clone();x[:,ACTIVE]=ct[:,ACTIVE]+z*(radius/4)
        pred=prior(x.reshape(-1,2,11))+torch_features(z)@weights
    return x.reshape(-1,2,11).numpy(),pred.numpy()


def check():
    task=TFIMTask(1,6,2,False,-1)
    ref=np.zeros((2,11));ref[0,6:9]=np.pi
    terms,energy,var=measure(task,ref,-.1,np.random.default_rng(10))
    assert np.max(abs(terms[:6]+1))<1e-12
    assert var>=0 and abs(energy+3)<.05
    z=np.random.default_rng(11).uniform(-1,1,(256,16))
    np.testing.assert_allclose(features(z),torch_features(torch.tensor(z)).numpy(),atol=0,rtol=0)
    w,detail=fit_correction(z,features(z)@np.linspace(-.1,.1,153),np.zeros(256))
    error=float(np.max(abs(features(z)@w-features(z)@np.linspace(-.1,.1,153))))
    assert error<.001,error
    return dict(finite_shot_reference=energy,reference_standard_error=float(np.sqrt(var)),local_basis_max_error=error)


def run():
    folder=OUT/'local_calibration';folder.mkdir(parents=True,exist_ok=True)
    protocol=dict(seeds=SEEDS,couplings=COUPLINGS,methods=METHODS,qubits=6,layers=2,
        original_fits='unchanged rank32 CQS-P and validation-tuned NC',
        local_only='same local quadratic, no pretrained surrogate',
        initial_center='explicit energy -3 feasible reference, identical every method',
        original_shallow_fit_used=False,
        rounds=3,radii=[.8,.6,.4],local_box='radius/4 per each of 16 active controls',
        train=192,val=64,basis='intercept + 16 linear + 136 quadratic features in standardized local coordinates',
        correction='ridge on measured energy minus original surrogate, free intercept',
        ridge_grid=ALPHAS,selection='64 fresh finite-shot validation controls, no validation refit',
        acquisition='1000 multinomial samples per joint all-X and all-Z group per control',shots_per_group=SHOTS,
        search='20 starts per round from the same local sampling rule, 100 Adam updates lr0.03, clip local standardized coordinates [-1,1]',
        acceptance='independently measured candidate plus 2*SE(difference) below independently measured incumbent',
        next_center='lowest measured accepted candidate, otherwise retain incumbent',
        final_score='mean true energy of all 20 accept-or-retain candidates in final round',
        measured_controls_per_round=277,simulated_shots_per_run=3*277*2*SHOTS,
        total_runs=len(SEEDS)*len(COUPLINGS)*len(METHODS),hardware_shots=0,
        extra_data='new finite-shot local data, not the original zero-extra-shot pipeline',
        exact_use='post-search scoring only, after every method/run is saved')
    p=folder/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(protocol))
    else:save(p,protocol)
    save(folder/'engineering_checks.json',check())
    task=TFIMTask(1,6,2,False,-1)
    for seed in SEEDS:
        models=load_models(seed,task)
        for ji,J in enumerate(COUPLINGS):
            funcs=predictors(models,task,seed,J)
            funcs['local_only']=lambda z,s:z.flatten(1).sum(1)*0
            for method in METHODS:
                dest=folder/f'seed{seed}_{method}_J{J:+.1f}';dest.mkdir(exist_ok=True)
                if (dest/'complete.json').exists():continue
                prior=lambda z,f=funcs[method]:f(z,0.)
                rng=np.random.default_rng(900000+seed*100+ji)
                center=np.zeros((2,11));center[0,6:9]=np.pi
                log=[]
                for iteration,radius in enumerate(protocol['radii']):
                    # Same sampling law and keyed random seed across methods, centers adapt.
                    z=rng.uniform(-1,1,(256,16))
                    x=np.repeat(center[None],256,axis=0).reshape(256,-1)
                    x[:,ACTIVE]+=z*(radius/4);x=x.reshape(256,2,11)
                    records=[measure(task,p,J,rng) for p in x]
                    measured=np.array([p[1] for p in records]);var=np.array([p[2] for p in records])
                    with torch.no_grad():initial_pred=prior(torch.tensor(x,dtype=torch.float64)).numpy()
                    w,selection=fit_correction(z,measured,initial_pred)
                    candidates,pred=propose(prior,center,radius,w,rng)
                    _,inc,incvar=measure(task,center,J,rng)
                    measured_candidates=[measure(task,p,J,rng) for p in candidates]
                    scores=np.array([p[1] for p in measured_candidates]);variances=np.array([p[2] for p in measured_candidates])
                    accept=scores+2*np.sqrt(variances+incvar)<inc
                    retained=np.where(accept[:,None,None],candidates,center[None])
                    next_index=int(np.argmin(np.where(accept,scores,np.inf))) if accept.any() else None
                    np.savez_compressed(dest/f'round{iteration}.npz',center=center,design=x,design_moments=np.stack([p[0] for p in records]),
                        design_energy=measured,design_variance=var,prior_prediction=initial_pred,correction_weights=w,
                        candidate_parameters=candidates,candidate_prediction=pred,candidate_energy=scores,candidate_variance=variances,
                        incumbent_energy=inc,incumbent_variance=incvar,accepted=accept,retained_parameters=retained)
                    log.append(dict(round=iteration,accepted=int(accept.sum()),incumbent_energy=inc,
                        candidate_measured_mean=float(scores.mean()),radius=radius,selection=selection))
                    if next_index is not None:center=candidates[next_index].copy()
                np.save(dest/'final_parameters.npy',retained)
                save(dest/'complete.json',dict(seed=seed,J=J,method=method,rounds=log,
                    simulated_shots=protocol['simulated_shots_per_run'],exact_evaluation_started=False))
                print('LOCAL SAVED',seed,J,method,[r['accepted'] for r in log],flush=True)
    save(folder/'all_searches_saved.json',dict(runs=protocol['total_runs'],complete=True,before_exact_scoring=True))
    rows=[]
    for seed in SEEDS:
        for J in COUPLINGS:
            for method in METHODS:
                dest=folder/f'seed{seed}_{method}_J{J:+.1f}'
                x=np.load(dest/'final_parameters.npy')
                true=np.array([task.energy(p,J,-.5) for p in x]);np.save(dest/'exact_final.npy',true)
                row=dict(seed=seed,J=J,method=method,mean_energy=float(true.mean()),
                    fraction_below_minus3=float(np.mean(true < -3-1e-8)),fraction_within_001=float(np.mean(true<=-2.99)),
                    mean_reference_improvement=float(np.mean(-3-true)),extra_shots=protocol['simulated_shots_per_run'])
                rows.append(row)
    save(folder/'results.json',rows)
    summary=[]
    for J in COUPLINGS:
        for method in METHODS:
            r=[x for x in rows if x['J']==J and x['method']==method];v=np.array([x['mean_energy'] for x in r])
            summary.append(dict(J=J,method=method,mean=float(v.mean()),sd=float(v.std(ddof=1)),n=len(v),
                fraction_within_001=float(np.mean([x['fraction_within_001'] for x in r])),
                below_minus3_fraction=float(np.mean([x['fraction_below_minus3'] for x in r])),values=v.tolist()))
    save(folder/'summary.json',summary)
    print('LOCAL COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    torch.set_num_threads(2)
    if args.check:print(json.dumps(check(),indent=2))
    else:run()
