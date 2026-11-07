"""Verify all frozen new-data searches before exact scoring and aggregation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import csv
import json
import sys
import time
import numpy as np
import torch
from scipy.stats import t

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[3]
sys.path[:0]=[str(WORK/'world_model_surrogate'),str(HERE.parent/'extensions_20261108'),
              str(HERE.parents[1])]
from calibration import select_correction,features,ACTIVE
from development import write
from worker import SEEDS,COUPLINGS,ARMS
from complete_fourier import SharedFeatureFourier
from resource_chain import ChainMeasurement
from src.wmps.data.nc_tfim import TFIMTask,shadow_terms
from src.wmps.models.periodic_redq import ProductMomentREDQ,causal_parameter_masks
from run_supplement_e1 import fit_fourier,fourier_kernel,kernel_predict


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),values=a.tolist())


def contrast(values):
    result=stats(values)
    half=float(t.ppf(.975,len(values)-1)*result['sd']/np.sqrt(len(values)))
    result.update(ci95=[result['mean']-half,result['mean']+half],
                  product_lower_datasets=int(sum(v<0 for v in values)))
    return result


def audit_fourier(folder,xtr,ytr,xval,yval,masks,destination):
    """Rebuild all ridge choices from measured labels and audit stored weights."""
    destination.mkdir(parents=True,exist_ok=True)
    rebuilt=fit_fourier(xtr,ytr,xval,yval,masks,destination)
    original=json.loads((folder/'selection.json').read_text())
    reference=json.loads((destination/'selection.json').read_text())
    np.testing.assert_allclose(original['validation_mse'],reference['validation_mse'],rtol=1e-10,atol=1e-10)
    assert original['selected']==reference['selected']
    max_weight_error=0.
    for file in sorted(destination.glob('kernel_group*.npz')):
        with np.load(folder/file.name) as old,np.load(file) as fresh:
            assert set(old.files)==set(fresh.files)
            for key in old.files:
                np.testing.assert_allclose(old[key],fresh[key],rtol=1e-8,atol=1e-9)
            max_weight_error=max(max_weight_error,float(abs(old['weights']-fresh['weights']).max()))
    exact=np.empty((min(8,len(xval)),11))
    for mask,outputs,fitted in rebuilt:
        exact[:,outputs]=kernel_predict(fitted,fourier_kernel(xval[:len(exact)],xtr,mask))[0]
    model=SharedFeatureFourier(folder)
    with torch.no_grad():
        shared=model(torch.tensor(xval[:len(exact)].reshape(-1,2,11))).numpy()
    np.testing.assert_allclose(shared,exact,atol=1e-9,rtol=1e-9)
    return dict(selected_ridge=reference['selected'],all_eight_validation_losses_reconstructed=True,
        selected_weights_max_error=max_weight_error,shared_evaluator_max_error=float(abs(shared-exact).max()))


def predict_batched(prior,x):
    with torch.no_grad():
        return np.concatenate([prior(torch.tensor(b)).numpy() for b in np.array_split(x,max(1,int(np.ceil(len(x)/64))))])


def main():
    torch.set_num_threads(2)
    root=HERE/'confirmation'
    assert all((root/f'seed{s}/all_saved.json').exists() for s in SEEDS)
    runs=[p for s in SEEDS for p in (root/f'seed{s}').glob('*/complete.json') if p.parent.name!='data']
    assert len(runs)==180,len(runs)
    cap=json.loads((HERE/'timing_cap.json').read_text())['search_seconds_per_round']
    # This marker is written before generating any exact terminal targets.
    write(HERE/'all_searches_frozen.json',dict(runs=180,datasets=5,before_exact_scoring=True))
    task=TFIMTask(1,6,2,False,-1)
    engine=ChainMeasurement()
    rows=[]
    max_physics_error=max_hist_error=max_prior_error=max_weight_error=max_pool_error=0.
    first_round_pairs=0
    fit_results=[];fourier_audits=[]
    for seed in SEEDS:
        base=root/f'seed{seed}'
        with np.load(base/'data/dataset.npz') as a:
            x,y=a['X'],a['measured_terms']
            assert 'exact_terms' not in a.files
        assert x.shape==(2800,2,11) and y.shape==(2400,11)
        with np.load(base/'data/shadow_records.npz') as a:
            replay=np.stack([shadow_terms(b,r,task.edges) for b,r in zip(a['bits'],a['recipes'])])
        np.testing.assert_array_equal(y,replay)
        p32=ProductMomentREDQ(6,task.edges,2,False,32,1).eval()
        p32.load_state_dict(torch.load(base/'product/product.pt',map_location='cpu',weights_only=True))
        history=json.loads((base/'product/product_history.json').read_text())
        best=min(history,key=lambda r:r['val_mse'])
        fit=json.loads((base/'product/fit.json').read_text())['details']
        assert best['epoch']==fit['best_epoch']
        with torch.no_grad():
            val=p32(torch.tensor(x[2000:2400],dtype=torch.float32))[0].numpy()
        assert abs(float(((val-y[2000:2400])**2).mean())-fit['best_val_mse'])<1e-5
        masks=causal_parameter_masks(6,task.edges,2,False).reshape(11,22)
        np.testing.assert_array_equal(p32.active.numpy(),masks)
        p=p32.double()
        for parameter in p.parameters():parameter.requires_grad_(False)
        f=SharedFeatureFourier(base/'fourier')
        selection=json.loads((base/'fourier/selection.json').read_text())
        assert selection['selected']==selection['ridges'][int(np.argmin(selection['validation_mse']))]
        for group in f.groups:
            np.testing.assert_array_equal(group['train'].numpy(),x[:2000].reshape(2000,22))
            for o in group['outputs']:np.testing.assert_array_equal(group['mask'],masks[o])
        fourier_audits.append(audit_fourier(base/'fourier',x[:2000].reshape(2000,22),y[:2000],
            x[2000:2400].reshape(400,22),y[2000:2400],masks,HERE/'verification_reconstruction'/f'seed{seed}'))
        for ji,j in enumerate(COUPLINGS):
            coefficient=torch.tensor(task.coefficients(j,-.5))
            funcs=dict(product=lambda z:p(z)[0]@coefficient,fourier=lambda z:f(z)@coefficient)
            pair_reference=None
            for arm in ARMS:
                for model in ('product','fourier'):
                    out=base/f'{arm}_{model}_J{j:+.1f}'
                    meta=json.loads((out/'complete.json').read_text())
                    assert not meta['exact_scoring_started'] and meta['total_shots']==1686000
                    with np.load(out/'measurements.npz') as m:
                        measured={k:m[k].copy() for k in m.files}
                    assert measured['parameters'].shape==(831,2,11)
                    shots=measured['shots_per_group']
                    assert (shots==1000).all() and 2*shots.sum()==1662000
                    cx,cz=measured['counts_x'],measured['counts_z']
                    np.testing.assert_array_equal(cx.sum(1),shots)
                    np.testing.assert_array_equal(cz.sum(1),shots)
                    vx,vz=engine.x_values(j),engine.z_values(j)
                    energy=(cx@vx+cz@vz)/shots
                    variance=(np.maximum(0,cx@vx**2/shots-(cx@vx/shots)**2)+
                        np.maximum(0,cz@vz**2/shots-(cz@vz/shots)**2))/(shots-1)
                    max_hist_error=max(max_hist_error,float(abs(energy-measured['energy']).max()),
                                       float(abs(variance-measured['variance']).max()))
                    np.testing.assert_allclose(energy,measured['energy'],atol=1e-12,rtol=1e-12)
                    np.testing.assert_allclose(variance,measured['variance'],atol=1e-12,rtol=1e-12)
                    accepted_total=blocks=updates=0
                    alphas=[];eligible_times=[];actual_times=[];overruns=[]
                    for r,radius in enumerate((.8,.6,.4)):
                        offset=277*r
                        with np.load(out/f'round{r}.npz') as a:
                            saved={k:a[k].copy() for k in a.files}
                        np.testing.assert_array_equal(saved['design'],measured['parameters'][offset:offset+256])
                        np.testing.assert_allclose(saved['design_energy'],energy[offset:offset+256],atol=1e-12)
                        np.testing.assert_array_equal(saved['candidate_parameters'],measured['parameters'][offset+257:offset+277])
                        if r==0:
                            if pair_reference is None:
                                pair_reference=(saved['design'],cx[:256],cz[:256])
                            else:
                                for actual,reference in zip((saved['design'],cx[:256],cz[:256]),pair_reference):
                                    np.testing.assert_array_equal(actual,reference)
                                first_round_pairs+=1
                        prior=predict_batched(funcs[model],saved['design'])
                        err=float(abs(prior-saved['prior_prediction']).max())
                        max_prior_error=max(max_prior_error,err)
                        assert err<1e-9
                        z=(saved['design'].reshape(256,22)[:,ACTIVE]-saved['center'].reshape(-1)[ACTIVE])/(radius/4)
                        weights,choice=select_correction(z,saved['design_energy'],prior,
                            alphas=(1.,) if arm=='fixed_original' else (0.,.5,1.))
                        stored=meta['rounds'][r]
                        assert choice['alpha']==stored['choice']['alpha'] and choice['ridge']==stored['choice']['ridge']
                        assert abs(choice['validation_mse']-stored['choice']['validation_mse'])<1e-10
                        max_weight_error=max(max_weight_error,float(abs(weights-saved['correction_weights']).max()))
                        np.testing.assert_allclose(weights,saved['correction_weights'],rtol=1e-8,atol=1e-9)
                        with np.load(out/f'search_round{r}.npz') as a:
                            params,pred,times=a['parameters'],a['predicted'],a['eligible_seconds']
                        sm=stored['search']
                        assert len(times)==sm['completed_blocks'] and len(times)>0
                        assert (np.diff(times)>=0).all()
                        if arm=='time_calibrated':
                            assert (times<=cap).all() and sm['eligible_seconds']<=cap
                            assert abs(sm['overrun_seconds']-max(0.,sm['actual_seconds']-cap))<1e-12
                        else:
                            assert len(times)==1 and sm['executed_updates']==100
                        flat=params.reshape(-1,2,11)
                        local_z=(flat.reshape(-1,22)[:,ACTIVE]-saved['center'].reshape(-1)[ACTIVE])/(radius/4)
                        assert abs(local_z).max()<=1+1e-12
                        expected=features(local_z)@weights
                        if choice['alpha']!=0:expected+=choice['alpha']*predict_batched(funcs[model],flat)
                        max_pool_error=max(max_pool_error,float(abs(expected-pred.reshape(-1)).max()))
                        np.testing.assert_allclose(expected,pred.reshape(-1),atol=1e-8,rtol=1e-8)
                        indices=np.argsort(pred.reshape(-1),kind='stable')[:20]
                        np.testing.assert_array_equal(flat[indices],saved['candidate_parameters'])
                        np.testing.assert_array_equal(pred.reshape(-1)[indices],saved['candidate_prediction'])
                        inc_e,inc_v=energy[offset+256],variance[offset+256]
                        candidate_e,candidate_v=energy[offset+257:offset+277],variance[offset+257:offset+277]
                        accept=candidate_e+2*np.sqrt(candidate_v+inc_v)<inc_e
                        np.testing.assert_array_equal(accept,saved['accepted'])
                        retained=np.where(accept[:,None,None],saved['candidate_parameters'],saved['center'][None])
                        np.testing.assert_array_equal(retained,saved['retained_parameters'])
                        if r<2:
                            next_center=saved['candidate_parameters'][int(np.argmin(np.where(accept,candidate_e,np.inf)))] if accept.any() else saved['center']
                            with np.load(out/f'round{r+1}.npz') as a:np.testing.assert_array_equal(next_center,a['center'])
                        accepted_total+=int(accept.sum());alphas.append(choice['alpha'])
                        blocks+=sm['completed_blocks'];updates+=sm['executed_updates']
                        eligible_times.append(sm['eligible_seconds']);actual_times.append(sm['actual_seconds']);overruns.append(sm['overrun_seconds'])
                    final=np.load(out/'final_parameters.npy')
                    np.testing.assert_array_equal(final,retained)
                    # True targets are first computed here, after every search was frozen.
                    pz,px=engine.probabilities(final)
                    true=px@engine.x_values(j)+pz@engine.z_values(j)
                    for index in (0,19):
                        replay=task.energy(final[index],j,-.5)
                        max_physics_error=max(max_physics_error,abs(replay-true[index]))
                    np.save(out/'exact_final.npy',true)
                    rows.append(dict(seed=seed,J=j,model=model,arm=arm,
                        mean_energy=float(true.mean()),within_run_sd=float(true.std(ddof=1)),
                        mean_reference_improvement=float(-3-true.mean()),fraction_below_reference=float(np.mean(true<-3-1e-8)),
                        alphas=alphas,completed_blocks=blocks,executed_updates=updates,
                        eligible_seconds_sum=sum(eligible_times),actual_search_seconds_sum=sum(actual_times),
                        max_round_overrun=max(overruns),accepted=accepted_total,total_shots=1686000))
                    print('VERIFIED AND SCORED',seed,j,arm,model,flush=True)
        # Fresh reserved controls are evaluated only after model/selection freezing.
        z,xprob=engine.probabilities(x[2400:])
        exact=np.c_[xprob@engine.signs,z@engine.zz]
        for name,m in [('product',p),('fourier',f)]:
            with torch.no_grad():
                pred=np.concatenate([(m(torch.tensor(b))[0] if name=='product' else m(torch.tensor(b))).numpy() for b in np.array_split(x[2400:],8)])
            np.savez_compressed(base/f'{name}/test_predictions.npz',prediction=pred,exact=exact)
            fit_results.append(dict(seed=seed,model=name,moment_mse=float(((pred-exact)**2).mean())))
    assert len(rows)==180 and first_round_pairs==150
    assert max_physics_error<1e-11 and max_hist_error<1e-11 and max_pool_error<1e-8
    write(HERE/'rows.json',rows)
    write(HERE/'fit_test_results.json',fit_results)
    summary=[]
    for j in COUPLINGS:
        for arm in ARMS:
            values={m:[next(r['mean_energy'] for r in rows if r['seed']==s and r['J']==j and r['model']==m and r['arm']==arm) for s in SEEDS] for m in ('product','fourier')}
            summary.append(dict(J=j,arm=arm,product=stats(values['product']),fourier=stats(values['fourier']),
                product_minus_fourier=contrast(np.asarray(values['product'])-values['fourier'])))
    overall=[]
    for arm in ARMS:
        values={m:[np.mean([r['mean_energy'] for r in rows if r['seed']==s and r['model']==m and r['arm']==arm]) for s in SEEDS] for m in ('product','fourier')}
        overall.append(dict(arm=arm,product=stats(values['product']),fourier=stats(values['fourier']),
            product_minus_fourier=contrast(np.asarray(values['product'])-values['fourier'])))
    write(HERE/'summary.json',dict(by_coupling=summary,overall=overall,
        primary='time_calibrated product-minus-Fourier paired five-dataset contrast, six equally weighted couplings; negative favors product'))
    with (HERE/'all_runs.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    verification=dict(passed=True,global_fits=10,searches=180,rounds=540,datasets=5,
        first_round_pairs_identical=first_round_pairs,all_alpha_and_ridge_selections_reconstructed=True,
        all_candidate_pools_and_deadline_eligibility_reconstructed=True,
        all_histogram_variances_and_measured_acceptance_reconstructed=True,
        histogram_max_error=max_hist_error,prior_prediction_max_error=max_prior_error,
        correction_weights_max_error=max_weight_error,candidate_prediction_max_error=max_pool_error,
        independent_qiskit_energy_max_error=max_physics_error,
        five_fourier_checkpoint_audits=fourier_audits,
        new_local_simulated_shots=180*1662000,new_global_acquisition_shots=5*24000,
        hardware_shots=0,exact_scoring_after_all180_searches_frozen=True,
        search_window_seconds=cap,max_observed_round_overrun=max(r['max_round_overrun'] for r in rows))
    write(HERE/'verification.json',verification)
    print(json.dumps(overall,indent=2),flush=True)


if __name__=='__main__':
    main()
