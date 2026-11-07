"""Full ten-pool preflight and checkpoint audit precede confirmation truth."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import json
import numpy as np
import torch
from confirm import HERE,ROOT,models
from fitting import energy_validation,write
from artifacts import preflight_searches,preflight_timed,save_npz
from search import ARMS,COUPLINGS,reference_starts
from timed import select_candidates
from analyze import ChainExact,TFIMTask,describe
from src.wmps.models.periodic_redq import ProductMomentREDQ

METHODS=('product','fourier','fourier_original')


def main():
    torch.set_num_threads(2)
    protocol=json.loads((HERE/'confirmation_protocol.json').read_text())
    seeds=protocol['seeds'];fixed=protocol['fixed_arms'];root=HERE/'confirmation'
    assert len(seeds)==10 and len(set(seeds))==10
    task=TFIMTask(1,6,2,False,-1);coeff=np.stack([task.coefficients(j,-.5) for j in COUPLINGS])
    records=[];timed_records=[];identities={};audit=[]
    max_validation_error=0.;max_export_error=0.;max_shadow_error=0.
    # Every pool is complete before even validation auditing starts.
    for seed in seeds:
        base=root/f'seed{seed}';assert (base/'all_saved.json').exists(),seed
    for seed in seeds:
        base=root/f'seed{seed}'
        assert json.loads((base/'all_saved.json').read_text())['exact_scoring_started'] is False
        with np.load(base/'data/dataset.npz') as a:
            assert set(a.files)=={'X','measured_terms'}
            x,y=a['X'],a['measured_terms']
        assert x.shape==(2800,2,11) and y.shape==(2400,11)
        # Uniform scaling differs by <=1 ulp between remote and local NumPy.
        np.testing.assert_allclose(x,np.random.default_rng(seed).uniform(-np.pi,np.pi,(2800,2,11)),rtol=0,atol=1e-14)
        with np.load(base/'data/shadow_records.npz') as a:
            bits,recipes=a['bits'],a['recipes']
        assert bits.shape==recipes.shape==(2400,10,6)
        assert np.isin(bits,[0,1]).all() and np.isin(recipes,[0,1,2]).all()
        signs=1.-2*bits
        xs=(3*signs*(recipes==0)).mean(1)
        zz=np.stack([(9*signs[:,:,i]*signs[:,:,j]*(recipes[:,:,i]==2)*(recipes[:,:,j]==2)).mean(1) for i,j in task.edges],axis=1)
        error=float(abs(np.concatenate([xs,zz],axis=1)-y).max());max_shadow_error=max(max_shadow_error,error)
        assert error<1e-12
        ps=json.loads((base/'product_selection.json').read_text());assert len(ps['candidates'])==31
        for row in ps['candidates']:
            model=ProductMomentREDQ(6,task.edges,2,False,row['rank'],1).float().eval()
            path=base/'original_product/model.pt' if row['candidate']=='original' else base/'products'/row['candidate']/'model.pt'
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            with torch.no_grad():pred=model(torch.tensor(x[2000:2400],dtype=torch.float32))[0].numpy()
            score=energy_validation(pred,y[2000:2400],coeff)
            error=abs(score-row['validation_energy_mse']);max_validation_error=max(max_validation_error,error)
            assert error<2e-6*max(1.,abs(score)),(seed,row['candidate'],error)
        assert min(ps['candidates'],key=lambda r:r['validation_energy_mse'])['candidate']==ps['selected']['candidate']
        fs=json.loads((base/'fourier_selection.json').read_text());assert len(fs['candidates'])==16
        with np.load(base/'fourier_validation_predictions.npz') as a:predictions=a['predictions']
        for row,pred in zip(fs['candidates'],predictions):
            assert abs(energy_validation(pred,y[2000:2400],coeff)-row['validation_energy_mse'])<1e-10
            assert abs(float(np.mean((pred-y[2000:2400])**2))-row['validation_moment_mse'])<1e-10
        assert min(range(16),key=lambda i:fs['candidates'][i]['validation_energy_mse'])==fs['selected_index']
        assert min((i for i,r in enumerate(fs['candidates']) if r['family']=='scaled_ridge'),key=lambda i:fs['candidates'][i]['validation_moment_mse'])==fs['original_index']
        _,_,fourier,old,ids=models(seed,base);identities[seed]=ids
        for model,index in [(fourier,fs['selected_index']),(old,fs['original_index'])]:
            with torch.no_grad():pred=np.concatenate([model(torch.tensor(batch)).numpy() for batch in np.array_split(x[2000:2400],16)])
            error=float(abs(pred-predictions[index]).max());max_export_error=max(max_export_error,error)
            assert error<1e-8,error
        with np.load(base/'starts.npz') as a:
            uniform,reference=a['uniform'],a['reference']
        np.testing.assert_allclose(uniform,np.random.default_rng(seed+10000).uniform(-np.pi,np.pi,(20,2,11)),rtol=0,atol=1e-14)
        np.testing.assert_allclose(reference,reference_starts(seed),rtol=0,atol=1e-14)
        for arm in fixed:
            boundary,initial,steps=ARMS[arm]
            first=uniform if initial=='uniform' else reference
            for method in METHODS:
                for j in COUPLINGS:
                    path=base/'search'/arm/f'{method}_J{j:+.1f}.npz'
                    records.append((path,steps,ids[method]))
                    # Schedule/start tests below occur before constructing the oracle.
                    with np.load(path) as a:
                        np.testing.assert_array_equal(a['parameters'][0],first)
                        np.testing.assert_allclose(a['sigma'],[max(0,1-k/(steps*.6)) for k in range(steps+1)],atol=1e-15)
        if protocol['primary_arm']=='time_matched':
            for method in METHODS:
                for j in COUPLINGS:
                    path=base/'timed'/f'{method}_J{j:+.1f}.npz';timed_records.append((path,ids[method]))
                    with np.load(path) as a:
                        np.testing.assert_array_equal(a['block_starts'][0],uniform)
                        pool=a['block_parameters'].reshape(-1,2,11);pred=a['block_predictions'].reshape(-1)
                        indices=select_candidates(pool,pred)
                        np.testing.assert_array_equal(pool[indices],a['retained_parameters'])
                        np.testing.assert_array_equal(pred[indices],a['retained_prediction'])
        audit.append(dict(seed=seed,product=ps['selected']['candidate'],fourier=fs['selected'],original_fourier=fs['original']))
    preflight_searches(records)
    if timed_records:preflight_timed(timed_records,protocol['time_cap_seconds'])
    write(root/'preflight.json',dict(passed=True,datasets=10,full_product_checkpoints=310,
        fourier_candidates=160,original_fourier_moment_selection_rebuilt=True,
        max_measured_label_reconstruction_error=max_shadow_error,max_validation_error=max_validation_error,
        max_fourier_export_error=max_export_error,searches=len(records)+len(timed_records),
        all_ten_datasets_and_searches_valid_before_truth=True,selections=audit))
    # This is the first confirmation oracle construction/call.
    write(root/'scoring_started.json',dict(full_preflight_passed=True,all_ten_saved=True))
    oracle=ChainExact(task);rows=[];max_physics=0.
    arms=fixed+(['time_matched'] if timed_records else [])
    for seed in seeds:
        base=root/f'seed{seed}'
        for arm in arms:
            for method in METHODS:
                for j in COUPLINGS:
                    path=(base/'timed'/f'{method}_J{j:+.1f}.npz') if arm=='time_matched' else base/'search'/arm/f'{method}_J{j:+.1f}.npz'
                    with np.load(path) as a:
                        parameters=a['retained_parameters'] if arm=='time_matched' else a['parameters'][-1]
                        first=a['block_starts'][0] if arm=='time_matched' else a['parameters'][0]
                        prediction=a['retained_prediction'] if arm=='time_matched' else a['predictions'][-1]
                        with torch.no_grad():
                            energy=oracle(torch.tensor(parameters),j,-1).numpy()
                            initial=oracle(torch.tensor(first),j,-1).numpy()
                        for index in (0,19):max_physics=max(max_physics,abs(float(energy[index])-task.energy(parameters[index],j,-.5)))
                    save_npz(path.with_suffix('.score.npz'),energy=energy,initial_energy=initial,prediction=prediction)
                    rows.append(dict(seed=seed,arm=arm,method=method,J=j,energy=float(energy.mean()),
                        initial_energy=float(initial.mean()),below_reference_fraction=float(np.mean(energy < -3.-1e-6)),
                        prediction_mae=float(abs(energy-prediction).mean())))
    overall=[];by_j=[]
    for arm in arms:
        values={m:[np.mean([r['energy'] for r in rows if r['seed']==seed and r['arm']==arm and r['method']==m]) for seed in seeds] for m in METHODS}
        record=dict(arm=arm,**{m:describe(values[m]) for m in METHODS},
            product_minus_fourier=describe(np.array(values['product'])-values['fourier']),
            product_minus_original_fourier=describe(np.array(values['product'])-values['fourier_original']))
        record['product_lower_datasets_vs_fourier']=int(np.sum(np.array(values['product'])<values['fourier']))
        record['product_lower_datasets_vs_original']=int(np.sum(np.array(values['product'])<values['fourier_original']))
        record['initial_energy']=float(np.mean([r['initial_energy'] for r in rows if r['arm']==arm and r['method']=='product']))
        overall.append(record)
        for j in COUPLINGS:
            v={m:[r['energy'] for r in rows if r['arm']==arm and r['method']==m and r['J']==j] for m in METHODS}
            by_j.append(dict(arm=arm,J=j,**{m:describe(v[m]) for m in METHODS},
                product_minus_fourier=describe(np.array(v['product'])-v['fourier']),
                product_minus_original_fourier=describe(np.array(v['product'])-v['fourier_original'])))
    primary=next(r for r in overall if r['arm']==protocol['primary_arm'])
    success=primary['product_minus_fourier']['ci95'][1]<0 and primary['product_minus_original_fourier']['ci95'][1]<0
    write(root/'rows.json',rows);write(root/'summary.json',dict(overall=overall,by_coupling=by_j))
    write(root/'decision.json',dict(primary_arm=protocol['primary_arm'],primary_success=success,
        delta='product-minus-Fourier; negative favors product',datasets=10,
        paired_CI_vs_tuned=primary['product_minus_fourier']['ci95'],
        paired_CI_vs_original=primary['product_minus_original_fourier']['ci95'],
        secondary_original_search_reported_separately=True))
    assert max_physics<1e-10
    write(root/'verification.json',dict(passed=True,full_datasets=10,full_product_fits=310,
        searches=len(rows),full_preflight_before_truth=True,measured_labels_reconstructed=True,
        every_checkpoint_selection_rebuilt=True,max_independent_statevector_error=max_physics,
        new_simulated_acquisition_shots=240000,added_search_shots=0,hardware_shots=0))
    for r in overall:print('CONFIRMED',r['arm'],r['product']['mean'],r['fourier']['mean'],r['fourier_original']['mean'],r['product_minus_fourier'],flush=True)
    print('PRIMARY SUCCESS',success,flush=True)


if __name__=='__main__':main()
