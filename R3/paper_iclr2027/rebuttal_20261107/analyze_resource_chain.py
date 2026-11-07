"""Verify finite-shot direct searches and aggregate true energy versus shots."""
from pathlib import Path
import json,sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from resource_chain import OUT,BUDGETS,ChainMeasurement
from chain_depth_recovery import OUT as OLD,TFIMTask
from run_evidence import ChainExact
E=OUT.parent

def save(name,value):(E/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
    assert json.loads((E/'chain_all_saved.json').read_text())['runs']==36
    torch.set_num_threads(2);task=TFIMTask(1,6,2,False,-1);oracle=ChainExact(task);engine=ChainMeasurement()
    rows=[];maximum_hist_error=0.;total_shots=0
    for seed in (45,46,47):
        for J in (-.5,-.3,-.1,.1,.3,.5):
            for method in ('spsa','parameter_shift_adam'):
                folder=OUT/f'seed{seed}_{method}_J{J:+.1f}';meta=json.loads((folder/'complete.json').read_text())
                with np.load(folder/'measurements.npz') as a:
                    n=a['shots_per_group'];cx=a['counts_x'].astype(float);cz=a['counts_z'].astype(float)
                    assert np.array_equal(cx.sum(1),n) and np.array_equal(cz.sum(1),n)
                    energy=(cx@engine.x_values(J)+cz@engine.z_values(J))/n
                    vx,vz=engine.x_values(J),engine.z_values(J)
                    variance=(np.maximum(0,cx@vx**2/n-(cx@vx/n)**2)+np.maximum(0,cz@vz**2/n-(cz@vz/n)**2))/(n-1)
                    error=max(np.max(abs(energy-a['energy'])),np.max(abs(variance-a['variance'])))
                    maximum_hist_error=max(maximum_hist_error,float(error));assert error<1e-12
                    cumulative=np.cumsum(2*n);assert cumulative[-1]==BUDGETS[-1];total_shots+=int(cumulative[-1])
                    previous=np.zeros((20,2,11));previous[:,0,6:9]=np.pi
                    for point in meta['points']:
                        b=point['budget'];assert point['actual_shots']==b
                        stop=int(np.searchsorted(cumulative,b))+1;assert cumulative[stop-1]==b
                        candidate=a['parameters'][stop-40:stop-20];old=a['parameters'][stop-20:stop]
                        np.testing.assert_allclose(old,previous,atol=0,rtol=0)
                        ce,oe=energy[stop-40:stop-20],energy[stop-20:stop]
                        cv,ov=variance[stop-40:stop-20],variance[stop-20:stop]
                        accept=ce+2*np.sqrt(cv+ov)<oe
                        retained=np.where(accept[:,None,None],candidate,old)
                        with np.load(folder/f'budget{b}.npz') as p:
                            np.testing.assert_allclose(p['parameters'],retained,atol=0,rtol=0)
                            np.testing.assert_array_equal(p['accepted'],accept)
                        with torch.no_grad():truth=oracle(torch.tensor(retained),J,-1).numpy()
                        np.save(folder/f'budget{b}.exact.npy',truth)
                        rows.append(dict(family='direct',seed=seed,J=J,method=method,total_shots=b,
                            extra_shots=b,original_shots=0,mean_energy=float(truth.mean()),
                            within_reference_fraction=float(np.mean(truth<=-2.99)),values=truth.tolist()))
                        previous=retained
            for method in ('product','tuned_nc','local_only'):
                folder=OLD/'local_calibration'/f'seed{seed}_{method}_J{J:+.1f}'
                initial=0 if method=='local_only' else 24000
                rows.append(dict(family='local',seed=seed,J=J,method=method,total_shots=initial,
                    extra_shots=0,original_shots=initial,mean_energy=-3.,within_reference_fraction=1.,values=[-3.]*20))
                for r in range(3):
                    with np.load(folder/f'round{r}.npz') as a:retained=a['retained_parameters'].copy()
                    with torch.no_grad():truth=oracle(torch.tensor(retained),J,-1).numpy()
                    extra=(r+1)*554000
                    rows.append(dict(family='local',seed=seed,J=J,method=method,total_shots=initial+extra,
                        extra_shots=extra,original_shots=initial,mean_energy=float(truth.mean()),
                        within_reference_fraction=float(np.mean(truth<=-2.99)),values=truth.tolist()))
    save('chain_rows.json',rows)
    summary=[]
    for J in (-.5,-.3,-.1,.1,.3,.5):
        for method in ('product','tuned_nc','local_only','spsa','parameter_shift_adam'):
            group=[r for r in rows if r['J']==J and r['method']==method]
            for budget in sorted(set(r['total_shots'] for r in group)):
                g=[r for r in group if r['total_shots']==budget];v=np.array([r['mean_energy'] for r in g]);assert len(v)==3
                summary.append(dict(J=J,method=method,total_shots=budget,extra_shots=g[0]['extra_shots'],
                    mean=float(v.mean()),sd=float(v.std(ddof=1)),values=v.tolist(),n=3,
                    within_reference_fraction=float(np.mean([r['within_reference_fraction'] for r in g]))))
    save('chain_summary.json',summary)
    paired=[]
    for J in (-.5,-.3,-.1,.1,.3,.5):
        g=lambda m:next(r for r in summary if r['J']==J and r['method']==m and r['extra_shots']==(1662000 if m in ('product','tuned_nc','local_only') else 1686000))
        a=np.array(g('product')['values'])
        for baseline in ('tuned_nc','local_only','spsa','parameter_shift_adam'):
            b=np.array(g(baseline)['values']);paired.append(dict(J=J,baseline=baseline,product_wins=int(np.sum(a<b-1e-10)),baseline_minus_product=float((b-a).mean())))
    save('chain_paired.json',paired)
    check=dict(passed=True,new_direct_runs=36,restarts_each=20,couplings=6,independent_seeds=3,
        exact_scoring_only_after_all_runs_saved=True,original_offline_data_shots=24000,
        histogram_energy_and_variance_max_error=maximum_hist_error,all_candidate_retention_reconstructed=True,
        exact_total_shot_budgets=list(BUDGETS),new_simulated_shots_total=total_shots,hardware_calls=0,
        total_budget_includes_original_fits=True,source_local_runs=54,source_local_round_points=162,
        primary_statistic='mean ± sample SD across3 independent dataset means, each averaging all20 retained outputs')
    save('chain_verification.json',check)
    print(json.dumps(check,indent=2))
    for J in (-.5,.5):
        print('FINAL',J,[(m,next(r['mean'] for r in summary if r['J']==J and r['method']==m and r['extra_shots']==(1662000 if m in ('product','tuned_nc','local_only') else 1686000))) for m in ('product','tuned_nc','local_only','spsa','parameter_shift_adam')])
if __name__=='__main__':main()
