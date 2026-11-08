"""Replay all reported hybrid metrics, checkpoints and terminal energies on CPU."""
from pathlib import Path
import argparse, importlib, json, sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
MAIN=ROOT/'paper_iclr2027/rebuttal_20261107/hybrid_main_20261108'
FEAS=MAIN.with_name('hybrid_feasibility_20261108')

def read(path): return json.loads(path.read_text())

def compare(a,b,atol=2e-7,rtol=2e-7):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid comparison shape or nonfinite value')
    np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
    return float(np.max(np.abs(a.astype(float)-b.astype(float))))

def replay_main():
    sys.path.insert(0,str(MAIN))
    run=importlib.import_module('run_main')
    audit=importlib.import_module('analyze_main')
    rows=read(MAIN/'rows.json')
    report=dict(fits=0,saved_test_arrays=0,checkpoint_samples_per_fit=16,searches=0,endpoints=0,
                max_checkpoint_prediction_error=0.,max_metric_error=0.,max_energy_error=0.)
    for c in run.cells():
        audit.identity(c)
        x,y,tr,va,te=run.load_data(c)
        folder=MAIN/'fits'/c['name']
        audit.check_saved_split(c,folder)
        pred=np.load(folder/'test_prediction.npy')
        model=audit.selected(c)
        with torch.no_grad(): actual=model(torch.tensor(x[te[:16]]))[0].numpy()
        error=compare(actual,pred[:16],atol=2e-5,rtol=2e-5)
        report['max_checkpoint_prediction_error']=max(error,report['max_checkpoint_prediction_error'])
        target=None
        with np.load(run.PROJECT/c['data']) as data:
            if c['task']=='tfim': target=data['y_exact'][te,None]
            elif c['task']=='vqe': target=data['exact_terms'][te]
            else: target=data['y'][te,:c['k']]
        for row in [r for r in rows if r['cell']==c['name'] and r['method']=='hybrid']:
            p,t=pred,target
            if c['task']=='vqe':
                task=run.task_object(c);coef=task.coefficients(row['J'],-.5)
                p,t=pred@coef,target@coef
                search=MAIN/'search'/c['name']/f"J{row['J']:+.1f}.npz"
                with np.load(search) as a:
                    endpoints=np.array([task.energy(z,row['J'],-.5) for z in a['parameters'][-1]])
                error=compare(endpoints,row['endpoints'],atol=2e-8,rtol=2e-8)
                report['max_energy_error']=max(error,report['max_energy_error'])
                compare(endpoints.mean(),row['energy'],atol=2e-8,rtol=2e-8)
                report['searches']+=1;report['endpoints']+=len(endpoints)
            scores=audit.metrics(t,p)
            for metric in ('mae','mse'):
                error=compare(scores[metric],row[metric],atol=2e-10,rtol=2e-10)
                report['max_metric_error']=max(error,report['max_metric_error'])
        report['fits']+=1;report['saved_test_arrays']+=1
        print('Replayed main',c['name'],flush=True)
    if (report['fits'],report['searches'],report['endpoints'])!=(66,33,660):
        raise ValueError('Incomplete main experiment')
    audit.validate_searches()
    # Recompute baseline metrics and circuit energies from the distributed records.
    baseline=audit.baselines()
    report['baseline_rows']=len(baseline['rows'])
    if report['baseline_rows']!=228: raise ValueError('Incomplete baseline matrix')
    # Reproduce every displayed group mean and sample standard deviation.
    for g in read(MAIN/'summary.json')['groups']:
        for method,expected in g['methods'].items():
            selected=[]
            for row in rows:
                key=['vqe',row['case'],row['J']] if row['task']=='vqe' else ['fspt',row['n'],row['k']] if row['task']=='fspt' else ['tfim',row['n']]
                if key==g['key'] and row['method']==method: selected.append(row[g['metric']])
            compare(np.mean(selected),expected['mean'],1e-12,1e-12)
            compare(np.std(selected,ddof=1),expected['sd'],1e-12,1e-12)
    return report

def replay_feasibility():
    sys.path.insert(0,str(FEAS))
    run=importlib.import_module('run_hybrid')
    audit=importlib.import_module('analyze_hybrid')
    rows=read(FEAS/'rows.json')
    report=dict(fits=0,saved_test_arrays=0,checkpoint_samples_per_fit=16,
                max_checkpoint_prediction_error=0.,max_metric_error=0.)
    for seed in run.SEEDS:
        audit.dataset_identity(seed)
        x,y=run.data(seed)
        for arm in audit.ARMS:
            pred=np.load(run.OUT/f'seed{seed}/{arm}/test_prediction.npy')
            m=audit.load(arm,seed)
            with torch.no_grad(): actual=m(torch.tensor(x[2400:2416]))[0].numpy()
            error=compare(actual,pred[:16],2e-5,2e-5)
            report['max_checkpoint_prediction_error']=max(error,report['max_checkpoint_prediction_error'])
            row=next(r for r in rows if r['seed']==seed and r['arm']==arm)
            for key,value in audit.metrics(pred,y[2400:]).items():
                error=compare(value,row[key],2e-7,2e-7)
                report['max_metric_error']=max(error,report['max_metric_error'])
            report['fits']+=1;report['saved_test_arrays']+=1
        print('Replayed feasibility',seed,flush=True)
    if report['fits']!=20: raise ValueError('Incomplete feasibility experiment')
    for arm,metrics in read(FEAS/'summary.json')['overall'].items():
        for metric,expected in metrics.items():
            values=[r[metric] for r in rows if r['arm']==arm]
            compare(np.mean(values),expected['mean'],1e-12,1e-12)
            compare(np.std(values,ddof=1),expected['sd'],1e-12,1e-12)
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite',choices=('all','main','feasibility'),default='all')
    args=ap.parse_args();torch.set_num_threads(2)
    report={}
    if args.suite in ('all','main'): report['main']=replay_main()
    if args.suite in ('all','feasibility'): report['feasibility']=replay_feasibility()
    report['passed']=True
    (ROOT/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
