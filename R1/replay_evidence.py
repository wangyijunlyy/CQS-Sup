"""Recompute response statistics from released numerical evidence."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
EXP=ROOT/'paper_iclr2027/rebuttal_20261107'
WORLD=ROOT/'world_model_surrogate'
EXT=EXP/'reviewer_weak_reject_2/extensions_20261108'
sys.path[:0]=[str(EXP),str(WORLD/'scripts'),str(WORLD),str(EXT),str(EXT/'parallel')]

def read(path):return json.loads(path.read_text())
def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(mean=float(a.mean()),sd=float(a.std(ddof=1)),values=a.tolist())
def same(a,b,atol=1e-10):np.testing.assert_allclose(a,b,rtol=0,atol=atol)

def masks():
    values={k:[] for k in ('P-local','P-full','F-local','F-full')}
    for seed in range(1001,1006):
        base=WORLD/f'results_supplementary/e1/confirmation/seed{seed}'
        with np.load(base/'data/dataset.npz') as a:exact=a['exact_terms'][2400:]
        for name in values:
            with np.load(base/name/'test_predictions.npz') as a:
                same(a['exact'],exact);values[name].append(float(np.mean((a['prediction']-exact)**2)))
    result={k:stats(v) for k,v in values.items()}
    same(result['P-local']['mean'],.07735,5e-6)
    same(result['F-local']['mean'],.08643,5e-6)
    return result

def common():
    from run_W1_common_penalty import build_model,initial_state,make_masks,fourier_kernel
    values={'product':[],'fourier':[]};maximum=0.
    for seed in range(1001,1006):
        base=EXP/f'evidence/W1_execution/common_penalty_refined/seed{seed}'
        sel=read(base/'product_selection.json');fsel=read(base/'fourier_selection.json');row=read(base/'results.json')
        assert sel['selected']==min(sel['candidates'],key=lambda x:x['validation_mse'])
        assert fsel['selected']==min(fsel['candidates'],key=lambda x:x['validation_mse'])
        with np.load(WORLD/f'results_supplementary/e1/confirmation/seed{seed}/data/dataset.npz') as a:
            x,y,truth=a['X'],a['measured_terms'],a['exact_terms'][2400:]
        model=build_model(initial_state(seed+1000),make_masks(seed+1000)['actual']).eval()
        model.load_state_dict(torch.load(base/f"product_lambda{row['product_lambda']:g}/model.pt",map_location='cpu',weights_only=True))
        with torch.no_grad():product=model(torch.tensor(x[2400:],dtype=torch.float32))[0].numpy()
        fourier=np.zeros_like(truth)
        for file in base.glob('fourier_group*.npz'):
            with np.load(file) as a:
                tr,mask,w,outputs=a['train_x'],a['mask'],a['weights'],a['outputs']
                mean,grand,ym=a['mean'],float(a['grand']),a['target_mean']
            same(tr,x[:2000].reshape(2000,-1))
            kernel=fourier_kernel(tr,tr,mask)-mean[:,None]-mean[None,:]+grand
            residual=kernel@w+2000*row['fourier_lambda']*w-(y[:2000,outputs]-ym)
            if row['fourier_lambda']==0:
                error=np.linalg.norm(kernel@residual)/max(1.,np.linalg.norm(kernel@(y[:2000,outputs]-ym)))
            else:error=np.linalg.norm(residual)/max(1.,np.linalg.norm(y[:2000,outputs]-ym))
            assert error<1e-6
            cross=fourier_kernel(x[2400:].reshape(400,-1),tr,mask)
            cross=cross-cross.mean(1,keepdims=True)-mean+grand
            fourier[:,outputs]=cross@w+ym
        with np.load(base/'predictions.npz') as a:
            same(a['exact'],truth);same(a['product'],product,1e-6);same(a['fourier'],fourier)
        for name,pred in [('product',product),('fourier',fourier)]:
            mse=float(np.mean((pred-truth)**2));same(mse,row[name+'_mse']);values[name].append(mse)
    result={k:stats(v) for k,v in values.items()}
    saved=read(EXP/'evidence/W1_execution/common_penalty_refined/summary.json')
    for name in values:same(result[name]['mean'],saved[name+'_mean'])
    result['paired_product_wins']=int(np.sum(np.array(values['product'])<values['fourier']))
    assert result['paired_product_wins']==5
    return result

def parallel(device='cpu'):
    from benchmark_parallel import load
    from parallel_models import CONFIGS
    root=EXT/'parallel';errors=[];results=[];strict=0
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    for seed in range(1001,1006):
        with np.load(WORLD/f'results_supplementary/e2/confirmation/seed{seed}/dataset.npz') as a:
            x,y=a['X'][2400:],a['y'][2400:]
        for name in ['cqs_r','query_gru','period_mlp',*CONFIGS]:
            model=load(name,seed,root,device)
            with torch.inference_mode():pred=model(torch.tensor(x,dtype=torch.float32,device=device)).cpu().numpy()
            with np.load(root/'evaluation'/f'seed{seed}_{name}.npz') as a:
                same(a['exact'],y);saved=a['prediction']
            error=float(abs(pred-saved).max())
            # The original PyTorch2.6 CUDA Transformer fast path produces
            # different values from its CPU/unfused path, even at the same
            # version. Preserve that actual execution instead of modifying
            # the model or disguising the difference with a loose tolerance.
            diagnostic=device=='cpu' and name.startswith('transformer')
            if not diagnostic:
                assert error<2e-5,(seed,name,error);strict+=1
            assert np.isfinite(pred).all()
            errors.append(error);results.append(dict(seed=seed,name=name,mae=float(abs(pred-y).mean()),
                max_difference_from_published_prediction=error,CPU_Transformer_diagnostic=diagnostic))
    import verify_parallel
    verify_parallel.main()
    return dict(checkpoints_evaluated=45,strict_checkpoint_matches=strict,device=device,
        CPU_Transformer_diagnostic_count=20 if device=='cpu' else 0,
        all400test_controls_per_checkpoint=True,max_difference_from_published_prediction=max(errors),
        strict_prediction_tolerance=2e-5,all45_published_test_metrics_recomputed_from_saved_predictions=True,results=results,
        timing='Published RTX4090 timings are audited from saved repetitions; run benchmark_parallel.py to remeasure.')

def reference():
    from run_evidence import TFIMTask,ChainExact
    task=TFIMTask(1,6,2,False,-1)
    base=EXP/'reviewer_weak_reject_2/chain_evidence'
    info=read(base/'reference.json');best=np.load(base/'best_found_ansatz_controls.npy')
    energy=task.energy(best,-.1,-.5);same(energy,info['best_found_ansatz_energy'])
    lower,upper=task.spectrum_bounds(-.1,-.5)
    same(lower,info['spectral_lower']);same(upper,info['spectral_upper'])
    same(energy-lower,info['ansatz_bracket_width'])
    return dict(ground_lower_bound=lower,best_found_ansatz_energy=energy,ansatz_bracket_width=energy-lower,
        global_ansatz_certificate=False)

def grid_continuation():
    from run_evidence import TFIMTask
    from run_vqe_fourier_controls import source_paths
    values={};count=0
    for base in sorted((WORLD/'results_review_controls').glob('grid*_seed*')):
        case=read(base/'protocol.json')['case'];task=TFIMTask(case['rows'],case['cols'],case['layers'],True,1)
        with np.load(base/'endpoints_J-0.1.npz') as a:
            endpoints={k:a[k] for k in a.files}
        for mode in ('direct','continuation'):
            with np.load(base/f'fourier_{mode}_J-0.1.npz') as a:parameters=a['parameters'][-1]
            truth=np.array([task.energy(x,-.1,-.5) for x in parameters])
            same(truth,endpoints['fourier_'+mode]);count+=1
        reference=np.array([task.energy(x,-.1,-.5) for x in endpoints['product_reference_controls']])
        same(reference,endpoints['product_reference_exact'])
        seed=int(base.name.rsplit('seed',1)[1])
        path=source_paths(case,seed)['J-0.1_redq_continuation']
        with np.load(path) as a:parameters=a['parameters'][-1]
        truth=np.array([task.energy(x,-.1,-.5) for x in parameters])
        same(truth,endpoints['product_continuation'])
        values.setdefault(case['id'],{'product':[],'fourier':[]})
        values[case['id']]['product'].append(float(truth.mean()))
        values[case['id']]['fourier'].append(float(endpoints['fourier_continuation'].mean()))
    assert count==24
    return {k:{m:stats(v) for m,v in d.items()} for k,d in values.items()}

def strong_grid():
    from run_evidence import TFIMTask
    task=TFIMTask(3,3,2,True,1);root=EXP/'evidence/warmstart_confirmation'
    saved=read(root/'summary.json');rows=[]
    for seed in range(5101,5106):
        base=root/f'seed{seed}'
        with np.load(base/'dataset.npz') as a:x,y=a['X'],a['measured_terms']
        for j in (-.1,-.3,-.5):
            index=int(np.argmin(y[:160]@task.coefficients(j,-.5)))
            candidate=task.energy(x[index],j,-.5)
            starts=np.remainder(x[index]+np.random.default_rng(seed+9000).normal(0,.05,(20,2,2))+np.pi,2*np.pi)-np.pi
            for method in ('product','fourier_energy_tuned','nc_energy_tuned'):
                with np.load(base/f'{method}_J{j:+.1f}.npz') as a:
                    same(a['starts'],starts,1e-14);parameters=a['parameters']
                truth=np.array([task.energy(z,j,-.5) for z in parameters])
                same(truth,np.load(base/f'{method}_J{j:+.1f}_exact.npy'))
                rows.append(dict(seed=seed,J=j,method=method,energy=float(truth.mean()),candidate=candidate))
    result={}
    for j in (-.1,-.3,-.5):
        result[str(j)]={}
        for method in ('product','fourier_energy_tuned','nc_energy_tuned'):
            values=[r['energy'] for r in rows if r['J']==j and r['method']==method]
            actual=stats(values);same(actual['mean'],saved[str(j)][method]['mean']);same(actual['sd'],saved[str(j)][method]['sd'])
            result[str(j)][method]=actual
    return result

def timing():
    a=read(EXT/'chain/timing.json')
    for operation,group in a['summary'].items():
        for method in ('product','complete_fourier','fourier_shared_features'):
            s=group[method];same(np.mean(s['dataset_medians']),s['mean']);same(np.std(s['dataset_medians'],ddof=1),s['sd'])
        same(group['fourier_shared_features']['mean']/group['product']['mean'],group['ratio_of_mean_medians'])
    return dict(published_timing_records_consistent=True,summary=a['summary'],
        boundary='Evaluation only; excludes fitting/acquisition. Fresh timings vary by hardware.')

FUNCTIONS={k:v for k,v in globals().copy().items() if k in ('masks','common','parallel','reference','grid_continuation','strong_grid','timing')}
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('suite',choices=FUNCTIONS)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    torch.set_num_threads(2);result=parallel(args.device) if args.suite=='parallel' else FUNCTIONS[args.suite]()
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    (out/f'replay_{args.suite}.json').write_text(json.dumps(dict(passed=True,result=result),indent=2)+'\n')
    print('PASS',args.suite,json.dumps(result)[:900],flush=True)
