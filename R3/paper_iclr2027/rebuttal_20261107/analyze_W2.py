"""Replay W2 evidence; evaluate continuation only after all fits complete."""
from pathlib import Path
import sys
import json
import csv
import argparse
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]/'world_model_surrogate'
OUT=HERE/'evidence/W2_analysis'
BLOCKS={'all':(0,24),'1_6':(0,6),'7_12':(6,12),'13_24':(12,24)}


def metrics(y,p):
    y,p=np.asarray(y,dtype=float),np.asarray(p,dtype=float)
    error=np.square(y-p)
    return dict(mae=float(np.abs(y-p).mean()),mse=float(error.mean()),
        pooled_r2=float(1-error.sum()/np.square(y-y.mean()).sum()),
        control_r2=float(1-error.sum()/np.square(y-y.mean(axis=0,keepdims=True)).sum()))


def summary(rows,keys,values):
    groups={}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys),[]).append(row)
    result=[]
    for key,records in groups.items():
        current=dict(zip(keys,key))
        for value in values:
            a=np.array([r[value] for r in records])
            current[value]=dict(mean=float(a.mean()),sample_sd=float(a.std(ddof=1)),values=a.tolist())
        result.append(current)
    return result


def main(curriculum):
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    if curriculum:
        assert (HERE/'evidence/W2_curriculum/queue_status.json').exists()
        import torch
        sys.path.insert(0,str(ROOT/'scripts'))
        from run_fspt_controls import factory
        from run_fspt8_fair import parameter_steps
        torch.set_num_threads(2)
    for n in (8,10):
        with np.load(ROOT/f'results_fspt_scaling/datasets/dataset_n{n}.npz') as d:
            y=d['y'].copy(); controls=d['X'].copy() if curriculum else None
        for seed in (42,43,44):
            folder=ROOT/f'results_fspt_diagnostics/n{n}_w128_seed{seed}'
            with np.load(folder/'split.npz') as split:
                tr,va,te=split['train'],split['val'],split['test']
            models={f'CQS-R cap{cap}':np.load(folder/f'pred_budget{cap}.npy') for cap in (300,600)}
            nc=HERE/f'evidence/W1_execution/fspt_matrix/n{n}_k24_seed{seed}/pred_per_output_alpha.npy'
            if n==8:
                nc=HERE/f'evidence/nc_tuned/fspt_n8_k24_seed{seed}/pred_per_output_alpha.npy'
            models['NC tuned']=np.load(nc)
            models['Training median']=np.broadcast_to(np.median(y[tr],axis=0),y[te].shape)
            if curriculum:
                f=HERE/f'evidence/W2_curriculum/n{n}_seed{seed}'
                training=json.loads((f/'training.json').read_text())
                history=json.loads((f/'history.json').read_text())
                selected=min(history,key=lambda h:h['val_mse'])
                assert selected['epoch']==training['best_epoch']
                assert selected['val_mse']==training['best_val_mse']
                saved_val=np.load(f/'pred_val.npy')
                replay_val=float(np.square(saved_val.astype(float)-y[va].astype(float)).mean())
                assert abs(replay_val-training['best_val_mse'])<1e-7
                m=factory('redq',n).eval()
                m.load_state_dict(torch.load(f/'selected.pt',map_location='cpu',weights_only=True))
                x=parameter_steps(controls[te],24,24)
                with torch.no_grad():
                    p=np.concatenate([m(torch.from_numpy(x[i:i+128]))[0].numpy() for i in range(0,len(x),128)])
                np.save(f/'pred_test.npy',p)
                models['CQS-R 12-to-24']=p
            for method,pred in models.items():
                assert pred.shape==y[te].shape and np.isfinite(pred).all()
                scores={}
                for block,(a,b) in BLOCKS.items():
                    score=metrics(y[te,a:b],pred[:,a:b]);scores[block]=score
                    rows.append(dict(N=n,seed=seed,method=method,block=block,**score))
                assert abs(scores['all']['mae']-(scores['1_6']['mae']*.25+scores['7_12']['mae']*.25+scores['13_24']['mae']*.5))<1e-12
    agg=summary(rows,['N','method','block'],['mae','mse','pooled_r2','control_r2'])
    query=[]
    for seed in range(1001,1006):
        folder=ROOT/f'results_supplementary/e2/confirmation/seed{seed}'
        for name in ('cqs_r','query_gru'):
            d=json.loads((folder/f'{name}_metrics.json').read_text())
            query.append(dict(seed=seed,method=name,mae=d['metrics']['test']['overall']['mae'],
                training_seconds=d['train_seconds'],fit_wall_seconds=d['fit_wall_seconds'],
                test_seconds=d['metrics']['test']['prediction_seconds']))
    result=dict(scope='same original data pools, three paired split/initialization runs per N',
        block_rows=rows,block_summary=agg,original_query_rows=query,
        original_query_summary=summary(query,['method'],['mae','training_seconds','fit_wall_seconds','test_seconds']),
        control_r2='1-SSE/sum((y-test mean across controls for each period/observable)^2)',
        checks=['test shapes and finite values','exact block-weighted MAE identity']+
            (['all six fits complete before test evaluation','validation argmin checkpoint replay'] if curriculum else []),
        curriculum=curriculum)
    name='with_curriculum' if curriculum else 'existing'
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/f'{name}.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for n in (8,10):
        print('N',n)
        for r in agg:
            if r['N']==n and r['block'] in ('all','13_24'):
                print(r['method'],r['block'],'MAE',round(r['mae']['mean'],6),'+/-',round(r['mae']['sample_sd'],6),
                      'control R2',round(r['control_r2']['mean'],6))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--curriculum',action='store_true')
    main(ap.parse_args().curriculum)
