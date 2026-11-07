"""Reconstruct all depth-recovery tables and check search/measurement accounting."""
from pathlib import Path
import sys
import json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from chain_depth_recovery import OUT,SEEDS,COUPLINGS,TFIMTask,save
from run_evidence import ChainExact


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    local=OUT/'local_calibration'
    p1=json.loads((OUT/'phase1_summary.json').read_text())
    p2=json.loads((local/'summary.json').read_text())
    raw1=json.loads((OUT/'phase1_results.json').read_text())
    raw2=json.loads((local/'results.json').read_text())
    protocol=json.loads((local/'protocol.json').read_text())
    assert len(raw1)==3*6*3*3==162 and len(raw2)==3*6*3==54
    for summary,raw in ((p1,raw1),(p2,raw2)):
        for row in summary:
            group=[x for x in raw if x['J']==row['J'] and x['method']==row['method']
                   and ('arm' not in row or x['arm']==row['arm'])]
            values=np.array([x['mean_energy'] for x in group])
            assert len(values)==row['n']==3
            assert abs(values.mean()-row['mean'])<1e-12
            assert abs(values.std(ddof=1)-row['sd'])<1e-12
    assert json.loads((local/'all_searches_saved.json').read_text())['runs']==54
    task=TFIMTask(1,6,2,False,-1);oracle=ChainExact(task)
    replay_error=0.;frozen_error=0.;records=[]
    for row in raw1:
        path=OUT/f"seed{row['seed']}"/f"{row['method']}_{row['arm']}_J{row['J']:+.1f}.npz"
        with np.load(path) as a:
            assert a['parameters'].shape==(101,20,2,11)
            if row['arm']=='shallow_staged':
                delta=a['parameters'][:51,:,0]-a['parameters'][0,:,0]
                error=np.max(abs(np.remainder(delta+np.pi,2*np.pi)-np.pi))
                frozen_error=max(frozen_error,float(error))
            x=a['parameters'][-1]
        true=np.load(path.with_suffix('.exact.npy'))
        assert abs(row['mean_energy']-true.mean())<1e-12
        with torch.no_grad():check=oracle(torch.tensor(x),row['J'],-1).numpy()
        replay_error=max(replay_error,float(np.max(abs(check-true))))
    for row in raw2:
        dest=local/f"seed{row['seed']}_{row['method']}_J{row['J']:+.1f}"
        log=json.loads((dest/'complete.json').read_text())
        assert len(log['rounds'])==3 and log['simulated_shots']==3*277*2*1000
        center=np.zeros((2,11));center[0,6:9]=np.pi
        for iteration in range(3):
            with np.load(dest/f'round{iteration}.npz') as a:
                np.testing.assert_allclose(a['center'],center,atol=1e-12,rtol=0)
                measured=a['design_moments']@task.coefficients(row['J'],-.5)
                np.testing.assert_allclose(measured,a['design_energy'],atol=1e-12,rtol=0)
                assert a['design_moments'].shape==(256,11)
                alpha=log['rounds'][iteration]['selection']['alpha']
                candidates=log['rounds'][iteration]['selection']['candidates']
                assert alpha==min(candidates,key=lambda v:v['validation_mse'])['alpha']
                accepted=a['candidate_energy']+2*np.sqrt(a['candidate_variance']+a['incumbent_variance'])<a['incumbent_energy']
                assert np.array_equal(accepted,a['accepted'])
                retained=np.where(accepted[:,None,None],a['candidate_parameters'],a['center'][None])
                np.testing.assert_allclose(retained,a['retained_parameters'],atol=0,rtol=0)
                assert np.max(abs(a['candidate_parameters'][:,0,:6]))==0
                if accepted.any():center=a['candidate_parameters'][np.argmin(np.where(accepted,a['candidate_energy'],np.inf))].copy()
                if iteration==2:
                    proposals=a['candidate_parameters'].copy();pred=a['candidate_prediction'].copy()
        np.testing.assert_allclose(retained,np.load(dest/'final_parameters.npy'),atol=0,rtol=0)
        truth=np.load(dest/'exact_final.npy')
        with torch.no_grad():check=oracle(torch.tensor(retained),row['J'],-1).numpy()
        replay_error=max(replay_error,float(np.max(abs(check-truth))))
        assert abs(truth.mean()-row['mean_energy'])<1e-12
        with torch.no_grad():proposal_truth=oracle(torch.tensor(proposals),row['J'],-1).numpy()
        records.append(dict(seed=row['seed'],J=row['J'],method=row['method'],
            corrected_endpoint_energy_mae=float(np.mean(abs(proposal_truth-pred))),
            proposed_true_mean=float(proposal_truth.mean()),retained_true_mean=float(truth.mean()),
            final_acceptance_fraction=float(accepted.mean())))
    assert replay_error<1e-10 and frozen_error<1e-12,(replay_error,frozen_error)
    save(OUT/'local_diagnostics.json',records)
    fmt=lambda x:f"{x['mean']:.3f} ± {x['sd']:.3f}"
    get1=lambda j,m,a:next(r for r in p1 if r['J']==j and r['method']==m and r['arm']==a)
    get2=lambda j,m:next(r for r in p2 if r['J']==j and r['method']==m)
    phase1=table(['J','CQS-P随机起点','CQS-P浅层热启动','CQS-P分阶段热启动','调参NC随机起点','调参NC浅层热启动','调参NC分阶段热启动'],
        [[f'{j:+.1f}']+[fmt(get1(j,m,a)) for m in ('product','tuned_nc') for a in ('uniform_joint','shallow_joint','shallow_staged')] for j in COUPLINGS])
    ranks=table(['J','验证选秩：随机起点','验证选秩：浅层热启动','验证选秩：分阶段热启动'],
        [[f'{j:+.1f}']+[fmt(get1(j,'product_valrank',a)) for a in ('uniform_joint','shallow_joint','shallow_staged')] for j in COUPLINGS])
    phase2=table(['J','CQS-P＋局部校正','调参NC＋局部校正','纯局部二次回归','可行参考'],
        [[f'{j:+.1f}']+[fmt(get2(j,m)) for m in ('product','tuned_nc','local_only')]+['−3.000'] for j in COUPLINGS])
    coverage=table(['J','CQS-P达到≤−2.990','调参NC达到≤−2.990','纯局部回归达到≤−2.990'],
        [[f'{j:+.1f}']+[f"{100*get2(j,m)['fraction_within_001']:.1f}%" for m in ('product','tuned_nc','local_only')] for j in COUPLINGS])
    paired=[]
    for j in COUPLINGS:
        a=np.array(get2(j,'product')['values']);b=np.array(get2(j,'tuned_nc')['values'])
        c=np.array(get2(j,'local_only')['values'])
        paired.append(dict(J=j,nc_minus_product=float(np.mean(b-a)),product_wins_nc=int(np.sum(a<b-1e-10)),
            local_minus_product=float(np.mean(c-a)),product_wins_local=int(np.sum(a<c-1e-10))))
    save(OUT/'paired_results.json',paired)
    report=dict(passed=True,phase1_runs=162,phase2_runs=54,total_final_candidates=4320,
        seeds=list(SEEDS),couplings=list(COUPLINGS),original_fits_unchanged=True,
        all20_endpoints_retained=True,exact_energy_replay_max_error=replay_error,
        staged_frozen_layer_max_error=frozen_error,measurement_acceptance_reconstructed=True,
        local_ridge_validation_selection_reconstructed=True,extra_simulated_shots_per_local_run=1662000,
        total_extra_simulated_shots=1662000*54,hardware_shots=0,
        statistics='mean ± sample SD across three independent original-fit/measurement runs, not 60 independent datasets')
    save(OUT/'verification.json',report)
    text=f'''# 链式深度退化：已完成实验

本次实验完整覆盖六比特两层链的6种原耦合，以及原3组独立数据。所有能量越低越好，表格报告3次运行均值 ± 样本标准差。每次搜索保留全部20个终点。精确能量仅在保存搜索结果后评分。

## 一、不增加测量的浅层热启动与分阶段优化

使用原CQS-P+C一层模型保存的全部20个终点，将第二层角度置零，再交给所有两层模型作为共同起点。新增层为恒等操作，扩展前后真实能量一致。随机起点对照使用原两层起点。所有新对照均使用100次Adam更新、学习率0.03、周期回绕和相同高斯平滑计划。分阶段流程前50步只更新新增层，后50步联合更新。每个方法都获得相同浅层候选，一层模型已有训练数据与成本单独计入工作流，不能将此流程表述为仅使用原两层数据。

{phase1}

验证选秩版本的完整结果为：

{ranks}

在J=−0.1时，原秩32模型由随机起点的−1.296改善到分阶段热启动的−1.870。改善不能稳定推广到全部耦合和模型，热启动本身尚不足以达到−3。

## 二、增加有限次测量的局部代理校正

三个方法均从显式可行参考−3开始，运行3轮局部优化。每轮测量192个训练控制参数、64个验证控制参数、1个当前候选和20个搜索候选。每组控制参数使用1000次全X和1000次全Z模拟测量。以标准化局部参数的线性项和二次项拟合能量残差，通过验证集选择岭系数，所有方法使用相同网格。纯局部回归不使用预训练代理。

每轮保留20个优化候选。仅当独立测量显示候选能量加上差值的两倍标准误仍低于当前候选时接受，否则保留当前候选。下一轮中心按测量能量选择。最终统计包含全部20个接受或保留的结果，不能称为原流程不增加测量的结果。

每次运行额外使用1,662,000次模拟shots，三个方法预算相同。全部54次运行总计89,748,000次模拟shots，新增QPU调用为0。

{phase2}

以距离−3参考不超过0.010作为接近参考的统计阈值，结果为：

{coverage}

## 三、复核与解释边界

全部162次第一阶段搜索和54次局部校正运行均保留。原检查点和数据不改动。候选接受规则、验证调参规则及测量量已从记录重建。保存终点在Qiskit与独立Pauli展开中的最大能量差为{replay_error:.3e}。新增测量流程与原始零新增测量流程分别报告。局部校正同时适用于NC与纯局部回归，是否体现CQS的额外价值必须依据同预算比较判断。

机器可读结果、逐轮候选和测量记录保存在同目录下的phase1_results.json与local_calibration子目录。
'''
    (OUT/'实验结果.md').write_text(text)
    print(json.dumps(report,indent=2))
    print('PAIRED',json.dumps(paired))
    print('PHASE2 TABLE\n'+phase2)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
