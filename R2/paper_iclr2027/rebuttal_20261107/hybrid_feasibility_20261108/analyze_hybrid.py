"""Audit20complete fits before test; no additional model selection."""
import json,time,hashlib
import numpy as np
import torch
from scipy.stats import t
from run_hybrid import HERE,OUT,SEEDS,BUDGET,SOURCE,data,write,preflight
from models import build,ARMS


def close(a,b,tol=2e-5):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid finite replay')
    error=float(abs(a.astype(float)-b.astype(float)).max())
    if not np.allclose(a,b,atol=tol,rtol=tol):raise ValueError(('Replay mismatch',error))
    return error


def physics(x,y):
    from qiskit.quantum_info import Statevector
    from src.wmps.data.fspt_data import build_period_block
    signs=1-2*((np.arange(256)[:,None]>>np.arange(8))&1);errors=[]
    for index in (0,2000):
        block=build_period_block(8,float(x[index,0]),x[index,1:].astype(float));state=Statevector.from_label('0'*8)
        for k in range(12):
            state=state.evolve(block);z=state.probabilities()@signs
            errors.append(close(z,y[index,k],2e-6))
    return max(errors)


def load(arm,seed):
    m=build(arm,seed).eval();m.load_state_dict(torch.load(OUT/f'seed{seed}/{arm}/selected.pt',map_location='cpu',weights_only=True));return m


def dataset_identity(seed):
    folder=SOURCE/f'seed{seed}';meta=json.loads((folder/'data_metadata.json').read_text())
    if meta['data_seed']!=seed or meta['source']!='fresh_exact':raise ValueError('Source seed mismatch')
    for name,key in [('dataset.npz','dataset_sha256'),('split.npz','split_sha256')]:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=meta[key]:raise ValueError('Copied source identity mismatch')
    with np.load(folder/'split.npz') as split:
        for name,lo,hi in [('train',0,2000),('val',2000,2400),('test',2400,2800)]:
            if not np.array_equal(split[name],np.arange(lo,hi)):raise ValueError('Source split mismatch')
    return dict(seed=seed,dataset_sha256=meta['dataset_sha256'],split_sha256=meta['split_sha256'])


def selected_paths(m,seed,x):
    initial=build('hybrid',seed).double()
    changes=dict(factors=float((m.factors-initial.factors).detach().abs().max()),
                 gru=float((m.gru.weight_ih-initial.gru.weight_ih).detach().abs().max()))
    inputs=[]
    def capture(module,args):
        args[0].retain_grad();inputs.append(args[0])
    hook=m.gru.register_forward_pre_hook(capture)
    direct=m.product_readout.detach().clone()
    with torch.no_grad():m.product_readout.zero_()
    m.zero_grad(set_to_none=True);prediction=m(torch.tensor(x[2000:2016],dtype=torch.float64))[0]
    prediction.square().sum().backward();hook.remove()
    def rms(v):return float(v.detach().square().mean().sqrt())
    evidence=dict(seed=seed,parameter_max_changes=changes,
        recurrent_only_output_rms=rms(prediction),factor_gradient_rms=rms(m.factors.grad),
        gru_gradient_rms=rms(m.gru.weight_ih.grad),
        current_product_input_gradient_rms=rms(torch.cat([i.grad[:,:m.rank] for i in inputs])),
        accumulated_product_input_gradient_rms=rms(torch.cat([i.grad[:,m.rank:2*m.rank] for i in inputs])))
    with torch.no_grad():m.product_readout.copy_(direct)
    q=m.period_features(m.angle_steps(torch.tensor(x[2000:2016],dtype=torch.float64)))
    evidence['direct_product_output_rms']=rms(((q.cumprod(1)-1)*m.product_readout).sum(-1))
    evidence['passed']=all(v>1e-12 for k,v in evidence.items() if k.endswith('_rms')) and all(v>1e-12 for v in changes.values())
    if not evidence['passed']:raise ValueError(('Inactive selected-model path',evidence))
    return evidence


def audit():
    fits=preflight();prediction_errors=[];cpu_errors=[];physical_errors=[];equivalence=[];identities=[];paths=[]
    for seed in SEEDS:
        identities.append(dataset_identity(seed));x,y=data(seed);physical_errors.append(physics(x,y))
        for arm in ARMS:
            b=OUT/f'seed{seed}'/arm;meta=json.loads((b/'training.json').read_text());history=json.loads((b/'history.json').read_text())
            if meta['actual_seconds']<BUDGET or meta['actual_seconds']>BUDGET+10:raise ValueError('Unexpected actual time overrun')
            eligible=[r for r in history if r['seconds']<=BUDGET];selected=min(eligible,key=lambda r:r['val_mse'])
            if selected!=meta['selected']:raise ValueError('Checkpoint is not validation best within budget')
            initial=build(arm,seed).state_dict();saved=torch.load(b/'initial.pt',map_location='cpu',weights_only=True)
            if initial.keys()!=saved.keys():raise ValueError('Changed architecture')
            for key in initial:close(initial[key].numpy(),saved[key].numpy(),0.)
            m=load(arm,seed)
            if sum(p.numel() for p in m.parameters())!=meta['parameters']:raise ValueError('Parameter mismatch')
            with torch.no_grad():
                cpu=m(torch.tensor(x[2000:2400]))[0].numpy()
                pred=m.cuda()(torch.tensor(x[2000:2400],device='cuda'))[0].cpu().numpy()
            saved=np.load(b/'validation_prediction.npy')
            cpu_errors.append(float(abs(cpu-saved).max()))
            prediction_errors.append(close(pred,saved,1e-7))
            close(np.mean((pred-y[2000:2400])**2),selected['val_mse'],2e-6)
        hi=torch.load(OUT/f'seed{seed}/hybrid/initial.pt',map_location='cpu',weights_only=True)
        ad=torch.load(OUT/f'seed{seed}/additive/initial.pt',map_location='cpu',weights_only=True)
        for key in hi:close(hi[key].numpy(),ad[key].numpy(),0.)
        m=load('hybrid',seed).double();xx=torch.tensor(x[2000:2004],dtype=torch.float64,requires_grad=True)
        a=m(xx)[0];ga=torch.autograd.grad(a.square().sum(),xx)[0]
        b=m(xx,independent=True)[0];gb=torch.autograd.grad(b.square().sum(),xx)[0]
        equivalence.append(dict(seed=seed,prediction_error=close(a.detach().numpy(),b.detach().numpy(),1e-10),
                           input_gradient_error=close(ga.numpy(),gb.numpy(),1e-9)))
        paths.append(selected_paths(m,seed,x))
        print('AUDITED HYBRID',seed,flush=True)
    if len({i['dataset_sha256'] for i in identities})!=5:raise ValueError('Repeated source pool')
    result=dict(source_identities=identities,selected_path_activity=paths,passed=True,fits=fits,before_new_test_scoring=True,max_validation_replay_error=max(prediction_errors),max_cpu_float32_difference=max(cpu_errors),
        replay_backend='training-matchedCUDAfloat32;TF32disabled',test_backend='CPUfloat32',
        max_physics_label_error=max(physical_errors),physics='independentQiskitStatevector;train/validationonly,float32storedlabels',
        shared_independent_equivalence=equivalence,shared_update_rounds=12,independent_update_rounds=78,
        observable_states_per_round=8,all20complete=True)
    write(HERE/'verification.json',result);return result


def stats(v):
    v=np.asarray(v,float);mean=float(v.mean());sd=float(v.std(ddof=1));half=float(t.ppf(.975,len(v)-1)*sd/len(v)**.5)
    return dict(mean=mean,sd=sd,ci95=[mean-half,mean+half],values=v.tolist())


def metrics(pred,target):
    error=pred-target
    return dict(mae=float(abs(error).mean()),mse=float((error**2).mean()),first6_mae=float(abs(error[:,:6]).mean()),last6_mae=float(abs(error[:,6:]).mean()))


def score():
    v=json.loads((HERE/'verification.json').read_text())
    if not v['passed'] or v['fits']!=20 or not v['before_new_test_scoring']:raise ValueError('Full audit required')
    rows=[]
    for seed in SEEDS:
        x,y=data(seed);target=y[2400:]
        mean=np.broadcast_to(y[:2000].mean(0)[None],target.shape)
        rows.append(dict(seed=seed,arm='training_mean',**metrics(mean,target)))
        for arm in ARMS:
            m=load(arm,seed)
            with torch.no_grad():pred=np.concatenate([m(torch.tensor(x[i:i+64]))[0].numpy() for i in range(2400,2800,64)])
            np.save(OUT/f'seed{seed}/{arm}/test_prediction.npy',pred)
            rows.append(dict(seed=seed,arm=arm,**metrics(pred,target)))
        print('SCORED HYBRID',seed,flush=True)
    names=ARMS+('training_mean',);overall={a:{k:stats([r[k] for r in rows if r['arm']==a]) for k in ('mae','mse','first6_mae','last6_mae')} for a in names}
    paired={}
    for other in ('recurrent','product','additive','training_mean'):
        diff=np.array(overall['hybrid']['mae']['values'])-overall[other]['mae']['values']
        paired[other]=dict(**stats(diff),hybrid_lower_pools=int((diff<0).sum()))
    costs={}
    for arm in ARMS:
        meta=[json.loads((OUT/f'seed{s}/{arm}/training.json').read_text()) for s in SEEDS]
        costs[arm]=dict(parameters=meta[0]['parameters'],actual_seconds=stats([r['actual_seconds'] for r in meta]),
                       optimizer_steps=stats([r['optimizer_steps'] for r in meta]),selected_epochs=[r['selected']['epoch'] for r in meta])
    result=dict(overall=overall,paired_mae=paired,costs=costs,feasibility=dict(selected_path_checks=all(r['passed'] for r in v['selected_path_activity']),
                stable_complete_fits=20,hybrid_beats_training_mean_on_all_pools=paired['training_mean']['hybrid_lower_pools']==5),
                scope='five reused independent control pools;feasibilitydevelopment;notfreshconfirmation',new_measurements=0,hardware_shots=0)
    write(HERE/'rows.json',rows);write(HERE/'summary.json',result);return result


def report(s):
    lines=['# CQS-P 与 CQS-R 架构融合：可行性实验','','融合模型同时维护每个可观测量的低秩乘积状态 z 和循环隐藏状态 h。每层先计算电路掩码约束的 Fourier 因子乘积 q，再执行 z←z⊙q，并将 q−1、z−1、周期编码送入同一个共享 GRU。读出同时使用 z 与 h。乘积状态进入状态更新，因而不是两个预测器的事后集成。','',
    '数据：复用五套独立生成的八比特、12周期精确物理轨迹池1001–1005，各2000/400/400训练/验证/测试。旧数据用于本次可行性开发，不能称新独立确认。各模型120秒训练预算，包含同步训练、验证与预算内检查点复制，按验证MSE选取。全部20个拟合核验后才评估本轮新模型测试结果。无新测量或QPU。','',
    '| 模型 | 参数量 | 测试MAE | 测试MSE | 前6周期MAE | 后6周期MAE |','| --- | --- | --- | --- | --- | --- |']
    for arm in ARMS:
        r=s['overall'][arm];c=s['costs'][arm]
        values=[f"{r[k]['mean']:.3f} ± {r[k]['sd']:.3f}" for k in ('mae','mse','first6_mae','last6_mae')]
        lines.append(f"| {arm} | {c['parameters']} | "+' | '.join(values)+' |')
    gain_r=100*(1-s['overall']['hybrid']['mae']['mean']/s['overall']['recurrent']['mae']['mean'])
    gain_p=100*(1-s['overall']['hybrid']['mae']['mean']/s['overall']['product']['mae']['mean'])
    conclusion=f"融合模型在五套数据上均优于纯循环、纯乘积和加法对照，MAE较同预算CQS-R降低{gain_r:.3f}%，较CQS-P降低{gain_p:.3f}%。结合训练后状态路径检查与前缀复用核验，本实验支持两种结构在架构层面联合使用的可行性，并显示在本次固定训练预算下的预测收益。"
    lines.insert(4,conclusion);lines.insert(5,'')
    mean=s['overall']['training_mean']['mae'];lines+=['',f"训练均值基线的测试MAE为{mean['mean']:.3f} ± {mean['sd']:.3f}。",'',
    '| 比较：融合减对照，负值有利于融合 | MAE均值差 | 融合更低池数 |','| --- | --- | --- |']
    for arm,r in s['paired_mae'].items():lines.append(f"| hybrid−{arm} | {r['mean']:+.3f} | {r['hybrid_lower_pools']}/5 |")
    lines+=['','| 模型 | 实际训练秒数 | 优化更新次数 | 选中轮次 |','| --- | --- | --- | --- |']
    for arm,c in s['costs'].items():lines.append(f"| {arm} | {c['actual_seconds']['mean']:.3f} ± {c['actual_seconds']['sd']:.3f} | {c['optimizer_steps']['mean']:.0f} | {c['selected_epochs']} |")
    lines+=['','所有统计量均以五套数据的均值作为五个样本。参数、训练/验证轨迹、选中权重、测试预测和物理标签核验记录均保存在本目录。验证预测以训练时CUDA float32设置重放，与保存值完全一致；测试预测统一采用CPU float32。','', '纯循环模型使用原CQS-R结构、宽度115。纯乘积模型保留原ProductFSPT的rank32及独立前缀头，因此参数更多；它和融合模型都采用常数1、正余弦系数标准差0.02的因子初始化。加法对照与融合模型参数量、初始权重和GRU完全相同，只将空间乘积与时间乘法累积改为加法，以区分乘积状态的作用。','',
    '结构核验覆盖运行乘积与直接前缀乘积一致、关闭直接乘积读出后因子仍能通过GRU影响输出、因子和GRU梯度非零、无未来输入泄漏，以及共享执行和独立前缀重算的预测/梯度一致。另在五个训练后选定模型上，以验证集检查直接乘积分支输出、关闭直接乘积读出后的因子/GRU梯度，以及对当前与累计乘积输入的独立梯度均非零，并核验因子和GRU参数均发生更新。12周期共享执行需要12轮状态更新，独立查询重算需要78轮；每轮更新8个可观测量状态，不能把这个计数写成实际加速倍数或直接等同于原CQS-R单状态成本。','',
    '固定全时域的前缀掩码并集保证编码器随查询前缀保持一致，代价是掩码比单个前缀更保守。非线性GRU融合不自动继承CQS-P的解析Gaussian continuation，本实验为轨迹预测，不包含VQE优化或+C优势主张。']
    (HERE/'实验结果.md').write_text('\n'.join(lines)+'\n')


def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not (HERE/'all_fits_saved.json').exists():raise ValueError('Full fit matrix must freeze')
    audit();s=score();report(s);write(HERE/'complete.json',dict(status='feasibility_complete',fits=20,finished_at=time.time()))
    print('HYBRID FEASIBILITY COMPLETE',json.dumps({a:s['overall'][a]['mae'] for a in ARMS}),flush=True)


if __name__=='__main__':main()
