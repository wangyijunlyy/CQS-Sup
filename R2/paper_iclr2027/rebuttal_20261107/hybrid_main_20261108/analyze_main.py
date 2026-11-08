"""Saved-baseline audit, all-fit audit, blind search, and final main-task scoring."""
import argparse,json,time,hashlib
import numpy as np
import torch
from run_main import HERE,PROJECT,cells,load_data,model,task_object,write,preflight


def read(p):return json.loads(p.read_text())
def metrics(target,pred):
 target,pred=np.asarray(target,dtype=float),np.asarray(pred,dtype=float)
 if target.shape!=pred.shape or not np.isfinite(pred).all():raise ValueError('Invalid scored prediction')
 return dict(mae=float(np.mean(abs(pred-target))),mse=float(np.mean((pred-target)**2)))
def check(a,b,atol=2e-7,rtol=2e-7):
 a,b=np.asarray(a),np.asarray(b)
 if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid finite comparison')
 np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
 return float(np.max(abs(a.astype(float)-b.astype(float))))
def selected(c):
 m=model(c);m.load_state_dict(torch.load(HERE/'fits'/c['name']/'joint.pt',map_location='cpu',weights_only=True));return m.eval()


def identity(c):
 protocol=read(PROJECT/c['product']/'protocol.json')
 expected=protocol['dataset_sha256']
 actual=hashlib.sha256((PROJECT/c['data']).read_bytes()).hexdigest()
 if actual!=expected:raise ValueError(('Original data changed',c['name']))
 return dict(cell=c['name'],original_dataset=c['data'],original_split=c['source']+'/split.npz' if c['task']!='vqe' else 'originalorderedtrain/validation/test',dataset_sha256=actual)


def physical_spot(c):
 from qiskit.quantum_info import Statevector
 with np.load(PROJECT/c['data']) as a:
  x=a['X'];y=a['y_exact'] if c['task']=='tfim' else a['y'] if c['task']=='fspt' else a['exact_terms']
 _,_,tr,va,_=load_data(c);error=[]
 for index in (int(tr[0]),int(va[0])):
  if c['task']=='fspt':
   from src.wmps.data.fspt_data import build_period_block
   block=build_period_block(c['n'],float(x[index,0]),x[index,1:].astype(float));sv=Statevector.from_label('0'*c['n'])
   signs=1-2*((np.arange(2**c['n'])[:,None]>>np.arange(c['n']))&1)
   for k in range(c['k']):
    sv=sv.evolve(block);error.append(check(sv.probabilities()@signs,y[index,k],2e-6,2e-6))
  elif c['task']=='tfim':
   from src.wmps.data.quantum_data import build_circuit,build_tfim_hamiltonian
   sv=Statevector.from_label('0'*c['n'])
   for layer in x[index]:sv=sv.evolve(build_circuit(layer,c['n']))
   error.append(check(float(sv.expectation_value(build_tfim_hamiltonian(c['n'],-.1,-.5)).real),y[index],2e-6,2e-6))
  else:error.append(check(task_object(c).exact_terms(x[index]),y[index],2e-8,2e-8))
 return max(error)


def baselines():
 rows=[];identities=[];physics=[]
 for c in cells():
  identities.append(identity(c));physics.append(dict(cell=c['name'],max_error=physical_spot(c)))
  x,y,tr,va,te=load_data(c);source=PROJECT/c['source'];product=PROJECT/c['product'];record=read(source/'results.json');pr=read(product/'results.json')
  with np.load(PROJECT/c['data']) as data:
   if c['task']=='tfim':
    target=data['y_exact'][te,None]
    for name,file,folder,expected in [('cqs_r','pred_redq_hcs.npy',source,record['models']['redq_hcs']['metrics']['mae']),('cqs_p','pred_product.npy',product,pr['models']['product']['mae'])]:
     scores=metrics(target,np.load(folder/file).reshape(-1,1));check(scores['mae'],expected,1e-10,1e-10)
     rows.append(dict(cell=c['name'],task=c['task'],n=c['n'],seed=c['seed'],method=name,**scores))
   elif c['task']=='fspt':
    target=data['y'][te,:c['k']]
    for name,file,folder,expected in [('cqs_r','pred_redq.npy',source,record['models']['redq']['overall']['mae']),('cqs_p','pred.npy',product,pr['metrics']['mae'])]:
     pred=np.load(folder/file);scores=metrics(target,pred);check(scores['mae'],expected,1e-10,1e-10)
     rows.append(dict(cell=c['name'],task=c['task'],n=c['n'],k=c['k'],seed=c['seed'],method=name,per_period_mae=np.mean(abs(pred.astype(float)-target),axis=(0,2)).tolist(),**scores))
   else:
    task=task_object(c);cont=read(PROJECT/c['continuation']/'results.json');mg=read(PROJECT/c['moment_gru']/'results.json')
    original_starts=np.load(source/'initial_parameters.npy')
    for J in c['case']['couplings']:
     cr=next(r for r in cont['records'] if np.isclose(r['J'],J));p=next(r for r in pr['records'] if np.isclose(r['J'],J));mr=next(r for r in mg['records'] if np.isclose(r['J'],J))
     specs=[('cqs_r',source/f'J{J:+.1f}'/'vqe_redq.npz','exact_energy',p['original_redq_final_mean']),('cqs_p',product/f'vqe_J{J:+.1f}.npz','exact_energy',p['final_mean']),('cqs_p_c',PROJECT/c['continuation']/f'redq_J{J:+.1f}.npz','exact_final',cr['methods']['redq']['final_mean']),('moment_gru',PROJECT/c['moment_gru']/f'vqe_J{J:+.1f}.npz','exact_energy',mr['final_mean'])]
     for name,file,key,expected in specs:
      with np.load(file) as a:
       if a['parameters'].shape[1]!=20:raise ValueError('Missing original endpoints')
       check(a['parameters'][0],original_starts,1e-10,1e-10)
       final=a[key][-1] if key=='exact_energy' else a[key];true=np.array([task.energy(z,J,-.5) for z in a['parameters'][-1]])
      check(true,final,2e-8,2e-8);check(float(true.mean()),expected,2e-8,2e-8)
      rows.append(dict(cell=c['name'],task='vqe',case=c['case']['id'],n=c['n'],seed=c['seed'],J=J,method=name,energy=float(true.mean()),endpoints=true.tolist()))
  print('BASELINE AUDITED',c['name'],flush=True)
 result=dict(passed=True,source_identities=identities,physical_spot_checks=physics,rows=rows,no_newmodel_test_scoring=True)
 write(HERE/'baseline_audit.json',result);return result


def check_saved_split(c,folder):
 _,_,tr,va,te=load_data(c)
 with np.load(folder/'split.npz') as saved:
  for key,expected in zip(('train','val','test'),(tr,va,te)):
   if not np.array_equal(saved[key],expected):raise ValueError(('Saved fit split differs from source',c['name'],key))
 if c['task']!='vqe':
  with np.load(PROJECT/c['product']/'split.npz') as product:
   for key,expected in zip(('train','val','test'),(tr,va,te)):
    if not np.array_equal(product[key],expected):raise ValueError(('Original product split differs',c['name'],key))
 return True


def validate_searches(manifest=None,root=None):
 manifest=cells() if manifest is None else manifest;root=HERE/'search' if root is None else root
 expected={root/c['name']/f'J{J:+.1f}.npz':c for c in manifest if c['task']=='vqe' for J in c['case']['couplings']}
 if set(root.glob('*/*.npz'))!=set(expected):raise ValueError('Blind search path set incomplete or unexpected')
 for path,c in expected.items():
  starts=np.load(PROJECT/c['source']/'initial_parameters.npy')
  with np.load(path) as a:
   positions,pred=a['parameters'],a['predictions']
  if positions.shape!=(101,20,*starts.shape[1:]) or pred.shape!=(101,20):raise ValueError('Blind search shape incomplete')
  if not np.isfinite(positions).all() or not np.isfinite(pred).all() or abs(positions).max()>np.pi+1e-12:raise ValueError('Invalid blind search values or bounds')
  if not np.array_equal(positions[0],starts):raise ValueError('Blind search starts differ from main experiment')
 return len(expected)


def audit():
 count=preflight()
 if not(HERE/'all_fits_saved.json').exists() or count!=66:raise ValueError('All66fitsmustfreeze')
 if not read(HERE/'baseline_audit.json')['passed']:raise ValueError('Original baseline audit missing')
 results=[]
 for c in cells():
  identity(c);x,y,tr,va,te=load_data(c);out=HERE/'fits'/c['name'];check_saved_split(c,out);meta=read(out/'training.json');m=selected(c)
  torch.manual_seed(c['seed']);initial=model(c).state_dict();saved=torch.load(out/'initial.pt',map_location='cpu',weights_only=True)
  for key in initial:check(initial[key].numpy(),saved[key].numpy(),0.,0.)
  if sum(p.numel() for p in m.parameters())!=meta['training']['parameters']:raise ValueError('Parameter count changed')
  with torch.no_grad():pred=m.cuda()(torch.tensor(x[va],device='cuda'))[0].cpu().numpy()
  replay=check(pred,np.load(out/'validation_prediction.npy'),1e-7,1e-7)
  check(np.mean((pred-y[va])**2),meta['training']['best_val_mse'],2e-7,2e-6)
  m=m.cpu().double();core=m.core if c['task']=='tfim' else m
  factor_delta=float((core.factors.detach()-saved['core.factors' if c['task']=='tfim' else 'factors'].double()).abs().max())
  gru_delta=float((core.gru.weight_ih.detach()-saved['core.gru.weight_ih' if c['task']=='tfim' else 'gru.weight_ih'].double()).abs().max())
  seen=[]
  def hook(_,args):args[0].retain_grad();seen.append(args[0])
  handle=core.gru.register_forward_pre_hook(hook);direct=core.product_readout.detach().clone()
  with torch.no_grad():core.product_readout.zero_()
  xx=torch.tensor(x[va[:4]],dtype=torch.float64,requires_grad=True);prediction=m(xx)[0];prediction.square().sum().backward();handle.remove()
  def rms(a):return float(a.detach().square().mean().sqrt())
  activity=dict(factor_change=factor_delta,gru_change=gru_delta,factor_gradient=rms(core.factors.grad),gru_gradient=rms(core.gru.weight_ih.grad),current_q_gradient=rms(torch.cat([a.grad[:,:32] for a in seen])),accumulated_z_gradient=rms(torch.cat([a.grad[:,32:64] for a in seen])))
  if not all(np.isfinite(v) and v>0 for v in activity.values()):raise ValueError(('Inactive learned path',c['name'],activity))
  with torch.no_grad():core.product_readout.copy_(direct)
  equivalence=None
  if c['task']=='fspt':
   xx=torch.tensor(x[va[:2]],dtype=torch.float64,requires_grad=True)
   a=m(xx)[0];ga=torch.autograd.grad(a.square().sum(),xx)[0];b=m(xx,independent=True)[0];gb=torch.autograd.grad(b.square().sum(),xx)[0]
   equivalence=dict(prediction=check(a.detach(),b.detach(),1e-11,1e-11),gradient=check(ga,gb,1e-9,1e-9))
  results.append(dict(cell=c['name'],validation_replay_max_error=replay,trained_paths=activity,prefix_equivalence=equivalence))
  print('FULL FIT AUDITED',c['name'],flush=True)
 write(HERE/'verification.json',dict(passed=True,fits=66,before_newmodel_test=True,checks=results))


def search():
 if not read(HERE/'verification.json')['passed']:raise ValueError('Audit required')
 for c in cells():
  if c['task']!='vqe':continue
  m=selected(c).double().cuda()
  for p in m.parameters():p.requires_grad_(False)
  task=task_object(c);starts=np.load(PROJECT/c['source']/'initial_parameters.npy');out=HERE/'search'/c['name'];out.mkdir(parents=True,exist_ok=True)
  for J in c['case']['couplings']:
   path=out/f'J{J:+.1f}.npz'
   if path.exists():continue
   x=torch.tensor(starts,dtype=torch.float64,device='cuda',requires_grad=True);opt=torch.optim.Adam([x],lr=.1,foreach=False);coef=torch.tensor(task.coefficients(J,-.5),device='cuda',dtype=torch.float64)
   positions=[];predictions=[]
   for step in range(101):
    opt.zero_grad(set_to_none=True);p=m(x)[0]@coef
    if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite search')
    positions.append(x.detach().cpu().numpy().copy());predictions.append(p.detach().cpu().numpy())
    if step==100:break
    p.sum().backward()
    if not torch.isfinite(x.grad).all():raise FloatingPointError('Nonfinite search gradient')
    opt.step()
    with torch.no_grad():x.clamp_(-np.pi,np.pi)
   np.savez_compressed(path,parameters=np.array(positions),predictions=np.array(predictions))
   print('BLIND HYBRID SEARCH SAVED',c['name'],J,flush=True)
 expected=validate_searches()
 write(HERE/'all_searches_saved.json',dict(searches=expected,endpoints_per_search=20,before_exact_scoring=True))


def stats(values):
 values=np.asarray(values,float);return dict(mean=float(values.mean()),sd=float(values.std(ddof=1)),values=values.tolist())


def score():
 if not read(HERE/'verification.json')['passed'] or not read(HERE/'all_searches_saved.json')['before_exact_scoring']:raise ValueError('Fit/audit/search gates required')
 validate_searches()
 rows=read(HERE/'baseline_audit.json')['rows'].copy()
 if (HERE/'strong_baseline_audit.json').exists():
  extra=read(HERE/'strong_baseline_audit.json')
  if not extra['passed']:raise ValueError('Strong reference audit failed')
  rows.extend(extra['rows'])
 for c in cells():
  x,y,tr,va,te=load_data(c);m=selected(c).cuda()
  with torch.no_grad():pred=np.concatenate([m(torch.tensor(x[ids],device='cuda'))[0].cpu().numpy() for ids in np.array_split(te,max(1,int(np.ceil(len(te)/32))))])
  np.save(HERE/'fits'/c['name']/'test_prediction.npy',pred)
  with np.load(PROJECT/c['data']) as data:
   if c['task']=='vqe':
    target=data['exact_terms'][te];task=task_object(c)
    for J in c['case']['couplings']:
     with np.load(HERE/'search'/c['name']/f'J{J:+.1f}.npz') as a:
      final=np.array([task.energy(z,J,-.5) for z in a['parameters'][-1]]);initial=np.array([task.energy(z,J,-.5) for z in a['parameters'][0]])
     rows.append(dict(cell=c['name'],task='vqe',case=c['case']['id'],n=c['n'],seed=c['seed'],J=J,method='hybrid',energy=float(final.mean()),endpoints=final.tolist(),initial_energy=float(initial.mean()),moment_mse=metrics(target,pred)['mse'],**metrics(target@task.coefficients(J,-.5),pred@task.coefficients(J,-.5))))
   else:
    target=data['y'][te,:c['k']] if c['task']=='fspt' else data['y_exact'][te,None]
    row=dict(cell=c['name'],task=c['task'],n=c['n'],seed=c['seed'],method='hybrid',**metrics(target,pred))
    if c['task']=='fspt':row.update(k=c['k'],per_period_mae=np.mean(abs(pred.astype(float)-target),axis=(0,2)).tolist())
    rows.append(row)
  print('MAIN TEST SCORED',c['name'],flush=True)
 groups={}
 for row in rows:
  key=('vqe',row['case'],row['J']) if row['task']=='vqe' else ('fspt',row['n'],row['k']) if row['task']=='fspt' else ('tfim',row['n'])
  groups.setdefault(key,{}).setdefault(row['method'],[]).append(row)
 summary=[]
 for key,methods in groups.items():
  metric='energy' if key[0]=='vqe' else 'mae';values={name:stats([r[metric] for r in rr]) for name,rr in methods.items()}
  if any(len(r['values'])!=3 for r in values.values()):raise ValueError('Missing paired run')
  wins={name:sum(next(r for r in methods['hybrid'] if r['seed']==b['seed'])[metric]<b[metric] for b in rr) for name,rr in methods.items() if name!='hybrid'}
  summary.append(dict(key=key,metric=metric,methods=values,hybrid_lower_runs=wins))
 write(HERE/'rows.json',rows);write(HERE/'summary.json',dict(groups=summary,full_fits=66,new_measurements=0,hardware_shots=0));report(summary)
 write(HERE/'complete.json',dict(status='full_main_experiment_complete',fits=66,finished_at=time.time()))


def report(summary):
 lines=['# 架构融合模型：论文主实验完整比较','','本实验沿用主实验原有数据、训练/验证/测试划分、监督标签与检查点选择。新增66个完整融合拟合，分别为12个TFIM、36个FSPT及18个VQE。全部拟合核验和33组盲搜索保存后才评价新模型测试指标及精确终点。旧120秒可行性实验单独保留。','',
 '主实验TFIM/FSPT的三个运行是同一数据池的三次配对划分，不称三套新独立数据。VQE采用原主表三个独立数据集。数值为三次运行均值±样本标准差，保留三位小数，不报告置信区间。','']
 for family in ('tfim','fspt','vqe'):
  lines+=['## '+family.upper(),'']
  names=('cqs_r','cqs_p','hybrid') if family!='vqe' else ('cqs_r','moment_gru','cqs_p','cqs_p_c','hybrid')
  lines+=['| 设置 | '+' | '.join(names)+' | 融合优于R/P的运行数 |','| --- | '+' | '.join('---:' for _ in names)+' | --- |']
  for g in summary:
   if g['key'][0]!=family or family=='vqe' and not np.isclose(g['key'][2],-.1):continue
   setting=','.join(map(str,g['key'][1:]));vals=[f"{g['methods'][n]['mean']:.3f} ± {g['methods'][n]['sd']:.3f}" for n in names]
   lines.append('| '+setting+' | '+' | '.join(vals)+f" | {g['hybrid_lower_runs']['cqs_r']}/3，{g['hybrid_lower_runs']['cqs_p']}/3 |")
  lines+=['']
 strong=[g for g in summary if g['key'][0]=='fspt' and 'cqs_r_curriculum' in g['methods']]
 for g in strong:
  a=g['methods']['cqs_r_curriculum'];b=g['methods']['hybrid']
  lines += [f"更强的已发表CQS-R分阶段参考，N={g['key'][1]},K=24：12周期预训练加24周期拟合的MAE为{a['mean']:.3f} ± {a['sd']:.3f}，本次单阶段融合模型为{b['mean']:.3f} ± {b['sd']:.3f}。该参考额外计入最多300轮12周期预训练，独立于上表原始300轮矩阵。",'']
 lines+=['VQE融合模型按与原CQS-R一致的100步直接优化执行，使用全部20个相同原始初值。其测量矩监督与CQS-P一致，因此另列原moment-GRU作为监督匹配的参考。原moment-GRU网格使用300轮上限及30轮耐心值，融合模型网格使用3000轮上限、至少500轮及100轮耐心值，因此两者不构成训练预算匹配的纯架构消融。CQS-P直接优化与+C分别列出；融合GRU不自动继承解析Gaussian continuation。网格融合训练与原CQS-P匹配3000轮上限、至少500轮及100轮耐心值，链300轮、耐心值30。TFIM为250轮、FSPT为300轮、耐心值均30。不是统一墙钟预算比较，参数量及实际训练成本见各拟合training.json。','',
 '融合模型保持rank32、隐藏宽度64，近恒等因子初始化。FSPT使用固定前缀支持掩码并集；TFIM沿用原电路逐层Hadamard掩码并仅接受标量能量标签；VQE沿用原初始plus态电路掩码与测量矩标签。所有原始基线从保存预测和端点重新核验，不为此比较重新拟合。所有耦合的逐运行结果保存在rows.json。']
 primary_tfim=[g for g in summary if g['key'][0]=='tfim']
 primary_fspt=[g for g in summary if g['key'][0]=='fspt']
 tw=sum(g['methods']['hybrid']['mean']<min(g['methods']['cqs_r']['mean'],g['methods']['cqs_p']['mean']) for g in primary_tfim)
 rw=sum(g['methods']['hybrid']['mean']<g['methods']['cqs_r']['mean'] for g in primary_fspt)
 pw=sum(g['methods']['hybrid']['mean']<g['methods']['cqs_p']['mean'] for g in primary_fspt)
 lines[2:2]=[f'本次完整主实验的结论：融合模型在{tw}/4个TFIM设置取得最低平均MAE；在{pw}/12个FSPT设置优于CQS-P、{rw}/12个设置优于原始CQS-R。八/十比特24周期精度接近已有分阶段CQS-R。VQE直接优化未超过CQS-P+C，预测收益与continuation的优化收益分别报告。','']
 (HERE/'实验结果.md').write_text('\n'.join(lines)+'\n')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('baselines','audit','search','score'));args=ap.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 globals()[args.stage]()

if __name__=='__main__':main()
