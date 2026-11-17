"""Post-search endpoint evaluation and dataset-level cost/quality summaries."""
from pathlib import Path
import csv,json,sys,time
import numpy as np
from scipy.stats import t
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from run_reuse import ORDERS,BASE,CAPS
from engine import PhysicalSampler,save,write
from run_evidence import TFIMTask

def describe(x):
 a=np.asarray(x,float);mean=float(a.mean());sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
 return dict(mean=mean,sd=sd,n=len(a),ci95=[mean-half,mean+half],values=a.tolist())
def csvout(path,rows):
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
 assert json.loads((HERE/'audit.json').read_text())['passed']
 write(HERE/'scoring_started.json',dict(time_utc=time.time(),all_workloads_saved=True,chronological_audit_passed=True))
 rows=[];prefix=[];maxerror=0.;checks=0
 for path in sorted(p.parent for p in (HERE/'runs').rglob('complete.json')):
  info=json.loads((path/'complete.json').read_text());kind=info['kind'];seed=info['seed'];order=info['order'];method=path.name
  engine=PhysicalSampler(kind,-.5);allparameters=[]
  for target in range(10):
   with np.load(path/f'target{target:02d}.npz') as z:allparameters.append(z['parameters'])
  parameters=np.concatenate(allparameters);pz,px=engine.probabilities(parameters)
  u=px@engine.signs.sum(1);v=pz@engine.zz.sum(1);j=np.repeat(ORDERS[order],20)
  energy=((-0.5 if kind=='grid' else .5)*u+(j if kind=='grid' else -j)*v).reshape(10,20)
  task=TFIMTask(3,3,2,True,1) if kind=='grid' else TFIMTask(1,6,2,False,-1)
  for target in [0,9]:
   for endpoint in [0,19]:
    exact=task.energy(allparameters[target][endpoint],ORDERS[order][target],-.5)
    maxerror=max(maxerror,abs(float(energy[target,endpoint])-exact));checks+=1
  save(path/'posthoc_scores.npz',energy=energy,J=np.array(ORDERS[order]))
  for r,e in zip(info['rows'],energy):
   rows.append(dict(kind=kind,seed=seed,order=order,method=method,target=r['target'],J=r['J'],mean_energy=float(e.mean()),endpoint_sd=float(e.std(ddof=1)),**{key:r[key] for key in ['initial_shots','online_shots','cumulative_shots','cumulative_query_controls','feedback_rounds','online_feedback_rounds','cache_hits']}))
  for m in [1,2,5,10]:
   r=info['rows'][m-1]
   prefix.append(dict(kind=kind,seed=seed,order=order,method=method,prefix=m,mean_energy=float(energy[:m].mean()),**{key:r[key] for key in ['initial_shots','online_shots','cumulative_shots','cumulative_query_controls','feedback_rounds','online_feedback_rounds','cache_hits']}))
 assert maxerror<1e-11
 summary=[];contrasts=[];lookup={}
 for key in sorted(set((r['kind'],r['order'],r['method'],r['prefix']) for r in prefix)):
  kind,order,method,m=key;group=sorted([r for r in prefix if (r['kind'],r['order'],r['method'],r['prefix'])==key],key=lambda r:r['seed']);assert len(group)==10
  row=dict(kind=kind,order=order,method=method,prefix=m,energy=describe([r['mean_energy'] for r in group]))
  for col in ['initial_shots','online_shots','cumulative_shots','cumulative_query_controls','feedback_rounds','online_feedback_rounds','cache_hits']:row[col]=describe([r[col] for r in group])
  summary.append(row);lookup[key]=row
 for kind in ['grid','chain']:
  for order in ORDERS:
   for m in [1,2,5,10]:
    cqs=lookup[kind,order,'cqs',m]['energy']['values']
    for regime in ['low','shared']:
     for cap in CAPS:
      other=lookup[kind,order,f'{regime}_cap{cap}',m]['energy']['values'];delta=np.array(cqs)-other
      contrasts.append(dict(kind=kind,order=order,prefix=m,comparison=f'cqs-minus-{regime}_cap{cap}',paired_cqs_wins=int((delta<0).sum()),**describe(delta)))
 frontier=[]
 for kind in ['grid','chain']:
  for order in ORDERS:
   for m in [1,2,5,10]:
    cqs=lookup[kind,order,'cqs',m];reference=cqs['energy']['mean']
    for regime in ['low','shared']:
     candidates=[lookup[kind,order,f'{regime}_cap{cap}',m] for cap in CAPS]
     qualifying=[r for r in candidates if r['energy']['mean']<=reference+.02]
     best=min(qualifying,key=lambda r:r['cumulative_shots']['mean']) if qualifying else None
     frontier.append(dict(kind=kind,order=order,prefix=m,regime=regime,cqs_energy=reference,cqs_shots=cqs['cumulative_shots']['mean'],spsa_qualifying_method=best['method'] if best else None,spsa_energy=best['energy']['mean'] if best else None,spsa_shots=best['cumulative_shots']['mean'] if best else None,spsa_feedback_rounds=best['feedback_rounds']['mean'] if best else None,cqs_shot_saving_vs_qualifying=1-cqs['cumulative_shots']['mean']/best['cumulative_shots']['mean'] if best and best['cumulative_shots']['mean'] else None))
 csvout(HERE/'per_target.csv',rows);csvout(HERE/'prefix_by_dataset.csv',prefix)
 write(HERE/'summary.json',summary);write(HERE/'paired_contrasts.json',contrasts);write(HERE/'descriptive_frontier.json',frontier)
 write(HERE/'verification.json',dict(passed=True,sequences=360,hamiltonian_outputs=len(rows),retained_endpoints=len(rows)*20,independent_global_fits=20,refits=0,qiskit_checks=checks,qiskit_energy_max_error=maxerror,hardware_shots=0,uncertainty='Unadjusted descriptive t95 intervals across10 independent existing dataset/model fits. Endpoints and Hamiltonians are not treated as independent replicates.',evaluation_cost='Post-hoc noiseless simulation only. Final hardware certification measurements excluded for both methods.',frontier='Grid of independently rerun per-target caps, compared after search. Not oracle-guided termination or proof of optimal SPSA cost.'))
 lines=['# 同一代理跨哈密顿量复用：完整结果','',
  '设置：九比特两层3×3网格与六比特两层一维链，分别使用此前独立训练的10个完整CQS-P模型，不重新拟合。固定h=−0.5，J取−0.05至−0.50的10个不同值。平滑顺序逐渐增强耦合，跳变顺序交替改变耦合强度。每个任务保留全部20个优化终点。以下误差按独立数据集统计，数值越低越好。','',
  'SPSA可重加权其全部历史X/ZZ测量，使用历史最优置信上界控制与上一任务终点热启动，并缓存重复控制。low初始池为网格40,000 shots或链式解析参考点；shared获得CQS全部初始测量。各顺序、初始池和预算组独立运行，零预算组仅从历史测量选点。新增测量精度在旧开发数据45、46上按独立读出选择，在正式运行前冻结。','',
  'CQS初始测量计入累计成本，网格为4,000,000、链为24,000 shots。同一模型依次优化10个哈密顿量，全程不更新模型、不新增查询。SPSA每个任务在线shots上限为0、160,000、640,000或1,280,000。开发测量单独记录，不计入正式任务成本。','',
  '反馈轮数指必须等待测量结果才能继续的逻辑批次。SPSA两个扰动方向、20条轨迹和两个测量组可在同一批次执行。固定初始采集计1轮，缓存读取计0轮。这不是实测硬件作业数或延迟。','']
 labels={'cqs':'固定CQS-P','low_cap0':'SPSA-low，仅历史','shared_cap0':'SPSA-shared，仅历史'}
 for kind in ['grid','chain']:
  for order in ORDERS:
   lines.extend([f'## {kind} / {order}：10个哈密顿量累计结果','', '| 方法 | 平均能量 ± 数据集间SD | 累计shots | 反馈轮数 |', '|---|---:|---:|---:|'])
   for method in ['cqs']+[f'{r}_cap{b}' for r in ['low','shared'] for b in CAPS]:
    r=lookup[kind,order,method,10];label=labels.get(method,method.replace('_cap','，每任务预算'))
    lines.append(f"| {label} | {r['energy']['mean']:.3f} ± {r['energy']['sd']:.3f} | {r['cumulative_shots']['mean']:,.0f} | {r['feedback_rounds']['mean']:.1f} |")
   lines.extend(['','### 随任务数增长的质量与成本','', '| 方法 | 任务数 | 平均能量 | 累计shots | 反馈轮数 |','|---|---:|---:|---:|---:|'])
   for m in [1,2,5,10]:
    for method in ['cqs','low_cap640000','shared_cap640000']:
     r=lookup[kind,order,method,m];lines.append(f"| {labels.get(method,method)} | {m} | {r['energy']['mean']:.3f} | {r['cumulative_shots']['mean']:,.0f} | {r['feedback_rounds']['mean']:.1f} |")
   lines.append('')
 lines.extend(['## 使用边界','',
  '此实验直接测试同一观测量集合、不同系数的TFIM家族，不代表改变测量基或电路结构后的任意哈密顿量迁移。初始模型来自此前实验，因此这是追加复用工作负载，不是新训练样本。所有正式任务在精确评分前完成，精确能量只用于事后统一评估。没有使用真实量子硬件。',
  '', '固定代理在线查询为零只说明交互被转移到预先采集与经典优化，不能单独证明同等精度下节省shots。必须同时比较能量，尤其检查零预算历史选点和允许复用的SPSA。质量匹配前沿采用平均能量容差0.020，是事后描述性分析，不是以精确能量指导SPSA停止。更低成本处仍可能存在未测试的更好配置。',
  '', '复现：`audit.py`重建全部测量历史、参数更新、保留决策和资源计数；`analyze.py`对全部72,000个终点评分并生成CSV。`per_target.csv`保留逐任务结果，`prefix_by_dataset.csv`保留独立数据集统计单位。'])
 (HERE/'results_report.md').write_text('\n'.join(lines)+'\n')
 write(HERE/'status.json',dict(stage='complete',all_audits_passed=True))
 print('COMPLETE',len(rows)*20,'endpoints; Qiskit max error',maxerror,flush=True)
 for row in summary:
  if row['prefix']==10:print(row['kind'],row['order'],row['method'],round(row['energy']['mean'],6),row['cumulative_shots']['mean'],row['feedback_rounds']['mean'],flush=True)

if __name__=='__main__':main()
