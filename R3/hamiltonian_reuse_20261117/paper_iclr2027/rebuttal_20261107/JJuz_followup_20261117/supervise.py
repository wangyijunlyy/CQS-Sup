"""Run old-seed development, freeze measured selection, then full fresh datasets."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from engine import write
from direct import SPSA_CONFIGS,COBYLA_CONFIGS
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')

def execute(job):
 task,seed,method,index=job;label=f'{task}_{seed}_{method}_{index}'
 cmd=[sys.executable,'-u',str(HERE/'worker.py'),'--task',task,'--seed',str(seed),'--method',method,'--index',str(index)]
 with (HERE/'logs'/f'{label}.log').open('a') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=ENV)
 if r.returncode:raise RuntimeError(f'Worker failed: {label}')
 return label

def main():
 (HERE/'logs').mkdir(exist_ok=True)
 protocol=dict(version='jjuz-matched-budget-v1',created_utc=time.time(),development_seeds=[45,46],grid_confirmation_seeds=list(range(17101,17111)),chain_confirmation_seeds=list(range(17201,17211)),
  grid=dict(qubits=9,layers=2,effective_controls=3,J=-.5,h=-.5,train=160,validation=40,shots_per_group=10000,total_acquisition=4000000,common_pilot_controls=20,common_pilot_shots_per_group=1000,pilot_cost_included=40000,starts=20,start_jitter_std=.05,product_rank=32,product_order=4,product_epochs=3000,minimum_epochs=500,patience=100,methods=['product','fourier','spsa','cobyla']),
  chain=dict(qubits=6,layers=2,effective_controls=16,couplings=[-.5,.5],train=2000,validation=400,random_pauli_shots=10,global_acquisition=24000,rank=32,order=1,epoch_cap=300,patience=30,local_rounds=3,local_shots_per_group=1000,local_training_controls=192,local_validation_controls=64,methods=['product','local_only','spsa'],checkpoints=[578000,1132000,1686000],initialization='reference plus19 shared perturbations within+-0.2; full trajectories retained on rejection',local_only_saved_acquisition='extra1000/group on12 first-round training controls:24000 shots'),
  configurations=dict(spsa=SPSA_CONFIGS,cobyla=COBYLA_CONFIGS,grid_acceptance_z=0,chain_acceptance_z=2),selection='Mean final independently measured incumbent energy across development datasets; chain equally weights both couplings. No exact values used.',
  primary='Mean true energy across all20 retained endpoints within each dataset; paired contrasts across10 independent full datasets,95% t intervals; no winner seed filtering',
  final_scoring='Separate phase after all development decisions and all confirmation searches saved',hardware_shots=0)
 if not(HERE/'frozen_selection.json').exists():write(HERE/'protocol.json',protocol)
 dev=[]
 for seed in [45,46]:
  dev += [('dev-grid',seed,'spsa',i) for i in range(len(SPSA_CONFIGS))]
  dev += [('dev-grid',seed,'cobyla',i) for i in range(len(COBYLA_CONFIGS))]
  dev += [('dev-chain',seed,'spsa',i) for i in range(len(SPSA_CONFIGS))]
 if not(HERE/'frozen_selection.json').exists():
  write(HERE/'status.json',dict(stage='development',jobs=len(dev)))
  with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
   for done in pool.map(execute,dev):print('DEV DONE',done,flush=True)
  selection={}
  for task,method,candidates in [('grid','spsa',SPSA_CONFIGS),('grid','cobyla',COBYLA_CONFIGS),('chain','spsa',SPSA_CONFIGS)]:
   values=[]
   for index,config in enumerate(candidates):
    scores=[]
    for seed in [45,46]:
     root=HERE/'development'/task/f'seed{seed}'/f'{method}{index}'
     paths=[root/'complete.json'] if task=='grid' else [root/f'J{j:+.1f}'/'complete.json' for j in [-.5,.5]]
     for p in paths:scores.append(json.loads(p.read_text())['points'][-1]['measured_score'])
    values.append(dict(index=index,config=config,mean_measured_score=sum(scores)/len(scores),scores=scores))
   best=min(values,key=lambda r:r['mean_measured_score'])
   selection[task+'_'+method]=dict(config=best['config'],index=best['index'],candidates=values)
  selection['frozen_utc']=time.time();selection['exact_values_used']=False
  selection['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
  write(HERE/'frozen_selection.json',selection)
 jobs=[('grid',s,'none',0) for s in range(17101,17111)]+[('chain',s,'none',0) for s in range(17201,17211)]
 write(HERE/'status.json',dict(stage='confirmation',jobs=len(jobs),source_and_selection_frozen=True))
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  for done in pool.map(execute,jobs):print('CONFIRM DONE',done,flush=True)
 write(HERE/'all_saved.json',dict(grid_datasets=10,chain_datasets=10,complete=True,exact_scoring_started=False,time_utc=time.time()))
 write(HERE/'status.json',dict(stage='all searches saved',exact_scoring_started=False))
 print('ALL CONFIRMATION OUTPUTS SAVED',flush=True)

if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(HERE/'status.json',dict(stage='failed',error=str(exc)));raise
