"""Develop finite-shot precision, freeze choices, execute isolated reuse workloads."""
from pathlib import Path
import concurrent.futures,hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_reuse import ORDERS,CAPS,data,BASE
from engine import write
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')

def execute(args):
 label='_'.join(str(x).replace('--','') for x in args)
 with (HERE/'logs'/f'{label}.log').open('a') as log:
  result=subprocess.run([sys.executable,'-u',str(HERE/'run_reuse.py'),*map(str,args)],stdout=log,stderr=subprocess.STDOUT,env=ENV)
 if result.returncode:raise RuntimeError('Worker failed: '+label)
 return label

def batch(jobs,stage):
 write(HERE/'status.json',dict(stage=stage,total=len(jobs),completed=0,time_utc=time.time()))
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
  futures=[pool.submit(execute,job) for job in jobs]
  for done,f in enumerate(concurrent.futures.as_completed(futures),1):
   print(stage,done,len(jobs),f.result(),flush=True)
   write(HERE/'status.json',dict(stage=stage,total=len(jobs),completed=done,time_utc=time.time()))

def main():
 (HERE/'logs').mkdir(exist_ok=True)
 if not (HERE/'protocol.json').exists():
  write(HERE/'protocol.json',dict(version='sequential-hamiltonian-reuse-v2',created_utc=time.time(),orders=ORDERS,caps_per_target=CAPS,h=-.5,
   tasks=dict(grid=dict(qubits=9,layers=2,controls=3,seeds=list(range(17101,17111)),initial_cqs=4000000,initial_spsa_low=40000),chain=dict(qubits=6,layers=2,controls=16,seeds=list(range(17201,17211)),initial_cqs=24000,initial_spsa_low=0)),
   datasets='Existing independent full fits from JJuz_followup_20261117, unchanged weights. Additional workload, not a fresh training cohort.',
   regimes=['low','shared'],outputs_per_target=20,history='Reweight all past measured X-sum/ZZ-sum statistics, exact-control cache with precision topups. Best historical mean+2SE control includes previous endpoints, with19 perturbed starts on every new target and incumbent retention. Each regime/order/budget owns an isolated history.',
   development=dict(seeds=[45,46],order='smooth',cap=640000,shots_per_group_candidates=[100,500],selection='Independent post-sequence 1000-shot/group readout of all200 retained endpoints; never used for search or warmstarts.',gain_c='Previously frozen grid .05/.05, chain .2/.2'),
   cqs=dict(steps=100,learning_rate=.1,refits=0,online_shots=0),
   costs='Initial acquisition charged once; all new shots charged. Both SPSA probes, trajectories and commuting groups batched per feedback round. Logical dependency rounds, not real backend jobs/latency.',
   scoring='Save all workloads before post-hoc exact evaluation. Dataset-level means over20 endpoints, then prefixes1/2/5/10 Hamiltonians. Descriptive budget frontier tolerance .02, not oracle stopping.',hardware_shots=0))
 if not(HERE/'frozen_selection.json').exists():
  for kind in ['grid','chain']:
   for seed in [45,46]:data(kind,seed,True)
  jobs=[['--kind',k,'--seed',s,'--regime',r,'--dev','--shots',n] for k in ['grid','chain'] for r in ['low','shared'] for n in [100,500] for s in [45,46]]
  batch(jobs,'development')
  selection={}
  for kind in ['grid','chain']:
   for regime in ['low','shared']:
    candidates=[]
    for n in [100,500]:
     records=[json.loads((HERE/'development'/kind/f'seed{s}'/f'{regime}_shots{n}'/'complete.json').read_text()) for s in [45,46]]
     scores=[r['validation']['mean_independent_measured_energy'] for r in records]
     candidates.append(dict(config=records[0]['config'],scores=scores,mean=sum(scores)/len(scores)))
    selection[kind+'_'+regime]=dict(config=min(candidates,key=lambda x:x['mean'])['config'],candidates=candidates)
  selection['frozen_utc']=time.time();selection['exact_values_used']=False
  selection['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
  selection['dependency_hashes']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'engine.py']}
  selection['model_and_data_hashes']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for kind,start in [('grid',17101),('chain',17201)] for s in range(start,start+10) for p in [BASE/kind/f'seed{s}'/'product.pt',BASE/kind/f'seed{s}'/'dataset.npz']}
  write(HERE/'frozen_selection.json',selection)
 jobs=[]
 for kind,start in [('grid',17101),('chain',17201)]:
  for seed in range(start,start+10):
   for order in ORDERS:
    prefix=['--kind',kind,'--seed',seed,'--order',order]
    jobs.append(prefix+['--cqs'])
    jobs.extend(prefix+['--regime',regime,'--cap',cap] for regime in ['low','shared'] for cap in CAPS)
 batch(jobs,'formal')
 write(HERE/'all_saved.json',dict(complete=True,sequences=len(jobs),targets=len(jobs)*10,endpoints=len(jobs)*200,exact_scoring_started=False,time_utc=time.time()))
 write(HERE/'status.json',dict(stage='all_saved',sequences=len(jobs)))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  write(HERE/'status.json',dict(stage='failed',error=str(exc)));raise
