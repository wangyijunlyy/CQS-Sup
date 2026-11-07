"""Replay the experiments supporting the R1 author response."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
EXP=Path('paper_iclr2027/rebuttal_20261107')
EXT=EXP/'reviewer_weak_reject_2/extensions_20261108'
FIXED=EXP/'reviewer_weak_reject_2/fixed_data_chain_20261108'
LOCAL=EXP/'reviewer_weak_reject_2/calibrated_budget_20261108'
SUITES={
 'masks':[(Path('replay_evidence.py'),['masks'])],
 'common':[(Path('replay_evidence.py'),['common'])],
 'parallel':[(Path('replay_evidence.py'),['parallel'])],
 'query':[(EXP/'analyze_resource_query.py',[])],
 'fixed-chain':[(FIXED/'analyze_confirmation.py',[])],
 'reference':[(Path('replay_evidence.py'),['reference'])],
 'grid-continuation':[(Path('replay_evidence.py'),['grid_continuation'])],
 'local-chain':[(EXP/'analyze_chain_depth_recovery.py',[]),(EXP/'analyze_resource_chain.py',[]),
    (EXT/'analyze_fourier_chain.py',[]),(EXT/'analyze_direct_jitter.py',[])],
 'local-confirmation':[(LOCAL/'analyze.py',[])],
 'strong-grid':[(Path('replay_evidence.py'),['strong_grid'])],
 'timing':[(Path('replay_evidence.py'),['timing'])],
}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--suite',choices=['all',*SUITES],default='all')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu',help='Device for the parallel checkpoint replay');args=ap.parse_args()
    subprocess.run([sys.executable,str(ROOT/'fetch_artifacts.py'),'--check'],check=True)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[key]='2'
    runs=[]
    for name in (SUITES if args.suite=='all' else [args.suite]):
        for script,arguments in SUITES[name]:
            start=time.perf_counter();print('RUN',name,script.name,flush=True)
            extra=['--device',args.device] if name=='parallel' else []
            subprocess.run([sys.executable,str(ROOT/script),*arguments,*extra],cwd=(ROOT/script).parent,env=env,check=True)
            runs.append(dict(suite=name,script=str(script),passed=True,seconds=time.perf_counter()-start))
    (ROOT/'reproduction_report.json').write_text(json.dumps(dict(passed=True,mode='numerical replay of published runs',runs=runs),indent=2)+'\n')
    print('All requested R1 replay suites passed.',flush=True)
if __name__=='__main__':main()
