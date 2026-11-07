"""Replay the R3 numerical evidence from released data, checkpoints and traces."""
from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
EXP=ROOT/'paper_iclr2027/rebuttal_20261107'
SUITES={
 'hardware':[("-c","import run_evidence as r;r.hardware();r.constants();r.endpoint_diagnostics()")],
 'w1':[("analyze_W1_existing.py",)],
 'w2':[("analyze_W2.py","--curriculum")],
 'query':[("analyze_resource_query.py",)],
 'chain':[("analyze_chain_depth_recovery.py",),("analyze_chain_mechanism.py",),("analyze_resource_chain.py",)],
 'chain-fixed':[("reviewer_weak_reject_2/fixed_data_chain_20261108/analyze_confirmation.py",)],
}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--suite',choices=['all',*SUITES],default='all');args=ap.parse_args()
    subprocess.run([sys.executable,str(ROOT/'fetch_artifacts.py'),'--check'],check=True,cwd=ROOT)
    runs=[]
    for name in (SUITES if args.suite=='all' else [args.suite]):
        for job in SUITES[name]:
            command=[sys.executable,*job];start=time.perf_counter()
            print(f'=== {name}: {job[0]} ===',flush=True)
            subprocess.run(command,check=True,cwd=EXP)
            runs.append({'suite':name,'script':job[0],'passed':True,'seconds':time.perf_counter()-start})
    report={'passed':True,'mode':'Replay from released datasets/checkpoints/traces','runs':runs}
    (ROOT/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('All requested replay suites passed.')
if __name__=='__main__':main()
