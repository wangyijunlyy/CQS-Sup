"""Replay saved reuse evidence, or rerun every workload with the frozen models."""
from pathlib import Path
import argparse,concurrent.futures,json,shutil,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
REL=HERE.relative_to(ROOT);BASE=REL.parent/'JJuz_followup_20261117'

def post(root):
 for name in ['replay_parallel.py','check_aux.py','analyze.py']:
  subprocess.run([sys.executable,str(root/REL/name)],cwd=root,check=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fresh',action='store_true');p.add_argument('--out',type=Path,default=ROOT/'fresh_reuse');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
 if not a.fresh:post(ROOT);return
 out=a.out.resolve()
 if out==ROOT or (out.exists() and any(out.iterdir())):raise SystemExit('Use a separate empty output directory.')
 out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((ROOT/'code_manifest.json').read_text())
 for row in manifest['files']:
  rel=Path(row['file']);target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
 for kind,start in [('grid',17101),('chain',17201)]:
  for seed in range(start,start+10):
   for name in ['dataset.npz','product.pt']:
    rel=BASE/kind/f'seed{seed}'/name;(out/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,out/rel)
 for name in ['protocol.json','frozen_selection.json']:
  shutil.copyfile(ROOT/REL/name,out/REL/name)
 jobs=[]
 for kind,start in [('grid',17101),('chain',17201)]:
  for seed in range(start,start+10):
   for order in ['smooth','jumping']:
    pre=['--kind',kind,'--seed',str(seed),'--order',order];jobs.append(pre+['--cqs'])
    jobs.extend(pre+['--regime',regime,'--cap',str(cap)] for regime in ['low','shared'] for cap in [0,160000,640000,1280000])
 def run(args):subprocess.run([sys.executable,str(out/REL/'run_reuse.py'),*args],cwd=out,check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for _ in pool.map(run,jobs):pass
 (out/REL/'all_saved.json').write_text(json.dumps(dict(complete=True,sequences=360,exact_scoring_started=False,time_utc=time.time())))
 post(out)

if __name__=='__main__':main()
