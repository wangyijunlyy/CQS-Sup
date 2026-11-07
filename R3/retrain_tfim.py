"""Refit the validation-tuned TFIM NC control from the released measured labels."""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'world_model_surrogate/scripts'),str(ROOT/'world_model_surrogate')]
from run_table1_tuned_control import load_inputs,run_one

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n',type=int,choices=(8,10,12,14));ap.add_argument('--seed',type=int,choices=(42,43,44))
    ap.add_argument('--check-inputs',action='store_true');args=ap.parse_args()
    ns=[args.n] if args.n else [8,10,12,14];seeds=[args.seed] if args.seed else [42,43,44]
    output=ROOT/'outputs/tfim';output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in ns:
        for seed in seeds:
            if args.check_inputs:load_inputs(n,seed);print('Validated measured labels, splits, coefficients and references:',n,seed,flush=True)
            else:rows.append(run_one(n,seed,output))
    if not args.check_inputs:(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
