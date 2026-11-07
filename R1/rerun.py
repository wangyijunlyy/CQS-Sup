"""Repeat complete fits/searches in a separate output directory."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
EXP=ROOT/'paper_iclr2027/rebuttal_20261107'
REVIEW=EXP/'reviewer_weak_reject_2'
WORLD=ROOT/'world_model_surrogate'
def command(script,*arguments):
    subprocess.run([sys.executable,str(script),*map(str,arguments)],cwd=script.parent,check=True)
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment',choices=['masks','common','parallel','query','fixed-chain','local-chain','local-confirmation','strong-grid','grid-continuation'])
    ap.add_argument('--out',type=Path,required=True,help='New output location; complete existing outputs resume')
    ap.add_argument('--seeds',type=int,nargs='+',help='Subset of the published seeds, where applicable')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(EXP),str(WORLD/'scripts'),str(WORLD)]
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # The wrappers change destinations only. Scientific grids and seeds remain
    # in their original modules and protocols.
    if args.experiment in ('fixed-chain','local-confirmation'):
        directory=REVIEW/('fixed_data_chain_20261108' if args.experiment=='fixed-chain' else 'calibrated_budget_20261108')
        sys.path.insert(0,str(directory))
        if args.experiment=='fixed-chain':
            import confirm as module
            shutil.copy2(directory/'confirmation_protocol.json',out/'confirmation_protocol.json')
            module.HERE=out;seeds=args.seeds or list(range(3101,3111))
            assert set(seeds)<=set(range(3101,3111))
            for seed in seeds:module.run(seed)
            if all((out/f'confirmation/seed{s}/all_saved.json').exists() for s in range(3101,3111)):
                import analyze_confirmation
                analyze_confirmation.main()
        else:
            import worker as module
            for name in ('protocol.json','timing_cap.json'):shutil.copy2(directory/name,out/name)
            module.HERE=out;seeds=args.seeds or list(range(2101,2106));assert set(seeds)<=set(module.SEEDS)
            for seed in seeds:
                sys.argv=['worker.py','--seed',str(seed)];module.main()
        return
    if args.experiment=='common':
        import run_W1_common_penalty as module
        module.DEST=out/'common_penalty';sys.argv=['run_W1_common_penalty.py'];module.main()
        import refine_W1_common_penalty
        sys.argv=['refine_W1_common_penalty.py'];refine_W1_common_penalty.main();return
    if args.experiment=='strong-grid':
        import confirm_warmstart as module
        module.FOLDER=out;module.main();return
    if args.experiment=='local-chain':
        import chain_local_calibration as module
        module.OUT=out;module.run();return
    if args.experiment=='grid-continuation':
        import run_vqe_fourier_controls as module
        module.OUTPUT=out;sys.argv=['run_vqe_fourier_controls.py'];module.main();return
    seeds=args.seeds or list(range(1001,1006));assert set(seeds)<=set(range(1001,1006))
    if args.experiment=='masks':
        for seed in seeds:command(WORLD/'scripts/run_supplement_e1.py','--seed',seed,'--out',out/f'seed{seed}','--device',args.device)
    elif args.experiment=='query':
        for seed in seeds:command(EXP/'resource_query.py','--seed',seed,'--data',WORLD/f'results_supplementary/e2/confirmation/seed{seed}/dataset.npz','--out',out/f'seed{seed}','--seconds',240,'--device',args.device)
    else:
        ext=REVIEW/'extensions_20261108';sys.path[:0]=[str(ext),str(ext/'parallel')]
        import numpy as np
        import run_parallel as module
        from parallel_models import CONFIGS
        data=out/'data';data.mkdir(exist_ok=True)
        for seed in seeds:
            with np.load(WORLD/f'results_supplementary/e2/confirmation/seed{seed}/dataset.npz') as a:
                np.savez_compressed(data/f'seed{seed}.npz',X=a['X'],y=a['y'])
            shutil.copytree(ext/f'parallel/historical/seed{seed}',out/f'historical/seed{seed}',dirs_exist_ok=True)
            for name in CONFIGS:module.fit(name,seed,data/f'seed{seed}.npz',out/'output',args.device)
        if set(seeds)==set(range(1001,1006)):
            command(ext/'benchmark_parallel.py','--root',out,'--device',args.device)
if __name__=='__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='2'
    main()
