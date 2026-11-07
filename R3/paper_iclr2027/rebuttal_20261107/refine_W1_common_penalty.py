"""Symmetric validation-only grid refinement of the shared-penalty control."""
import shutil
import json
import argparse
import torch
from threadpoolctl import threadpool_limits
import run_W1_common_penalty as experiment


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(1001,1006)))
    args=parser.parse_args()
    assert set(args.seeds)<=set(range(1001,1006))
    torch.set_num_threads(2)
    source=experiment.DEST
    experiment.DEST=source.parent/'common_penalty_refined'
    experiment.GRID=[0.,1e-6,1e-4,.01,.1,1.,10.,100.]
    experiment.DEST.mkdir(parents=True,exist_ok=True)
    experiment.write(experiment.DEST/'protocol.json',dict(
        seeds=list(range(1001,1006)),grid=experiment.GRID,
        refinement_rule='add lambda0.1 and10 symmetrically for both methods around validation-best coarse lambda1',
        selection='validation-only; all coarse candidates retained',
        objective='same mean moment MSE plus nonconstant orthonormal Fourier coefficient squared norm',
        unchanged=dict(rank=32,epochs=300,patience=30,weight_decay=0,shots=10),
        scope='same existing independent E1 pools; controlled fitting comparison'))
    for seed in args.seeds:
        folder=experiment.DEST/f'seed{seed}'
        folder.mkdir(exist_ok=True)
        for candidate in (source/f'seed{seed}').glob('product_lambda*'):
            if (candidate/'fit.json').exists() and not (folder/candidate.name).exists():
                shutil.copytree(candidate,folder/candidate.name)
        experiment.run(seed)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
