"""Constructive capacity check and identical-encoding execution benchmark."""
from pathlib import Path
import sys
import json
import time
import argparse
from collections import Counter
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_evidence import ROOT,OUT,TFIMTask,ChainExact
from src.wmps.models.periodic_redq import ProductMomentREDQ
from src.wmps.models.world_model import ObsWorldModel
from run_fspt8_fair import parameter_steps


def representation():
    rows=[]
    for layers in (1,2):
        task=TFIMTask(1,6,layers,False,-1)
        exact=ChainExact(task)
        count=Counter(o for o,_,_,_ in exact.terms)
        model=ProductMomentREDQ(6,task.edges,layers,False,width=32,harmonics=1).double()
        with torch.no_grad():
            model.factors.zero_()
            model.factors[:,:,:,0]=1.
            model.readout.zero_()
            model.bias.zero_()
            next_channel=Counter()
            for o,ci,si,coefficient in exact.terms:
                r=next_channel[o]
                next_channel[o]+=1
                for indices,basis in ((ci,1),(si,2)):
                    for d in indices:
                        assert model.active[o,d]==1
                        model.factors[o,d,r,:]=0
                        model.factors[o,d,r,basis]=1/np.sqrt(2.)
                model.readout[o,r]=coefficient
        x=np.random.default_rng(878).uniform(-np.pi,np.pi,(32,*task.control_shape))
        truth=np.stack([task.exact_terms(z) for z in x])
        with torch.no_grad():
            pred=model(torch.tensor(x,dtype=torch.float64))[0].numpy()
        error=float(np.max(np.abs(pred-truth)))
        assert error<1e-12
        row=dict(layers=layers,total_monomials=len(exact.terms),monomials_per_observable=[count[o] for o in range(task.block)],
                 sufficient_rank=max(count.values()),paper_rank=32,Q=1,max_qiskit_moment_error=error,
                 coefficients='constructed analytically for capacity diagnostic; not fitted and not a learned comparator')
        torch.save(model.state_dict(),OUT/f'analytic_capacity_chain6_l{layers}.pt')
        rows.append(row)
    (OUT/'constructive_capacity.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('CAPACITY',rows,flush=True)


def recomputed(model,x):
    """Independent zero state per output; exactly the same j/24 history.

    Batched independent queries, with finished queries removed. Encodings are
    shared by both executions, so this isolates reuse of recurrent states.
    """
    emb=model.encoder(x)
    b,k,width=emb.shape
    h=emb.new_zeros(b,k,model.rssm.hidden)
    outputs=[]
    for j in range(k):
        step=emb[:,j,None,:].expand(b,k-j,width).reshape(-1,width)
        h=model.rssm.gru(step,h.reshape(-1,model.rssm.hidden)).reshape(b,k-j,model.rssm.hidden)
        outputs.append(h[:,0])
        h=h[:,1:]
    return model.predictor(torch.stack(outputs,1))


def execution():
    rows=[]
    for seed in range(1001,1006):
        src=ROOT/f'results_supplementary/e2/confirmation/seed{seed}'
        model=ObsWorldModel(obs_dim=10,out_dim=8,encoder_hidden=128,rssm_hidden=128,predictor_hidden=128,stochastic=False).eval()
        saved=torch.load(src/'cqs_r.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict'])
        with np.load(src/'dataset.npz') as a:
            x=torch.tensor(parameter_steps(a['X'][2400:],12,24))
            target=a['y'][2400:]
        with torch.inference_mode():
            shared=model(x)[0]
            separate=recomputed(model,x)
        discrepancy=float((shared-separate).abs().max())
        np.testing.assert_allclose(shared.numpy(),separate.numpy(),atol=2e-6,rtol=2e-6)
        # Identical task loss and gradients, using the same complete 12 targets.
        small=x[:32]
        yy=torch.tensor(target[:32])
        model.zero_grad(set_to_none=True)
        loss=(model(small)[0]-yy).square().mean()
        loss.backward()
        grads=[p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        loss2=(recomputed(model,small)-yy).square().mean()
        loss2.backward()
        grad_error=max(float((p.grad-g).abs().max()) for p,g in zip(model.parameters(),grads))
        assert grad_error<2e-6
        # Same host, dtype, data, threads, batch, and output contract; alternating
        # order prevents systematic timing-order bias. Exclude model loading.
        modes={'shared':lambda:model(x)[0],'recomputed':lambda:recomputed(model,x)}
        times={key:[] for key in modes}
        with torch.inference_mode():
            for f in modes.values():
                for _ in range(3):
                    f()
            for repeat in range(20):
                order=list(modes) if repeat%2==0 else list(reversed(modes))
                for key in order:
                    t=time.perf_counter()
                    modes[key]()
                    times[key].append(time.perf_counter()-t)
        row=dict(seed=seed,N=8,K=12,trajectories=400,dtype='float32',threads=2,device='cpu',
                 same_weights=True,same_complete_period_history=True,max_prediction_error=discrepancy,
                 max_gradient_error=grad_error,loss=float(loss.detach()),recomputed_loss=float(loss2.detach()),
                 mae=float(np.abs(shared.numpy()-target).mean()),
                 shared_cells_per_trajectory=12,recomputed_cells_per_trajectory=78,
                 shared_median_seconds=float(np.median(times['shared'])),
                 recomputed_median_seconds=float(np.median(times['recomputed'])),raw_seconds=times,
                 scope='execution equivalence of one fitted model; not a retrained fixed-query Query-GRU or a matched-training-walltime accuracy experiment')
        rows.append(row)
        print('EXECUTION',seed,row['shared_median_seconds'],row['recomputed_median_seconds'],discrepancy,flush=True)
    (OUT/'matched_encoding_execution.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',choices=('all','capacity','execution'),default='all')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if args.part in ('all','capacity'):
        representation()
    if args.part in ('all','execution'):
        execution()
