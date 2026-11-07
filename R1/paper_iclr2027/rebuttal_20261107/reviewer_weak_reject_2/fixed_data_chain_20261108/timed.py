"""Pure global-prior continuation within one inclusive wall-clock window."""
import time
import numpy as np
from search import search


def select_candidates(parameters,predictions,count=20,min_rms=.2):
    flat=parameters.reshape(len(parameters),22)[:,6:]
    order=np.argsort(predictions,kind='stable');chosen=[]
    for index in order:
        if not chosen:chosen.append(int(index))
        else:
            difference=(flat[chosen]-flat[index]+np.pi)%(2*np.pi)-np.pi
            if np.sqrt(np.mean(difference**2,axis=1)).min()>=min_rms:
                chosen.append(int(index))
        if len(chosen)==count:break
    # Preserve a defined20-output metric even if a model converges to fewer basins.
    if len(chosen)<count:
        chosen.extend(int(i) for i in order if int(i) not in chosen)
    return np.array(chosen[:count],dtype=int)


def timed_search(predict,first_starts,seed,cap,clock=time.perf_counter):
    rng=np.random.default_rng(seed)
    started=clock();deadline=started+cap
    all_x=[];all_p=[];all_starts=[];eligible=[];retained=None;retained_p=None
    while clock()<deadline:
        starts=first_starts.copy() if not all_x else rng.uniform(-np.pi,np.pi,(20,2,11))
        trace=search(predict,starts,steps=100,boundary='wrap',deadline=deadline,clock=clock)
        if trace is None:break
        x,p=trace['parameters'][-1].copy(),trace['predictions'][-1].copy()
        pool=np.concatenate(all_x+[x]);scores=np.concatenate(all_p+[p])
        indices=select_candidates(pool,scores)
        new_x,new_p=pool[indices].copy(),scores[indices].copy()
        completed=clock()-started
        if completed>cap:break
        all_x.append(x);all_p.append(p);all_starts.append(starts)
        retained,retained_p=new_x,new_p;eligible.append(completed)
    elapsed=clock()-started
    if retained is None:raise RuntimeError('No complete search block within frozen time cap')
    return dict(retained_parameters=retained,retained_prediction=retained_p,
        block_parameters=np.stack(all_x),block_predictions=np.stack(all_p),
        block_starts=np.stack(all_starts),eligible_seconds=np.array(eligible),
        completed_blocks=len(all_x),actual_seconds=elapsed,cap=cap,
        min_rms=.2,added_shots=0)
