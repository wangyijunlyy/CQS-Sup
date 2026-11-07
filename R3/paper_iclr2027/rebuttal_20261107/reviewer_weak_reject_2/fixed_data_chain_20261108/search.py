"""Paired periodic continuation; fitted models and labels are unchanged."""
import time
import numpy as np
import torch

COUPLINGS=(-.5,-.3,-.1,.1,.3,.5)
ARMS={'clamp_uniform100':('clamp','uniform',100),
      'wrap_uniform100':('wrap','uniform',100),
      'wrap_uniform300':('wrap','uniform',300),
      'wrap_reference100':('wrap','reference',100)}


def reference_starts(seed):
    center=np.zeros((2,11));center[0,6:9]=np.pi
    x=np.repeat(center[None],20,axis=0).reshape(20,22)
    x[:,6:]+=np.random.default_rng(970000+seed).uniform(-.2,.2,(20,16))
    return ((x+np.pi)%(2*np.pi)-np.pi).reshape(20,2,11)


def search(predict,starts,steps=100,boundary='wrap',lr=.1,sigma=1.,fraction=.6,deadline=None,clock=time.perf_counter):
    if boundary not in ('wrap','clamp'):
        raise ValueError(boundary)
    x=torch.tensor(starts,dtype=torch.float64,requires_grad=True)
    opt=torch.optim.Adam([x],lr=lr,foreach=False)
    parameters,values,scales=[],[],[]
    for step in range(steps+1):
        if deadline is not None and clock()>deadline:
            return None
        scale=sigma*max(0.,1-step/(steps*fraction))
        opt.zero_grad()
        pred=predict(x,scale)
        if not torch.isfinite(pred).all():
            raise FloatingPointError('Nonfinite objective')
        parameters.append(x.detach().numpy().copy())
        values.append(pred.detach().numpy())
        scales.append(scale)
        if step==steps:
            break
        pred.sum().backward()
        if not torch.isfinite(x.grad).all():
            raise FloatingPointError('Nonfinite gradient')
        opt.step()
        with torch.no_grad():
            if boundary=='clamp':
                x.clamp_(-np.pi,np.pi)
            else:
                x.copy_((x+np.pi).remainder(2*np.pi)-np.pi)
    return dict(parameters=np.array(parameters),predictions=np.array(values),sigma=np.array(scales))
