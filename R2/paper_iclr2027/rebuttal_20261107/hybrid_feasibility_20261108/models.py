"""Masked low-rank product states inside a shared recurrent cell."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
WORK=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(WORK/'world_model_surrogate'))
from src.wmps.models.fspt_product import fspt_masks,ProductFSPT
from src.wmps.models.world_model import ObsWorldModel
ARMS=('recurrent','product','hybrid','additive')


def union_masks(n,k):
    mask=np.zeros((n,k,n+1),dtype=np.float32)
    for length in range(1,k+1):mask[:,:length]=np.maximum(mask[:,:length],fspt_masks(n,length))
    return mask


def initialize_factors(factors):
    with torch.no_grad():
        factors[...,0]=1.;factors[...,1:].normal_(0,.02)


class Hybrid(nn.Module):
    def __init__(self,n=8,horizon=12,rank=32,hidden=64,mode='hybrid'):
        super().__init__()
        if mode not in ('hybrid','additive'):raise ValueError(mode)
        self.n=n;self.horizon=horizon;self.rank=rank;self.hidden=hidden;self.mode=mode
        self.register_buffer('active',torch.tensor(union_masks(n,horizon)))
        self.factors=nn.Parameter(torch.empty(n,horizon,n+1,rank,3));initialize_factors(self.factors)
        self.gru=nn.GRUCell(2*rank+1,hidden)
        self.product_readout=nn.Parameter(torch.randn(n,rank)*(.1/rank**.5))
        self.recurrent_readout=nn.Parameter(torch.randn(n,hidden)*(.1/hidden**.5))
        self.bias=nn.Parameter(torch.zeros(n))

    def angle_steps(self,controls):
        angles=torch.cat((torch.pi-2*controls[:,:1],-2*controls[:,1:]),-1)
        return angles[:,None,:].expand(-1,self.horizon,-1)

    def period_features(self,angles):
        length=angles.shape[1]
        phi=torch.stack((torch.ones_like(angles),2**.5*angles.cos(),2**.5*angles.sin()),-1)
        gates=torch.einsum('bldf,oldrf->blodr',phi,self.factors[:,:length])
        active=self.active[:,:length].permute(1,0,2)[None,:,:,:,None]
        gates=gates*active+(1-active)
        return gates.prod(3) if self.mode=='hybrid' else 1+(gates-1).sum(3)

    def scan_steps(self,angles,states=False):
        q=self.period_features(angles);batch,length=q.shape[:2]
        z=q.new_ones(batch,self.n,self.rank);h=q.new_zeros(batch,self.n,self.hidden)
        out=[];products=[];hidden=[]
        for i in range(length):
            z=z*q[:,i] if self.mode=='hybrid' else z+(q[:,i]-1)
            time=q.new_full((batch,self.n,1),(i+1)/24)
            inp=torch.cat((q[:,i]-1,z-1,time),-1)
            h=self.gru(inp.reshape(batch*self.n,-1),h.reshape(batch*self.n,-1)).reshape(batch,self.n,self.hidden)
            y=((z-1)*self.product_readout).sum(-1)+(h*self.recurrent_readout).sum(-1)+self.bias
            out.append(y);products.append(z);hidden.append(h)
        prediction=torch.stack(out,1)
        return (prediction,torch.stack(products,1),torch.stack(hidden,1)) if states else prediction

    def forward(self,controls,independent=False):
        angles=self.angle_steps(controls)
        if independent:return torch.stack([self.scan_steps(angles[:,:k])[:,-1] for k in range(1,self.horizon+1)],1),None
        return self.scan_steps(angles),None


class Recurrent(ObsWorldModel):
    def __init__(self):super().__init__(10,8,115,115,115,False)
    def forward(self,controls):
        t=torch.arange(1,13,device=controls.device,dtype=controls.dtype)[None,:,None]/24
        steps=torch.cat((controls[:,None,:].expand(-1,12,-1),t.expand(len(controls),-1,-1)),-1)
        return super().forward(steps)


def build(arm,seed):
    torch.manual_seed(seed+1000)
    if arm=='recurrent':return Recurrent()
    if arm=='product':
        m=ProductFSPT(8,12,32)
        for head in m.heads:initialize_factors(head.factors)
        return m
    if arm in ('hybrid','additive'):return Hybrid(mode=arm)
    raise ValueError(arm)
