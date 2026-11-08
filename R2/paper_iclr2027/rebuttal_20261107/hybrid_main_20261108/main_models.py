"""The same joint product–GRU state on the three original main tasks."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[3]/'world_model_surrogate';sys.path.insert(0,str(ROOT))
from src.wmps.models.fspt_product import fspt_masks
from src.wmps.models.periodic_redq import causal_parameter_masks


class JointCore(nn.Module):
 def __init__(self,masks,harmonics=1,rank=32,hidden=64):
  super().__init__();masks=np.asarray(masks,dtype=np.float32)
  if masks.ndim!=3 or not np.isin(masks,[0,1]).all():raise ValueError('Expected binary observable/layer/control masks')
  self.n,self.horizon,self.inputs=masks.shape;self.rank=rank;self.hidden=hidden
  self.register_buffer('active',torch.tensor(masks))
  self.register_buffer('harmonics',torch.arange(1,harmonics+1,dtype=torch.float32))
  self.factors=nn.Parameter(torch.empty(self.n,self.horizon,self.inputs,rank,1+2*harmonics))
  with torch.no_grad():self.factors[...,0]=1.;self.factors[...,1:].normal_(0,.02)
  self.gru=nn.GRUCell(2*rank+1,hidden)
  self.product_readout=nn.Parameter(torch.randn(self.n,rank)*(.1/rank**.5))
  self.recurrent_readout=nn.Parameter(torch.randn(self.n,hidden)*(.1/hidden**.5))
  self.bias=nn.Parameter(torch.zeros(self.n))
 def period_features(self,angles):
  phase=angles[...,None]*self.harmonics
  phi=torch.cat((torch.ones_like(phase[...,:1]),2**.5*phase.cos(),2**.5*phase.sin()),-1)
  length=angles.shape[1];gates=torch.einsum('bldf,oldrf->blodr',phi,self.factors[:,:length])
  active=self.active[:,:length].permute(1,0,2)[None,:,:,:,None]
  return (gates*active+1-active).prod(3)
 def scan_steps(self,angles):
  q=self.period_features(angles);batch,length=q.shape[:2]
  z=q.new_ones(batch,self.n,self.rank);h=q.new_zeros(batch,self.n,self.hidden);out=[]
  for i in range(length):
   z=z*q[:,i];t=q.new_full((batch,self.n,1),(i+1)/24)
   inp=torch.cat((q[:,i]-1,z-1,t),-1)
   h=self.gru(inp.reshape(batch*self.n,-1),h.reshape(batch*self.n,-1)).reshape(batch,self.n,self.hidden)
   out.append(((z-1)*self.product_readout).sum(-1)+(h*self.recurrent_readout).sum(-1)+self.bias)
  return torch.stack(out,1)


class FSPTJoint(JointCore):
 def __init__(self,n,horizon):
  masks=np.zeros((n,horizon,n+1),np.float32)
  for k in range(1,horizon+1):masks[:,:k]=np.maximum(masks[:,:k],fspt_masks(n,k))
  super().__init__(masks)
 def angle_steps(self,controls):
  angles=torch.cat((torch.pi-2*controls[:,:1],-2*controls[:,1:]),-1)
  return angles[:,None,:].expand(-1,self.horizon,-1)
 def forward(self,controls,independent=False):
  angles=self.angle_steps(controls)
  if independent:return torch.stack([self.scan_steps(angles[:,:k])[:,-1] for k in range(1,self.horizon+1)],1),None
  return self.scan_steps(angles),None


class TFIMJoint(nn.Module):
 def __init__(self,n):
  super().__init__();edges=[(i,i+1) for i in range(n-1)]
  self.core=JointCore(causal_parameter_masks(n,edges,2,False,True))
  self.register_buffer('coefficients',torch.tensor([.5]*n+[.1]*(n-1)))
 def forward(self,controls):return (self.core.scan_steps(controls)[:,-1]@self.coefficients)[:,None],None


class VQEJoint(JointCore):
 def __init__(self,task):
  super().__init__(causal_parameter_masks(task.n,task.edges,task.layers,task.shared),4 if task.shared else 1)
 def forward(self,controls):return self.scan_steps(controls)[:,-1],None
