"""Fixed-horizon RED-Q-M adapter for the cached RX / cluster-Pauli drive.

One factor per independent control per period, as in shared-control TFIM.
Each supervised (period, observable) has its own head; no temporal extrapolator
is implied. Masks are conservative operator light cones, not state labels.
"""
import numpy as np
import torch
from torch import nn


def expanded_angles(controls, periods):
    n = controls.shape[-1]-1
    return np.tile(np.concatenate((np.repeat(np.pi-2*controls[:, :1], n, axis=1),
                                   -2*controls[:, 1:]), axis=1), (1, periods))


def fspt_masks(n, periods):
    masks = np.zeros((n, periods, n+1), dtype=np.float32)
    gates = []
    for t in range(periods):
        gates.extend((t, 0, 1 << i, 0) for i in range(n))
        for i in range(n):
            z = (1 << (i-1) if i else 0) | (1 << (i+1) if i+1 < n else 0)
            gates.append((t, i+1, 1 << i, z))
    for obs in range(n):
        # Union supports overapproximate Pauli branches, so omitted controls
        # provably cannot affect the observable, even at long horizons.
        px, pz = 0, 1 << obs
        for t, control, gx, gz in reversed(gates):
            if (px & gz) or (pz & gx):
                masks[obs, t, control] = 1
                px, pz = px | gx, pz | gz
    return masks


class ProductHead(nn.Module):
    def __init__(self, n, periods, width):
        super().__init__()
        active = fspt_masks(n, periods).reshape(n, -1)
        self.periods = periods
        self.register_buffer('active', torch.tensor(active))
        self.factors = nn.Parameter(torch.randn(n, active.shape[1], width, 3)/np.sqrt(3))
        self.readout = nn.Parameter(torch.randn(n, width)*.1/np.sqrt(width))
        self.bias = nn.Parameter(torch.zeros(n))

    def forward(self, angles):
        x = angles.repeat(1, self.periods)
        phi = torch.stack((torch.ones_like(x), 2**.5*x.cos(), 2**.5*x.sin()), -1)
        gates = torch.einsum('bdf,odrf->bodr', phi, self.factors)
        gates = gates*self.active[None, :, :, None]+(1-self.active[None, :, :, None])
        return torch.einsum('bor,or->bo', gates.prod(2), self.readout)+self.bias


class ProductFSPT(nn.Module):
    def __init__(self, n, horizon, width=32):
        super().__init__()
        self.horizon = horizon
        self.heads = nn.ModuleList([ProductHead(n, t, width) for t in range(1, horizon+1)])

    def period(self, controls, index):
        angles = torch.cat((torch.pi-2*controls[:, :1], -2*controls[:, 1:]), -1)
        return self.heads[index](angles)

    def period_group(self, controls, start, stop):
        from torch.nn.functional import pad
        heads = self.heads[start:stop]
        n, count, rank, _ = heads[-1].factors.shape
        factors = torch.cat([pad(h.factors, (0,0,0,0,0,count-h.factors.shape[1])) for h in heads])
        active = torch.cat([pad(h.active,(0,count-h.active.shape[1])) for h in heads])
        readout = torch.cat([h.readout for h in heads])
        bias = torch.cat([h.bias for h in heads])
        angles = torch.cat((torch.pi-2*controls[:,:1],-2*controls[:,1:]),-1).repeat(1,stop)
        phi = torch.stack((torch.ones_like(angles),2**.5*angles.cos(),2**.5*angles.sin()),-1)
        gates = torch.einsum('bdf,odrf->bodr',phi,factors)
        gates = gates*active[None,:,:,None]+(1-active[None,:,:,None])
        pred = torch.einsum('bor,or->bo',gates.prod(2),readout)+bias
        return pred.reshape(len(controls),stop-start,n)

    def forward(self, controls, horizon=None):
        horizon = self.horizon if horizon is None else horizon
        if not 1 <= horizon <= self.horizon:
            raise ValueError('No fitted head for an unsupervised period')
        return torch.stack([self.period(controls, t) for t in range(horizon)], 1), None
