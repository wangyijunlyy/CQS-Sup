"""Periodic recurrent moment predictor for plus-initialized RX/RZZ circuits."""

import torch
from torch import nn
import numpy as np


def causal_parameter_masks(n, edges, layers, shared, hadamards_each_layer=False):
    """Conservative Pauli support propagation, without expectation evaluation."""
    block = n + len(edges)
    controls = 2 if shared else block
    masks = np.zeros((block, layers, controls), dtype=np.float32)
    gates = []
    for layer in range(layers):
        if hadamards_each_layer:
            gates.append((layer, -1, 0, 0))
        gates.extend((layer, wire, 1 << wire, 0) for wire in range(n))
        gates.extend((layer, n + idx, 0, (1 << i) | (1 << j)) for idx, (i, j) in enumerate(edges))
    observables = [(1 << i, 0) for i in range(n)] + [(0, (1 << i) | (1 << j)) for i, j in edges]
    for term, observable in enumerate(observables):
        support = {observable}
        for layer, gate, gx, gz in reversed(gates):
            if gate == -1:
                support = {(pz, px) for px, pz in support}
                continue
            if layer == 0 and gate < n:
                continue
            branches = {(px ^ gx, pz ^ gz) for px, pz in support
                        if ((px & gz).bit_count() + (pz & gx).bit_count()) % 2}
            if branches:
                control = (0 if gate < n else 1) if shared else gate
                masks[term, layer, control] = 1
                support |= branches
    return masks


class CausalMomentREDQ(nn.Module):
    """Separate low-dimensional recurrent heads with exact dependency masks."""
    def __init__(self, n, edges, layers, shared, width=32, harmonics=1):
        super().__init__()
        self.register_buffer('masks', torch.tensor(causal_parameter_masks(n, edges, layers, shared)))
        self.heads = nn.ModuleList([
            PeriodicMomentREDQ(n, edges, shared, width, harmonics)
            for _ in range(n + len(edges))])
        for head in self.heads:
            head.readout[-1] = nn.Linear(width, 1)

    def forward(self, x):
        values = [head(x * mask)[0] for head, mask in zip(self.heads, self.masks)]
        return torch.cat(values, dim=1), None


class ProductMomentREDQ(nn.Module):
    """Low-rank multiplicative latent recurrence over causal control parameters.

    Each factor is learned from measured moments, including its constant part.
    No simulator amplitudes or analytic moment coefficients enter the model.
    """
    def __init__(self, n, edges, layers, shared, width=32, harmonics=1, hadamards_each_layer=False):
        super().__init__()
        masks = causal_parameter_masks(n, edges, layers, shared, hadamards_each_layer)
        active = masks.reshape(len(masks), -1)
        count = active.shape[1]
        self.register_buffer('active', torch.tensor(active))
        self.register_buffer('harmonics', torch.arange(1, harmonics + 1, dtype=torch.float32))
        self.factors = nn.Parameter(torch.randn(len(masks), count, width, 1 + 2 * harmonics) / np.sqrt(1 + 2 * harmonics))
        self.readout = nn.Parameter(torch.randn(len(masks), width) * .1 / np.sqrt(width))
        self.bias = nn.Parameter(torch.zeros(len(masks)))

    def forward(self, x, smoothing=0.0):
        phase = x.flatten(1).unsqueeze(-1) * self.harmonics
        damping = torch.exp(-.5 * smoothing**2 * self.harmonics**2)
        phi = torch.cat((torch.ones_like(phase[..., :1]), 2**.5 * phase.cos() * damping, 2**.5 * phase.sin() * damping), -1)
        gates = torch.einsum('bdf,odrf->bodr', phi, self.factors)
        gates = gates * self.active[None, :, :, None] + (1 - self.active[None, :, :, None])
        latent = gates.prod(dim=2)
        return torch.einsum('bor,or->bo', latent, self.readout) + self.bias, None


class ProductEnergyREDQ(nn.Module):
    """Scalar supervision of observable-local components for Table 1 circuits."""
    def __init__(self, n, width=32, harmonics=1, J=-.1, h=-.5):
        super().__init__()
        edges = [(i, i+1) for i in range(n-1)]
        self.moments = ProductMomentREDQ(n, edges, 2, False, width, harmonics,
                                        hadamards_each_layer=True)
        self.register_buffer('coefficients', torch.tensor([-h]*n + [-J]*(n-1)))

    def forward(self, x):
        return (self.moments(x)[0] @ self.coefficients)[:, None], None


class PeriodicMomentREDQ(nn.Module):
    def __init__(self, n, edges, shared, width=64, harmonics=4):
        super().__init__()
        self.n = n
        self.shared = shared
        self.input_dim = 2 if shared else n + len(edges)
        self.register_buffer('harmonics', torch.arange(1, harmonics + 1, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(2 * harmonics * self.input_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU())
        self.gru = nn.GRUCell(width, width)
        self.readout = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, n + len(edges)))
        self.width = width

    def forward(self, x):
        mask = torch.ones_like(x)
        mask[:, 0, :1 if self.shared else self.n] = 0
        phase = (x * mask).unsqueeze(-1) * self.harmonics
        features = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        embeddings = self.encoder(features)
        state = x.new_zeros((len(x), self.width))
        for step in range(x.shape[1]):
            state = self.gru(embeddings[:, step], state)
        return self.readout(state).tanh(), None
