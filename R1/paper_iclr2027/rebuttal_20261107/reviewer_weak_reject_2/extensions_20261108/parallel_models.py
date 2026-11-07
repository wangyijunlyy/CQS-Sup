"""Full configured parallel predictors of the complete FSPT trajectory."""
import torch
from torch import nn


CONFIGS = {
    'joint_mlp_w128': dict(family='joint_mlp', width=128, depth=3),
    'joint_mlp_w256': dict(family='joint_mlp', width=256, depth=3),
    'transformer_w64_l2': dict(family='transformer', width=64, depth=2),
    'transformer_w64_l4': dict(family='transformer', width=64, depth=4),
    'transformer_w128_l2': dict(family='transformer', width=128, depth=2),
    'transformer_w128_l4': dict(family='transformer', width=128, depth=4),
}


class JointMLP(nn.Module):
    def __init__(self, width, horizon=12, sites=8, depth=3):
        super().__init__()
        self.horizon, self.sites = horizon, sites
        layers = [nn.Linear(sites+1, width), nn.SiLU()]
        for _ in range(depth-1):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        layers.append(nn.Linear(width, horizon*sites))
        self.net = nn.Sequential(*layers)

    def forward(self, controls):
        return self.net(controls).reshape(len(controls), self.horizon, self.sites)


class TrajectoryTransformer(nn.Module):
    def __init__(self, width, depth, horizon=12, sites=8):
        super().__init__()
        self.horizon = horizon
        self.register_buffer('periods', torch.arange(1, horizon+1, dtype=torch.float32)/24)
        self.encoder = nn.Sequential(nn.Linear(sites+2, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        layer = nn.TransformerEncoderLayer(width, 4, dim_feedforward=4*width,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        # Encoder clones a prototype. Initialize the actual blocks independently.
        for block in self.blocks.layers:
            block.self_attn._reset_parameters()
            block.linear1.reset_parameters()
            block.linear2.reset_parameters()
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                     nn.SiLU(), nn.Linear(width, sites))

    def forward(self, controls):
        inputs = torch.cat((controls[:, None].expand(-1, self.horizon, -1),
            self.periods[None, :, None].expand(len(controls), -1, -1)), -1)
        return self.readout(self.blocks(self.encoder(inputs)))


def build_model(name):
    c = CONFIGS[name]
    if c['family'] == 'joint_mlp':
        return JointMLP(c['width'], depth=c['depth'])
    return TrajectoryTransformer(c['width'], c['depth'])


def select_under_budget(records, budget_ms):
    eligible = [r for r in records if r['latency_ms'] <= budget_ms]
    return min(eligible, key=lambda r: r['val_mse']) if eligible else None
