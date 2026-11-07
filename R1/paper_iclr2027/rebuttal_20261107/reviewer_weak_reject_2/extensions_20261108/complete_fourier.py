"""Exact complete Q1 tensor-Fourier kernel and Gaussian convolution."""
from pathlib import Path
import numpy as np
import torch


def smoothed_kernel(x, z, mask, sigma=0.):
    indices = np.flatnonzero(mask)
    difference = x[:, None, indices] - z[None, :, indices]
    return (1+2*np.exp(-.5*sigma**2)*difference.cos()).prod(-1)


class CompleteFourier(torch.nn.Module):
    def __init__(self, folder):
        super().__init__()
        self.groups = []
        for path in sorted(Path(folder).glob('kernel_group*.npz')):
            with np.load(path) as a:
                self.groups.append(dict(
                    train=torch.tensor(a['train_x'], dtype=torch.float64),
                    mask=a['mask'], outputs=a['outputs'].tolist(),
                    mean=torch.tensor(a['mean']), overall=float(a['overall']),
                    scale=float(a['scale']), target_mean=torch.tensor(a['target_mean']),
                    weights=torch.tensor(a['weights'][0])))
        if sum(len(g['outputs']) for g in self.groups) != 11:
            raise ValueError('Expected all eleven original chain observables')

    def forward(self, controls, sigma=0.):
        flat = controls.flatten(1)
        result = flat.new_zeros((len(flat), 11))
        for g in self.groups:
            kernel = smoothed_kernel(flat, g['train'], g['mask'], sigma)
            centered = (kernel-kernel.mean(1, keepdim=True)-g['mean'][None]
                        +g['overall'])/g['scale']
            result[:, g['outputs']] = centered @ g['weights'] + g['target_mean']
        return result


class SharedFeatureFourier(CompleteFourier):
    """Same fitted kernel, reusing trigonometric features across observables."""
    def __init__(self, folder):
        super().__init__(folder)
        self.union = np.flatnonzero(np.any([g['mask'] for g in self.groups], axis=0))
        self.shared_train = self.groups[0]['train'][:, self.union]
        for g in self.groups:
            if not torch.equal(g['train'], self.groups[0]['train']):
                raise ValueError('Shared evaluation requires identical training controls')
            g['positions'] = np.flatnonzero(g['mask'][self.union])

    def forward(self, controls, sigma=0.):
        flat = controls.flatten(1)
        difference = flat[:, None, self.union]-self.shared_train[None]
        factors = 1+2*np.exp(-.5*sigma**2)*difference.cos()
        result = flat.new_zeros((len(flat), 11))
        for g in self.groups:
            kernel = factors[:, :, g['positions']].prod(-1)
            centered = (kernel-kernel.mean(1, keepdim=True)-g['mean'][None]
                        +g['overall'])/g['scale']
            result[:, g['outputs']] = centered@g['weights']+g['target_mean']
        return result
