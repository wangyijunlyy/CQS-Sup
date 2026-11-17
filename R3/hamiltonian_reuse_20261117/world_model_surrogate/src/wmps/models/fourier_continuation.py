"""Closed-form Gaussian smoothing for the support-two NC feature map."""

import torch


def smoothed_nc_features(fmap, x, sigma):
    c, s = fmap.cos_counts, fmap.sin_counts
    if int((c + s).max()) > 2:
        raise ValueError('Analytic convolution supports exponents up to two')
    out = x.new_ones((len(x), len(c)))
    sigma = x.new_tensor(sigma)
    first, second = torch.exp(-.5 * sigma**2), torch.exp(-2 * sigma**2)
    for d in range(x.shape[1]):
        angle = x[:, d:d+1]
        factor = torch.ones_like(out)
        factor = torch.where((c[:, d] == 1) & (s[:, d] == 0), angle.cos() * first, factor)
        factor = torch.where((s[:, d] == 1) & (c[:, d] == 0), angle.sin() * first, factor)
        factor = torch.where(c[:, d] == 2, .5 * (1 + (2 * angle).cos() * second), factor)
        factor = torch.where(s[:, d] == 2, .5 * (1 - (2 * angle).cos() * second), factor)
        factor = torch.where((c[:, d] == 1) & (s[:, d] == 1), .5 * (2 * angle).sin() * second, factor)
        out = out * factor
    return out
