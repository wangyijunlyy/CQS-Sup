"""Numerical engineering checks, not experimental accuracy evidence."""
import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
from parallel_models import JointMLP, TrajectoryTransformer, select_under_budget
from complete_fourier import smoothed_kernel


class ExtensionTests(unittest.TestCase):
    def test_full_trajectory_models_use_controls_and_backpropagate(self):
        torch.manual_seed(33)
        x = torch.randn(2, 9, requires_grad=True)
        for model in (JointMLP(128), TrajectoryTransformer(64, 2)):
            y = model(x)
            self.assertEqual(tuple(y.shape), (2, 12, 8))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad.abs().sum()), 0.)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()
                                   if p.grad is not None), 0.)
            x.grad = None

    def test_latency_filter_then_validation_selection_ignores_test_metric(self):
        rows = [dict(name='validation_best', latency_ms=10., val_mse=.1, test_mae=100.),
                dict(name='test_best', latency_ms=5., val_mse=.2, test_mae=0.),
                dict(name='too_slow', latency_ms=20., val_mse=.01, test_mae=0.)]
        self.assertEqual(select_under_budget(rows, 11.)['name'], 'validation_best')
        self.assertIsNone(select_under_budget(rows, 1.))

    def test_kernel_smoothing_matches_complete_fourier_expansion(self):
        x = torch.tensor([[.2, -.7, 1.1], [.8, .1, .6]], dtype=torch.float64,
                         requires_grad=True)
        z = torch.tensor([[-.4, .3, -.6], [.1, .2, .3]], dtype=torch.float64)
        mask = np.array([True, False, True])
        sigma = .6
        frequencies = torch.tensor([(a, b) for a in (-1, 0, 1)
                                    for b in (-1, 0, 1)], dtype=torch.float64)
        phase = (x[:, None, mask] - z[None, :, mask]) @ frequencies.T
        expected = (torch.cos(phase)*torch.exp(-.5*sigma**2*frequencies.square().sum(1))).sum(-1)
        actual = smoothed_kernel(x, z, mask, sigma)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(float(x.grad[:, 1].abs().sum()), 0.)

    def test_direct_search_uses_distinct_local_starts_and_retains_reference(self):
        from direct_jitter import initial_controls
        x, incumbent = initial_controls(45, -.5)
        self.assertEqual(x.shape, (20, 2, 11))
        np.testing.assert_array_equal(x[0], incumbent[0])
        delta = (x-incumbent).reshape(20, 22)
        self.assertTrue(np.all(np.abs(delta[:, 6:]) <= .2))
        self.assertTrue(np.all(delta[1:, 6:] != 0))
        np.testing.assert_array_equal(delta[:, :6], 0)

    def test_shared_kernel_features_preserve_values_and_input_gradients(self):
        from complete_fourier import CompleteFourier, SharedFeatureFourier
        rng = np.random.default_rng(91)
        train = rng.normal(size=(8, 22))
        with tempfile.TemporaryDirectory() as temp:
            for output in range(11):
                mask = np.zeros(22, dtype=bool)
                mask[6+(np.arange(3+output % 5)+output) % 16] = True
                np.savez(Path(temp)/f'kernel_group{output}.npz', train_x=train,
                    mask=mask, outputs=[output], mean=rng.normal(size=8),
                    overall=.1, scale=2., target_mean=np.array([.3]),
                    weights=rng.normal(size=(1, 8, 1)))
            a, b = CompleteFourier(temp), SharedFeatureFourier(temp)
            x = torch.tensor(rng.normal(size=(3, 2, 11)), requires_grad=True)
            p, q = a(x, .6), b(x, .6)
            torch.testing.assert_close(p, q, atol=1e-12, rtol=1e-12)
            g = torch.autograd.grad(p.square().sum(), x)[0]
            h = torch.autograd.grad(q.square().sum(), x)[0]
            torch.testing.assert_close(g, h, atol=1e-11, rtol=1e-11)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
