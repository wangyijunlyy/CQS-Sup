"""Critical numerical checks; separate from scientific result evidence."""
import unittest
import tempfile
import json
import numpy as np
import torch
import calibration as c
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


class CalibrationTests(unittest.TestCase):
    def test_primary_contrast_preserves_product_minus_fourier_sign(self):
        from analyze import contrast
        result=contrast(np.array([-.2,-.1,.1]))
        self.assertEqual(result['product_lower_datasets'],2)
        self.assertLess(result['mean'],0.)
        json.dumps(result)

    def test_fourier_audit_rejects_a_corrupted_saved_checkpoint(self):
        from analyze import audit_fourier
        from run_supplement_e1 import fit_fourier
        rng=np.random.default_rng(55)
        x=rng.uniform(-np.pi,np.pi,(30,22))
        y=np.cos(x[:,0,None])*np.linspace(.5,1.5,11)[None]
        masks=np.zeros((11,22),dtype=bool);masks[:,:2]=True
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fit_dir=root/'fit';fit_dir.mkdir()
            fit_fourier(x[:24],y[:24],x[24:],y[24:],masks,fit_dir)
            audit_fourier(fit_dir,x[:24],y[:24],x[24:],y[24:],masks,root/'audit')
            file=fit_dir/'kernel_group0.npz'
            with np.load(file) as a:data={k:a[k].copy() for k in a.files}
            data['weights']+=.01
            np.savez_compressed(file,**data)
            with self.assertRaises(AssertionError):
                audit_fourier(fit_dir,x[:24],y[:24],x[24:],y[24:],masks,root/'audit_bad')

    def test_validation_recovers_fractional_prior_without_exact_targets(self):
        rng = np.random.default_rng(13)
        z = rng.uniform(-1, 1, (256, 16))
        prior = np.sin(9*z[:, 0])
        measured = .5*prior+.2*z[:, 1]-.3
        w, choice = c.select_correction(z, measured, prior)
        self.assertEqual(choice['alpha'], .5)
        self.assertLess(choice['validation_mse'], 1e-7)
        self.assertEqual(len(choice['candidates']), 18)

    def test_alpha_one_matches_existing_local_correction(self):
        rng = np.random.default_rng(14)
        z = rng.uniform(-1, 1, (256, 16))
        y, prior = rng.normal(size=(2, 256))
        w, choice = c.select_correction(z, y, prior, alphas=(1.,))
        from chain_local_calibration import fit_correction
        old, detail = fit_correction(z, y, prior)
        np.testing.assert_allclose(w, old, atol=1e-11, rtol=1e-11)
        self.assertAlmostEqual(choice['validation_mse'], detail['validation_mse'])

    def test_zero_alpha_never_evaluates_unused_prior(self):
        def forbidden(x):
            raise AssertionError('Unused prior was evaluated')
        z = torch.zeros((20, 16), dtype=torch.float64, requires_grad=True)
        center = np.zeros((2, 11))
        w = np.zeros(153)
        w[1] = 1.
        value = c.local_objective(z, forbidden, center, .8, w, 0.)
        value.sum().backward()
        np.testing.assert_allclose(z.grad[:, 0], 1.)

    def test_late_candidate_checkpoint_cannot_win_selection(self):
        records = [dict(seconds=.5, predicted=[-1., -2.], parameters=['a', 'b']),
                   dict(seconds=1.01, predicted=[-100.], parameters=['late'])]
        result = c.select_eligible(records, 1., count=2)
        self.assertEqual(result['parameters'], ['b', 'a'])
        self.assertEqual(result['eligible_blocks'], 1)

    def test_real_timed_search_returns_only_eligible_complete_blocks(self):
        torch.set_num_threads(2)
        center=np.zeros((2,11))
        weights=np.zeros(153)
        weights[17]=1.
        def forbidden(x):
            raise AssertionError('Unused prior was evaluated')
        x,p,meta=c.search(forbidden,center,.8,weights,0.,np.random.default_rng(20),budget=1.)
        self.assertGreater(meta['completed_blocks'],0)
        self.assertLessEqual(meta['eligible_seconds'],1.)
        expected=c.select_eligible(meta['eligible_block_records'],1.)
        np.testing.assert_allclose(x,np.asarray(expected['parameters']))
        np.testing.assert_allclose(p,expected['predicted'])
        self.assertEqual(x.shape,(20,2,11))
        self.assertLessEqual(float(abs(x).max()),.2)


if __name__ == '__main__':
    unittest.main()
