"""Validation criterion and exact full-Fourier fitting, not paper results."""
from pathlib import Path
import sys,unittest
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from fitting import energy_validation,choose,fit_fourier_candidates,coefficient_norm,should_confirm


class FitTests(unittest.TestCase):
    def test_confirmation_gate_uses_negative_product_minus_fourier(self):
        row=lambda value:{'product_minus_fourier':{'mean':value}}
        self.assertFalse(should_confirm([row(.4),row(.1)]))
        self.assertFalse(should_confirm([row(0.)]))
        self.assertTrue(should_confirm([row(.4),row(-.02)]))

    def test_coefficient_penalty_matches_explicit_tensor_and_gradient(self):
        import torch,itertools
        from types import SimpleNamespace
        torch.manual_seed(8)
        factors=torch.randn(2,2,3,3,dtype=torch.float64,requires_grad=True)
        readout=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
        model=SimpleNamespace(factors=factors,readout=readout,active=torch.ones(2,2))
        explicit=[]
        for o in range(2):
            c=torch.stack([(readout[o]*factors[o,0,:,a]*factors[o,1,:,b]).sum()
                for a,b in itertools.product(range(3),repeat=2)])
            explicit.append(c[1:].square().sum())
        explicit=torch.stack(explicit);actual=coefficient_norm(model)
        torch.testing.assert_close(actual,explicit,atol=1e-10,rtol=1e-10)
        a=torch.autograd.grad(actual.sum(),factors,retain_graph=True)[0]
        b=torch.autograd.grad(explicit.sum(),factors)[0]
        torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)

    def test_energy_validation_recombines_each_hamiltonian(self):
        rng=np.random.default_rng(8)
        p,y=rng.normal(size=(2,13,11))
        c=rng.normal(size=(6,11))
        manual=np.mean([np.mean((p@row-y@row)**2) for row in c])
        self.assertAlmostEqual(energy_validation(p,y,c),manual,places=12)
        records=[{'validation_energy_mse':3.,'name':'first'},
                 {'validation_energy_mse':1.,'name':'second'},
                 {'validation_energy_mse':2.,'name':'third'}]
        self.assertEqual(choose(records)['name'],'second')

    def test_complete_kernel_matches_orthonormal_primal_ridge(self):
        rng=np.random.default_rng(9)
        x=rng.uniform(-np.pi,np.pi,(40,2));z=rng.uniform(-np.pi,np.pi,(7,2))
        y=rng.normal(size=(40,11));yv=rng.normal(size=(7,11))
        masks=np.ones((11,2),bool)
        c=rng.normal(size=(6,11))
        candidates,groups=fit_fourier_candidates(x,y,z,yv,masks,c,
            raw_grid=(.1,),scaled_grid=())
        def basis(a):
            q=np.stack([np.ones_like(a),np.sqrt(2)*np.cos(a),np.sqrt(2)*np.sin(a)],axis=-1)
            return np.einsum('bi,bj->bij',q[:,0],q[:,1]).reshape(len(a),9)
        a,b=basis(x)[:,1:],basis(z)[:,1:]
        ac=a-a.mean(0);yc=y-y.mean(0)
        beta=np.linalg.solve(ac.T@ac+len(x)*.1*np.eye(8),ac.T@yc)
        expected=(b-a.mean(0))@beta+y.mean(0)
        np.testing.assert_allclose(candidates[0]['prediction'],expected,atol=1e-10,rtol=1e-10)
        self.assertEqual(len(groups),1)
        self.assertAlmostEqual(candidates[0]['validation_energy_mse'],energy_validation(expected,yv,c),places=11)


if __name__=='__main__':unittest.main()
