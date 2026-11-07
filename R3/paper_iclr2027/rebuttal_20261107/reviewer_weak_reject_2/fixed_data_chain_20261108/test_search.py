"""Critical scientific invariants for the paired fixed-data search."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
sys.path[:0] = [str(HERE), str(WORK/'world_model_surrogate/scripts'), str(WORK/'world_model_surrogate')]
from search import search, reference_starts
from run_redq_continuation import continuation


class SearchTests(unittest.TestCase):
    def test_clamp_reproduces_original_trace(self):
        x=np.random.default_rng(4).uniform(-np.pi,np.pi,(4,2,11))
        fn=lambda z,s: ((z.cos()+.3*z.sin())*np.exp(-s*s/2)).sum((1,2))
        old=continuation(fn,x,dict(steps=12,lr=.1,clip=np.pi),1.,.6)
        new=search(fn,x,steps=12,boundary='clamp')
        for key in ('parameters','predictions','sigma'):
            np.testing.assert_array_equal(new[key],old[key])

    def test_wrap_preserves_periodic_objective_and_escapes_clamp_boundary(self):
        x=np.full((2,2,11),np.pi-.01)
        fn=lambda z,s: np.exp(-s*s/2)*z.sin().sum((1,2))
        wrapped=search(fn,x,steps=8,boundary='wrap')
        clamped=search(fn,x,steps=8,boundary='clamp')
        self.assertLess(wrapped['predictions'][-1].mean(),clamped['predictions'][-1].mean())
        z=torch.tensor(wrapped['parameters'][-1],requires_grad=True)
        a=fn(z,0.);ga=torch.autograd.grad(a.sum(),z)[0]
        b=fn(z+2*np.pi,0.);gb=torch.autograd.grad(b.sum(),z)[0]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-12)
        self.assertLessEqual(abs(wrapped['parameters'][1:]).max(),np.pi)

    def test_paired_reference_starts_and_inactive_slots(self):
        a=reference_starts(45);b=reference_starts(45)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[:,:,:6][:,0],np.zeros((20,6)))
        center=np.zeros((2,11));center[0,6:9]=np.pi
        difference=(a-center+np.pi)%(2*np.pi)-np.pi
        np.testing.assert_array_equal(difference.reshape(20,22)[:,:6],np.zeros((20,6)))
        self.assertLessEqual(abs(difference).max(),.2+1e-14)


if __name__=='__main__':
    unittest.main()
