from pathlib import Path
import sys,unittest,importlib
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))


class HybridTests(unittest.TestCase):
    def model(self,mode='hybrid'):
        self.assertTrue((HERE/'models.py').exists(),'Missing architecture-level fusion')
        m=importlib.import_module('models');torch.manual_seed(7)
        return m.Hybrid(4,4,3,5,mode).double()

    def test_running_product_equals_direct_masked_prefix_product(self):
        m=self.model();x=torch.randn(2,4,5,dtype=torch.float64)
        q=m.period_features(x);_,zs,_=m.scan_steps(x,states=True)
        torch.testing.assert_close(zs,q.cumprod(1),atol=1e-13,rtol=1e-13)
        from src.wmps.models.fspt_product import fspt_masks
        expected=np.zeros((4,4,5))
        for k in range(1,5):expected[:,:k]=np.maximum(expected[:,:k],fspt_masks(4,k))
        np.testing.assert_array_equal(m.active.numpy(),expected)

    def test_product_factors_enter_gru_with_direct_product_readout_disabled(self):
        m=self.model()
        with torch.no_grad():m.product_readout.zero_()
        m(torch.randn(3,5,dtype=torch.float64))[0].square().sum().backward()
        self.assertGreater(m.factors.grad.abs().sum().item(),0)
        self.assertGreater(m.gru.weight_ih.grad.abs().sum().item(),0)
        inactive=(1-m.active).bool()[:,:, :,None,None].expand_as(m.factors)
        self.assertEqual(m.factors.grad[inactive].abs().sum().item(),0)

    def test_shared_prefix_outputs_and_gradients_equal_independent_queries(self):
        m=self.model();x=torch.randn(2,5,dtype=torch.float64,requires_grad=True)
        a=m(x)[0];ga=torch.autograd.grad(a.square().sum(),(x,m.factors,m.gru.weight_ih))
        b=m(x,independent=True)[0];gb=torch.autograd.grad(b.square().sum(),(x,m.factors,m.gru.weight_ih))
        torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
        for aa,bb in zip(ga,gb):torch.testing.assert_close(aa,bb,atol=1e-11,rtol=1e-11)

    def test_no_future_step_input_leakage(self):
        m=self.model();x=torch.randn(2,4,5,dtype=torch.float64);other=x.clone();other[:,2:]+=1
        torch.testing.assert_close(m.scan_steps(x)[:,:2],m.scan_steps(other)[:,:2],atol=0,rtol=0)

    def test_parameter_counts_and_identical_additive_initialization(self):
        m=importlib.import_module('models') if (HERE/'models.py').exists() else None
        self.assertIsNotNone(m)
        models={a:m.build(a,1001) for a in m.ARMS}
        self.assertEqual(sum(p.numel() for p in models['hybrid'].parameters()),108872)
        self.assertEqual(sum(p.numel() for p in models['recurrent'].parameters()),108913)
        self.assertEqual(sum(p.numel() for p in models['product'].parameters()),542304)
        for k,v in models['hybrid'].state_dict().items():torch.testing.assert_close(v,models['additive'].state_dict()[k],atol=0,rtol=0)


if __name__=='__main__':unittest.main(verbosity=2)
