import sys,unittest
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'hybrid_feasibility_20261108'))

class MainAdapters(unittest.TestCase):
 def test_fspt_exactly_reuses_existing_joint_architecture(self):
  from models import Hybrid
  from main_models import FSPTJoint
  torch.manual_seed(9);old=Hybrid(4,6,32,64).double();new=FSPTJoint(4,6).double()
  missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
  self.assertEqual(missing,['harmonics']);self.assertEqual(unexpected,[])
  x=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
  a=old(x)[0];b=new(x)[0];torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
  torch.testing.assert_close(torch.autograd.grad(a.square().sum(),x)[0],torch.autograd.grad(b.square().sum(),x)[0],atol=1e-10,rtol=1e-10)
 def test_scalar_tfim_has_correct_masks_and_no_additional_labels(self):
  from main_models import TFIMJoint
  from src.wmps.models.periodic_redq import causal_parameter_masks
  m=TFIMJoint(4).double();expected=causal_parameter_masks(4,[(0,1),(1,2),(2,3)],2,False,True)
  np.testing.assert_array_equal(m.core.active.numpy(),expected)
  x=torch.randn(3,2,7,dtype=torch.float64,requires_grad=True);mom=m.core.scan_steps(x)[:,-1];energy=m(x)[0]
  torch.testing.assert_close(energy,(mom@m.coefficients)[:,None]);self.assertEqual(energy.shape,(3,1))
  energy.square().sum().backward();self.assertEqual(float(x.grad[:,0,:4].abs().sum()),0.)
  self.assertGreater(float(m.core.gru.weight_ih.grad.abs().sum()),0.)
  inactive=(1-m.core.active).bool()[...,None,None].expand_as(m.core.factors)
  self.assertEqual(float(m.core.factors.grad[inactive].abs().sum()),0.)
 def test_vqe_adapter_all_geometries_orders_and_double_gradients(self):
  from main_models import VQEJoint
  from src.wmps.data.nc_tfim import TFIMTask
  from src.wmps.models.periodic_redq import causal_parameter_masks
  for rows,cols,layers,shared in [(1,6,1,False),(1,6,2,False),(2,3,1,True),(2,3,2,True),(3,3,1,True),(3,3,2,True)]:
   task=TFIMTask(rows,cols,layers,shared,1 if shared else -1);m=VQEJoint(task).double()
   np.testing.assert_array_equal(m.active.numpy(),causal_parameter_masks(task.n,task.edges,layers,shared))
   self.assertEqual(len(m.harmonics),4 if shared else 1)
   x=torch.randn(2,*task.control_shape,dtype=torch.float64,requires_grad=True);p=m(x)[0]
   self.assertEqual(p.shape,(2,task.block));p.square().sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
   self.assertEqual(float(x.grad[:,0,:1 if shared else task.n].abs().sum()),0.)
 def test_generic_scan_and_independent_queries_predictions_and_gradients(self):
  from main_models import FSPTJoint
  m=FSPTJoint(4,6).double();x=torch.randn(2,5,dtype=torch.float64,requires_grad=True)
  a=m(x)[0];ga=torch.autograd.grad(a.square().sum(),(x,m.factors,m.gru.weight_ih))
  b=m(x,independent=True)[0];gb=torch.autograd.grad(b.square().sum(),(x,m.factors,m.gru.weight_ih))
  torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
  for aa,bb in zip(ga,gb):torch.testing.assert_close(aa,bb,atol=1e-10,rtol=1e-10)

if __name__=='__main__':unittest.main(verbosity=2)
