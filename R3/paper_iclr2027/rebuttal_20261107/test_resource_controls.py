import unittest
import numpy as np
import torch
from resource_query import MatchedQueryGRU, build_model, eligible_checkpoint
from resource_chain import ChainMeasurement, shifted_gradients, shot_cost

class ResourceTests(unittest.TestCase):
    def test_complete_history_and_gradients(self):
        torch.manual_seed(5)
        a=build_model('cqs_r').double();b=build_model('matched_query').double()
        b.load_state_dict(a.state_dict())
        x=torch.randn(3,12,10,dtype=torch.float64)
        # Original recurrent implementation initializes its state in float32.
        a.float();b.float();x=x.float()
        y=a(x)[0];z=b(x)[0]
        torch.testing.assert_close(y,z,atol=1e-6,rtol=1e-6)
        y.square().mean().backward();z.square().mean().backward()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
        self.assertTrue(eligible_checkpoint(239.99,240))
        self.assertFalse(eligible_checkpoint(240.01,240))

    def test_joint_histograms_physics_and_accounting(self):
        from chain_depth_recovery import TFIMTask
        engine=ChainMeasurement(100)
        rng=np.random.default_rng(19);x=rng.uniform(-np.pi,np.pi,(5,2,11))
        task=TFIMTask(1,6,2,False,-1)
        pz,px=engine.probabilities(x)
        for k in range(5):
            state=task.state(x[k])
            np.testing.assert_allclose(pz[k],state.probabilities(),atol=1e-12,rtol=0)
        # Exact grouped expectations agree with the independent Qiskit energy.
        for J in (-.5,.1,.5):
            energy=px@engine.x_values(J)+pz@engine.z_values(J)
            np.testing.assert_allclose(energy,[task.energy(t,J,-.5) for t in x],atol=1e-12,rtol=0)
        record=engine.measure(x,-.5,rng)
        self.assertEqual(engine.shots,shot_cost(5,100))
        self.assertEqual(record['counts_x'].shape,(5,64))
        np.testing.assert_equal(record['counts_x'].sum(1),100)
        grads=shifted_gradients(engine,x[:2],-.5,rng)
        self.assertEqual(grads.shape,(2,16))
        self.assertEqual(engine.shots,shot_cost(5+2*16*2,100))

if __name__=='__main__':unittest.main()
