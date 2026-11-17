from pathlib import Path
import sys,unittest
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
from engine import PhysicalSampler,grid_dataset,shadow_dataset
from run_evidence import TFIMTask

class Measurements(unittest.TestCase):
 def test_original_circuit_probabilities_both_tasks(self):
  from qiskit import QuantumCircuit
  for kind in ['grid','chain']:
   e=PhysicalSampler(kind,-.5);task=TFIMTask(3,3,2,True,1) if kind=='grid' else TFIMTask(1,6,2,False,-1)
   x=np.random.default_rng(40).uniform(-np.pi,np.pi,(3,*task.control_shape))
   pz,px=e.probabilities(x)
   for i,c in enumerate(x):
    state=task.state(c);q=QuantumCircuit(task.n);q.h(range(task.n))
    np.testing.assert_allclose(pz[i],state.probabilities(),atol=2e-14)
    np.testing.assert_allclose(px[i],state.evolve(q).probabilities(),atol=2e-14)
 def test_full_pilot_cost_is_reused_in_training_counts(self):
  data=grid_dataset(45)
  self.assertEqual(int(data['shots_used']),4000000)
  np.testing.assert_array_equal(data['counts_x'].sum(1),np.full(200,10000))
  np.testing.assert_array_equal(data['counts_z'].sum(1),np.full(200,10000))
  self.assertEqual(int(data['pilot_counts_x'].sum()+data['pilot_counts_z'].sum()),40000)
  self.assertTrue(np.all(data['counts_x'][:20]>=data['pilot_counts_x']))
  e=PhysicalSampler('grid',-.5)
  np.testing.assert_allclose(e.terms(data['counts_z'],data['counts_x']),data['measured_terms'],atol=1e-14)
 def test_shadow_record_labels_reconstruct(self):
  d=shadow_dataset(45,controls=12,shots=10)
  b,r=d['bits'],d['recipes'];s=1.-2*b;e=PhysicalSampler('chain',-.5)
  y=np.c_[(3*s*(r==0)).mean(1),np.stack([(9*s[:,:,i]*s[:,:,j]*(r[:,:,i]==2)*(r[:,:,j]==2)).mean(1) for i,j in e.edges],axis=1)]
  np.testing.assert_array_equal(y,d['measured_terms'])
  self.assertEqual(int(d['shots_used']),120)
if __name__=='__main__':unittest.main()
