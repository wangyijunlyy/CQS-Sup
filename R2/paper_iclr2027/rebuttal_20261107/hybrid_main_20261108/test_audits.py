import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import analyze_main as audit
from run_main import cells,load_data,PROJECT

class AuditGates(unittest.TestCase):
 def test_saved_training_membership_must_match_source_even_if_validation_unchanged(self):
  c=cells()[0];_,_,tr,va,te=load_data(c);tr=tr.copy();te=te.copy();tr[0],te[0]=te[0],tr[0]
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);np.savez(p/'split.npz',train=tr,val=va,test=te)
   with self.assertRaises(ValueError):audit.check_saved_split(c,p)
 def test_extra_file_cannot_replace_expected_blind_search(self):
  c=next(c for c in cells() if c['task']=='vqe');starts=np.load(PROJECT/c['source']/'initial_parameters.npy')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/c['name'];p.mkdir();np.savez(p/'J+0.1.npz',parameters=np.repeat(starts[None],101,0),predictions=np.zeros((101,20)))
   with self.assertRaises(ValueError):audit.validate_searches([c],Path(d))
 def test_blind_search_requires_all_iterations_starts_and_finite_predictions(self):
  c=next(c for c in cells() if c['task']=='vqe');starts=np.load(PROJECT/c['source']/'initial_parameters.npy')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/c['name'];p.mkdir();file=p/'J-0.1.npz'
   for positions,pred in [(np.repeat(starts[None],100,0),np.zeros((100,20))), (np.repeat((starts+.1)[None],101,0),np.zeros((101,20))), (np.repeat(starts[None],101,0),np.full((101,20),np.nan))]:
    np.savez(file,parameters=positions,predictions=pred)
    with self.assertRaises(ValueError):audit.validate_searches([c],Path(d))
   np.savez(file,parameters=np.repeat(starts[None],101,0),predictions=np.zeros((101,20)))
   self.assertEqual(audit.validate_searches([c],Path(d)),1)
 def test_worker_failure_cleanup_terminates_and_waits_only_surviving_owned_children(self):
  from safety import stop_owned
  running=Mock();running.poll.return_value=None;done=Mock();done.poll.return_value=0
  stop_owned([(0,running),(1,done)])
  running.terminate.assert_called_once();running.wait.assert_called_once();done.terminate.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
