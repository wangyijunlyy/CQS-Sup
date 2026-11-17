from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from history import HistoryCache

class CacheTests(unittest.TestCase):
 def test_cached_energy_reweights_and_query_deduplicates(self):
  c=HistoryCache('grid');x=np.zeros((2,2));x[0,1]=.3;r=np.random.default_rng(2)
  c.request(np.repeat(x[None],3,0),100,r,'first')
  self.assertEqual(c.online_shots,200);self.assertEqual(c.feedback_rounds,1);self.assertEqual(c.new_points,1)
  for j in [-.1,-.5]:
   mean,var=c.evaluate(x[None],j)
   self.assertTrue(np.isfinite(mean[0]));self.assertGreaterEqual(var[0],0)
  before=c.online_shots;c.request(x[None],100,r,'cache')
  self.assertEqual(c.online_shots,before);self.assertEqual(c.feedback_rounds,1)
  c.request(x[None],150,r,'topup');self.assertEqual(c.online_shots,300);self.assertEqual(c.feedback_rounds,2)
 def test_grouped_history_recombination_matches_raw_record(self):
  c=HistoryCache('chain');x=np.random.default_rng(4).normal(0,.3,(3,2,11));r=np.random.default_rng(5)
  c.request(x,200,r,'sample')
  for j in [-.05,-.5]:
   mean,var=c.evaluate(x,j)
   rows=c.raw_records
   vx=.5*c.x_bins;vz=-j*c.z_bins
   mx=rows[0]['counts_x']@vx/200;mz=rows[0]['counts_z']@vz/200
   expected=(rows[0]['counts_x']@vx**2/200-mx**2+rows[0]['counts_z']@vz**2/200-mz**2)/199
   np.testing.assert_allclose(mean,mx+mz,atol=1e-12);np.testing.assert_allclose(var,expected,atol=1e-12)
if __name__=='__main__':unittest.main()
