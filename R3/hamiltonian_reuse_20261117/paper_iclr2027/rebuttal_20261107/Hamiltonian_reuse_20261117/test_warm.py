from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_reuse import warm
from history import HistoryCache

class Warmstarts(unittest.TestCase):
 def test_later_targets_retain_perturbed_starts_away_from_chain_saddle(self):
  cache=HistoryCache('chain');previous=np.broadcast_to(cache.reference,(20,2,11)).copy()
  x,inc,center=warm(cache,-.1,previous,np.random.default_rng(13))
  np.testing.assert_array_equal(inc,previous)
  delta=(x.reshape(20,-1)[1:,cache.active]-center.reshape(-1)[cache.active]+np.pi)%(2*np.pi)-np.pi
  self.assertTrue(np.all(np.linalg.norm(delta,axis=1)>0))
  self.assertTrue(np.all(np.abs(delta)<=.2))

if __name__=='__main__':unittest.main()
