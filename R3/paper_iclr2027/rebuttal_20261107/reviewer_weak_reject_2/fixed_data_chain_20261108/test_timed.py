"""Deadline and periodic diversity checks for the pure fixed-data time arm."""
from pathlib import Path
import sys,unittest
import numpy as np
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from timed import select_candidates,timed_search


class TimedTests(unittest.TestCase):
    def test_late_pool_selection_does_not_commit_completed_block(self):
        class Clock:
            value=0.
            def __call__(self):self.value+=.001;return self.value
        clock=Clock();calls=[]
        def delayed_selection(*args,**kwargs):
            result=select_candidates(*args,**kwargs);calls.append(1)
            if len(calls)==2:clock.value=.5
            return result
        x=np.random.default_rng(2).uniform(-np.pi,np.pi,(20,2,11))
        fn=lambda z,s:(z.cos()*np.exp(-s*s/2)).sum((1,2))
        with patch('timed.select_candidates',side_effect=delayed_selection):
            result=timed_search(fn,x,seed=4,cap=.4,clock=clock)
        self.assertEqual(len(calls),2)
        self.assertEqual(result['completed_blocks'],1)
        np.testing.assert_allclose(np.sort(result['retained_prediction']),np.sort(result['block_predictions'][0]))

    def test_diversity_uses_periodic_active_coordinates(self):
        x=np.zeros((4,2,11));x[0,1,:]=np.pi-.01;x[1,1,:]=-np.pi+.01
        x[2,1,:]=0.;x[3,1,:]=1.
        chosen=select_candidates(x,np.array([-4.,-3.9,-3.,-2.]),count=2,min_rms=.2)
        np.testing.assert_array_equal(chosen,np.array([0,2]))

    def test_incomplete_or_late_blocks_cannot_change_retained_pool(self):
        class Clock:
            value=0.
            def __call__(self):
                self.value+=.001
                return self.value
        clock=Clock();rng=np.random.default_rng(8)
        starts=rng.uniform(-np.pi,np.pi,(20,2,11))
        fn=lambda z,s:(z.cos()*np.exp(-s*s/2)).sum((1,2))
        result=timed_search(fn,starts,seed=4,cap=.13,clock=clock)
        self.assertEqual(result['completed_blocks'],1)
        self.assertTrue(all(v<=.13 for v in result['eligible_seconds']))
        np.testing.assert_allclose(np.sort(result['retained_prediction']),np.sort(result['block_predictions'][0]))
        self.assertEqual(len(result['retained_parameters']),20)


if __name__=='__main__':unittest.main()
