"""Interrupted saves and preflight rejection before terminal scoring."""
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from artifacts import save_npz,valid_search,preflight_searches


class ArtifactTests(unittest.TestCase):
    def test_partial_or_missing_second_search_rejected_by_full_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=[Path(folder)/name for name in ('a.npz','b.npz')]
            values=dict(parameters=np.zeros((3,20,2,11)),predictions=np.zeros((3,20)),sigma=np.array([1.,1/6,0.]),seconds=1.)
            save_npz(a,**values)
            self.assertTrue(valid_search(a,2))
            self.assertFalse(valid_search(b,2))
            with self.assertRaises(ValueError):preflight_searches([(a,2,None),(b,2,None)])
            b.write_bytes(b'partial write')
            self.assertFalse(valid_search(b,2))
            with self.assertRaises(ValueError):preflight_searches([(a,2,None),(b,2,None)])
            b.write_bytes(a.read_bytes()[:40])
            self.assertFalse(valid_search(b,2))
            save_npz(b,**values,model_identity='different')
            with self.assertRaises(ValueError):preflight_searches([(a,2,None),(b,2,'expected')])
            self.assertFalse(list(Path(folder).glob('*.tmp')))
