"""Atomic scientific outputs and full-matrix validation before truth scoring."""
from pathlib import Path
from zipfile import BadZipFile
import numpy as np


def save_npz(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**values)
    temp.replace(path)


def valid_search(path,steps,identity=None):
    try:
        with Path(path).open('rb') as stream, np.load(stream) as a:
            expected={'parameters':(steps+1,20,2,11),'predictions':(steps+1,20),'sigma':(steps+1,)}
            for key,shape in expected.items():
                if a[key].shape!=shape or not np.isfinite(a[key]).all():return False
            if not np.isfinite(float(a['seconds'])):return False
            if identity is not None and str(a['model_identity'])!=identity:return False
        return True
    except (OSError,KeyError,ValueError,EOFError,BadZipFile):return False


def valid_timed(path,cap=None,identity=None):
    try:
        with Path(path).open('rb') as stream, np.load(stream) as a:
            blocks=int(a['completed_blocks'])
            expected={'retained_parameters':(20,2,11),'retained_prediction':(20,),
                'block_parameters':(blocks,20,2,11),'block_predictions':(blocks,20),
                'block_starts':(blocks,20,2,11),'eligible_seconds':(blocks,)}
            if blocks<1:return False
            for key,shape in expected.items():
                if a[key].shape!=shape or not np.isfinite(a[key]).all():return False
            if not np.isfinite(float(a['actual_seconds'])):return False
            actual_cap=float(a['cap'])
            if cap is not None and actual_cap!=cap:return False
            if np.any(a['eligible_seconds']>actual_cap):return False
            if identity is not None and str(a['model_identity'])!=identity:return False
        return True
    except (OSError,KeyError,ValueError,EOFError,BadZipFile):return False


def preflight_searches(records):
    for path,steps,identity in records:
        if not valid_search(path,steps,identity):raise ValueError('Incomplete search matrix: '+str(path))


def preflight_timed(records,cap):
    for path,identity in records:
        if not valid_timed(path,cap,identity):raise ValueError('Incomplete timed matrix: '+str(path))
