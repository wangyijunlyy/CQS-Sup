"""Replay chronological finite-shot histories, updates, retention and cost accounting."""
from pathlib import Path
import concurrent.futures,hashlib,json,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from run_reuse import BASE,ORDERS,data,warm
from history import HistoryCache
from engine import wrap,write

def close(a,b):np.testing.assert_allclose(a,b,rtol=0,atol=2e-11)
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}

def replay(path):
 info=json.loads((path/'complete.json').read_text());kind=info['kind'];seed=info['seed'];order=info['order']
 cache=HistoryCache(kind,data(kind,seed),'shared' if info.get('method')=='fixed_cqs' else info['regime'])
 if info.get('method')=='fixed_cqs':
  assert info['refits']==0 and info['global_model_hash_before']==info['global_model_hash_after']
  actual=hashlib.sha256((BASE/kind/f'seed{seed}'/'product.pt').read_bytes()).hexdigest();assert actual==info['global_model_hash_before']
  for r in info['rows']:
   assert r['J']==ORDERS[order][r['target']]
   for k,v in cache.counters().items():assert r[k]==v
   assert np.isfinite(load(path/f"target{r['target']:02d}.npz")['parameters']).all()
  return dict(path=str(path.relative_to(HERE)),method='cqs',events=0,draws=0,shots=cache.initial_shots)
 events=load(path/'history_events.npz');draws=load(path/'history_draws.npz');updates=load(path/'updates.npz');pp=events['point_parameters'];offsets=events['offsets']
 close(pp[:cache.initial_count],np.array([p['parameters'] for p in cache.points]))
 update_by_event={int(e):i for i,e in enumerate(updates['event'])};event_start=0;previous=None;draw_end=0
 conf=info['config'];cap=info['online_cap_per_target'];n=conf['shots'];updates_used=0
 for row in info['rows']:
  target=row['target'];j=ORDERS[order][target];assert row['J']==j
  before=cache.online_shots;record=load(path/f'target{target:02d}.npz')
  x,inc,center=warm(cache,j,previous,np.random.default_rng(seed+500000+target*100))
  close(x,record['initial_iterates']);close(inc,record['initial_incumbents']);close(center,record['center'])
  iteration=0;directions=np.random.default_rng(seed+600000+target*100)
  for event in range(event_start,row['event_end']):
   assert events['target_index'][event]==target and events['j'][event]==j
   ids=events['point_indices'][offsets[event]:offsets[event+1]].astype(int);points=pp[ids]
   cache.hits+=sum(cache._missing(t,n)==0 for t in points);unique={cache.key(t):t for t in points};cache.deduplications+=len(points)-len(unique)
   assigned=np.array([cache._ensure(t) for t in points]);np.testing.assert_array_equal(ids,assigned)
   missing=[(cache.keys[cache.key(t)],cache._missing(t,n)) for t in unique.values() if cache._missing(t,n)>0]
   a=int(events['draw_start'][event]);b=int(events['draw_end'][event]);assert a==draw_end and b-a==len(missing)
   if missing:
    np.testing.assert_array_equal(draws['point_indices'][a:b],[z[0] for z in missing]);np.testing.assert_array_equal(draws['shots_per_group'][a:b],[z[1] for z in missing])
    close(draws['parameters'][a:b],pp[draws['point_indices'][a:b]])
    for group in ['counts_x','counts_z']:np.testing.assert_array_equal(draws[group][a:b].sum(1),draws['shots_per_group'][a:b])
    cache.feedback_rounds+=1;cache.new_points+=b-a;cache.online_shots+=int(2*draws['shots_per_group'][a:b].sum())
    assert np.all(draws['round_id'][a:b]==cache.feedback_rounds) and np.all(draws['target_index'][a:b]==target)
    for index,cx,cz in zip(draws['point_indices'][a:b],draws['counts_x'][a:b],draws['counts_z'][a:b]):cache._add_group(index,cx,cz)
   draw_end=b
   assert events['round_id'][event]==cache.feedback_rounds and events['online_shots'][event]==cache.online_shots
   mean,var=cache.evaluate(points,j);close(mean,events['energy'][offsets[event]:offsets[event+1]]);close(var,events['variance'][offsets[event]:offsets[event+1]])
   stage=str(events['stage'][event]);assert int(events['min_shots'][event])==n
   if stage=='gradient':
    ui=update_by_event[event];updates_used+=1;assert updates['target'][ui]==target
    delta=directions.choice([-1.,1.],(20,len(cache.active)));c=conf['c']/(iteration+1)**.101
    close(delta,updates['delta'][ui]);close(c,updates['c'][ui])
    xp=x.reshape(20,-1).copy();xm=xp.copy();xp[:,cache.active]+=c*delta;xm[:,cache.active]-=c*delta
    expected=wrap(np.r_[xp,xm]);close(points.reshape(40,-1)[:,cache.active],expected[:,cache.active])
    iteration+=1;gain=conf['gain']*(11/(iteration+10))**.602;close(gain,updates['gain'][ui])
    flat=x.reshape(20,-1).copy();flat[:,cache.active]-=gain*(mean[:20]-mean[20:])[:,None]*delta/(2*c);x=wrap(flat).reshape(x.shape);close(x,updates['after'][ui])
   else:
    assert stage=='retention' and event==row['event_end']-1
    close(points.reshape(40,-1)[:,cache.active],np.r_[x,inc].reshape(40,-1)[:,cache.active])
    z=0. if kind=='grid' else 2.;take=mean[:20]+z*np.sqrt(var[:20]+var[20:])<mean[20:];inc=np.where(take[:,None,None],x,inc)
  close(record['final_iterates'],x);close(record['parameters'],inc)
  # Cache keys are exact float bytes. Arithmetic replay can differ by an ulp,
  # so use the verified stored control for the next exact-key lookup.
  previous=record['parameters'].copy()
  assert iteration==row['iterations'] and cache.online_shots-before==row['online_used_for_target'] and cache.online_shots-before<=cap
  for k,v in cache.counters().items():assert row[k]==v,(path,target,k,row[k],v)
  event_start=row['event_end']
 assert event_start==len(events['j']) and draw_end==len(draws['point_indices']) and updates_used==len(updates['event'])
 assert len(cache.points)==len(pp)
 return dict(path=str(path.relative_to(HERE)),method='spsa',events=event_start,draws=draw_end,shots=cache.initial_shots+cache.online_shots)

def main():
 assert (HERE/'all_saved.json').exists(),'All workloads must be saved before audit/scoring.'
 frozen=json.loads((HERE/'frozen_selection.json').read_text())
 for name,expected in frozen['source_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
 for name,expected in {**frozen['dependency_hashes'],**frozen['model_and_data_hashes']}.items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==expected
 roots=sorted(p.parent for p in (HERE/'runs').rglob('complete.json'));assert len(roots)==360
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(replay,roots))
 write(HERE/'audit.json',dict(passed=True,sequences=len(records),events=sum(r['events'] for r in records),draws=sum(r['draws'] for r in records),checks=['chronological target-local history, no future labels','all grouped draws and precision topups','cached recombination energies and variances','warmstarts and frozen SPSA updates','retention and all20 saved outputs','actual shots and logical feedback rounds','unchanged existing global models and datasets','source hashes frozen before formal workloads'],records=records,time_utc=time.time()))
 print('AUDIT PASSED',len(records),flush=True)

if __name__=='__main__':main()
