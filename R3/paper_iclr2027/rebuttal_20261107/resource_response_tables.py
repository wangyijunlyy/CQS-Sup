"""Result tables from the verified resource-matched controls."""
from pathlib import Path
import json
E=Path(__file__).resolve().parent/'evidence/resource_controls_20261108'
def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def query_table():
 assert json.loads((E/'query_verification.json').read_text())['passed']
 data=json.loads((E/'query_summary.json').read_text());labels={'cqs_r':'CQS-R','matched_query':'Query-GRU, chronological','original_query':'Query-GRU, repeated target'}
 f=lambda r:f"{r['mean']:.3f} ± {r['sd']:.3f}"
 return table(['Method','Test MAE','Actual training time (s)','Optimizer updates'],[[labels[r['arm']],f(r['test_mae']),f(r['actual_seconds']),f"{r['optimizer_steps']['mean']:.0f}"] for r in data])
def chain_curve_table():
 assert json.loads((E/'chain_verification.json').read_text())['passed']
 data=json.loads((E/'chain_summary.json').read_text());rows=[]
 for J in (-.5,.5):
  for budget in (578000,1132000,1686000):
   values=[]
   for method in ('product','spsa','parameter_shift_adam'):
    r=next(r for r in data if (r['J'],r['method'],r['total_shots'])==(J,method,budget));values.append(f"{r['mean']:.3f} ± {r['sd']:.3f}")
   rows.append([f'{J:+.1f}',f'{budget:,}',*values])
 return table(['J','Total shots','CQS-P + local correction','Direct SPSA','Direct parameter-shift Adam'],rows)
