"""Read-only run-level accounting and aggregate decoder agreement supplement."""
import json,sqlite3,glob
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; root=ROOT/'artifacts/participant-data/jev-expanded-grid-v1-2026-10-27'
public=ROOT/'docs/analysis/jev-expanded-grid-results-2026-10-27.json'
report=json.loads(public.read_text(encoding='utf-8')); status=report['status']
if status['status'] not in ('COMPLETE','PARTIAL_PREDICTION_FAILURE'): raise ValueError('run not fully observed')
rows=[json.loads(line) for line in (root/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
q=[x['step']['decisions'] for x in rows if x['step']['stage']=='quantity']
uri=(root/'cost-ledger.sqlite').resolve().as_uri()+'?mode=ro'
with sqlite3.connect(uri,uri=True) as db: counts=list(db.execute('SELECT state,COUNT(*),SUM(committed),SUM(input_tokens) FROM attempts GROUP BY state'))
if len(counts)!=1 or counts[0][0]!='settled' or counts[0][1]!=status['physical_requests']: raise ValueError('ledger not fully settled')
timing=[json.loads(Path(path).read_text(encoding='utf-8')) for path in glob.glob(str(root/'wire-evidence/*/timing.json'))]
if len(timing)!=counts[0][1]: raise ValueError('timing count mismatch')
result={'schema':'jev-expanded-grid-run-supplement-v1','quantity_5_3_agreement_n':sum(d['cutoff5'].get('selected_id')==d['cutoff3'].get('selected_id') and d['cutoff5'].get('selected_id') is not None for d in q),'quantity_decision_n':len(q),'settled_request_n':counts[0][1],'settled_input_tokens':counts[0][3],'settled_cost_nanodollars':counts[0][2],'first_to_last_http_elapsed_seconds':max(x['finished_monotonic'] for x in timing)-min(x['started_monotonic'] for x in timing)}
output=public.parent/(public.stem+'-supplement.json');output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
