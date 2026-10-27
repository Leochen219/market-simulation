"""Read-only aggregate run accounting; call only after validated replay."""
import json,sqlite3,hashlib
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];root=ROOT/'artifacts/participant-data/jev-long-history-v1-2026-10-27';public=ROOT/'docs/analysis/jev-long-history-results-2026-10-27.json'
report=json.loads(public.read_text(encoding='utf-8')); status=report['status']
if status['status'] not in ('COMPLETE','PARTIAL_PREDICTION_FAILURE'):raise ValueError('run not complete enough to report')
rows=[json.loads(line) for line in (root/'trace.jsonl').read_text(encoding='utf-8').splitlines()];uri=(root/'cost-ledger.sqlite').resolve().as_uri()+'?mode=ro'
with sqlite3.connect(uri,uri=True) as db: records=list(db.execute('SELECT request_id,state,committed,input_tokens FROM attempts'))
if len(records)!=status['physical_requests'] or any(x[1]!='settled' for x in records):raise ValueError('ledger not fully settled')
ledger={r[0]:r for r in records};timings=[];by=defaultdict(lambda:{'requests':0,'body_bytes_sum':0,'input_tokens_sum':0})
for row in rows:
    step=row['step'];rid=step['request_id'];folder=root/'wire-evidence'/hashlib.sha256(rid.encode()).hexdigest();attempt=json.loads((folder/'attempt.json').read_text(encoding='utf-8'));timing=json.loads((folder/'timing.json').read_text(encoding='utf-8'));timings.append(timing);g=by[row['path']['condition']];g['requests']+=1;g['body_bytes_sum']+=attempt['body_bytes'];g['input_tokens_sum']+=ledger[rid][3]
if len(timings)!=len(records):raise ValueError('timing count mismatch')
for group in by.values():group['mean_body_bytes']=group['body_bytes_sum']/group['requests'];group['mean_input_tokens']=group['input_tokens_sum']/group['requests']
out={'schema':'jev-long-history-run-supplement-v1','plan_sha256':hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),'settled_requests':len(records),'input_tokens':sum(x[3] for x in records),'cost_nanodollars':sum(x[2] for x in records),'elapsed_seconds':max(x['finished_monotonic'] for x in timings)-min(x['started_monotonic'] for x in timings),'conditions':dict(by)}
path=public.parent/(public.stem+'-supplement.json');path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))
