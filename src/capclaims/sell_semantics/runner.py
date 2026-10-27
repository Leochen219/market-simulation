"""Frozen matched-price study, live request ledger, and strict evidence replay."""
from datetime import datetime, timezone
from pathlib import Path
from capclaims.behavioral_memory.transport import CostLedger as BaseLedger, MODEL, RESERVATION

BUDGET=250_000_000
LIMIT=64
STUDY='jev-sell-semantics-v1'

class Ledger(BaseLedger):
    def __init__(self,path,*,preview_hash):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (request_id TEXT PRIMARY KEY,category TEXT NOT NULL,state TEXT NOT NULL,committed INTEGER NOT NULL,input_tokens INTEGER,created_utc TEXT NOT NULL,detail TEXT)')
            wanted={'preview_hash':preview_hash,'budget':str(BUDGET),'model':MODEL,'reservation':str(RESERVATION),'study':STUDY,'request_cap':str(LIMIT)}
            old=dict(db.execute('SELECT key,value FROM config'))
            if old and old!=wanted: raise ValueError('ledger mismatch')
            if not old: db.executemany('INSERT INTO config VALUES (?,?)',wanted.items())
    def reserve(self,request_id,category='human'):
        if not isinstance(request_id,str) or not request_id or category!='human': raise ValueError('bad request')
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM attempts WHERE request_id=?',(request_id,)).fetchone(): raise ValueError('duplicate request')
            count,total=db.execute('SELECT COUNT(*),COALESCE(SUM(committed),0) FROM attempts').fetchone()
            if count>=LIMIT or total+RESERVATION>BUDGET: raise ValueError('request or budget cap')
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',(request_id,category,'reserved',RESERVATION,None,datetime.now(timezone.utc).isoformat(),None))
    def summary(self):
        result=super().summary()
        result.update(request_cap=LIMIT,budget_nanodollars=BUDGET,study=STUDY)
        return result

import asyncio, hashlib, json, math, os, sqlite3
from capclaims.behavioral_memory.transport import JevWireClient, serialize_body, strict_json, MAX_BODY, MAX_INPUT_TOKENS, NANODOLLARS_PER_TOKEN
from capclaims.flat_grid_study import runner as baseline
from capclaims.sell_semantics import design

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/participant-data/jev-sell-semantics-v1-2026-10-27'
BASELINE_SHA='8891b9242ff05f4df240ceac454a9aa7c3cccfe47ef581d9f458febbbe8aa281'
CONTRACT={'max_input_tokens':MAX_INPUT_TOKENS,'nanodollars_per_input_token':NANODOLLARS_PER_TOKEN,'reservation_nanodollars':RESERVATION,'max_body_bytes':MAX_BODY,'max_inflight':10,'requests_per_second':40,'reserved_bytes_per_second':50000,'batch_size':10,'batch_timeout_seconds':600,'retries':0,'long_context':False,'compression':False}

def canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value): return hashlib.sha256(value).hexdigest()
def append(path,value):
    with path.open('ab') as f:
        f.write(canonical(value)+b'\n'); f.flush(); os.fsync(f.fileno())
def hashes():
    files=[Path(__file__),Path(design.__file__),ROOT/'src/capclaims/sell_semantics/__init__.py',ROOT/'src/capclaims/sell_semantics/analysis.py',ROOT/'tools/probes/jev_sell_semantics.py',ROOT/'docs/research/jev-sell-semantics-design-2026-10-27.md',ROOT/'docs/engineering/jev-sell-semantics.md',*map(lambda p: ROOT/p,baseline.hashes())]
    return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in files}
async def body(ctx,row):
    state,options=design.price_spec(ctx,row['conditioned_quantity'],row['arm'])
    data=await serialize_body(state=state,instructions=design.PROMPT_PRICE,criteria=options)
    if len(data)>MAX_BODY: raise ValueError('request exceeds body limit')
    return data
async def preview():
    if sha((baseline.OUT/'plan.json').read_bytes())!=BASELINE_SHA: raise ValueError('baseline plan hash mismatch')
    old,predictions,status=await baseline.replay()
    if status['status']!='COMPLETE' or len(predictions)!=72: raise ValueError('baseline incomplete')
    contexts=[c for c in old['contexts'] if c['side']=='SELL']
    if len(contexts)!=16 or len({c['person_id'] for c in contexts})!=8: raise ValueError('invalid SELL cohort')
    ids={(c['person_id'],c['formal_round'],c['condition']) for c in contexts}
    paths=[r for r in old['schedule'] if design.key(r)[:3] in ids]
    if len(paths)!=32: raise ValueError('invalid baseline paths')
    schedule=design.schedule(paths)
    index={(c['person_id'],c['formal_round'],c['condition']):c for c in contexts}
    bodies={}
    for row in schedule:
        data=await body(index[design.key(row)[:3]],row)
        if row['arm']=='original' and {'sha256':sha(data),'bytes':len(data)} != old['bodies'][design.parent.request_id(row)]: raise ValueError('original differs from baseline')
        bodies[design.request_id(row)]={'sha256':sha(data),'bytes':len(data)}
    return {**old,'contexts':contexts},schedule,bodies
async def freeze(root=OUT):
    old,schedule,bodies=await preview()
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    labels=(baseline.OUT/'labels-private.json').read_bytes()
    with (root/'labels-private.json').open('xb') as f: f.write(labels)
    plan={'schema':STUDY,'contract':CONTRACT,'status':'FROZEN','model':MODEL,'seed':design.old_design.SEED,'schedule_seed':design.SCHEDULE_SEED,'economic_definitions':design.DEFINITIONS,'budget_nanodollars':BUDGET,'request_cap':LIMIT,'baseline_sha256':BASELINE_SHA,'baseline_status':old['status'],'contexts':old['contexts'],'schedule':schedule,'bodies':bodies,'prompt':design.PROMPT_PRICE,'labels_sha256':sha(labels),'code_sha256':hashes(),'max_preview_body_bytes':max(x['bytes'] for x in bodies.values())}
    raw=canonical(plan)+b'\n'
    with (root/'plan.json').open('xb') as f: f.write(raw)
    with (root/'manifest.sha256').open('x',encoding='ascii') as f: f.write(sha(raw)+'\n')
    return {'status':'FROZEN','plan_sha256':sha(raw),'paths':len(schedule),'max_body_bytes':plan['max_preview_body_bytes']}
async def valid_plan(root):
    root=Path(root); raw=(root/'plan.json').read_bytes(); plan=strict_json(raw)
    if sha(raw)!=(root/'manifest.sha256').read_text(encoding='ascii').strip(): raise ValueError('manifest mismatch')
    if (plan.get('contract')!=CONTRACT or plan.get('schedule_seed')!=design.SCHEDULE_SEED or plan.get('economic_definitions')!=design.DEFINITIONS or plan.get('schema')!=STUDY or plan.get('model')!=MODEL or plan.get('seed')!=design.old_design.SEED or plan.get('budget_nanodollars')!=BUDGET or plan.get('request_cap')!=LIMIT or plan.get('baseline_sha256')!=BASELINE_SHA or plan.get('prompt')!=design.PROMPT_PRICE or plan.get('code_sha256')!=hashes()): raise ValueError('frozen contract mismatch')
    if sha((root/'labels-private.json').read_bytes())!=plan['labels_sha256'] or (root/'labels-private.json').read_bytes()!=(baseline.OUT/'labels-private.json').read_bytes(): raise ValueError('labels mismatch')
    old,schedule,bodies=await preview()
    if plan['contexts']!=old['contexts'] or plan['schedule']!=schedule or plan['bodies']!=bodies: raise ValueError('baseline, requests or schedule mismatch')
    return plan

def evidence(root,request_id):
    folder=root/'wire-evidence'/sha(request_id.encode())
    return {name:(sha((folder/f'{name}.json').read_bytes()) if (folder/f'{name}.json').exists() else None) for name in ('request','response','attempt','timing','failure')}
async def call(client,root,ctx,row,expected_body=None):
    request_id=design.request_id(row); data=None
    try:
        data=await body(ctx,row)
        if expected_body is not None and expected_body!={'sha256':sha(data),'bytes':len(data)}: raise ValueError('body differs from frozen request')
        wire=await client.send_body(request_id=request_id,body=data)
        answer=wire['answers']['decision']; raw=answer['probabilities']
        _,options=design.price_spec(ctx,row['conditioned_quantity'],row['arm']); decision=design.decide(raw,list(options),row)
        norm,_=design.normalize_jev_probabilities(raw,list(options)); best=max(norm.values())
        choice=answer.get('choice')
        record={'path':row,'request_id':request_id,'candidate_ids':list(options),'raw_probabilities':raw,'provider_choice':choice,'provider_nonmax':choice in norm and norm[choice]<best,'decision':decision,'evidence_sha256':evidence(root,request_id)}
        append(root/'trace.jsonl',record)
        return record
    except BaseException as exc:
        client.halted=True
        with sqlite3.connect(root/'cost-ledger.sqlite') as db: dispatched=db.execute('SELECT 1 FROM attempts WHERE request_id=?',(request_id,)).fetchone() is not None
        append(root/'technical.jsonl',{'path':row,'request_id':request_id,'constructed_request_sha256':sha(data) if data is not None else None,'dispatched':dispatched,'evidence_sha256':evidence(root,request_id),'error_type':type(exc).__name__})
        raise
async def run(root=OUT):
    root=Path(root); plan=await valid_plan(root)
    marker=root/'started'
    if marker.exists(): raise FileExistsError('already started; no retry')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False); api_key=os.environ.get('TYPESAFE_API_KEY')
    if not api_key: raise ValueError('provider key missing')
    ledger=Ledger(root/'cost-ledger.sqlite',preview_hash=sha((root/'plan.json').read_bytes()))
    fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try: os.write(fd,(sha((root/'plan.json').read_bytes())+'\n').encode()); os.fsync(fd)
    finally: os.close(fd)
    contexts={(c['person_id'],c['formal_round'],c['condition']):c for c in plan['contexts']}
    phase='client_start'
    try:
        async with JevWireClient(api_key=api_key,ledger=ledger,evidence_dir=root/'wire-evidence') as client:
            phase='dispatch'
            for repeat in (0,1):
                block=[r for r in plan['schedule'] if r['repeat']==repeat]
                for i in range(0,len(block),10):
                    results=await asyncio.wait_for(asyncio.gather(*(call(client,root,contexts[design.key(r)[:3]],r,plan['bodies'][design.request_id(r)]) for r in block[i:i+10]),return_exceptions=True),600)
                    error=next((r for r in results if isinstance(r,BaseException)),None)
                    if error: raise error
    except BaseException as exc:
        append(root/'status.jsonl',{'status':'PARTIAL_TECHNICAL','physical_requests':ledger.summary()['physical_requests'],'failure_type':type(exc).__name__,'phase':phase})
        raise
    result={'status':'COMPLETE','physical_requests':ledger.summary()['physical_requests']}
    append(root/'status.jsonl',result); return result

def readlines(path):
    return [strict_json(line.encode()) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
async def replay(root=OUT):
    root=Path(root); plan=await valid_plan(root); plan_hash=sha((root/'plan.json').read_bytes())
    if (root/'started').read_text(encoding='ascii').strip()!=plan_hash: raise ValueError('started marker mismatch')
    statuses=readlines(root/'status.jsonl')
    if len(statuses)!=1 or statuses[0].get('status') not in ('COMPLETE','PARTIAL_TECHNICAL'): raise ValueError('status mismatch')
    status=statuses[0]
    with sqlite3.connect(root/'cost-ledger.sqlite') as db:
        ledger={r[0]:r[1:] for r in db.execute('SELECT request_id,state,committed,input_tokens FROM attempts')}
        config=dict(db.execute('SELECT key,value FROM config'))
    expected={'preview_hash':plan_hash,'budget':str(BUDGET),'model':MODEL,'reservation':str(RESERVATION),'study':STUDY,'request_cap':str(LIMIT)}
    if config!=expected or len(ledger)>LIMIT or sum(v[1] for v in ledger.values())>BUDGET or status['physical_requests']!=len(ledger): raise ValueError('ledger mismatch')
    scheduled={design.request_id(r):r for r in plan['schedule']}
    contexts={(c['person_id'],c['formal_round'],c['condition']):c for c in plan['contexts']}
    traces=readlines(root/'trace.jsonl'); failures=readlines(root/'technical.jsonl'); seen=set(); predictions=[]
    for event in traces+failures:
        rid=event.get('request_id'); row=scheduled.get(rid)
        if row is None or row!=event.get('path') or rid in seen: raise ValueError('identity mismatch')
        seen.add(rid); actual=evidence(root,rid)
        if actual!=event.get('evidence_sha256'): raise ValueError('evidence mismatch')
        failed=event in failures
        try:
            data=await body(contexts[design.key(row)[:3]],row)
        except Exception:
            if not failed or event.get('constructed_request_sha256') is not None or event.get('dispatched') or rid in ledger or any(actual.values()): raise ValueError('construction failure evidence mismatch')
            continue
        if sha(data)!=plan['bodies'][rid]['sha256'] or len(data)!=plan['bodies'][rid]['bytes']: raise ValueError('frozen body mismatch')
        if failed:
            constructed=event.get('constructed_request_sha256')
            if event.get('dispatched')!=(rid in ledger): raise ValueError('failure dispatch mismatch')
            if constructed is None:
                if rid in ledger or any(actual.values()): raise ValueError('construction failure evidence mismatch')
            elif constructed!=sha(data): raise ValueError('failure body mismatch')
        if rid not in ledger:
            if not failed or any(actual.values()): raise ValueError('unaccounted evidence')
            continue
        if not actual['request'] or not actual['attempt'] or actual['request']!=sha(data): raise ValueError('missing or different posted request')
        folder=root/'wire-evidence'/sha(rid.encode())
        if (folder/'request.json').read_bytes()!=data: raise ValueError('request bytes mismatch')
        attempt=strict_json((folder/'attempt.json').read_bytes())
        if (attempt.get('request_id')!=rid or attempt.get('request_sha256')!=sha(data) or attempt.get('body_bytes')!=len(data) or attempt.get('method')!='POST' or attempt.get('url')!='https://api.typesafe.ai/v1/systemone' or attempt.get('reserved_bytes')!=MAX_BODY or attempt.get('reserved_nanodollars')!=RESERVATION or attempt.get('long_context') is not False or attempt.get('compression') is not False or attempt.get('retry_count')!=0): raise ValueError('attempt mismatch')
        state,committed,tokens=ledger[rid]
        if state in ('unknown','reserved'):
            if not failed or committed!=RESERVATION or tokens is not None or not actual['failure']: raise ValueError('unknown reservation mismatch')
        elif state=='settled':
            if type(tokens) is not int or not 0<=tokens<=MAX_INPUT_TOKENS or committed!=tokens*NANODOLLARS_PER_TOKEN or not actual['response'] or not actual['timing']: raise ValueError('settlement mismatch')
            response=strict_json((folder/'response.json').read_bytes()); usage=response.get('usage',{})
            if response.get('model')!=MODEL or usage.get('input_tokens')!=tokens or type(usage.get('output_tokens')) is not int or usage['output_tokens']<0: raise ValueError('response usage mismatch')
        else: raise ValueError('invalid ledger state')
        if actual['timing']:
            timing=strict_json((folder/'timing.json').read_bytes())
            if not all(type(timing.get(k)) in (int,float) and math.isfinite(timing[k]) for k in ('started_monotonic','finished_monotonic','elapsed_seconds')) or timing['finished_monotonic']<timing['started_monotonic'] or timing['elapsed_seconds']<0: raise ValueError('timing mismatch')
        if failed:
            if actual['failure']:
                failure=strict_json((folder/'failure.json').read_bytes())
                if (failure.get('failure_type')!=event['error_type'] or not all(type(failure.get(k)) in (int,float) and math.isfinite(failure[k]) for k in ('started_monotonic','finished_monotonic')) or failure['finished_monotonic']<failure['started_monotonic']): raise ValueError('failure metadata mismatch')
            if actual['timing'] and type(timing.get('status_code')) is not int: raise ValueError('failure HTTP status mismatch')
            continue
        if state!='settled' or timing.get('status_code')!=200: raise ValueError('success transport mismatch')
        _,options=design.price_spec(contexts[design.key(row)[:3]],row['conditioned_quantity'],row['arm']); ids=list(options)
        answer=response['answers']['decision']; raw=answer['probabilities']; decision=design.decide(raw,ids,row)
        norm,_=design.normalize_jev_probabilities(raw,ids); choice=answer.get('choice')
        if event['candidate_ids']!=ids or event['raw_probabilities']!=raw or event['provider_choice']!=choice or event['provider_nonmax']!=(choice in norm and norm[choice]<max(norm.values())) or event['decision']!=decision: raise ValueError('decision mismatch')
        sid=decision['selected_id']
        predictions.append({**{k:row[k] for k in ('person_id','formal_round','condition','repeat','arm')},'fixed_quantity':row['conditioned_quantity'],'price':float(options[sid]) if sid else None,'outcome':decision['status'],'raw_probabilities':raw,'decision':decision})
    if set(ledger)!={r for r in seen if r in ledger}: raise ValueError('unaccounted attempts')
    if status['status']=='COMPLETE' and (failures or len(traces)!=64 or len(ledger)!=64 or seen!=set(scheduled)): raise ValueError('incomplete claimed complete')
    if status['status']=='PARTIAL_TECHNICAL' and not failures and not (status.get('phase')=='client_start' and not ledger and not traces and status.get('failure_type')): raise ValueError('missing technical evidence')
    return plan,predictions,status
