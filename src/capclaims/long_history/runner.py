"""Frozen controlled history study, strict replay, and bounded cost control."""
from __future__ import annotations
import asyncio,hashlib,json,os,sqlite3,math
from datetime import datetime,timezone
from pathlib import Path
from capclaims.long_history.source import load as load_source
from capclaims.behavioral_memory.transport import CostLedger as BaseLedger,JevWireClient,MODEL,MAX_BODY,MAX_INPUT_TOKENS,NANODOLLARS_PER_TOKEN,RESERVATION,serialize_body,strict_json
from capclaims.backends.jev import normalize_jev_probabilities
from capclaims.grid_sampling import decode as decode5
from capclaims.grid_sampling_v3 import decode as decode3
from capclaims.long_history import design,source
from tools.probes import jev_grid_probability as v1

ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'artifacts/participant-data/jev-long-history-v1-2026-10-27'
BUDGET=1_500_000_000; LIMIT=432; STUDY='jev-long-history-v1'; STAGES=('quantity','band','price')

def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b): return hashlib.sha256(b).hexdigest()
def append(path,value):
    with path.open('ab') as f: f.write(canonical(value)+b'\n'); f.flush(); os.fsync(f.fileno())
def paths():
    from tools.probes import jev_grid_probability_v2 as v2
    return [Path(__file__),Path(design.__file__),Path(source.__file__),ROOT/'src/capclaims/long_history/analysis.py',ROOT/'tools/probes/jev_long_history.py',ROOT/'tools/analysis/compare_jev_grid_human_distributions.py',ROOT/'docs/research/jev-long-history-design-2026-10-27.md',ROOT/'docs/engineering/jev-long-history.md',ROOT/'src/capclaims/grid_sampling.py',ROOT/'src/capclaims/grid_sampling_v3.py',*v2._source_code_paths()]
def hashes(): return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in paths()}
def rid(row,stage): return f"lh1-{row['condition']}-r{row['repeat']}-{row['person_id']}-t{row['formal_round']}-{stage}"
def band_options(): return {f'b{i}':f'{lo*2.5:.2f}-{hi*2.5:.2f} dollars' for i,(lo,hi) in enumerate(v1.BANDS)}
def stage_spec(ctx,stage,q=None,band=None):
    base=ctx['base_state']; side=ctx['side']
    if stage=='quantity': return base,v1.criteria(v1.quantity_grid(side,base['own_q_units']),'q')
    if q is None: raise ValueError('missing selected quantity')
    state=dict(base,own_qty_units=q,model_chosen_quantity=q)
    if stage=='band': return state,band_options()
    if stage!='price' or band not in range(4): raise ValueError('invalid price band')
    state['model_chosen_band']=band
    return state,v1.criteria(v1.price_grid(*v1.BANDS[band]),'p')
async def body(state,stage,options): return await serialize_body(state=state,instructions=v1._instructions(stage),criteria=options)

def decisions(raw,ids,row,stage):
    norm,meta=normalize_jev_probabilities(raw,ids); argmax=v1.choose(ids,norm)
    result={'normalized':norm,'normalization':meta,'argmax':{'selected_id':argmax}}
    for label,decoder in (('cutoff5',decode5),('cutoff3',decode3)):
        try: result[label]=decoder(raw,ids,seed=design.SEED,person_id=design.coordinate(row['person_id'],row['formal_round']),repeat=row['repeat'],stage=stage)
        except ValueError as exc:
            if 'no surviving positive-mass candidate' not in str(exc): raise
            result[label]={'error':str(exc)}
    return result

def selection(stage,sid,ids,options):
    if stage=='quantity': return int(options[sid])
    if stage=='band': return int(sid[1:])
    return float(options[sid])

async def preview():
    cases,targets,meta=load_source(); contexts,excluded=design.cohort(cases); max_len=0
    for ctx in contexts:
        qmax=max(v1.quantity_grid(ctx['side'],ctx['base_state']['own_q_units']))
        initial=(ctx['base_state'],v1.criteria(v1.quantity_grid(ctx['side'],ctx['base_state']['own_q_units']),'q'))
        specs=[('quantity',*initial),('band',*stage_spec(ctx,'band',qmax))]
        for band in range(4): specs.append(('price',*stage_spec(ctx,'price',qmax,band)))
        for stage,state,options in specs:
            size=len(await body(state,stage,options)); max_len=max(max_len,size)
            if size>MAX_BODY: raise ValueError('preview exceeds body cap')
    return contexts,targets,meta,max_len

async def freeze(root=OUT):
    contexts,targets,meta,max_len=await preview(); excluded=18-len(contexts)//2
    if root.exists(): raise FileExistsError('output exists; no reset or resume')
    root.mkdir(parents=True); labels_bytes=canonical(targets)+b'\n'; (root/'labels-private.json').write_bytes(labels_bytes)
    plan={'schema':STUDY,'status':'FROZEN','model':MODEL,'seed':design.SEED,'people_n':18,'contexts_n':len(contexts),'paths_n':len(design.schedule(contexts)),'budget_nanodollars':BUDGET,'request_cap':LIMIT,'history_windows':{'short3':3,'long10':10},'excluded_equal':excluded,'prompt':v1.PROMPT,'price_bands':v1.BANDS,'contexts':contexts,'schedule':design.schedule(contexts),'metadata':meta,'labels_sha256':sha(labels_bytes),'code_sha256':hashes(),'max_preview_body_bytes':max_len}
    raw=canonical(plan)+b'\n'; (root/'plan.json').write_bytes(raw); (root/'manifest.sha256').write_text(sha(raw)+'\n',encoding='ascii'); return {'status':'FROZEN','plan_sha256':sha(raw),'contexts':len(contexts),'paths':len(design.schedule(contexts)),'max_body_bytes':max_len}

def valid_plan(root):
    raw=(root/'plan.json').read_bytes(); plan=strict_json(raw)
    if sha(raw)!=(root/'manifest.sha256').read_text(encoding='ascii').strip() or plan.get('schema')!=STUDY: raise ValueError('plan hash/schema mismatch')
    if (plan.get('model')!=MODEL or plan.get('seed')!=design.SEED or plan.get('budget_nanodollars')!=BUDGET or plan.get('request_cap')!=LIMIT or plan.get('code_sha256')!=hashes() or plan.get('prompt')!=v1.PROMPT or plan.get('price_bands')!=[list(x) for x in v1.BANDS]): raise ValueError('frozen contract changed')
    if sha((root/'labels-private.json').read_bytes())!=plan['labels_sha256']: raise ValueError('target file mismatch')
    cases,targets,meta=load_source(); contexts,excluded=design.cohort(cases)
    if plan['contexts']!=contexts or plan['schedule']!=design.schedule(contexts) or plan['excluded_equal']!=excluded or plan['metadata']!=meta or strict_json((root/'labels-private.json').read_bytes())!=targets: raise ValueError('source or schedule differs from plan')
    return plan

class Ledger(BaseLedger):
    def __init__(self,path,*,preview_hash):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (request_id TEXT PRIMARY KEY,category TEXT NOT NULL,state TEXT NOT NULL,committed INTEGER NOT NULL,input_tokens INTEGER,created_utc TEXT NOT NULL,detail TEXT)')
            wanted={'preview_hash':preview_hash,'budget':str(BUDGET),'model':MODEL,'reservation':str(RESERVATION),'study':STUDY,'request_cap':str(LIMIT)}; old=dict(db.execute('SELECT key,value FROM config'))
            if old and old!=wanted: raise ValueError('ledger mismatch')
            if not old: db.executemany('INSERT INTO config VALUES (?,?)',wanted.items())
    def summary(self):
        result=super().summary();result.update(request_cap=LIMIT,budget_nanodollars=BUDGET,study=STUDY);return result
    def reserve(self,request_id,category='human'):
        if not isinstance(request_id,str) or not request_id or category!='human': raise ValueError('bad request')
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM attempts WHERE request_id=?',(request_id,)).fetchone(): raise ValueError('duplicate request')
            count,total=db.execute('SELECT COUNT(*),COALESCE(SUM(committed),0) FROM attempts').fetchone()
            if count>=LIMIT or total+RESERVATION>BUDGET: raise ValueError('request or budget cap')
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',(request_id,category,'reserved',RESERVATION,None,datetime.now(timezone.utc).isoformat(),None))

async def chain(client,root,ctx,row):
    async def call(stage,q=None,band=None):
        wire_stage=f'price-b{band}' if stage=='price' else stage
        state,options=stage_spec(ctx,stage,q,band); ids=list(options); data=await body(state,stage,options)
        if len(data)>MAX_BODY: raise ValueError('body cap')
        request_id=rid(row,wire_stage)
        try:
            wire=await client.send_body(request_id=request_id,body=data)
            answer=wire['answers']['decision']; raw=answer['probabilities']; dec=decisions(raw,ids,row,stage)
        except BaseException as exc:
            client.halted=True
            folder=root/'wire-evidence'/sha(request_id.encode()); names=('request','response','attempt','timing','failure')
            evidence={name:(sha((folder/f'{name}.json').read_bytes()) if (folder/f'{name}.json').exists() else None) for name in names}
            with sqlite3.connect(root/'cost-ledger.sqlite') as db: attempted=db.execute('SELECT 1 FROM attempts WHERE request_id=?',(request_id,)).fetchone() is not None
            append(root/'technical.jsonl',{'path':row,'stage':stage,'band':band,'request_id':request_id,'constructed_request_sha256':sha(data),'dispatched':attempted,'evidence_sha256':evidence,'error_type':type(exc).__name__})
            raise
        entry={'stage':stage,'band':band,'request_id':rid(row,wire_stage),'request_sha256':wire['_wire']['request_sha256'],'response_sha256':wire['_wire']['response_sha256'],
            'attempt_sha256':wire['_wire']['attempt_sha256'],'timing_sha256':wire['_wire']['timing_sha256'],'candidate_ids':ids,'raw_probabilities':raw,'provider_choice':answer.get('choice'),'decisions':dec}
        append(root/'trace.jsonl',{'path':row,'step':entry})
        return entry,options
    quantity,options=await call('quantity'); qid=quantity['decisions']['cutoff3'].get('selected_id')
    qfail=sum('selected_id' not in quantity['decisions'][arm] for arm in design.ARMS)
    if qid is None: return {'path_status':'QUANTITY_FAILURE','quantity_failures':qfail,'price_failures':3}
    q=selection('quantity',qid,quantity['candidate_ids'],options)
    band,_=await call('band',q); prices={}
    for b in row['band_order']: prices[b],_=await call('price',q,b)
    failures=0
    for arm in design.ARMS:
        bid=band['decisions'][arm].get('selected_id')
        if bid is None or prices[int(bid[1:])]['decisions'][arm].get('selected_id') is None: failures+=1
    return {'path_status':'QUERIED','quantity_failures':qfail,'price_failures':failures}

def partial_progress(root,total=72):
    grouped={}
    trace=root/'trace.jsonl'
    if trace.exists():
        for line in trace.read_text(encoding='utf-8').splitlines():
            row=json.loads(line); path=row['path']; key=(path['person_id'],path['formal_round'],path['condition'],path['repeat']); step=row['step']
            grouped.setdefault(key,{})[f"price-b{step['band']}" if step['stage']=='price' else step['stage']]=step
    technical=root/'technical.jsonl'
    if technical.exists():
        for line in technical.read_text(encoding='utf-8').splitlines():
            row=json.loads(line); path=row['path']; grouped.setdefault((path['person_id'],path['formal_round'],path['condition'],path['repeat']),{})
    queried=qtyfailed=pricefailed=qpredfailed=incomplete=0
    for chain in grouped.values():
        if 'quantity' not in chain: incomplete+=1; continue
        dec=chain['quantity']['decisions']; qpredfailed+=sum('selected_id' not in dec[a] for a in design.ARMS)
        if 'selected_id' not in dec['cutoff3']: qtyfailed+=1; pricefailed+=3; continue
        if len(chain)!=6: incomplete+=1; continue
        queried+=1
        for arm in design.ARMS:
            bid=chain['band']['decisions'][arm].get('selected_id')
            if bid is None or 'selected_id' not in chain['price-'+bid]['decisions'][arm]: pricefailed+=1
    return {'queried_paths':queried,'quantity_failed_paths':qtyfailed,'quantity_failed_predictions':qpredfailed,
        'price_failed_predictions':pricefailed,'incomplete_started_paths':incomplete,'not_observed_paths':total-len(grouped)}


def batches(schedule):
    for repeat in (0,1):
        block=[row for row in schedule if row['repeat']==repeat]
        for i in range(0,len(block),10): yield block[i:i+10]


async def run(root=OUT):
    plan=valid_plan(root); marker=root/'started'
    if marker.exists(): raise FileExistsError('already started; no retry')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False); key=os.environ.get('TYPESAFE_API_KEY')
    if not key: raise ValueError('provider key missing')
    ledger=Ledger(root/'cost-ledger.sqlite',preview_hash=sha((root/'plan.json').read_bytes()))
    fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.write(fd,(sha((root/'plan.json').read_bytes())+'\n').encode()); os.fsync(fd); os.close(fd)
    contexts={(x['person_id'],x['formal_round'],x['condition']):x for x in plan['contexts']}; queried=qty_failed=quantity_predictions_failed=price_failed=0
    try:
        async with JevWireClient(api_key=key,ledger=ledger,evidence_dir=root/'wire-evidence') as client:
            for batch in batches(plan['schedule']):
                results=await asyncio.wait_for(asyncio.gather(*(chain(client,root,contexts[(r['person_id'],r['formal_round'],r['condition'])],r) for r in batch),return_exceptions=True),timeout=600)
                queried+=sum(x.get('path_status')=='QUERIED' for x in results if isinstance(x,dict))
                qty_failed+=sum(x.get('path_status')=='QUANTITY_FAILURE' for x in results if isinstance(x,dict))
                quantity_predictions_failed+=sum(x['quantity_failures'] for x in results if isinstance(x,dict))
                price_failed+=sum(x['price_failures'] for x in results if isinstance(x,dict))
                error=next((x for x in results if isinstance(x,BaseException)),None)
                if error is not None: raise error
    except BaseException as exc:
        progress=partial_progress(root,len(plan['schedule']))
        append(root/'status.jsonl',{'status':'PARTIAL_TECHNICAL',**progress,'physical_requests':ledger.summary()['physical_requests'],'failure_type':type(exc).__name__}); raise
    status='PARTIAL_PREDICTION_FAILURE' if price_failed or quantity_predictions_failed else 'COMPLETE'
    result={'status':status,'queried_paths':queried,'quantity_failed_paths':qty_failed,'quantity_failed_predictions':quantity_predictions_failed,'price_failed_predictions':price_failed,'physical_requests':ledger.summary()['physical_requests']}; append(root/'status.jsonl',result); return result

def replay(root=OUT):
    plan=valid_plan(root)
    if (root/'started').read_text(encoding='ascii').strip()!=sha((root/'plan.json').read_bytes()): raise ValueError('started marker mismatch')
    statuses=[json.loads(line) for line in (root/'status.jsonl').read_text(encoding='utf-8').splitlines()]
    if not statuses or statuses[-1]['status'] not in ('COMPLETE','PARTIAL_PREDICTION_FAILURE','PARTIAL_TECHNICAL'): raise ValueError('unknown or missing status')
    status=statuses[-1]; contexts={(x['person_id'],x['formal_round'],x['condition']):x for x in plan['contexts']}; scheduled={(x['person_id'],x['formal_round'],x['condition'],x['repeat']):x for x in plan['schedule']}
    trace_path=root/'trace.jsonl'; rows=[json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()] if trace_path.exists() else []; chains={}; seen=set()
    with sqlite3.connect(root/'cost-ledger.sqlite') as db:
        ledger={row[0]:row[1:] for row in db.execute('SELECT request_id,state,committed,input_tokens FROM attempts')}
        config=dict(db.execute('SELECT key,value FROM config'))
    expected_config={'preview_hash':sha((root/'plan.json').read_bytes()),'budget':str(BUDGET),'model':MODEL,'reservation':str(RESERVATION),'study':STUDY,'request_cap':str(LIMIT)}
    if config!=expected_config: raise ValueError('ledger configuration mismatch')
    if status.get('physical_requests',len(ledger))!=len(ledger): raise ValueError('ledger physical count mismatch')
    if status['status']!='PARTIAL_TECHNICAL' and len(ledger)!=len(rows): raise ValueError('ledger attempt count mismatch')
    for trace in rows:
        row=trace['path']; key=(row['person_id'],row['formal_round'],row['condition'],row['repeat']); step=trace['step']; stage=step['stage']; band=step['band']
        if scheduled.get(key)!=row or stage not in STAGES or (stage=='price' and band not in range(4)) or (stage!='price' and band is not None): raise ValueError('trace identity mismatch')
        node=(key,stage,band)
        if node in seen: raise ValueError('duplicate node')
        seen.add(node); chain=chains.setdefault(key,{}); ctx=contexts[key[:3]]
        q=None
        if stage!='quantity':
            if 'quantity' not in chain: raise ValueError('missing quantity predecessor')
            q=chain['quantity']['q']
            if q is None: raise ValueError('continued after quantity failure')
        if stage=='price' and 'band' not in chain: raise ValueError('missing band predecessor')
        state,options=stage_spec(ctx,stage,q,band); ids=list(options); wire_stage=f'price-b{band}' if stage=='price' else stage
        expected_id=rid(row,wire_stage)
        if step['request_id']!=expected_id or step['candidate_ids']!=ids: raise ValueError('request identity or candidate mismatch')
        folder=root/'wire-evidence'/sha(expected_id.encode()); req=(folder/'request.json').read_bytes(); resp=(folder/'response.json').read_bytes()
        if sha(req)!=step['request_sha256'] or sha(resp)!=step['response_sha256']: raise ValueError('wire hash mismatch')
        attempt_bytes=(folder/'attempt.json').read_bytes(); timing_bytes=(folder/'timing.json').read_bytes()
        if sha(attempt_bytes)!=step['attempt_sha256'] or sha(timing_bytes)!=step['timing_sha256']: raise ValueError('attempt or timing hash mismatch')
        attempt=strict_json(attempt_bytes); timing=strict_json(timing_bytes)
        if attempt.get('request_id')!=expected_id or attempt.get('request_sha256')!=sha(req) or attempt.get('body_bytes')!=len(req) or attempt.get('method')!='POST' or attempt.get('url')!='https://api.typesafe.ai/v1/systemone' or attempt.get('reserved_bytes')!=MAX_BODY or attempt.get('reserved_nanodollars')!=RESERVATION or attempt.get('long_context') is not False or attempt.get('compression') is not False or attempt.get('retry_count')!=0: raise ValueError('attempt metadata mismatch')
        if timing.get('status_code')!=200 or not all(isinstance(timing.get(k),(float,int)) and math.isfinite(timing[k]) for k in ('started_monotonic','finished_monotonic','elapsed_seconds')) or timing['finished_monotonic']<timing['started_monotonic'] or timing['elapsed_seconds']<0: raise ValueError('timing metadata mismatch')
        request=strict_json(req); response=strict_json(resp); question=request.get('questions',{}).get('decision',{}); answer=response.get('answers',{}).get('decision',{})
        usage=response.get('usage',{}); it=usage.get('input_tokens'); ot=usage.get('output_tokens')
        if type(it)is not int or not 0<=it<=MAX_INPUT_TOKENS or type(ot)is not int or ot<0 or ledger.get(expected_id)!=('settled',it*NANODOLLARS_PER_TOKEN,it): raise ValueError('usage/ledger mismatch')
        if set(request)!={'model','state','questions'} or set(request['questions'])!={'decision'} or set(question)!={'type','instructions','criteria'} or question['type']!='choice': raise ValueError('wire request shape mismatch')
        if request.get('model')!=MODEL or response.get('model')!=MODEL or request.get('state')!=state or question.get('criteria')!=options or list(question.get('criteria',{}))!=ids or question.get('instructions')!=v1._instructions(stage): raise ValueError('wire request mismatch')
        raw=answer.get('probabilities')
        if raw!=step['raw_probabilities'] or answer.get('choice')!=step['provider_choice'] or decisions(raw,ids,row,stage)!=step['decisions']: raise ValueError('response or decoder mismatch')
        keyname=f'price-b{band}' if stage=='price' else stage
        entry={'step':step,'request':request}
        if stage=='quantity':
            sid=step['decisions']['cutoff3'].get('selected_id'); entry['q']=selection(stage,sid,ids,options) if sid else None
        chain[keyname]=entry
        chain.setdefault('_sequence',[]).append(keyname)
    technical_path=root/'technical.jsonl'; technical=[json.loads(line) for line in technical_path.read_text(encoding='utf-8').splitlines()] if technical_path.exists() else []
    technical_ids=set(); trace_ids={r['step']['request_id'] for r in rows}
    for event in technical:
        row=event['path']; key=(row['person_id'],row['formal_round'],row['condition'],row['repeat']); stage=event['stage']; band=event['band']
        if scheduled.get(key)!=row or stage not in STAGES or (stage=='price' and band not in range(4)) or (stage!='price' and band is not None): raise ValueError('technical event identity mismatch')
        reqid=rid(row,f'price-b{band}' if stage=='price' else stage)
        if reqid!=event['request_id'] or reqid in technical_ids or reqid in trace_ids: raise ValueError('duplicate technical event identity')
        technical_ids.add(reqid); folder=root/'wire-evidence'/sha(reqid.encode()); evidence=event['evidence_sha256']
        if set(evidence)!={'request','response','attempt','timing','failure'}: raise ValueError('technical evidence shape mismatch')
        for name in evidence:
            path=folder/f'{name}.json'; actual=sha(path.read_bytes()) if path.exists() else None
            if actual!=evidence[name]: raise ValueError('technical evidence hash mismatch')
        if event['dispatched']!=(reqid in ledger): raise ValueError('technical dispatch ledger mismatch')
        if reqid not in ledger:
            if any(evidence.values()): raise ValueError('undispatched event has wire files')
            continue
        state,committed,tokens=ledger[reqid]
        if state in ('unknown','reserved'):
            if committed!=RESERVATION or tokens is not None or not evidence['failure']: raise ValueError('unknown cost reservation or failure evidence mismatch')
        elif state=='settled':
            if type(tokens)is not int or not 0<=tokens<=MAX_INPUT_TOKENS or committed!=tokens*NANODOLLARS_PER_TOKEN: raise ValueError('technical settlement mismatch')
        else: raise ValueError('unknown ledger state')
        if not evidence['request'] or not evidence['attempt'] or evidence['request']!=event['constructed_request_sha256']: raise ValueError('technical durable request missing or differs')
        request=strict_json((folder/'request.json').read_bytes()); attempt=strict_json((folder/'attempt.json').read_bytes()); question=request.get('questions',{}).get('decision',{})
        q=None
        if stage!='quantity':
            if key not in chains or 'quantity' not in chains[key] or chains[key]['quantity']['q'] is None: raise ValueError('technical event missing quantity predecessor')
            q=chains[key]['quantity']['q']
        if stage=='price' and (key not in chains or 'band' not in chains[key]): raise ValueError('technical event missing band predecessor')
        expected_state,options=stage_spec(contexts[key[:3]],stage,q,band)
        if set(request)!={'model','state','questions'} or set(request['questions'])!={'decision'} or set(question)!={'type','instructions','criteria'} or question.get('type')!='choice' or request.get('model')!=MODEL or request.get('state')!=expected_state or question.get('instructions')!=v1._instructions(stage) or question.get('criteria')!=options or list(question['criteria'])!=list(options): raise ValueError('technical request mismatch')
        if attempt.get('request_id')!=reqid or attempt.get('request_sha256')!=evidence['request'] or attempt.get('body_bytes')!=(folder/'request.json').stat().st_size or attempt.get('reserved_nanodollars')!=RESERVATION or attempt.get('reserved_bytes')!=MAX_BODY or attempt.get('method')!='POST' or attempt.get('url')!='https://api.typesafe.ai/v1/systemone' or attempt.get('retry_count')!=0 or attempt.get('long_context') is not False or attempt.get('compression') is not False: raise ValueError('technical attempt mismatch')
        if evidence['timing']:
            timing=strict_json((folder/'timing.json').read_bytes())
            if not all(isinstance(timing.get(k),(float,int)) and math.isfinite(timing[k]) for k in ('started_monotonic','finished_monotonic','elapsed_seconds')) or timing['finished_monotonic']<timing['started_monotonic'] or timing['elapsed_seconds']<0: raise ValueError('technical timing mismatch')
        if state=='settled':
            if not evidence['response'] or not evidence['timing']: raise ValueError('settled technical response missing')
            response=strict_json((folder/'response.json').read_bytes()); usage=response.get('usage',{})
            if response.get('model')!=MODEL or usage.get('input_tokens')!=tokens or type(usage.get('output_tokens')) is not int: raise ValueError('technical usage mismatch')
    if status['status']!='PARTIAL_TECHNICAL' and technical: raise ValueError('technical evidence in terminal prediction status')
    if set(ledger)!=trace_ids|{rid for rid in technical_ids if rid in ledger}: raise ValueError('unaccounted ledger attempt')
    queried=failed=quantity_fail=price_fail=0; predictions=[]
    for key,chain in chains.items():
        if 'quantity' not in chain: raise ValueError('missing quantity')
        sequence=chain.pop('_sequence'); expected=['quantity','band']+[f'price-b{b}' for b in scheduled[key]['band_order']]
        if sequence!=expected[:len(sequence)]: raise ValueError('branch order mismatch')
        q=chain['quantity']['q']; ctx=contexts[key[:3]]
        quantity_fail+=sum('selected_id' not in chain['quantity']['step']['decisions'][arm] for arm in design.ARMS)
        if q is None:
            if len(chain)!=1: raise ValueError('continued after quantity failure')
            failed+=1; price_fail+=3
            qdec=chain['quantity']['step']['decisions']; quantities={arm:(int(qdec[arm]['selected_id'][1:]) if 'selected_id' in qdec[arm] else None) for arm in design.ARMS}
            predictions.append({'person_id':key[0],'formal_round':key[1],'condition':key[2],'repeat':key[3],'side':ctx['side'],'stratum':ctx['stratum'],'quantities':quantities,'conditioned_quantity':None,'prices':{arm:None for arm in design.ARMS},'price_failure_reason':{arm:'upstream_quantity' for arm in design.ARMS},'request_hashes':{'quantity':chain['quantity']['step']['request_sha256']},'probabilities':{'quantity':chain['quantity']['step']['raw_probabilities']}})
            continue
        if len(chain)!=6:
            if status['status']!='PARTIAL_TECHNICAL': raise ValueError('missing complete branch')
            continue
        queried+=1; bdec=chain['band']['step']['decisions']; prices={}; reasons={}
        for arm in design.ARMS:
            bid=bdec[arm].get('selected_id')
            if bid is None: prices[arm]=None; reasons[arm]='band_decoder'; price_fail+=1; continue
            pstep=chain['price-'+bid]['step']; pid=pstep['decisions'][arm].get('selected_id')
            if pid is None: prices[arm]=None; reasons[arm]='selected_price_decoder'; price_fail+=1; continue
            prices[arm]=float(pid[1:])*2.5; reasons[arm]=None
        qdec=chain['quantity']['step']['decisions']; quantities={arm:(int(d['selected_id'][1:]) if 'selected_id' in d else None) for arm,d in qdec.items() if arm in design.ARMS}
        predictions.append({'person_id':key[0],'formal_round':key[1],'condition':key[2],'repeat':key[3],'side':ctx['side'],'stratum':ctx['stratum'],'quantities':quantities,'conditioned_quantity':q,'prices':prices,'price_failure_reason':reasons,
            'request_hashes':{node:x['step']['request_sha256'] for node,x in chain.items()},'probabilities':{node:x['step']['raw_probabilities'] for node,x in chain.items()}})
    if status['status']!='PARTIAL_TECHNICAL':
        if len(chains)!=len(plan['schedule']) or queried+failed!=len(plan['schedule']) or status['queried_paths']!=queried or status['quantity_failed_paths']!=failed or status['quantity_failed_predictions']!=quantity_fail or status['price_failed_predictions']!=price_fail or status['physical_requests']!=len(rows): raise ValueError('terminal path coverage mismatch')
        if (status['status']=='COMPLETE')!=(price_fail==0 and quantity_fail==0): raise ValueError('terminal success status mismatch')
    return plan,predictions,status
