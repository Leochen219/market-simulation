"""Versioned top3_cutoff05 Jev grid probe. Preview, run, replay and report only."""
from __future__ import annotations
import argparse, asyncio, base64, hashlib, json, os, statistics, sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from capclaims.grid_sampling import decode, SCHEMA as DECODER_SCHEMA
from capclaims.behavioral_memory.transport import CostLedger as BaseLedger, JevWireClient, MAX_BODY, MODEL, RESERVATION, serialize_body, strict_json
from capclaims.behavioral_memory.memory import compile_behavioral_memory
from capclaims.behavioral_memory.design import build_prompt
from tools.probes import jev_grid_probability as v1

OUT=ROOT/'artifacts'/'participant-data'/'jev-grid-probability-v2-2026-10-27'
STUDY='jev_grid_probability_v2_20261027'; PLAN_SCHEMA='jev-grid-probability-v2'; SEED='grid-prob-20261027-v2'
MAX_REQUESTS=60; BUDGET=200_000_000; CUTOFF=.05; TOLERANCE=1e-12
PROMPT=v1.PROMPT

def _canonical(value): return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _sha(data): return hashlib.sha256(data).hexdigest()
def _append(path:Path,value:dict):
    with path.open('ab') as stream: stream.write(_canonical(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())
def _instructions(stage): return v1._instructions(stage)
async def _body(state,stage,criteria): return await serialize_body(state=state,instructions=_instructions(stage),criteria=criteria)
def grid_criteria(values,kind):
    if kind not in {'q','p'}: raise ValueError('numeric grid criterion kind must be q or p')
    return {f'{kind}{value}':(str(value) if kind=='q' else f'{v1.cents(value)/100:.2f}') for value in values}

def _source_code_paths():
    return [Path(__file__),ROOT/'src/capclaims/grid_sampling.py',ROOT/'tools/probes/jev_grid_probability.py',
        ROOT/'src/capclaims/backends/jev.py',ROOT/'src/capclaims/behavioral_memory/memory.py',ROOT/'src/capclaims/behavioral_memory/design.py',
        ROOT/'src/capclaims/behavioral_memory/transport.py',ROOT/'src/capclaims/behavioral_memory/source.py',ROOT/'src/capclaims/behavioral_memory/projection.py',
        ROOT/'src/capclaims/conditional_behavior/transport.py',ROOT/'src/capclaims/conditional_behavior/source.py',ROOT/'src/capclaims/conditional_behavior/protocol.py',ROOT/'src/capclaims/conditional_behavior/prompts.py',ROOT/'src/capclaims/personalization_xlsx.py']

def _hashes(): return {str(path.relative_to(ROOT)):_sha(path.read_bytes()) for path in _source_code_paths()}

async def _preview_body(state,stage,criteria):
    body=await _body(state,stage,criteria)
    if len(body)>MAX_BODY: raise ValueError('serialized request exceeds 12,000-byte cap')
    return body

def freeze(cases:list[dict],metadata:dict,output:Path=OUT)->dict:
    chosen=v1.select_cases(cases); contexts=[]; max_bytes=0
    for case in chosen:
        side='BUY' if case['current_state']['D']>case['current_state']['Q'] else 'SELL'
        base=v1.make_state(case,side)
        contexts.append({'person_id':case['person_id'],'stratum':'shortage' if side=='BUY' else 'surplus','side':side,'base_state':base})
        qvalues=v1.quantity_grid(side,case['current_state']['Q'])
        qcrit=grid_criteria(qvalues,'q'); maxq=990
        candidate_bands={f'b{i}':f'{lo*2.5:.2f}-{hi*2.5:.2f} dollars' for i,(lo,hi) in enumerate(v1.BANDS)}
        fixtures=[('quantity',base,qcrit)]
        for band_index,(lo,hi) in enumerate(v1.BANDS):
            qstate=dict(base,own_qty_units=maxq,model_chosen_quantity=maxq)
            fixtures.append(('band',qstate,candidate_bands))
            pstate=dict(qstate,model_chosen_band=band_index)
            fixtures.append(('price',pstate,grid_criteria(v1.price_grid(lo,hi),'p')))
        for stage,state,criteria in fixtures:
            body=asyncio.run(_preview_body(state,stage,criteria)); max_bytes=max(max_bytes,len(body))
    if output.exists(): raise FileExistsError('v2 output exists; no reset or resume')
    plan={'schema':PLAN_SCHEMA,'status':'FROZEN','decoder':DECODER_SCHEMA,'cutoff':CUTOFF,'cutoff_tolerance':TOLERANCE,'top_k':3,
        'seed':SEED,'model':MODEL,'people_n':10,'repeat_blocks':2,'calls_per_chain':3,'planned_requests':60,
        'request_cap':MAX_REQUESTS,'budget_nanodollars':BUDGET,'max_body_bytes':max_bytes,
        'metadata':metadata,'prompt':PROMPT,'contexts':contexts,'price_bands':[list(x) for x in v1.BANDS],'code_sha256':_hashes()}
    output.mkdir(parents=True)
    raw=_canonical(plan)+b'\n'; (output/'plan.json').write_bytes(raw); (output/'manifest.sha256').write_text(_sha(raw)+'\n',encoding='ascii')
    return {'status':'FROZEN','schema':PLAN_SCHEMA,'plan_sha256':_sha(raw),'contexts':10,'request_cap':60,'max_body_bytes':max_bytes}

def _valid_plan(root:Path)->dict:
    raw=(root/'plan.json').read_bytes(); plan=json.loads(raw); digest=(root/'manifest.sha256').read_text(encoding='ascii').strip()
    if _sha(raw)!=digest: raise ValueError('v2 plan digest mismatch')
    if (plan.get('schema')!=PLAN_SCHEMA or plan.get('status')!='FROZEN' or plan.get('decoder')!=DECODER_SCHEMA
            or plan.get('cutoff')!=CUTOFF or plan.get('cutoff_tolerance')!=TOLERANCE or plan.get('top_k')!=3
            or plan.get('seed')!=SEED or plan.get('model')!=MODEL or plan.get('budget_nanodollars')!=BUDGET
            or plan.get('request_cap')!=MAX_REQUESTS or plan.get('people_n')!=10 or plan.get('repeat_blocks')!=2
            or plan.get('calls_per_chain')!=3 or plan.get('planned_requests')!=60 or len(plan.get('contexts',[]))!=10):
        raise ValueError('unsupported or incomplete v2 plan')
    if plan.get('code_sha256')!=_hashes(): raise ValueError('v2 frozen code changed')
    return plan

def _verify_request(request:dict,stage:str,state:dict,options:dict)->None:
    if request.get('state')!=state: raise ValueError('v2 request state replay mismatch')
    question=request.get('questions',{}).get('decision',{})
    if question.get('criteria')!=options or question.get('instructions')!=_instructions(stage):
        raise ValueError('v2 request candidate/prompt replay mismatch')

def _replay_decision(step:dict,raw:dict,*,person:str,repeat:int)->dict:
    stage=step['stage']; ids=step['candidate_ids']; decoded=decode(raw,ids,seed=SEED,person_id=person,repeat=repeat,stage=stage)
    if step.get('decode')!=decoded: raise ValueError('v2 deterministic decoder output was tampered')
    if step.get('selected_id')!=decoded['selected_id']: raise ValueError('v2 deterministic sampled selection was tampered')
    if step.get('raw_sum')!=__import__('math').fsum(raw.values()): raise ValueError('v2 raw sum mismatch')
    argmax=v1.choose(ids,decoded['all_candidate_normalized_probabilities'])
    if argmax!=step.get('argmax_id'): raise ValueError('v2 all-candidate argmax mismatch')
    provider=step.get('provider_choice'); norm=decoded['all_candidate_normalized_probabilities']; maximum=max(norm.values())
    tied=provider in norm and provider!=argmax and norm[provider]==maximum
    strict_nonmax=provider is not None and (provider not in norm or norm[provider]<maximum)
    if step.get('provider_choice_matches_argmax')!=(provider==argmax) or step.get('provider_choice_tie_resolved')!=tied or step.get('provider_choice_strict_nonmax')!=strict_nonmax:
        raise ValueError('v2 provider-choice diagnostic tampered')
    value=step.get('selected_value'); index=step.get('selected_index')
    if stage=='quantity' and (type(value)is not int or decoded['selected_id']!=f'q{value}' or index!=ids.index(decoded['selected_id'])):
        raise ValueError('v2 selected quantity identity mismatch')
    if stage=='band' and (type(index)is not int or decoded['selected_id']!=f'b{index}' or index not in range(4) or value!=decoded['selected_id']):
        raise ValueError('v2 selected band identity mismatch')
    if stage=='price':
        if isinstance(value,bool) or not isinstance(value,(int,float)):
            raise ValueError('v2 selected price value is not numeric')
        tick=int(decoded['selected_id'][1:])
        if decoded['selected_id']!=f'p{round(float(value)*100/250)}' or float(value)!=tick*2.5 or index!=ids.index(decoded['selected_id']):
            raise ValueError('v2 selected price identity/value mismatch')
    return decoded

class V2CostLedger(BaseLedger):
    """Independent atomic v2 budget/request ledger."""
    def __init__(self,path,*,preview_hash,budget_nanodollars=BUDGET):
        if type(budget_nanodollars)is not int or budget_nanodollars!=BUDGET: raise ValueError('fixed v2 budget is USD 0.20')
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (request_id TEXT PRIMARY KEY,category TEXT NOT NULL,state TEXT NOT NULL,committed INTEGER NOT NULL,input_tokens INTEGER,created_utc TEXT NOT NULL,detail TEXT)')
            wanted={'preview_hash':preview_hash,'budget':str(BUDGET),'model':MODEL,'reservation':str(RESERVATION),'study':STUDY,'request_cap':str(MAX_REQUESTS)}
            config=dict(db.execute('SELECT key,value FROM config'))
            if config and config!=wanted: raise ValueError('v2 ledger belongs to another plan or cost contract')
            if not config: db.executemany('INSERT INTO config VALUES (?,?)',wanted.items())
    def reserve(self,request_id,category='human'):
        if not isinstance(request_id,str) or not request_id or category!='human': raise ValueError('invalid v2 request identity')
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM attempts WHERE request_id=?',(request_id,)).fetchone(): raise ValueError('duplicate request forbidden')
            count,committed=db.execute('SELECT COUNT(*),COALESCE(SUM(committed),0) FROM attempts').fetchone()
            budget=int(db.execute("SELECT value FROM config WHERE key='budget'").fetchone()[0])
            if count>=MAX_REQUESTS: raise ValueError('v2 60-request cap reached')
            if committed+RESERVATION>budget: raise ValueError('v2 USD 0.20 budget exhausted before send')
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',(request_id,'human','reserved',RESERVATION,None,datetime.now(timezone.utc).isoformat(),None))
    def summary(self):
        result=super().summary(); result.update({'request_cap':MAX_REQUESTS,'budget_nanodollars':BUDGET,'study':STUDY}); return result

async def _chain(client,root:Path,context:dict,repeat:int)->None:
    person=context['person_id']; side=context['side']; base=context['base_state']; prefix=f'gpr-v2-r{repeat}-{person}'
    qvalues=v1.quantity_grid(side,base['own_q_units'])
    specs=[('quantity',base,grid_criteria(qvalues,'q'))]; selected={}
    band_options={f'b{i}':f'{lo*2.5:.2f}-{hi*2.5:.2f} dollars' for i,(lo,hi) in enumerate(v1.BANDS)}
    for stage_no in range(3):
        if stage_no==0: stage,state,options=specs[0]
        elif stage_no==1:
            quantity=selected['quantity']['selected_value']; state=dict(base,own_qty_units=quantity,model_chosen_quantity=quantity)
            stage,options='band',band_options
        else:
            quantity=selected['quantity']['selected_value']; band=selected['band']['selected_index']
            state=dict(base,own_qty_units=quantity,model_chosen_quantity=quantity,model_chosen_band=band)
            stage,options='price',grid_criteria(v1.price_grid(*v1.BANDS[band]),'p')
        request_id=f'{prefix}-{stage}'; body=await _body(state,stage,options); wire=await client.send_body(request_id=request_id,body=body)
        answer=wire.get('answers',{}).get('decision',{}); raw=answer.get('probabilities'); ids=list(options)
        try:
            decoded=decode(raw,ids,seed=SEED,person_id=person,repeat=repeat,stage=stage)
        except ValueError as exc:
            failed={'stage':stage,'request_id':wire['_wire']['request_id'],'request_sha256':wire['_wire']['request_sha256'],
                'response_sha256':wire['_wire']['response_sha256'],'candidate_ids':ids,'raw_probabilities':raw,
                'provider_choice':answer.get('choice'),'decode_error':str(exc),'conditional_on':{'side':side,**({'quantity':selected['quantity']['selected_value']} if stage_no>=1 else {}),**({'band':selected['band']['selected_index']} if stage_no>=2 else {})}}
            _append(root/'trace.jsonl',{'schema':'grid-probability-v2-trace','record_type':'DECODE_FAILURE','repeat':repeat,'context':person,'step':failed})
            raise
        argmax=v1.choose(ids,decoded['all_candidate_normalized_probabilities']); provider_choice=answer.get('choice')
        if stage=='quantity': value=int(options[decoded['selected_id']]); index=ids.index(decoded['selected_id'])
        elif stage=='band': value=decoded['selected_id']; index=int(value[1:])
        else: value=float(options[decoded['selected_id']]); index=ids.index(decoded['selected_id'])
        full_argmax=max(decoded['all_candidate_normalized_probabilities'],key=lambda key:(decoded['all_candidate_normalized_probabilities'][key],-ids.index(key)))
        choice_tie=provider_choice in decoded['all_candidate_normalized_probabilities'] and provider_choice!=full_argmax and decoded['all_candidate_normalized_probabilities'][provider_choice]==decoded['all_candidate_normalized_probabilities'][full_argmax]
        choice_strict_nonmax=provider_choice is not None and (provider_choice not in decoded['all_candidate_normalized_probabilities'] or decoded['all_candidate_normalized_probabilities'][provider_choice]<decoded['all_candidate_normalized_probabilities'][full_argmax])
        step={'stage':stage,'request_id':wire['_wire']['request_id'],'request_sha256':wire['_wire']['request_sha256'],'response_sha256':wire['_wire']['response_sha256'],
            'candidate_ids':ids,'raw_probabilities':raw,'raw_sum':__import__('math').fsum(raw.values()),'all_candidate_normalized_probabilities':decoded['all_candidate_normalized_probabilities'],
            'normalization':decoded['normalization'],'decode':decoded,'argmax_id':argmax,'provider_choice':provider_choice,
            'selected_id':decoded['selected_id'],
            'provider_choice_matches_argmax':provider_choice==argmax,'provider_choice_tie_resolved':choice_tie,'provider_choice_strict_nonmax':choice_strict_nonmax,
            'selected_value':value,'selected_index':index,'conditional_on':{'side':side,**({'quantity':selected['quantity']['selected_value']} if stage_no>=1 else {}),**({'band':selected['band']['selected_index']} if stage_no>=2 else {})}}
        selected[stage]=step
        _append(root/'trace.jsonl',{'schema':'grid-probability-v2-trace','record_type':'STEP','repeat':repeat,'context':person,'step':step})

def _read_plan_bound(root,plan):
    marker=root/'started'
    if marker.exists() and marker.read_text(encoding='ascii').strip()!=_sha((root/'plan.json').read_bytes()): raise ValueError('started marker plan mismatch')

async def run(root:Path)->dict:
    plan=_valid_plan(root); marker=root/'started'
    if marker.exists(): raise FileExistsError('v2 run already started; no resume/retry')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False); key=os.environ.get('TYPESAFE_API_KEY')
    if not key: raise ValueError('provider key missing; no request sent')
    ledger=V2CostLedger(root/'cost-ledger.sqlite',preview_hash=_sha((root/'plan.json').read_bytes()))
    fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.write(fd,(_sha((root/'plan.json').read_bytes())+'\n').encode()); os.fsync(fd); os.close(fd)
    client=JevWireClient(api_key=key,ledger=ledger,evidence_dir=root/'wire-evidence'); completed=0
    try:
        async with client:
            for repeat in (0,1):
                batch=[_chain(client,root,context,repeat) for context in plan['contexts']]
                outcomes=await asyncio.wait_for(asyncio.gather(*batch,return_exceptions=True),timeout=600)
                failure=next((x for x in outcomes if isinstance(x,BaseException)),None)
                if failure: raise failure
                completed+=len(outcomes)
    except BaseException as exc:
        _append(root/'status.jsonl',{'status':'PARTIAL_FAILURE','completed_chains':completed,'failure_type':type(exc).__name__})
        raise
    _append(root/'status.jsonl',{'status':'COMPLETE','completed_chains':completed,'physical_requests':ledger.summary()['physical_requests']})
    return {'status':'COMPLETE','chains':completed,'physical_requests':ledger.summary()['physical_requests']}

def _valid_trace_id(row,contexts):
    if row.get('schema')!='grid-probability-v2-trace' or row.get('record_type') not in {'STEP','DECODE_FAILURE'}:
        raise ValueError('unknown v2 trace record')
    if row.get('repeat') not in (0,1) or row.get('context') not in contexts:
        raise ValueError('unknown v2 trace coordinates')

def replay(root:Path)->tuple[dict,list[dict]]:
    plan=_valid_plan(root)
    if not (root/'started').is_file(): raise ValueError('v2 run has no durable started marker')
    _read_plan_bound(root,plan)
    statuses=[json.loads(line) for line in (root/'status.jsonl').read_text(encoding='utf-8').splitlines()] if (root/'status.jsonl').exists() else []
    complete=bool(statuses and statuses[-1].get('status')=='COMPLETE')
    rows=[json.loads(line) for line in (root/'trace.jsonl').read_text(encoding='utf-8').splitlines()] if (root/'trace.jsonl').exists() else []
    contexts={x['person_id']:x for x in plan['contexts']}; seen=set(); seen_requests=set(); chains={}; requests={}; records=[]
    for row in rows:
        _valid_trace_id(row,contexts); person=row['context']; repeat=row['repeat']; step=row['step']; stage=step.get('stage')
        if stage not in {'quantity','band','price'}: raise ValueError('unknown v2 stage')
        key=(repeat,person,stage)
        if key in seen: raise ValueError('duplicate v2 stage response')
        seen.add(key)
        rid=step.get('request_id'); expected_id=f'gpr-v2-r{repeat}-{person}-{stage}'
        if rid!=expected_id or rid in seen_requests: raise ValueError('v2 request ID is not unique or canonical')
        seen_requests.add(rid)
        folder=root/'wire-evidence'/hashlib.sha256(rid.encode()).hexdigest()
        request_bytes=(folder/'request.json').read_bytes(); response_bytes=(folder/'response.json').read_bytes()
        if _sha(request_bytes)!=step.get('request_sha256') or _sha(response_bytes)!=step.get('response_sha256'):
            raise ValueError('v2 wire evidence hash mismatch')
        request=strict_json(request_bytes); response=strict_json(response_bytes)
        if request.get('model')!=MODEL or response.get('model')!=MODEL: raise ValueError('unexpected v2 model')
        question=request.get('questions',{}).get('decision',{}); options=question.get('criteria'); ids=step.get('candidate_ids')
        if not isinstance(options,dict) or list(options)!=ids: raise ValueError('v2 request candidate order mismatch')
        answer=response.get('answers',{}).get('decision',{}); raw=answer.get('probabilities')
        if not isinstance(raw,dict) or set(raw)!=set(ids): raise ValueError('v2 response probability keys mismatch')
        if raw!=step.get('raw_probabilities') or answer.get('choice')!=step.get('provider_choice'):
            raise ValueError('v2 trace differs from saved raw response')
        requests[rid]=request
        ctx=contexts[person]; side=ctx['side']; group_key=(repeat,person)
        chain=chains.setdefault(group_key,[]); chain.append((row,request,options,answer))
        if row['record_type']=='DECODE_FAILURE':
            try: decode(raw,ids,seed=SEED,person_id=person,repeat=repeat,stage=stage)
            except ValueError as exc:
                if str(exc)!=step.get('decode_error'): raise ValueError('v2 decoder failure replay mismatch')
            else: raise ValueError('v2 logged decode failure but decoder found support')
            records.append({'record_type':'DECODE_FAILURE','repeat':repeat,'context':person,'stage':stage,'decode_error':step['decode_error'],'candidate_count':len(ids),'raw_probabilities':raw,'request_sha256':step['request_sha256'],'response_sha256':step['response_sha256']})
            continue
        decoded=_replay_decision(step,raw,person=person,repeat=repeat)
        argmax=step['argmax_id']
        if step.get('selected_id')!=decoded['selected_id']: raise ValueError('v2 deterministic sampled selection mismatch')
        value=step.get('selected_value'); index=step.get('selected_index')
        records.append({'record_type':'STEP','repeat':repeat,'context':person,'stage':stage,'side':side,'conditional_on':step.get('conditional_on'),
            'raw_probabilities':raw,'decoder':decoded,'argmax_id':argmax,'provider_choice':answer.get('choice'),'selected_id':decoded['selected_id'],'selected_value':value,'selected_index':index,
            'request_sha256':step['request_sha256'],'response_sha256':step['response_sha256']})
    if len(seen)>MAX_REQUESTS: raise ValueError('v2 trace exceeded 60-request cap')
    if complete and (len(seen)!=60 or statuses[-1].get('physical_requests')!=60 or statuses[-1].get('completed_chains')!=20):
        raise ValueError('v2 complete status requires 60 requests and 20 complete chains')
    for (repeat,person),chain in chains.items():
        chain.sort(key=lambda item:('quantity','band','price').index(item[0]['step']['stage']))
        stages=[item[0]['step']['stage'] for item in chain]
        if stages!=['quantity','band','price'][:len(stages)]: raise ValueError('v2 chain order is invalid')
        failure=chain[-1][0]['record_type']=='DECODE_FAILURE'
        if any(item[0]['record_type']=='DECODE_FAILURE' for item in chain[:-1]): raise ValueError('v2 chain continued after decoder failure')
        if complete and (len(stages)!=3 or failure): raise ValueError('v2 complete chain has a decode failure')
        context=contexts[person]; base=context['base_state']; qty=None; band=None
        for row,request,options,answer in chain:
            step=row['step']; stage=step['stage']
            if stage=='quantity': state=base; expected=v1.criteria(v1.quantity_grid(context['side'],base['own_q_units']),'q')
            elif stage=='band':
                prior=next(item[0]['step'] for item in chain if item[0]['step']['stage']=='quantity' and item[0]['record_type']=='STEP')
                qty=prior['selected_value']; state=dict(base,own_qty_units=qty,model_chosen_quantity=qty)
                expected={f'b{i}':f'{lo*2.5:.2f}-{hi*2.5:.2f} dollars' for i,(lo,hi) in enumerate(v1.BANDS)}
            else:
                prior_q=next(item[0]['step'] for item in chain if item[0]['step']['stage']=='quantity' and item[0]['record_type']=='STEP')
                prior_b=next(item[0]['step'] for item in chain if item[0]['step']['stage']=='band' and item[0]['record_type']=='STEP')
                qty=prior_q['selected_value']; band=prior_b['selected_index']; state=dict(base,own_qty_units=qty,model_chosen_quantity=qty,model_chosen_band=band)
                expected=v1.criteria(v1.price_grid(*v1.BANDS[band]),'p')
            if options!=expected: raise ValueError('trace candidate IDs differ from deterministic stage candidates')
            _verify_request(request,stage,state,expected)
            expected_conditional={'side':context['side'],**({'quantity':qty} if stage in ('band','price') else {}),**({'band':band} if stage=='price' else {})}
            if step.get('conditional_on')!=expected_conditional: raise ValueError('v2 trace conditional chain mismatch')
    if complete and len(chains)!=20: raise ValueError('v2 complete run requires 20 participant/repeat chains')
    return plan,records

def _aggregate(records):
    groups={}
    for row in records:
        if row['record_type']!='STEP': continue
        stage=row['stage']; side=row['side']; band=(row['conditional_on'] or {}).get('band')
        key=('BUYquantity' if stage=='quantity' and side=='BUY' else 'SELLquantity' if stage=='quantity' else f'{side}_{stage}'+(f'_band{band}' if band is not None else ''))
        groups.setdefault(key,[]).append(row)
    output={}
    for key,rows in groups.items():
        raw=[x['raw_probabilities'] for x in rows]
        out=v1.summarize(raw,[len(x['decoder']['candidate_ids']) for x in rows])
        out['decoder']={'schema':DECODER_SCHEMA,'selected_n':len(rows),
            'retained_raw_mass_median':statistics.median(x['decoder']['retained_mass'] for x in rows),
            'cutoff_dropped_mass_median':statistics.median(x['decoder']['cutoff_dropped_mass'] for x in rows),
            'topk_dropped_mass_median':statistics.median(x['decoder']['topk_dropped_mass'] for x in rows),
            'mean_retained_candidates':statistics.mean(len(x['decoder']['retained_ids']) for x in rows)}
        output[key]=out
    return output

def report(root:Path,public_output:Path,private_output:Path)->dict:
    plan,records=replay(root)
    public={'schema':'jev-grid-probability-v2-report','decoder':DECODER_SCHEMA,'cutoff':CUTOFF,'tolerance':TOLERANCE,'top_k':3,
        'interpretation':'Aggregate conditional raw Choice probabilities. The v2 decoder draws from the at-most-three candidates passing the 5% raw-probability cutoff; this is a discrete behavioral sampling rule, not a calibrated probability or accuracy claim. Price distributions are conditional on supplied side, sampled quantity, and sampled band.','completed_steps':sum(x['record_type']=='STEP' for x in records),'decode_failures':sum(x['record_type']=='DECODE_FAILURE' for x in records),'conditional_groups':_aggregate(records)}
    private={'schema':'jev-grid-probability-v2-private-report','plan_sha256':_sha((root/'plan.json').read_bytes()),'records':records}
    public_output.write_bytes(_canonical(public)+b'\n'); private_output.write_bytes(_canonical(private)+b'\n')
    return {'status':'REPLAY_VALID','steps':public['completed_steps'],'decode_failures':public['decode_failures']}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=['preview','run','report','replay'])
    parser.add_argument('--plan-dir',type=Path,default=OUT); parser.add_argument('--public-output',type=Path); parser.add_argument('--private-output',type=Path)
    args=parser.parse_args(argv)
    if args.command=='preview':
        from capclaims.behavioral_memory.source import build_authorized_cases
        cases,labels,metadata=build_authorized_cases(); del labels
        result=freeze(cases,metadata,args.plan_dir)
    elif args.command=='run': result=asyncio.run(run(args.plan_dir))
    else:
        if not args.public_output or not args.private_output: parser.error('report/replay require both output paths')
        result=report(args.plan_dir,args.public_output,args.private_output)
    print(json.dumps(result,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
