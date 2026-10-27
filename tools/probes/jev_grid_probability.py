"""Standalone offline-frozen Jev probability shape diagnostic; never predicts direction."""
from __future__ import annotations
import argparse, asyncio, hashlib, json, os, statistics
from pathlib import Path
from typing import Any
from capclaims.backends.jev import normalize_jev_probabilities
from capclaims.behavioral_memory.memory import compile_behavioral_memory
from capclaims.behavioral_memory.design import build_prompt
from capclaims.behavioral_memory.transport import CostLedger as BaseLedger, JevWireClient, serialize_body, strict_json, MAX_BODY, MODEL, RESERVATION

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts'/'participant-data'/'jev-grid-probability-v1-2026-10-27'
SEED='grid-prob-20261027'; BUDGET=200_000_000; MAX_REQUESTS=60
BANDS=((1,100),(101,200),(201,300),(301,399))
PROMPT=('You are an observer predicting a human participant\'s numeric choice, given the supplied side and one order. '
        'This is a behavioral forecast, not advice or profit maximization. Use the supplied historical facts and recorded rules unchanged; '
        'The current diagnostic grid differs from the historical integer-dollar interface. One diagnostic order is being described; '
        'although the original interface allows up to five paper orders, this numeric probe conditions on one. '
        'Predict the human numeric choice from the supplied complete candidate values. Do not infer the participant\'s direction; '
        'the side is supplied for this diagnostic. Do not use target orders, target final profit, or answer keys.')

def quantity_grid(side:str,Q:int)->list[int]:
    if side=='BUY': return list(range(10,1000,10))
    if side=='SELL': return list(range(10,min(990,(Q//10)*10)+1,10))
    raise ValueError('side must be BUY or SELL')

def price_grid(lo:int,hi:int)->list[int]:
    if type(lo)is not int or type(hi)is not int or not 1<=lo<=hi<=399: raise ValueError('bad price band')
    return list(range(lo,hi+1))
def cents(tick:int)->int:
    if type(tick)is not int or not 1<=tick<=399: raise ValueError('bad price tick')
    return tick*250

def choose(ids:list[str], probs:dict[str,float])->str:
    return min(ids,key=lambda x:(-probs[x],ids.index(x)))
def normalized(raw:dict[str,Any], ids:list[str]): return normalize_jev_probabilities(raw,ids)
def summarize(distributions:list[dict[str,float]], candidate_n:list[int])->dict[str,Any]:
    metrics=[]
    for p,n in zip(distributions,candidate_n):
        ordered=sorted(p.items(),key=lambda x:(-x[1],x[0]))
        vals=[x[1] for x in ordered]
        metrics.append({'n':n,'empty_support':sum(v==0 for v in p.values()),'top1':vals[0] if vals else 0,
          'top3_sum':sum(vals[:3]),'third_rank':vals[2] if len(vals)>2 else None,
          'max_ties':sum(v==max(vals) for v in vals) if vals else 0,
          **{f'mass_ge_{int(t*100)}':sum(v for v in vals if v>=t) for t in (.03,.05,.10)},
          **{f'count_ge_{int(t*100)}':sum(v>=t for v in vals) for t in (.03,.05,.10)},
          **{f'cutoff_top3_mass_{int(t*100)}':sum(v for v in vals[:3] if v>=t) for t in (.03,.05,.10)},
          **{f'empty_after_cutoff_{int(t*100)}':int(not any(v>=t for v in vals)) for t in (.03,.05,.10)},
          'retained_top3_mass':sum(vals[:3])})
    out={}
    for key in ('top1','top3_sum','third_rank','max_ties','mass_ge_3','mass_ge_5','mass_ge_10','count_ge_3','count_ge_5','count_ge_10','cutoff_top3_mass_3','cutoff_top3_mass_5','cutoff_top3_mass_10','empty_after_cutoff_3','empty_after_cutoff_5','empty_after_cutoff_10','retained_top3_mass','empty_support'):
        vals=[m[key] for m in metrics if m[key] is not None]
        out[key]={'n':len(vals),'median':statistics.median(vals) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None}
    return {'n':len(metrics),'summary':out,'per_distribution':metrics,'interpretation':'conditional distributions; offline descriptive statistics only'}

def select_cases(cases:list[dict])->list[dict]:
    def key(c): return hashlib.sha256((SEED+':'+c['person_id']).encode()).hexdigest()
    shortage=sorted((c for c in cases if c['current_state']['D']>c['current_state']['Q']),key=key)[:5]
    surplus=sorted((c for c in cases if c['current_state']['Q']>c['current_state']['D'] and c['current_state']['Q']>=10),key=key)[:5]
    if len(shortage)!=5 or len(surplus)!=5 or len({c['person_id'] for c in shortage+surplus})!=10: raise ValueError('need five unique cases in each stratum')
    return shortage+surplus

def make_state(case:dict, side:str, quantity:int|None=None)->dict:
    memory,_=compile_behavioral_memory(case['history'],case['current_state'],12)
    # Preserve only the M1 observation compiler output; replace its count-first
    # language with the frozen numeric diagnostic instructions.
    m1=json.loads(build_prompt(arm='M1',current_state=case['current_state'],history=memory['history'],memory=memory))
    state=m1['observation']
    for key in ('numeric_prefix','decimal_place','numeric_prefixes','decimal_places'):
        state.pop(key,None)
    if isinstance(state.get('public_instructions'),dict):
        state['public_instructions']=dict(state['public_instructions'])
        state['public_instructions']['trading']=('Diagnostic grid only: quantity choices are complete multiples of 10; '
            'price choices are complete $2.50 ticks. The model predicts numeric choices conditional on the supplied side; '
            'this probe does not predict direction.')
    state.pop('decision_interface',None); state.pop('protocol_interface',None)
    if isinstance(state.get('submitted_order_ledger'),dict):
        state['submitted_order_ledger']=dict(state['submitted_order_ledger'])
        state['submitted_order_ledger'].update({'submitted_order_count':0,'remaining_order_slots':1,'submitted_buy_quantity_units':0,'submitted_sell_quantity_units':0,'remaining_sell_submission_capacity_units':state.get('own_q_units',0)})
    state.pop('own_qty_units',None)
    state.update({'selected_order_count':1,'remaining_planned_order_slots':1,'slot':0,'partial_orders':[], 'own_side':side})
    if quantity is not None: state['model_chosen_quantity']=quantity; state['own_qty_units']=quantity
    return state

def criteria(values:list[int], kind:str)->dict[str,str]:
    return {f'{kind}{v}': (str(v) if kind=='q' else f'{cents(v)/100:.2f}') for v in values}

class GridLedger(BaseLedger):
    """Study-specific contract; inherited SQLite transaction is atomic."""
    def __init__(self,path,*,preview_hash,budget_nanodollars=BUDGET):
        if type(budget_nanodollars)is not int or budget_nanodollars!=BUDGET: raise ValueError('grid probe budget is fixed at USD 0.20')
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (request_id TEXT PRIMARY KEY,category TEXT NOT NULL,state TEXT NOT NULL,committed INTEGER NOT NULL,input_tokens INTEGER,created_utc TEXT NOT NULL,detail TEXT)')
            wanted={'preview_hash':preview_hash,'budget':str(BUDGET),'model':MODEL,'reservation':str(RESERVATION),'study':'jev_grid_probability_v1_20261027','request_cap':str(MAX_REQUESTS)}
            config=dict(db.execute('SELECT key,value FROM config'))
            if config and config!=wanted: raise ValueError('ledger belongs to a different frozen plan or study contract')
            if not config: db.executemany('INSERT INTO config VALUES (?,?)',wanted.items())
    def reserve(self,request_id:str,category:str='human')->None:
        if category!='human' or not request_id: raise ValueError('invalid attempt')
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM attempts WHERE request_id=?',(request_id,)).fetchone(): raise ValueError('duplicate attempt forbidden')
            count,committed=db.execute('SELECT COUNT(*),COALESCE(SUM(committed),0) FROM attempts').fetchone()
            budget=int(db.execute("SELECT value FROM config WHERE key='budget'").fetchone()[0])
            if count>=MAX_REQUESTS: raise ValueError('60-request cap reached')
            if committed+RESERVATION>budget: raise ValueError('USD 0.20 budget exhausted')
            from datetime import datetime,timezone
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',(request_id,'human','reserved',RESERVATION,None,datetime.now(timezone.utc).isoformat(),None))
    def summary(self):
        result=super().summary(); result.update({'request_cap':MAX_REQUESTS,'budget_nanodollars':BUDGET}); return result

def canonical(obj): return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def sha(data): return hashlib.sha256(data).hexdigest()
def _instructions(step:str)->str:
    if step=='quantity': return PROMPT+' Choose exactly one complete quantity value.'
    if step=='band': return PROMPT+' Choose exactly one disjoint price-band label for the already chosen quantity.'
    return PROMPT+' Choose exactly one complete dollar price within the selected band; this is conditional on the chosen side, quantity, and band.'

async def _body(state,step,options): return await serialize_body(state=state,instructions=_instructions(step),criteria=options)

def freeze(cases:list[dict],metadata:dict, output:Path=OUT)->dict:
    selected=select_cases(cases); plans=[]
    previews={}; max_body=0
    for case in selected:
        side='BUY' if case['current_state']['D']>case['current_state']['Q'] else 'SELL'
        mem,_=compile_behavioral_memory(case['history'],case['current_state'],12)
        context={'person_id':case['person_id'],'stratum':'shortage' if side=='BUY' else 'surplus','side':side,'base_state':make_state(case,side)}
        plans.append(context)
        qvals=quantity_grid(side,case['current_state']['Q'])
        qcrit=criteria(qvals,'q')
        # all four band follow-ups use a conservative 990-quantity state so the
        # certificate covers the longest legal quantity rendering.
        max_quantity=990
        qstate=make_state(case,side)
        max_body=max(max_body,len(asyncio.run(_body(qstate,'quantity',qcrit))))
        for i,band in enumerate(BANDS):
            bstate=make_state(case,side,max_quantity)
            options={f'b{j}':f'{lo*250/100:.2f}-{hi*250/100:.2f} dollars' for j,(lo,hi) in enumerate(BANDS)}
            max_body=max(max_body,len(asyncio.run(_body(bstate,'band',options))))
            prices=criteria(price_grid(*band),'p')
            pstate=make_state(case,side,max_quantity); pstate['model_chosen_band']=i
            max_body=max(max_body,len(asyncio.run(_body(pstate,'price',prices))))
        if max_body>MAX_BODY: raise ValueError('frozen request exceeds 12,000-byte cap')
    manifest={'schema':'jev-grid-probability-v1','status':'FROZEN','seed':SEED,'model':MODEL,'people_n':10,'repeat_blocks':2,'chains_per_context':1,'calls_per_chain':3,'planned_requests':60,'request_cap':MAX_REQUESTS,'budget_nanodollars':BUDGET,'metadata':metadata,'prompt':PROMPT,'contexts':plans,'thresholds':[.03,.05,.10], 'price_bands':[list(x) for x in BANDS],'max_serialized_body_bytes':max_body, 'code_sha256':{str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in [Path(__file__),ROOT/'src/capclaims/backends/jev.py',ROOT/'src/capclaims/behavioral_memory/memory.py',ROOT/'src/capclaims/behavioral_memory/design.py',ROOT/'src/capclaims/behavioral_memory/transport.py',ROOT/'src/capclaims/behavioral_memory/source.py',ROOT/'src/capclaims/behavioral_memory/projection.py',ROOT/'src/capclaims/conditional_behavior/transport.py',ROOT/'src/capclaims/conditional_behavior/source.py',ROOT/'src/capclaims/conditional_behavior/protocol.py',ROOT/'src/capclaims/conditional_behavior/prompts.py']}}
    if output.exists(): raise FileExistsError('preview already exists; no reset/resume')
    output.mkdir(parents=True); (output/'plan.json').write_bytes(canonical(manifest)+b'\n')
    (output/'manifest.sha256').write_text(sha((output/'plan.json').read_bytes())+'\n',encoding='ascii')
    # Private examples are exact SDK serialization, stored only in the participant directory.
    examples=[]
    context=plans[0]; q=990
    fixtures=[('quantity',context['base_state'],criteria(quantity_grid(context['side'],context['base_state'].get('own_q_units',context['base_state'].get('Q',10))),'q'))]
    for bi,band in enumerate(BANDS):
        fixtures.append(('band',dict(context['base_state'],model_chosen_quantity=q,own_qty_units=q),{f'b{i}':f'{lo*2.5:.2f}-{hi*2.5:.2f} dollars' for i,(lo,hi) in enumerate(BANDS)}))
        fixtures.append(('price',dict(context['base_state'],model_chosen_quantity=q,own_qty_units=q,model_chosen_band=bi),criteria(price_grid(*band),'p')))
    for step,state,opts in fixtures:
        body=asyncio.run(_body(state,step,opts)); examples.append({'step':step,'body_base64':__import__('base64').b64encode(body).decode(),'body_sha256':sha(body),'body_bytes':len(body)})
    (output/'wire-examples.json').write_bytes(canonical(examples)+b'\n')
    return {'status':'FROZEN','plan_sha256':sha((output/'plan.json').read_bytes()),'contexts':10,'max_body_bytes':max_body}

def _valid_plan(root:Path)->dict:
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8')); raw=(root/'plan.json').read_bytes()
    digest=(root/'manifest.sha256').read_text(encoding='ascii').strip()
    if sha(raw)!=digest: raise ValueError('plan digest mismatch')
    if plan.get('schema')!='jev-grid-probability-v1' or plan.get('status')!='FROZEN' or len(plan.get('contexts',[]))!=10: raise ValueError('unsupported plan')
    for filename,digest in plan['code_sha256'].items():
        if sha((ROOT/filename).read_bytes())!=digest: raise ValueError('frozen code changed: '+filename)
    return plan

def _append(path:Path,row:dict):
    with path.open('ab') as f: f.write(canonical(row)+b'\n'); f.flush(); os.fsync(f.fileno())

async def _chain(client,root:Path,context:dict,repeat:int,chain:int)->dict:
    base=context['base_state']; side=context['side']; prefix=f"gpr-{repeat}-{context['person_id']}-{chain}"
    results=[]
    # build_observation exposes initial own quantity under own_q_units; fallback Q.
    qvals=quantity_grid(side,base.get('own_q_units',base.get('Q',0)))
    steps=[('quantity',base,criteria(qvals,'q'))]
    for step_no in range(3):
        if step_no==0: stage,state,opts=steps[0]
        elif step_no==1:
            q=results[0]['selected_value']; state=dict(base,model_chosen_quantity=q,own_qty_units=q)
            opts={f'b{i}':f'{lo*2.5:.2f}-{hi*2.5:.2f} dollars' for i,(lo,hi) in enumerate(BANDS)}; stage='band'
        else:
            bi=results[1]['selected_index']; q=results[0]['selected_value']; state=dict(base,model_chosen_quantity=q,own_qty_units=q,model_chosen_band=bi)
            opts=criteria(price_grid(*BANDS[bi]),'p'); stage='price'
        body=await _body(state,stage,opts)
        wire=await client.send_body(request_id=f'{prefix}-{stage}',body=body)
        answer=wire.get('answers',{}).get('decision',{})
        raw=answer.get('probabilities'); ids=list(opts)
        probs,meta=normalized(raw,ids)
        selected_id=choose(ids,probs); provider_choice=answer.get('choice')
        if stage=='quantity': value=int(opts[selected_id]); selected_index=ids.index(selected_id)
        elif stage=='band': value=selected_id; selected_index=int(selected_id[1:])
        else: value=float(opts[selected_id]); selected_index=ids.index(selected_id)
        tie_resolved=provider_choice in probs and probs[provider_choice]==max(probs.values()) and provider_choice!=selected_id
        strict_nonmax=provider_choice is not None and (provider_choice not in probs or probs[provider_choice]<max(probs.values()))
        step={'stage':stage,'request_id':wire['_wire']['request_id'],'request_sha256':wire['_wire']['request_sha256'],'response_sha256':wire['_wire']['response_sha256'],'candidate_ids':ids,'raw_probabilities':raw,'normalized_probabilities':probs,'normalization':meta,'raw_sum':sum(raw.values()),'provider_choice':provider_choice,'argmax_id':selected_id,'choice_argmax_conflict':strict_nonmax,'choice_tie_resolved':tie_resolved,'choice_strict_nonmax':strict_nonmax,'argmax_tied':sum(v==max(probs.values()) for v in probs.values())>1,'selected_value':value,'selected_index':selected_index,'conditional_on':{'side':side,**({'quantity':results[0]['selected_value']} if step_no>=1 else {}),**({'band':results[1]['selected_index']} if step_no>=2 else {})}}
        _append(root/'trace.jsonl',{'repeat':repeat,'context':context['person_id'],'chain':chain,'step':step}); results.append(step)
    return {'steps':results}

async def run(root:Path)->dict:
    plan=_valid_plan(root); marker=root/'started'
    if marker.exists(): raise FileExistsError('run already started; automatic resume/retry is forbidden')
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env',override=False)
    key=os.environ.get('TYPESAFE_API_KEY')
    if not key: raise ValueError('provider key missing; no request sent')
    ledger=GridLedger(root/'cost-ledger.sqlite',preview_hash=sha((root/'plan.json').read_bytes()),budget_nanodollars=BUDGET)
    fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); os.write(fd,(sha((root/'plan.json').read_bytes())+'\n').encode()); os.fsync(fd); os.close(fd)
    # Wire client owns an exclusive evidence directory and halts on any error.
    client=JevWireClient(api_key=key,ledger=ledger,evidence_dir=root/'wire-evidence')
    completed=0
    try:
        async with client:
            for repeat in (0,1):
                # Ten independent contexts may overlap, while every dependent
                # quantity -> band -> price chain stays strictly sequential.
                batch=[_chain(client,root,context,repeat,0) for context in plan['contexts']]
                results=await asyncio.wait_for(asyncio.gather(*batch,return_exceptions=True),timeout=600)
                failure=next((result for result in results if isinstance(result,BaseException)),None)
                if failure is not None: raise failure
                completed+=len(results)
    except BaseException as exc:
        _append(root/'status.jsonl',{'status':'PARTIAL_FAILURE','completed_chains':completed,'failure_type':type(exc).__name__})
        raise
    _append(root/'status.jsonl',{'status':'COMPLETE','completed_chains':completed,'physical_requests':ledger.summary()['physical_requests']})
    return {'status':'COMPLETE','chains':completed,'physical_requests':ledger.summary()['physical_requests']}

def report(root:Path,output:Path|None=None)->dict:
    plan=_valid_plan(root)
    if (root/'started').exists() and (root/'started').read_text(encoding='ascii').strip()!=sha((root/'plan.json').read_bytes()): raise ValueError('started marker does not bind to current plan')
    traces=[json.loads(line) for line in (root/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    status_rows=[json.loads(line) for line in (root/'status.jsonl').read_text(encoding='utf-8').splitlines()] if (root/'status.jsonl').exists() else []
    is_complete=bool(status_rows and status_rows[-1].get('status')=='COMPLETE')
    grouped={}; requests_by_id={}; contexts={c['person_id']:c for c in plan['contexts']}; seen=set()
    chain_steps={}
    for row in traces:
        step=row['step']; raw=step['raw_probabilities']; ids=step['candidate_ids']; valid,meta=normalized(raw,ids)
        if valid!=step['normalized_probabilities']: raise ValueError('normalization replay mismatch')
        if meta!=step['normalization'] or sum(raw.values())!=step['raw_sum']: raise ValueError('raw probability audit mismatch')
        if choose(ids,valid)!=step['argmax_id']: raise ValueError('argmax replay mismatch')
        provider=step['provider_choice']; maximum=max(valid.values())
        expected_tie=provider in valid and provider!=step['argmax_id'] and valid[provider]==maximum
        expected_nonmax=provider is not None and (provider not in valid or valid[provider]<maximum)
        if step['choice_tie_resolved']!=expected_tie or step['choice_strict_nonmax']!=expected_nonmax or step['choice_argmax_conflict']!=expected_nonmax:
            raise ValueError('provider choice comparison replay mismatch')
        selected=step['argmax_id']
        if step['stage']=='quantity' and selected!=f"q{int(step['selected_value'])}": raise ValueError('quantity candidate/value mismatch')
        if step['stage']=='band' and selected!=f"b{step['selected_index']}": raise ValueError('band candidate/value mismatch')
        if step['stage']=='price' and selected!=f"p{round(float(step['selected_value'])*100/250)}": raise ValueError('price candidate/value mismatch')
        chain_steps.setdefault((row['repeat'],row['context'],row['chain']),[]).append(step)
        group=(step['stage'],step['conditional_on']['side'],step['conditional_on'].get('band'))
        grouped.setdefault(group,[]).append((raw,len(ids),step))
        req=step['request_id']; evidence=root/'wire-evidence'/hashlib.sha256(req.encode()).hexdigest()
        request_bytes=(evidence/'request.json').read_bytes(); response_bytes=(evidence/'response.json').read_bytes()
        if sha(request_bytes)!=step['request_sha256'] or sha(response_bytes)!=step['response_sha256']:
            raise ValueError('wire evidence hash mismatch')
        response=strict_json(response_bytes); answer=response.get('answers',{}).get('decision',{})
        if answer.get('probabilities')!=raw or answer.get('choice')!=step['provider_choice']:
            raise ValueError('trace differs from preserved provider response')
        request=strict_json(request_bytes)
        if request.get('model')!=MODEL or not isinstance(request.get('state'),dict):
            raise ValueError('saved wire request is malformed')
        if list(request.get('questions',{}).get('decision',{}).get('criteria',{}))!=ids:
            raise ValueError('trace candidate IDs differ from saved request')
        requests_by_id[req]=request
        unique=(row['repeat'],row['context'],row['chain'],step['stage'])
        if unique in seen or row['repeat'] not in (0,1) or row['context'] not in contexts or row['chain']!=0:
            raise ValueError('duplicate or unknown trace identity')
        seen.add(unique)
    for chain_key,steps in chain_steps.items():
        steps.sort(key=lambda x:('quantity','band','price').index(x['stage']))
        if [s['stage'] for s in steps]!=['quantity','band','price'][:len(steps)] or (is_complete and len(steps)!=3): raise ValueError('incomplete or out-of-order chain')
        if (len(steps)>1 and steps[1]['conditional_on'].get('quantity')!=steps[0]['selected_value']) or (len(steps)>2 and (steps[2]['conditional_on'].get('quantity')!=steps[0]['selected_value'] or steps[2]['conditional_on'].get('band')!=steps[1]['selected_index'])):
            raise ValueError('conditional chain replay mismatch')
        # Replay the state and all Choice candidate labels that were derived
        # from the previous response in this exact sequential chain.
        first=steps[0]; context=contexts[chain_key[1]]; base=context['base_state']
        expected_states=[base]
        expected_options=[criteria(quantity_grid(context['side'],base.get('own_q_units',base.get('Q',0))),'q')]
        if len(steps)>1:
            expected_states.append(dict(base,model_chosen_quantity=first['selected_value'],own_qty_units=first['selected_value']))
            expected_options.append({f'b{i}':f'{lo*2.5:.2f}-{hi*2.5:.2f} dollars' for i,(lo,hi) in enumerate(BANDS)})
        if len(steps)>2:
            expected_states.append(dict(base,model_chosen_quantity=first['selected_value'],own_qty_units=first['selected_value'],model_chosen_band=steps[1]['selected_index']))
            expected_options.append(criteria(price_grid(*BANDS[steps[1]['selected_index']]),'p'))
        for step,state,options in zip(steps,expected_states,expected_options):
            question=requests_by_id[step['request_id']]['questions']['decision']
            if requests_by_id[step['request_id']]['state']!=state or question['criteria']!=options or question['instructions']!=_instructions(step['stage']):
                raise ValueError('saved request state/options do not replay from chain')
    if len(seen)!=len(traces) or len(traces)>MAX_REQUESTS: raise ValueError('trace request count exceeds frozen cap')
    if is_complete and len(traces)!=60: raise ValueError('complete status requires all 60 response records')
    stats={}
    for (stage,side,band),rows in grouped.items():
        key=('BUYquantity' if stage=='quantity' and side=='BUY' else 'SELLquantity' if stage=='quantity' else f'{side}_{stage}'+(f'_band{band}' if band is not None else ''))
        s=summarize([v for v,n,_ in rows],[n for v,n,_ in rows])
        s['raw_sum_median']=statistics.median(x['raw_sum'] for _,_,x in rows)
        s['raw_sum_min']=min(x['raw_sum'] for _,_,x in rows); s['raw_sum_max']=max(x['raw_sum'] for _,_,x in rows)
        s['rounding_normalized_n']=sum(x['normalization']['rounding_normalized'] for _,_,x in rows)
        s['source_precision_decimals']={str(d):sum(x['normalization']['source_precision_decimals']==d for _,_,x in rows) for d in (None,2)}
        s['choice_argmax_conflicts']=sum(x['choice_argmax_conflict'] for _,_,x in rows)
        s['choice_tie_resolved_n']=sum(x['choice_tie_resolved'] for _,_,x in rows)
        s['choice_strict_nonmax_n']=sum(x['choice_strict_nonmax'] for _,_,x in rows)
        s['argmax_tied_n']=sum(x['argmax_tied'] for _,_,x in rows)
        stats[key]=s
    result={'schema':'jev-grid-probability-report-v1','interpretation':'Conditional distributions, not joint probabilities; price probabilities are conditional on side, selected quantity, and selected band. Supplied-side diagnostics do not predict direction. No accuracy or calibration claim. Thresholds use raw provider probabilities; normalization is used for legality and argmax only.','completed_response_steps':len(traces),'conditional_groups':stats,'status_lines':status_rows}
    if output: output.write_bytes(canonical(result)+b'\n')
    return result

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['preview','run','report']); p.add_argument('--plan-dir',type=Path,default=OUT); p.add_argument('--output',type=Path); a=p.parse_args(argv)
    if a.command=='preview':
        from capclaims.behavioral_memory.source import build_authorized_cases
        cases,labels,meta=build_authorized_cases(); del labels
        print(json.dumps(freeze(cases,meta,a.plan_dir),sort_keys=True)); return 0
    if a.command=='run': result=asyncio.run(run(a.plan_dir))
    else: result=report(a.plan_dir,a.output)
    print(json.dumps(result,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
