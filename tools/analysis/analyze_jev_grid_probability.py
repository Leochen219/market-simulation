"""Strict offline replay and reporting for the frozen Jev grid probability probe.

The run trace is append-only evidence. This analyzer never repairs or rewrites it.
"""
from __future__ import annotations
import argparse, hashlib, json, statistics
from pathlib import Path
from typing import Any
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))

from capclaims.behavioral_memory.transport import strict_json
from tools.probes import jev_grid_probability as frozen


def _sha(data:bytes)->str: return hashlib.sha256(data).hexdigest()
def _json(value:Any)->bytes: return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def _verify_probability_evidence(step:dict, answer:dict)->dict:
    raw=answer.get('probabilities')
    if not isinstance(raw,dict): raise ValueError('raw response probability map missing')
    ids=step.get('candidate_ids')
    if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or set(raw)!=set(ids):
        raise ValueError('invalid candidate identity or probability keys')
    if raw!=step.get('raw_probabilities') or answer.get('choice')!=step.get('provider_choice'):
        raise ValueError('trace probability or provider choice differs from preserved response')
    raw_sum=sum(raw.values())
    if raw_sum!=step.get('raw_sum'):
        raise ValueError('trace raw sum differs from original response-order sum')
    normalized,meta=frozen.normalized(raw,ids)
    if normalized!=step.get('normalized_probabilities') or meta!=step.get('normalization'):
        raise ValueError('strict probability normalization replay mismatch')
    return {'raw':raw,'raw_sum':raw_sum,'normalized':normalized,'normalization':meta}

def _verify_request(request:dict, stage:str, expected_state:dict, expected_options:dict)->None:
    if request.get('state')!=expected_state:
        raise ValueError('saved request state fails chain replay')
    question=request.get('questions',{}).get('decision',{})
    if question.get('criteria')!=expected_options or question.get('instructions')!=frozen._instructions(stage):
        raise ValueError('saved request candidates or prompt fail chain replay')

def _boundary_distribution(probabilities:dict[str,float],threshold:float,tolerance:float=1e-12)->dict:
    values=list(probabilities.values())
    exact=[v for v in values if v>=threshold]
    tolerant=[v for v in values if v+tolerance>=threshold]
    boundary=[v for v in values if v<threshold and threshold-v<=tolerance]
    top3=sorted(values,reverse=True)[:3]
    return {'threshold':threshold,'tolerance':tolerance,'boundary_candidate_n':len(boundary),
        'strict_support_count':len(exact),'tolerant_support_count':len(tolerant),
        'strict_support_mass':sum(exact),'tolerant_support_mass':sum(tolerant),
        'strict_empty':not exact,'tolerant_empty':not tolerant,
        'strict_top3_retained_mass':sum(v for v in top3 if v>=threshold),
        'tolerant_top3_retained_mass':sum(v for v in top3 if v+tolerance>=threshold)}


def _replay(run_dir:Path)->tuple[dict,list[dict]]:
    plan=frozen._valid_plan(run_dir)
    plan_bytes=(run_dir/'plan.json').read_bytes()
    if (run_dir/'started').read_text(encoding='ascii').strip()!=_sha(plan_bytes):
        raise ValueError('started marker does not bind to the frozen plan')
    status=[json.loads(line) for line in (run_dir/'status.jsonl').read_text(encoding='utf-8').splitlines()]
    if not status or status[-1].get('status')!='COMPLETE' or status[-1].get('physical_requests')!=60:
        raise ValueError('run is not a complete 60-request study')
    traces=[json.loads(line) for line in (run_dir/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(traces)!=60: raise ValueError('complete run must have exactly 60 response records')
    contexts={context['person_id']:context for context in plan['contexts']}
    chains={}; requests={}; records=[]; seen=set()
    for row in traces:
        repeat,person,chain=row.get('repeat'),row.get('context'),row.get('chain')
        step=row.get('step',{})
        stage=step.get('stage')
        if repeat not in (0,1) or person not in contexts or chain!=0 or stage not in ('quantity','band','price'):
            raise ValueError('invalid or unknown trace identity')
        unique=(repeat,person,chain,stage)
        if unique in seen: raise ValueError('duplicate response identity')
        seen.add(unique)
        request_id=step.get('request_id')
        folder=run_dir/'wire-evidence'/hashlib.sha256(request_id.encode('utf-8')).hexdigest()
        req_bytes=(folder/'request.json').read_bytes(); resp_bytes=(folder/'response.json').read_bytes()
        if _sha(req_bytes)!=step.get('request_sha256') or _sha(resp_bytes)!=step.get('response_sha256'):
            raise ValueError('preserved wire body hash mismatch')
        request=strict_json(req_bytes); response=strict_json(resp_bytes)
        if request.get('model')!=frozen.MODEL or response.get('model')!=frozen.MODEL:
            raise ValueError('unexpected model in preserved wire evidence')
        answer=response.get('answers',{}).get('decision',{})
        raw=response.get('answers',{}).get('decision',{}).get('probabilities')
        ids=step.get('candidate_ids')
        if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)):
            raise ValueError('invalid frozen candidate identity list')
        criteria=request.get('questions',{}).get('decision',{}).get('criteria')
        if not isinstance(criteria,dict) or list(criteria)!=ids or set(raw)!=set(ids):
            raise ValueError('saved candidate order differs from request or response')
        # Preserve service insertion order here. Trace JSON uses sorted keys, which
        # can change the final floating-point bit of Python's left-to-right sum.
        probability=_verify_probability_evidence(step,answer)
        raw=probability['raw']; raw_sum=probability['raw_sum']; normalized=probability['normalized']; meta=probability['normalization']
        argmax=frozen.choose(ids,normalized)
        provider_choice=answer.get('choice')
        if argmax!=step.get('argmax_id'): raise ValueError('argmax replay mismatch')
        maximum=max(normalized.values())
        tie_resolved=(provider_choice in normalized and provider_choice!=argmax and normalized[provider_choice]==maximum)
        strict_nonmax=(provider_choice is not None and (provider_choice not in normalized or normalized[provider_choice]<maximum))
        if step.get('choice_tie_resolved')!=tie_resolved or step.get('choice_strict_nonmax')!=strict_nonmax or step.get('choice_argmax_conflict')!=strict_nonmax:
            raise ValueError('provider choice comparison replay mismatch')
        value=step.get('selected_value'); index=step.get('selected_index')
        if stage=='quantity' and (argmax!=f'q{int(value)}' or type(value)is not int): raise ValueError('quantity choice identity mismatch')
        if stage=='band' and (argmax!=f'b{index}' or type(index)is not int): raise ValueError('band choice identity mismatch')
        if stage=='price' and (argmax!=f'p{round(float(value)*100/250)}'): raise ValueError('price choice identity mismatch')
        requests[request_id]=request
        chains.setdefault((repeat,person,chain),[]).append(step)
        records.append({'repeat':repeat,'context':person,'chain':chain,'stage':stage,
            'candidate_ids':ids,'raw_probabilities':raw,'raw_sum':raw_sum,
            'raw_sum_trace':step['raw_sum'],'normalization':meta,'provider_choice':provider_choice,
            'argmax_id':argmax,'selected_value':value,'selected_index':index,
            'choice_tie_resolved':tie_resolved,'choice_strict_nonmax':strict_nonmax,
            'request_sha256':step['request_sha256'],'response_sha256':step['response_sha256'],
            'conditional_on':step.get('conditional_on')})
    if len(seen)!=60: raise ValueError('response identities are incomplete')
    for key,steps in chains.items():
        steps.sort(key=lambda step:('quantity','band','price').index(step['stage']))
        if [s['stage'] for s in steps]!=['quantity','band','price']: raise ValueError('chain stages missing or out of order')
        quantity,band,price=steps
        if band.get('conditional_on',{}).get('quantity')!=quantity['selected_value']:
            raise ValueError('band request does not follow selected quantity')
        if price.get('conditional_on',{}).get('quantity')!=quantity['selected_value'] or price.get('conditional_on',{}).get('band')!=band['selected_index']:
            raise ValueError('price request does not follow selected quantity and band')
        context=contexts[key[1]]; base=context['base_state']
        expected_states=[base,
            dict(base,model_chosen_quantity=quantity['selected_value'],own_qty_units=quantity['selected_value']),
            dict(base,model_chosen_quantity=quantity['selected_value'],own_qty_units=quantity['selected_value'],model_chosen_band=band['selected_index'])]
        expected_options=[frozen.criteria(frozen.quantity_grid(context['side'],base['own_q_units']),'q'),
            {f'b{i}':f'{lo*2.5:.2f}-{hi*2.5:.2f} dollars' for i,(lo,hi) in enumerate(frozen.BANDS)},
            frozen.criteria(frozen.price_grid(*frozen.BANDS[band['selected_index']]),'p')]
        for step,state,options in zip(steps,expected_states,expected_options):
            req=requests[step['request_id']]
            _verify_request(req,step['stage'],state,options)
    return plan,records


def _aggregate(records:list[dict])->dict:
    groups={}
    for record in records:
        side=record['conditional_on']['side']; stage=record['stage']; band=record['conditional_on'].get('band')
        key=('BUYquantity' if stage=='quantity' and side=='BUY' else 'SELLquantity' if stage=='quantity' else f'{side}_{stage}'+(f'_band{band}' if band is not None else ''))
        groups.setdefault(key,[]).append(record)
    result={}
    for key,rows in groups.items():
        stat=frozen.summarize([row['raw_probabilities'] for row in rows],[len(row['candidate_ids']) for row in rows])
        raw_sums=[r['raw_sum'] for r in rows]
        stat['raw_sum']={'n':len(raw_sums),'median':statistics.median(raw_sums),'min':min(raw_sums),'max':max(raw_sums)}
        stat['rounding_normalized_n']=sum(r['normalization']['rounding_normalized'] for r in rows)
        stat['source_precision_decimals']={str(d):sum(r['normalization']['source_precision_decimals']==d for r in rows) for d in (None,2)}
        stat['provider_choice_tie_resolved_n']=sum(r['choice_tie_resolved'] for r in rows)
        stat['provider_choice_strict_nonmax_n']=sum(r['choice_strict_nonmax'] for r in rows)
        boundary={f'{threshold:.2f}':[_boundary_distribution(r['raw_probabilities'],threshold) for r in rows]
                  for threshold in (.03,.05,.10)}
        stat['boundary_sensitivity']={key:{
            'distribution_n':len(values),
            'distributions_with_boundary_candidate':sum(v['boundary_candidate_n']>0 for v in values),
            'boundary_candidate_n':sum(v['boundary_candidate_n'] for v in values),
            'strict_empty_n':sum(v['strict_empty'] for v in values),
            'tolerant_empty_n':sum(v['tolerant_empty'] for v in values),
            'strict_top3_retained_mass_median':statistics.median(v['strict_top3_retained_mass'] for v in values),
            'tolerant_top3_retained_mass_median':statistics.median(v['tolerant_top3_retained_mass'] for v in values),
            'tolerance':1e-12} for key,values in boundary.items()}
        result[key]=stat
    return result


def analyze(run_dir:str|Path, public_output:str|Path, private_output:str|Path)->dict:
    root=Path(run_dir); plan,records=_replay(root)
    public={'schema':'jev-grid-probability-public-report-v1','interpretation':'Aggregate conditional Choice distributions only. Price distributions are conditional on supplied side, chosen quantity, and selected band. The supplied side is not a direction prediction. No accuracy or calibration claim. Threshold counts and masses use original raw probabilities; normalization is only for validation and argmax. This offline report applies the report-only compatibility fix: raw sums are replayed in original saved response key order, rather than trace key order.','model':plan['model'],'participant_context_count':10,'repeat_blocks':2,'response_step_count':len(records),'conditional_groups':_aggregate(records)}
    private={'schema':'jev-grid-probability-private-report-v1','compatibility_note':'Report-only repair for floating-point addition order: trace JSON sorts probability-map keys, while Python summation is order-sensitive in the last bit. Raw sums are recomputed from preserved response JSON insertion order and compared exactly with trace values. No trace, plan, manifest, raw probability, or wire record was rewritten.','plan_sha256':_sha((root/'plan.json').read_bytes()),'records':records}
    Path(public_output).write_bytes(_json(public)+b'\n'); Path(private_output).write_bytes(_json(private)+b'\n')
    return {'status':'REPLAY_VALID','response_steps':len(records),'public_output':str(public_output),'private_output':str(private_output)}


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--public-output',required=True,type=Path)
    parser.add_argument('--private-output',required=True,type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(analyze(args.run_dir,args.public_output,args.private_output),sort_keys=True))
    return 0

if __name__=='__main__': raise SystemExit(main())
