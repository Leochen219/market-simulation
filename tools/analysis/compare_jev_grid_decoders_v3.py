"""Replay and compare two complete Jev grid studies without exposing contexts."""
from __future__ import annotations
import argparse, json, statistics, sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
from tools.analysis.analyze_jev_grid_probability import _replay as replay_v1
from tools.probes import jev_grid_probability as v1, jev_grid_probability_v3 as v3
from capclaims.grid_sampling import decode as decode5
from capclaims.grid_sampling_v3 import decode as decode3

STAGES=('quantity','band','price')

def _matched(p1,p2):
    for k in ('model','prompt','price_bands'):
        if p1.get(k)!=p2.get(k): raise ValueError(f'plans differ in {k}')
    c1={x['person_id']:x for x in p1['contexts']}; c2={x['person_id']:x for x in p2['contexts']}
    if len(c1)!=10 or c1!=c2 or Counter(x['side'] for x in c1.values())!={'BUY':5,'SELL':5}:
        raise ValueError('context identities, states or supplied sides differ')
    for stage in STAGES:
        if v1._instructions(stage)!=v3._instructions(stage): raise ValueError('stage prompts differ')
    for context in c1.values():
        grid=v1.quantity_grid(context['side'],context['base_state']['own_q_units'])
        if v1.criteria(grid,'q')!=v3.grid_criteria(grid,'q'): raise ValueError('quantity candidates differ')
    for lo,hi in v1.BANDS:
        grid=v1.price_grid(lo,hi)
        if v1.criteria(grid,'p')!=v3.grid_criteria(grid,'p'): raise ValueError('price candidates differ')
    return c1

def _chains(records,contexts,version):
    result={}
    for r in records:
        if version==2 and r.get('record_type') not in ('STEP','DECODE_FAILURE'): raise ValueError('unknown v3 record type')
        if version==2 and r.get('record_type')=='DECODE_FAILURE' and r.get('stage')!='price': raise ValueError('non-price decode failure')
        key=(r['context'],r['repeat']); stage=r['stage']
        if key[0] not in contexts or key[1] not in (0,1) or stage not in STAGES: raise ValueError('unknown chain identity')
        chain=result.setdefault(key,{})
        if stage in chain: raise ValueError('duplicate chain stage')
        chain[stage]=r
    if set(result)!={(c,r) for c in contexts for r in (0,1)} or any(set(c)!=set(STAGES) for c in result.values()):
        raise ValueError('incomplete chains')
    return result

def _stats(values):
    s=sorted(values); n=len(s); midpoint=n//2
    # Median of each half, excluding the middle observation for odd n.
    q1=statistics.median(s[:midpoint]) if n>1 else s[0]
    q3=statistics.median(s[(n+1)//2:]) if n>1 else s[0]
    return {'n':n,'histogram':{str(k):v for k,v in sorted(Counter(values).items())},'mean':statistics.mean(values),
        'median':statistics.median(values),'min':s[0],'max':s[-1],'range':s[-1]-s[0],
        'q1':q1,'q3':q3,'iqr':q3-q1,'std_population':statistics.pstdev(values),'unique_values':len(set(values))}

def _group(chains,contexts,side):
    keys=sorted(k for k in chains if side is None or contexts[k[0]]['side']==side)
    quantity=[chains[k]['quantity']['selected_value'] for k in keys]
    price=[chains[k]['price']['selected_value'] for k in keys if chains[k]['price'].get('record_type')!='DECODE_FAILURE']
    bands=[chains[k]['band']['selected_index'] for k in keys]
    people=sorted({k[0] for k in keys}); repeat={}
    for stage in ('quantity','price'):
        pairs=[(chains[(p,0)][stage],chains[(p,1)][stage]) for p in people if all(chains[(p,r)][stage].get('record_type')!='DECODE_FAILURE' for r in (0,1))]
        a=[x['selected_value'] for x,y in pairs]; b=[y['selected_value'] for x,y in pairs]
        repeat[stage]={'context_n':len(pairs),'different_n':sum(x!=y for x,y in zip(a,b)),
            'absolute_difference':_stats([abs(x-y) for x,y in zip(a,b)])}
    return {'quantity':_stats(quantity),'price':_stats(price),
        'price_10_dollar_multiple_n':sum(abs(x/10-round(x/10))<1e-9 for x in price),
        'price_10_dollar_multiple_share':sum(abs(x/10-round(x/10))<1e-9 for x in price)/len(price),
        'price_gt_250_n':sum(x>250 for x in price),'price_denominator':len(price),
        'band_counts':{str(i):bands.count(i) for i in range(4)},'repeat_diversity':repeat}

def _paired(a,b,contexts,side):
    keys=sorted(k for k in a if side is None or contexts[k[0]]['side']==side)
    out={}
    for stage in ('quantity','price'):
        stage_keys=[k for k in keys if b[k][stage].get('record_type')!='DECODE_FAILURE']
        diffs=[b[k][stage]['selected_value']-a[k][stage]['selected_value'] for k in stage_keys]
        out[stage]={'n':len(stage_keys),'increased_n':sum(x>0 for x in diffs),'decreased_n':sum(x<0 for x in diffs),'unchanged_n':sum(x==0 for x in diffs),'difference':_stats(diffs)}
    out['band']={'n':len(keys),'changed_n':sum(a[k]['band']['selected_index']!=b[k]['band']['selected_index'] for k in keys)}
    return out

def compare(p1,r1,p2,r2):
    contexts=_matched(p1,p2); a=_chains(r1,contexts,1); b=_chains(r2,contexts,2)
    failures=[(key,c['price']) for key,c in b.items() if c['price'].get('record_type')=='DECODE_FAILURE']
    groups={}
    for label,side in (('all',None),('BUY','BUY'),('SELL','SELL')):
        old=_group(a,contexts,side); new=_group(b,contexts,side)
        matched_keys={k for k in b if b[k]['price'].get('record_type')!='DECODE_FAILURE'}
        matched_price=[a[k]['price']['selected_value'] for k in sorted(matched_keys) if side is None or contexts[k[0]]['side']==side]
        old_matched_price=_stats(matched_price)
        marginal={}
        for stage in ('quantity','price'):
            baseline=old_matched_price if stage=='price' else old[stage]
            h1=baseline['histogram']; h2=new[stage]['histogram']
            marginal[stage]={'n_per_run':baseline['n'],'histogram_count_v3_minus_v1':{k:h2.get(k,0)-h1.get(k,0) for k in sorted(set(h1)|set(h2),key=float)},'mean_v3_minus_v1':new[stage]['mean']-baseline['mean']}
        marginal['band_count_v3_minus_v1']={k:new['band_counts'][k]-old['band_counts'][k] for k in old['band_counts']}
        groups[label]={'v1_argmax':old,'v3_top3_cutoff3':new,'paired_v3_minus_v1':_paired(a,b,contexts,side),'marginal_v3_minus_v1':marginal,'v1_price_matched_to_v3_successes':old_matched_price}
    direct={}
    for stage in STAGES:
        rows=[c[stage] for c in b.values() if c[stage].get('record_type')=='STEP']
        direct[stage]={'n':len(rows),'sample_differs_from_same_response_argmax_n':sum(x['selected_id']!=x['argmax_id'] for x in rows)}
    return {'schema':'jev-grid-decoder-comparison-v3','status':'PARTIAL_FAILURE' if failures else 'COMPLETE','context_count':10,'planned_chains_per_run':20,'responses_per_run':60,
        'price_decode_failures':{'n':len(failures),'denominator':20,'share':len(failures)/20,'details':[{'candidate_n':r['candidate_count'],'max_raw_probability':max(r['raw_probabilities'].values())} for _,r in failures]},'complete_v3_chains':20-len(failures),
        'groups':groups,'v3_same_response_decoder_deviation':direct,
        'interpretation':'Supplied BUY/SELL sides are not predictions. Both runs replayed strictly; price summaries for v3 include successful decodes only. The runs match on model, prompt, context, base state, candidate rules and price bands. Later requests are conditional on the earlier selections in each run. Same-response argmax at each v3 stage is not a full argmax counterfactual chain. Runs use separate calls and time blocks; differences combine decoder and response variation. Ten contexts and two repeats provide no significance or human-likeness conclusion.',
        'quartiles':'Median of halves, excluding the median for odd n; population standard deviation; histogram counts have n as denominator.'}

def threshold_sensitivity(rows):
    """Apply both frozen decoders to already replay-validated trace responses."""
    results={stage:{'response_n':0,'failure_5_n':0,'failure_3_n':0,'both_selected_n':0,'support_diff_n':0,'selected_diff_n':0} for stage in STAGES}
    for row in rows:
        step=row['step']; stage=step['stage']; result=results[stage]; result['response_n']+=1
        outputs=[]
        for decoder,label in ((decode5,'5'),(decode3,'3')):
            try: outputs.append(decoder(step['raw_probabilities'],step['candidate_ids'],seed=v3.SEED,person_id=row['context'],repeat=row['repeat'],stage=stage))
            except ValueError: outputs.append(None); result[f'failure_{label}_n']+=1
        a,b=outputs
        if a is not None and b is not None:
            result['both_selected_n']+=1
            result['support_diff_n']+=set(a['retained_ids'])!=set(b['retained_ids'])
            result['selected_diff_n']+=a['selected_id']!=b['selected_id']
    return results


def analyze(old,new):
    p1,r1=replay_v1(old); p2,r2=v3.replay(new)
    statuses=[json.loads(x) for x in (new/'status.jsonl').read_text(encoding='utf-8').splitlines()]
    if not statuses or statuses[-1].get('status') not in ('COMPLETE','PARTIAL_FAILURE') or len(r2)!=60: raise ValueError('v3 run incomplete')
    result=compare(p1,r1,p2,r2)
    old5=new.parent/'jev-grid-probability-v2-2026-10-27'
    from tools.probes import jev_grid_probability_v2
    _,checked5=jev_grid_probability_v2.replay(old5)
    if len(checked5)!=60: raise ValueError('v2 sensitivity run incomplete')
    result['same_response_threshold_sensitivity']={label:threshold_sensitivity([json.loads(line) for line in (path/'trace.jsonl').read_text(encoding='utf-8').splitlines()]) for label,path in (('v2_responses',old5),('v3_responses',new))}
    if result['status']!=statuses[-1]['status']: raise ValueError('v3 status and replay disagree')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--v1',type=Path,required=True); parser.add_argument('--v3',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try: result=analyze(args.v1,args.v3)
    except (ValueError,FileNotFoundError) as exc:
        result={'schema':'jev-grid-decoder-comparison-v3','status':'INCOMPLETE_OR_INVALID','reason':str(exc)}
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8'); return 1
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8'); return 0
if __name__=='__main__': raise SystemExit(main())
