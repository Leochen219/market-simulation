"""Recompute descriptive calculator diagnostics from saved synthetic evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize_arm(rows):
    unique = [r for r in rows if not r['economic_tie']]
    resolved = [r for r in unique if r['selected_id'] is not None]
    favorable = []
    for r in unique:
        sides = {c['candidate_id']:c['value'] for c in r['calls'][0]['request']['candidates']}
        if any(sides.get(k)=='SELL' and v>r['payoffs_cents']['c0'] for k,v in r['payoffs_cents'].items()):
            favorable.append(r)
    coverage = Counter(len(r['tool_results']) for r in rows)
    all_seen = [r for r in unique if r['all_actions_calculated']]
    return {'sessions':len(rows), 'unique_n':len(unique), 'unique_correct':sum(r['correct'] for r in unique),
            'unique_accuracy':sum(r['correct'] for r in unique)/len(unique) if unique else None,
            'economic_ties':len(rows)-len(unique), 'probability_ties':sum(r['selected_id'] is None for r in rows),
            'favorable_sell_n':len(favorable), 'favorable_sell_stop':sum(r['selected']=='STOP' for r in favorable),
            'harmful_sell':sum(r['selected']=='SELL' and r['profit_cents']<r['payoffs_cents']['c0'] for r in unique),
            'autonomous_tool_use_sessions':sum(r['model_tool_calls']>0 for r in rows),
            'model_tool_calls':sum(r['model_tool_calls'] for r in rows),
            'result_coverage_sessions':dict(sorted(coverage.items())),
            'all_calculated_unique_n':len(all_seen), 'all_calculated_wrong':sum(not r['correct'] for r in all_seen),
            'resolved_unique_n':len(resolved),
            'sum_regret_cents_resolved_unique':sum(r['regret_cents'] for r in resolved),
            'mean_regret_cents_resolved_unique':sum(r['regret_cents'] for r in resolved)/len(resolved) if resolved else None}


def analyze(directory):
    def read(name):
        return json.loads((directory/name).read_text(encoding='utf-8'))
    rows=read('raw-evidence.json'); run=read('result.json'); plan=read('frozen-plan.json')
    transcript=read('call-transcript.json'); failures=read('failures.json')
    if run['status']!='COMPLETE' or failures or len(rows)!=216:
        raise ValueError('Expected complete frozen experiment; do not silently score partial data')
    by_sequence={r['sequence']:r for r in rows}
    if len(by_sequence)!=216 or set(by_sequence)!={r['sequence'] for r in plan['sessions']}:
        raise ValueError('session identity mismatch')
    for r in rows:
        for call in r['calls']:
            if call['wire']['state']!=call['request']['observation']:
                raise ValueError('mutable or mismatched wire snapshot')
    if len(transcript)!=run['budget_usage']['requests_completed']:
        raise ValueError('call transcript coverage mismatch')
    arms={arm:summarize_arm([r for r in rows if r['arm']==arm]) for arm in ('baseline','optional','provided')}
    by_case=[]
    for case in sorted({r['case_id'] for r in rows}):
        case_rows=[r for r in rows if r['case_id']==case]
        by_case.append({'case_id':case, 'payoffs_cents':case_rows[0]['payoffs_cents'],
                        'optimal_ids':case_rows[0]['optimal_ids'],
                        'arms':{a:{'correct':sum(r['correct'] for r in case_rows if r['arm']==a),
                                   'n':sum(r['arm']==a for r in case_rows),
                                   'actions':dict(Counter(r['selected'] for r in case_rows if r['arm']==a))}
                                for a in arms}})
    primary=['sell_surplus','expiry_zero_demand','new_surplus']
    primary_stats={a:summarize_arm([r for r in rows if r['arm']==a and r['case_id'] in primary]) for a in arms}
    pairwise={}
    for arm in ('optional','provided'):
        pairs=[]
        for r in rows:
            if r['arm']!=arm or r['economic_tie']:
                continue
            base=next(b for b in rows if b['arm']=='baseline' and all(b[k]==r[k] for k in ('case_id','order','repetition')))
            pairs.append((base['correct'],r['correct']))
        pairwise[arm]={'pairs':len(pairs),'wrong_to_correct':sum(not a and b for a,b in pairs),
                       'correct_to_wrong':sum(a and not b for a,b in pairs)}
    events=[]
    for t in run['transport_timings']:
        events.extend([(t['started_at'],1),(t['finished_at'],-1)])
    active=maximum=0
    for _,delta in sorted(events):
        active+=delta; maximum=max(maximum,active)
    frozen_sources=plan['source_sha256']
    root=Path(__file__).resolve().parents[2]
    return {'model':plan['model'],'plan_sha256':plan['plan_sha256'], 'status':run['status'],
            'arms':arms,'primary_failures':primary_stats,'by_case':by_case,'paired_transitions':pairwise,
            'budget_usage':run['budget_usage'], 'actual_http_max_overlap':maximum,
            'frozen_source_files_still_match':all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in frozen_sources.items()),
            'evidence_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                               for name in ('frozen-plan.json','raw-evidence.json','call-transcript.json','result.json','failures.json')},
            'examples':{arm:next(r for r in rows if r['arm']==arm and r['case_id']=='sell_surplus' and r['order']=='original' and r['repetition']==0)
                        for arm in arms},
            'limitations':['Repeated calls are not independent economic states.',
                           'Optional arm changes instruction/options/length; no isolated arithmetic causality.',
                           'Provided results are automatic support, not autonomous tool use.',
                           'Probability ties have undefined payoff/regret and are excluded from payoff means.',
                           'No20round market or general readiness claim.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.directory)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','arms','actual_http_max_overlap','frozen_source_files_still_match')}))


if __name__=='__main__':
    main()
