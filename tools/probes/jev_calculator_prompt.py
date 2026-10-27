"""Single frozen v1/v2 prompt contrast; all tool calls remain model-selected."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from capclaims.randomness import canonical_json
from tools.analysis.analyze_jev_calculator import summarize_arm
from tools.probes.jev_calculator import build_plan as legacy_plan, _tree, _write, execute

REQUEST_CAP = 444
TOKEN_CAP = 2_000_000
SEED = 20261024


def build_plan():
    rows=[]
    for old in legacy_plan()['sessions']:
        if old['arm']!='optional':
            continue
        for condition in ('v1','v2'):
            row=deepcopy(old)
            row['prompt_condition']=condition
            rows.append(row)
    random.Random(SEED).shuffle(rows)
    total_tokens=total_requests=0
    for sequence,row in enumerate(rows):
        row['sequence']=sequence
        nodes,cost,count=_tree(row,[])
        row.update(reachable_nodes=nodes,max_estimated_input_tokens=cost,max_requests=count)
        total_tokens+=cost; total_requests+=count
    if total_requests!=REQUEST_CAP or total_tokens>TOKEN_CAP:
        raise ValueError('prompt contrast exceeds frozen budget')
    plan={'model':'jev-1.14.0','seed':SEED,'sessions':rows,'request_cap':REQUEST_CAP,'token_cap':TOKEN_CAP,
          'max_estimated_input_tokens':total_tokens,'instruction_profile':'calculator_v2','max_in_flight':10,
          'extra_source_paths':['tools/probes/jev_calculator.py','tools/probes/jev_calculator_prompt.py',
                                'tools/analysis/analyze_jev_calculator.py','src/capclaims/domain/types.py']}
    return {**plan,'plan_sha256':hashlib.sha256(canonical_json(plan).encode()).hexdigest()}


def analyze(directory, conditions=('v1', 'v2')):
    def read(name):
        return json.loads((directory/name).read_text(encoding='utf-8'))
    run=read('result.json'); plan=read('frozen-plan.json'); rows=read('raw-evidence.json')
    failures=read('failures.json'); transcript=read('call-transcript.json')
    if run['status']!='COMPLETE' or failures or len(rows)!=144:
        raise ValueError('Only the complete frozen contrast can be analyzed')
    if len({r['sequence'] for r in rows})!=144 or {r['sequence'] for r in rows}!={r['sequence'] for r in plan['sessions']}:
        raise ValueError('session coverage mismatch')
    if len(transcript)!=run['budget_usage']['requests_completed']:
        raise ValueError('transcript coverage mismatch')
    summaries={}; primary={}; cases={}
    for condition in conditions:
        group=[r for r in rows if r['prompt_condition']==condition]
        summary=summarize_arm(group)
        summary['query_count_histogram']=dict(sorted(Counter(r['model_tool_calls'] for r in group).items()))
        summary['query_order_histogram']=dict(Counter(','.join(t['candidate_id'] for t in r['tool_results']) for r in group))
        summaries[condition]=summary
        primary[condition]=summarize_arm([r for r in group if r['case_id'] in ('sell_surplus','expiry_zero_demand','new_surplus')])
        cases[condition]={case:{'n':len(g),'correct':sum(r['correct'] for r in g),
                               'actions':dict(Counter(r['selected'] for r in g)),
                               'tool_use':sum(r['model_tool_calls']>0 for r in g),
                               'all_calculated':sum(r['all_actions_calculated'] for r in g)}
                          for case in sorted({r['case_id'] for r in group})
                          for g in [[r for r in group if r['case_id']==case]]}
    for call in transcript:
        if call['wire']['state']!=call['request']['observation']:
            raise ValueError('wire snapshot mismatch')
    events=[]
    for t in run['transport_timings']:
        events.extend([(t['started_at'],1),(t['finished_at'],-1)])
    active=maximum=0
    for _,delta in sorted(events):
        active+=delta; maximum=max(maximum,active)
    root=Path(__file__).resolve().parents[2]
    return {'status':run['status'],'model':plan['model'],'plan_sha256':plan['plan_sha256'],
            'conditions':summaries,'primary_cases':primary,'by_case':cases,'budget_usage':run['budget_usage'],
            'actual_http_max_overlap':maximum,
            'frozen_source_files_still_match':all(hashlib.sha256((root/p).read_bytes()).hexdigest()==v for p,v in plan['source_sha256'].items()),
            'evidence_sha256':{name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                               for name in ('frozen-plan.json','raw-evidence.json','call-transcript.json','result.json','failures.json')},
            'examples':{condition:next(r for r in rows if r['prompt_condition']==condition and r['case_id']=='sell_surplus' and r['order']=='original' and r['repetition']==0) for condition in conditions}}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--analyze',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.analyze:
        if args.execute or args.output_dir or not args.output:
            parser.error('--analyze requires --output and excludes execution/preview')
        result=analyze(args.analyze)
        _write(args.output,result)
        print(json.dumps({k:result[k] for k in ('status','conditions','actual_http_max_overlap','frozen_source_files_still_match')}))
        return 0
    if not args.output_dir:
        parser.error('--output-dir is required for preview/execution')
    plan=build_plan()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    if args.execute:
        result=asyncio.run(execute(plan,args.output_dir))
        # The shared runner's old aggregate summary is preserved; condition-aware analysis is separate.
        result={k:result[k] for k in ('status','plan_sha256','budget_usage')}
    else:
        _write(args.output_dir/'preview.json',plan)
        result={'status':'PREVIEW','sessions':len(plan['sessions']),'request_cap':REQUEST_CAP,
                'max_estimated_input_tokens':plan['max_estimated_input_tokens'],'plan_sha256':plan['plan_sha256']}
    print(json.dumps(result))
    return 0 if result['status'] in ('PREVIEW','COMPLETE') else 1


if __name__=='__main__':
    raise SystemExit(main())
