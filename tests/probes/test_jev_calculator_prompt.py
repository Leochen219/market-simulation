import asyncio
from copy import deepcopy
from functools import lru_cache

from capclaims.backends.jev_instructions import build_instruction
from capclaims.engine.protocol import DecisionReply
from tools.probes.jev_calculator import build_plan as legacy_plan, make_request, run_session, wire
from tools.probes.jev_calculator_prompt import build_plan

build_plan = lru_cache(maxsize=1)(build_plan)


def test_old_plan_and_prompt_are_unchanged_new_prompt_only_changes_instruction():
    assert legacy_plan()['plan_sha256']=='a66818ccfa43fb31fd73d939ed5a9ef672d2af100071a8e796954bd8a0bb3517'
    plan=build_plan()
    assert len(plan['sessions'])==144 and plan['request_cap']==444
    assert plan['max_estimated_input_tokens']<=plan['token_cap']==2_000_000
    for row in plan['sessions']:
        other=next(r for r in plan['sessions'] if r['prompt_condition']!=row['prompt_condition'] and
                   all(r[k]==row[k] for k in ('order','repetition')) and r['fixture']==row['fixture'])
        w=wire(make_request(row,[])); o=wire(make_request(other,[]))
        assert w['state']==o['state'] and w['criteria']==o['criteria']
        assert w['instruction']!=o['instruction']
        if row['prompt_condition']=='v2':
            assert 'NEXT STEP' in w['instruction']
            assert build_instruction(make_request(row,[]),'calculator_v2')==w['instruction']
        for node in row['reachable_nodes']:
            if len(node['query_prefix'])==len(row['actions']):
                assert node['wire']['instruction']==build_instruction(None,'ablation_v1')


class AllQueryBackend:
    async def decide_many(self,requests):
        r=requests[0]
        tool=next((c for c in r.candidates if c.value=='CALCULATE'),None)
        choice=tool.candidate_id if tool else 'c0'
        return [DecisionReply.from_request(r,{c.candidate_id:float(c.candidate_id==choice) for c in r.candidates})]


def test_every_reachable_maximum_query_path_remains_callable_and_private():
    plan=build_plan(); frozen=deepcopy(plan)
    async def run():
        return [await run_session(row,AllQueryBackend()) for row in plan['sessions']]
    rows=asyncio.run(run())
    assert sum(len(r['calls']) for r in rows)==444
    assert all(r['model_tool_calls']==len(r['tool_results']) for r in rows)
    assert all(r['all_actions_calculated'] for r in rows)
    assert plan==frozen
