from copy import deepcopy

import pytest

from capclaims.backends.jev import JevBackend
from capclaims.backends.jev_instructions import build_instruction
from tools.probes.jev_calculator import build_plan as legacy_plan, make_request, tool_result, wire
from tools.probes.jev_calculator_completion import build_plan
from tools.probes.jev_calculator_prompt import analyze
from capclaims.engine.protocol import DecisionReply


def test_legacy_hashes_and_default_analysis_conditions_are_preserved(tmp_path, monkeypatch):
    assert legacy_plan()['plan_sha256'] == 'a66818ccfa43fb31fd73d939ed5a9ef672d2af100071a8e796954bd8a0bb3517'
    from tools.probes.jev_calculator_prompt import build_plan as old_prompt_plan
    assert old_prompt_plan()['plan_sha256'] == '105469c42bc0f1bccbc04a97b4a553db7f7e5b5149701db22eb1691236a06f51'
    assert analyze.__defaults__ == (('v1', 'v2'),)


def test_v3_is_versioned_instruction_only_and_keeps_every_action_available():
    plan = build_plan()
    assert plan['instruction_profile'] == 'calculator_v3'
    assert len(plan['sessions']) == 144
    assert plan['request_cap'] == 444
    for row in plan['sessions']:
        assert row['prompt_condition'] in ('v2', 'v3')
        assert len(row['reachable_nodes']) > 0
        for node in row['reachable_nodes']:
            request = node['request']
            if request['stage'] == 'jev:calculator_tool_v3':
                text = node['wire']['instruction']
                for candidate in request['candidates']:
                    if candidate['value'] in ('STOP', 'BUY', 'SELL'):
                        assert candidate['candidate_id'] in text
                assert 'STOP is one candidate' in text
                assert 'calculated only if its candidate_id appears in calculator_results' in text
                assert 'choose one remaining CALCULATE' in text
                assert 'not a final action' in text
                assert 'never zero' in text
                assert 'Do not prefer an action based only on its name or candidate ID' in text
                assert 'rank' not in text.lower() and 'best candidate is' not in text.lower()
                assert [c['candidate_id'] for c in request['candidates'] if c['value'] in ('STOP', 'BUY', 'SELL')]
                assert node['wire']['state'] == request['observation']


def test_v2_v3_prefix_states_criteria_and_final_instruction_match():
    plan = build_plan()
    pairs = {}
    for row in plan['sessions']:
        pairs.setdefault((row['fixture']['case_id'], row['order'], row['repetition']), {})[row['prompt_condition']] = row
    for pair in pairs.values():
        left, right = pair['v2'], pair['v3']
        assert left['actions'] == right['actions']
        for a, b in zip(left['reachable_nodes'], right['reachable_nodes']):
            assert a['query_prefix'] == b['query_prefix']
            assert a['wire']['state'] == b['wire']['state']
            assert a['wire']['criteria'] == b['wire']['criteria']
            if a['request']['stage'] == 'jev:one_step_trade':
                assert a['wire']['instruction'] == b['wire']['instruction']
            elif a['request']['stage'] == 'jev:calculator_tool_v2':
                assert b['request']['stage'] == 'jev:calculator_tool_v3'
                assert a['wire']['instruction'] != b['wire']['instruction']
    assert sum(row['max_requests'] for row in plan['sessions']) == 444


def test_stage_dispatch_and_backend_allowlist_are_backwards_compatible():
    plan = build_plan()
    node_count = 0
    for row in plan['sessions']:
        profile = f"calculator_{row['prompt_condition']}"
        for node in row['reachable_nodes']:
            node_count += 1
            prefix_results = [tool_result(row, candidate_id) for candidate_id in node['query_prefix']]
            request = make_request(row, prefix_results)
            assert wire(request) == node['wire']
            assert build_instruction(request, profile) == node['wire']['instruction']
            assert build_instruction(request, plan['instruction_profile']) == node['wire']['instruction']
    assert node_count == 852
    row = next(r for r in plan['sessions'] if r['prompt_condition'] == 'v3')
    request = make_request(row, [])
    assert request.stage == 'jev:calculator_tool_v3'
    assert build_instruction(request, 'calculator_v2')


def test_preview_plan_is_offline_and_has_rebuildable_tree():
    plan = build_plan()
    assert plan['seed'] == 20261024
    assert plan['token_cap'] == 2_000_000
    assert plan['max_estimated_input_tokens'] <= plan['token_cap']
    assert len({row['sequence'] for row in plan['sessions']}) == 144
    assert all(row['max_requests'] <= 4 for row in plan['sessions'])


class AllQueryBackend:
    async def decide_many(self, requests):
        request = requests[0]
        candidate = next((c for c in request.candidates if c.value == 'CALCULATE'), None)
        selected = candidate.candidate_id if candidate else 'c0'
        return [DecisionReply.from_request(request, {
            c.candidate_id: float(c.candidate_id == selected) for c in request.candidates})]


def test_all_144_maximum_query_paths_total_444_calls():
    import asyncio
    from tools.probes.jev_calculator import run_session
    plan = build_plan()
    async def run_all():
        return await asyncio.gather(*(run_session(row, AllQueryBackend()) for row in plan['sessions']))
    results = asyncio.run(run_all())
    assert len(results) == 144
    assert sum(len(row['calls']) for row in results) == 444


def test_preview_does_not_load_env_or_construct_backend(tmp_path, monkeypatch):
    from capclaims.backends import jev as jev_backend
    from tools.probes import jev_calculator
    from tools.probes import jev_calculator_completion
    def forbidden(*args, **kwargs):
        raise AssertionError('offline preview touched execution setup')
    monkeypatch.setattr(jev_calculator, 'JevBackend', forbidden)
    monkeypatch.setattr(jev_backend, 'load_local_env', forbidden)
    output = tmp_path / 'preview'
    assert jev_calculator_completion.main(['--output-dir', str(output)]) == 0
    assert (output / 'preview.json').is_file()
