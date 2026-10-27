import asyncio
from copy import deepcopy
from functools import lru_cache

import pytest

from capclaims.backends.jev_instructions import build_instruction
from capclaims.engine.protocol import DecisionReply
from tools.probes.jev_calculator import build_plan, make_request, run_session, summarize

uncached_build_plan = build_plan
build_plan = lru_cache(maxsize=1)(build_plan)


class FakeBackend:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.requests = []

    async def decide_many(self, requests):
        assert len(requests) == 1
        r = requests[0]
        self.requests.append(r)
        choice = next(self.choices)
        if choice == 'ERROR':
            raise RuntimeError('provider failed')
        if choice == 'TIE':
            probs = {c.candidate_id: 1 / len(r.candidates) for c in r.candidates}
        else:
            assert choice in {c.candidate_id for c in r.candidates}
            probs = {c.candidate_id: float(c.candidate_id == choice) for c in r.candidates}
        return [DecisionReply.from_request(r, probs)]


def session(arm='optional', case='sell_surplus'):
    return next(x for x in build_plan()['sessions'] if x['arm'] == arm and
                x['fixture']['case_id'] == case and x['order'] == 'original' and x['repetition'] == 0)


def test_plan_and_clean_wire_are_frozen_without_oracle():
    from tools.probes.jev_input_ablation import Condition, _request, fixtures, _wire_payload
    plan = build_plan()
    assert len(plan['sessions']) == 216
    assert plan['request_cap'] == 366
    assert plan['max_estimated_input_tokens'] <= 2_000_000
    assert plan == uncached_build_plan()
    row = session('baseline')
    r = make_request(row, [])
    old, _ = _request(fixtures()[0], Condition('x', False, None, 'both_explicit'), 'original', 0, 0)
    assert _wire_payload(r) == _wire_payload(old)
    assert not any(k in str(r.observation) for k in ('oracle', 'expected', 'arm', 'repetition'))
    assert 'does not execute' in build_instruction(make_request(session(), []), 'calculator_v1')


@pytest.mark.parametrize('choices,query_ids', [(['c1'], []), (['t1', 'c1'], ['c1']),
                                            (['t0', 't1', 'c1'], ['c0', 'c1'])])
def test_optional_loop_only_executes_selected_query_and_keeps_state_private(choices, query_ids):
    row = session(); frozen = deepcopy(row)
    backend = FakeBackend(choices)
    result = asyncio.run(run_session(row, backend))
    assert result['selected'] == 'SELL' and result['correct']
    assert [q['candidate_id'] for q in result['tool_results']] == query_ids
    assert row == frozen
    for request in backend.requests[1:]:
        assert request.observation['procured_capacity_units'] == 60
    if query_ids:
        assert 't' + query_ids[0][1:] not in {c.candidate_id for c in backend.requests[-1].candidates}
    if len(query_ids) == 2:
        assert [c.candidate_id for c in backend.requests[-1].candidates] == ['c0', 'c1']
        assert result['tool_results'][1]['result']['final_profit_cents'] == 110000
        assert 'calculator_results' not in result['calls'][0]['wire']['state']
        assert len(result['calls'][1]['wire']['state']['calculator_results']) == 1
        assert result['calls'][1]['wire']['state']['calculator_results'][0]['candidate_id'] == 'c0'
        assert len(backend.requests[1].observation['calculator_results']) == 1


@pytest.mark.parametrize('order', ['original', 'reversed'])
def test_auto_results_are_not_counted_as_autonomous_calls(order):
    backend = FakeBackend(['c1'])
    row = deepcopy(session('provided'))
    row['order'] = order
    row = next(r for r in build_plan()['sessions'] if r['sequence'] != row['sequence'] and
               r['arm']=='provided' and r['order']==order and r['fixture']['case_id']=='sell_surplus') if order=='reversed' else row
    result = asyncio.run(run_session(row, backend))
    assert len(backend.requests) == 1
    assert len(result['tool_results']) == 2
    assert result['model_tool_calls'] == 0
    assert result['correct']


def test_tie_stops_without_silently_executing_a_tool():
    result = asyncio.run(run_session(session(), FakeBackend(['TIE'])))
    assert result['selected'] == 'TOP_TIE'
    assert result['tool_results'] == []
    assert not result['correct']


def test_errors_propagate_and_no_fallback_action_is_fabricated():
    transcript = []
    with pytest.raises(RuntimeError):
        asyncio.run(run_session(session(), FakeBackend(['ERROR']), transcript))
    assert transcript[0]['error_type'] == 'RuntimeError'
    assert 'reply' not in transcript[0]
    assert transcript[0]['wire']['state']['procured_capacity_units'] == 60


def test_accounting_every_fixture_matches_independent_cashflow_and_ties_separate():
    plan = build_plan()
    for row in plan['sessions']:
        r = make_request(row, [])
        for choice in row['actions']:
            f = row['fixture']; a = choice['net_quantity']; m = choice['market_price_cents']
            assert row['payoffs_cents'][choice['candidate_id']] == (-100*f['w']*f['Q'] - m*a + 100*f['p']*min(f['D'], f['Q']+a))
    row = session('baseline', 'equal_profit')
    result = asyncio.run(run_session(row, FakeBackend(['c0'])))
    s = summarize([result])
    assert s['arms']['baseline']['unique_n'] == 0
    assert s['arms']['baseline']['economic_tie_n'] == 1
