import json
from copy import deepcopy

import pytest

from capclaims.behavioral_memory.memory import compile_behavioral_memory
from capclaims.behavioral_memory.design import build_prompt
from capclaims.behavioral_memory.sampling import sample_trajectory
from capclaims.paper.codec import prefix_mask


def test_every_full_integer_remains_reachable():
    for maximum in (1, 9, 10, 99, 100, 400, 999):
        prefixes = ['']
        for _ in range(3):
            prefixes = [p + str(d) for p in prefixes for d in prefix_mask(p, 1, maximum)]
        assert {int(p) for p in prefixes} == set(range(1, maximum + 1))


@pytest.mark.parametrize('bad', [
    {'profit_cents': None}, {'D': 1, 'd': 2}, {'Q': 1, 'q': 2}, {'w': 10}, {'fills': []},
    {'orders': [{'side': 'sell', 'qty': 2, 'price': 1}]},
    {'orders': [{'side': 'buy', 'qty': 0, 'price': 1}]},
    {'orders': [{'side': 'buy', 'qty': 1, 'price': 1000}]},
])
def test_bad_history_rejected_with_otherwise_valid_cutoff(bad):
    history = [{'round': 6, 'p': 1, 'Q': 1, 'D': 1, 'profit_cents': 0, 'orders': [], **bad}]
    with pytest.raises(ValueError):
        compile_behavioral_memory(history, {'round': 13, 'p': 1, 'Q': 1, 'D': 1}, 12)


def test_zero_states_duplicate_five_orders_and_arm_information_parity():
    history = [{'round': r, 'p': 1, 'Q': 0, 'D': 0, 'profit_cents': 0,
                'orders': [{'side': 'buy', 'qty': 999, 'price': 999}] * 5 if r == 6 else []}
               for r in range(6, 13)]
    state = {'round': 13, 'p': 1, 'w': 10, 'Q': 0, 'D': 0}
    memory, audit = compile_behavioral_memory(history, state, 12)
    first = memory['episodes'][0]['derived']
    assert first['duplicate_submissions'] == 4
    assert first['buy_quantity'] == 4995
    assert first['sell_weighted_price'] is None
    observations = [json.loads(build_prompt(arm=arm, current_state=state,
                    history=memory['history'], memory=memory))['observation']
                    for arm in ('M0', 'M1', 'M2')]
    raw = deepcopy(observations[0])
    assert 'behavioral_memory' not in raw
    for observation in observations[1:]:
        addition = observation.pop('behavioral_memory')
        assert addition['episode_table']['rows'][0][0] == 'round:6'
        assert observation == raw


@pytest.mark.asyncio
async def test_mixed_five_orders_exhaust_sell_cap_without_shrinking_buy_domain():
    class Provider:
        async def choose(self, *, request_id, state, instructions, criteria):
            field = state['decision_field']
            if field == 'order_count':
                choice = 'N5'
            elif field == 'side':
                choice = 'SELL' if not state['partial_orders'] else 'BUY'
                if state['partial_orders']:
                    assert 'SELL' not in criteria
            else:
                target = 400 if field == 'quantity' and state['own_side'] == 'sell' else 999
                choice = 'digit' + f'{target:03d}'[len(state['numeric_prefix'])]
                if field == 'price':
                    assert state['own_qty_units'] in (400, 999)
            assert choice in criteria
            return {'model': 'jev-1.14.0', 'answers': {'decision': {'type': 'choice', 'choice': choice,
                'probabilities': {key: float(key == choice) for key in criteria}}}}
    state = {'round': 13, 'p': 1, 'w': 10, 'Q': 400, 'D': 0}
    row = {'person_id': 'test', 'target_round': 13, 'repeat': 0, 'arm': 'M0',
           'request_prefix': 'test-full-domain', 'current_state': state,
           'prompt': build_prompt(arm='M0', current_state=state, history=[])}
    result = await sample_trajectory(Provider(), row)
    assert result['physical_requests'] == 36
    assert result['orders'] == [{'side': 'sell', 'qty': 400, 'price': 999}] + [
        {'side': 'buy', 'qty': 999, 'price': 999}] * 4
