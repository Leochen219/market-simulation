from copy import deepcopy

from capclaims.behavioral_memory.memory import compile_behavioral_memory
from capclaims.behavioral_memory.projection import project_behavioral_memory


def memory():
    return compile_behavioral_memory([
        {'round': 6, 'p': 30, 'Q': 10, 'D': 20, 'profit_cents': 100,
         'orders': [{'side': 'buy', 'qty': 12, 'price': 21},
                    {'side': 'buy', 'qty': 12, 'price': 21},
                    {'side': 'sell', 'qty': 3, 'price': 35}]},
        {'round': 7, 'p': 40, 'Q': 10, 'D': 20, 'profit_cents': -10, 'orders': []},
        {'round': 8, 'p': 20, 'Q': 20, 'D': 0, 'profit_cents': 0,
         'orders': [{'side': 'sell', 'qty': 11, 'price': 999}]},
    ], {'round': 13, 'p': 25, 'Q': 10, 'D': 20}, 12)[0]


def test_arm_projection_preserves_facts_but_not_audit_duplication():
    original = memory()
    before = deepcopy(original)
    assert project_behavioral_memory(original, 'M0') == {}
    m1 = project_behavioral_memory(original, 'M1')
    assert set(m1) == {'definitions', 'episode_table'}
    rows = [dict(zip(m1['episode_table']['columns'], row)) for row in m1['episode_table']['rows']]
    for projected, episode in zip(rows, original['episodes']):
        derived = episode['derived']
        assert projected['source_ref'] == episode['source_ref']
        assert projected['formal_round'] == episode['raw']['round']
        assert projected['duplicate_count'] == derived['duplicate_submissions']
        assert projected['BUY_quantity'] == derived['buy_quantity']
        assert projected['SELL_weighted_price'] == derived['sell_weighted_price']
        assert 'relevance' not in projected
    m2 = project_behavioral_memory(original, 'M2')
    assert m2['episode_table'] == m1['episode_table']
    assert m2['current_relevance']['closest_source_refs'] == original['relevance']['closest_source_refs']
    for compact, group in zip(m2['conditions'], original['conditional_summaries']):
        assert compact['source_refs'] == group['source_refs']
        assert compact['n'] == group['n']
        assert compact['BUY']['quantity']['median'] == group['quantity']['buy']['median']
        assert compact['BUY']['price']['valid_n'] == group['prices']['buy']['valid_n']
        assert compact['SELL']['price']['median'] == group['prices']['sell']['median']
    assert original == before


def test_absent_prices_stay_null_and_no_prediction_or_advice_is_created():
    result = project_behavioral_memory(memory(), 'M2')
    empty = dict(zip(result['episode_table']['columns'], result['episode_table']['rows'][1]))
    assert empty['BUY_quantity'] == 0
    assert empty['BUY_weighted_price'] is None
    assert empty['SELL_quote_range'] is None
    import json
    encoded = json.dumps(result)
    for forbidden in ('recommended_action', 'personality', 'target_profit', 'predicted_order', 'wholesale'):
        assert forbidden not in encoded
