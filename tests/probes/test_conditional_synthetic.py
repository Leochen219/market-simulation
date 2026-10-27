from collections import Counter

from capclaims.conditional_behavior.synthetic import (complete_cases, numeric_cases,
    atomic_cases, atomic_packets, public_packet, validate_fixtures, INTEGER_BINS)


def test_full_case_oracles_follow_stated_policy_and_cover_zero_to_five():
    rows = complete_cases()
    assert len(rows) == 32
    assert {len(r['expected_orders']) for r in rows} == set(range(6))
    for r in rows:
        p = r['payload']
        orders = r['expected_orders']
        if r['family'] == 0:
            assert all(o == {'side': 'buy', 'qty': p['D'], 'price': p['p'] - 7} for o in orders)
        elif r['family'] == 1:
            h = p['history'][0]
            offset = h['orders'][0]['qty'] - (h['D'] - h['Q'])
            assert orders == [{'side': 'buy', 'qty': p['D'] - p['Q'] + offset, 'price': p['p'] - 3}]
        elif r['family'] == 2:
            winner = max(p['history'], key=lambda h: h['final_profit_dollars'])
            assert orders == winner['orders']
            for h in p['history']:
                net = sum((1 if o['side'] == 'buy' else -1) * o['qty'] for o in h['orders'])
                cost = sum((1 if o['side'] == 'buy' else -1) * o['qty'] * o['price'] for o in h['orders'])
                assert h['final_profit_dollars'] == h['p'] * min(h['D'], h['Q'] + net) - 10 * h['Q'] - cost
        else:
            for position, order in enumerate(orders):
                assert order == {'side': 'buy' if position % 2 == 0 else 'sell',
                    'qty': p['Q'] // (len(orders) + 1), 'price': p['p'] + 5}
        assert 'expected_orders' not in p


def test_numeric_bins_cover_every_legal_integer_exactly_once():
    counts = Counter(n for lo, hi in INTEGER_BINS for n in range(lo, hi + 1))
    assert set(counts) == set(range(1, 1000)) and set(counts.values()) == {1}
    assert max(hi - lo + 1 for lo, hi in INTEGER_BINS) <= 255
    assert all(r['payload']['a'] - r['payload']['b'] == r['expected_integer'] for r in numeric_cases())


def test_atomic_oracles_pair_separation_and_exact_repeats():
    rows = atomic_cases()
    for r in rows:
        p = r['payload']
        if r['family'] <= 1 and r['condition'] == 0:
            assert r['expected_choice'] is None and not r['identifiable']
        elif r['family'] == 0:
            h = p['history'][0]
            answer = p['D'] - p['Q'] + h['submitted_buy_quantity'] - (h['D'] - h['Q'])
            assert r['criteria'][r['expected_choice']] == str(answer)
        elif r['family'] == 1:
            answer = max(p['history'], key=lambda h: h['final_profit_dollars'])['buy_quantity']
            assert r['criteria'][r['expected_choice']] == str(answer)
        else:
            answer = p['Q'] + sum((1 if o['side'] == 'buy' else -1) * o['qty'] for o in p['submitted_orders'])
            assert r['criteria'][r['expected_choice']] == str(answer)
    appearances = Counter()
    for packet in atomic_packets():
        assert len({r['base_id'] for r in packet['questions']}) == 4
        appearances.update(r['id'] for r in packet['questions'])
        for question in public_packet(packet)['questions']:
            assert set(question) == {'id', 'facts', 'candidates'}
    assert set(appearances.values()) == {2} and len(appearances) == 96
    assert validate_fixtures()['status'] == 'OFFLINE_FIXTURES_VALID'
