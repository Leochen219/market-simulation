"""Public deterministic diagnostic fixtures; no participant data are used here.

Expected answers are kept outside payloads. Ambiguous information controls have
no unique action label and are excluded from action-accuracy denominators.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from typing import Any

from .protocol import validate_orders

SEED = 20261026
NUMERIC_TARGETS = (1, 9, 10, 11, 99, 100, 101, 249, 250, 251, 499, 500,
                   501, 749, 750, 751, 899, 900, 901, 989, 990, 991, 998, 999)
INTEGER_BINS = ((1, 250), (251, 500), (501, 750), (751, 999))


def _order(side, qty, price):
    return {'side': side, 'qty': qty, 'price': price}


def complete_cases() -> list[dict[str, Any]]:
    cases = []
    for i in range(32):
        family, variant = divmod(i, 8)
        p, q, d = 60 + variant, 150 + variant * 3, 190 - variant
        n = variant % 6
        history = []
        if family == 0:
            policy = f'The participant submits exactly {n} BUY orders, including identical repeated orders. Each has quantity D and limit price p minus 7. If the count is zero, submit nothing.'
            expected = [_order('buy', d, p - 7) for _ in range(n)]
        elif family == 1:
            offset = 1 + variant
            policy = ('The participant uses exactly one of two rules in every round: one BUY order with quantity D-Q, '
                      f'or one BUY order with quantity D-Q+{offset}. The limit price always equals p-3. '
                      'All displayed rounds have D greater than Q. Infer which rule from the historical submissions.')
            for r in range(11, 14):
                hp, hq, hd = p - (14 - r), 90 + r, 150 + r
                history.append({'round': r, 'p': hp, 'Q': hq, 'D': hd,
                    'orders': [_order('buy', hd - hq + offset, hp - 3)]})
            expected = [_order('buy', d - q + offset, p - 3)]
        elif family == 2:
            a = [_order('sell', 11 + variant, 17 + variant)]
            b = [_order('buy', 5 + variant, 9 + variant), _order('buy', 5 + variant, 9 + variant)]
            win_b = variant % 2 == 0
            # Construct feasible historical accounting outcomes. The oracle uses
            # all fills at the identical within-round limit price; execution
            # details are not supplied to the model, only historical final profit.
            pa, pb = (51, 90) if win_b else (90, 51)
            sold, bought = a[0]['qty'], sum(o['qty'] for o in b)
            profit_a = pa * min(d, q - sold) - 10 * q + sold * a[0]['price']
            profit_b = pb * min(d, q + bought) - 10 * q - sum(o['qty'] * o['price'] for o in b)
            history = [{'round': 12, 'p': pa, 'Q': q, 'D': d, 'orders': a,
                        'final_profit_dollars': profit_a},
                       {'round': 13, 'p': pb, 'Q': q, 'D': d, 'orders': b,
                        'final_profit_dollars': profit_b}]
            policy = ('Repeat exactly the whole submitted portfolio, with its original quantities and prices, '
                      'from whichever of the last two displayed rounds earned higher final profit. '
                      'Do not optimize that portfolio for the current state.')
            expected = deepcopy(max(history, key=lambda h: h['final_profit_dollars'])['orders'])
        else:
            policy = (f'Submit exactly {n} orders. Number positions starting at zero. Even positions BUY and odd positions SELL. '
                      f'Every order has quantity floor(Q/({n}+1)) and price p+5. Keep repeated orders as separate submissions. '
                      'This fixed personal rule need not maximize profit.')
            expected = [_order('buy' if j % 2 == 0 else 'sell', q // (n + 1), p + 5) for j in range(n)]
        expected = list(validate_orders(expected, initial_q=q))
        cases.append({'id': f'portfolio-{i:02d}', 'family': family,
            'payload': {'round': 14, 'p': p, 'w': 10, 'Q': q, 'D': d,
                        'history': history, 'participant_policy': policy}, 'expected_orders': expected})
    return cases


def numeric_cases() -> list[dict[str, Any]]:
    return [{'id': f'number-{i:02d}', 'payload': {'task': 'Return the integer difference a minus b.',
            'a': 1000 + i, 'b': 1000 + i - value}, 'expected_integer': value}
            for i, value in enumerate(NUMERIC_TARGETS)]


def atomic_cases() -> list[dict[str, Any]]:
    cases = []
    for i in range(48):
        family, v = divmod(i, 12)
        base = f'atomic-{i:02d}'
        p, q, d = 70 + v, 100 + v, 150 + v
        if family == 0:
            k = v + 1
            common = {'p': p, 'Q': q, 'D': d,
                'policy_family': f'Always one BUY order, with quantity either D-Q or D-Q+{k}. Which quantity does this participant submit now?'}
            options = {'a': str(d - q), 'b': str(d - q + k), 'c': 'Cannot identify from supplied information'}
            payloads = [{**common, 'history': []}, {**common, 'history': [
                {'Q': 90, 'D': 130, 'submitted_buy_quantity': 40 + k},
                {'Q': 110, 'D': 170, 'submitted_buy_quantity': 60 + k}]}]
            expected = [None, 'b']
        elif family == 1:
            a, b = 11 + v, 29 + v
            common = {'p': p, 'Q': q, 'D': d,
                'policy': 'Repeat the BUY quantity from the higher-profit of these two prior rounds. Both limit prices are 20.'}
            h = [{'round': 12, 'buy_quantity': a}, {'round': 13, 'buy_quantity': b}]
            full = [{**h[0], 'final_profit_dollars': 200 if v % 2 else 100},
                    {**h[1], 'final_profit_dollars': 100 if v % 2 else 200}]
            options = {'a': str(a), 'b': str(b), 'c': 'Cannot identify from supplied information'}
            payloads = [{**common, 'history': h}, {**common, 'history': full}]
            expected = [None, 'a' if v % 2 else 'b']
        else:
            orders = [_order('buy', 11 + v, 21), _order('sell', 7 + v, 31),
                      _order('buy', 11 + v, 21), _order('sell', 3, 41)]
            buys, sells = 22 + 2 * v, 10 + v
            common = {'Q': q, 'submitted_orders': orders,
                'question': 'For this hypothetical arithmetic calculation only, assume all submitted orders fill in full. What is inventory after all fills?'}
            correct = q + buys - sells
            options = {'a': str(correct), 'b': str(q), 'c': str(q + buys + sells)}
            if family == 2:
                payloads = [common, {**common, 'deterministic_ledger': {
                    'submitted_buy_total': buys, 'submitted_sell_total': sells,
                    'net_units_if_all_filled': buys - sells}}]
            else:
                payloads = [common, {**common, 'submitted_orders': [orders[j] for j in (3, 1, 2, 0)]}]
            expected = ['a', 'a']
        for condition in range(2):
            cases.append({'id': f'{base}-{condition}', 'base_id': base, 'family': family,
                          'condition': condition, 'payload': payloads[condition],
                          'criteria': options, 'expected_choice': expected[condition],
                          'identifiable': expected[condition] is not None})
    return cases


def atomic_packets() -> list[dict[str, Any]]:
    """48 packets, four unrelated bases each; paired variants never co-occur."""
    by_key = {(row['base_id'], row['condition']): row for row in atomic_cases()}
    packets = []
    for repeat in range(2):
        for condition in range(2):
            ids = [f'atomic-{i:02d}' for i in range(48)]
            random.Random(SEED + repeat * 100 + condition).shuffle(ids)
            for start in range(0, 48, 4):
                rows = [by_key[(base, condition)] for base in ids[start:start + 4]]
                packets.append({'id': f'packet-r{repeat}-c{condition}-{start // 4:02d}',
                    'repeat': repeat, 'questions': rows})
    return packets


def public_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {'packet_id': packet['id'], 'questions': [
        {'id': row['id'], 'facts': row['payload'], 'candidates': row['criteria']}
        for row in packet['questions']]}


def fixture_digest() -> str:
    data = {'complete': complete_cases(), 'numeric': numeric_cases(), 'atomic': atomic_packets()}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def validate_fixtures() -> dict[str, Any]:
    complete, numeric, atomic = complete_cases(), numeric_cases(), atomic_cases()
    assert len(complete) == 32 and len(numeric) == 24 and len(atomic) == 96
    assert {len(row['expected_orders']) for row in complete} == set(range(6))
    for row in complete:
        validate_orders(row['expected_orders'], initial_q=row['payload']['Q'])
    for row in numeric:
        assert row['payload']['a'] - row['payload']['b'] == row['expected_integer']
        assert sum(lo <= row['expected_integer'] <= hi for lo, hi in INTEGER_BINS) == 1
    assert len(atomic_packets()) == 48
    for packet in atomic_packets():
        assert len({q['base_id'] for q in packet['questions']}) == 4
        assert all('expected_choice' not in q for q in public_packet(packet)['questions'])
    return {'status': 'OFFLINE_FIXTURES_VALID', 'complete_cases': 32, 'numeric_cases': 24,
            'atomic_cases': 96, 'atomic_packets': 48, 'fixture_sha256': fixture_digest()}
