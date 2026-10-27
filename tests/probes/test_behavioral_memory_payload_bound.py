"""Check the maximizer against alternative legal states and exact SDK bytes."""
import asyncio
import itertools

from capclaims.behavioral_memory.wire_preview import (
    _instruction, _order_state, _wire_length, maximum_legal_request,
)
from capclaims.behavioral_memory.transport import serialize_body
from capclaims.paper.codec import prefix_mask
from capclaims.paper.interface import build_candidates


def test_bound_covers_shorter_prefixes_and_decimal_positions():
    for q in (0, 1, 9, 10, 99, 100, 400):
        biggest, instruction, candidates = maximum_legal_request(q)
        bound = _wire_length(biggest, instruction, candidates)
        assert biggest['submitted_order_ledger']['submitted_sell_quantity_units'] <= q
        assert len(biggest['partial_orders']) == 4
        base = {'own_q_units': q, 'decision_interface': 'economic_v2'}
        for n in range(5):
            for sell_n in range(n + 1):
                if sell_n > q:
                    continue
                prior = [{'side': 'sell', 'qty': 1, 'price': 999}] * sell_n
                prior += [{'side': 'buy', 'qty': 999, 'price': 999}] * (n - sell_n)
                remaining = q - sell_n
                for side, field, prefix in itertools.product(('buy', 'sell'), ('quantity', 'price'), ('', '0', '1', '9', '00', '09', '10', '99')):
                    if side == 'sell' and not remaining:
                        continue
                    maximum = remaining if side == 'sell' and field == 'quantity' else 999
                    digits = prefix_mask(prefix, 1, maximum)
                    if not digits:
                        continue
                    state = {**base, **_order_state(base, prior), 'phase': 'orders',
                        'decision_field': field, 'slot': n, 'own_side': side,
                        'numeric_prefix': prefix, 'choice_minimum': 1, 'choice_maximum': maximum,
                        'decimal_place': ('h', 't', 'u')[len(prefix)], 'selected_order_count': 5,
                        'remaining_planned_order_slots': 5 - n}
                    if field == 'price':
                        state['own_qty_units'] = remaining if side == 'sell' else 999
                    criteria = {c.candidate_id: c.label for c in build_candidates(digits, state)}
                    text = _instruction(field, slot=n, side=side, prefix=prefix)
                    assert _wire_length(state, text, criteria) <= bound


def test_sdk_size_matches_maximization_encoding():
    async def exercise():
        for q in (0, 400):
            state, text, criteria = maximum_legal_request(q)
            raw = await serialize_body(state=state, instructions=text, criteria=criteria)
            assert len(raw) == _wire_length(state, text, criteria)
            # Largest digit labels are not necessarily the final '99' branch.
            assert state['numeric_prefix'] == ''
    asyncio.run(exercise())
