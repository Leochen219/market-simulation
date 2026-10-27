"""Offline-capable single-order flat $5 price diagnostic (unfrozen)."""
from __future__ import annotations
import asyncio
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from capclaims.conditional_behavior.transport import serialize_body, MAX_BODY
from capclaims.grid_sampling_v3 import decode

VERSION = 'flat_price_grid5_v1'
PRICE_VALUES = {f'p{v:03d}': v for v in range(5, 1000, 5)}
FORECAST = ('You are an observer predicting this human from the supplied history and current state. '
            'Forecast behavior, not advice or profit maximization. Predict one order on the supplied side; '
            'no target answers are supplied. ')
INSTRUCTIONS = {
    'quantity': FORECAST + 'Choose one complete order quantity in units.',
    'price': FORECAST + 'Choose one complete limit price in dollars per unit, conditional on the selected quantity.'}
TRADING = ('Diagnostic only: choose a complete quantity in units and a complete limit price in dollars per unit, '
           'on a $5 grid from $5 to $995. Side is supplied; submitted intentions do not execute trades.')
EMPTY = 'top3_cutoff03 has no surviving positive-mass candidate; no fallback is allowed'


def stage_spec(state, side, stage, quantity=None):
    if not isinstance(state, dict) or side not in ('BUY', 'SELL'):
        raise ValueError('state and BUY/SELL side required')
    copied = deepcopy(state)
    copied.pop('model_chosen_band', None)
    copied.pop('model_chosen_quantity', None)
    copied.pop('own_qty_units', None)
    copied['own_side'] = side
    instructions = copied.setdefault('public_instructions', {})
    if not isinstance(instructions, dict): raise ValueError('public instructions must be a mapping')
    instructions['trading'] = TRADING
    own = copied.get('own_q_units')
    orders = copied.get('partial_orders', [])
    if not isinstance(orders, list) or len(orders) >= 5:
        raise ValueError('invalid submitted prefix or no remaining order slots')
    buy = sold = 0
    for index, order in enumerate(orders):
        if not isinstance(order, dict) or set(order) != {'slot', 'side', 'quantity_units', 'limit_price_dollars_per_unit'}:
            raise ValueError('invalid canonical submitted order')
        qty, price = order['quantity_units'], order['limit_price_dollars_per_unit']
        if (type(order['slot']) is not int or order['slot'] != index or order['side'] not in ('BUY', 'SELL')
                or type(qty) is not int or qty <= 0 or qty > 990 or qty % 10 or isinstance(price, bool)):
            raise ValueError('invalid submitted order fields')
        try:
            parsed_price = Decimal(str(price))
        except (InvalidOperation, ValueError):
            raise ValueError('invalid submitted price') from None
        if not parsed_price.is_finite() or parsed_price <= 0:
            raise ValueError('invalid submitted price')
        if order['side'] == 'SELL': sold += qty
        else: buy += qty
    if type(own) is not int or own < 0 or sold > own:
        raise ValueError('invalid inventory or submitted SELL quantity')
    ledger = copied.get('submitted_order_ledger')
    if ledger is None:
        ledger = {'submitted_order_count': len(orders), 'submitted_buy_quantity_units': buy,
                  'submitted_sell_quantity_units': sold, 'remaining_sell_submission_capacity_units': own - sold,
                  'remaining_order_slots': 5 - len(orders)}
        copied['submitted_order_ledger'] = ledger
    if not isinstance(ledger, dict): raise ValueError('invalid submitted ledger')
    expected = {'submitted_order_count': len(orders), 'submitted_buy_quantity_units': buy,
                'submitted_sell_quantity_units': sold, 'remaining_sell_submission_capacity_units': own - sold}
    for key, value in expected.items():
        if key in ledger and (type(ledger[key]) is not int or ledger[key] != value):
            raise ValueError('submitted ledger disagrees with order prefix')
    if 'remaining_order_slots' in ledger and (type(ledger['remaining_order_slots']) is not int
            or not 1 <= ledger['remaining_order_slots'] <= 5 - len(orders)):
        raise ValueError('invalid remaining order slots')
    maximum = min(990, own - sold) if side == 'SELL' else 990
    quantities = list(range(10, maximum + 1, 10))
    if stage == 'quantity':
        if not quantities: raise ValueError('no legal quantity')
        return copied, {f'q{q}': str(q) for q in quantities}
    if stage != 'price' or type(quantity) is not int or quantity not in quantities:
        raise ValueError('price requires a legal selected quantity')
    copied.update(own_qty_units=quantity, model_chosen_quantity=quantity)
    return copied, {key: f'{value:.2f}' for key, value in PRICE_VALUES.items()}


def decode_choice(raw, ids, *, seed, person_id, repeat, stage):
    if stage not in ('quantity', 'price'): raise ValueError('invalid stage')
    try:
        result = decode(raw, ids, seed=seed, person_id=person_id, repeat=repeat, stage=stage)
    except ValueError as exc:
        if str(exc) != EMPTY: raise
        return {'status': 'client_stop', 'selected_id': None, 'reason': 'empty_valid_numeric_support'}
    return {'status': 'selected', 'selected_id': result['selected_id'], 'decision': result}


async def checked_body(state, stage, criteria):
    if not 1 <= len(criteria) <= 255: raise ValueError('Choice must contain 1..255 candidates')
    body = await serialize_body(state=state, instructions=INSTRUCTIONS[stage], criteria=criteria)
    if len(body) > MAX_BODY: raise ValueError('body exceeds 12000 bytes')
    return body


async def predict(client, path, *, seed, semaphore=None):
    person, repeat, side = path['person_id'], path['repeat'], path['side']
    steps = []
    orders = deepcopy(path['state'].get('partial_orders', []))
    if not isinstance(orders, list): raise ValueError('partial_orders must be a list')
    quantity = None
    for stage in ('quantity', 'price'):
        state, options = stage_spec(path['state'], side, stage, quantity)
        request_id = f'{person}:{repeat}:{stage}'
        body = None
        response = None
        try:
            body = await checked_body(state, stage, options)
            async def call():
                return await client.choose(request_id=request_id, state=state,
                    instructions=INSTRUCTIONS[stage], criteria=options)
            if semaphore is None: response = await call()
            else:
                async with semaphore: response = await call()
            answer = response['answers']['decision']
            raw = answer['probabilities']
            decision = decode_choice(raw, list(options), seed=seed, person_id=person, repeat=repeat, stage=stage)
        except Exception as exc:
            # Preserve completed steps and failing request for the caller's audit.
            exc.flat_grid_audit = {'path': deepcopy(path), 'steps': deepcopy(steps),
                                   'failure': {'stage': stage, 'request_id': request_id, 'state': deepcopy(state),
                                               'instructions': INSTRUCTIONS[stage], 'criteria': deepcopy(options),
                                               'body': body.decode('utf-8') if body is not None else None,
                                               'response': deepcopy(response), 'type': type(exc).__name__, 'message': str(exc)}}
            raise
        steps.append({'stage': stage, 'request_id': request_id, 'state': state, 'instructions': INSTRUCTIONS[stage],
                      'criteria': options, 'body_bytes': len(body), 'body': body.decode('utf-8'), 'response': deepcopy(response), 'raw_probabilities': deepcopy(raw),
                      'provider_choice': answer.get('choice'), 'decision': decision})
        sid = decision['selected_id']
        if sid is None:
            return {'version': VERSION, 'steps': steps, 'status': 'client_stop', 'submitted_orders': orders, 'pending_draft': None}
        if stage == 'quantity': quantity = int(options[sid])
        else:
            orders.append({'slot': len(orders), 'side': side, 'quantity_units': quantity, 'limit_price_dollars_per_unit': PRICE_VALUES[sid]})
    return {'version': VERSION, 'steps': steps, 'status': 'selected', 'submitted_orders': orders, 'pending_draft': None}


async def run_batch(client, paths, *, seed):
    identities = [(p['person_id'], p['repeat']) for p in paths]
    if len(set(identities)) != len(identities): raise ValueError('duplicate path identity')
    semaphore = asyncio.Semaphore(10)
    outcomes = await asyncio.gather(*(predict(client, p, seed=seed, semaphore=semaphore) for p in paths), return_exceptions=True)
    errors = [x for x in outcomes if isinstance(x, BaseException)]
    if errors:
        error = RuntimeError('flat-grid batch failed; all started paths have settled')
        error.outcomes = outcomes
        raise error from errors[0]
    return outcomes


async def replay(paths, results, *, seed):
    """Reconstruct every request and selection; reject omissions, additions and edits."""
    if len(paths) != len(results) or len({(p['person_id'], p['repeat']) for p in paths}) != len(paths):
        raise ValueError('path coverage mismatch')
    for path, result in zip(paths, results):
        steps = result['steps']; quantity = None; expected_orders = deepcopy(path['state'].get('partial_orders', []))
        if not isinstance(steps, list) or not 1 <= len(steps) <= 2: raise ValueError('invalid step count')
        for idx, step in enumerate(steps):
            stage = ('quantity', 'price')[idx]
            state, options = stage_spec(path['state'], path['side'], stage, quantity)
            expected = decode_choice(step['raw_probabilities'], list(options), seed=seed,
                                     person_id=path['person_id'], repeat=path['repeat'], stage=stage)
            body = await checked_body(state, stage, options)
            if (step['stage'] != stage or step['request_id'] != f"{path['person_id']}:{path['repeat']}:{stage}"
                or step['state'] != state or step['criteria'] != options or list(step['criteria']) != list(options)
                or step['instructions'] != INSTRUCTIONS[stage] or step['decision'] != expected
                or step['body_bytes'] != len(body) or step['body'] != body.decode('utf-8')
                or step['response']['answers']['decision']['probabilities'] != step['raw_probabilities']
                or step['response']['answers']['decision'].get('choice') != step['provider_choice']):
                raise ValueError('request or decision mismatch')
            sid = expected['selected_id']
            if sid is None:
                if idx != len(steps)-1: raise ValueError('continued after stop')
            elif stage == 'quantity': quantity = int(options[sid])
            else: expected_orders.append({'slot': len(expected_orders), 'side': path['side'], 'quantity_units': quantity, 'limit_price_dollars_per_unit': PRICE_VALUES[sid]})
        status = 'client_stop' if steps[-1]['decision']['selected_id'] is None else 'selected'
        if status == 'selected' and len(steps) != 2: raise ValueError('missing price')
        if result != {'version': VERSION, 'steps': steps, 'status': status, 'submitted_orders': expected_orders, 'pending_draft': None}:
            raise ValueError('result mismatch')
    return results


class SyntheticClient:
    """No-network deterministic mock, optionally synchronizing the first ten requests."""
    def __init__(self, barrier=0):
        self.barrier, self.active, self.peak, self.arrived = barrier, 0, 0, 0
        self.ready = asyncio.Event()

    async def choose(self, *, request_id, state, instructions, criteria):
        self.active += 1; self.peak = max(self.peak, self.active)
        try:
            if self.barrier and request_id.endswith(':quantity'):
                self.arrived += 1
                if self.arrived >= self.barrier: self.ready.set()
                await self.ready.wait()
            await asyncio.sleep(0)
            first = next(iter(criteria))
            return {'answers': {'decision': {'choice': first, 'probabilities': {k: float(k == first) for k in criteria}}}}
        finally: self.active -= 1
