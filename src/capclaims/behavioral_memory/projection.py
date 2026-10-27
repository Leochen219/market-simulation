"""Compact named-field presentation; all arithmetic lives in the audited compiler.

Definitions and shared denominators occur once instead of being repeated for every
cell. This is lossless for the planned model-visible facts, not history truncation.
The full compiler object and exact rational audit remain local artifacts.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _stat(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in ('median', 'min', 'max')}


def project_behavioral_memory(memory: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if arm not in {'M0', 'M1', 'M2'}:
        raise ValueError('unknown memory arm')
    if arm == 'M0':
        return {}
    result: dict[str, Any] = {
        'definitions': {
            'formula_version': 'episode-relations-1',
            'units': 'quantity: units; price: dollars/unit; counts: submissions or rounds as named',
            'source': 'round:N references formal round N in the unchanged raw history',
            'gap_surplus': 'D_minus_Q=D-Q; positive_gap=max(D-Q,0); surplus=max(Q-D,0)',
            'side_fields': 'quantity=sum(qty); count=number of orders; quote_range=[min(price),max(price)]; '
                           'weighted_price=sum(qty*price)/sum(qty); price_minus_p=weighted_price-historical_p',
            'offsets': 'BUY_minus_gap=BUY.quantity-positive_gap; SELL_minus_surplus=SELL.quantity-surplus',
            'duplicates': 'duplicate_count=sum(max(multiplicity-1,0)) for identical(side,qty,price) within a round',
            'null_price': 'null means no orders of that side; it is not zero',
            'precision': 'Computed prices rounded to 2 decimals, ROUND_HALF_UP.',
            'interpretation': 'Submitted intentions, not fills. Relations describe observations, not recommended actions. '
                              'Within-round order position is not a reliable time sequence.',
        },
        'episode_table': {
            'columns': ['source_ref', 'formal_round', 'order_count', 'duplicate_count',
                'D_minus_Q', 'positive_gap', 'surplus', 'BUY_minus_gap', 'SELL_minus_surplus',
                'BUY_quantity', 'BUY_count', 'BUY_quote_range', 'BUY_weighted_price', 'BUY_price_minus_p',
                'SELL_quantity', 'SELL_count', 'SELL_quote_range', 'SELL_weighted_price', 'SELL_price_minus_p'],
            'rows': [],
        },
    }
    for episode in memory['episodes']:
        derived = episode['derived']
        row: dict[str, Any] = {
            'source_ref': episode['source_ref'], 'formal_round': episode['raw']['round'],
            'order_count': derived['order_count'], 'duplicate_count': derived['duplicate_submissions'],
            'D_minus_Q': derived['D_minus_Q'], 'positive_gap': derived['gap'],
            'surplus': derived['surplus'], 'BUY_minus_gap': derived['buy_minus_gap'],
            'SELL_minus_surplus': derived['sell_minus_surplus'],
        }
        for side in ('buy', 'sell'):
            price_range = derived[f'{side}_price_range']
            row[side.upper()] = {
                'quantity': derived[f'{side}_quantity'], 'count': derived[f'{side}_order_count'],
                'quote_range': [price_range['min'], price_range['max']] if price_range else None,
                'weighted_price': derived[f'{side}_weighted_price'],
                'price_minus_p': derived[f'{side}_weighted_price_minus_p'],
            }
        flattened = {key: value for key, value in row.items() if key not in ('BUY', 'SELL')}
        for side in ('BUY', 'SELL'):
            flattened.update({side + '_' + key: value for key, value in row[side].items()})
        result['episode_table']['rows'].append(
            [flattened[key] for key in result['episode_table']['columns']])
    if arm == 'M2':
        result['definitions']['conditional_formula_version'] = 'conditional-groups-1'
        result['definitions']['conditions'] = (
            'shortage: D>Q; equal: D=Q; surplus: D<Q. Each group uses the listed source_refs and denominator n. '
            'Quantity and order_count statistics include zero in rounds without orders. '
            'Price statistics summarize per-round quantity-weighted prices only where that side exists; '
            'valid_n and source_refs give their denominator and contributors. '
            'quote_range spans all actual quotes of that side. Empty groups have null statistics.')
        result['conditions'] = []
        for group in memory['conditional_summaries']:
            compact: dict[str, Any] = {
                'condition': group['demand_relation'], 'n': group['n'], 'source_refs': group['source_refs'],
                'any_BUY_count': group['any_buy_count'], 'any_SELL_count': group['any_sell_count'],
                'empty_count': group['empty_count'], 'mixed_count': group['mixed_count'],
                'order_count': _stat(group['order_count']),
            }
            for side in ('buy', 'sell'):
                prices = group['prices'][side]
                compact[side.upper()] = {
                    'quantity': _stat(group['quantity'][side]),
                    'price': {**_stat(prices), 'valid_n': prices['valid_n'],
                              'source_refs': prices['source_refs'],
                              'quote_range': ([prices['quote_price_min'], prices['quote_price_max']]
                                              if prices['valid_n'] else None)},
                }
            result['conditions'].append(compact)
        result['current_relevance'] = {
            **memory['relevance'],
            'tie_rule': 'equal exact distance: more recent formal round first; all history remains present',
        }
    return deepcopy(result)
