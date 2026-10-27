"""Frozen conditional price-distribution comparison; targets are scoring-only."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
from pathlib import Path

from capclaims.backends.jev import normalize_jev_probabilities
from capclaims.flat_price_grid import PRICE_VALUES
from tools.analysis.compare_jev_grid_human_distributions import weighted, w1
from tools.probes.jev_grid_probability import BANDS

METHODS = ('hierarchical_2_5', 'flat_5')
EVENTS = ('ge700', '700_to_800', 'gt250', 'above_retail')


def identity(row):
    return tuple(row[k] for k in ('person_id', 'formal_round', 'condition', 'repeat'))


def project_price(value, step):
    values = [i * step for i in range(1, int(999 // step) + 1)]
    return min(values, key=lambda p: (abs(p - value), -p))


def distribution(points):
    if not points:
        return {'mean': None, 'median': None, 'p95': None, 'cdf': []}
    values = sorted({v for v, _ in points})
    rows = [{'value': v, 'mass': math.fsum(w for x, w in points if x == v),
             'cdf': math.fsum(w for x, w in points if x <= v)} for v in values]
    def quantile(q):
        return next((r['value'] for r in rows if r['cdf'] + 1e-12 >= q), rows[-1]['value'])
    return {'mean': math.fsum(v * w for v, w in points), 'median': quantile(.5),
            'p95': quantile(.95), 'cdf': rows}


def event(value, retail, name):
    if name == 'ge700': return value >= 700
    if name == '700_to_800': return 700 <= value < 800
    if name == 'gt250': return value > 250
    if name == 'above_retail': return value > retail
    raise ValueError('unknown event')


def event_counts(prices, retail, name):
    """No quote is a separate outcome, not a zero or an inferred human STOP."""
    n = len(prices)
    known = [(p, r) for p, r in zip(prices, retail) if p is not None]
    count = sum(event(p, r, name) for p, r in known)
    missing = n - len(known)
    return {'events': count, 'planned': n, 'quoted': len(known), 'no_quote': missing,
            'confirmed_quote_event_per_planned': count / n if n else None,
            'among_quotes': count / len(known) if known else None,
            'hypothetical_unreported_price_bounds': [count / n, (count + missing) / n] if n else None}


def raw_tail(prediction, method):
    """Unfiltered model-score mixture at the identical fixed quantity, not calibration."""
    if method == 'flat_5':
        raw = prediction.get('raw_probabilities')
        if raw is None: return None
        probs, _ = normalize_jev_probabilities(raw, list(PRICE_VALUES))
        return math.fsum(p for key, p in probs.items() if PRICE_VALUES[key] >= 700)
    saved = prediction.get('probabilities')
    if saved is None: return None
    bands, _ = normalize_jev_probabilities(saved['band'], [f'b{i}' for i in range(4)])
    parts = []
    for i, (lo, hi) in enumerate(BANDS):
        ids = [f'p{tick}' for tick in range(lo, hi + 1)]
        probs, _ = normalize_jev_probabilities(saved[f'price-b{i}'], ids)
        parts.append(bands[f'b{i}'] * math.fsum(p for key, p in probs.items() if int(key[1:]) * 2.5 >= 700))
    return math.fsum(parts)


def summarize(plan, predictions, old_plan, old_predictions, targets):
    old = {identity(p): p for p in old_predictions}
    new = {identity(p): p for p in predictions}
    scheduled = {identity(p): p for p in plan['schedule']}
    if len(old) != len(old_predictions) or len(new) != len(predictions) or len(scheduled) != len(plan['schedule']):
        raise ValueError('duplicate path identity')
    if set(new) != set(scheduled) or set(old) != set(scheduled):
        raise ValueError('complete comparison requires every planned response')
    contexts = {(c['person_id'], c['formal_round'], c['condition']): c for c in old_plan['contexts']}
    if plan['contexts'] != old_plan['contexts']: raise ValueError('contexts differ')
    for key, row in new.items():
        if row['fixed_quantity'] != old[key]['conditioned_quantity']:
            raise ValueError('quantity was not held fixed')
        if row['outcome'] not in ('selected', 'client_numeric_abstain'):
            raise ValueError('technical or unknown outcome cannot be scored as complete')
        value = row['price']
        if row['outcome'] == 'selected':
            if type(value) not in (int, float) or value not in PRICE_VALUES.values():
                raise ValueError('invalid complete flat-grid price')
        elif value is not None: raise ValueError('abstention has a price')
    results = {}
    for condition in sorted({k[2] for k in scheduled}):
        results[condition] = {}
        for side in ('BUY', 'SELL'):
            keys = [k for k in scheduled if k[2] == condition and contexts[k[:3]]['side'] == side]
            people = sorted({k[0] for k in keys})
            if not people: continue
            by_person = {p: sorted([k for k in keys if k[0] == p], key=lambda k: k[3]) for p in people}
            if any(len(v) != 2 or {k[3] for k in v} != {0, 1} for v in by_person.values()):
                raise ValueError('expected two repeats per person')
            human = {p: [o['price'] for o in targets[p]['orders'] if o['side'].upper() == side] for p in people}
            if any(not v for v in human.values()): raise ValueError('human lacks supplied-side orders')
            prices = {METHODS[0]: {k: old[k]['prices']['cutoff3'] for k in keys},
                      METHODS[1]: {k: new[k]['price'] for k in keys}}
            common = {p: [k for k in by_person[p] if all(prices[m][k] is not None for m in METHODS)] for p in people}
            covered = [p for p in people if common[p]]
            group = {'eligible_people': len(people), 'planned_paths_per_method': len(keys),
                     'common_people': len(covered), 'common_paths': sum(len(common[p]) for p in covered),
                     'methods': {}, 'human_events': {}, 'common_distributions': {}}
            raw_all = weighted([human[p] for p in people])
            group['human_full_distribution'] = distribution(raw_all)
            for name in EVENTS:
                group['human_events'][name] = math.fsum(
                    sum(event(v, contexts[by_person[p][0][:3]]['current_state']['p'], name) for v in human[p])
                    / (len(people) * len(human[p])) for p in people)
            raw = weighted([human[p] for p in covered]) if covered else []
            five = weighted([[project_price(v, 5) for v in human[p]] for p in covered]) if covered else []
            old_grid = weighted([[project_price(v, 2.5) for v in human[p]] for p in covered]) if covered else []
            group['common_distributions'].update(human_raw=distribution(raw), human_grid5=distribution(five))
            for method in METHODS:
                quoted = [k for k in keys if prices[method][k] is not None]
                entry = {'quoted': len(quoted), 'no_quote': len(keys) - len(quoted), 'events': {}}
                retail = [contexts[k[:3]]['current_state']['p'] for k in keys]
                for name in EVENTS:
                    entry['events'][name] = event_counts([prices[method][k] for k in keys], retail, name)
                points = weighted([[prices[method][k] for k in common[p]] for p in covered]) if covered else []
                entry.update(raw_human_w1=w1(raw, points) if covered else None,
                             common_grid5_human_w1=w1(five, points) if covered else None,
                             own_grid_human_w1=w1(old_grid if method == METHODS[0] else five, points) if covered else None)
                group['common_distributions'][method] = distribution(points)
                raw_masses = [raw_tail(old[k] if method == METHODS[0] else new[k], method) for k in keys]
                entry['raw_model_tail_available_paths'] = sum(v is not None for v in raw_masses)
                entry['raw_model_score_mass_ge700'] = math.fsum(raw_masses) / len(keys) if all(v is not None for v in raw_masses) else None
                group['methods'][method] = entry
            group['flat_minus_hierarchical_raw_w1'] = (group['methods'][METHODS[1]]['raw_human_w1'] - group['methods'][METHODS[0]]['raw_human_w1']) if covered else None
            results[condition][side] = group
    return {'schema': 'jev-flat-grid5-comparison-v1', 'people': len({k[0] for k in scheduled}),
            'planned_paths': len(scheduled), 'fixed_quantity_mismatches': 0,
            'outcomes': {'hierarchical_legacy_no_quote': sum(p['prices']['cutoff3'] is None for p in old.values()),
                         'flat_client_numeric_abstain': sum(p['outcome'] == 'client_numeric_abstain' for p in new.values())},
            'groups': results,
            'interpretation': 'Conditional prices at quantities chosen by the frozen old policy, not a full two-stage rerun. Primary W1 compares person-balanced pooled price distributions on identical successful paths, not average per-person errors. No quote is never a zero price or human STOP. High-event/planned and quoted-only fractions are separate; unreported-price bounds are hypothetical sensitivity, not identified probabilities for abstentions. Raw score mixtures are uncalibrated and conditional on fixed quantity. Grid, hierarchical interface and necessary wording differ; no unique causal attribution. Two repeats and one-market development participants do not establish calibration or generalization.'}


def accounting(root):
    uri = (Path(root) / 'cost-ledger.sqlite').resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        rows = list(db.execute('SELECT request_id,state,committed,input_tokens FROM attempts'))
    if any(r[1] != 'settled' for r in rows): raise ValueError('unsettled completed study')
    timings = []
    for rid, _, _, _ in rows:
        folder = Path(root) / 'wire-evidence' / hashlib.sha256(rid.encode()).hexdigest()
        timings.append(json.loads((folder / 'timing.json').read_text(encoding='utf-8')))
    starts = [r['started_monotonic'] for r in timings]
    points = sorted([(r['started_monotonic'], 1) for r in timings] + [(r['finished_monotonic'], -1) for r in timings])
    active = peak = 0
    for _, delta in points: active += delta; peak = max(peak, active)
    rolling = max((sum(0 <= t - s < 1 for s in starts) for t in starts), default=0)
    return {'requests': len(rows), 'input_tokens': sum(r[3] for r in rows),
            'cost_nanodollars': sum(r[2] for r in rows), 'observed_peak_inflight': peak,
            'max_rolling_second_requests': rolling, 'max_rolling_reserved_bytes': rolling * 12000,
            'elapsed_seconds': max((r['finished_monotonic'] for r in timings), default=0) - min(starts, default=0)}


async def report_async(root, output):
    from capclaims.flat_grid_study import runner
    plan, predictions, status = await runner.replay(root)
    if status['status'] != 'COMPLETE': raise ValueError('technical partial study is not a complete comparison')
    old_plan, old_predictions, _ = runner.baseline.replay()
    targets = json.loads((Path(root) / 'labels-private.json').read_text(encoding='utf-8'))
    result = summarize(plan, predictions, old_plan, old_predictions, targets)
    result.update(status=status, plan_sha256=runner.sha((Path(root) / 'plan.json').read_bytes()),
                  baseline_plan_sha256=plan['baseline_sha256'], accounting=accounting(root))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def report(root=None, output=None):
    from capclaims.flat_grid_study import runner
    return asyncio.run(report_async(root or runner.OUT, output or runner.ROOT / 'docs/analysis/jev-flat-grid5-comparison-2026-10-27.json'))
