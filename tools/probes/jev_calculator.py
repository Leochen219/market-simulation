"""Frozen small three-arm calculator diagnostic; never runs a market session."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any

from capclaims.backends.jev import JevBackend, estimate_input_tokens
from capclaims.backends.jev_instructions import build_instruction
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest, validate_reply
from capclaims.paper.payoff_tool import calculate_payoff
from capclaims.randomness import canonical_json
from tools.probes.jev_input_ablation import Fixture, _action_labels, fixtures as old_fixtures

SEED = 20261024
REQUEST_CAP = 366
TOKEN_CAP = 2_000_000
ARMS = ('baseline', 'optional', 'provided')


def fixtures():
    old = {f.case_id: f for f in old_fixtures()}
    rows = [old[k] for k in ('sell_surplus', 'expiry_zero_demand', 'new_surplus')]
    def add(name, p, q, d, m, x, family):
        rows.append(Fixture(name, 'frozen_validation', family, p, q, d, 10, (('SELL', x, m),)))
    add('Q27', 70, 27, 20, 15, 10, 'boundary')
    add('Q28', 70, 28, 20, 15, 10, 'boundary')
    add('D32', 70, 40, 32, 15, 10, 'boundary')
    add('D33', 70, 40, 33, 15, 10, 'boundary')
    add('needed_harmful', 100, 20, 20, 70, 10, 'needed')
    add('needed_premium', 70, 20, 20, 85, 10, 'needed')
    add('equal_profit', 70, 20, 20, 70, 10, 'tie')
    add('new_surplus_83', 83, 57, 31, 17, 13, 'fresh_surplus')
    add('new_needed_62', 62, 25, 24, 19, 6, 'fresh_needed')
    return rows


def _sha(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def tool_result(row, candidate_id):
    """Bind only the model-selected listed scenario; no future-state lookup."""
    action = next(a for a in row['actions'] if a['candidate_id'] == candidate_id)
    f = row['fixture']
    args = dict(p=f['p'], wholesale_price=f['w'], q=f['Q'], demand=f['D'],
                net_quantity=action['net_quantity'], market_price_cents=action['market_price_cents'])
    return {'candidate_id': candidate_id, 'arguments': args, 'result': calculate_payoff(**args)}


def make_request(row, results):
    f = row['fixture']
    obs = {'retail_value_per_unit_dollars': f['p'], 'procured_capacity_units': f['Q'],
           'customer_demand_units': f['D'], 'unit_procurement_cost_dollars': f['w'],
           'procurement_already_paid': True}
    if results:
        obs['calculator_results'] = deepcopy(results)
        obs['calculator_units'] = 'All monetary fields ending in _cents are cents; 100 cents = $1.'
    actions = [Candidate(**a) for a in row['candidates']]
    if row['order'] == 'reversed':
        actions.reverse()
    queried = {r['candidate_id'] for r in results}
    tools = []
    if row['arm'] == 'optional':
        for candidate in actions:
            if candidate.candidate_id not in queried:
                args = next(a for a in row['actions'] if a['candidate_id'] == candidate.candidate_id)
                tools.append(Candidate('t'+candidate.candidate_id[1:], 'CALCULATE',
                    f"CALCULATE hypothetical accounting only for {candidate.candidate_id}: "
                    f"{candidate.label}. Parameters p={f['p']}, wholesale_price={f['w']}, "
                    f"q={f['Q']}, demand={f['D']}, net_quantity={args['net_quantity']}, "
                    f"market_price_cents={args['market_price_cents']}; money returned in cents."))
    condition = row.get('prompt_condition')
    tool_stage = {'v2': 'jev:calculator_tool_v2', 'v3': 'jev:calculator_tool_v3'}.get(condition, 'jev:calculator_tool')
    stage = tool_stage if tools else 'jev:one_step_trade'
    # Reverse all options as a block so position effects include tools; IDs stay fixed.
    candidates = actions + tools if row['order'] == 'original' else tools + actions
    return DecisionRequest(f"calculator_{row['sequence']:03d}_{len(results)}", 0, stage, obs, tuple(candidates))


def wire(request):
    profile = {'jev:calculator_tool_v2': 'calculator_v2', 'jev:calculator_tool_v3': 'calculator_v3'}.get(
        request.stage, 'calculator_v1')
    return {'state': dict(request.observation), 'instruction': build_instruction(request, profile),
            'criteria': {c.candidate_id: c.label for c in request.candidates}}


def _tree(row, results):
    request = make_request(row, results)
    branches = []
    cost = estimate_input_tokens(request, wire(request)['instruction'])
    count = 1
    nodes = [{'query_prefix': [r['candidate_id'] for r in results],
              'request': asdict(request), 'wire': wire(request),
              'wire_sha256': _sha({**wire(request), 'criteria': list(wire(request)['criteria'].items())})}]
    for c in request.candidates:
        if c.value == 'CALCULATE':
            child_nodes, child_cost, child_count = _tree(row, results + [tool_result(row, 'c'+c.candidate_id[1:])])
            nodes.extend(child_nodes)
            branches.append((child_cost, child_count))
    if branches:
        cost += max(x[0] for x in branches)
        count += max(x[1] for x in branches)
    return nodes, cost, count


def build_plan():
    schedule = list(itertools.product(fixtures(), ARMS, ('original', 'reversed'), range(3)))
    random.Random(SEED).shuffle(schedule)
    rows = []; total_cost = total_calls = 0
    for sequence, (f, arm, order, repetition) in enumerate(schedule):
        candidates, profits = _action_labels(f, 'both_explicit')
        specs = [('STOP', 0, 0), *f.actions]
        actions = [{'candidate_id': f'c{i}', 'side': side,
                    'net_quantity': qty if side == 'BUY' else -qty if side == 'SELL' else 0,
                    'market_price_cents': price*100} for i, (side, qty, price) in enumerate(specs)]
        best = max(profits.values())
        row = {'sequence': sequence, 'fixture': asdict(f), 'arm': arm, 'order': order,
               'repetition': repetition, 'candidates': [asdict(c) for c in candidates],
               'actions': actions, 'payoffs_cents': {k: 100*v for k,v in profits.items()},
               'optimal_ids': [k for k,v in profits.items() if v == best]}
        results = [tool_result(row, c['candidate_id']) for c in row['candidates']] if arm == 'provided' else []
        nodes, cost, calls = _tree(row, results)
        row.update({'reachable_nodes': nodes, 'max_estimated_input_tokens': cost, 'max_requests': calls})
        rows.append(row); total_cost += cost; total_calls += calls
    if total_calls != REQUEST_CAP or total_cost > TOKEN_CAP:
        raise ValueError('frozen plan exceeds configured budget')
    plan = {'model': 'jev-1.14.0', 'seed': SEED, 'sessions': rows, 'request_cap': REQUEST_CAP,
            'token_cap': TOKEN_CAP, 'max_estimated_input_tokens': total_cost,
            'max_in_flight': 10, 'instruction_profile': 'calculator_v1'}
    return {**plan, 'plan_sha256': _sha(plan)}


async def run_session(row, backend, transcript=None):
    results = []
    if row['arm'] == 'provided':
        results = [tool_result(row, c['candidate_id']) for c in row['candidates']]
    calls = []; selected = 'TOP_TIE'; selected_id = None
    for _ in range(row['max_requests']):
        request = make_request(row, results)
        # Every reachable request was frozen before dispatch, including ordering.
        matched = next((n for n in row['reachable_nodes'] if n['query_prefix'] == [r['candidate_id'] for r in results]), None)
        if matched is None or wire(request) != matched['wire']:
            raise ValueError('runtime wire differs from frozen request tree')
        try:
            reply = (await backend.decide_many([request]))[0]
        except Exception as exc:
            if transcript is not None:
                transcript.append({'sequence':row['sequence'], 'request':asdict(request),
                                   'wire':wire(request), 'error_type':type(exc).__name__})
            raise
        validate_reply(request, reply)
        call = {'sequence': row['sequence'], 'request': asdict(request), 'wire': wire(request), 'reply': asdict(reply)}
        calls.append(call)
        if transcript is not None:
            transcript.append(call)
        top = max(reply.probabilities.values())
        winners = [c for c in request.candidates if reply.probabilities[c.candidate_id] == top]
        if len(winners) != 1:
            break
        chosen = winners[0]
        if chosen.value == 'CALCULATE':
            result = tool_result(row, 'c'+chosen.candidate_id[1:])
            call['tool_result'] = result
            results.append(result)
        else:
            selected = chosen.value; selected_id = chosen.candidate_id
            break
    else:
        raise RuntimeError('session exhausted without final action')
    unique = len(row['optimal_ids']) == 1
    final_profit = row['payoffs_cents'].get(selected_id)
    return {**({'prompt_condition': row['prompt_condition']} if 'prompt_condition' in row else {}),
            'sequence': row['sequence'], 'case_id': row['fixture']['case_id'],
            'arm': row['arm'], 'order': row['order'], 'repetition': row['repetition'],
            'selected': selected, 'selected_id': selected_id, 'economic_tie': not unique,
            'optimal_ids': row['optimal_ids'], 'correct': selected_id in row['optimal_ids'],
            'payoffs_cents': row['payoffs_cents'], 'profit_cents': final_profit,
            'regret_cents': max(row['payoffs_cents'].values())-final_profit if final_profit is not None else None,
            'tool_results': results, 'model_tool_calls': len(results) if row['arm']=='optional' else 0,
            'all_actions_calculated': len(results) == len(row['actions']), 'calls': calls}


def summarize(rows):
    arms = {}
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        unique = [r for r in group if not r['economic_tie']]
        def counts(g):
            return {'n': len(g), 'correct': sum(r['correct'] for r in g),
                    'stop': sum(r['selected']=='STOP' for r in g),
                    'top_ties': sum(r['selected']=='TOP_TIE' for r in g)}
        favorable = [r for r in unique if any(k != 'c0' and v > r['payoffs_cents']['c0']
                                            for k,v in r['payoffs_cents'].items())]
        queried = [r for r in group if r['model_tool_calls']]
        complete = [r for r in group if r['all_actions_calculated'] and not r['economic_tie']]
        arms[arm] = {'n':len(group), 'unique_n':len(unique), 'unique_correct':sum(r['correct'] for r in unique),
                     'economic_tie_n':len(group)-len(unique), 'top_ties':sum(r['selected']=='TOP_TIE' for r in group),
                     'favorable_trade_n':len(favorable), 'favorable_trade_stop':sum(r['selected']=='STOP' for r in favorable),
                     'tool_use_sessions':len(queried), 'all_calculated_unique_n':len(complete),
                     'all_calculated_wrong':sum(not r['correct'] for r in complete),
                     'harmful_sell':sum(r['selected']=='SELL' and r['profit_cents']<r['payoffs_cents']['c0'] for r in unique),
                     'total_regret_cents':sum(r['regret_cents'] or 0 for r in group),
                     'by_case':{case: counts([r for r in group if r['case_id']==case]) for case in sorted({r['case_id'] for r in group})}}
    return {'arms':arms, 'completed_sessions':len(rows)}


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


async def execute(plan, output):
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve(), root/'src/capclaims/paper/payoff_tool.py',
             root/'src/capclaims/paper/projections.py', root/'src/capclaims/backends/jev.py',
             root/'src/capclaims/backends/jev_instructions.py', root/'tools/probes/jev_input_ablation.py',
             root/'src/capclaims/engine/protocol.py', root/'src/capclaims/randomness.py']
    paths += [root/p for p in plan.get('extra_source_paths', [])]
    _write(output/'frozen-plan.json', {**plan,'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    evidence = []; failures = []; transcript = []; backend = None
    try:
        backend = JevBackend(instruction_profile=plan['instruction_profile'], max_requests=plan['request_cap'],
                             max_estimated_input_tokens=plan['token_cap'], rps=40, tps=50000,
                             max_in_flight=10)
        for offset in range(0, len(plan['sessions']), 10):
            wave = plan['sessions'][offset:offset+10]
            replies = await asyncio.gather(*(run_session(r, backend, transcript) for r in wave), return_exceptions=True)
            for row, reply in zip(wave,replies):
                if isinstance(reply, BaseException):
                    failures.append({'sequence':row['sequence'],'error_type':type(reply).__name__})
                else:
                    evidence.append(reply)
            _write(output/'raw-evidence.json',evidence)
            _write(output/'call-transcript.json',transcript)
            if failures:
                break
    except Exception as exc:
        failures.append({'phase':'initialization_or_execution','error_type':type(exc).__name__})
    finally:
        if backend is not None:
            await backend.aclose()
    _write(output/'failures.json',failures)
    result = {'status':'COMPLETE' if len(evidence)==len(plan['sessions']) and not failures else 'PARTIAL',
              'plan_sha256':plan['plan_sha256'], 'summary':summarize(evidence),
              'budget_usage':backend.budget_usage if backend else None,
              'window_timings':list(backend.window_timings) if backend else [],
              'transport_timings':list(backend.transport_timings) if backend else []}
    _write(output/'result.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args=parser.parse_args(argv)
    plan=build_plan()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    if args.execute:
        result=asyncio.run(execute(plan,args.output_dir))
    else:
        _write(args.output_dir/'preview.json',plan)
        result={'status':'PREVIEW','sessions':len(plan['sessions']), 'request_cap':REQUEST_CAP,
                'max_estimated_input_tokens':plan['max_estimated_input_tokens'], 'plan_sha256':plan['plan_sha256']}
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status'] in ('PREVIEW','COMPLETE') else 1


if __name__=='__main__':
    raise SystemExit(main())
