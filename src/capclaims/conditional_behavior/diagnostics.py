"""Frozen synthetic diagnostics and existing-helper Opus reference consultations.

Complete and numeric cases use the same raw-probability sampling as the human
study. Atomic classification uses client argmax; Opus supplies choices only.
This diagnostic distinction is frozen before outcomes, not fitted afterwards.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from . import VERSION
from .protocol import build_observation, deterministic_draw, sample_categorical
from .sampling import sample_portfolio
from .study import verify_preview
from .synthetic import (SEED, INTEGER_BINS, complete_cases, numeric_cases,
                        atomic_packets, public_packet, fixture_digest, validate_fixtures)
from .transport import serialize_questions, serialize_body, strict_json, _save, _json

ATOMIC_INSTRUCTION = ('You are advising on independent synthetic decision questions. '
    'For each question, use only its stated facts and select one listed candidate. '
    'These are specified-policy and arithmetic tasks, not profit-maximization advice. '
    'When the supplied facts cannot identify an action, choose the cannot-identify option if present.')
HELPER = Path.home() / '.codex/skills/cross-ai-collaboration/scripts/collaborate.py'
AUTH_PREFLIGHT = Path.home() / '.codex/skills/claude-opus-consult/scripts/claude_auth_preflight.py'


def complete_row(case, interface, repeat, *, oracle=False):
    payload = case['payload']
    history = []
    for episode in payload['history']:
        row = {k: episode[k] for k in ('round', 'p', 'Q', 'D', 'orders')}
        if 'final_profit_dollars' in episode:
            row['profit_cents'] = episode['final_profit_dollars'] * 100
        history.append(row)
    arm = 'C' if case['family'] == 2 else 'B'
    observation = build_observation(state={k: payload[k] for k in ('round', 'p', 'w', 'Q', 'D')},
                                    history=history, arm=arm)
    observation['protocol_interface'] = interface
    observation['participant_policy'] = payload['participant_policy']
    ident = f"{case['id']}:{interface}:{repeat}:oracle={oracle}"
    return {'case_id': hashlib.sha256(ident.encode()).hexdigest(), 'person_id': case['id'],
            'target_round': 14, 'arm': 'E' if interface == 'J2' else arm,
            'sample': repeat, 'interface': interface, 'observation': observation}


async def numeric_sample(client, case, encoding, repeat):
    from capclaims.paper.codec import prefix_mask
    from capclaims.paper.interface import build_candidates
    trace = []
    async def choose(stage, criteria, state):
        request_id = hashlib.sha256(f"{VERSION}:{case['id']}:{encoding}:{repeat}:{stage}".encode()).hexdigest()
        instructions = 'Compute the requested integer. Choose the current encoding branch from its full feasible integer range; leading zeros count.'
        response = await client.choose(request_id=request_id, state=state, instructions=instructions, criteria=criteria)
        raw = response['answers']['decision']['probabilities']
        draw = deterministic_draw(seed=SEED, version=VERSION, arm=encoding, person=case['id'],
                                  target=0, sample=repeat, stage=stage)
        selected = sample_categorical(raw, draw, tuple(criteria))
        trace.append({'request_id': request_id, 'state': state, 'instructions': instructions,
                      'criteria': criteria, 'raw_response': response, 'draw': draw, 'selected': selected})
        return selected
    if encoding == 'decimal':
        prefix = ''
        for place in 'htu':
            state = {**case['payload'], 'numeric_prefix': prefix, 'choice_minimum': 1,
                     'choice_maximum': 999, 'decision_field': 'quantity', 'decimal_place': place,
                     'decision_interface': 'economic_v2'}
            criteria = {c.candidate_id: c.label for c in build_candidates(prefix_mask(prefix, 1, 999), state)}
            prefix += (await choose(place, criteria, state)).removeprefix('digit')
        result = int(prefix)
    elif encoding == 'interval':
        selected_bin = await choose('bin', {f'b{i}': f'{lo}..{hi}' for i, (lo, hi) in enumerate(INTEGER_BINS)}, case['payload'])
        lo, hi = INTEGER_BINS[int(selected_bin[1:])]
        selected = await choose('integer', {f'n{i}': str(i) for i in range(lo, hi + 1)},
                                {**case['payload'], 'selected_interval': [lo, hi]})
        result = int(selected[1:])
    else:
        raise ValueError('unknown diagnostic numeric encoding')
    return {'id': case['id'], 'encoding': encoding, 'repeat': repeat, 'value': result,
            'expected': case['expected_integer'], 'trace': trace}


def packet_request(packet):
    public = public_packet(packet)
    questions = {row['id']: {'instructions': ATOMIC_INSTRUCTION +
        f" Evaluate only question {row['id']} in state.questions.", 'criteria': row['candidates']}
        for row in public['questions']}
    return public, questions


def freeze_diagnostics(human_preview: Path, output: Path):
    manifest = json.loads((human_preview / 'manifest.json').read_text(encoding='utf-8'))
    verify_preview(human_preview, manifest['preview_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    record = {'schema': VERSION, 'human_preview_sha256': manifest['preview_sha256'],
              **validate_fixtures(), 'complete_decoder': 'raw_distribution_sample',
              'numeric_decoder': 'raw_distribution_sample', 'atomic_decoder': 'client_argmax',
              'opus_helper_sha256': hashlib.sha256(HELPER.read_bytes()).hexdigest(),
              'opus_auth_preflight_sha256': hashlib.sha256(AUTH_PREFLIGHT.read_bytes()).hexdigest(),
              'opus_role': 'architect', 'opus_timeout_seconds': 180, 'opus_consultations': 48}
    _save(output / 'manifest.json', _json(record))
    _save(output / 'fixtures.json', _json({'complete': complete_cases(), 'numeric': numeric_cases(), 'atomic': atomic_packets()}))
    for packet in atomic_packets():
        request = ATOMIC_INSTRUCTION + '\nReturn a JSON object mapping every question ID to its chosen candidate ID.\n' + json.dumps(public_packet(packet), ensure_ascii=False)
        _save(output / f"{packet['id']}.txt", request.encode())
    return record


async def _batches(rows, call, save_dir):
    results = []
    async def one(index, row):
        result = await call(row)
        _save(save_dir / f'{index:04d}.json', _json(result))
        return result
    for start in range(0, len(rows), 10):
        values = await asyncio.wait_for(asyncio.gather(*(one(i, rows[i])
            for i in range(start, min(start + 10, len(rows))))), 600)
        results.extend(values)
    return results


async def run_jev_diagnostics(client, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    portfolios = [(c, j, r, False) for c in complete_cases() for j in ('J1', 'J2') for r in range(2)]
    portfolios += [(c, 'J2', 0, True) for c in complete_cases()]
    async def portfolio(spec):
        case, interface, repeat, oracle = spec
        row = complete_row(case, interface, repeat, oracle=oracle)
        result = await sample_portfolio(client, row, seed=SEED, category='synthetic',
            oracle_order_count=len(case['expected_orders']) if oracle else None)
        return {'case_id': case['id'], 'interface': interface, 'repeat': repeat, 'oracle': oracle,
                'row': row, 'result': result, 'expected': case['expected_orders']}
    async def atomic(packet):
        state, questions = packet_request(packet)
        body = await serialize_questions(state=state, questions=questions)
        response = await client.send_body(request_id=f'synthetic:{packet["id"]}', body=body)
        if set(response['answers']) != set(questions):
            raise ValueError('atomic packet response questions mismatch')
        answers = []
        for question in packet['questions']:
            raw = response['answers'][question['id']]['probabilities']
            # Validate the same rounded categorical contract before selecting a maximum.
            sample_categorical(raw, 0.5, tuple(question['criteria']))
            best = max(question['criteria'], key=lambda key: (raw[key], -list(question['criteria']).index(key)))
            answers.append({'id': question['id'], 'selected': best,
                            'expected': question['expected_choice'], 'identifiable': question['identifiable']})
        return {'id': packet['id'], 'response': response, 'answers': answers}
    subdirs = [output / name for name in ('complete', 'numeric', 'atomic')]
    for path in subdirs:
        path.mkdir()
    complete = await _batches(portfolios, portfolio, subdirs[0])
    numbers = [(c, encoding, r) for c in numeric_cases() for encoding in ('decimal', 'interval') for r in range(2)]
    numeric = await _batches(numbers, lambda spec: numeric_sample(client, *spec), subdirs[1])
    atomic_results = await _batches(atomic_packets(), atomic, subdirs[2])
    summary = {'status': 'COMPLETE', 'complete_portfolios': len(complete), 'numeric_predictions': len(numeric),
               'atomic_packets': len(atomic_results)}
    summary['file_sha256'] = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(output.glob('*/*.json'))}
    _save(output / 'terminal.json', _json(summary))
    return summary


def parse_opus_choices(stdout: bytes, packet):
    events = [strict_json(line) for line in stdout.splitlines() if line.strip().startswith(b'{')]
    results = [row for row in events if row.get('type') == 'result']
    if not results:
        raise ValueError('Opus result event missing')
    text = results[-1].get('result', '')
    possible = [text] + re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    expected = {q['id']: q['criteria'] for q in packet['questions']}
    for candidate in possible:
        try:
            value = strict_json(candidate.encode())
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and set(value) == set(expected) and all(value[k] in expected[k] for k in expected):
            return value
    raise ValueError('Opus choice result is not a complete legal four-question object')


async def run_opus_diagnostics(preview: Path, output: Path):
    """Use the installed helper without changing its account, auth, tools or model."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    # Serial calls make consultation provenance and a failed-call stop unambiguous.
    for packet in atomic_packets():
        result_dir = output / packet['id']
        # This is the installed credential-expiry preflight for an actual
        # authorized consultation, not the user-paused quota/routing workflow.
        auth = await asyncio.create_subprocess_exec('ssh', '-o', 'BatchMode=yes', '-o',
            'ConnectTimeout=10', 'claude-vps', 'python3', '-', stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            auth_stdout, _ = await asyncio.wait_for(auth.communicate(AUTH_PREFLIGHT.read_bytes()), 40)
        except TimeoutError:
            auth.kill()
            await auth.communicate()
            raise ValueError('Claude expiry preflight timed out; no consultation started')
        auth_status = strict_json(auth_stdout)
        _save(output / f"{packet['id']}.auth.json", _json(auth_status))
        if auth.returncode or auth_status.get('status') not in ('valid', 'refreshed'):
            raise ValueError('Claude expiry preflight failed; no consultation started')
        proc = await asyncio.create_subprocess_exec(sys.executable, str(HELPER), 'consult',
            '--request', str(preview / f"{packet['id']}.txt"), '--output-dir', str(result_dir),
            '--role', 'architect', '--timeout', '180', '--max-budget-usd', '1.0',
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        _save(output / f"{packet['id']}.helper-status.json", _json({'returncode': proc.returncode,
            'stdout_sha256': hashlib.sha256(stdout).hexdigest(), 'stderr_sha256': hashlib.sha256(stderr).hexdigest()}))
        if proc.returncode:
            raise ValueError('Opus consultation failed; automatic retry disabled')
        receipt = strict_json((result_dir / 'receipt.json').read_bytes())
        if (receipt.get('status') != 'complete' or receipt.get('canonical_model') != 'claude-opus-5-5'
                or 'claude-opus-5-5' not in receipt.get('model_usage_keys', [])
                or receipt.get('api_key_source') != 'none' or receipt.get('tools') != []
                or receipt.get('mcp_servers') != []):
            raise ValueError('Opus identity or restricted-channel receipt invalid')
        choices = parse_opus_choices((result_dir / 'stdout.log').read_bytes(), packet)
        row = {'packet_id': packet['id'], 'choices': choices,
               'list_cost_estimate_usd': receipt.get('total_cost_usd')}
        _save(result_dir / 'choices.json', _json(row))
        rows.append(row)
    _save(output / 'terminal.json', _json({'status': 'COMPLETE', 'consultations': len(rows),
        'file_sha256': {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(output.rglob('*')) if p.is_file()}}))
    return rows


def verify_diagnostics(preview: Path, human_preview: Path):
    frozen = strict_json((preview / 'manifest.json').read_bytes())
    verify_preview(human_preview, frozen['human_preview_sha256'])
    if (frozen['fixture_sha256'] != fixture_digest()
            or frozen['opus_helper_sha256'] != hashlib.sha256(HELPER.read_bytes()).hexdigest()
            or frozen['opus_auth_preflight_sha256'] != hashlib.sha256(AUTH_PREFLIGHT.read_bytes()).hexdigest()):
        raise ValueError('synthetic fixtures or installed helper changed after freeze')
    expected = {'complete': complete_cases(), 'numeric': numeric_cases(), 'atomic': atomic_packets()}
    if (preview / 'fixtures.json').read_bytes() != _json(expected):
        raise ValueError('saved diagnostic fixtures changed')
    for packet in atomic_packets():
        request = ATOMIC_INSTRUCTION + '\nReturn a JSON object mapping every question ID to its chosen candidate ID.\n' + json.dumps(public_packet(packet), ensure_ascii=False)
        if (preview / f"{packet['id']}.txt").read_bytes() != request.encode():
            raise ValueError('Opus public synthetic request changed')
    return frozen


def _verify_receipt(receipt, request_bytes):
    if (receipt.get('status') != 'complete' or receipt.get('canonical_model') != 'claude-opus-5-5'
            or 'claude-opus-5-5' not in receipt.get('model_usage_keys', [])
            or receipt.get('api_key_source') != 'none' or receipt.get('tools') != []
            or receipt.get('mcp_servers') != []
            or receipt.get('request_sha256') != hashlib.sha256(request_bytes).hexdigest()
            or receipt.get('role') != 'architect' or receipt.get('timeout_seconds') != 180):
        raise ValueError('Opus receipt differs from the frozen request and channel')


def _check_output_hashes(output, terminal):
    hashes = terminal.get('file_sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('terminal lacks evidence hashes')
    for relative, digest in hashes.items():
        path = (output / relative).resolve()
        if output.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('diagnostic evidence hash or path differs')


def _atomic_summary(rows):
    groups = {}
    for row in rows:
        key = f"family_{row['family']}_condition_{row['condition']}"
        group = groups.setdefault(key, {'identifiable_n': 0, 'correct': 0,
                                       'negative_controls': 0, 'negative_cannot_identify': 0})
        if row['identifiable']:
            group['identifiable_n'] += 1
            group['correct'] += row['selected'] == row['expected']
        else:
            group['negative_controls'] += 1
            group['negative_cannot_identify'] += row['cannot_identify']
    return groups


def replay_opus_diagnostics(preview: Path, output: Path):
    terminal = strict_json((output / 'terminal.json').read_bytes())
    if terminal.get('status') != 'COMPLETE' or terminal.get('consultations') != 48:
        raise ValueError('Opus comparison is incomplete')
    _check_output_hashes(output, terminal)
    rows, costs = [], []
    for packet in atomic_packets():
        folder = output / packet['id']
        request = (preview / f"{packet['id']}.txt").read_bytes()
        receipt = strict_json((folder / 'receipt.json').read_bytes())
        _verify_receipt(receipt, request)
        choices = parse_opus_choices((folder / 'stdout.log').read_bytes(), packet)
        saved = strict_json((folder / 'choices.json').read_bytes())
        if (saved.get('packet_id') != packet['id'] or saved.get('choices') != choices
                or saved.get('list_cost_estimate_usd') != receipt.get('total_cost_usd')):
            raise ValueError('Opus saved choices do not reconstruct from raw output')
        cost = receipt.get('total_cost_usd')
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            costs.append(cost)
        for q in packet['questions']:
            rows.append({'id': q['id'], 'base_id': q['base_id'], 'repeat': packet['repeat'],
                'family': q['family'], 'condition': q['condition'], 'selected': choices[q['id']],
                'expected': q['expected_choice'], 'identifiable': q['identifiable'],
                'cannot_identify': 'Cannot identify' in q['criteria'][choices[q['id']]]})
    return {'status': 'OPUS_REPLAY_VALID', 'consultations': 48, 'question_responses': 192,
            'list_cost_estimate_usd': sum(costs) if len(costs) == 48 else None,
            'subscription_charge': 'not reported; membership quota, not list-cost debit',
            'scores': _atomic_summary(rows), 'rows': rows}


async def compare_atomic_diagnostics(preview: Path, jev_output: Path, opus_output: Path):
    opus = replay_opus_diagnostics(preview, opus_output)
    terminal = strict_json((jev_output / 'terminal.json').read_bytes())
    if terminal.get('status') != 'COMPLETE' or terminal.get('atomic_packets') != 48:
        raise ValueError('Jev diagnostic comparison is incomplete')
    _check_output_hashes(jev_output, terminal)
    client = _RecordedClient(jev_output.with_name(jev_output.name + '.wire'))
    rows = []
    for index, packet in enumerate(atomic_packets()):
        saved = strict_json((jev_output / 'atomic' / f'{index:04d}.json').read_bytes())
        state, questions = packet_request(packet)
        response = await client.send_body(request_id=f'synthetic:{packet["id"]}',
                                          body=await serialize_questions(state=state, questions=questions))
        if response != saved['response']:
            raise ValueError('Jev atomic evidence differs')
        for question in packet['questions']:
            raw = response['answers'][question['id']]['probabilities']
            sample_categorical(raw, .5, tuple(question['criteria']))
            choice = max(question['criteria'], key=lambda key: raw[key])
            rows.append({'id': question['id'], 'base_id': question['base_id'], 'repeat': packet['repeat'],
                'family': question['family'], 'condition': question['condition'], 'selected': choice,
                'expected': question['expected_choice'], 'identifiable': question['identifiable'],
                'cannot_identify': 'Cannot identify' in question['criteria'][choice]})
    opus_by_key = {(r['id'], r['repeat']): r for r in opus['rows']}
    paired = {'both_correct': 0, 'jev_only_correct': 0, 'opus_only_correct': 0, 'neither_correct': 0,
              'negative_controls_excluded': 0}
    for jev in rows:
        other = opus_by_key[(jev['id'], jev['repeat'])]
        if not jev['identifiable']:
            paired['negative_controls_excluded'] += 1
            continue
        j, o = jev['selected'] == jev['expected'], other['selected'] == jev['expected']
        paired['both_correct' if j and o else 'jev_only_correct' if j else 'opus_only_correct' if o else 'neither_correct'] += 1
    return {'status': 'PAIRED_SYNTHETIC_COMPARISON', 'paired_counts': paired,
        'jev_scores': _atomic_summary(rows), 'opus_scores': opus['scores'],
        'opus_consultations': 48, 'opus_list_cost_estimate_usd': opus['list_cost_estimate_usd'],
        'limitations': ['Only synthetic specified-policy and arithmetic questions are compared.',
            'Opus uses the existing adviser wrapper and high-effort Claude Code channel; Jev uses atomic client argmax.',
            'Facts and candidates match, but this is not a pure model-only intervention; no Opus probabilities are invented.',
            'Ambiguous controls are excluded from action-accuracy denominators; repeats do not add independent tasks.']}


class _RecordedClient:
    def __init__(self, root):
        self.root = Path(root)
        self.used = set()

    async def choose(self, *, request_id, state, instructions, criteria):
        return await self.send_body(request_id=request_id,
            body=await serialize_body(state=state, instructions=instructions, criteria=criteria))

    async def send_body(self, *, request_id, body):
        if request_id in self.used:
            raise ValueError('duplicate replay request')
        self.used.add(request_id)
        folder = self.root / hashlib.sha256(request_id.encode()).hexdigest()
        if (folder / 'request.json').read_bytes() != body:
            raise ValueError('reconstructed synthetic wire differs')
        response = (folder / 'response.json').read_bytes()
        raw = strict_json(response)
        return {**raw, '_wire': {'request_id': request_id, 'directory': str(folder.resolve()),
            'request_sha256': hashlib.sha256(body).hexdigest(), 'response_sha256': hashlib.sha256(response).hexdigest(),
            'timing_sha256': hashlib.sha256((folder / 'timing.json').read_bytes()).hexdigest(),
            'attempt_sha256': hashlib.sha256((folder / 'attempt.json').read_bytes()).hexdigest()}}


async def replay_jev_diagnostics(output: Path, wire_root: Path):
    terminal = strict_json((output / 'terminal.json').read_bytes())
    for name, digest in terminal['file_sha256'].items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError('diagnostic result evidence changed')
    client = _RecordedClient(wire_root)
    score = {'complete': {}, 'numeric': {}, 'atomic': {'correct': 0, 'identifiable_n': 0,
             'negative_controls': 0, 'negative_cannot_identify': 0}}
    complete_by_id = {row['id']: row for row in complete_cases()}
    for path in sorted((output / 'complete').glob('*.json')):
        saved = strict_json(path.read_bytes())
        case = complete_by_id[saved['case_id']]
        row = complete_row(case, saved['interface'], saved['repeat'], oracle=saved['oracle'])
        result = await sample_portfolio(client, row, seed=SEED, category='synthetic',
            oracle_order_count=len(case['expected_orders']) if saved['oracle'] else None)
        if result != saved['result'] or row != saved['row'] or saved['expected'] != case['expected_orders']:
            raise ValueError('complete diagnostic semantic replay failed')
        key = 'oracle_N' if saved['oracle'] else saved['interface']
        stats = score['complete'].setdefault(key, {'correct': 0, 'n': 0})
        canonical = lambda orders: Counter((o['side'], o['qty'], o['price']) for o in orders)
        stats['correct'] += canonical(result['orders']) == canonical(case['expected_orders'])
        stats['n'] += 1
    numeric_by_id = {row['id']: row for row in numeric_cases()}
    for path in sorted((output / 'numeric').glob('*.json')):
        saved = strict_json(path.read_bytes())
        result = await numeric_sample(client, numeric_by_id[saved['id']], saved['encoding'], saved['repeat'])
        if result != saved:
            raise ValueError('numeric diagnostic semantic replay failed')
        stats = score['numeric'].setdefault(saved['encoding'], {'correct': 0, 'n': 0, 'absolute_error_sum': 0})
        stats['correct'] += result['value'] == result['expected']
        stats['n'] += 1
        stats['absolute_error_sum'] += abs(result['value'] - result['expected'])
    for index, packet in enumerate(atomic_packets()):
        saved = strict_json((output / 'atomic' / f'{index:04d}.json').read_bytes())
        state, questions = packet_request(packet)
        response = await client.send_body(request_id=f'synthetic:{packet["id"]}',
                                          body=await serialize_questions(state=state, questions=questions))
        if saved['response'] != response:
            raise ValueError('atomic response evidence differs')
        for item, question in zip(saved['answers'], packet['questions'], strict=True):
            probabilities = response['answers'][question['id']]['probabilities']
            sample_categorical(probabilities, 0.5, tuple(question['criteria']))
            selected = max(question['criteria'], key=lambda key: probabilities[key])
            if item['selected'] != selected or item['expected'] != question['expected_choice']:
                raise ValueError('atomic selected answer differs')
            if question['identifiable']:
                score['atomic']['identifiable_n'] += 1
                score['atomic']['correct'] += selected == question['expected_choice']
            else:
                score['atomic']['negative_controls'] += 1
                score['atomic']['negative_cannot_identify'] += 'Cannot identify' in question['criteria'][selected]
    if (sum(v['n'] for v in score['complete'].values()) != 160 or
            sum(v['n'] for v in score['numeric'].values()) != 96 or len(client.used) > 5952):
        raise ValueError('synthetic execution cardinality or request cap differs')
    return {'status': 'SYNTHETIC_REPLAY_VALID', 'physical_requests': len(client.used), 'scores': score}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preview', 'run-jev', 'run-opus', 'replay', 'replay-opus', 'compare'))
    parser.add_argument('--human-preview', required=True, type=Path)
    parser.add_argument('--preview', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--human-run', type=Path)
    parser.add_argument('--jev-output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'preview':
        result = freeze_diagnostics(args.human_preview, args.preview)
    else:
        manifest = verify_diagnostics(args.preview, args.human_preview)
        if args.output is None:
            parser.error('--output required')
        if args.command == 'run-opus':
            result = {'status': 'COMPLETE', 'consultations': len(asyncio.run(run_opus_diagnostics(args.preview, args.output)))}
        elif args.command == 'replay':
            result = asyncio.run(replay_jev_diagnostics(args.output, args.output.with_name(args.output.name + '.wire')))
        elif args.command == 'replay-opus':
            full = replay_opus_diagnostics(args.preview, args.output)
            result = {k: v for k, v in full.items() if k != 'rows'}
        elif args.command == 'compare':
            if args.jev_output is None:
                parser.error('--jev-output required for paired comparison')
            result = asyncio.run(compare_atomic_diagnostics(args.preview, args.jev_output, args.output))
        else:
            if args.human_run is None:
                parser.error('--human-run required to protect the primary study budget')
            from .runner import replay_run
            real = replay_run(args.human_preview, args.human_run)
            if real.get('mode') != 'provider' or real.get('sample_n') != 680:
                raise ValueError('complete valid real human study required before synthetic Jev spending')
            from .cli import _attestation
            from .transport import CostLedger, JevWireClient
            from capclaims.backends.jev import load_local_env
            import os
            root = Path(__file__).resolve().parents[3]
            human_manifest = strict_json((args.human_preview / 'manifest.json').read_bytes())
            ledger = CostLedger(root / 'artifacts/participant-data/jev-conditional-v1/cost-ledger.sqlite',
                                preview_hash=manifest['human_preview_sha256'])
            attestation, verifier = _attestation(args.human_preview / 'cost-attestation.json', human_manifest, ledger)
            if not verifier(attestation, human_manifest):
                raise ValueError('synthetic cost attestation verification failed')
            load_local_env()
            async def execute():
                async with JevWireClient(api_key=os.environ['TYPESAFE_API_KEY'], ledger=ledger,
                    evidence_dir=args.output.with_name(args.output.name + '.wire'), category='synthetic') as client:
                    return await run_jev_diagnostics(client, args.output)
            result = asyncio.run(execute())
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
