import asyncio
import json
import hashlib

import pytest

from capclaims.conditional_behavior.diagnostics import (complete_row, numeric_sample,
    packet_request, parse_opus_choices, replay_opus_diagnostics)
from capclaims.conditional_behavior.sampling import sample_portfolio
from capclaims.conditional_behavior.synthetic import complete_cases, numeric_cases, atomic_packets
from capclaims.conditional_behavior.transport import serialize_questions


class OracleClient:
    def __init__(self, expected=None):
        self.expected = expected
        self.calls = 0
        self.instructions = []

    async def choose(self, *, request_id, state, instructions, criteria):
        self.calls += 1
        self.instructions.append(instructions)
        if self.expected is not None:
            field = state['decision_field']
            slot = state.get('slot', 0)
            if field == 'order_count':
                selected = 'N' + str(len(self.expected))
            elif field == 'side':
                selected = 'STOP' if slot >= len(self.expected) else self.expected[slot]['side'].upper()
            else:
                integer = self.expected[slot]['qty' if field == 'quantity' else 'price']
                digit = f'{integer:03d}'[len(state['numeric_prefix'])]
                selected = 'digit' + digit
        else:
            integer = state['a'] - state['b']
            if 'b0' in criteria:
                selected = next(k for k, v in criteria.items() if int(v.split('..')[0]) <= integer <= int(v.split('..')[1]))
            elif any(k.startswith('n') for k in criteria):
                selected = f'n{integer}'
            else:
                selected = 'digit' + f'{integer:03d}'[len(state['numeric_prefix'])]
        assert selected in criteria
        return {'model': 'jev-1.14.0', 'usage': {'input_tokens': 1, 'output_tokens': 1},
            'answers': {'decision': {'type': 'choice', 'choice': selected, 'confidence': 1,
                'probabilities': {key: float(key == selected) for key in criteria}}}}


def test_every_complete_diagnostic_reaches_oracle_through_each_legal_protocol():
    async def exercise():
        for case in complete_cases():
            for interface, oracle in (('J1', False), ('J2', False), ('J2', True)):
                client = OracleClient(case['expected_orders'])
                result = await sample_portfolio(client, complete_row(case, interface, 0, oracle=oracle),
                    seed=20261026, category='synthetic',
                    oracle_order_count=len(case['expected_orders']) if oracle else None)
                assert result['orders'] == case['expected_orders']
                assert client.calls <= (36 if interface == 'J2' and not oracle else 35)
                n = len(case['payload']['history'])
                profits = sum('final_profit_dollars' in h for h in case['payload']['history'])
                assert all(f'contains {n} historical example(s)' in text and
                    f'profit is present in {profits} of those examples' in text for text in client.instructions)
                assert all('formal rounds 6 through 13' not in text and
                           'formal rounds 11 through 13' not in text for text in client.instructions)
    asyncio.run(exercise())


def test_every_numeric_diagnostic_reaches_expected_integer_in_both_encodings():
    async def exercise():
        for case in numeric_cases():
            for encoding, steps in (('decimal', 3), ('interval', 2)):
                client = OracleClient()
                result = await numeric_sample(client, case, encoding, 0)
                assert result['value'] == case['expected_integer'] and client.calls == steps
    asyncio.run(exercise())


def test_all_atomic_packets_fit_wire_limit_and_keep_labels_out():
    async def exercise():
        for packet in atomic_packets():
            state, questions = packet_request(packet)
            body = await serialize_questions(state=state, questions=questions)
            assert len(body) <= 12000
            payload = json.loads(body)
            assert len(payload['questions']) == 4
            assert 'expected_choice' not in body.decode() and 'identifiable' not in body.decode()
    asyncio.run(exercise())


def test_opus_parser_requires_complete_legal_four_answer_object():
    packet = atomic_packets()[0]
    answer = {q['id']: next(iter(q['criteria'])) for q in packet['questions']}
    result = json.dumps({'type': 'result', 'result': '```json\n' + json.dumps(answer) + '\n```'}).encode()
    assert parse_opus_choices(result, packet) == answer
    answer.pop(next(iter(answer)))
    with pytest.raises(ValueError):
        parse_opus_choices(json.dumps({'type': 'result', 'result': json.dumps(answer)}).encode(), packet)


def test_opus_offline_replay_checks_all_receipts_hashes_and_excludes_ambiguous(tmp_path):
    preview, output = tmp_path / 'preview', tmp_path / 'opus'
    preview.mkdir()
    output.mkdir()
    for packet in atomic_packets():
        request = ('Synthetic frozen packet ' + packet['id']).encode()
        (preview / (packet['id'] + '.txt')).write_bytes(request)
        folder = output / packet['id']
        folder.mkdir()
        choices = {q['id']: q['expected_choice'] if q['identifiable'] else 'c' for q in packet['questions']}
        receipt = {'status': 'complete', 'canonical_model': 'claude-opus-5-5',
            'model_usage_keys': ['claude-opus-5-5'], 'api_key_source': 'none', 'tools': [], 'mcp_servers': [],
            'role': 'architect', 'timeout_seconds': 180, 'request_sha256': hashlib.sha256(request).hexdigest(),
            'total_cost_usd': .01}
        (folder / 'receipt.json').write_text(json.dumps(receipt))
        (folder / 'stdout.log').write_text(json.dumps({'type': 'result', 'result': json.dumps(choices)}))
        (folder / 'choices.json').write_text(json.dumps({'packet_id': packet['id'], 'choices': choices,
                                                      'list_cost_estimate_usd': .01}))
    terminal = {'status': 'COMPLETE', 'consultations': 48,
        'file_sha256': {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in output.rglob('*') if p.is_file()}}
    (output / 'terminal.json').write_text(json.dumps(terminal))
    result = replay_opus_diagnostics(preview, output)
    assert result['consultations'] == 48 and result['question_responses'] == 192
    assert sum(g['identifiable_n'] for g in result['scores'].values()) == 144
    assert sum(g['correct'] for g in result['scores'].values()) == 144
    assert sum(g['negative_controls'] for g in result['scores'].values()) == 48
    first_request = next(preview.glob('*.txt'))
    first_request.write_bytes(b'altered frozen request')
    with pytest.raises(ValueError, match='receipt'):
        replay_opus_diagnostics(preview, output)
