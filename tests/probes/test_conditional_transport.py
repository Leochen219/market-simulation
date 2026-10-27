import asyncio
import json

import pytest

from capclaims.conditional_behavior.transport import CostLedger, JevWireClient, serialize_body, strict_json


def test_ledger_reserves_before_settlement_and_persists_unknown_charges(tmp_path):
    path = tmp_path / 'ledger.sqlite'
    ledger = CostLedger(path, preview_hash='a' * 64)
    ledger.reserve('first', 'human')
    assert ledger.summary()['reserved_nanodollars'] == 2752512
    ledger.settle('first', 100)
    ledger.reserve('unknown', 'human')
    ledger.mark_unknown('unknown', 'timeout')
    again = CostLedger(path, preview_hash='a' * 64)
    assert again.summary()['committed_nanodollars'] == 4200 + 2752512
    with pytest.raises(ValueError, match='already'):
        again.reserve('first', 'human')
    with pytest.raises(ValueError, match='preview'):
        CostLedger(path, preview_hash='b' * 64)


def test_ledger_cannot_dispatch_past_budget_or_release_invalid_usage(tmp_path):
    ledger = CostLedger(tmp_path / 'budget.sqlite', preview_hash='a' * 64,
                        budget_nanodollars=2752512)
    ledger.reserve('one', 'human')
    with pytest.raises(ValueError, match='budget'):
        ledger.reserve('two', 'human')
    for usage in (True, -1, 65537, 1.5):
        with pytest.raises(ValueError):
            ledger.settle('one', usage)
    assert ledger.summary()['committed_nanodollars'] == 2752512
    ledger.settle('one', 0)
    ledger.reserve('two', 'human')


def test_wire_client_records_exact_sdk_body_usage_and_no_secret(tmp_path):
    import httpx2
    async def exercise():
        captured = []
        async def provider(request):
            captured.append(await request.aread())
            return httpx2.Response(200, json={'model': 'jev-1.14.0',
                'usage': {'input_tokens': 300, 'output_tokens': 20},
                'answers': {'decision': {'type': 'choice', 'choice': 'a',
                    'confidence': 0.6, 'probabilities': {'a': 0.6, 'b': 0.4}}}})
        ledger = CostLedger(tmp_path / 'cost.sqlite', preview_hash='c' * 64)
        async with JevWireClient(api_key='test-secret', ledger=ledger,
                evidence_dir=tmp_path / 'wire', transport=httpx2.MockTransport(provider)) as client:
            result = await client.choose(request_id='id', state={'Q': 1},
                instructions='Choose.', criteria={'a': 'A', 'b': 'B'})
        expected = await serialize_body(state={'Q': 1}, instructions='Choose.', criteria={'a': 'A', 'b': 'B'})
        assert captured == [expected]
        assert result['usage']['input_tokens'] == 300
        assert ledger.summary()['settled_nanodollars'] == 12600
        files = list((tmp_path / 'wire').rglob('*'))
        assert all(b'test-secret' not in p.read_bytes() for p in files if p.is_file())
    asyncio.run(exercise())


def test_failure_keeps_reservation_and_blocks_followup(tmp_path):
    import httpx2
    async def exercise():
        calls = 0
        async def provider(request):
            nonlocal calls
            calls += 1
            return httpx2.Response(200, json={'model': 'jev-1.14.0', 'answers': {}})
        ledger = CostLedger(tmp_path / 'cost.sqlite', preview_hash='d' * 64)
        async with JevWireClient(api_key='test-secret', ledger=ledger,
                evidence_dir=tmp_path / 'wire', transport=httpx2.MockTransport(provider)) as client:
            for ident in ('one', 'two'):
                with pytest.raises(ValueError):
                    await client.choose(request_id=ident, state={}, instructions='x', criteria={'a': 'A'})
        assert calls == 1
        assert ledger.summary()['committed_nanodollars'] == 2752512
    asyncio.run(exercise())


def test_ten_distinct_mock_requests_overlap_and_all_evidence_is_recorded(tmp_path):
    import httpx2
    async def exercise():
        arrived = set()
        barrier = asyncio.Event()
        async def provider(request):
            payload = json.loads(await request.aread())
            arrived.add(payload['state']['input_id'])
            if len(arrived) == 10:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), 8)
            return httpx2.Response(200, json={'model': 'jev-1.14.0',
                'usage': {'input_tokens': 100, 'output_tokens': 10},
                'answers': {'decision': {'type': 'choice', 'choice': 'a',
                    'confidence': 1, 'probabilities': {'a': 1}}}})
        ledger = CostLedger(tmp_path / 'cost.sqlite', preview_hash='e' * 64)
        async with JevWireClient(api_key='test-secret', ledger=ledger,
                evidence_dir=tmp_path / 'wire', transport=httpx2.MockTransport(provider)) as client:
            await asyncio.gather(*(client.choose(request_id=f'case-{i}', state={'input_id': i},
                instructions='Choose.', criteria={'a': 'A'}) for i in range(10)))
            assert client.peak == 10
        assert len(arrived) == 10
        attempts = [json.loads(p.read_text()) for p in (tmp_path / 'wire').glob('*/attempt.json')]
        starts = sorted(json.loads(p.read_text())['started_monotonic'] for p in (tmp_path / 'wire').glob('*/timing.json'))
        assert len(attempts) == 10
        assert all(sum(0 <= other - start < 1 for other in starts) <= 4 for start in starts)
        assert ledger.summary()['physical_requests'] == 10
    asyncio.run(exercise())


def test_concurrent_ledger_users_cannot_overcommit(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / 'shared.sqlite'
    CostLedger(path, preview_hash='f' * 64, budget_nanodollars=4 * 2752512)
    def attempt(index):
        ledger = CostLedger(path, preview_hash='f' * 64, budget_nanodollars=4 * 2752512)
        try:
            ledger.reserve(str(index), 'human')
            return True
        except ValueError as exc:
            assert 'budget' in str(exc)
            return False
    with ThreadPoolExecutor(max_workers=10) as pool:
        assert sum(pool.map(attempt, range(10))) == 4


@pytest.mark.parametrize('body', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_strict_json_rejects_ambiguous_provider_response(body):
    with pytest.raises(ValueError):
        strict_json(body)
