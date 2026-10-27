"""Behavioral-memory budget and real asynchronous wire acceptance (offline)."""
import asyncio
import json
import sqlite3

import pytest

from capclaims.behavioral_memory.transport import (
    BUDGET, MAX_REQUESTS, RESERVATION, CostLedger, JevWireClient, serialize_body,
)


def test_new_contract_and_unknown_attempts_remain_reserved(tmp_path):
    assert BUDGET == 12_000_000_000
    assert MAX_REQUESTS == 3888
    assert MAX_REQUESTS * RESERVATION == 10_701_766_656
    path = tmp_path / 'ledger.sqlite'
    ledger = CostLedger(path, preview_hash='a' * 64)
    ledger.reserve('one')
    ledger.mark_unknown('one', 'timeout')
    assert ledger.summary()['reserved_nanodollars'] == RESERVATION
    with pytest.raises(ValueError, match='reserved'):
        ledger.settle('one', 1)
    with pytest.raises(ValueError, match='retry'):
        ledger.reserve('one')
    with pytest.raises(ValueError, match='different'):
        CostLedger(path, preview_hash='b' * 64)
    with pytest.raises(ValueError):
        CostLedger(tmp_path / 'too-high.sqlite', preview_hash='a' * 64,
                   budget_nanodollars=BUDGET + 1)


def test_physical_attempt_cap_even_when_settled_usage_is_zero(tmp_path):
    ledger = CostLedger(tmp_path / 'ledger.sqlite', preview_hash='c' * 64)
    # Insert already-settled evidence to exercise the exact cap without 3888 fsyncs.
    with sqlite3.connect(ledger.path) as db:
        db.executemany('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',
            [(str(i), 'human', 'settled', 0, 0, '2026-10-26T00:00:00Z', None)
             for i in range(MAX_REQUESTS)])
    with pytest.raises(ValueError, match='physical request'):
        ledger.reserve('over-cap')
    assert ledger.summary()['physical_requests'] == MAX_REQUESTS


def test_atomic_budget_and_usage_validation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / 'ledger.sqlite'
    ledger = CostLedger(path, preview_hash='d' * 64, budget_nanodollars=4 * RESERVATION)
    def reserve(index):
        try:
            CostLedger(path, preview_hash='d' * 64,
                       budget_nanodollars=4 * RESERVATION).reserve(str(index))
            return True
        except ValueError as exc:
            assert 'budget' in str(exc)
            return False
    with ThreadPoolExecutor(max_workers=10) as pool:
        assert sum(pool.map(reserve, range(10))) == 4
    for usage in (True, -1, 65537, 1.5, None):
        with pytest.raises(ValueError):
            ledger.settle('0', usage)
    assert ledger.summary()['committed_nanodollars'] == 4 * RESERVATION


def test_ten_distinct_requests_overlap_with_exact_wire_and_durable_evidence(tmp_path):
    import httpx2
    async def exercise():
        arrived = set()
        captured = {}
        barrier = asyncio.Event()
        async def provider(request):
            body = await request.aread()
            ident = json.loads(body)['state']['input_id']
            arrived.add(ident)
            captured[ident] = body
            if len(arrived) == 10:
                barrier.set()
            await asyncio.wait_for(barrier.wait(), 10)
            return httpx2.Response(200, json={'model': 'jev-1.14.0',
                'usage': {'input_tokens': 100, 'output_tokens': 1},
                'answers': {'decision': {'type': 'choice', 'choice': 'a',
                    'confidence': 1, 'probabilities': {'a': 1}}}})
        ledger = CostLedger(tmp_path / 'ledger.sqlite', preview_hash='e' * 64)
        async with JevWireClient(api_key='test-secret-never-persist', ledger=ledger,
                evidence_dir=tmp_path / 'wire', transport=httpx2.MockTransport(provider)) as client:
            await asyncio.gather(*(client.choose(request_id=f'input-{i}',
                state={'input_id': i}, instructions='Choose.', criteria={'a': 'A'})
                for i in range(10)))
            assert client.peak == 10
        assert len(arrived) == 10
        assert captured[0] == await serialize_body(state={'input_id': 0},
                                                  instructions='Choose.', criteria={'a': 'A'})
        starts = sorted(json.loads(p.read_text())['started_monotonic']
                        for p in (tmp_path / 'wire').glob('*/timing.json'))
        assert len(starts) == 10
        assert all(sum(0 <= other - start < 1 for other in starts) <= 4 for start in starts)
        assert ledger.summary()['settled_nanodollars'] == 10 * 100 * 42
        assert all(b'test-secret-never-persist' not in p.read_bytes()
                   for p in (tmp_path / 'wire').rglob('*') if p.is_file())
    asyncio.run(exercise())


def test_uncertain_failure_halts_and_reserves_without_retry(tmp_path):
    import httpx2
    async def exercise():
        calls = 0
        async def provider(request):
            nonlocal calls
            calls += 1
            return httpx2.Response(503, json={'error': 'test'})
        ledger = CostLedger(tmp_path / 'ledger.sqlite', preview_hash='f' * 64)
        async with JevWireClient(api_key='offline-secret', ledger=ledger,
                evidence_dir=tmp_path / 'wire', transport=httpx2.MockTransport(provider)) as client:
            for ident in ('one', 'two'):
                with pytest.raises(ValueError):
                    await client.choose(request_id=ident, state={}, instructions='Choose.',
                                        criteria={'a': 'A'})
        assert calls == 1
        assert ledger.summary()['states'] == {'unknown': 1}
        assert ledger.summary()['reserved_nanodollars'] == RESERVATION
        assert len(list((tmp_path / 'wire').glob('*/failure.json'))) == 1
    asyncio.run(exercise())
