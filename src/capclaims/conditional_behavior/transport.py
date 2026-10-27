"""Exact SDK wire encoding, durable attempt evidence and rolling cost reservation.

Money is stored as integer nanodollars: each input token costs 42 nanodollars.
SQLite transactions protect the same study ledger across processes and batches.
The live HTTP client posts captured SDK bytes exactly once with retries disabled.
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

MODEL = 'jev-1.14.0'
MAX_INPUT_TOKENS = 65_536
NANODOLLARS_PER_TOKEN = 42
RESERVATION = MAX_INPUT_TOKENS * NANODOLLARS_PER_TOKEN
BUDGET = 20_000_000_000
MAX_BODY = 12_000


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                      sort_keys=True, allow_nan=False).encode('utf-8')


def strict_json(data: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON field')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('nonfinite JSON constant')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)


def _save(path: Path, data: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class CostLedger:
    """An append-only-in-purpose study ledger; unknown sends retain full reserves."""

    def __init__(self, path: str | Path, *, preview_hash: str,
                 budget_nanodollars: int = BUDGET):
        if (type(budget_nanodollars) is not int or not 0 < budget_nanodollars <= BUDGET
                or len(preview_hash) != 64 or any(c not in '0123456789abcdef' for c in preview_hash)):
            raise ValueError('invalid cost contract or preview hash')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (request_id TEXT PRIMARY KEY, category TEXT NOT NULL, '
                       'state TEXT NOT NULL, committed INTEGER NOT NULL, input_tokens INTEGER, '
                       'created_utc TEXT NOT NULL, detail TEXT)')
            config = dict(db.execute('SELECT key,value FROM config'))
            wanted = {'preview_hash': preview_hash, 'budget': str(budget_nanodollars),
                      'model': MODEL, 'reservation': str(RESERVATION)}
            if config and config != wanted:
                raise ValueError('cost ledger belongs to a different preview or cost contract')
            if not config:
                db.executemany('INSERT INTO config VALUES (?,?)', wanted.items())

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.execute('PRAGMA synchronous=FULL')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def reserve(self, request_id: str, category: str = 'human') -> None:
        if not isinstance(request_id, str) or not request_id or category not in ('human', 'synthetic'):
            raise ValueError('invalid attempt identity or category')
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM attempts WHERE request_id=?', (request_id,)).fetchone():
                raise ValueError('attempt already exists; automatic retry is forbidden')
            count, committed = db.execute('SELECT COUNT(*),COALESCE(SUM(committed),0) FROM attempts').fetchone()
            category_count = db.execute('SELECT COUNT(*) FROM attempts WHERE category=?', (category,)).fetchone()[0]
            category_cap = 23936 if category == 'human' else 5952
            if count >= 29888 or category_count >= category_cap:
                raise ValueError('physical request budget exhausted')
            budget = int(db.execute("SELECT value FROM config WHERE key='budget'").fetchone()[0])
            if committed + RESERVATION > budget:
                raise ValueError('study cost budget exhausted before dispatch')
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',
                       (request_id, category, 'reserved', RESERVATION, None,
                        datetime.now(timezone.utc).isoformat(), None))

    def settle(self, request_id: str, input_tokens: int) -> None:
        if type(input_tokens) is not int or not 0 <= input_tokens <= MAX_INPUT_TOKENS:
            raise ValueError('authoritative usage violates the frozen token bound')
        with self._transaction() as db:
            row = db.execute('SELECT state FROM attempts WHERE request_id=?', (request_id,)).fetchone()
            if row != ('reserved',):
                raise ValueError('only a reserved attempt may settle once')
            db.execute('UPDATE attempts SET state=?,committed=?,input_tokens=? WHERE request_id=?',
                       ('settled', input_tokens * NANODOLLARS_PER_TOKEN, input_tokens, request_id))

    def mark_unknown(self, request_id: str, detail: str) -> None:
        with self._transaction() as db:
            db.execute("UPDATE attempts SET state='unknown',detail=? WHERE request_id=? AND state='reserved'",
                       (detail, request_id))

    def summary(self) -> dict[str, Any]:
        with self._transaction() as db:
            rows = list(db.execute('SELECT state,COUNT(*),SUM(committed) FROM attempts GROUP BY state'))
        counts = {state: n for state, n, _ in rows}
        sums = {state: value for state, _, value in rows}
        return {'physical_requests': sum(counts.values()), 'states': counts,
                'committed_nanodollars': sum(sums.values()),
                'settled_nanodollars': sums.get('settled', 0),
                'reserved_nanodollars': sums.get('reserved', 0) + sums.get('unknown', 0)}


async def serialize_body(*, state: Any, instructions: str, criteria: dict[str, str]) -> bytes:
    return await serialize_questions(state=state, questions={
        'decision': {'instructions': instructions, 'criteria': criteria}})


async def serialize_questions(*, state: Any, questions: dict[str, dict[str, Any]]) -> bytes:
    """Obtain the installed frozen SDK's exact bytes using only a mock transport."""
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient, Choice
    from typesafe_sdk._core.retry import RetryPolicy
    if importlib.metadata.version('typesafe-sdk') != '0.7.5':
        raise ValueError('frozen study requires typesafe-sdk 0.7.5')
    if not isinstance(questions, dict) or not questions:
        raise ValueError('questions must be nonempty')
    for question in questions.values():
        if not isinstance(question.get('criteria'), dict) or not 1 <= len(question['criteria']) <= 255:
            raise ValueError('Choice must contain 1..255 candidates')
    captured = []
    async def capture(request):
        captured.append(await request.aread())
        answers = {}
        for name, question in questions.items():
            criteria = question['criteria']
            first = next(iter(criteria))
            answers[name] = {'type': 'choice', 'choice': first, 'confidence': 1.0,
                'probabilities': {key: float(key == first) for key in criteria}}
        return httpx2.Response(200, json={'model': MODEL, 'usage': {'input_tokens': 1, 'output_tokens': 1},
            'answers': answers})
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(capture), follow_redirects=False)
    client = AsyncTypeSafeClient(api_key='offline-dummy', model=MODEL,
        retry=RetryPolicy(max_retries=0), http_client=http,
        base_url='https://api.typesafe.ai', compression=False)
    try:
        await client.system_one(state=state, questions={name: Choice(
            instructions=question['instructions'], criteria=question['criteria'])
            for name, question in questions.items()}, model=MODEL)
    finally:
        await client.aclose()
    if len(captured) != 1 or len(captured[0]) > MAX_BODY:
        raise ValueError('complete serialized request exceeds 12000 bytes or is not singular')
    return captured[0]


class JevWireClient:
    """Shared 10-request asynchronous transport; callers own dependent decision order."""

    def __init__(self, *, api_key: str, ledger: CostLedger, evidence_dir: str | Path,
                 transport: Any = None, category: str = 'human'):
        import httpx2
        if not api_key or category not in ('human', 'synthetic'):
            raise ValueError('API key and valid request category required')
        self.ledger, self.category = ledger, category
        self.root = Path(evidence_dir)
        self.root.mkdir(parents=True, exist_ok=False)
        self.http = httpx2.AsyncClient(transport=transport or httpx2.AsyncHTTPTransport(retries=0),
            follow_redirects=False, timeout=60.0, trust_env=False,
            headers={'Authorization': 'Bearer ' + api_key, 'Content-Type': 'application/json',
                     'Accept': 'application/json'})
        self.semaphore = asyncio.Semaphore(10)
        self.rate_lock = asyncio.Lock()
        self.times = deque()
        self.halted = False
        self.active = self.peak = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def _rate_slot(self):
        # Caller holds rate_lock through durable writes and records the actual
        # forwarding time, rather than reserving an earlier permission time.
        while True:
            now = time.monotonic()
            while self.times and now - self.times[0] >= 1.0:
                self.times.popleft()
            if len(self.times) < 40 and (len(self.times) + 1) * MAX_BODY <= 50_000:
                return
            await asyncio.sleep(max(0.001, 1.001 - (now - self.times[0])))

    async def choose(self, *, request_id: str, state: Any, instructions: str,
                     criteria: dict[str, str]) -> dict[str, Any]:
        body = await serialize_body(state=state, instructions=instructions, criteria=criteria)
        return await self.send_body(request_id=request_id, body=body)

    async def send_body(self, *, request_id: str, body: bytes) -> dict[str, Any]:
        if not isinstance(body, bytes) or len(body) > MAX_BODY:
            raise ValueError('complete body must fit 12000 bytes')
        parsed = strict_json(body)
        if parsed.get('model') != MODEL or 'assets' in parsed:
            raise ValueError('request does not use pinned text-only model')
        async with self.semaphore:
            if self.halted:
                raise ValueError('transport halted after an ambiguous or failed attempt')
            async with self.rate_lock:
                await self._rate_slot()
                if self.halted:
                    raise ValueError('transport halted before dispatch')
                self.ledger.reserve(request_id, self.category)
                folder = self.root / hashlib.sha256(request_id.encode()).hexdigest()
                folder.mkdir(exist_ok=False)
                _save(folder / 'request.json', body)
                _save(folder / 'attempt.json', _json({'request_id': request_id, 'request_sha256': hashlib.sha256(body).hexdigest(),
                    'method': 'POST', 'url': 'https://api.typesafe.ai/v1/systemone', 'body_bytes': len(body),
                    'reserved_bytes': MAX_BODY, 'reserved_nanodollars': RESERVATION,
                    'utc': datetime.now(timezone.utc).isoformat(),
                    'long_context': False, 'compression': False, 'retry_count': 0}))
                start = time.monotonic()
                self.times.append(start)
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                response = await self.http.post('https://api.typesafe.ai/v1/systemone', content=body)
                _save(folder / 'response.json', response.content)
                _save(folder / 'timing.json', _json({'status_code': response.status_code,
                    'started_monotonic': start, 'finished_monotonic': time.monotonic(),
                    'elapsed_seconds': time.monotonic() - start}))
                if response.status_code != 200:
                    raise ValueError('provider returned unsuccessful HTTP status')
                result = strict_json(response.content)
                if result.get('model') != MODEL:
                    raise ValueError('provider returned unexpected model')
                usage = result.get('usage', {}).get('input_tokens')
                output_tokens = result.get('usage', {}).get('output_tokens')
                if type(output_tokens) is not int or output_tokens < 0:
                    raise ValueError('provider output usage is malformed')
                self.ledger.settle(request_id, usage)
                return {**result, '_wire': {'request_id': request_id, 'directory': str(folder.resolve()),
                    'request_sha256': hashlib.sha256(body).hexdigest(),
                    'response_sha256': hashlib.sha256(response.content).hexdigest(),
                    'timing_sha256': hashlib.sha256((folder / 'timing.json').read_bytes()).hexdigest(),
                    'attempt_sha256': hashlib.sha256((folder / 'attempt.json').read_bytes()).hexdigest()}}
            except BaseException as exc:
                self.halted = True
                self.ledger.mark_unknown(request_id, type(exc).__name__)
                _save(folder / 'failure.json', _json({'failure_type': type(exc).__name__,
                    'started_monotonic': start, 'finished_monotonic': time.monotonic()}))
                raise
            finally:
                self.active -= 1
