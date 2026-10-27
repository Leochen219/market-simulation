"""Independent USD12 study ledger; reuse frozen exact-wire transport unchanged.

The shared wire implementation has the same pinned model, input token reservation,
12k body limit, 10-inflight/40RPS/50k reserved-byte rate caps and zero retries.
Only this ledger owns this study's spending and physical-attempt policy. No old
study globals or ledgers are changed. Values are integer nanodollars.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any

from capclaims.conditional_behavior.transport import (
    JevWireClient, MAX_BODY, MAX_INPUT_TOKENS, MODEL, NANODOLLARS_PER_TOKEN,
    RESERVATION, serialize_body, serialize_questions, strict_json,
)

BUDGET = 12_000_000_000
MAX_REQUESTS = 3_888
DEFAULT_LEDGER = (Path(__file__).resolve().parents[3] / 'artifacts' /
                  'participant-data' / 'jev-behavioral-memory-v1' / 'cost-ledger.sqlite')


class CostLedger:
    """Reserve before dispatch atomically; uncertain attempts retain full cost."""

    def __init__(self, path: str | Path, *, preview_hash: str,
                 budget_nanodollars: int = BUDGET):
        if (type(budget_nanodollars) is not int or not 0 < budget_nanodollars <= BUDGET
                or not isinstance(preview_hash, str) or len(preview_hash) != 64
                or any(c not in '0123456789abcdef' for c in preview_hash)):
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
                      'model': MODEL, 'reservation': str(RESERVATION),
                      'study': 'jev_behavioral_memory_v1', 'request_cap': str(MAX_REQUESTS)}
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
        if not isinstance(request_id, str) or not request_id or category != 'human':
            raise ValueError('invalid attempt identity or category for behavioral-memory study')
        with self._transaction() as db:
            if db.execute('SELECT 1 FROM attempts WHERE request_id=?', (request_id,)).fetchone():
                raise ValueError('attempt already exists; automatic retry is forbidden')
            count, committed = db.execute('SELECT COUNT(*),COALESCE(SUM(committed),0) FROM attempts').fetchone()
            if count >= MAX_REQUESTS:
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
                       (str(detail), request_id))

    def summary(self) -> dict[str, Any]:
        with self._transaction() as db:
            rows = list(db.execute('SELECT state,COUNT(*),SUM(committed) FROM attempts GROUP BY state'))
        counts = {state: n for state, n, _ in rows}
        sums = {state: value for state, _, value in rows}
        return {'physical_requests': sum(counts.values()), 'states': counts,
                'committed_nanodollars': sum(sums.values()),
                'settled_nanodollars': sums.get('settled', 0),
                'reserved_nanodollars': sums.get('reserved', 0) + sums.get('unknown', 0),
                'request_cap': MAX_REQUESTS, 'budget_nanodollars': BUDGET}
