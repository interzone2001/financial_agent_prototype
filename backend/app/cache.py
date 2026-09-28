"""SQLite TTL cache for Alpha Vantage responses (spec §3.5: av_cache(key, body, fetched_at)).

One connection per operation: the graph runs nodes in parallel threads and sqlite3
connections must not be shared across threads.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


class SqliteCache:
    def __init__(self, path: Path, clock: Clock = utc_now):
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._execute(
            "CREATE TABLE IF NOT EXISTS av_cache "
            "(key TEXT PRIMARY KEY, body TEXT NOT NULL, fetched_at TEXT NOT NULL)"
        )

    def _execute(self, sql: str, params: tuple = ()) -> tuple | None:
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            with conn:  # commits on success
                return conn.execute(sql, params).fetchone()
        finally:
            conn.close()

    def get(self, key: str, ttl: timedelta) -> tuple[dict, datetime] | None:
        row = self._execute("SELECT body, fetched_at FROM av_cache WHERE key = ?", (key,))
        if row is None:
            return None
        fetched_at = datetime.fromisoformat(row[1])
        if self.clock() - fetched_at >= ttl:
            return None
        return json.loads(row[0]), fetched_at

    def set(self, key: str, body: dict, fetched_at: datetime) -> None:
        self._execute(
            "INSERT OR REPLACE INTO av_cache (key, body, fetched_at) VALUES (?, ?, ?)",
            (key, json.dumps(body), fetched_at.isoformat()),
        )
