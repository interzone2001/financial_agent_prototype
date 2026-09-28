"""Per-IP sliding-window limits for the endpoints that spend API quota (deploy hardening).

In-memory and per-process: fine for a single Railway replica, resets on redeploy.
Enabled by env: RATE_LIMIT_REPORTS_PER_HOUR / RATE_LIMIT_CHATS_PER_HOUR (unset or 0 = off).
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Request


class RateLimitedError(Exception):
    def __init__(self, what: str, limit: int):
        super().__init__(
            f"Limit reached: {limit} {what} per hour from your network. Please try again later."
        )


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 3600, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window_s, self._clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


def client_ip(request: Request) -> str:
    # Railway's edge proxy sets X-Forwarded-For; the first entry is the real client.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def limiter_dependency(env_var: str, what: str):
    """FastAPI dependency enforcing `env_var` requests/hour per client IP (no-op if unset)."""
    limit = int(os.getenv(env_var) or 0)
    limiter = RateLimiter(limit) if limit > 0 else None

    def check(request: Request) -> None:
        if limiter is not None and not limiter.allow(client_ip(request)):
            raise RateLimitedError(what, limit)

    return check
