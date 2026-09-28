import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

KEY = "TESTKEY123SECRET"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
T0 = datetime(2026, 9, 28, 18, 0, tzinfo=UTC)


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class FakeClock:
    def __init__(self, now: datetime = T0):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


class FakeAV:
    """MockTransport handler. bodies: function -> dict | httpx.Response | Exception."""

    def __init__(self, **bodies):
        self.bodies = bodies
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        out = self.bodies[request.url.params["function"]]
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, httpx.Response) else httpx.Response(200, json=out)

    def count(self, function: str) -> int:
        return sum(1 for r in self.calls if r.url.params["function"] == function)
