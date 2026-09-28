"""Deploy hardening: per-IP hourly limits on the endpoints that spend API quota."""
from fastapi.testclient import TestClient

from app.api import create_app, stub_mode_deps
from app.ratelimit import RateLimiter


def test_limiter_window_and_per_key():
    t = [0.0]
    rl = RateLimiter(limit=2, window_s=3600, clock=lambda: t[0])
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a")
    assert rl.allow("b")                      # separate bucket per IP
    t[0] = 3601
    assert rl.allow("a")                      # window slid


def _client(monkeypatch, reports="2", chats="3"):
    monkeypatch.setenv("RATE_LIMIT_REPORTS_PER_HOUR", reports)
    monkeypatch.setenv("RATE_LIMIT_CHATS_PER_HOUR", chats)
    return TestClient(create_app(stub_mode_deps()))


def test_report_limit_returns_429_error_response(monkeypatch):
    c = _client(monkeypatch)
    tid = c.post("/api/sessions").json()["thread_id"]
    body = {"thread_id": tid, "ticker": "AAPL"}
    assert c.post("/api/report", json=body).status_code == 200
    assert c.post("/api/report", json=body).status_code == 200
    r = c.post("/api/report", json=body)
    assert r.status_code == 429 and r.json()["error_code"] == "rate_limited"
    assert "hour" in r.json()["message"]


def test_limit_keyed_on_forwarded_for(monkeypatch):
    c = _client(monkeypatch, reports="1")
    tid = c.post("/api/sessions").json()["thread_id"]
    body = {"thread_id": tid, "ticker": "AAPL"}
    h1, h2 = {"x-forwarded-for": "1.1.1.1, 10.0.0.1"}, {"x-forwarded-for": "2.2.2.2"}
    assert c.post("/api/report", json=body, headers=h1).status_code == 200
    assert c.post("/api/report", json=body, headers=h1).status_code == 429
    assert c.post("/api/report", json=body, headers=h2).status_code == 200


def test_no_limits_when_unset(monkeypatch):
    monkeypatch.delenv("RATE_LIMIT_REPORTS_PER_HOUR", raising=False)
    c = TestClient(create_app(stub_mode_deps()))
    tid = c.post("/api/sessions").json()["thread_id"]
    for _ in range(5):
        assert c.post("/api/report", json={"thread_id": tid, "ticker": "AAPL"}).status_code == 200


def test_serves_frontend_dist_when_configured(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<title>Security Brief</title>")
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path))
    c = TestClient(create_app(stub_mode_deps()))
    assert "Security Brief" in c.get("/").text
    assert c.get("/api/health").json() == {"status": "ok"}   # API still wins
