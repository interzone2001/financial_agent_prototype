"""Phase 0: stub API honours the §3.4 contract (WS4 must keep these passing)."""
from fastapi.testclient import TestClient

from app.api import create_app


def test_stub_api_contract():
    c = TestClient(create_app())
    assert c.get("/api/health").json() == {"status": "ok"}
    tid = c.post("/api/sessions").json()["thread_id"]
    assert c.post("/api/chat", json={"thread_id": tid, "message": "hi"}).status_code == 409
    r = c.post("/api/report", json={"thread_id": tid, "ticker": "aapl"})
    assert r.status_code == 200
    body = r.json()
    assert body["market"]["quote"]["source"]["provider"] == "alpha_vantage"
    assert body["filings"]["citation_index"] and body["disclaimer"]
    assert c.post("/api/report", json={"thread_id": tid, "ticker": "ZZZZ"}).json()["error_code"] == "ticker_not_found"
    assert c.post("/api/report", json={"thread_id": tid, "ticker": "1BAD"}).json()["error_code"] == "invalid_input"
    chat = c.post("/api/chat", json={"thread_id": tid, "message": "risk factors?"})
    assert chat.status_code == 200 and chat.json()["citations"]
