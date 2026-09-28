import uuid

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api import create_app
from app.models import DISCLAIMER, DataSourceError, RateLimitedError
from tests.ws4_orch.helpers import chat_llm, deps_with, raiser

AAPL = {"thread_id": "t1", "ticker": "AAPL"}


@pytest.fixture
def client(tmp_data_dir):
    return lambda llm=None, **o: TestClient(create_app(deps_with(llm, **o), InMemorySaver()))


def _err(resp, status, code):
    assert resp.status_code == status, resp.text
    assert resp.json()["error_code"] == code and resp.json()["message"]


def test_health_and_session(client):
    c = client()
    assert c.get("/api/health").json() == {"status": "ok"}
    uuid.UUID(c.post("/api/sessions").json()["thread_id"])


def test_report_happy_path(client):
    body = client().post("/api/report", json=AAPL).json()
    assert body["disclaimer"] == DISCLAIMER
    assert body["market"]["quote"]["source"]["url"].startswith("https://www.alphavantage.co")
    assert body["filings"]["citation_index"] and body["filings"]["risk_factors"]


def test_ticker_normalised(client):
    r = client().post("/api/report", json={"thread_id": "t1", "ticker": " aapl "})
    assert r.status_code == 200 and r.json()["ticker"] == "AAPL"


def test_rate_limited_market_is_partial_200(client):
    c = client(get_market_snapshot=raiser(RateLimitedError("alpha_vantage", "quota")))
    r = c.post("/api/report", json=AAPL)
    assert r.status_code == 200 and r.json()["market"] is None and r.json()["warnings"]


def test_error_codes(client):
    c = client()
    _err(c.post("/api/report", json={"thread_id": "t1", "ticker": "ZZZZ"}), 404, "ticker_not_found")
    _err(c.post("/api/report", json={"thread_id": "t1", "ticker": "123!"}), 422, "invalid_input")
    _err(c.post("/api/chat", json={"thread_id": "t1", "message": "x" * 1001}), 422, "invalid_input")
    _err(c.post("/api/chat", json={"thread_id": "t1", "message": "hi"}), 409, "no_report_yet")


def test_chat_after_report_and_thread_isolation(client):
    c = client(chat_llm("filings"))
    c.post("/api/report", json=AAPL)
    r = c.post("/api/chat", json={"thread_id": "t1", "message": "Main risks?"})
    assert r.status_code == 200 and r.json()["route"] == "filings"
    assert r.json()["disclaimer"] == DISCLAIMER
    _err(c.post("/api/chat", json={"thread_id": "t2", "message": "hi"}), 409, "no_report_yet")


def test_failed_report_does_not_poison_chat(client):
    llm = chat_llm("market")
    c = client(llm)
    c.post("/api/report", json=AAPL)
    _err(c.post("/api/report", json={"thread_id": "t1", "ticker": "ZZZZ"}), 404, "ticker_not_found")
    r = c.post("/api/chat", json={"thread_id": "t1", "message": "Price?"})
    assert r.status_code == 200 and "AAPL" in r.json()["text"]
    assert "AAPL" in llm.calls[0].messages[0]["content"]


def test_upstream_failure_in_chat_503(client):
    c = client(chat_llm("filings"),
               answer_filings_question=raiser(DataSourceError("sec_edgar", "down")))
    c.post("/api/report", json=AAPL)
    _err(c.post("/api/chat", json={"thread_id": "t1", "message": "Risks?"}),
         503, "upstream_unavailable")


def test_cors_allows_vite_origin(client):
    r = client().options("/api/report", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_default_stub_app_is_offline(tmp_data_dir, monkeypatch):
    # Before Phase 2 the no-deps app serves stubs; it must never build a real LLM (no key/network).
    def boom(*a, **k):
        raise AssertionError("stub app constructed AnthropicLLM")

    monkeypatch.setattr("app.llm.AnthropicLLM", boom)
    c = TestClient(create_app(checkpointer=InMemorySaver()))
    c.post("/api/report", json=AAPL)
    r = c.post("/api/chat", json={"thread_id": "t1", "message": "Main risks?"})
    assert r.status_code == 200 and r.json()["route"] == "filings" and r.json()["citations"]


def test_unexpected_chat_error_is_error_response(tmp_data_dir):
    app = create_app(deps_with(chat_llm("filings"),
                               answer_filings_question=raiser(RuntimeError("secret detail"))),
                     InMemorySaver())
    c = TestClient(app, raise_server_exceptions=False)
    c.post("/api/report", json=AAPL)
    r = c.post("/api/chat", json={"thread_id": "t1", "message": "Risks?"})
    _err(r, 500, "internal_error")
    assert "secret detail" not in r.text
