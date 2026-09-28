import inspect

import pytest

from app.agents import market
from app.contracts import AnswerMarketQuestion, GetMarketSnapshot, stub_market_snapshot
from app.models import DataSourceError, RateLimitedError
from tests.ws1_market.helpers import load_fixture

QUOTE = load_fixture("av_global_quote_AAPL.json")
OVERVIEW = load_fixture("av_overview_AAPL.json")


def _sig(fn):
    # unwrap: @traceable adds a keyword-only `config=None` to the public signature
    s = inspect.signature(inspect.unwrap(fn))
    return [(p.name, p.annotation) for p in s.parameters.values() if p.name != "self"], (
        s.return_annotation)


def test_signatures_match_contracts():
    assert _sig(market.get_market_snapshot) == _sig(GetMarketSnapshot.__call__)
    assert _sig(market.answer_market_question) == _sig(AnswerMarketQuestion.__call__)


def test_snapshot_normalises_ticker(make_client):
    client, av = make_client(GLOBAL_QUOTE=QUOTE, OVERVIEW=OVERVIEW)
    snap = market._snapshot_with(client, " aapl ")
    assert snap.quote.symbol == snap.overview.symbol == "AAPL"
    assert {r.url.params["symbol"] for r in av.calls} == {"AAPL"}
    market._snapshot_with(client, "AAPL")
    assert len(av.calls) == 2  # second snapshot fully served from cache


def test_get_market_snapshot_uses_default_client(monkeypatch, make_client):
    client, _ = make_client(GLOBAL_QUOTE=QUOTE, OVERVIEW=OVERVIEW)
    monkeypatch.setattr(market, "_default_client", client)
    snap = market.get_market_snapshot("AAPL")
    assert snap.overview.name == "Apple Inc"
    assert snap.quote.source.provider == snap.overview.source.provider == "alpha_vantage"


def test_rate_limit_propagates(monkeypatch, make_client):
    client, _ = make_client(GLOBAL_QUOTE=load_fixture("av_rate_limited.json"))
    monkeypatch.setattr(market, "_default_client", client)
    with pytest.raises(RateLimitedError):
        market.get_market_snapshot("AAPL")


def test_missing_api_key_is_data_source_error(monkeypatch, tmp_data_dir):
    monkeypatch.setattr(market, "_default_client", None)
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(DataSourceError) as exc:
        market.get_market_snapshot("AAPL")
    assert exc.value.provider == "alpha_vantage"


def test_default_client_built_lazily_from_config(monkeypatch, tmp_data_dir):
    monkeypatch.setattr(market, "_default_client", None)
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "k")
    client = market._client()
    assert client is market._client()  # built once
    assert client._cache.path == tmp_data_dir / "cache.sqlite"


def test_answer_passes_snapshot_and_returns_both_sources(fake_llm):
    snap = stub_market_snapshot("AAPL")
    fake_llm.queue("text", "  AAPL last traded at $255.46.  ")
    ans = market.answer_market_question("What's the P/E?", snap, fake_llm)
    call = fake_llm.calls[0]
    assert (call.method, call.role) == ("text", "agent")
    content = call.messages[0]["content"]
    assert "What's the P/E?" in content and "255.46" in content and "38.87" in content
    assert "investment advice" in call.system.lower()
    assert ans.text == "AAPL last traded at $255.46."
    assert ans.citations == [snap.quote.source, snap.overview.source]
