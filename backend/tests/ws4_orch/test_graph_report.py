import pytest

from app.graph import get_report, run_config, run_report
from app.llm import LLMError
from app.models import (
    DISCLAIMER,
    Claim,
    DataSourceError,
    FilingsSummary,
    RateLimitedError,
    TickerNotFoundError,
)
from tests.fakes import FakeLLM
from tests.ws4_orch.helpers import VALID_ID, make_graph, raiser, summarize_plus


def test_report_happy_path_no_llm_spend():
    llm = FakeLLM()
    g = make_graph(llm)
    r = run_report(g, "t1", "AAPL")
    assert r.company_name == "Apple Inc." and r.disclaimer == DISCLAIMER and r.warnings == []
    assert r.market.quote.source.provider == "alpha_vantage"
    claims = r.filings.all_claims()
    assert claims and all(c in r.filings.citation_index for cl in claims for c in cl.citations)
    assert llm.calls == [] and get_report(g, "t1") == r


def test_market_rate_limited_gives_partial_report():
    r = run_report(make_graph(get_market_snapshot=raiser(RateLimitedError("alpha_vantage", "q"))),
                   "t1", "AAPL")
    assert r.market is None and r.filings is not None
    assert any("Market data unavailable" in w for w in r.warnings)


def test_filings_error_gives_partial_report():
    r = run_report(make_graph(ingest_recent_filings=raiser(DataSourceError("sec_edgar", "503"))),
                   "t1", "AAPL")
    assert r.filings is None and r.market is not None
    assert any("Filings unavailable" in w for w in r.warnings)


def test_llm_error_in_summary_is_partial():
    r = run_report(make_graph(summarize_filings=raiser(LLMError("max_tokens"))), "t1", "AAPL")
    assert r.filings is None and any("Filings unavailable" in w for w in r.warnings)


def test_unknown_ticker_raises_before_any_spend():
    calls = []
    g = make_graph(get_market_snapshot=calls.append, ingest_recent_filings=calls.append)
    with pytest.raises(TickerNotFoundError):
        run_report(g, "t1", "ZZZZ")
    assert calls == []


def test_advice_and_unknown_citation_claims_dropped():
    advice = Claim(text="Investors should buy AAPL ahead of earnings.", citations=[VALID_ID])
    bogus = Claim(text="Revenue doubled.", citations=["fake:id:0"])
    r = run_report(make_graph(summarize_filings=summarize_plus(advice, bogus)), "t1", "AAPL")
    texts = [c.text for c in r.filings.all_claims()]
    assert advice.text not in texts and bogus.text not in texts
    assert any("advice" in w for w in r.warnings) and any("ungrounded" in w for w in r.warnings)


def test_run_config_carries_tracing_metadata():
    cfg = run_config("t9", "report", "AAPL")
    assert cfg["configurable"] == {"thread_id": "t9"} and "report" in cfg["tags"]
    assert cfg["run_name"] == "financial_agent.report"
    assert cfg["metadata"] == {"ticker": "AAPL", "thread_id": "t9", "mode": "report"}


def test_market_ticker_not_found_is_partial_not_404():
    # SEC decides existence; AV miss (e.g. ETF without OVERVIEW) is only a missing branch.
    r = run_report(make_graph(get_market_snapshot=raiser(TickerNotFoundError("AAPL"))),
                   "t1", "AAPL")
    assert r.market is None and r.filings is not None
    assert any("No market data available for AAPL" in w for w in r.warnings)


def test_empty_summary_warns():
    def empty(ticker, company_name, filings, retriever, llm):
        return FilingsSummary(ticker=ticker, company_name=company_name, filings=filings)

    r = run_report(make_graph(summarize_filings=empty), "t1", "AAPL")
    assert r.filings is not None and r.filings.all_claims() == []
    assert any("No filing content could be retrieved" in w for w in r.warnings)


def test_unexpected_branch_error_is_partial():
    # e.g. an Anthropic 529 or Chroma error that isn't a DataSourceError/LLMError
    r = run_report(make_graph(summarize_filings=raiser(RuntimeError("overloaded"))), "t1", "AAPL")
    assert r.filings is None and r.market is not None
    assert any("Filings unavailable" in w for w in r.warnings)
    r = run_report(make_graph(get_market_snapshot=raiser(RuntimeError("boom"))), "t1", "AAPL")
    assert r.market is None and r.filings is not None
    assert any("Market data unavailable" in w for w in r.warnings)
