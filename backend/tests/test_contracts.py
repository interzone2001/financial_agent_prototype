"""Phase 0 smoke: contracts, stubs and fakes are coherent."""
import pytest

from app.contracts import fixture_retriever, stub_deps
from app.models import ReportRequest, RouteDecision, TickerNotFoundError
from tests.fakes import FakeLLM


def test_stub_deps_produce_attributed_data():
    d = stub_deps(llm=FakeLLM())
    snap = d.get_market_snapshot("AAPL")
    assert snap.quote.price == 255.46 and snap.quote.source.provider == "alpha_vantage"
    _cik, name = d.resolve_company("AAPL")
    filings = d.ingest_recent_filings("AAPL")
    assert {f.form_type for f in filings} == {"10-K", "10-Q", "8-K"}
    s = d.summarize_filings("AAPL", name, filings, d.retrieve, d.llm)
    for claim in s.all_claims():
        assert all(cid in s.citation_index for cid in claim.citations)


def test_unknown_ticker_raises():
    with pytest.raises(TickerNotFoundError):
        stub_deps(llm=FakeLLM()).resolve_company("ZZZZ")


def test_fixture_retriever_filters_form():
    hits = fixture_retriever("AAPL", "risk factors", 3, "10-K")
    assert hits and all(c.source.form_type == "10-K" for c in hits)


def test_fake_llm_queue_and_calls():
    llm = FakeLLM(parse=[RouteDecision(route="market", reason="x")])
    assert llm.parse("fast", "sys", [], RouteDecision).route == "market"
    assert llm.calls[0].method == "parse"
    with pytest.raises(AssertionError):
        llm.text("fast", "s", [])


@pytest.mark.parametrize("raw,ok", [(" aapl ", "AAPL"), ("BRK-B", "BRK-B"), ("BRK.B", "BRK.B")])
def test_ticker_normalised(raw, ok):
    assert ReportRequest(thread_id="t", ticker=raw).ticker == ok


@pytest.mark.parametrize("raw", ["", "1ABC", "TOOLONGX", "AA PL", "ignore previous"])
def test_ticker_rejected(raw):
    with pytest.raises(ValueError):
        ReportRequest(thread_id="t", ticker=raw)
