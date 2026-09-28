from datetime import date

import httpx
import pytest

from app import vectorstore
from app.ingest import ingest_recent_filings, retrieve
from app.models import TickerNotFoundError
from tests.ws2_ingest import helpers as h

ALL = [h.ACC_10K, h.ACC_10Q, h.ACC_8K]

def _ingest(client, collection, ticker="AAPL"):
    return ingest_recent_filings(ticker, client=client, collection=collection,
                                 today=date(2026, 9, 28))

def test_ingest_then_retrieve_with_complete_citations(edgar_api, client, collection):
    assert [f.accession_no for f in _ingest(client, collection)] == ALL
    chunks = retrieve("AAPL", "principal risk factors", k=50, collection=collection)
    assert chunks and all(c.ticker == "AAPL" for c in chunks)
    for c in chunks:
        s = c.source
        assert None not in (s.accession_no, s.form_type, s.filed_date, s.section, s.chunk_id)
        assert s.chunk_id == c.chunk_id and s.url.startswith("https://www.sec.gov/Archives/")
    assert {"Item 1A. Risk Factors", "Part II, Item 1A. Risk Factors",
            "8-K Items 2.02,9.01"} <= {c.source.section for c in chunks}
    only_q = retrieve("aapl", "demand", k=50, form_type="10-Q", collection=collection)
    assert only_q and {c.source.accession_no for c in only_q} == {h.ACC_10Q}

def test_reingest_is_noop(edgar_api, client, collection):
    _ingest(client, collection)
    count = collection.count()
    assert [f.accession_no for f in _ingest(client, collection)] == ALL
    assert collection.count() == count
    for name in ("doc_10k", "doc_10q", "doc_8k"):
        assert edgar_api.routes[name].call_count == 1

def test_failed_document_is_skipped_then_retried(edgar_api, client, collection):
    edgar_api.routes["doc_10q"].mock(
        side_effect=[httpx.Response(500), httpx.Response(200, text=h.TEN_Q_HTML)])
    assert [f.accession_no for f in _ingest(client, collection)] == [h.ACC_10K, h.ACC_8K]
    assert not vectorstore.has_accession(collection, h.ACC_10Q)
    assert [f.accession_no for f in _ingest(client, collection)] == ALL

def test_unknown_tickers(edgar_api, client, collection):
    with pytest.raises(TickerNotFoundError):
        _ingest(client, collection, ticker="ZZZZZ")
    assert edgar_api.routes["submissions"].call_count == 0
    assert retrieve("NOPE", "anything", collection=collection) == []

@pytest.mark.live
def test_live_aapl_end_to_end(tmp_data_dir):
    filings = ingest_recent_filings("AAPL")  # real EDGAR + real default embedding model
    assert {"10-K", "10-Q"} <= {f.form_type for f in filings}
    chunks = retrieve("AAPL", "principal risk factors", k=4, form_type="10-K")
    assert chunks and all(c.source.url.startswith("https://www.sec.gov/Archives/") for c in chunks)
    metas = vectorstore.get_collection().get(where={"form_type": "10-K"})["metadatas"]
    assert "Item 1A. Risk Factors" in {m["section"] for m in metas}
