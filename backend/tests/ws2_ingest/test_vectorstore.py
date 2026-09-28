from datetime import date

from app import vectorstore
from app.vectorstore import ChunkRecord
from tests.ws2_ingest.helpers import HashEmbedding

AAPL_CIK, MSFT_CIK = "0000320193", "0000789019"

def _rec(cid, ticker, form, acc, cik):
    return ChunkRecord(cid, f"{ticker} {form} text about supply chain risk", {
        "ticker": ticker, "cik": cik, "accession_no": acc, "form_type": form,
        "filed_date": "2025-10-31", "section": "Item 1A. Risk Factors",
        "url": f"https://www.sec.gov/Archives/edgar/data/1/{acc}/d.htm", "chunk_idx": 0,
        "retrieved_at": "2026-09-28T18:00:00+00:00"})

RECS = [_rec("A-1:item-1a:0", "AAPL", "10-K", "A-1", AAPL_CIK),
        _rec("A-2:8-k:0", "AAPL", "8-K", "A-2", AAPL_CIK),
        _rec("M-1:item-1a:0", "MSFT", "10-K", "M-1", MSFT_CIK)]

def test_query_filters_by_ticker_and_form(collection):
    assert vectorstore.add_records(collection, RECS) == 3
    got = vectorstore.query_chunks(collection, AAPL_CIK, "AAPL", "risk", k=6)
    assert {c.chunk_id for c in got} == {"A-1:item-1a:0", "A-2:8-k:0"}
    only_8k = vectorstore.query_chunks(collection, AAPL_CIK, "AAPL", "risk", k=6, form_type="8-K")
    assert [c.chunk_id for c in only_8k] == ["A-2:8-k:0"]
    assert vectorstore.has_accession(collection, "A-1")
    assert not vectorstore.has_accession(collection, "Z-9")

def test_query_builds_complete_source_ref(collection):
    vectorstore.add_records(collection, RECS)
    (c,) = vectorstore.query_chunks(collection, MSFT_CIK, "MSFT", "risk", k=6)
    s = c.source
    assert c.ticker == "MSFT" and c.text.startswith("MSFT 10-K")
    assert (s.provider, s.accession_no, s.form_type, s.section, s.chunk_id) == (
        "sec_edgar", "M-1", "10-K", "Item 1A. Risk Factors", "M-1:item-1a:0")
    assert s.filed_date == date(2025, 10, 31) and s.retrieved_at.tzinfo is not None
    assert s.url.startswith("https://www.sec.gov/Archives/")

def test_empty_and_persistence(tmp_data_dir, collection):
    assert vectorstore.add_records(collection, []) == 0
    assert vectorstore.query_chunks(collection, AAPL_CIK, "AAPL", "anything") == []
    assert vectorstore.query_chunks(collection, AAPL_CIK, "AAPL", "anything", k=0) == []
    vectorstore.add_records(collection, RECS)
    again = vectorstore.get_collection(tmp_data_dir, embedding_function=HashEmbedding())
    assert again.count() == 3 and (tmp_data_dir / "chroma").is_dir()
