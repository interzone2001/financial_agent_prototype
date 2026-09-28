"""8-K Exhibit 99.1 press release ingestion (stretch, spec Task 6)."""

from __future__ import annotations

from datetime import date

from app.ingest import ingest_recent_filings
from app.sources.edgar import exhibit_991_url
from tests.ws2_ingest.helpers import ACC_8K, ARCH

EX_URL = f"{ARCH}/000032019326000018/a8-kex991q3202606272026.htm"
NAMES = ["0000320193-26-000018-index.html", "aapl-20260730.htm", "a8-kex991q3202606272026.htm",
         "a8-kex992.htm", "R1.htm"]


def test_exhibit_991_url_picks_99_1_only():
    assert exhibit_991_url(NAMES, "0000320193", ACC_8K) == EX_URL
    assert exhibit_991_url(["aapl-20260730.htm", "ex-99.2.htm"], "0000320193", ACC_8K) is None


def test_8k_exhibit_ingested_best_effort(edgar_api, client, collection):
    ingest_recent_filings("AAPL", client=client, collection=collection, today=date(2026, 9, 28))
    assert collection.get(where={"accession_no": ACC_8K})["ids"]  # index 404 -> primary only
    assert not collection.get(where={"section": "8-K Exhibit 99.1"})["ids"]
    collection.delete(where={"accession_no": ACC_8K})
    edgar_api.routes["index_8k"].respond(json={"directory": {"item": [{"name": n} for n in NAMES]}})
    edgar_api.get(EX_URL).respond(text="<p>Apple reports third quarter results.</p>")
    ingest_recent_filings("AAPL", client=client, collection=collection, today=date(2026, 9, 28))
    got = collection.get(where={"section": "8-K Exhibit 99.1"})
    assert got["ids"] == [f"{ACC_8K}:8-k-exhibit-99-1:0"]
    assert got["metadatas"][0]["url"] == EX_URL


def test_8k_exhibit_malformed_index_is_best_effort(edgar_api, client, collection):
    """A malformed (non-index-shaped) index.json must not sink the whole ingest."""
    edgar_api.routes["index_8k"].respond(json={"unexpected": 1})
    ingest_recent_filings("AAPL", client=client, collection=collection, today=date(2026, 9, 28))
    assert collection.get(where={"accession_no": ACC_8K})["ids"]  # primary 8-K still stored
    assert not collection.get(where={"section": "8-K Exhibit 99.1"})["ids"]
