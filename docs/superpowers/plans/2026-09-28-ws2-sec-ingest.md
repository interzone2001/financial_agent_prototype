# WS2 SEC Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve a ticker on SEC EDGAR, ingest its latest 10-K, latest 10-Q and recent 8-Ks into Chroma as sectioned chunks, and serve cited retrieval.

**Architecture:** `app/sources/edgar.py` holds all network code (throttled httpx client) plus pure `select_filings`. `app/vectorstore.py` wraps the Chroma `filings` collection with an injectable embedding fn. `app/ingest.py` does HTML→text→sections→chunks→records and exposes `ingest_recent_filings` / `retrieve`. Tests inject a respx-mocked client, a tmp-dir collection with a hash embedding fn, and a fixed `today`.

**Tech Stack:** Python 3.12, httpx, respx, BeautifulSoup+lxml, chromadb 1.5, langsmith `@traceable`, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-28-financial-agent-design.md` (§1–5, §6.2).

Run everything from `backend/`. Every code block below was assembled task by task and run against the fixtures before this plan was written (test counts per task are real; the live test passed against EDGAR).

## Global Constraints

- Own only `backend/app/sources/edgar.py`, `backend/app/ingest.py`, `backend/app/vectorstore.py`, `backend/tests/ws2_ingest/*`. `models.py`, `contracts.py`, `llm.py`, `tracing.py`, `config.py`, `tests/conftest.py`, `tests/fixtures/*` are READ-ONLY. If a contract looks wrong, STOP and tell the human.
- Public signatures exactly (§3.2): `resolve_company(ticker: str) -> tuple[str, str]`, `ingest_recent_filings(ticker: str) -> list[FilingMeta]`, `retrieve(ticker: str, query: str, k: int = 6, form_type: str | None = None) -> list[Chunk]`. Injection only via extra **keyword-only** args with defaults (`client=`, `collection=`, `today=`).
- EDGAR: `User-Agent` from `config.sec_user_agent()`, `Accept-Encoding: gzip`, ≤10 req/s. No live network in default tests; exactly one `@pytest.mark.live` test.
- Tracing (§3.6): `@traceable` from `app.tracing`, names `"<module>.<function>"`; `edgar.*` and `ingest.retrieve` use `run_type="tool"`, `ingest.ingest_recent_filings` uses `"chain"`.
- Chroma (§3.5): `PersistentClient(DATA_DIR/"chroma")`, collection `filings`, doc id = chunk_id `"{accession_no}:{section_slug}:{idx}"`, metadata `ticker, cik, accession_no, form_type, filed_date (ISO str), section, url, chunk_idx` plus `retrieved_at` (ISO, feeds `SourceRef.retrieved_at`), all scalar.
- Chunks ~1500 chars / 200 overlap, cap 150 per filing (log when capped). Suite < 10 s; `uv run ruff check` clean (use `--fix` for import sorting).

## Review Focus

- Share-class and messy tickers (`" brk.b "`) must resolve like SEC's `BRK-B`. Pinned by Task 1 `test_resolve_company`.
- One filing document failing (HTTP 500) must not sink the whole ingest, and must be retried on the next call (not marked stored). Pinned by Task 5 `test_failed_document_is_skipped_then_retried`.
- Empty or cover-page-only documents produce zero chunks, and Chroma rejects empty upserts. Pinned by Task 3 `test_empty_and_persistence` and Task 4 `test_records_ids_metadata_cap_and_empty`.
- Retrieval for a never-ingested ticker returns `[]`, not an exception. Pinned by Task 5 `test_unknown_tickers`.
- Real filings use uppercase `ITEM 1A.` with `&nbsp;`, and 10-Qs repeat `Item 2` in Part I (MD&A) and Part II (repurchases). Pinned by Task 4 `test_uppercase_and_nbsp_headings_are_found` and `test_10q_sections_pick_part_i_mdna_and_part_ii_risk`.

---

### Task 1: EdgarClient + `resolve_company`

**Files:** Create `backend/app/sources/edgar.py`, `backend/tests/ws2_ingest/helpers.py`, `backend/tests/ws2_ingest/conftest.py`, `backend/tests/ws2_ingest/test_edgar.py`

**Interfaces:**
- Produces: `EdgarClient(user_agent=None, http=None, min_interval=0.11, sleep=time.sleep)` with `.get_company_tickers() -> dict[str, tuple[str, str]]`, `.get_submissions(cik) -> dict` and `.get_document(url) -> str`; `default_client()`; `normalize_ticker(t) -> str`; `resolve_company(ticker, *, client=None) -> (cik10, title)`. HTTP 429 raises `RateLimitedError`; any other non-200 or transport error raises `DataSourceError("sec_edgar", …)`.
- Test fixtures (conftest): `edgar_api` (respx router with named routes `tickers, submissions, doc_10k, doc_10q, doc_8k`), `client` and `collection` (tmp Chroma + `HashEmbedding`).

- [ ] **Step 0: Warm the embedding model once (~80MB download, no-op if cached; can run in background)**
Run: `uv run python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as D; D()(['warm'])"`. Expected: exit 0.

- [ ] **Step 1: Test helpers.** Create `backend/tests/ws2_ingest/helpers.py`:
```python
"""Shared WS2 test data (importable, unlike conftest)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from chromadb import Documents, EmbeddingFunction, Embeddings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
UA = "WS2Tests test@example.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000320193.json"
ARCH = "https://www.sec.gov/Archives/edgar/data/320193"
ACC_10K, ACC_10Q, ACC_8K = "0000320193-25-000079", "0000320193-26-000020", "0000320193-26-000018"
URL_10K = f"{ARCH}/000032019325000079/aapl-20250927.htm"
URL_10Q = f"{ARCH}/000032019326000020/aapl-20260627.htm"
URL_8K = f"{ARCH}/000032019326000018/aapl-20260730.htm"

# Synthetic 10-Q: TOC first; Part II Item 2 is LONGER than Part I Item 2 (MD&A), so a naive
# "longest span" pick would grab share repurchases instead of MD&A.
TEN_Q_HTML = (
    "<html><body>"
    "<p>PART I Item 1. Financial Statements 1 Item 2. Management’s Discussion and Analysis 13 "
    "PART II Item 1A. Risk Factors 20 Item 2. Unregistered Sales of Equity Securities 21</p>"
    "<p>PART I Item 1. Financial Statements " + "Net sales table row. " * 40 + "</p>"
    "<p>Item 2. Management’s Discussion and Analysis "
    + "Services net sales grew on strong demand. " * 60 + "</p>"
    "<p>Item 3. Quantitative and Qualitative Disclosures About Market Risk None.</p>"
    "<p>PART II Item 1. Legal Proceedings None.</p>"
    "<p>Item 1A. Risk Factors " + "Tariffs and export controls may affect supply. " * 30 + "</p>"
    "<p>Item 2. Unregistered Sales of Equity Securities and Use of Proceeds "
    + "Share repurchases detail row. " * 120 + "</p>"
    "<p>Item 5. Other Information None.</p></body></html>"
)


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class HashEmbedding(EmbeddingFunction[Documents]):
    """Deterministic, offline 16-d embedding so tests never load the ONNX model."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:16]] for t in input]

    @staticmethod
    def name() -> str:
        return "ws2-test-hash"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> HashEmbedding:
        return HashEmbedding()
```
Create `backend/tests/ws2_ingest/conftest.py` (app imports are lazy so Task 1 runs before Task 3 exists):
```python
import json

import httpx
import pytest
import respx

from tests.ws2_ingest import helpers as h


@pytest.fixture
def edgar_api():
    """All EDGAR endpoints mocked from fixtures. Unmocked URLs fail loudly."""
    with respx.mock(assert_all_called=False) as mock:
        tickers = json.loads(h.fixture_text("sec_company_tickers.json"))
        mock.get(h.TICKERS_URL, name="tickers").respond(json=tickers)
        subs = json.loads(h.fixture_text("sec_submissions_AAPL.json"))
        mock.get(h.SUBMISSIONS_URL, name="submissions").respond(json=subs)
        tenk = h.fixture_text("sec_10k_AAPL_excerpt.html")
        mock.get(h.URL_10K, name="doc_10k").respond(text=tenk)
        mock.get(h.URL_10Q, name="doc_10q").respond(text=h.TEN_Q_HTML)
        mock.get(h.URL_8K, name="doc_8k").respond(text=h.fixture_text("sec_8k_AAPL.html"))
        yield mock


@pytest.fixture
def client():
    from app.sources.edgar import EdgarClient

    with httpx.Client() as http:
        yield EdgarClient(user_agent=h.UA, http=http, min_interval=0)


@pytest.fixture
def collection(tmp_data_dir):
    from app import vectorstore

    return vectorstore.get_collection(embedding_function=h.HashEmbedding())
```

- [ ] **Step 2: Failing tests.** Create `backend/tests/ws2_ingest/test_edgar.py`:
```python
import httpx
import pytest

from app.models import DataSourceError, RateLimitedError, TickerNotFoundError
from app.sources.edgar import EdgarClient, resolve_company
from tests.ws2_ingest.helpers import UA


def test_resolve_company(edgar_api, client):
    assert resolve_company("AAPL", client=client) == ("0000320193", "Apple Inc.")
    assert resolve_company(" brk.b ", client=client) == ("0001067983", "BERKSHIRE HATHAWAY INC")
    with pytest.raises(TickerNotFoundError):
        resolve_company("ZZZZZ", client=client)
    assert edgar_api.routes["tickers"].call_count == 1  # map cached per client


def test_requests_carry_sec_user_agent(edgar_api, client):
    client.get_submissions("0000320193")
    req = edgar_api.routes["submissions"].calls.last.request
    assert req.headers["User-Agent"] == UA and "gzip" in req.headers["Accept-Encoding"]


@pytest.mark.parametrize(("status", "exc"), [(503, DataSourceError), (429, RateLimitedError)])
def test_http_errors_map_to_datasource_errors(edgar_api, client, status, exc):
    edgar_api.routes["submissions"].respond(status)
    with pytest.raises(exc) as info:
        client.get_submissions("0000320193")
    assert info.value.provider == "sec_edgar"


def test_throttle_spaces_consecutive_requests(edgar_api):
    sleeps: list[float] = []
    c = EdgarClient(user_agent=UA, http=httpx.Client(), min_interval=0.5, sleep=sleeps.append)
    c.get_submissions("0000320193")
    c.get_submissions("0000320193")
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 0.5
```

- [ ] **Step 3: Run, expect FAIL.** `uv run pytest tests/ws2_ingest/test_edgar.py -q` → `ModuleNotFoundError: No module named 'app.sources.edgar'`.

- [ ] **Step 4: Implement.** Create `backend/app/sources/edgar.py`:
```python
"""SEC EDGAR client: ticker -> CIK, submissions, filing selection, documents (spec §6.2)."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import lru_cache

import httpx

from app import config
from app.models import DataSourceError, RateLimitedError, TickerNotFoundError
from app.tracing import traceable

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _size(out) -> dict:  # keep multi-MB payloads out of LangSmith
    return {"chars": len(out)}


class EdgarClient:
    """Thin httpx wrapper: SEC User-Agent, <=10 req/s throttle, errors -> DataSourceError."""

    def __init__(self, user_agent: str | None = None, http: httpx.Client | None = None,
                 min_interval: float = 0.11, sleep: Callable[[float], None] = time.sleep):
        self.http = http or httpx.Client(timeout=30.0, follow_redirects=True)
        self.headers = {"User-Agent": user_agent or config.sec_user_agent(),
                        "Accept-Encoding": "gzip, deflate"}
        self.min_interval = min_interval
        self._sleep = sleep
        self._last = float("-inf")
        self._tickers: dict[str, tuple[str, str]] | None = None

    def _get(self, url: str) -> httpx.Response:
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._last = time.monotonic()
        try:
            resp = self.http.get(url, headers=self.headers)
        except httpx.HTTPError as e:
            raise DataSourceError("sec_edgar", f"EDGAR request failed: {url}: {e}") from e
        if resp.status_code == 429:
            raise RateLimitedError("sec_edgar", f"EDGAR rate limited: {url}")
        if resp.status_code != 200:
            raise DataSourceError("sec_edgar", f"EDGAR HTTP {resp.status_code}: {url}")
        return resp

    @traceable(run_type="tool", name="edgar.get_company_tickers", process_outputs=_size)
    def get_company_tickers(self) -> dict[str, tuple[str, str]]:
        """TICKER -> (10-digit cik, title). Fetched once per client (process lifetime)."""
        if self._tickers is None:
            raw = self._get(TICKERS_URL).json()
            self._tickers = {row["ticker"].upper(): (str(row["cik_str"]).zfill(10), row["title"])
                             for row in raw.values()}
        return self._tickers

    @traceable(run_type="tool", name="edgar.get_submissions", process_outputs=_size)
    def get_submissions(self, cik: str) -> dict:
        return self._get(SUBMISSIONS_URL.format(cik=cik)).json()

    @traceable(run_type="tool", name="edgar.get_document", process_outputs=_size)
    def get_document(self, url: str) -> str:
        return self._get(url).text


@lru_cache(maxsize=1)
def default_client() -> EdgarClient:
    return EdgarClient()


def normalize_ticker(ticker: str) -> str:
    """SEC uses '-' for share classes: 'brk.b ' -> 'BRK-B'."""
    return ticker.strip().upper().replace(".", "-")


@traceable(run_type="tool", name="edgar.resolve_company")
def resolve_company(ticker: str, *, client: EdgarClient | None = None) -> tuple[str, str]:
    """(10-digit cik, company title). Raises TickerNotFoundError."""
    hit = (client or default_client()).get_company_tickers().get(normalize_ticker(ticker))
    if hit is None:
        raise TickerNotFoundError(ticker)
    return hit
```

- [ ] **Step 5: Run, expect PASS.** `uv run pytest tests/ws2_ingest/test_edgar.py -q` → `5 passed`.
- [ ] **Step 6: Commit.** `git add app/sources/edgar.py tests/ws2_ingest && git commit -m "feat(ws2): EDGAR client with UA, throttle, resolve_company"`

---

### Task 2: Filing selection + document URLs

**Files:** Modify `backend/app/sources/edgar.py` (append); Create `backend/tests/ws2_ingest/test_selection.py`

**Interfaces:**
- Produces: `filing_base_url(cik, accession_no)`, `doc_url(cik, accession_no, primary_document)`, and `select_filings(submissions: dict, cik: str, today: date) -> list[FilingMeta]`. The result is ordered `[10-K, 10-Q, *8-Ks newest-first]`. 8-K `items` are parsed from `"2.02,9.01"`, an empty `reportDate` becomes `None`, and filings dated after `today` are ignored.

Fixture facts (verified in `sec_submissions_AAPL.json`): as of 2026-09-28 the 8-K window starts 2026-06-30. The only newer 8-K-type filing besides `0000320193-26-000018` is the amendment `8-K/A 0001140361-26-035325`, which is excluded.

- [ ] **Step 1: Failing tests.** Create `backend/tests/ws2_ingest/test_selection.py`:
```python
import json
from datetime import date

from app.sources.edgar import select_filings
from tests.ws2_ingest import helpers as h

CIK = "0000320193"
AAPL = lambda: json.loads(h.fixture_text("sec_submissions_AAPL.json"))

def _subs(rows):
    """rows: (form, accession, filed ISO, items) -> minimal submissions JSON."""
    return {"filings": {"recent": {
        "form": [r[0] for r in rows], "accessionNumber": [r[1] for r in rows],
        "filingDate": [r[2] for r in rows], "reportDate": ["" for _ in rows],
        "primaryDocument": ["d.htm" for _ in rows], "items": [r[3] for r in rows]}}}

def test_select_from_fixture_as_of_2026_09_28():
    got = select_filings(AAPL(), CIK, date(2026, 9, 28))
    assert [(f.form_type, f.accession_no) for f in got] == [
        ("10-K", h.ACC_10K), ("10-Q", h.ACC_10Q), ("8-K", h.ACC_8K)]
    assert [f.primary_doc_url for f in got] == [h.URL_10K, h.URL_10Q, h.URL_8K]  # URL build
    assert got[2].items == ["2.02", "9.01"] and got[0].items == []
    assert got[1].filed_date == date(2026, 7, 31) and got[1].report_date == date(2026, 6, 27)

def test_select_is_as_of_today_injected():
    assert [f.accession_no for f in select_filings(AAPL(), CIK, date(2026, 5, 1))] == [
        "0000320193-25-000079",  # 10-K 2025-10-31
        "0000320193-26-000013",  # 10-Q 2026-05-01 (filed on 'today' counts)
        "0000320193-26-000011",  # 8-K 2026-04-30
        "0001140361-26-015711",  # 8-K 2026-04-20
        "0001140361-26-006577",  # 8-K 2026-02-24 (window starts 2026-01-31)
    ]

def test_select_caps_8ks_at_five_newest_and_skips_amendments():
    rows = [("8-K", f"0000000001-26-00000{i}", f"2026-09-{20 - i:02d}", "7.01") for i in range(7)]
    rows += [("8-K/A", "0000000001-26-000099", "2026-09-27", "5.02"),
             ("10-K/A", "0000000001-26-000098", "2026-09-26", "")]
    got = select_filings(_subs(rows), "0000000001", date(2026, 9, 28))
    assert [f.accession_no for f in got] == [f"0000000001-26-00000{i}" for i in range(5)]
    assert got[0].report_date is None

def test_select_90_day_window_boundary():
    rows = [("8-K", "0000000001-26-000001", "2026-06-30", ""),   # exactly 90 days -> in
            ("8-K", "0000000001-26-000002", "2026-06-29", "")]   # 91 days -> out
    got = select_filings(_subs(rows), "0000000001", date(2026, 9, 28))
    assert [f.accession_no for f in got] == ["0000000001-26-000001"] and got[0].items == []
```

- [ ] **Step 2: Run, expect FAIL.** `uv run pytest tests/ws2_ingest/test_selection.py -q` → `ImportError: cannot import name 'select_filings'`.

- [ ] **Step 3: Implement.** In `edgar.py`, add `from datetime import date, timedelta` and `FilingMeta` to the imports, add these constants under `SUBMISSIONS_URL`:
```python
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}"
EIGHT_K_WINDOW_DAYS = 90
MAX_8K = 5
```
and append:
```python
def filing_base_url(cik: str, accession_no: str) -> str:
    return ARCHIVES_URL.format(cik_int=int(cik), acc_nodash=accession_no.replace("-", ""))


def doc_url(cik: str, accession_no: str, primary_document: str) -> str:
    return f"{filing_base_url(cik, accession_no)}/{primary_document}"


def select_filings(submissions: dict, cik: str, today: date) -> list[FilingMeta]:
    """Latest 10-K, latest 10-Q, 8-Ks filed in [today-90d, today] (newest 5). No /A forms."""
    recent = submissions["filings"]["recent"]
    n = len(recent["form"])
    col = lambda k: recent.get(k) or [""] * n
    rows = [
        (form, acc, date.fromisoformat(filed), report, doc, items)
        for form, acc, filed, report, doc, items in zip(
            col("form"), col("accessionNumber"), col("filingDate"),
            col("reportDate"), col("primaryDocument"), col("items"),
        )
        if form in ("10-K", "10-Q", "8-K") and date.fromisoformat(filed) <= today
    ]
    rows.sort(key=lambda r: r[2], reverse=True)

    def meta(form, acc, filed, report, doc, items) -> FilingMeta:
        return FilingMeta(
            cik=cik, accession_no=acc, form_type=form, filed_date=filed,
            report_date=date.fromisoformat(report) if report else None,
            primary_doc_url=doc_url(cik, acc, doc),
            items=[i.strip() for i in items.split(",") if i.strip()] if form == "8-K" else [],
        )

    out: list[FilingMeta] = []
    for form in ("10-K", "10-Q"):
        if latest := next((r for r in rows if r[0] == form), None):
            out.append(meta(*latest))
    cutoff = today - timedelta(days=EIGHT_K_WINDOW_DAYS)
    out += [meta(*r) for r in rows if r[0] == "8-K" and r[2] >= cutoff][:MAX_8K]
    return out
```

- [ ] **Step 4: Run, expect PASS.** `uv run pytest tests/ws2_ingest -q` → `9 passed`.
- [ ] **Step 5: Commit.** `git add app/sources/edgar.py tests/ws2_ingest/test_selection.py && git commit -m "feat(ws2): select latest 10-K/10-Q and 90-day 8-Ks"`

---

### Task 3: Chroma vector store

**Files:** Create `backend/app/vectorstore.py`, `backend/tests/ws2_ingest/test_vectorstore.py`

**Interfaces:**
- Produces: `ChunkRecord(id, text, metadata)` NamedTuple; `get_collection(data_dir: Path | None = None, embedding_function=None) -> Collection`; `warm_embedding_model()`; `has_accession(col, accession_no) -> bool`; `add_records(col, records) -> int` (upsert, `[]` → 0); `query_chunks(col, ticker, query, k=6, form_type=None) -> list[Chunk]`.

- [ ] **Step 1: Failing tests.** Create `backend/tests/ws2_ingest/test_vectorstore.py`:
```python
from datetime import date

from app import vectorstore
from app.vectorstore import ChunkRecord
from tests.ws2_ingest.helpers import HashEmbedding


def _rec(cid, ticker, form, acc):
    return ChunkRecord(cid, f"{ticker} {form} text about supply chain risk", {
        "ticker": ticker, "cik": "0000320193", "accession_no": acc, "form_type": form,
        "filed_date": "2025-10-31", "section": "Item 1A. Risk Factors",
        "url": f"https://www.sec.gov/Archives/edgar/data/1/{acc}/d.htm", "chunk_idx": 0,
        "retrieved_at": "2026-09-28T18:00:00+00:00"})

RECS = [_rec("A-1:item-1a:0", "AAPL", "10-K", "A-1"), _rec("A-2:8-k:0", "AAPL", "8-K", "A-2"),
        _rec("M-1:item-1a:0", "MSFT", "10-K", "M-1")]

def test_query_filters_by_ticker_and_form(collection):
    assert vectorstore.add_records(collection, RECS) == 3
    got = vectorstore.query_chunks(collection, "AAPL", "risk", k=6)
    assert {c.chunk_id for c in got} == {"A-1:item-1a:0", "A-2:8-k:0"}
    only_8k = vectorstore.query_chunks(collection, "AAPL", "risk", k=6, form_type="8-K")
    assert [c.chunk_id for c in only_8k] == ["A-2:8-k:0"]
    assert vectorstore.has_accession(collection, "A-1")
    assert not vectorstore.has_accession(collection, "Z-9")

def test_query_builds_complete_source_ref(collection):
    vectorstore.add_records(collection, RECS)
    (c,) = vectorstore.query_chunks(collection, "MSFT", "risk", k=6)
    s = c.source
    assert c.ticker == "MSFT" and c.text.startswith("MSFT 10-K")
    assert (s.provider, s.accession_no, s.form_type, s.section, s.chunk_id) == (
        "sec_edgar", "M-1", "10-K", "Item 1A. Risk Factors", "M-1:item-1a:0")
    assert s.filed_date == date(2025, 10, 31) and s.retrieved_at.tzinfo is not None
    assert s.url.startswith("https://www.sec.gov/Archives/")

def test_empty_and_persistence(tmp_data_dir, collection):
    assert vectorstore.add_records(collection, []) == 0
    assert vectorstore.query_chunks(collection, "AAPL", "anything") == []
    assert vectorstore.query_chunks(collection, "AAPL", "anything", k=0) == []
    vectorstore.add_records(collection, RECS)
    again = vectorstore.get_collection(tmp_data_dir, embedding_function=HashEmbedding())
    assert again.count() == 3 and (tmp_data_dir / "chroma").is_dir()
```

- [ ] **Step 2: Run, expect FAIL.** `uv run pytest tests/ws2_ingest/test_vectorstore.py -q` → `ImportError: cannot import name 'vectorstore'`.

- [ ] **Step 3: Implement.** Create `backend/app/vectorstore.py`:
```python
"""Chroma persistence for filing chunks (spec §3.5). Embedding fn is injectable for tests."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import NamedTuple

import chromadb
from chromadb.api.models.Collection import Collection

from app import config
from app.models import Chunk, SourceRef

COLLECTION_NAME = "filings"


class ChunkRecord(NamedTuple):
    id: str  # chunk_id "{accession_no}:{section_slug}:{idx}"
    text: str
    metadata: dict[str, str | int]  # §3.5 keys (+ retrieved_at ISO), all scalar


def get_collection(data_dir: Path | None = None, embedding_function=None) -> Collection:
    """`filings` collection under DATA_DIR/chroma. None => Chroma's default ONNX MiniLM."""
    client = chromadb.PersistentClient(path=str((data_dir or config.data_dir()) / "chroma"))
    if embedding_function is None:
        return client.get_or_create_collection(COLLECTION_NAME)
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=embedding_function)


def warm_embedding_model() -> None:
    """Downloads (~80MB, once) and loads the default embedding model."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    DefaultEmbeddingFunction()(["warm up"])


def has_accession(collection: Collection, accession_no: str) -> bool:
    return bool(collection.get(where={"accession_no": accession_no}, limit=1)["ids"])


def add_records(collection: Collection, records: list[ChunkRecord]) -> int:
    if not records:  # Chroma rejects empty upserts
        return 0
    collection.upsert(ids=[r.id for r in records], documents=[r.text for r in records],
                      metadatas=[r.metadata for r in records])
    return len(records)


def _where(ticker: str, form_type: str | None) -> dict:
    if form_type is None:
        return {"ticker": ticker}
    return {"$and": [{"ticker": ticker}, {"form_type": form_type}]}


def query_chunks(collection: Collection, ticker: str, query: str, k: int = 6,
                 form_type: str | None = None) -> list[Chunk]:
    if k < 1:
        return []
    res = collection.query(query_texts=[query], n_results=k, where=_where(ticker, form_type))
    return [_to_chunk(cid, doc, meta)
            for cid, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])]


def _to_chunk(chunk_id: str, text: str, m: dict) -> Chunk:
    return Chunk(chunk_id=chunk_id, ticker=m["ticker"], text=text, source=SourceRef(
        provider="sec_edgar", url=m["url"], retrieved_at=datetime.fromisoformat(m["retrieved_at"]),
        accession_no=m["accession_no"], form_type=m["form_type"],
        filed_date=date.fromisoformat(m["filed_date"]), section=m["section"], chunk_id=chunk_id))
```

- [ ] **Step 4: Run, expect PASS.** `uv run pytest tests/ws2_ingest -q` → `12 passed`.
- [ ] **Step 5: Commit.** `git add app/vectorstore.py tests/ws2_ingest/test_vectorstore.py && git commit -m "feat(ws2): Chroma filings collection with injectable embeddings"`

---

### Task 4: HTML → text → sections → chunk records

**Files:** Create `backend/app/ingest.py` (parsing half), `backend/tests/ws2_ingest/test_parsing.py`

**Interfaces:**
- Consumes: `ChunkRecord` (Task 3), `FilingMeta`.
- Produces: `html_to_text(html)`, `split_sections(text, form_type) -> dict[label, text]`, `chunk_text(text, size=1500, overlap=200) -> list[str]`, `slugify(s)`, `filing_sections(filing, html)`, `build_records(ticker, filing, sections, retrieved_at, max_chunks=150) -> list[ChunkRecord]`, and `FULL_TEXT = "Full text"`.
- Labels: 10-K uses `"Item 1. Business"`, `"Item 1A. Risk Factors"`, `"Item 7. Management's Discussion and Analysis"` and `"Item 7A. Quantitative and Qualitative Disclosures About Market Risk"`. 10-Q uses `"Part I, Item 2. Management's Discussion and Analysis"` and `"Part II, Item 1A. Risk Factors"`. 8-K uses `"8-K Items 2.02,9.01"`.
- Algorithm: a heading is `Item <code>[.]` followed by a capitalised title, so cross-refs like "Item 1A of this Form" don't count. A span runs from one heading to the next. For each wanted code, prefer headings whose title starts with the expected word (this separates the two 10-Q `Item 2`s), then take the **longest span** (this skips the TOC, whose spans are ~20 chars). Whitespace is already collapsed, so chunks cut at a sentence end (". ") in the back half of each window instead of at paragraph breaks.

- [ ] **Step 1: Failing tests.** Create `backend/tests/ws2_ingest/test_parsing.py`:
```python
import logging
import warnings
from datetime import UTC, date, datetime
from itertools import pairwise

from app import ingest
from app.models import FilingMeta
from tests.ws2_ingest import helpers as h

NOW = datetime(2026, 9, 28, 18, 0, tzinfo=UTC)
EIGHT_K = FilingMeta(cik="0000320193", accession_no=h.ACC_8K, form_type="8-K",
                     filed_date=date(2026, 7, 30), primary_doc_url=h.URL_8K, items=["2.02", "9.01"])

def _ten_k_text():
    return ingest.html_to_text(h.fixture_text("sec_10k_AAPL_excerpt.html"))

def test_html_to_text_strips_script_style_and_ixbrl_header():
    text = _ten_k_text()
    assert text.startswith("Item 1. Business 1 Item 1A. Risk Factors 5")
    for noise in ("HIDDEN_XBRL_NOISE", "var x", "margin:0", "  ", "\n"):
        assert noise not in text

def test_html_to_text_suppresses_xml_parsed_as_html_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert ingest.html_to_text('<?xml version="1.0"?><root><p>Hello</p></root>') == "Hello"
    assert not [w for w in caught if w.category.__name__ == "XMLParsedAsHTMLWarning"]

def test_10k_sections_skip_table_of_contents():
    secs = ingest.split_sections(_ten_k_text(), "10-K")
    assert list(secs) == ["Item 1. Business", "Item 1A. Risk Factors",
                          "Item 7. Management's Discussion and Analysis",
                          "Item 7A. Quantitative and Qualitative Disclosures About Market Risk"]
    risk = secs["Item 1A. Risk Factors"]
    assert risk.startswith("Item 1A. Risk Factors The following summarizes")
    assert "Item 1B." not in risk and len(risk) > 2000
    assert secs["Item 1. Business"].startswith("Item 1. Business Company Background")

def test_10q_sections_pick_part_i_mdna_and_part_ii_risk():
    secs = ingest.split_sections(ingest.html_to_text(h.TEN_Q_HTML), "10-Q")
    mdna = secs["Part I, Item 2. Management's Discussion and Analysis"]
    risk = secs["Part II, Item 1A. Risk Factors"]
    assert mdna.startswith("Item 2. Management") and "Services net sales" in mdna
    assert "Share repurchases" not in mdna  # longer Part II Item 2 must not win
    assert "Tariffs" in risk and "Share repurchases" not in risk

def test_uppercase_and_nbsp_headings_are_found():
    body = "x " * 50
    html = (f"<p>ITEM\xa01A.\xa0RISK FACTORS {body}Risks here.</p>"
            f"<p>ITEM 1B. UNRESOLVED STAFF COMMENTS None.</p><p>ITEM 7. MANAGEMENT {body}</p>")
    risk = ingest.split_sections(ingest.html_to_text(html), "10-K")["Item 1A. Risk Factors"]
    assert "Risks here." in risk and "UNRESOLVED" not in risk
    assert ingest.split_sections("Cover page.", "10-K") == {ingest.FULL_TEXT: "Cover page."}

def test_chunk_text_size_and_overlap_bounds():
    text = " ".join(f"Sentence {i} discusses revenue and margin trends." for i in range(300))
    chunks = ingest.chunk_text(text)
    assert len(chunks) > 5 and all(len(c) <= 1500 for c in chunks)
    assert all(len(c) >= 700 for c in chunks[:-1])
    for a, b in pairwise(chunks):
        assert a.find(b[:60]) >= len(a) - 220  # ~200-char overlap, carried from a's tail
    assert chunks[0].startswith("Sentence 0 ")
    assert chunks[-1].endswith("Sentence 299 discusses revenue and margin trends.")
    assert ingest.chunk_text("") == [] and len(ingest.chunk_text("x" * 4000)) == 3

def test_records_ids_metadata_cap_and_empty(caplog):
    sections = ingest.filing_sections(EIGHT_K, h.fixture_text("sec_8k_AAPL.html"))
    recs = ingest.build_records("AAPL", EIGHT_K, sections, NOW)
    assert recs and recs[0].id == f"{h.ACC_8K}:8-k-items-2-02-9-01:0"
    assert recs[0].metadata == {
        "ticker": "AAPL", "cik": "0000320193", "accession_no": h.ACC_8K, "form_type": "8-K",
        "filed_date": "2026-07-30", "section": "8-K Items 2.02,9.01", "url": h.URL_8K,
        "chunk_idx": 0, "retrieved_at": NOW.isoformat()}
    long = " ".join(f"Sentence {i} about risk." for i in range(2000))
    with caplog.at_level(logging.WARNING):
        capped = ingest.build_records("AAPL", EIGHT_K, {"A": long, "B": long}, NOW, max_chunks=3)
    assert len(capped) == 3 and "capped" in caplog.text
    assert ingest.build_records("AAPL", EIGHT_K, {ingest.FULL_TEXT: ""}, NOW) == []
```

- [ ] **Step 2: Run, expect FAIL.** `uv run pytest tests/ws2_ingest/test_parsing.py -q` → `ImportError: cannot import name 'ingest'`.

- [ ] **Step 3: Implement.** Create `backend/app/ingest.py`:
```python
"""EDGAR HTML -> text -> sections -> chunks -> Chroma; plus cited retrieval (spec §6.2)."""

from __future__ import annotations

import logging
import re
import warnings
from datetime import datetime

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from app.models import FilingMeta
from app.vectorstore import ChunkRecord

log = logging.getLogger(__name__)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_FILING = 150
FULL_TEXT = "Full text"

# A heading is "Item <code>[.]" + a capitalised title; "Item 1A of this Form" is not.
HEADING_RE = re.compile(r"\b(?i:item)\s+(\d{1,2}[A-C]?)\.?\s+(?=[A-Z\[‘“\"'])")
# form -> [(item code, expected title prefix, section label)]
WANTED = {
    "10-K": [
        ("1", "business", "Item 1. Business"),
        ("1A", "risk", "Item 1A. Risk Factors"),
        ("7", "management", "Item 7. Management's Discussion and Analysis"),
        ("7A", "quantitative", "Item 7A. Quantitative and Qualitative Disclosures About Market Risk"),
    ],
    "10-Q": [
        ("2", "management", "Part I, Item 2. Management's Discussion and Analysis"),
        ("1A", "risk", "Part II, Item 1A. Risk Factors"),
    ],
}


def html_to_text(html: str) -> str:
    with warnings.catch_warnings():  # inline-XBRL XHTML parsed as HTML on purpose
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def split_sections(text: str, form_type: str) -> dict[str, str]:
    """Per wanted Item, the heading occurrence starting the longest span (TOC spans are tiny)."""
    heads = [(m.start(), m.group(1).upper(), text[m.end() : m.end() + 40].lower())
             for m in HEADING_RE.finditer(text)]
    spans = [(code, title, start, heads[i + 1][0] if i + 1 < len(heads) else len(text))
             for i, (start, code, title) in enumerate(heads)]
    out: dict[str, str] = {}
    for code, prefix, label in WANTED.get(form_type, []):
        cands = [s for s in spans if s[0] == code]
        cands = [s for s in cands if s[1].startswith(prefix)] or cands
        if cands:
            _, _, a, b = max(cands, key=lambda s: s[3] - s[2])
            out[label] = text[a:b].strip()
    return out or {FULL_TEXT: text}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """~size-char windows cut at a sentence end in the back half; next window starts ~overlap
    chars earlier on a word boundary."""
    text = text.strip()
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n and (cut := text.rfind(". ", start + size // 2, end)) != -1:
            end = cut + 1
        if piece := text[start:end].strip():
            chunks.append(piece)
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        sp = text.find(" ", nxt, end)
        start = sp + 1 if sp != -1 else nxt
    return chunks


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def filing_sections(filing: FilingMeta, html: str) -> dict[str, str]:
    text = html_to_text(html)
    if filing.form_type == "8-K":
        return {"8-K Items " + ",".join(filing.items) if filing.items else "8-K": text}
    return split_sections(text, filing.form_type)


def build_records(ticker: str, filing: FilingMeta, sections: dict[str, str],
                  retrieved_at: datetime, max_chunks: int = MAX_CHUNKS_PER_FILING,
                  ) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for section, body in sections.items():
        slug = slugify(section)
        for idx, piece in enumerate(chunk_text(body)):
            if len(records) >= max_chunks:
                log.warning("capped %s at %d chunks", filing.accession_no, max_chunks)
                return records
            records.append(ChunkRecord(f"{filing.accession_no}:{slug}:{idx}", piece, {
                "ticker": ticker, "cik": filing.cik, "accession_no": filing.accession_no,
                "form_type": filing.form_type, "filed_date": filing.filed_date.isoformat(),
                "section": section, "url": filing.primary_doc_url, "chunk_idx": idx,
                "retrieved_at": retrieved_at.isoformat()}))
    return records
```

- [ ] **Step 4: Run, expect PASS.** `uv run pytest tests/ws2_ingest -q` → `19 passed`.
- [ ] **Step 5: Commit.** `git add app/ingest.py tests/ws2_ingest/test_parsing.py && git commit -m "feat(ws2): text extraction, TOC-proof sectioning, chunking"`

---

### Task 5: `ingest_recent_filings` + `retrieve` (+ live test)

**Files:** Modify `backend/app/ingest.py` (imports + append); Create `backend/tests/ws2_ingest/test_ingest.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces (§3.2, public): `ingest_recent_filings(ticker, *, client=None, collection=None, today=None) -> list[FilingMeta]`. It returns the filings now in the store, including ones stored earlier and excluding failed fetches. It raises `TickerNotFoundError` for an unknown ticker, and `DataSourceError` if every selected filing failed. Also `retrieve(ticker, query, k=6, form_type=None, *, collection=None) -> list[Chunk]`. Tickers are normalised (`BRK.B` → `BRK-B`) on both write and read.

- [ ] **Step 1: Failing tests.** Create `backend/tests/ws2_ingest/test_ingest.py`:
```python
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
```

- [ ] **Step 2: Run, expect FAIL.** `uv run pytest tests/ws2_ingest/test_ingest.py -q` → `ImportError: cannot import name 'ingest_recent_filings'`.

- [ ] **Step 3: Implement.** Replace the import block of `ingest.py` (from `import logging` through `from app.vectorstore import ChunkRecord`) with:
```python
import logging
import re
import warnings
from datetime import UTC, date, datetime

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from chromadb.api.models.Collection import Collection

from app import vectorstore
from app.models import Chunk, DataSourceError, FilingMeta
from app.sources import edgar
from app.tracing import traceable
from app.vectorstore import ChunkRecord
```
and append:
```python
@traceable(run_type="chain", name="ingest.ingest_recent_filings")
def ingest_recent_filings(ticker: str, *, client: edgar.EdgarClient | None = None,
                          collection: Collection | None = None,
                          today: date | None = None) -> list[FilingMeta]:
    """Fetch + store the selected filings. Idempotent: stored accession_nos are not re-fetched.
    A filing whose document fails is skipped (and retried on the next call)."""
    client = client or edgar.default_client()
    col = collection if collection is not None else vectorstore.get_collection()
    symbol = edgar.normalize_ticker(ticker)
    cik, _ = edgar.resolve_company(symbol, client=client)
    as_of = today or datetime.now(UTC).date()
    filings = edgar.select_filings(client.get_submissions(cik), cik, as_of)
    stored: list[FilingMeta] = []
    for filing in filings:
        if vectorstore.has_accession(col, filing.accession_no):
            stored.append(filing)
            continue
        try:
            html = client.get_document(filing.primary_doc_url)
        except DataSourceError as e:
            log.warning("skipping %s %s: %s", filing.form_type, filing.accession_no, e)
            continue
        now = datetime.now(UTC)
        records = build_records(symbol, filing, filing_sections(filing, html), now)
        vectorstore.add_records(col, records)
        stored.append(filing)
    if filings and not stored:
        raise DataSourceError("sec_edgar", f"no filings could be fetched for {symbol}")
    return stored


@traceable(run_type="tool", name="ingest.retrieve")
def retrieve(ticker: str, query: str, k: int = 6, form_type: str | None = None, *,
             collection: Collection | None = None) -> list[Chunk]:
    col = collection if collection is not None else vectorstore.get_collection()
    return vectorstore.query_chunks(col, edgar.normalize_ticker(ticker), query, k, form_type)
```

- [ ] **Step 4: Run, expect PASS.** `uv run pytest tests/ws2_ingest -q` → `23 passed, 1 deselected` (~2 s). Then `uv run ruff check --fix app/sources/edgar.py app/ingest.py app/vectorstore.py tests/ws2_ingest` → `All checks passed!` (only import-sort / unused-noqa autofixes expected).
- [ ] **Step 5: Live check (once).** `uv run pytest tests/ws2_ingest -q -m live` → `1 passed` (~5 s with the model cached; uses `SEC_USER_AGENT` from `.env`).
- [ ] **Step 6: Commit.** `git add app/ingest.py tests/ws2_ingest/test_ingest.py && git commit -m "feat(ws2): idempotent ingest_recent_filings and cited retrieve"`

---

### Task 6 (OPTIONAL STRETCH, only if Tasks 1–5 are green with time left): 8-K EX-99.1

**Files:** Modify `backend/app/sources/edgar.py`, `backend/app/ingest.py`, `backend/tests/ws2_ingest/helpers.py`, `backend/tests/ws2_ingest/conftest.py`; Create `backend/tests/ws2_ingest/test_exhibits.py`

**Interfaces:** Produces `EdgarClient.get_filing_index(cik, accession_no) -> list[str]` and `exhibit_991_url(names, cik, accession_no) -> str | None`. 8-K ingest adds chunks with section `"8-K Exhibit 99.1"` and `url` = the exhibit URL. This is best effort: any EDGAR failure means no exhibit chunks, and the primary doc is still stored. The real AAPL exhibit name, verified live, is `a8-kex991q3202606272026.htm`.

- [ ] **Step 1: Scaffolding.** In `helpers.py` add `INDEX_8K = f"{ARCH}/000032019326000018/index.json"`. In the conftest `edgar_api`, before `yield mock`, add `mock.get(h.INDEX_8K, name="index_8k").respond(404)  # no exhibit unless a test says so`.
- [ ] **Step 2: Failing tests.** Create `backend/tests/ws2_ingest/test_exhibits.py`, then run `uv run pytest tests/ws2_ingest/test_exhibits.py -q` → FAIL (`cannot import name 'exhibit_991_url'`):
```python
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
```
- [ ] **Step 3: Implement.** In `edgar.py`, add `import re` and add this method to `EdgarClient`:
```python
    @traceable(run_type="tool", name="edgar.get_filing_index")
    def get_filing_index(self, cik: str, accession_no: str) -> list[str]:
        data = self._get(f"{filing_base_url(cik, accession_no)}/index.json").json()
        return [item["name"] for item in data["directory"]["item"]]
```
then append:
```python
EX_991_RE = re.compile(r"(?:exhibit|ex)[-_]?99[-_.]?0?1(?!\d).*\.html?$", re.IGNORECASE)


def exhibit_991_url(names: list[str], cik: str, accession_no: str) -> str | None:
    hit = next((n for n in names if EX_991_RE.search(n)), None)
    return f"{filing_base_url(cik, accession_no)}/{hit}" if hit else None
```
In `ingest.py`, add above `ingest_recent_filings`:
```python
def _exhibit_records(client: edgar.EdgarClient, ticker: str, filing: FilingMeta,
                     now: datetime) -> list[ChunkRecord]:
    try:
        names = client.get_filing_index(filing.cik, filing.accession_no)
        url = edgar.exhibit_991_url(names, filing.cik, filing.accession_no)
        if url is None:
            return []
        html = client.get_document(url)
    except DataSourceError as e:
        log.warning("no EX-99.1 for %s: %s", filing.accession_no, e)
        return []
    exhibit = filing.model_copy(update={"primary_doc_url": url})
    return build_records(ticker, exhibit, {"8-K Exhibit 99.1": html_to_text(html)}, now)
```
and in `ingest_recent_filings`, between `records = build_records(...)` and `vectorstore.add_records(...)`, add:
```python
        if filing.form_type == "8-K":
            records += _exhibit_records(client, symbol, filing, now)
```
- [ ] **Step 4: Run.** `uv run pytest tests/ws2_ingest -q` → `25 passed, 1 deselected`, then `ruff check --fix` should report clean.
- [ ] **Step 5: Commit.** `git add app tests/ws2_ingest && git commit -m "feat(ws2): ingest 8-K EX-99.1 press release (stretch)"`

---

## Done checklist

- [ ] `uv run pytest tests/ws2_ingest -q` is green (23 tests, or 25 with the stretch), runs in under 10 s, and makes no network calls.
- [ ] `uv run pytest tests/ws2_ingest -q -m live` → 1 passed.
- [ ] `uv run ruff check` is clean for owned files.
- [ ] `resolve_company`, `ingest_recent_filings` and `retrieve` are importable with the §3.2 signatures (extra args keyword-only).
- [ ] `@traceable` is on `edgar.get_company_tickers`, `get_submissions`, `get_document` and `resolve_company`, on `ingest.ingest_recent_filings` (chain) and on `ingest.retrieve` (tool).
- [ ] No edits outside owned files; contracts, fixtures and `tests/conftest.py` are untouched.

**Report status to the human: tests, the live result, whether the stretch landed, and deviations. The known deviations are the extra `retrieved_at` metadata key and chunk cuts at sentence ends rather than paragraphs, because whitespace is collapsed first. Do not merge; the integrator merges in Phase 2.**
