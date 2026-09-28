"""Workstream interfaces + fixture-backed stubs. FROZEN after Phase 0 — spec §3.2.

Each workstream implements its functions in its own module with EXACTLY these
signatures. Until then, `stub_deps()` lets everyone else run against fixture data.
Phase 2 adds `real_deps()` here, wiring the real modules in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

from app.llm import LLM
from app.models import (
    AgentAnswer,
    ChatTurn,
    Chunk,
    Claim,
    FilingMeta,
    FilingsSummary,
    MarketSnapshot,
    Overview,
    Quote,
    SourceRef,
    TickerNotFoundError,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# Same shape as ws2's `retrieve(ticker, query, k=6, form_type=None)`.
Retriever = Callable[[str, str, int, str | None], list[Chunk]]


# --- Protocols (one per public function; implementations must match) --------


class GetMarketSnapshot(Protocol):  # WS1 app/agents/market.py
    def __call__(self, ticker: str) -> MarketSnapshot: ...


class AnswerMarketQuestion(Protocol):  # WS1 app/agents/market.py
    def __call__(self, question: str, snapshot: MarketSnapshot, llm: LLM) -> AgentAnswer: ...


class ResolveCompany(Protocol):  # WS2 app/sources/edgar.py -> (cik, company_name)
    def __call__(self, ticker: str) -> tuple[str, str]: ...


class IngestRecentFilings(Protocol):  # WS2 app/ingest.py
    def __call__(self, ticker: str) -> list[FilingMeta]: ...


class SummarizeFilings(Protocol):  # WS3 app/agents/filings.py
    def __call__(
        self,
        ticker: str,
        company_name: str,
        filings: list[FilingMeta],
        retriever: Retriever,
        llm: LLM,
    ) -> FilingsSummary: ...


class AnswerFilingsQuestion(Protocol):  # WS3 app/agents/filings.py
    def __call__(
        self,
        ticker: str,
        question: str,
        history: list[ChatTurn],
        retriever: Retriever,
        llm: LLM,
    ) -> AgentAnswer: ...


@dataclass
class AgentDeps:
    """Everything the graph (WS4) needs. Integration swaps stubs -> real functions."""

    get_market_snapshot: GetMarketSnapshot
    answer_market_question: AnswerMarketQuestion
    resolve_company: ResolveCompany
    ingest_recent_filings: IngestRecentFilings
    retrieve: Retriever
    summarize_filings: SummarizeFilings
    answer_filings_question: AnswerFilingsQuestion
    llm: LLM


# --- Fixture-backed stubs ------------------------------------------------------

_STUB_TIME = datetime(2026, 9, 28, 18, 0, tzinfo=UTC)
_KNOWN = {"AAPL": ("0000320193", "Apple Inc.")}


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


def stub_market_snapshot(ticker: str) -> MarketSnapshot:
    if ticker not in _KNOWN:
        raise TickerNotFoundError(ticker)
    q = _load("av_global_quote_AAPL.json")["Global Quote"]
    o = _load("av_overview_AAPL.json")
    url = "https://www.alphavantage.co/query?function={}&symbol=AAPL&apikey=REDACTED"
    src = lambda fn: SourceRef(
        provider="alpha_vantage", url=url.format(fn), retrieved_at=_STUB_TIME
    )
    return MarketSnapshot(
        quote=Quote(
            symbol=q["01. symbol"],
            price=float(q["05. price"]),
            change=float(q["09. change"]),
            change_pct=float(q["10. change percent"].rstrip("%")),
            volume=int(q["06. volume"]),
            latest_trading_day=date.fromisoformat(q["07. latest trading day"]),
            source=src("GLOBAL_QUOTE"),
        ),
        overview=Overview(
            symbol=o["Symbol"],
            name=o["Name"],
            sector=o["Sector"],
            industry=o["Industry"],
            market_cap=int(o["MarketCapitalization"]),
            pe_ratio=float(o["PERatio"]),
            week52_high=float(o["52WeekHigh"]),
            week52_low=float(o["52WeekLow"]),
            description=o["Description"],
            source=src("OVERVIEW"),
        ),
    )


def stub_answer_market_question(question: str, snapshot: MarketSnapshot, llm: LLM) -> AgentAnswer:
    q = snapshot.quote
    return AgentAnswer(
        text=f"{q.symbol} last traded at ${q.price:.2f} ({q.change_pct:+.2f}%) on "
        f"{q.latest_trading_day}.",
        citations=[q.source, snapshot.overview.source],
    )


def stub_resolve_company(ticker: str) -> tuple[str, str]:
    if ticker not in _KNOWN:
        raise TickerNotFoundError(ticker)
    return _KNOWN[ticker]


def fixture_chunks() -> list[Chunk]:
    return [Chunk.model_validate(c) for c in _load("chunks_AAPL.json")]


def stub_ingest_recent_filings(ticker: str) -> list[FilingMeta]:
    stub_resolve_company(ticker)
    seen: dict[str, FilingMeta] = {}
    for c in fixture_chunks():
        s = c.source
        if s.accession_no not in seen:
            seen[s.accession_no] = FilingMeta(
                cik="0000320193",
                accession_no=s.accession_no,
                form_type=s.form_type,
                filed_date=s.filed_date,
                primary_doc_url=s.url,
                items=["2.02", "9.01"] if s.form_type == "8-K" else [],
            )
    return list(seen.values())


def fixture_retriever(
    ticker: str, query: str, k: int = 6, form_type: str | None = None
) -> list[Chunk]:
    """Naive keyword-overlap ranking over fixture chunks. Good enough for tests."""
    words = set(re.findall(r"[a-z]{4,}", query.lower()))
    pool = [
        c
        for c in fixture_chunks()
        if c.ticker == ticker and (form_type is None or c.source.form_type == form_type)
    ]
    pool.sort(key=lambda c: -len(words & set(re.findall(r"[a-z]{4,}", c.text.lower()))))
    return pool[:k]


def stub_summarize_filings(
    ticker: str, company_name: str, filings: list[FilingMeta], retriever: Retriever, llm: LLM
) -> FilingsSummary:
    chunks = fixture_chunks()
    by_form = lambda f: [c for c in chunks if c.source.form_type == f]
    k, q, e = by_form("10-K"), by_form("10-Q"), by_form("8-K")
    claim = lambda text, cs: Claim(text=text, citations=[c.chunk_id for c in cs])
    return FilingsSummary(
        ticker=ticker,
        company_name=company_name,
        filings=filings,
        key_developments=[claim("Stub: key development from the latest 10-Q MD&A.", q[:1])],
        risk_factors=[claim("Stub: risk factors summarised from the 10-K Item 1A.", k[:2])],
        financial_highlights=[claim("Stub: results of operations from the 10-K MD&A.", k[4:5])],
        material_events=[claim("Stub: 8-K Item 2.02 reported quarterly results.", e[:1])],
        citation_index={c.chunk_id: c.source for c in (q[:1] + k[:2] + k[4:5] + e[:1])},
    )


def stub_answer_filings_question(
    ticker: str, question: str, history: list[ChatTurn], retriever: Retriever, llm: LLM
) -> AgentAnswer:
    hits = retriever(ticker, question, 2, None)
    return AgentAnswer(
        text="Stub filings answer. " + " ".join(f"[{c.chunk_id}]" for c in hits),
        citations=[c.source for c in hits],
    )


def stub_deps(llm: LLM | None = None) -> AgentDeps:
    if llm is None:
        from app.llm import AnthropicLLM

        llm = AnthropicLLM()
    return AgentDeps(
        get_market_snapshot=stub_market_snapshot,
        answer_market_question=stub_answer_market_question,
        resolve_company=stub_resolve_company,
        ingest_recent_filings=stub_ingest_recent_filings,
        retrieve=fixture_retriever,
        summarize_filings=stub_summarize_filings,
        answer_filings_question=stub_answer_filings_question,
        llm=llm,
    )
