"""Shared Pydantic contracts. FROZEN after Phase 0 — see spec §3.1.

Every fact that reaches an advisor carries a SourceRef. That is the attribution
guarantee: if a value has no SourceRef, it has no business being on screen.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Provider = Literal["alpha_vantage", "sec_edgar"]
FormType = Literal["10-K", "10-Q", "8-K"]
Route = Literal["market", "filings", "both", "off_topic"]

TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,5}$")

DISCLAIMER = (
    "For informational purposes only. Not investment advice. Data from SEC EDGAR "
    "and Alpha Vantage; verify against original sources before acting."
)


class SourceRef(BaseModel):
    """Where a fact came from. Attached to every quote, overview and filing chunk."""

    provider: Provider
    url: str  # human-clickable source URL (API keys redacted)
    retrieved_at: datetime  # UTC
    accession_no: str | None = None  # filings only, e.g. "0000320193-24-000123"
    form_type: str | None = None  # "10-K" | "10-Q" | "8-K"
    filed_date: date | None = None
    section: str | None = None  # e.g. "Item 1A. Risk Factors"
    chunk_id: str | None = None  # "{accession_no}:{section_slug}:{idx}"


# --- Market data -------------------------------------------------------------


class Quote(BaseModel):
    symbol: str
    price: float
    change: float
    change_pct: float  # percent units: 1.23 means +1.23%
    volume: int
    latest_trading_day: date
    source: SourceRef


class Overview(BaseModel):
    symbol: str
    name: str
    sector: str | None = None
    industry: str | None = None
    market_cap: int | None = None
    pe_ratio: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    description: str | None = None
    source: SourceRef


class MarketSnapshot(BaseModel):
    quote: Quote
    overview: Overview


# --- Filings -----------------------------------------------------------------


class FilingMeta(BaseModel):
    cik: str  # 10-digit zero-padded
    accession_no: str
    form_type: FormType
    filed_date: date
    report_date: date | None = None
    primary_doc_url: str
    items: list[str] = Field(default_factory=list)  # 8-K item codes, e.g. ["2.02"]


class Chunk(BaseModel):
    chunk_id: str
    ticker: str
    text: str
    source: SourceRef


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(min_length=1)  # chunk_ids — no uncited claims


class FilingsSummary(BaseModel):
    ticker: str
    company_name: str
    filings: list[FilingMeta]
    key_developments: list[Claim] = Field(default_factory=list)
    risk_factors: list[Claim] = Field(default_factory=list)
    financial_highlights: list[Claim] = Field(default_factory=list)
    material_events: list[Claim] = Field(default_factory=list)
    citation_index: dict[str, SourceRef] = Field(default_factory=dict)  # chunk_id -> source

    def all_claims(self) -> list[Claim]:
        return (
            self.key_developments
            + self.risk_factors
            + self.financial_highlights
            + self.material_events
        )


# --- Report & chat -----------------------------------------------------------


class Report(BaseModel):
    ticker: str
    company_name: str
    market: MarketSnapshot | None = None  # None => market branch failed; see warnings
    filings: FilingsSummary | None = None  # None => filings branch failed; see warnings
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER
    generated_at: datetime


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RouteDecision(BaseModel):
    route: Route
    reason: str


class AgentAnswer(BaseModel):
    """What an agent returns for one chat turn."""

    text: str
    citations: list[SourceRef] = Field(default_factory=list)


class ChatAnswer(BaseModel):
    """What the API returns for one chat turn."""

    text: str
    citations: list[SourceRef] = Field(default_factory=list)
    route: Route
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


# --- API I/O -----------------------------------------------------------------


class CreateSessionResponse(BaseModel):
    thread_id: str


class ReportRequest(BaseModel):
    thread_id: str
    ticker: str

    @field_validator("ticker")
    @classmethod
    def _normalise_ticker(cls, v: str) -> str:
        v = v.strip().upper()
        if not TICKER_RE.match(v):
            raise ValueError("ticker must be 1-6 chars: letters, '.' or '-', starting with a letter")
        return v


class ChatRequest(BaseModel):
    thread_id: str
    message: str = Field(min_length=1, max_length=1000)


class ErrorResponse(BaseModel):
    error_code: str
    message: str


# --- Errors ------------------------------------------------------------------


class DataSourceError(Exception):
    """An upstream data provider failed. Nodes turn this into a report warning."""

    def __init__(self, provider: Provider, message: str):
        super().__init__(message)
        self.provider = provider


class RateLimitedError(DataSourceError):
    """Provider refused due to quota (Alpha Vantage: HTTP 200 with a Note/Information body)."""


class TickerNotFoundError(Exception):
    """Ticker is not a known SEC registrant / has no quote. API maps to 404."""

    def __init__(self, ticker: str):
        super().__init__(f"Ticker not found: {ticker}")
        self.ticker = ticker
