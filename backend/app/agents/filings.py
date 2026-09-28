"""Filings agent (WS3): cited summaries and follow-up Q&A over retrieved filing chunks.

Implements `summarize_filings` and `answer_filings_question` exactly as the
Protocols in app/contracts.py. Knows nothing about LangGraph.
"""

from __future__ import annotations

import re

from anthropic import beta_tool
from pydantic import BaseModel, Field

from app.contracts import Retriever
from app.llm import LLM
from app.models import AgentAnswer, ChatTurn, Chunk, Claim, FilingMeta, FilingsSummary
from app.prompts.filings import (
    CHAT_SYSTEM,
    SUMMARY_SYSTEM,
    build_summary_user,
    format_chunks,
)
from app.tracing import traceable

K_PER_QUERY = 6
MAX_PROMPT_CHUNKS = 24
HISTORY_TURNS = 6
FORM_TYPES = {"10-K", "10-Q", "8-K"}

# (query, form_type) per summary section; order = priority when the chunk cap bites.
SECTION_QUERIES: dict[str, list[tuple[str, str | None]]] = {
    "risk_factors": [
        ("principal risk factors uncertainties adverse effect", "10-K"),
        ("risk factors changes since annual report", "10-Q"),
    ],
    "material_events": [("material event announcement press release results", "8-K")],
    "financial_highlights": [("net sales revenue gross margin results of operations", None)],
    "key_developments": [("significant developments strategy outlook products", None)],
}

# Matches ids like 0000320193-25-000079:item-1a:0 or ...:part1-item-2:2 or ...:8k-items:0
CHUNK_ID_RE = re.compile(r"\d{10}-\d{2}-\d{6}:[A-Za-z0-9.\-]+:\d+")


class _SummaryDraft(BaseModel):
    key_developments: list[Claim] = Field(default_factory=list)
    risk_factors: list[Claim] = Field(default_factory=list)
    financial_highlights: list[Claim] = Field(default_factory=list)
    material_events: list[Claim] = Field(default_factory=list)


_SECTIONS = ("key_developments", "risk_factors", "financial_highlights", "material_events")


def _gather_chunks(ticker: str, retriever: Retriever) -> list[Chunk]:
    """Run every section query, dedupe by chunk_id (first hit wins), cap the total."""
    seen: dict[str, Chunk] = {}
    for queries in SECTION_QUERIES.values():
        for query, form_type in queries:
            for chunk in retriever(ticker, query, K_PER_QUERY, form_type):
                seen.setdefault(chunk.chunk_id, chunk)
    return list(seen.values())[:MAX_PROMPT_CHUNKS]


def _clean_claims(claims: list[Claim], known: set[str]) -> list[Claim]:
    """Local grounding pre-guard: trim unknown citation ids; drop claims left uncited."""
    kept = []
    for claim in claims:
        valid = [cid for cid in dict.fromkeys(claim.citations) if cid in known]
        if valid:
            kept.append(Claim(text=claim.text, citations=valid))
    return kept


@traceable(run_type="chain", name="filings.summarize_filings")
def summarize_filings(
    ticker: str,
    company_name: str,
    filings: list[FilingMeta],
    retriever: Retriever,
    llm: LLM,
) -> FilingsSummary:
    chunks = _gather_chunks(ticker, retriever)
    if not chunks:  # nothing ingested/retrieved: an LLM call could only hallucinate
        return FilingsSummary(ticker=ticker, company_name=company_name, filings=filings)
    user = build_summary_user(ticker, company_name, filings, chunks)
    draft = llm.parse("agent", SUMMARY_SYSTEM, [{"role": "user", "content": user}], _SummaryDraft)
    index = {c.chunk_id: c.source for c in chunks}
    sections = {name: _clean_claims(getattr(draft, name), set(index)) for name in _SECTIONS}
    return FilingsSummary(
        ticker=ticker,
        company_name=company_name,
        filings=filings,
        citation_index=index,
        **sections,
    )
