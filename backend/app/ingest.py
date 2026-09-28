"""EDGAR HTML -> text -> sections -> chunks -> Chroma; plus cited retrieval (spec §6.2)."""

from __future__ import annotations

import logging
import re
import warnings
from datetime import UTC, date, datetime

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from chromadb.api.models.Collection import Collection

from app import vectorstore
from app.models import Chunk, DataSourceError, FilingMeta, TickerNotFoundError
from app.sources import edgar
from app.tracing import traceable
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
             collection: Collection | None = None, client: edgar.EdgarClient | None = None,
             ) -> list[Chunk]:
    """Filters by CIK so a company's share classes (same CIK, different ticker) share chunks."""
    col = collection if collection is not None else vectorstore.get_collection()
    symbol = edgar.normalize_ticker(ticker)
    try:
        cik, _ = edgar.resolve_company(symbol, client=client)
    except TickerNotFoundError:
        return []
    return vectorstore.query_chunks(col, cik, symbol, query, k, form_type)
