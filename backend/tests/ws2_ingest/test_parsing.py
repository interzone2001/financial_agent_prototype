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
