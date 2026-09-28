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
