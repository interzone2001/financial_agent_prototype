from app.agents.filings import MAX_PROMPT_CHUNKS, _SummaryDraft, summarize_filings
from app.contracts import fixture_chunks, fixture_retriever, stub_ingest_recent_filings
from app.models import Claim
from tests.fakes import FakeLLM

K_RISK = "0000320193-25-000079:item-1a:0"
K_MDA = "0000320193-25-000079:item-7:0"
Q_MDA = "0000320193-26-000020:part1-item-2:0"
EIGHT_K = "0000320193-26-000018:8k-items:0"
FAKE = "0000320193-99-999999:item-1a:0"


def _spy(calls):
    def retriever(ticker, query, k=6, form_type=None):
        calls.append((ticker, query, k, form_type))
        return fixture_retriever(ticker, query, k, form_type)

    return retriever


def _run(draft, retriever=fixture_retriever):
    llm = FakeLLM(parse=[draft])
    filings = stub_ingest_recent_filings("AAPL")
    return summarize_filings("AAPL", "Apple Inc.", filings, retriever, llm), llm


def test_retriever_called_with_section_form_filters():
    calls = []
    _run(_SummaryDraft(), retriever=_spy(calls))
    forms = [c[3] for c in calls]
    assert {"10-K", "10-Q", "8-K", None} == set(forms)
    assert all(c[0] == "AAPL" for c in calls)
    risk_forms = {c[3] for c in calls if "risk" in c[1]}
    assert risk_forms == {"10-K", "10-Q"}


def test_happy_path_citation_index_covers_every_cited_id():
    draft = _SummaryDraft(
        key_developments=[Claim(text="Q3 update.", citations=[Q_MDA])],
        risk_factors=[Claim(text="Global economy risk.", citations=[K_RISK])],
        financial_highlights=[Claim(text="Net sales discussed.", citations=[K_MDA])],
        material_events=[Claim(text="Results announced via 8-K.", citations=[EIGHT_K])],
    )
    summary, llm = _run(draft)
    assert summary.ticker == "AAPL" and summary.company_name == "Apple Inc."
    assert len(summary.filings) == 3
    assert len(summary.all_claims()) == 4
    for claim in summary.all_claims():
        assert set(claim.citations) <= set(summary.citation_index)
    assert summary.citation_index[K_RISK].form_type == "10-K"
    assert llm.calls[0].method == "parse" and llm.calls[0].role == "agent"
    assert llm.calls[0].extra is _SummaryDraft


def test_fabricated_citation_dropped_and_mixed_citation_trimmed():
    draft = _SummaryDraft(
        risk_factors=[
            Claim(text="Made-up risk.", citations=[FAKE]),
            Claim(text="Real risk.", citations=[K_RISK, FAKE]),
        ]
    )
    summary, _ = _run(draft)
    assert [c.text for c in summary.risk_factors] == ["Real risk."]
    assert summary.risk_factors[0].citations == [K_RISK]
    assert FAKE not in summary.citation_index


def test_prompt_lists_chunk_ids_and_caps_chunk_count():
    n = iter(range(1000))  # every call returns k brand-new chunks: 5 queries x 6 = 30
    base = fixture_chunks()[0]

    def flood(ticker, query, k=6, form_type=None):
        return [
            base.model_copy(update={"chunk_id": f"{base.chunk_id}-f{next(n)}"}) for _ in range(k)
        ]

    summary, llm = _run(_SummaryDraft(), retriever=flood)
    prompt = llm.calls[0].messages[0]["content"]
    assert prompt.count(f"[{base.chunk_id}-f") == MAX_PROMPT_CHUNKS
    assert len(summary.citation_index) == MAX_PROMPT_CHUNKS
    assert all(f"[{cid}]" in prompt for cid in summary.citation_index)


def test_duplicate_chunks_listed_once():
    _, llm = _run(_SummaryDraft())
    prompt = llm.calls[0].messages[0]["content"]
    assert prompt.count(f"[{K_RISK}]") == 1


def test_no_chunks_returns_empty_summary_without_llm_call():
    llm = FakeLLM()  # nothing queued: any call would raise
    summary = summarize_filings("AAPL", "Apple Inc.", [], lambda *a: [], llm)
    assert summary.all_claims() == [] and summary.citation_index == {}
    assert llm.calls == []
