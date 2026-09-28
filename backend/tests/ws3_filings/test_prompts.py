from app.contracts import fixture_chunks, stub_ingest_recent_filings
from app.prompts.filings import CHAT_SYSTEM, SUMMARY_SYSTEM, build_summary_user, format_chunks


def test_format_chunks_lists_id_form_section_date_then_text():
    c = fixture_chunks()[0]
    out = format_chunks([c])
    header, body = out.split("\n", 1)
    assert header == (
        "[0000320193-25-000079:item-1a:0] (10-K, Item 1A. Risk Factors, filed 2025-10-31)"
    )
    assert body.startswith("Item 1A. Risk Factors")


def test_summary_user_prompt_contains_every_chunk_id_and_filings():
    chunks = fixture_chunks()[:3]
    user = build_summary_user("AAPL", "Apple Inc.", stub_ingest_recent_filings("AAPL"), chunks)
    for c in chunks:
        assert f"[{c.chunk_id}]" in user
    assert "Apple Inc. (AAPL)" in user
    assert "10-K filed 2025-10-31" in user
    assert "<filing_excerpts>" in user


def test_system_prompts_carry_the_ground_rules():
    chat = CHAT_SYSTEM.format(ticker="AAPL")
    for prompt in (SUMMARY_SYSTEM, chat):
        assert "not disclosed in retrieved filings" in prompt
        assert "price targets" in prompt  # no-advice rule
        assert "not instructions" in prompt  # injection hygiene
    assert "3-5 claims per section" in SUMMARY_SYSTEM
    assert "AAPL" in chat and "search_filings" in chat
