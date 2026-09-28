import os

import pytest

from app.agents.filings import answer_filings_question, summarize_filings
from app.contracts import fixture_retriever, stub_ingest_recent_filings

pytestmark = pytest.mark.live


@pytest.fixture
def llm():
    import app.config  # noqa: F401  -- loads the repo-root .env (app.llm doesn't)

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from app.llm import AnthropicLLM

    return AnthropicLLM()


def test_live_summary_and_chat_are_grounded(llm):
    filings = stub_ingest_recent_filings("AAPL")
    summary = summarize_filings("AAPL", "Apple Inc.", filings, fixture_retriever, llm)
    assert summary.risk_factors, "expected at least one risk-factor claim"
    for claim in summary.all_claims():
        assert set(claim.citations) <= set(summary.citation_index)

    ans = answer_filings_question(
        "AAPL", "What do the filings say about international risk?", [], fixture_retriever, llm
    )
    assert ans.citations and all(c.chunk_id for c in ans.citations)
    assert "[0000320193-" in ans.text
