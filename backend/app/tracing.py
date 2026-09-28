"""LangSmith tracing — REQUIRED (spec §3.6).

Tracing is controlled by the standard LangSmith env flag:
    LANGSMITH_TRACING=true  LANGSMITH_API_KEY=...  LANGSMITH_PROJECT=financial-agent-prototype

LangGraph runs are traced automatically when the flag is on. Everything else that
does real work (agent functions, data-source calls, guardrails) must be decorated
with `@traceable` from THIS module so run names/types stay consistent:

    from app.tracing import traceable

    @traceable(run_type="tool", name="market.get_market_snapshot")
    def get_market_snapshot(ticker: str) -> MarketSnapshot: ...

run_type conventions: "chain" = agent/orchestration step, "tool" = data source or
retrieval, "llm" = model call (app.llm only), "parser" = guardrail check.
Tests run with tracing forced off (tests/conftest.py).
"""

from __future__ import annotations

import os

from langsmith import traceable

__all__ = ["traceable", "tracing_enabled"]


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true"
