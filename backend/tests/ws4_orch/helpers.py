import dataclasses

from langgraph.checkpoint.memory import InMemorySaver

from app.contracts import stub_deps, stub_summarize_filings
from app.graph import build_graph
from app.guardrails import AdviceCheck
from app.models import Claim, RouteDecision
from tests.fakes import FakeLLM

VALID_ID = "0000320193-25-000079:item-1a:0"


def deps_with(llm=None, **overrides):
    return dataclasses.replace(stub_deps(llm=llm or FakeLLM()), **overrides)


def make_graph(llm=None, **overrides):
    return build_graph(deps_with(llm, **overrides), InMemorySaver())


def chat_llm(*routes, advice=False):
    """FakeLLM queued per chat turn: RouteDecision, then AdviceCheck unless off_topic."""
    llm = FakeLLM()
    for r in routes:
        llm.queue("parse", RouteDecision(route=r, reason="test"))
        if r != "off_topic":
            llm.queue("parse", AdviceCheck(is_advice=advice, reason="test"))
    return llm


def summarize_plus(*extra: Claim):
    def _s(ticker, company_name, filings, retriever, llm):
        s = stub_summarize_filings(ticker, company_name, filings, retriever, llm)
        return s.model_copy(update={"risk_factors": s.risk_factors + list(extra)})
    return _s


def raiser(exc):
    def _f(*a, **k):
        raise exc
    return _f
