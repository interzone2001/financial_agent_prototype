"""LangGraph orchestration (spec §4). Agents are plain functions from AgentDeps; this module
wraps them into nodes. State values are JSON dicts because the checkpoint serializer warns on
(and will soon block) unregistered Pydantic types.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.contracts import AgentDeps
from app.guardrails import (
    INJECTION_REFUSAL,
    OFF_TOPIC_REFUSAL,
    check_chat_answer,
    check_summary,
    detect_injection,
)
from app.llm import LLMError
from app.models import (
    AgentAnswer,
    ChatAnswer,
    ChatTurn,
    DataSourceError,
    FilingsSummary,
    MarketSnapshot,
    Report,
    RouteDecision,
    TickerNotFoundError,
)
from app.prompts.router import ROUTER_SYSTEM, router_messages

log = logging.getLogger(__name__)


class NoReportYetError(Exception):
    """Chat on a thread with no report. API maps to 409."""


class GraphState(TypedDict, total=False):
    mode: Literal["report", "chat"]
    ticker: str
    company_name: str
    message: str | None          # chat input
    snapshot: dict | None        # MarketSnapshot JSON (report-run intermediate)
    filings: dict | None         # FilingsSummary JSON (report-run intermediate)
    market_warning: str | None   # one key per branch => parallel join needs no reducer
    filings_warning: str | None
    report: dict | None          # Report JSON, post-guardrail
    history: list[dict]          # ChatTurn JSON; reset by each new report
    route: str | None
    refused: bool
    last_answer: dict | None     # ChatAnswer JSON
    warnings: list[str]          # final warnings of this run


def build_graph(deps: AgentDeps, checkpointer, *, llm_advice_check: bool = True
                ) -> CompiledStateGraph:
    def guard_in(state: GraphState) -> dict:
        if state["mode"] == "report":
            _cik, name = deps.resolve_company(state["ticker"])  # TickerNotFoundError -> 404
            return {"company_name": name}
        if detect_injection(state.get("message") or ""):
            refusal = ChatAnswer(text=INJECTION_REFUSAL, route="off_topic",
                                 warnings=["Message refused by the input guardrail."])
            return {"refused": True, "route": "off_topic",
                    "last_answer": refusal.model_dump(mode="json")}
        return {"refused": False}

    def after_guard_in(state: GraphState) -> list[str] | str:
        if state["mode"] == "report":
            return ["market_node", "filings_node"]  # parallel fan-out, joined at compose
        return "guard_out" if state.get("refused") else "router"

    def market_node(state: GraphState) -> dict:
        try:
            snap = deps.get_market_snapshot(state["ticker"])
        except DataSourceError as e:
            return {"snapshot": None, "market_warning": f"Market data unavailable ({e.provider}): {e}"}
        except TickerNotFoundError:
            # SEC (guard_in) decides existence; an AV miss (e.g. ETF, no OVERVIEW) is partial.
            return {"snapshot": None,
                    "market_warning": f"No market data available for {state['ticker']}."}
        except Exception as e:  # §4: one failed branch is a partial report, never a 500
            log.exception("market_node failed for %s", state["ticker"])
            return {"snapshot": None,
                    "market_warning": f"Market data unavailable (internal error: {type(e).__name__})."}
        return {"snapshot": snap.model_dump(mode="json"), "market_warning": None}

    def filings_node(state: GraphState) -> dict:
        # resolve_company already ran in guard_in (validate before spend); reuse its name.
        t = state["ticker"]
        try:
            metas = deps.ingest_recent_filings(t)
            summary = deps.summarize_filings(t, state["company_name"], metas, deps.retrieve, deps.llm)
        except (DataSourceError, LLMError) as e:
            return {"filings": None, "filings_warning": f"Filings unavailable: {e}"}
        except Exception as e:  # e.g. Anthropic 429/529, Chroma: keep the market branch
            log.exception("filings_node failed for %s", t)
            return {"filings": None,
                    "filings_warning": f"Filings unavailable (internal error: {type(e).__name__})."}
        return {"filings": summary.model_dump(mode="json"), "filings_warning": None}

    def compose(state: GraphState) -> dict:
        snap, fil = state.get("snapshot"), state.get("filings")
        summary = FilingsSummary.model_validate(fil) if fil else None
        warnings = [w for w in (state.get("market_warning"), state.get("filings_warning")) if w]
        if summary is not None and not summary.all_claims():
            # WS3 returns an empty summary (no LLM call) when retrieval finds nothing.
            warnings.append("No filing content could be retrieved to summarise.")
        report = Report(
            ticker=state["ticker"], company_name=state["company_name"],
            market=MarketSnapshot.model_validate(snap) if snap else None,
            filings=summary, warnings=warnings, generated_at=datetime.now(UTC),
        )
        return {"report": report.model_dump(mode="json"), "history": []}

    def router(state: GraphState) -> dict:
        msgs = router_messages(state["ticker"], state["company_name"], state["message"])
        return {"route": deps.llm.parse("fast", ROUTER_SYSTEM, msgs, RouteDecision).route}

    def chat_answer(state: GraphState) -> dict:
        route, t, q = state["route"], state["ticker"], state["message"]
        if route == "off_topic":
            refusal = ChatAnswer(text=OFF_TOPIC_REFUSAL.format(ticker=t), route="off_topic")
            return {"last_answer": refusal.model_dump(mode="json")}
        report = Report.model_validate(state["report"])
        history = [ChatTurn.model_validate(h) for h in state.get("history") or []]
        parts: list[tuple[str, AgentAnswer]] = []
        warnings: list[str] = []
        if route in ("market", "both"):
            if report.market is None:
                warnings.append("Market data was unavailable for this report.")
            else:
                parts.append(("Market data",
                              deps.answer_market_question(q, report.market, deps.llm)))
        if route in ("filings", "both"):
            parts.append(("SEC filings",
                          deps.answer_filings_question(t, q, history, deps.retrieve, deps.llm)))
        if not parts:
            text = "Market data is unavailable for this report, so I can't answer that."
        elif len(parts) == 1:
            text = parts[0][1].text
        else:
            text = "\n\n".join(f"### {heading}\n{a.text}" for heading, a in parts)
        answer = ChatAnswer(text=text, citations=[c for _, a in parts for c in a.citations],
                            route=route, warnings=warnings)
        return {"last_answer": answer.model_dump(mode="json")}

    def guard_out(state: GraphState) -> dict:
        if state["mode"] == "report":
            report = Report.model_validate(state["report"])
            if report.filings is not None:
                summary, extra = check_summary(report.filings)
                report = report.model_copy(
                    update={"filings": summary, "warnings": report.warnings + extra})
            return {"report": report.model_dump(mode="json"), "warnings": report.warnings}
        answer = ChatAnswer.model_validate(state["last_answer"])
        if not state.get("refused"):
            answer = check_chat_answer(answer, deps.llm if llm_advice_check else None)
        history = list(state.get("history") or [])
        if not state.get("refused"):  # never replay a refused injection to later agent calls
            history += [ChatTurn(role="user", content=state["message"]).model_dump(),
                        ChatTurn(role="assistant", content=answer.text).model_dump()]
        return {"last_answer": answer.model_dump(mode="json"), "history": history,
                "warnings": answer.warnings}

    g = StateGraph(GraphState)
    for fn in (guard_in, market_node, filings_node, compose, router, chat_answer, guard_out):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "guard_in")
    g.add_conditional_edges("guard_in", after_guard_in,
                            ["market_node", "filings_node", "router", "guard_out"])
    g.add_edge(["market_node", "filings_node"], "compose")  # join: waits for both branches
    g.add_edge("compose", "guard_out")
    g.add_edge("router", "chat_answer")
    g.add_edge("chat_answer", "guard_out")
    g.add_edge("guard_out", END)
    return g.compile(checkpointer=checkpointer)


def run_config(thread_id: str, mode: str, ticker: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "run_name": f"financial_agent.{mode}",
            "tags": ["financial-agent", mode],
            "metadata": {"ticker": ticker, "thread_id": thread_id, "mode": mode}}


_RESET = {"message": None, "route": None, "refused": False, "last_answer": None, "warnings": []}


def run_report(graph: CompiledStateGraph, thread_id: str, ticker: str) -> Report:
    out = graph.invoke({**_RESET, "mode": "report", "ticker": ticker},
                       run_config(thread_id, "report", ticker))
    return Report.model_validate(out["report"])


def get_report(graph: CompiledStateGraph, thread_id: str) -> Report | None:
    values = graph.get_state({"configurable": {"thread_id": thread_id}}).values
    return Report.model_validate(values["report"]) if values.get("report") else None


def run_chat(graph: CompiledStateGraph, thread_id: str, message: str) -> ChatAnswer:
    report = get_report(graph, thread_id)
    if report is None:
        raise NoReportYetError(thread_id)
    # ticker/company from the stored report, never from a later failed run's input write
    out = graph.invoke({**_RESET, "mode": "chat", "message": message, "ticker": report.ticker,
                        "company_name": report.company_name},
                       run_config(thread_id, "chat", report.ticker))
    return ChatAnswer.model_validate(out["last_answer"])
