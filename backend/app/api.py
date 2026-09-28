"""HTTP API (spec §3.4). Routes/response models frozen; internals run the WS4 graph.

Run: cd backend && uv run uvicorn app.api:app --port 8000
Default deps = stub_deps() with an offline stub LLM until Phase 2 wires real_deps().
"""

from __future__ import annotations

import sqlite3
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import data_dir
from app.contracts import AgentDeps, stub_deps
from app.graph import NoReportYetError, build_graph, run_chat, run_report
from app.guardrails import AdviceCheck
from app.llm import LLMError
from app.models import (
    ChatAnswer,
    ChatRequest,
    CreateSessionResponse,
    DataSourceError,
    ErrorResponse,
    Report,
    ReportRequest,
    RouteDecision,
    TickerNotFoundError,
)

_ERRORS = {s: {"model": ErrorResponse} for s in (404, 409, 422, 503)}


def _err(status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error_code=code, message=message).model_dump()
    return JSONResponse(status_code=status, content=body)


class _StubModeLLM:
    """Offline LLM for the no-deps stub app (pre-Phase 2), mirroring the Phase 0 stub: every
    chat routes to filings and nothing reads as advice. Keeps the stub app keyless for WS5 and
    its tests network-free. Stub agents never call text()/run_tools()."""

    def parse(self, role, system, messages, schema):
        if schema is RouteDecision:
            return RouteDecision(route="filings", reason="stub mode")
        if schema is AdviceCheck:
            return AdviceCheck(is_advice=False, reason="stub mode")
        raise LLMError(f"stub mode: no LLM for {schema.__name__}")

    def text(self, role, system, messages):
        raise LLMError("stub mode: no LLM")

    def run_tools(self, role, system, messages, tools):
        raise LLMError("stub mode: no LLM")


def default_checkpointer() -> SqliteSaver:
    return SqliteSaver(sqlite3.connect(data_dir() / "checkpoints.sqlite", check_same_thread=False))


def create_app(deps: AgentDeps | None = None, checkpointer=None) -> FastAPI:
    graph = build_graph(deps if deps is not None else stub_deps(llm=_StubModeLLM()),
                        checkpointer if checkpointer is not None else default_checkpointer())
    app = FastAPI(title="Financial Data Agent")
    app.state.graph = graph
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError):
        return _err(422, "invalid_input", "; ".join(e["msg"] for e in exc.errors()))

    @app.exception_handler(TickerNotFoundError)
    async def _not_found(_: Request, exc: TickerNotFoundError):
        return _err(404, "ticker_not_found", str(exc))

    @app.exception_handler(NoReportYetError)
    async def _no_report(_: Request, exc: NoReportYetError):
        return _err(409, "no_report_yet", "Generate a report for a ticker first.")

    @app.exception_handler(Exception)
    async def _internal(_: Request, exc: Exception):
        # §3.4: all errors are ErrorResponse. Detail stays in server logs, not the body.
        return _err(500, "internal_error", "Unexpected error; please retry.")

    @app.exception_handler(DataSourceError)
    async def _upstream(_: Request, exc: DataSourceError):
        return _err(503, "upstream_unavailable", f"{exc.provider} unavailable: {exc}")

    @app.exception_handler(LLMError)
    async def _llm_down(_: Request, exc: LLMError):
        return _err(503, "upstream_unavailable", f"Language model unavailable: {exc}")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/sessions", response_model=CreateSessionResponse)
    def create_session():
        return CreateSessionResponse(thread_id=str(uuid.uuid4()))

    @app.post("/api/report", response_model=Report, responses=_ERRORS)
    def report(req: ReportRequest):
        return run_report(graph, req.thread_id, req.ticker)

    @app.post("/api/chat", response_model=ChatAnswer, responses=_ERRORS)
    def chat(req: ChatRequest):
        return run_chat(graph, req.thread_id, req.message)

    return app


app = create_app()
