"""PHASE 0 STUB API — routes and schemas per spec §3.4, fixture data only.

WS4 replaces the internals (graph, checkpointer, guardrails) WITHOUT changing routes
or response models. WS5 builds the UI against this.
Run: cd backend && uv run uvicorn app.api:app --port 8000
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.contracts import (
    fixture_retriever,
    stub_answer_filings_question,
    stub_ingest_recent_filings,
    stub_market_snapshot,
    stub_resolve_company,
    stub_summarize_filings,
)
from app.models import (
    ChatAnswer,
    ChatRequest,
    CreateSessionResponse,
    ErrorResponse,
    Report,
    ReportRequest,
    TickerNotFoundError,
)


def _err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(error_code=code, message=message).model_dump())


def create_app() -> FastAPI:
    app = FastAPI(title="Financial Data Agent (stub)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    reports: dict[str, Report] = {}  # thread_id -> last report (stub only; WS4 uses checkpointer)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError):
        return _err(422, "invalid_input", "; ".join(e["msg"] for e in exc.errors()))

    @app.exception_handler(TickerNotFoundError)
    async def _not_found(_: Request, exc: TickerNotFoundError):
        return _err(404, "ticker_not_found", str(exc))

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/sessions", response_model=CreateSessionResponse)
    def create_session():
        return CreateSessionResponse(thread_id=str(uuid.uuid4()))

    @app.post("/api/report", response_model=Report)
    def report(req: ReportRequest):
        _cik, name = stub_resolve_company(req.ticker)
        filings = stub_ingest_recent_filings(req.ticker)
        r = Report(
            ticker=req.ticker,
            company_name=name,
            market=stub_market_snapshot(req.ticker),
            filings=stub_summarize_filings(req.ticker, name, filings, fixture_retriever, None),
            warnings=["Stub data: served from fixtures, not live sources."],
            generated_at=datetime.now(UTC),
        )
        reports[req.thread_id] = r
        return r

    @app.post("/api/chat", response_model=ChatAnswer, responses={409: {"model": ErrorResponse}})
    def chat(req: ChatRequest):
        r = reports.get(req.thread_id)
        if r is None:
            return _err(409, "no_report_yet", "Generate a report for a ticker first.")
        a = stub_answer_filings_question(r.ticker, req.message, [], fixture_retriever, None)
        return ChatAnswer(text=a.text, citations=a.citations, route="filings",
                          warnings=["Stub answer."])

    return app


app = create_app()
