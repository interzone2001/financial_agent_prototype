# Security Brief — Financial Data Agent (prototype)

A LangGraph multi-agent orchestrator that gets a wealth advisor up to speed on a stock:
a live quote (Alpha Vantage) plus a **cited** summary of recent SEC filings (EDGAR → Chroma →
Claude), with follow-up chat remembered per session. Every fact carries its source.

- Design spec: [`docs/superpowers/specs/2026-09-28-financial-agent-design.md`](docs/superpowers/specs/2026-09-28-financial-agent-design.md)
- Workstream plans: [`docs/superpowers/plans/`](docs/superpowers/plans/)
- E2E results: [`docs/e2e-2026-09-28.md`](docs/e2e-2026-09-28.md)
- Business one-pager: [`docs/business/advisor-stock-briefing.pdf`](docs/business/advisor-stock-briefing.pdf)

## Architecture

```
 input ─▶ guard_in ─▶ orchestrator ─┬─▶ market_node  (Alpha Vantage) ──┐
                                    └─▶ filings_node (EDGAR→Chroma→Claude) ┴─▶ compose ─▶ guard_out
          chat: router (Haiku) ─▶ market | filings | both | off_topic ─▶ guard_out
```

LangGraph + SQLite checkpointing (per-session memory), Chroma (filing chunks), Pydantic on every
boundary, LangSmith tracing, guardrails for input validation / injection, scope, no investment
advice, and citation grounding. Claude Sonnet 5 writes; Claude Haiku 4.5 routes and checks.

## Run it

```bash
cp .env.example .env          # fill in ANTHROPIC, ALPHA_VANTAGE, LANGSMITH keys + SEC_USER_AGENT
cd backend && uv sync && uv run uvicorn app.api:app --port 8000
cd frontend && npm ci && npm run dev    # http://localhost:5173 (proxies /api → :8000)
```

No keys? `APP_MODE=stub uv run uvicorn app.api:app --port 8000` serves fixture data offline.
Backend on another port: `VITE_API_TARGET=http://localhost:8001 npm run dev`.

## Test

```bash
cd backend && uv run pytest -q && uv run ruff check .    # offline (fixtures + FakeLLM)
cd backend && uv run pytest -m live -q                   # live APIs — spends AV quota (25/day)
cd frontend && npm run typecheck && npm test && npm run build
```
