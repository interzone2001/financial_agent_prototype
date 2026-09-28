# Financial Data Agent — prototype

LangGraph multi-agent orchestrator: Alpha Vantage quote + cited SEC filing summaries
for wealth advisors, with follow-up chat. **~120 min timebox.**

- **Spec (source of truth):** `docs/superpowers/specs/2026-09-28-financial-agent-design.md`
- **Your plan:** `docs/superpowers/plans/2026-09-28-ws<N>-*.md`

## If you are a workstream session (WS1–WS5)

1. Read spec §1–5 and your §6.N, then your plan. **They are approved — do not
   brainstorm; execute the plan with TDD.**
2. **Only edit files your spec section/plan says you own**, plus tests under
   `backend/tests/ws<N>_*/` (or `frontend/` for WS5).
3. **Read-only contracts:** `backend/app/{models,contracts,llm,tracing,config}.py`,
   `backend/tests/{fakes,conftest}.py`, `backend/tests/fixtures/`,
   `backend/tests/test_contracts.py`, `backend/tests/test_stub_api.py`. If one looks
   wrong, STOP and tell the human (who relays to the integrator). Do not edit it.
4. **No live network in default tests.** Fixtures + `FakeLLM`. Live tests use
   `@pytest.mark.live` (skipped unless `-m live`). **Alpha Vantage quota is 25/day,
   shared by everyone.**
5. **Tracing is required:** decorate public functions and data-source calls with
   `from app.tracing import traceable` (conventions in `app/tracing.py`, spec §3.6).
   Never put API keys in traced inputs or `SourceRef.url`.
6. **Testing depth:** happy path + the edge cases in your plan. **No mutation testing.**
7. Commit often on your branch. **Don't merge to main** — the integrator does.
8. Done = `cd backend && uv run pytest -q && uv run ruff check .` (or
   `cd frontend && npm run typecheck && npm test && npm run build`) green; report to the human.

## Commands

```bash
cd backend && uv run pytest -q              # all offline tests
cd backend && uv run pytest -m live -q      # live API tests (spend quota!)
cd backend && uv run ruff check .
cd backend && uv run uvicorn app.api:app --port 8000
cd frontend && npm run dev                  # http://localhost:5173, proxies /api -> :8000
cd frontend && npm run typecheck && npm test && npm run build
```

## Environment

`.env` lives at the main repo root; each worktree has a symlink to it. Never copy
or commit keys. Vars: see `.env.example`. LangSmith tracing: `LANGSMITH_TRACING=true`.

## Gotchas found in Phase 0

- `anthropic` SDK is **1.9.0**. Use `client.messages.parse(output_format=Model)` and
  `client.beta.messages.tool_runner(...)` with `@beta_tool`. Only `app/llm.py` imports it.
- `wrap_anthropic` doesn't trace `parse`/tool runner → `AnthropicLLM` methods carry `@traceable`.
- 10-K/10-Q HTML is inline-XBRL: TOC repeats every "Item N." heading; suppress
  BeautifulSoup `XMLParsedAsHTMLWarning`.
- 8-K primary docs are mostly cover pages; the substance is in EX-99.1.
- Alpha Vantage fixtures are hand-built in the documented format (WS1 live test verifies).
