# Financial Data Agent — Design Spec

**Date:** 2026-09-28 · **Status:** Approved (brainstorm) · **Timebox:** ~120 min total

> **If you are a Claude Code session starting a workstream:** read §1–§5 (shared,
> ~5 min), then jump to **your** section in §6. That section is your approved spec —
> **do not brainstorm, go straight to TDD.** Everything in §3 (contracts) is
> **read-only** for you. If a contract seems wrong, STOP and tell the human; do not
> edit it.

---

## 1. Goal

A wealth-management advisor enters a ticker and within ~30s gets:

1. **Current market data** — quote + company overview (Alpha Vantage).
2. **A readable summary of recent SEC filings** — latest 10-K, latest 10-Q, 8-Ks from
   the last 90 days (max 5), organised as key developments / risk factors /
   financial highlights / material events.
3. **Follow-up chat** on that security ("what did they say about China risk?"),
   remembered per session.

**Non-negotiables:** every fact is attributed to its source (URL, provider,
retrieval time, and for filings: accession no., form, section); no investment advice;
Pydantic on every boundary; LangGraph checkpointing.

**Out of scope (YAGNI):** auth, multi-user, streaming responses, cross-session memory,
insider/13D filings, exhibits, charts, deployment.

## 2. Stack & decisions

| Concern | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** `StateGraph` (deterministic graph; LLM only routes chat) | Predictable, testable report path; real routing where it matters |
| LLM | **Anthropic Python SDK** called directly inside nodes (no LangChain chat-model wrappers) | Fewer layers; `messages.parse(output_format=Model)` gives validated Pydantic output |
| Models | `claude-sonnet-5` (agents), `claude-haiku-4-5` (router, advice check) | Quality where it's read, cheap where it's classification |
| Vector DB | **Chroma** (persistent, default local embedding fn — no extra key) | Filing chunks + metadata for cited retrieval |
| Relational | **SQLite** — AV response cache + LangGraph checkpoints | Already required by checkpointer |
| Checkpointing | `langgraph-checkpoint-sqlite` `SqliteSaver`, `thread_id` per UI session | Enables follow-up chat |
| Market data | **Alpha Vantage** `GLOBAL_QUOTE` + `OVERVIEW` | Free key, **25 req/day** → cache + fixtures mandatory |
| Filings | **SEC EDGAR** (no key; `User-Agent` header required, ≤10 req/s) | Authoritative source |
| API | **FastAPI** + Pydantic | |
| UI | **Vite + React + TypeScript**, plain CSS | One page, no component lib |
| Tracing | **LangSmith** (REQUIRED), toggled by `LANGSMITH_TRACING=true` | Every graph run, agent step, data-source call and LLM call is inspectable |
| Python tooling | `uv`, Python 3.12, `pytest`, `ruff` | |

Env vars (`.env`, never committed; `.env.example` is): `ANTHROPIC_API_KEY`,
`ALPHA_VANTAGE_API_KEY`, `SEC_USER_AGENT` (e.g. `"FinAgentPrototype you@example.com"`),
`DATA_DIR` (default `./data`), `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`,
`LANGSMITH_PROJECT=financial-agent-prototype`. `.env` lives at the main repo root;
each worktree gets a symlink to it (never copy keys).

## 3. Shared contracts (FROZEN after Phase 0 — owned by the integrator session)

All live in `backend/app/models.py`, `backend/app/contracts.py`, `backend/app/llm.py`.
Signatures below are normative; field docs are in the code.

### 3.1 Attribution & data models (`models.py`)

```python
Provider = Literal["alpha_vantage", "sec_edgar"]

class SourceRef(BaseModel):          # attached to EVERY fact
    provider: Provider
    url: str                         # human-clickable source URL
    retrieved_at: datetime           # UTC
    accession_no: str | None = None  # filings only
    form_type: str | None = None     # "10-K" | "10-Q" | "8-K"
    filed_date: date | None = None
    section: str | None = None       # e.g. "Item 1A. Risk Factors"
    chunk_id: str | None = None      # "{accession_no}:{section_slug}:{idx}"

class Quote(BaseModel):
    symbol: str; price: float; change: float; change_pct: float
    volume: int; latest_trading_day: date; source: SourceRef

class Overview(BaseModel):
    symbol: str; name: str; sector: str | None; industry: str | None
    market_cap: int | None; pe_ratio: float | None
    week52_high: float | None; week52_low: float | None
    description: str | None; source: SourceRef

class MarketSnapshot(BaseModel):
    quote: Quote; overview: Overview

class FilingMeta(BaseModel):
    cik: str                         # 10-digit zero-padded
    accession_no: str                # "0000320193-24-000123"
    form_type: Literal["10-K", "10-Q", "8-K"]
    filed_date: date; report_date: date | None
    primary_doc_url: str
    items: list[str] = []            # 8-K item codes, e.g. ["2.02", "9.01"]

class Chunk(BaseModel):
    chunk_id: str; ticker: str; text: str; source: SourceRef

class Claim(BaseModel):
    text: str
    citations: list[str] = Field(min_length=1)   # chunk_ids — NO uncited claims

class FilingsSummary(BaseModel):
    ticker: str; company_name: str
    filings: list[FilingMeta]
    key_developments: list[Claim]; risk_factors: list[Claim]
    financial_highlights: list[Claim]; material_events: list[Claim]
    citation_index: dict[str, SourceRef]         # chunk_id -> SourceRef (UI resolves links)

class Report(BaseModel):
    ticker: str; company_name: str
    market: MarketSnapshot | None                # None => market failed; see warnings
    filings: FilingsSummary | None               # None => filings failed; see warnings
    warnings: list[str] = []
    disclaimer: str = DISCLAIMER
    generated_at: datetime

Route = Literal["market", "filings", "both", "off_topic"]
class RouteDecision(BaseModel): route: Route; reason: str

class AgentAnswer(BaseModel):                    # what an agent returns for a chat turn
    text: str; citations: list[SourceRef] = []

class ChatAnswer(BaseModel):                     # what the API returns for a chat turn
    text: str; citations: list[SourceRef]; route: Route; warnings: list[str] = []

# API I/O
class CreateSessionResponse(BaseModel): thread_id: str
class ReportRequest(BaseModel): thread_id: str; ticker: str   # validator: ^[A-Z][A-Z.\-]{0,5}$ after .upper().strip()
class ChatRequest(BaseModel): thread_id: str; message: str = Field(max_length=1000)
class ErrorResponse(BaseModel): error_code: str; message: str

DISCLAIMER = ("For informational purposes only. Not investment advice. Data from SEC EDGAR "
              "and Alpha Vantage; verify against original sources before acting.")

# Errors (raise these; the API maps them)
class DataSourceError(Exception):  provider: Provider
class RateLimitedError(DataSourceError): ...          # -> warning / 503
class TickerNotFoundError(Exception): ...             # -> 404
```

### 3.2 Workstream interfaces (`contracts.py`)

Each workstream implements exactly these functions in its own module. `contracts.py`
holds the `Protocol`s plus **stub implementations backed by fixtures** so every other
workstream can run before yours exists.

```python
# WS1 Market  -> app/agents/market.py
def get_market_snapshot(ticker: str) -> MarketSnapshot                      # raises RateLimitedError, DataSourceError
def answer_market_question(question: str, snapshot: MarketSnapshot, llm: LLM) -> AgentAnswer

# WS2 SEC ingestion -> app/sources/edgar.py, app/ingest.py
def resolve_company(ticker: str) -> tuple[str, str]                         # (cik, company_name); raises TickerNotFoundError
def ingest_recent_filings(ticker: str) -> list[FilingMeta]                  # idempotent; skips stored accession_nos
def retrieve(ticker: str, query: str, k: int = 6,
             form_type: str | None = None) -> list[Chunk]

# WS3 Filings agent -> app/agents/filings.py
Retriever = Callable[[str, str, int, str | None], list[Chunk]]              # same shape as retrieve()
def summarize_filings(ticker: str, company_name: str, filings: list[FilingMeta],
                      retriever: Retriever, llm: LLM) -> FilingsSummary
def answer_filings_question(ticker: str, question: str, history: list[ChatTurn],
                            retriever: Retriever, llm: LLM) -> AgentAnswer

@dataclass
class AgentDeps:        # WS4 builds the graph from this; integration swaps stubs -> real
    get_market_snapshot; answer_market_question
    resolve_company; ingest_recent_filings; retrieve
    summarize_filings; answer_filings_question
    llm: LLM

def stub_deps() -> AgentDeps     # fixture-backed; used by WS4 tests & the stub API
```

`ChatTurn(BaseModel): role: Literal["user","assistant"]; content: str` (in `models.py`).

### 3.3 LLM wrapper (`llm.py`)

Agents never import `anthropic` directly — they take an `LLM`, so tests inject a fake.

```python
ModelRole = Literal["agent", "fast"]   # agent=claude-sonnet-5, fast=claude-haiku-4-5

class LLM(Protocol):
    def parse(self, role: ModelRole, system: str, messages: list[dict],
              schema: type[T]) -> T: ...                 # client.messages.parse(output_format=schema).parsed_output
    def text(self, role: ModelRole, system: str, messages: list[dict]) -> str: ...
    def run_tools(self, role: ModelRole, system: str, messages: list[dict],
                  tools: list) -> str: ...               # client.beta.messages.tool_runner(...) with @beta_tool fns; returns final text

class AnthropicLLM(LLM): ...          # real
class FakeLLM(LLM): ...               # tests/fakes.py: queue of canned returns per method; records calls
```

### 3.4 HTTP API (implemented by WS4, consumed by WS5)

| Method & path | Body | Returns | Errors |
|---|---|---|---|
| `POST /api/sessions` | — | `CreateSessionResponse` | |
| `POST /api/report` | `ReportRequest` | `Report` | 404 `ticker_not_found`, 422 `invalid_input` |
| `POST /api/chat` | `ChatRequest` | `ChatAnswer` | 409 `no_report_yet`, 422 `invalid_input` |
| `GET /api/health` | — | `{"status":"ok"}` | |

All errors return `ErrorResponse`. CORS allows `http://localhost:5173`. Backend on
`:8000`. Frontend proxies `/api` → `:8000` via Vite config.

### 3.5 Storage layout (`DATA_DIR`, gitignored)

- `data/cache.sqlite` — AV cache table `av_cache(key TEXT PK, body TEXT, fetched_at TEXT)`
- `data/checkpoints.sqlite` — LangGraph `SqliteSaver`
- `data/chroma/` — collection **`filings`**; metadata keys: `ticker, cik, accession_no,
  form_type, filed_date (ISO str), section, url, chunk_idx`. Doc id = `chunk_id`.
- `data/edgar/` — optional raw HTML cache keyed by accession.

### 3.6 Tracing (LangSmith — REQUIRED)

- Flag: `LANGSMITH_TRACING=true` (+ `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`). Off in
  tests (`tests/conftest.py`).
- LangGraph runs trace automatically. `AnthropicLLM` methods are `@traceable(run_type="llm")`.
  `wrap_anthropic` is NOT used (langsmith 0.14 + anthropic 1.x: crashes on
  `client.completions`). Note: `@traceable` adds a keyword-only `config=None` to a
  wrapped function's signature — contract/signature tests must compare
  `inspect.unwrap(fn)`.
- **Every workstream decorates its public functions and its data-source calls** with
  `from app.tracing import traceable`:
  - `run_type="chain"` agent/orchestration step (`market.answer_market_question`,
    `filings.summarize_filings`, graph nodes), `"tool"` data source / retrieval
    (`alpha_vantage.fetch`, `edgar.get_submissions`, `ingest.retrieve`), `"parser"`
    guardrail checks. Name format: `"<module>.<function>"`.
- Tracing must never leak secrets: no API keys in inputs/outputs (AV URLs are redacted).
- Acceptance: with the flag on, one report run shows a single trace tree
  `graph → market_node / filings_node → agent fns → tool calls / llm calls`.

## 4. Architecture

```
                        ┌──────────── report path ────────────────┐
 input ─▶ guard_in ─▶ orchestrator ─┬─▶ market_node  ──┐          │
          (validate)   (mode)       └─▶ filings_node ──┴─▶ compose ─▶ guard_out ─▶ END
                            │           (parallel)       (no LLM)   (advice, grounding)
                            └── chat: router (Haiku) ─▶ market | filings | both | off_topic ─▶ guard_out ─▶ END
```

- **Agents are plain functions** (§3.2) and know nothing about LangGraph. WS4 wraps
  them into nodes. That boundary is what makes parallel development possible.
- **Report path:** `market_node` calls `get_market_snapshot`; `filings_node` calls
  `resolve_company` → `ingest_recent_filings` → `summarize_filings`. A failure in one
  branch produces a partial `Report` (that field `None` + a `warnings` entry), never
  a 500. `TickerNotFoundError` aborts with 404.
- **Chat path:** requires a prior report in the thread (else 409). Router returns
  `RouteDecision`; `both` runs both agents and concatenates under headings.
- **State** (WS4, `graph.py`): `mode, ticker, company_name, snapshot, filings,
  report, history: list[ChatTurn], last_answer, warnings`. Persisted by `SqliteSaver`
  keyed on `thread_id`.

### 4.1 Guardrails (WS4, `guardrails.py`)

| Guardrail | Where | Mechanism |
|---|---|---|
| Input validation | `guard_in` + Pydantic | Ticker regex; `resolve_company` existence check before any LLM/AV spend; chat ≤1000 chars; injection heuristics (e.g. "ignore previous", "system prompt", role-play markers) → refuse |
| Scope filter | router | `off_topic` route → polite refusal, no agent call |
| No investment advice | `guard_out` | Regex (`\b(buy|sell|hold|strong buy|price target|should (invest|purchase)|overweight|underweight)\b` in advisory phrasing) on every Claim + chat text; chat also gets a Haiku yes/no check. Offending claim dropped / chat answer replaced with refusal + warning. Disclaimer always present |
| Grounding | `guard_out` | Every `Claim.citations` id ∈ `citation_index` (violations dropped + warning). Filings chat answers with no `chunk_id` citation are kept but flagged with a warning (Phase 2 decision: an unverified-flagged answer beats none) |

## 5. Rules for every session

1. **Your §6 section is the approved spec.** Skip brainstorming; use TDD.
2. **Only touch files you own** (listed in your section) plus new test files under
   `backend/tests/<your_ws>/`. Contracts (§3) are read-only.
3. **No live network in default tests.** Use fixtures in `backend/tests/fixtures/`
   and `FakeLLM`. Live tests are `@pytest.mark.live` and skipped unless `-m live`.
   Alpha Vantage quota is 25/day **shared across all sessions** — WS1 only, and sparingly.
4. **Testing depth:** happy path + the edge/error cases listed in your section.
   **No mutation testing.** Aim for fast (<10s) suites.
5. Work on your branch in your worktree; commit often with clear messages. Don't
   merge — the integrator (Phase 2) merges.
6. Run `uv run pytest backend/tests/<your_ws> -q` and `uv run ruff check` before
   declaring done. Report done-criteria status to the human.

### 5.1 Session map & timeline

| Phase | Session | Branch / worktree | Time |
|---|---|---|---|
| 0 | **Integrator** (this session): contracts, stubs, fixtures, scaffolds, worktrees | `main` | 0–20 min |
| 1 | **WS1 Market Data** | `ws/market` → `../fap-ws1-market` | 20–85 |
| 1 | **WS2 SEC Ingestion** | `ws/sec-ingest` → `../fap-ws2-ingest` | 20–85 |
| 1 | **WS3 Filings Agent** | `ws/filings-agent` → `../fap-ws3-filings` | 20–85 |
| 1 | **WS4 Orchestrator + Guardrails + API** | `ws/orchestrator` → `../fap-ws4-orch` | 20–85 |
| 1 | **WS5 Frontend** | `ws/frontend` → `../fap-ws5-fe` | 20–85 |
| 2 | **Integrator**: merge WS1→WS2→WS3→WS4→WS5, swap stubs for real deps, full suite, live smoke | `main` | 85–110 |
| — | Buffer / demo | | 110–120 |

**Starting a session:** `cd ../fap-wsN-xxx && claude`, then:
> "Read `docs/superpowers/specs/2026-09-28-financial-agent-design.md` §1–5 and §6.N,
> and `docs/superpowers/plans/` for your workstream plan. Execute it with TDD."

---

## 6. Workstreams

### 6.1 WS1 — Market Data Agent

**Owns:** `backend/app/sources/alpha_vantage.py`, `backend/app/cache.py`,
`backend/app/agents/market.py`, `backend/tests/ws1_market/`.
**Implements:** `get_market_snapshot`, `answer_market_question` (§3.2).
**Fixtures provided:** `fixtures/av_global_quote_AAPL.json`, `fixtures/av_overview_AAPL.json`.

Behaviour:
- `AlphaVantageClient(api_key, http=httpx.Client, cache=SqliteCache)`; `GET
  https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=X&apikey=…` and
  `function=OVERVIEW`.
- Cache TTL: quote 60s, overview 24h. Cache key excludes the API key.
- **Rate-limit detection:** AV returns HTTP 200 with a body containing `"Note"` or
  `"Information"` → raise `RateLimitedError`. Empty `"Global Quote": {}` → `TickerNotFoundError`.
- Parse AV strings (`"05. price": "227.5200"`, `"10. change percent": "1.23%"`,
  `"None"`/`"-"` → `None`) into `Quote`/`Overview`. Each gets a `SourceRef(provider=
  "alpha_vantage", url=<query URL with apikey REDACTED>, retrieved_at=<fetch time,
  from cache if cached>)`.
- `answer_market_question`: `llm.text("agent", …)` with the snapshot JSON as the only
  data; system prompt: answer only from supplied data, cite nothing else, no advice.
  Returns `AgentAnswer(citations=[quote.source, overview.source])`.

**Tests (must pass):** parse quote fixture; parse overview fixture incl. `"None"`
fields; rate-limit `Note` → `RateLimitedError`; empty quote → `TickerNotFoundError`;
cache hit avoids second HTTP call (mock transport counts calls); cache expiry refetches;
API key never appears in `SourceRef.url`; `answer_market_question` passes snapshot to
LLM and returns both sources. One `@pytest.mark.live` test for AAPL.

**Done when:** tests green, ruff clean, `get_market_snapshot` importable with the
§3.2 signature.

### 6.2 WS2 — SEC Ingestion (EDGAR → Chroma)

**Owns:** `backend/app/sources/edgar.py`, `backend/app/ingest.py`,
`backend/app/vectorstore.py`, `backend/tests/ws2_ingest/`.
**Implements:** `resolve_company`, `ingest_recent_filings`, `retrieve` (§3.2).
**Fixtures provided:** `fixtures/sec_company_tickers.json` (trimmed),
`fixtures/sec_submissions_AAPL.json`, `fixtures/sec_10k_AAPL_excerpt.html`,
`fixtures/sec_8k_AAPL.html`.

Behaviour:
- `EdgarClient(user_agent, http)`: headers `User-Agent`, `Accept-Encoding: gzip`;
  simple throttle ≤10 req/s.
  - Ticker map: `https://www.sec.gov/files/company_tickers.json` (cache to disk for
    the process lifetime). CIK zero-padded to 10.
  - Submissions: `https://data.sec.gov/submissions/CIK##########.json` →
    `filings.recent` parallel arrays (`form`, `accessionNumber`, `filingDate`,
    `reportDate`, `primaryDocument`, `items`).
  - Doc URL: `https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_nodash}/{primaryDocument}`.
- Selection: latest `10-K`, latest `10-Q`, `8-K`s filed in last 90 days (max 5).
  Ignore amendments (`10-K/A`).
- **8-K content lives in exhibits.** Phase 0 found the 8-K primary doc is mostly a
  cover page ("press release attached as Exhibit 99.1"). Stretch task: also fetch
  EX-99.x from the filing index (`…/{accession_nodash}/index.json`) and ingest with
  section `"8-K Exhibit 99.1"`. Core requirement is the primary doc only.
- 10-K/10-Q primary docs are inline-XBRL XHTML: BeautifulSoup emits
  `XMLParsedAsHTMLWarning` — suppress it; still parse with `lxml`.
- Text extraction: BeautifulSoup (`lxml`), drop `script/style` and inline-XBRL hidden
  `ix:header`; collapse whitespace.
- Sectioning (10-K/10-Q): regex on `Item\s+(1A|1|2|3|7A|7)\.?` headings; the **table
  of contents repeats headings** — take the occurrence that starts the longest span.
  Keep: 10-K → Items 1, 1A, 7, 7A; 10-Q → Part I Item 2, Part II Item 1A. Fallback if
  no headings found: whole doc as section `"Full text"`. 8-K: whole doc, section =
  `"8-K Items " + ",".join(items)`.
- Chunking: ~1500 chars, 200 overlap, split on paragraph boundaries when possible.
  Cap 150 chunks per filing (log if capped).
- Chroma: `PersistentClient(DATA_DIR/chroma)`, collection `filings`, default embedding
  function, metadata per §3.5. Idempotency: skip filing if any doc with its
  `accession_no` exists.
- `retrieve`: `where={"ticker": T}` (+ `form_type` if given) → `list[Chunk]` with a
  full `SourceRef(provider="sec_edgar", url=primary_doc_url, …, chunk_id)`.

**Tests (must pass):** ticker→CIK (and unknown → `TickerNotFoundError`); selection from
submissions fixture (right forms, 90-day window with injected `today`, max 5, no
amendments); URL construction; section split on 10-K fixture ignores TOC duplicate;
chunk size/overlap bounds; ingest into a tmp-dir Chroma then `retrieve` returns chunks
with complete `SourceRef`s filtered by ticker/form; re-ingest is a no-op. One
`@pytest.mark.live` AAPL end-to-end.

**Done when:** tests green, ruff clean. Note: first Chroma run downloads a ~80MB ONNX
embedding model — do it once early.

### 6.3 WS3 — Filings Agent

**Owns:** `backend/app/agents/filings.py`, `backend/app/prompts/filings.py`,
`backend/tests/ws3_filings/`.
**Implements:** `summarize_filings`, `answer_filings_question` (§3.2).
**Fixtures provided:** `fixtures/chunks_AAPL.json` (~12 real chunks across 10-K/10-Q/8-K
with full `SourceRef`s), `fixture_retriever()` in `contracts.py` stubs.

Behaviour:
- `summarize_filings`: for each section run 1–2 targeted `retriever` queries
  (e.g. risk_factors: "principal risk factors", form 10-K/10-Q; material_events:
  8-K form; financial_highlights: "net sales revenue gross margin results of
  operations"; key_developments: "significant developments strategy outlook"). Dedupe
  chunks; build prompt listing chunks as `[chunk_id] (form, section, date)\ntext`.
  One `llm.parse("agent", …, schema=_SummaryDraft)` where `_SummaryDraft` has the four
  `list[Claim]` fields. Build `FilingsSummary` with `citation_index` = the retrieved
  chunks only.
- Prompt rules: plain advisor-friendly language; 3–5 claims per section; every claim
  cites ≥1 chunk_id **from the list provided**; no recommendations/opinions on the
  stock; say "not disclosed in retrieved filings" rather than guess.
- Local pre-guard: drop any claim citing an unknown chunk_id (WS4 re-checks).
- `answer_filings_question`: `llm.run_tools("agent", …, tools=[search_filings])` where
  `search_filings(query: str, form_type: str | None)` wraps `retriever` and records
  every chunk returned. System prompt requires `[chunk_id]` inline citations. Parse
  cited ids from the final text; `AgentAnswer.citations` = SourceRefs of cited ids
  that were actually retrieved. Include last 6 `history` turns.

**Tests (must pass):** summarize with `FakeLLM` returns FilingsSummary whose
`citation_index` covers all cited ids; claim with fabricated chunk_id is dropped;
retriever called with the right form filters; prompt contains chunk ids and no
more than N chunks; chat answer extracts citations and ignores ids never retrieved;
history truncated to 6. One `@pytest.mark.live` test using the fixture retriever +
real Claude.

**Done when:** tests green, ruff clean.

### 6.4 WS4 — Orchestrator, Guardrails, API

**Owns:** `backend/app/graph.py`, `backend/app/guardrails.py`, `backend/app/api.py`,
`backend/app/prompts/router.py`, `backend/tests/ws4_orch/`.
**Builds against:** `stub_deps()` from `contracts.py` and `FakeLLM`.

Behaviour:
- `build_graph(deps: AgentDeps, checkpointer) -> CompiledGraph` per §4. Report fan-out
  uses parallel edges from `orchestrator` to `market_node` and `filings_node`, joined
  at `compose`. Nodes catch `DataSourceError` → warning; let `TickerNotFoundError`
  propagate.
- `compose` (no LLM) builds `Report`; `guard_out` applies §4.1 and finalises warnings.
- Router: `deps.llm.parse("fast", …, schema=RouteDecision)` with the ticker/company in
  context.
- `api.py`: FastAPI app factory `create_app(deps: AgentDeps | None = None)`; default =
  real deps (after integration) — for now default to `stub_deps()`. `SqliteSaver` at
  `DATA_DIR/checkpoints.sqlite`. Endpoints and error mapping per §3.4. CORS.
- **Phase 0 already ships a stub `api.py`** returning fixtures; WS4 replaces its internals
  without changing routes or schemas.

**Tests (must pass, via `TestClient` + stub deps + FakeLLM):** report happy path
(both sections, disclaimer, sources present); market `RateLimitedError` → partial
report + warning (200); unknown ticker → 404 `ticker_not_found`; bad ticker format → 422;
chat before report → 409; chat routed to market / filings / both; off_topic refused
without agent call; injection string refused; advice phrase in a claim → claim dropped
+ warning; uncited/unknown-citation claim dropped; **checkpoint persistence**: new
graph instance with the same SqliteSaver file + thread_id sees prior report.

**Done when:** tests green, ruff clean, `uv run uvicorn app.api:app` serves stub data.

### 6.5 WS5 — Frontend

**Owns:** everything under `frontend/`.
**Builds against:** Phase 0 stub API (`uv run uvicorn app.api:app --port 8000` in
`backend/`), or `frontend/src/fixtures/*.json` (copies of stub responses) via a
`VITE_USE_FIXTURES=1` flag.

Behaviour:
- `src/api.ts`: typed client; TS types in `src/types.ts` mirror §3.1 exactly (hand-written).
- Flow: on load `POST /api/sessions` (store `thread_id` in state + `sessionStorage`) →
  ticker form → `POST /api/report` with loading state ("Fetching quote & filings…") →
  render:
  - **Quote card:** price, change (green/red), volume, market cap, P/E, 52w range,
    sector; footer "Source: Alpha Vantage · retrieved {time}" linking to source.
  - **Filings report:** list of filings (form, filed date, link to sec.gov); four
    sections of claims, each claim followed by superscript citation numbers `[1][2]`;
    hovering/clicking shows form · section · filed date and links to `url`.
    Numbered **Sources** list at the bottom.
  - **Warnings** banner if `warnings` non-empty; **disclaimer** always visible.
  - **Chat panel:** message list, input (max 1000), shows `route` badge and citations
    per answer.
- Error states: 404 → "Ticker not found", 409 → "Generate a report first", network →
  retry button.

**Tests (must pass):** `npm run typecheck`, `npm run build`; Vitest: citation renderer
maps chunk ids → numbered sources correctly and links to URL; api client handles
`ErrorResponse`.

**Done when:** the above pass and the page works end-to-end against the stub API.

#### 6.5.1 UI design addendum (approved 2026-09-28)

This section refines §6.5. It does **not** change §3 contracts or the API; everything below is client-side.

- **Layout:** a sticky header (app name · ticker form · disclaimer, always visible in both themes). Below it, two panes: the report on the left (about 65%, scrolls independently) and the chat on the right (about 35%, full height, stays put). Below 900px the panes stack (report, then chat) and the page scrolls normally. No horizontal scroll at 375px.
- **Theme:** "terminal-lite". Dark by default; light palette under `prefers-color-scheme: light`. All colours are CSS variables. Tabular numbers throughout; mono font for tickers, prices and citation marks. Price change shows ▲/▼ as well as green/red.
- **Citations: one registry per session.** Report claims are numbered first, by first appearance in section order. Chat answers reuse existing numbers and append new sources to the same numbered **Sources** list, which is rendered once at the bottom of the report pane. Registry key = `SourceRef.chunk_id`, falling back to `url` for Alpha Vantage sources. The registry resets when a new report is generated.
- **Chat inline markers:** answer text contains `[<chunk_id>]` (per §6.3). The UI parses these into `[n]` citation marks. Unknown ids stay as literal text. Citations not mentioned inline show as `[n]` chips under the answer. Each answer carries a `route` badge.
- **`[n]` interaction:** hover or keyboard focus opens a popover (form · section · filed date · "Open on sec.gov ↗"; for Alpha Vantage: provider · retrieved time); Escape closes it. Click scrolls to the matching Sources entry and flashes it. An id missing from `citation_index` still gets a number, and its entry reads "Source unavailable".
- **Loading:** "Fetching quote & filings…" plus a *cosmetic* timed stage line (resolving ticker → quote → reading filings → summarising) with elapsed seconds, and skeleton cards. The API doesn't stream, so the stages are an estimate, not real progress.
- **Chat gating:** the input is disabled until a report exists. A 409 (for example, after a server restart) still shows "Generate a report first".
- **Cut order** (extends §9): popover (fall back to a `title` attribute) → timed stages (fall back to static copy) → light theme.

---

## 7. Phase 0 deliverables (integrator, before sessions start)

- Repo scaffold: `backend/` (uv project, deps: langgraph, langgraph-checkpoint-sqlite,
  anthropic, chromadb, fastapi, uvicorn, httpx, pydantic, beautifulsoup4, lxml,
  python-dotenv; dev: pytest, ruff, respx), `frontend/` (Vite React TS + vitest).
- `models.py`, `contracts.py` (Protocols + `stub_deps()` + `fixture_retriever()`),
  `llm.py` (`AnthropicLLM`), `tests/fakes.py` (`FakeLLM`), `tests/conftest.py`
  (`live` marker, `tmp_data_dir` fixture).
- Recorded fixtures (§6.1–6.3 lists) — AV recorded **once** (2 calls).
- Stub `api.py` serving fixture data on the §3.4 routes.
- `CLAUDE.md` (the §5 rules), `.env.example`, `.gitignore` (`.env`, `data/`, `node_modules/`).
- Per-workstream plans in `docs/superpowers/plans/` and git worktrees/branches per §5.1.

## 8. Phase 2 integration checklist

1. Merge branches in order WS1 → WS2 → WS3 → WS4 → WS5 (file ownership ⇒ no conflicts
   expected outside `pyproject.toml`/lockfile).
2. `real_deps()` in `contracts.py` wiring real functions + `AnthropicLLM`; `create_app()`
   defaults to it.
3. Full `uv run pytest -q` + frontend build.
4. Live smoke: AAPL report → follow-up "What are the main risk factors?" → follow-up
   "Should I buy it?" (expect refusal) → restart server, same thread_id, follow-up
   still has context.

## 9. Risks

| Risk | Mitigation |
|---|---|
| AV 25/day quota | Cache, fixtures, only WS1 live test, integration smoke uses ≤4 calls |
| 10-K sectioning brittle | Fallback to full-text chunks; retrieval still works |
| Chroma embedding model download | Warm once in Phase 0 |
| Contract drift between sessions | Frozen contracts + stop-and-ask rule |
| Time overrun | Cut order: Haiku advice check → both-route → 8-K section labels → UI hover cards |

## 10. Known limitations (Phase 2)

- SEC unreachable ⇒ whole report 503 (SEC is the ticker-existence authority, checked first).
- SEC throttles with HTTP 403, surfaced as a generic `DataSourceError`.
- EDGAR client throttle is per-process and not thread-safe across concurrent reports.
- Superseded 10-Qs / old 8-Ks remain retrievable in a long-lived `DATA_DIR`.
- A filing yielding 0 chunks is still listed in `FilingsSummary.filings`.
- Chat citations only cover chunks retrieved in that turn.
