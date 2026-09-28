# WS5 Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A one-page advisor UI: enter a ticker → quote card + cited SEC filings summary → follow-up chat, against the §3.4 API.

**Architecture:** Hand-written TS types mirror `backend/app/models.py`. `api.ts` is the only code touching `fetch`/`sessionStorage` and turns every failure into `ApiError`. `citations.ts` + `format.ts` are pure and unit-tested; components are thin function components; `App.tsx` owns session/report/error state.

**Tech Stack:** Vite + React + TypeScript, plain CSS (`src/index.css`), Vitest + jsdom + @testing-library/react.

**Spec:** `docs/superpowers/specs/2026-09-28-financial-agent-design.md` (§1–5, §3.1, §3.4, §6.5, **§6.5.1 UI design addendum**)

**Approved UI design (§6.5.1, in brief):**
- **Layout:** two panes. The report is on the left (about 65%) and scrolls. The chat is on the right (about 35%) and stays put. Below 900px the panes stack. A sticky header holds the ticker form and the always-visible disclaimer.
- **Theme:** "terminal-lite": dark by default, light under `prefers-color-scheme`, mono font for tickers and citations, and ▲/▼ next to green/red so the change never relies on colour alone.
- **Citations:** one numbering for the whole session, shared by the report and chat. Chat's inline `[chunk_id]` markers are rendered as the same `[n]`. Hovering or focusing `[n]` opens a popover. Clicking scrolls to the one Sources list and flashes the entry.
- **Loading:** timed stage text plus skeleton cards. This is cosmetic, since the API doesn't stream.
- **Cut order if time runs short:** popover (fall back to a `title` attribute) → timed stages (fall back to static copy) → light theme.

## Global Constraints

- Touch only `frontend/**`. Contracts (`backend/app/models.py`) are read-only; if one looks wrong, STOP and tell the human.
- TS types use the **exact snake_case field names** of `models.py`; `date` → `"YYYY-MM-DD"` string, `datetime` → ISO string. Never pass a date-only string to `new Date()` (shifts a day in US time zones) — show it verbatim.
- No component libraries, no new runtime deps. Dev deps only: `vitest`, `jsdom`, `@testing-library/react`, `@testing-library/dom`.
- All styling in `src/index.css`; numbers `font-variant-numeric: tabular-nums`; no horizontal scroll at 375px.
- Copy (verbatim): loading `Fetching quote & filings…` (plus a timed stage line under it); 404 `Ticker not found`; 409 `Generate a report first`; 422 shows server `message`; network/5xx → `Retry` button.
- Chat input max 1000 chars. Disclaimer always on screen (before and after a report).
- `sessionStorage` key `fap.thread_id`; every access in try/catch.
- Vite react-ts template sets `verbatimModuleSyntax` + `erasableSyntaxOnly`: use `import type`, no TS parameter properties, no `enum`.
- Tests: no network (`vi.stubGlobal("fetch", …)`); no vitest globals — call `afterEach(cleanup)` explicitly; no jest-dom matchers.

## Review Focus

1. Backend down: Vite proxy returns a non-JSON 500 → `ApiError` + Retry, not a JSON-parse crash (Task 2 test; Task 5 retry test).
2. FastAPI's default 422 body `{"detail":[{"msg":…}]}` (raised before WS4's handler) → show `msg`, never "undefined" (Task 2 test).
3. A claim cites a chunk_id missing from `citation_index` → still numbered, rendered unlinked, "Source unavailable" (Task 3 & 4 tests).
4. Partial report (`market: null` once Alpha Vantage's 25/day quota is spent) → placeholder + warnings banner, filings still render (Task 5 test).
5. `sessionStorage` throws (blocked storage / private mode) → app still gets a session (Task 2 test).
6. Chat text with inline `[chunk_id]` markers → rendered as `[n]` using the report's numbers. Unknown markers stay as literal text, and raw accession ids are never shown as citations (Task 3 + Task 5 tests).

---

### Task 1: Verify scaffold, add types and fixtures

**Files:** Check `frontend/package.json`, `frontend/vite.config.ts`, `frontend/src/fixtures/*.json` (already on main from Phase 0). Create `frontend/src/types.ts`. Delete `frontend/src/App.css`, `frontend/src/assets/`.

**Interfaces:** Produces every type below (used by all later tasks) and the fixtures `api.ts` loads when `VITE_USE_FIXTURES=1`.

- [ ] **Step 1: Verify scaffold**

Run: `cd frontend && npm pkg get scripts && npm ls vitest jsdom @testing-library/react @testing-library/dom && cat vite.config.ts`
Expected: scripts `dev`, `build`, `typecheck`, `test`; all four packages present. Fix gaps:

```bash
npm i -D vitest jsdom @testing-library/react @testing-library/dom
npm pkg set scripts.typecheck="tsc -b" scripts.test="vitest run"
```

Phase 0 already ships all of this (plus `port: 5173`, `globals: true` and `setupFiles: ['./src/setupTests.ts']` with jest-dom). **Keep the existing config; don't rewrite it.** It must at least contain:

```ts
/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
  test: { environment: "jsdom" },
});
```

Confirm `vite/client` types are wired (`src/vite-env.d.ts` or `"types": ["vite/client"]` in `tsconfig.app.json`).

- [ ] **Step 2: Write `src/types.ts`**

```ts
// Mirrors backend/app/models.py (spec §3.1). date => "YYYY-MM-DD", datetime => ISO string.
export type Provider = "alpha_vantage" | "sec_edgar";
export type FormType = "10-K" | "10-Q" | "8-K";
export type Route = "market" | "filings" | "both" | "off_topic";
export type ISODate = string;
export type ISODateTime = string;
export const DISCLAIMER = "For informational purposes only. Not investment advice. Data from SEC EDGAR " +
  "and Alpha Vantage; verify against original sources before acting.";

export interface SourceRef {
  provider: Provider; url: string; retrieved_at: ISODateTime;
  accession_no?: string | null; form_type?: string | null; filed_date?: ISODate | null;
  section?: string | null; chunk_id?: string | null;
}
export interface Quote {
  symbol: string; price: number; change: number; change_pct: number; // 1.23 means +1.23%
  volume: number; latest_trading_day: ISODate; source: SourceRef;
}
export interface Overview {
  symbol: string; name: string; sector: string | null; industry: string | null;
  market_cap: number | null; pe_ratio: number | null; week52_high: number | null;
  week52_low: number | null; description: string | null; source: SourceRef;
}
export interface MarketSnapshot { quote: Quote; overview: Overview }
export interface FilingMeta {
  cik: string; accession_no: string; form_type: FormType; filed_date: ISODate;
  report_date: ISODate | null; primary_doc_url: string; items: string[];
}
export interface Chunk { chunk_id: string; ticker: string; text: string; source: SourceRef }
export interface Claim { text: string; citations: string[] }
export interface FilingsSummary {
  ticker: string; company_name: string; filings: FilingMeta[];
  key_developments: Claim[]; risk_factors: Claim[]; financial_highlights: Claim[]; material_events: Claim[];
  citation_index: Record<string, SourceRef>;
}
export interface Report {
  ticker: string; company_name: string; market: MarketSnapshot | null; filings: FilingsSummary | null;
  warnings: string[]; disclaimer: string; generated_at: ISODateTime;
}
export interface ChatTurn { role: "user" | "assistant"; content: string }
export interface RouteDecision { route: Route; reason: string }
export interface AgentAnswer { text: string; citations: SourceRef[] }
export interface ChatAnswer { text: string; citations: SourceRef[]; route: Route; warnings: string[]; disclaimer: string }
export interface CreateSessionResponse { thread_id: string }
export interface ReportRequest { thread_id: string; ticker: string }
export interface ChatRequest { thread_id: string; message: string }
export interface ErrorResponse { error_code: string; message: string }
```

- [ ] **Step 3: Check the fixtures (already exported in Phase 0)**

The Phase 0 stub responses are in `src/fixtures/{session,report,chat}.json`. Just run the `node -e` key check at the end of the block below. Re-record with the curl commands only if the check fails or you want fresh data. Start the backend in the background: `cd backend && uv run uvicorn app.api:app --port 8000`. Then:

```bash
cd frontend && mkdir -p src/fixtures
curl -s -X POST localhost:8000/api/sessions > src/fixtures/session.json
TID=$(node -p "require('./src/fixtures/session.json').thread_id")
curl -s -X POST localhost:8000/api/report -H 'Content-Type: application/json' \
  -d "{\"thread_id\":\"$TID\",\"ticker\":\"AAPL\"}" > src/fixtures/report.json
curl -s -X POST localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d "{\"thread_id\":\"$TID\",\"message\":\"What are the main risk factors?\"}" > src/fixtures/chat.json
node -e "for (const f of ['session','report','chat']) console.log(f, Object.keys(require('./src/fixtures/'+f+'.json')))"
```

Expected: `report` keys include `ticker, company_name, market, filings, warnings, disclaimer, generated_at`; `chat` includes `text, citations, route`. If the stub API won't run, STOP and tell the human — don't hand-write fixtures.

- [ ] **Step 4: Remove template leftovers, typecheck**

`rm -rf src/App.css src/assets`; replace `src/App.tsx` with `export default function App() { return null; }` (Task 5 writes the real one). Run `npm run typecheck` → exit 0.

- [ ] **Step 5: Commit** — `git add frontend && git commit -m "feat(fe): scaffold check, TS model mirror, recorded stub fixtures"`

---

### Task 2: API client (TDD)

**Files:** Create `frontend/src/api.ts`, `frontend/src/test/sample.ts`. Test: `frontend/src/api.test.ts`.

**Interfaces:**
- Consumes: Task 1 types and fixtures.
- Produces: `class ApiError extends Error { status: number; error_code: string }` (status 0 = network); `api.createSession(): Promise<CreateSessionResponse>`, `api.getReport(threadId, ticker): Promise<Report>`, `api.chat(threadId, message): Promise<ChatAnswer>`; `ensureSession(): Promise<string>`; `errorMessage(e: unknown): string`; `isRetryable(e: unknown): boolean`; `THREAD_KEY`. `test/sample.ts` exports `T`, `src10k`, `src8k`, `avSrc(fn)`, `sampleReport`, `json(status, body)`.

- [ ] **Step 1: Shared test data `src/test/sample.ts`**

```ts
import type { Report, SourceRef } from "../types";

export const T = "2026-09-28T18:00:00Z";
export const src10k: SourceRef = { provider: "sec_edgar", retrieved_at: T,
  url: "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
  accession_no: "0000320193-25-000079", form_type: "10-K", filed_date: "2025-10-31",
  section: "Item 1A. Risk Factors", chunk_id: "K:item-1a:0" };
export const src8k: SourceRef = { ...src10k,
  url: "https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/aapl-20260730.htm",
  accession_no: "0000320193-26-000018", form_type: "8-K", filed_date: "2026-07-30",
  section: "8-K Items 2.02,9.01", chunk_id: "E:8k-items:0" };
export const avSrc = (fn: string): SourceRef => ({ provider: "alpha_vantage", retrieved_at: T,
  url: `https://www.alphavantage.co/query?function=${fn}&symbol=AAPL&apikey=REDACTED` });
const filing = (s: SourceRef, form_type: "10-K" | "8-K", items: string[]) => ({ cik: "0000320193",
  accession_no: s.accession_no!, form_type, filed_date: s.filed_date!, report_date: null, primary_doc_url: s.url, items });
export const sampleReport: Report = {
  ticker: "AAPL", company_name: "Apple Inc.", generated_at: T, warnings: [],
  disclaimer: "For informational purposes only. Not investment advice.",
  market: {
    quote: { symbol: "AAPL", price: 227.52, change: -1.25, change_pct: -0.55, volume: 45123456,
      latest_trading_day: "2026-09-25", source: avSrc("GLOBAL_QUOTE") },
    overview: { symbol: "AAPL", name: "Apple Inc.", sector: "TECHNOLOGY", industry: null, market_cap: 3410000000000,
      pe_ratio: null, week52_high: 260.1, week52_low: 169.21, description: null, source: avSrc("OVERVIEW") },
  },
  filings: {
    ticker: "AAPL", company_name: "Apple Inc.",
    filings: [filing(src10k, "10-K", []), filing(src8k, "8-K", ["2.02", "9.01"])],
    key_developments: [{ text: "Services revenue grew.", citations: ["E:8k-items:0"] }],
    risk_factors: [{ text: "Supply chain concentrated in Asia.", citations: ["K:item-1a:0", "E:8k-items:0"] }],
    financial_highlights: [],
    material_events: [{ text: "Reported quarterly results.", citations: ["E:8k-items:0", "missing:id"] }],
    citation_index: { "K:item-1a:0": src10k, "E:8k-items:0": src8k },
  },
};
export const json = (status: number, body: unknown) => new Response(
  typeof body === "string" ? body : JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
```

- [ ] **Step 2: Write the failing tests `src/api.test.ts`**

```ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, THREAD_KEY, api, ensureSession, errorMessage, isRetryable } from "./api";
import { json, sampleReport } from "./test/sample";

const stubFetch = (impl: () => Promise<Response>) => { const fn = vi.fn(impl); vi.stubGlobal("fetch", fn); return fn; };
const caught = (p: Promise<unknown>) => p.then(() => { throw new Error("expected rejection"); }, (e) => e as ApiError);
const fields = (e: ApiError) => [e.status, e.error_code, e.message];

beforeEach(() => sessionStorage.clear());
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.restoreAllMocks(); });

describe("api client", () => {
  it("POSTs the report request as JSON and returns the body", async () => {
    const fetch = stubFetch(async () => json(200, sampleReport));
    expect((await api.getReport("t1", "AAPL")).company_name).toBe("Apple Inc.");
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect([url, init.method]).toEqual(["/api/report", "POST"]);
    expect(JSON.parse(init.body as string)).toEqual({ thread_id: "t1", ticker: "AAPL" });
  });
  it("parses ErrorResponse into ApiError", async () => {
    stubFetch(async () => json(404, { error_code: "ticker_not_found", message: "Ticker not found: ZZZZ" }));
    const e = await caught(api.getReport("t1", "ZZZZ"));
    expect(e).toBeInstanceOf(ApiError);
    expect(fields(e)).toEqual([404, "ticker_not_found", "Ticker not found: ZZZZ"]);
  });
  it("handles FastAPI's default 422 detail list", async () => {
    stubFetch(async () => json(422, { detail: [{ msg: "String should have at most 1000 characters" }] }));
    expect(fields(await caught(api.chat("t1", "x"))))
      .toEqual([422, "invalid_input", "String should have at most 1000 characters"]);
  });
  it("handles a non-JSON error body (proxy error page) as retryable", async () => {
    stubFetch(async () => new Response("Internal Server Error", { status: 500 }));
    const e = await caught(api.createSession());
    expect([e.status, e.error_code, isRetryable(e)]).toEqual([500, "http_error", true]);
  });
  it("maps a fetch rejection to a retryable network error", async () => {
    stubFetch(async () => { throw new TypeError("Failed to fetch"); });
    const e = await caught(api.createSession());
    expect([e.status, e.error_code, isRetryable(e)]).toEqual([0, "network_error", true]);
  });
  it("maps errors to advisor-facing copy", () => {
    expect(errorMessage(new ApiError(404, "ticker_not_found", "x"))).toMatch(/^Ticker not found/);
    expect(errorMessage(new ApiError(409, "no_report_yet", "x"))).toBe("Generate a report first");
    expect(errorMessage(new ApiError(422, "invalid_input", "bad ticker"))).toBe("bad ticker");
    expect(isRetryable(new ApiError(404, "ticker_not_found", "x"))).toBe(false);
  });
  it("serves fixtures without fetch when VITE_USE_FIXTURES=1", async () => {
    vi.stubEnv("VITE_USE_FIXTURES", "1");
    const fetch = stubFetch(async () => json(200, {}));
    expect((await api.getReport("t", "AAPL")).ticker).toBe("AAPL");
    expect((await caught(api.getReport("t", "ZZZZ"))).status).toBe(404);
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe("ensureSession", () => {
  it("reuses a stored thread id", async () => {
    sessionStorage.setItem(THREAD_KEY, "stored");
    const fetch = stubFetch(async () => json(200, { thread_id: "new" }));
    expect(await ensureSession()).toBe("stored");
    expect(fetch).not.toHaveBeenCalled();
  });
  it("creates and stores a session", async () => {
    stubFetch(async () => json(200, { thread_id: "new" }));
    expect(await ensureSession()).toBe("new");
    expect(sessionStorage.getItem(THREAD_KEY)).toBe("new");
  });
  it("survives storage that throws", async () => {
    const boom = () => { throw new Error("blocked"); };
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(boom);
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(boom);
    stubFetch(async () => json(200, { thread_id: "new" }));
    expect(await ensureSession()).toBe("new");
  });
});
```

- [ ] **Step 3: Run** `cd frontend && npx vitest run src/api.test.ts` → Expected: FAIL, cannot resolve `./api`.

- [ ] **Step 4: Implement `src/api.ts`**

```ts
import type { ChatAnswer, CreateSessionResponse, ErrorResponse, Report } from "./types";

export class ApiError extends Error {
  status: number; // 0 = network failure
  error_code: string;
  constructor(status: number, error_code: string, message: string) {
    super(message);
    this.name = "ApiError"; this.status = status; this.error_code = error_code;
  }
}

export const THREAD_KEY = "fap.thread_id";
const useFixtures = () => import.meta.env.VITE_USE_FIXTURES === "1"; // read per call so tests can stubEnv
const fixtureFiles = import.meta.glob<{ default: unknown }>("./fixtures/*.json");

async function fixture<T>(name: "session" | "report" | "chat"): Promise<T> {
  const load = fixtureFiles[`./fixtures/${name}.json`];
  if (!load) throw new ApiError(0, "fixture_missing", `Missing fixture ${name}.json`);
  return (await load()).default as T;
}

async function toApiError(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try { body = await res.json(); } catch { /* non-JSON body, e.g. proxy error page */ }
  if (body && typeof body === "object") {
    const b = body as Partial<ErrorResponse> & { detail?: unknown };
    if (typeof b.error_code === "string" && typeof b.message === "string") return new ApiError(res.status, b.error_code, b.message);
    if (typeof b.detail === "string") return new ApiError(res.status, "invalid_input", b.detail);
    if (Array.isArray(b.detail) && typeof b.detail[0]?.msg === "string") return new ApiError(res.status, "invalid_input", b.detail[0].msg);
  }
  return new ApiError(res.status, "http_error", `Request failed (HTTP ${res.status})`);
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body) });
  } catch {
    throw new ApiError(0, "network_error", "Can't reach the server. Check your connection and retry.");
  }
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

export const api = {
  createSession: (): Promise<CreateSessionResponse> => (useFixtures() ? fixture("session") : post("/api/sessions")),
  getReport: async (thread_id: string, ticker: string): Promise<Report> => {
    if (!useFixtures()) return post("/api/report", { thread_id, ticker });
    if (ticker !== "AAPL") throw new ApiError(404, "ticker_not_found", `Ticker not found: ${ticker}`);
    return fixture("report");
  },
  chat: (thread_id: string, message: string): Promise<ChatAnswer> =>
    useFixtures() ? fixture("chat") : post("/api/chat", { thread_id, message }),
};

export async function ensureSession(): Promise<string> {
  try { const stored = sessionStorage.getItem(THREAD_KEY); if (stored) return stored; }
  catch { /* storage blocked: fall through to a fresh session */ }
  const { thread_id } = await api.createSession();
  try { sessionStorage.setItem(THREAD_KEY, thread_id); } catch { /* keep in memory only */ }
  return thread_id;
}

export function errorMessage(e: unknown): string {
  if (!(e instanceof ApiError)) return "Something went wrong. Please retry.";
  if (e.status === 404) return "Ticker not found. Check the symbol and try again.";
  if (e.status === 409) return "Generate a report first";
  return e.message;
}

export const isRetryable = (e: unknown) => e instanceof ApiError && (e.status === 0 || e.status >= 500);
```

- [ ] **Step 5: Run** `npx vitest run src/api.test.ts` → Expected: 10 passed. `npm run typecheck` → exit 0.
- [ ] **Step 6: Commit** — `git add frontend/src && git commit -m "feat(fe): typed API client with ApiError, fixtures mode, session storage"`

---

### Task 3: Citation registry + formatters (TDD)

**Files:** Create `frontend/src/citations.ts`, `frontend/src/format.ts`. Test `frontend/src/citations.test.ts`.

**Design (§6.5.1):** ONE numbering for the whole session. Report claims are numbered first, in display order. Chat answers reuse existing numbers and append new sources. The registry is an immutable value (every update returns a new object), so React state updates stay trivial and tests stay pure.

**Interfaces:**
- Produces: `SECTIONS`, `type ClaimSection`; `interface NumberedSource { n: number; key: string; source: SourceRef | null }`; `interface Registry { numberOf: ReadonlyMap<string, number>; sources: readonly NumberedSource[] }`; `EMPTY_REGISTRY`; `sourceKey(s: SourceRef): string` (= `chunk_id ?? url`); `sourceFor(reg, key): SourceRef | null`; `registerReport(s: FilingsSummary): Registry`; `registerAnswer(reg: Registry, a: ChatAnswer): Registry`; `type Segment = { kind: "text"; text: string } | { kind: "cite"; n: number; key: string }`; `parseInline(text, numberOf): Segment[]`; `describeSource(s: SourceRef): string`. `format.ts`: `formatPrice, formatCompactUsd, formatInt, formatNumber` (accept `number | null | undefined`, `"—"` for missing), `formatChange(change, pct)`, `formatDateTime(iso)`.

- [ ] **Step 1: Write failing tests `src/citations.test.ts`**

```ts
import { describe, expect, it } from "vitest";
import { describeSource, parseInline, registerAnswer, registerReport } from "./citations";
import { formatChange, formatCompactUsd, formatDateTime, formatInt, formatPrice } from "./format";
import { avSrc, sampleReport, src10k, src8k } from "./test/sample";
import type { ChatAnswer, SourceRef } from "./types";

const answer = (text: string, citations: SourceRef[]): ChatAnswer =>
  ({ text, citations, route: "filings", warnings: [], disclaimer: "" });
const srcQ: SourceRef = { ...src10k, chunk_id: "Q:item-2:0", section: "Item 2. MD&A" };

describe("registry", () => {
  const reg = registerReport(sampleReport.filings!);

  it("numbers report chunk ids by first appearance in section order; null source if unindexed", () => {
    expect([...reg.numberOf.entries()]).toEqual([["E:8k-items:0", 1], ["K:item-1a:0", 2], ["missing:id", 3]]);
    expect(reg.sources.map((s) => s.source?.url ?? null)).toEqual([src8k.url, src10k.url, null]);
  });

  it("chat answers reuse report numbers and append new sources, inline order first", () => {
    const next = registerAnswer(reg, answer("Growth [Q:item-2:0]; risk [K:item-1a:0].", [src10k, avSrc("OVERVIEW"), srcQ]));
    expect(next.numberOf.get("K:item-1a:0")).toBe(2);
    expect(next.sources.slice(3).map((s) => [s.n, s.key])).toEqual([[4, "Q:item-2:0"], [5, avSrc("OVERVIEW").url]]);
    expect(reg.sources).toHaveLength(3); // immutable: original untouched
  });

  it("registering an already-known answer changes nothing", () => {
    const once = registerAnswer(reg, answer("x [Q:item-2:0]", [srcQ]));
    expect(registerAnswer(once, answer("y [Q:item-2:0]", [srcQ])).sources).toHaveLength(4);
  });
});

describe("parseInline", () => {
  const m = new Map([["K:item-1a:0", 1], ["E:8k-items:0", 2]]);
  it("splits text and known markers into segments", () => {
    expect(parseInline("See [K:item-1a:0] and [E:8k-items:0].", m)).toEqual([
      { kind: "text", text: "See " }, { kind: "cite", n: 1, key: "K:item-1a:0" },
      { kind: "text", text: " and " }, { kind: "cite", n: 2, key: "E:8k-items:0" },
      { kind: "text", text: "." },
    ]);
  });
  it("keeps unknown markers as literal text (never silently dropped)", () => {
    expect(parseInline("x [nope] y", m)).toEqual([{ kind: "text", text: "x [nope] y" }]);
  });
  it("handles adjacent markers and plain text", () => {
    expect(parseInline("[K:item-1a:0][E:8k-items:0]", m).map((s) => s.kind)).toEqual(["cite", "cite"]);
    expect(parseInline("plain", m)).toEqual([{ kind: "text", text: "plain" }]);
  });
});

describe("describeSource / format", () => {
  it("describes filings as form · section · filed date (verbatim date)", () => {
    expect(describeSource(src10k)).toBe("10-K · Item 1A. Risk Factors · filed 2025-10-31");
    expect(describeSource(avSrc("OVERVIEW"))).toMatch(/^Alpha Vantage · retrieved /);
  });
  it("formats numbers", () => {
    expect(formatPrice(227.52)).toBe("$227.52");
    expect(formatPrice(null)).toBe("—");
    expect(formatCompactUsd(3410000000000)).toBe("$3.41T");
    expect(formatInt(1234567)).toBe("1,234,567");
    expect(formatChange(1.2, 0.53)).toBe("+1.20 (+0.53%)");
    expect(formatChange(-2.5, -1.1)).toBe("−2.50 (−1.10%)");
    expect(formatDateTime("garbage")).toBe("garbage");
  });
});
```

- [ ] **Step 2: Run** `npx vitest run src/citations.test.ts` → Expected: FAIL, cannot resolve `./citations`.

- [ ] **Step 3: Implement `src/format.ts`**

```ts
const DASH = "—";
type N = number | null | undefined;
const ok = (v: N): v is number => typeof v === "number" && Number.isFinite(v);
const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const compactUsd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 2 });
const int = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const dec = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

export const formatPrice = (v: N) => (ok(v) ? usd.format(v) : DASH);
export const formatCompactUsd = (v: N) => (ok(v) ? compactUsd.format(v) : DASH);
export const formatInt = (v: N) => (ok(v) ? int.format(v) : DASH);
export const formatNumber = (v: N) => (ok(v) ? dec.format(v) : DASH);

export function formatChange(change: number, pct: number): string {
  const sign = change > 0 ? "+" : change < 0 ? "−" : "";
  return `${sign}${Math.abs(change).toFixed(2)} (${sign}${Math.abs(pct).toFixed(2)}%)`;
}

/** For datetimes only. Date-only strings ("2025-10-31") are displayed verbatim elsewhere. */
export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}
```

- [ ] **Step 4: Implement `src/citations.ts`**

```ts
import { formatDateTime } from "./format";
import type { ChatAnswer, FilingsSummary, SourceRef } from "./types";

export type ClaimSection = "key_developments" | "risk_factors" | "financial_highlights" | "material_events";
export const SECTIONS: ReadonlyArray<readonly [ClaimSection, string]> = [
  ["key_developments", "Key developments"],
  ["risk_factors", "Risk factors"],
  ["financial_highlights", "Financial highlights"],
  ["material_events", "Material events"],
];

export interface NumberedSource { n: number; key: string; source: SourceRef | null }

/** One citation numbering for the whole session: report claims first, chat answers append. */
export interface Registry { numberOf: ReadonlyMap<string, number>; sources: readonly NumberedSource[] }

export const EMPTY_REGISTRY: Registry = { numberOf: new Map(), sources: [] };
export const sourceKey = (s: SourceRef): string => s.chunk_id ?? s.url;

export function sourceFor(reg: Registry, key: string): SourceRef | null {
  const n = reg.numberOf.get(key);
  return n === undefined ? null : reg.sources[n - 1].source;
}

function add(reg: Registry, key: string, source: SourceRef | null): Registry {
  if (reg.numberOf.has(key)) return reg;
  const n = reg.sources.length + 1;
  return { numberOf: new Map(reg.numberOf).set(key, n), sources: [...reg.sources, { n, key, source }] };
}

export function registerReport(summary: FilingsSummary): Registry {
  let reg = EMPTY_REGISTRY;
  for (const [section] of SECTIONS) {
    for (const claim of summary[section]) {
      for (const id of claim.citations) reg = add(reg, id, summary.citation_index[id] ?? null);
    }
  }
  return reg;
}

// Chat answers cite inline as "[<chunk_id>]" (spec §6.3).
const MARKER = /\[([^[\]\s]+)\]/g;

export function registerAnswer(reg: Registry, answer: ChatAnswer): Registry {
  const byKey = new Map(answer.citations.map((s) => [sourceKey(s), s] as const));
  const inline = [...answer.text.matchAll(MARKER)].map((m) => m[1]).filter((id) => byKey.has(id));
  for (const key of [...inline, ...byKey.keys()]) reg = add(reg, key, byKey.get(key)!);
  return reg;
}

export type Segment = { kind: "text"; text: string } | { kind: "cite"; n: number; key: string };

export function parseInline(text: string, numberOf: ReadonlyMap<string, number>): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const n = numberOf.get(m[1]);
    if (n === undefined) continue; // unknown id: stays in the text verbatim
    if (m.index > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    out.push({ kind: "cite", n, key: m[1] });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

export function describeSource(s: SourceRef): string {
  if (s.provider === "alpha_vantage") return `Alpha Vantage · retrieved ${formatDateTime(s.retrieved_at)}`;
  return [s.form_type, s.section, s.filed_date && `filed ${s.filed_date}`].filter(Boolean).join(" · ");
}
```

- [ ] **Step 5: Run** `npx vitest run src/citations.test.ts` → Expected: 8 passed. `npm run typecheck` → exit 0.

- [ ] **Step 6: Commit**: `git add frontend/src && git commit -m "feat(fe): session-wide citation registry, inline marker parsing, formatters"`

---

### Task 4: Report display components (TDD)

**Files:** Create `frontend/src/components/{Cite,QuoteCard,FilingsReport,SourceList,Notices,LoadingStages}.tsx`. Test `frontend/src/components/report.test.tsx`.

**Interfaces:**
- Consumes: Task 3 exports; `sampleReport` from `src/test/sample.ts`.
- Produces (named exports): `Cite({ n, source: SourceRef | null })`, `sourceAnchor(n) => "src-{n}"`, `QuoteCard({ market })`, `FilingsReport({ summary, numberOf })`, `SourceList({ sources: readonly NumberedSource[] })`, `Warnings({ warnings })`, `Disclaimer({ text })`, `LoadingStages()`.
- `[n]` behaviour (§6.5.1): hover or focus opens a popover (form · section · filed date · "Open on sec.gov ↗"), and Escape closes it. Clicking scrolls to `#src-{n}` in the one global Sources list and flashes that entry. Sources are rendered once, by `App` (Task 5), **not** inside `FilingsReport`.

- [ ] **Step 1: Write failing tests `src/components/report.test.tsx`**

```tsx
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { registerReport } from "../citations";
import { sampleReport, src10k } from "../test/sample";
import { FilingsReport } from "./FilingsReport";
import { LoadingStages } from "./LoadingStages";
import { QuoteCard } from "./QuoteCard";
import { SourceList } from "./SourceList";

afterEach(() => { cleanup(); vi.useRealTimers(); });

describe("QuoteCard", () => {
  it("shows price, red change with ▼, dashes for missing fields, and the source", () => {
    render(<QuoteCard market={sampleReport.market!} />);
    expect(screen.getByText("$227.52")).toBeTruthy();
    expect(screen.getByText("−1.25 (−0.55%)").className).toContain("down");
    expect(screen.getByText("▼")).toBeTruthy(); // never colour alone
    expect(screen.getByText("$3.41T")).toBeTruthy();
    expect(screen.getAllByText("—")).toHaveLength(2); // pe_ratio + industry null
    expect(screen.getByRole("link", { name: "quote" }).getAttribute("href")).toContain("GLOBAL_QUOTE");
  });
});

describe("FilingsReport + SourceList", () => {
  const summary = sampleReport.filings!;
  const reg = registerReport(summary);
  const renderReport = () => render(
    <><FilingsReport summary={summary} numberOf={reg.numberOf} /><SourceList sources={reg.sources} /></>,
  );

  it("renders sections, numbered citations and one Sources list", () => {
    renderReport();
    for (const h of ["Key developments", "Risk factors", "Financial highlights", "Material events"]) {
      expect(screen.getByRole("heading", { name: h })).toBeTruthy();
    }
    expect(screen.getAllByText("[1]")).toHaveLength(3);
    expect(screen.getByText("[2]").getAttribute("href")).toBe("#src-2");
    expect(screen.getByText(/Source unavailable \(missing:id\)/)).toBeTruthy();
    expect(screen.getByText(/10-K · Item 1A\. Risk Factors · filed 2025-10-31/)).toBeTruthy();
    expect(document.getElementById("src-2")).toBeTruthy();
    expect(screen.getAllByRole("link", { name: "View on sec.gov" })).toHaveLength(2);
    expect(screen.getByText(/Nothing reported/)).toBeTruthy(); // empty financial_highlights
  });

  it("[n] shows a popover with source details and flashes its Sources entry on click", () => {
    renderReport();
    const two = screen.getByText("[2]");
    fireEvent.mouseEnter(two.parentElement!);
    const tip = screen.getByRole("tooltip");
    expect(tip.textContent).toContain("10-K · Item 1A. Risk Factors · filed 2025-10-31");
    expect(within(tip).getByRole("link").getAttribute("href")).toBe(src10k.url);
    fireEvent.keyDown(two, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).toBeNull();
    fireEvent.click(two);
    expect(document.getElementById("src-2")!.className).toContain("flash");
  });
});

describe("LoadingStages", () => {
  it("keeps the spec copy and advances the stage line on a timer", () => {
    vi.useFakeTimers();
    render(<LoadingStages />);
    expect(screen.getByText("Fetching quote & filings…")).toBeTruthy();
    expect(screen.getByText(/Resolving ticker…/)).toBeTruthy();
    act(() => { vi.advanceTimersByTime(6000); });
    expect(screen.getByText(/Reading SEC filings…/)).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run** `npx vitest run src/components` → Expected: FAIL, cannot resolve `./FilingsReport`.

- [ ] **Step 3: Implement `src/components/Cite.tsx`**

```tsx
import { useId, useState, type MouseEvent } from "react";
import { describeSource } from "../citations";
import type { SourceRef } from "../types";

export const sourceAnchor = (n: number) => `src-${n}`;

function jumpTo(e: MouseEvent, n: number) {
  e.preventDefault();
  const el = document.getElementById(sourceAnchor(n));
  if (!el) return;
  el.scrollIntoView?.({ behavior: "smooth", block: "center" }); // absent in jsdom
  el.classList.remove("flash");
  void el.offsetWidth; // force reflow so the animation restarts on repeat clicks
  el.classList.add("flash");
}

export function Cite({ n, source }: { n: number; source: SourceRef | null }) {
  const [open, setOpen] = useState(false);
  const tipId = useId();
  return (
    <sup
      className="cite"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOpen(false); }}
      onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}
    >
      <a href={`#${sourceAnchor(n)}`} onClick={(e) => jumpTo(e, n)} aria-describedby={open ? tipId : undefined}>
        {`[${n}]`}
      </a>
      {open && (
        <span role="tooltip" id={tipId} className="cite-pop">
          {source ? (
            <>
              <span>{describeSource(source)}</span>
              <a href={source.url} target="_blank" rel="noreferrer">
                Open on {source.provider === "sec_edgar" ? "sec.gov" : "alphavantage.co"} ↗
              </a>
            </>
          ) : (
            <span className="muted">Source unavailable</span>
          )}
        </span>
      )}
    </sup>
  );
}
```

- [ ] **Step 4: Implement `src/components/SourceList.tsx` and `Notices.tsx`**

```tsx
import { describeSource, type NumberedSource } from "../citations";
import { sourceAnchor } from "./Cite";

export function SourceList({ sources }: { sources: readonly NumberedSource[] }) {
  return (
    <section className="card sources" aria-label="Sources">
      <h3>Sources</h3>
      <ol>
        {sources.map(({ n, key, source }) => (
          <li key={n} value={n} id={sourceAnchor(n)}>
            {source ? (
              <>
                {describeSource(source)} ·{" "}
                <a href={source.url} target="_blank" rel="noreferrer">
                  {source.provider === "sec_edgar" ? "sec.gov" : "alphavantage.co"}
                </a>
              </>
            ) : (
              <span className="muted">Source unavailable ({key})</span>
            )}
          </li>
        ))}
      </ol>
    </section>
  );
}
```

```tsx
export function Warnings({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <div className="warnings" role="alert">
      <strong>Heads up:</strong>
      <ul>{warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
    </div>
  );
}

export function Disclaimer({ text }: { text: string }) {
  return <p className="disclaimer">{text}</p>;
}
```

- [ ] **Step 5: Implement `src/components/QuoteCard.tsx`**

```tsx
import { formatChange, formatCompactUsd, formatDateTime, formatInt, formatNumber, formatPrice } from "../format";
import type { MarketSnapshot } from "../types";

export function QuoteCard({ market }: { market: MarketSnapshot }) {
  const { quote: q, overview: o } = market;
  const dir = q.change > 0 ? "up" : q.change < 0 ? "down" : "flat";
  return (
    <section className="card quote-card" aria-label="Market data">
      <h2>{o.name} <span className="muted">{q.symbol}</span> {o.sector && <span className="tag">{o.sector}</span>}</h2>
      <div className="price-row">
        <span className="price num">{formatPrice(q.price)}</span>
        {dir !== "flat" && <span className={`arrow ${dir}`} aria-hidden="true">{dir === "up" ? "▲" : "▼"}</span>}
        <span className={`change num ${dir}`}>{formatChange(q.change, q.change_pct)}</span>
      </div>
      <dl className="stats">
        <div><dt>Volume</dt><dd>{formatInt(q.volume)}</dd></div>
        <div><dt>Market cap</dt><dd>{formatCompactUsd(o.market_cap)}</dd></div>
        <div><dt>P/E</dt><dd>{formatNumber(o.pe_ratio)}</dd></div>
        <div><dt>52-week range</dt><dd>{formatPrice(o.week52_low)} – {formatPrice(o.week52_high)}</dd></div>
        <div><dt>Trading day</dt><dd>{q.latest_trading_day}</dd></div>
        <div><dt>Industry</dt><dd>{o.industry ?? "—"}</dd></div>
      </dl>
      <footer className="source-line">
        Source: Alpha Vantage (
        <a href={q.source.url} target="_blank" rel="noreferrer">quote</a>,{" "}
        <a href={o.source.url} target="_blank" rel="noreferrer">overview</a>) · retrieved{" "}
        {formatDateTime(q.source.retrieved_at)}
      </footer>
    </section>
  );
}
```

- [ ] **Step 6: Implement `src/components/FilingsReport.tsx`**

```tsx
import { SECTIONS } from "../citations";
import type { FilingsSummary } from "../types";
import { Cite } from "./Cite";

export function FilingsReport({ summary, numberOf }: {
  summary: FilingsSummary; numberOf: ReadonlyMap<string, number>;
}) {
  return (
    <section className="card filings" aria-label="SEC filings summary">
      <h2>SEC filings</h2>
      {summary.filings.length === 0 && <p className="muted">No recent 10-K, 10-Q or 8-K filings found.</p>}
      <ul className="filing-list">
        {summary.filings.map((f) => (
          <li key={f.accession_no}>
            <span className="form-badge">{f.form_type}</span> filed <span className="num">{f.filed_date}</span>
            {f.items.length > 0 && ` · Items ${f.items.join(", ")}`} ·{" "}
            <a href={f.primary_doc_url} target="_blank" rel="noreferrer">View on sec.gov</a>
          </li>
        ))}
      </ul>
      {SECTIONS.map(([key, title]) => (
        <div key={key} className="filing-section">
          <h3>{title}</h3>
          {summary[key].length === 0 ? (
            <p className="muted">Nothing reported in the retrieved filings.</p>
          ) : (
            <ul>
              {summary[key].map((claim, i) => (
                <li key={i}>
                  {claim.text}
                  {[...new Set(claim.citations)].map((id) => {
                    const n = numberOf.get(id);
                    return n === undefined ? null : <Cite key={id} n={n} source={summary.citation_index[id] ?? null} />;
                  })}
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </section>
  );
}
```

- [ ] **Step 7: Implement `src/components/LoadingStages.tsx`**

```tsx
import { useEffect, useState } from "react";

// Cosmetic only: the API doesn't stream progress. Timings roughly match a cold report run.
const STAGES: ReadonlyArray<readonly [number, string]> = [
  [0, "Resolving ticker…"],
  [2, "Fetching quote…"],
  [5, "Reading SEC filings…"],
  [15, "Summarising filings…"],
];

export function LoadingStages() {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);
  const stage = [...STAGES].reverse().find(([at]) => elapsed >= at)![1];
  return (
    <div className="loading" role="status">
      <p>
        <strong>Fetching quote & filings…</strong>{" "}
        <span className="muted">{stage} <span className="num">{elapsed}s</span></span>
      </p>
      <div className="card skeleton" aria-hidden="true" />
      <div className="card skeleton tall" aria-hidden="true" />
    </div>
  );
}
```

- [ ] **Step 8: Run** `npx vitest run src/components` → Expected: 4 passed. `npm run typecheck` → exit 0.

- [ ] **Step 9: Commit**: `git add frontend/src && git commit -m "feat(fe): quote card, cited filings report, citation popovers, sources, loading stages"`

---

### Task 5: Ticker form, chat panel, two-pane App, theme (TDD)

**Files:** Create `frontend/src/components/{TickerForm,ChatPanel}.tsx`, test `frontend/src/App.test.tsx`. Replace `frontend/src/App.tsx`, `frontend/src/main.tsx`, `frontend/src/index.css`.

**Interfaces:**
- Consumes: `api`, `ensureSession`, `errorMessage`, `isRetryable` (Task 2); `Registry`, `EMPTY_REGISTRY`, `registerReport`, `registerAnswer`, `parseInline`, `sourceKey`, `sourceFor` (Task 3); Task 4 components.
- Produces: `TickerForm({ disabled, onSubmit(ticker) })`, `ChatPanel({ threadId, ticker: string | null, registry: Registry, onAnswer(a: ChatAnswer): void })`, default-export `App`.
- `App` owns the `Registry`. It resets the registry on each new report and folds each chat answer in via `onAnswer`. Chat input is **disabled until a report exists**. A 409 can still happen (for example, the stub server restarted and lost its in-memory reports) and shows "Generate a report first".

- [ ] **Step 1: Write failing tests `src/App.test.tsx`**

```tsx
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { json, sampleReport, src10k } from "./test/sample";

type Handler = () => Response | Promise<Response>;
function route(handlers: Record<string, Handler[]>) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const h = handlers[url]?.shift();
    if (!h) throw new Error(`unexpected fetch ${url}`);
    return h();
  }));
}
const session = () => json(200, { thread_id: "t1" });

async function submit(ticker: string) {
  const btn = screen.getByRole("button", { name: "Get report" }) as HTMLButtonElement;
  fireEvent.change(screen.getByLabelText("Ticker"), { target: { value: ticker } });
  await waitFor(() => expect(btn.disabled).toBe(false));
  fireEvent.click(btn);
}
async function ask(message: string) {
  const box = screen.getByLabelText("Ask a follow-up") as HTMLTextAreaElement;
  await waitFor(() => expect(box.disabled).toBe(false));
  fireEvent.change(box, { target: { value: message } });
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
}

beforeEach(() => sessionStorage.clear());
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("shows loading copy, then the report; disclaimer always visible", async () => {
  let resolve!: (r: Response) => void;
  route({ "/api/sessions": [session], "/api/report": [() => new Promise<Response>((r) => { resolve = r; })] });
  render(<App />);
  expect(screen.getByText(/Not investment advice/)).toBeTruthy();
  await submit("aapl");
  expect(await screen.findByText(/Fetching quote & filings…/)).toBeTruthy();
  resolve(json(200, sampleReport));
  expect(await screen.findByText("Apple Inc. (AAPL)")).toBeTruthy();
  expect(screen.getAllByText("[1]").length).toBeGreaterThan(0);
  expect(screen.getByRole("region", { name: "Sources" })).toBeTruthy();
});

it("404 shows Ticker not found without a retry button", async () => {
  route({ "/api/sessions": [session],
    "/api/report": [() => json(404, { error_code: "ticker_not_found", message: "Ticker not found: ZZZZ" })] });
  render(<App />);
  await submit("ZZZZ");
  expect(await screen.findByText(/Ticker not found/)).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
});

it("network error offers Retry, which refetches", async () => {
  route({ "/api/sessions": [session],
    "/api/report": [() => { throw new TypeError("Failed to fetch"); }, () => json(200, sampleReport)] });
  render(<App />);
  await submit("AAPL");
  fireEvent.click(await screen.findByRole("button", { name: "Retry" }));
  expect(await screen.findByText("Apple Inc. (AAPL)")).toBeTruthy();
});

it("partial report: market placeholder + warnings, filings still shown", async () => {
  const partial = { ...sampleReport, market: null, warnings: ["Market data unavailable: rate limited"] };
  route({ "/api/sessions": [session], "/api/report": [() => json(200, partial)] });
  render(<App />);
  await submit("AAPL");
  expect(await screen.findByText(/Market data unavailable — see warnings/)).toBeTruthy();
  expect(screen.getByText("Market data unavailable: rate limited")).toBeTruthy();
  expect(screen.getByRole("heading", { name: "Risk factors" })).toBeTruthy();
});

it("chat is disabled until a report exists", async () => {
  route({ "/api/sessions": [session] });
  render(<App />);
  const box = (await screen.findByLabelText("Ask a follow-up")) as HTMLTextAreaElement;
  expect(box.disabled).toBe(true);
  expect(box.placeholder).toBe("Generate a report first");
});

it("409 from chat (server lost the report) shows Generate a report first", async () => {
  route({ "/api/sessions": [session], "/api/report": [() => json(200, sampleReport)],
    "/api/chat": [() => json(409, { error_code: "no_report_yet", message: "No report" })] });
  render(<App />);
  await submit("AAPL");
  await screen.findByText("Apple Inc. (AAPL)");
  await ask("What are the risks?");
  expect(await screen.findByText("Generate a report first")).toBeTruthy();
});

it("chat reuses report citation numbers and appends new sources to the list", async () => {
  const srcQ = { ...src10k, chunk_id: "Q:item-2:0", section: "Item 2. MD&A" };
  route({ "/api/sessions": [session], "/api/report": [() => json(200, sampleReport)],
    "/api/chat": [() => json(200, { text: "Risk [K:item-1a:0]; growth [Q:item-2:0].",
      citations: [src10k, srcQ], route: "filings", warnings: [], disclaimer: "x" })] });
  render(<App />);
  await submit("AAPL");
  await screen.findByText("Apple Inc. (AAPL)");
  await ask("risks and growth?");
  await screen.findByText("[4]");
  const answer = document.querySelector(".msg-assistant")!;
  expect([...answer.querySelectorAll(".cite > a")].map((a) => a.textContent)).toEqual(["[2]", "[4]"]);
  expect(answer.textContent).toContain("SEC filings"); // route badge
  expect(document.getElementById("src-4")!.textContent).toContain("Item 2. MD&A");
});
```

- [ ] **Step 2: Run** `npx vitest run src/App.test.tsx` → Expected: FAIL (the placeholder `App` renders nothing).

- [ ] **Step 3: Implement `src/components/TickerForm.tsx`**

```tsx
import { useState } from "react";

const TICKER_RE = /^[A-Z][A-Z.-]{0,5}$/;

export function TickerForm({ disabled, onSubmit }: { disabled: boolean; onSubmit: (ticker: string) => void }) {
  const [value, setValue] = useState("");
  const ticker = value.trim().toUpperCase();
  const valid = TICKER_RE.test(ticker);
  return (
    <form className="ticker-form" onSubmit={(e) => { e.preventDefault(); if (valid && !disabled) onSubmit(ticker); }}>
      <label htmlFor="ticker">Ticker</label>
      <input id="ticker" value={value} onChange={(e) => setValue(e.target.value)} placeholder="AAPL"
        maxLength={8} autoComplete="off" spellCheck={false} />
      <button type="submit" disabled={disabled || !valid}>Get report</button>
    </form>
  );
}
```

- [ ] **Step 4: Implement `src/components/ChatPanel.tsx`**

```tsx
import { Fragment, useState, type FormEvent, type KeyboardEvent } from "react";
import { api, errorMessage } from "../api";
import { parseInline, sourceFor, sourceKey, type Registry } from "../citations";
import type { ChatAnswer, Route } from "../types";
import { Cite } from "./Cite";

const MAX = 1000;
const ROUTE_LABEL: Record<Route, string> = {
  market: "Market data", filings: "SEC filings", both: "Market + filings", off_topic: "Out of scope",
};
type Item = { role: "user"; text: string } | { role: "assistant"; answer: ChatAnswer } | { role: "error"; text: string };

function Answer({ answer, registry }: { answer: ChatAnswer; registry: Registry }) {
  const segments = parseInline(answer.text, registry.numberOf);
  const inline = new Set(segments.flatMap((s) => (s.kind === "cite" ? [s.key] : [])));
  // Citations the text didn't mention inline still get shown, as chips under the answer.
  const extra = [...new Set(answer.citations.map(sourceKey))].filter((k) => !inline.has(k) && registry.numberOf.has(k));
  return (
    <>
      <span className={`route route-${answer.route}`}>{ROUTE_LABEL[answer.route]}</span>
      <p className="answer-text">
        {segments.map((s, i) =>
          s.kind === "text" ? <Fragment key={i}>{s.text}</Fragment>
            : <Cite key={i} n={s.n} source={sourceFor(registry, s.key)} />)}
      </p>
      {extra.length > 0 && (
        <p className="answer-sources">
          Sources:{" "}
          {extra.map((k) => <Cite key={k} n={registry.numberOf.get(k)!} source={sourceFor(registry, k)} />)}
        </p>
      )}
      {answer.warnings.length > 0 && <p className="msg-warning">{answer.warnings.join(" ")}</p>}
    </>
  );
}

export function ChatPanel({ threadId, ticker, registry, onAnswer }: {
  threadId: string; ticker: string | null; registry: Registry; onAnswer: (a: ChatAnswer) => void;
}) {
  const [items, setItems] = useState<Item[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const ready = ticker !== null;

  async function send(e?: FormEvent) {
    e?.preventDefault();
    const msg = input.trim();
    if (!ready || !msg || busy || msg.length > MAX) return;
    setItems((xs) => [...xs, { role: "user", text: msg }]);
    setInput("");
    setBusy(true);
    try {
      const answer = await api.chat(threadId, msg);
      onAnswer(answer); // registry update + message append batch into one render
      setItems((xs) => [...xs, { role: "assistant", answer }]);
    } catch (err) {
      setItems((xs) => [...xs, { role: "error", text: errorMessage(err) }]);
      setInput(msg); // let them resend without retyping
    } finally {
      setBusy(false);
    }
  }
  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); }
  }

  return (
    <section className="chat" aria-label="Follow-up chat">
      <h2>Ask about {ticker ?? "a security"}</h2>
      <ol className="chat-log" aria-live="polite">
        {items.length === 0 && (
          <li className="muted">{ready ? "e.g. What did they say about supply-chain risk?" : "Generate a report, then ask follow-up questions here."}</li>
        )}
        {items.map((it, i) => (
          <li key={i} className={`msg msg-${it.role}`}>
            {it.role === "assistant" ? <Answer answer={it.answer} registry={registry} /> : it.text}
          </li>
        ))}
        {busy && <li className="msg muted">Thinking…</li>}
      </ol>
      <form className="chat-form" onSubmit={send}>
        <textarea aria-label="Ask a follow-up" value={input} maxLength={MAX} rows={2} disabled={!ready}
          onChange={(e) => setInput(e.target.value)} onKeyDown={onKey}
          placeholder={ready ? "Ask a follow-up…" : "Generate a report first"} />
        <div className="chat-actions">
          <span className="muted num">{input.length}/{MAX}</span>
          <button type="submit" disabled={!ready || busy || !input.trim()}>Send</button>
        </div>
      </form>
    </section>
  );
}
```

- [ ] **Step 5: Implement `src/App.tsx` and `src/main.tsx`**

```tsx
import { useCallback, useEffect, useState } from "react";
import { api, ensureSession, errorMessage, isRetryable } from "./api";
import { EMPTY_REGISTRY, registerAnswer, registerReport, type Registry } from "./citations";
import { ChatPanel } from "./components/ChatPanel";
import { FilingsReport } from "./components/FilingsReport";
import { LoadingStages } from "./components/LoadingStages";
import { Disclaimer, Warnings } from "./components/Notices";
import { QuoteCard } from "./components/QuoteCard";
import { SourceList } from "./components/SourceList";
import { TickerForm } from "./components/TickerForm";
import { formatDateTime } from "./format";
import { DISCLAIMER, type ChatAnswer, type Report } from "./types";

type Failure = { message: string; retry?: () => void };

function ErrorBox({ message, retry }: Failure) {
  return (
    <div className="error" role="alert">
      <span>{message}</span>
      {retry && <button type="button" onClick={retry}>Retry</button>}
    </div>
  );
}

export default function App() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [sessionError, setSessionError] = useState<Failure | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [registry, setRegistry] = useState<Registry>(EMPTY_REGISTRY);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Failure | null>(null);

  const startSession = useCallback(async () => {
    setSessionError(null);
    try { setThreadId(await ensureSession()); }
    catch (e) { setSessionError({ message: errorMessage(e), retry: () => void startSession() }); }
  }, []);
  useEffect(() => { void startSession(); }, [startSession]);

  async function loadReport(ticker: string) {
    if (!threadId) return;
    setLoading(true); setError(null); setReport(null); setRegistry(EMPTY_REGISTRY);
    try {
      const r = await api.getReport(threadId, ticker);
      setRegistry(r.filings ? registerReport(r.filings) : EMPTY_REGISTRY);
      setReport(r);
    } catch (e) {
      setError({ message: errorMessage(e), retry: isRetryable(e) ? () => void loadReport(ticker) : undefined });
    } finally {
      setLoading(false);
    }
  }
  const onAnswer = useCallback((a: ChatAnswer) => setRegistry((r) => registerAnswer(r, a)), []);

  return (
    <div className="app">
      <header className="topbar">
        <h1>Security Brief</h1>
        <TickerForm disabled={!threadId || loading} onSubmit={(t) => void loadReport(t)} />
        <Disclaimer text={report?.disclaimer ?? DISCLAIMER} />
      </header>
      <div className="layout">
        <main className="report-pane" aria-busy={loading}>
          {sessionError && <ErrorBox {...sessionError} />}
          {error && <ErrorBox {...error} />}
          {loading && <LoadingStages />}
          {!report && !loading && !error && (
            <p className="muted empty">
              Enter a ticker for a sourced brief: live quote plus a summary of the latest 10-K, 10-Q and recent 8-Ks.
            </p>
          )}
          {report && (
            <>
              <div className="report-head">
                <h2 className="company">{report.company_name} ({report.ticker})</h2>
                <span className="muted">Generated {formatDateTime(report.generated_at)}</span>
              </div>
              <Warnings warnings={report.warnings} />
              {report.market ? <QuoteCard market={report.market} />
                : <section className="card muted">Market data unavailable — see warnings.</section>}
              {report.filings ? <FilingsReport summary={report.filings} numberOf={registry.numberOf} />
                : <section className="card muted">Filings summary unavailable — see warnings.</section>}
            </>
          )}
          {registry.sources.length > 0 && <SourceList sources={registry.sources} />}
        </main>
        <aside className="chat-pane">
          {threadId && (
            <ChatPanel key={report?.generated_at ?? "none"} threadId={threadId}
              ticker={report?.ticker ?? null} registry={registry} onAnswer={onAnswer} />
          )}
        </aside>
      </div>
    </div>
  );
}
```

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
```

- [ ] **Step 6: Replace `src/index.css`**: "terminal-lite" theme, dark by default, light under `prefers-color-scheme`

```css
:root {
  color-scheme: dark;
  --bg: #0d1117; --panel: #151b23; --panel-2: #1c232d; --border: #2a323d;
  --text: #e6edf3; --muted: #8b98a8; --accent: #58a6ff; --accent-soft: rgba(88, 166, 255, 0.18);
  --up: #3fb950; --down: #f85149; --warn: #d29922; --warn-bg: rgba(210, 153, 34, 0.12); --err-bg: rgba(248, 81, 73, 0.12);
  --mono: ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 15px; line-height: 1.5; color: var(--text);
}
@media (prefers-color-scheme: light) {
  :root {
    color-scheme: light;
    --bg: #f6f8fa; --panel: #ffffff; --panel-2: #f0f3f6; --border: #d0d7de;
    --text: #1f2328; --muted: #59636e; --accent: #0969da; --accent-soft: rgba(9, 105, 218, 0.14);
    --up: #1a7f37; --down: #cf222e; --warn: #9a6700; --warn-bg: #fff8c5; --err-bg: #ffebe9;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); }
a { color: var(--accent); }
.muted { color: var(--muted); }
.num, .price, .change, .stats dd { font-variant-numeric: tabular-nums; }

/* Shell: header + two panes. Report pane scrolls; chat pane stays put. */
.app { height: 100vh; display: flex; flex-direction: column; }
.topbar { flex: none; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 20px; padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border); }
.topbar h1 { margin: 0; font-size: 0.95rem; font-family: var(--mono); letter-spacing: 0.06em; text-transform: uppercase; }
.disclaimer { margin: 0 0 0 auto; max-width: 46rem; font-size: 0.75rem; color: var(--muted); border-left: 2px solid var(--warn); padding-left: 8px; }
.layout { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 65fr) minmax(0, 35fr); }
.report-pane { overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
.chat-pane { min-height: 0; display: flex; border-left: 1px solid var(--border); background: var(--panel); }
@media (max-width: 900px) {
  .app { height: auto; }
  .layout { grid-template-columns: minmax(0, 1fr); }
  .report-pane { overflow: visible; }
  .chat-pane { border-left: 0; border-top: 1px solid var(--border); }
  .chat-log { max-height: 60vh; }
  .disclaimer { margin-left: 0; }
}

/* Form controls */
.ticker-form { display: flex; gap: 8px; align-items: center; }
.ticker-form label { font-size: 0.75rem; color: var(--muted); }
input, textarea { font: inherit; color: var(--text); background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px; }
.ticker-form input { width: 8rem; font-family: var(--mono); text-transform: uppercase; }
button { font: inherit; font-weight: 600; padding: 7px 14px; border-radius: 6px; border: 1px solid var(--accent); background: var(--accent); color: var(--bg); cursor: pointer; }
button:disabled, textarea:disabled { opacity: 0.45; cursor: not-allowed; }
:is(input, textarea, button, a):focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }

/* Notices */
.error { display: flex; gap: 12px; align-items: center; justify-content: space-between; background: var(--err-bg); border: 1px solid var(--down); border-radius: 8px; padding: 10px 12px; }
.warnings { background: var(--warn-bg); border: 1px solid var(--warn); border-radius: 8px; padding: 10px 12px; }
.warnings ul { margin: 4px 0 0; padding-left: 20px; }
.empty { margin: 48px auto; max-width: 32rem; text-align: center; }

/* Loading */
.loading p { margin: 0 0 12px; }
.skeleton { height: 140px; margin-bottom: 12px; background: linear-gradient(90deg, var(--panel) 0%, var(--panel-2) 50%, var(--panel) 100%); background-size: 200% 100%; animation: shimmer 1.4s linear infinite; }
.skeleton.tall { height: 320px; }
@keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }

/* Cards */
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; min-width: 0; }
.card h2 { margin: 0 0 8px; font-size: 1rem; }
.report-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.company { margin: 0; font-size: 1.25rem; }
.price-row { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
.price { font-size: 2rem; font-weight: 600; font-family: var(--mono); }
.change, .arrow { font-family: var(--mono); }
.up { color: var(--up); } .down { color: var(--down); }
.stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px 16px; margin: 12px 0; }
@media (max-width: 600px) { .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.stats dt { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.stats dd { margin: 0; font-family: var(--mono); }
.tag, .form-badge, .route { display: inline-block; font-family: var(--mono); font-size: 0.72rem; font-weight: 600; padding: 1px 8px; border-radius: 999px; background: var(--accent-soft); color: var(--accent); }
.route-off_topic { background: var(--err-bg); color: var(--down); }
.source-line { font-size: 0.78rem; color: var(--muted); border-top: 1px solid var(--border); padding-top: 8px; }

/* Filings */
.filing-list { list-style: none; padding: 0; margin: 0 0 8px; }
.filing-list li { padding: 3px 0; overflow-wrap: anywhere; }
.filing-section h3, .sources h3 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin: 18px 0 6px; }
.filing-section ul { margin: 0; padding-left: 20px; }
.filing-section li { margin: 5px 0; }

/* Citations */
.cite { position: relative; font-family: var(--mono); font-size: 0.7rem; margin-left: 1px; }
.cite > a { text-decoration: none; }
.cite-pop { position: absolute; z-index: 10; top: 100%; left: 0; display: flex; flex-direction: column; gap: 4px; width: max-content; max-width: min(22rem, 80vw); padding: 8px 10px; font-family: system-ui, sans-serif; font-size: 0.8rem; line-height: 1.4; color: var(--text); background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35); }
.chat .cite-pop { left: auto; right: 0; }
.sources { font-size: 0.82rem; }
.sources h3 { margin-top: 0; }
.sources ol { margin: 0; padding-left: 28px; }
.sources li { overflow-wrap: anywhere; margin: 2px 0; border-radius: 4px; }
.flash { animation: flash 1.6s ease-out; }
@keyframes flash { from { background: var(--accent-soft); } to { background: transparent; } }

/* Chat */
.chat { flex: 1; min-height: 0; display: flex; flex-direction: column; padding: 12px 16px; }
.chat h2 { margin: 0 0 8px; font-size: 1rem; }
.chat-log { flex: 1; min-height: 0; overflow-y: auto; list-style: none; padding: 0; margin: 0 0 12px; display: flex; flex-direction: column; gap: 8px; }
.msg { padding: 8px 12px; border-radius: 8px; overflow-wrap: anywhere; }
.msg-user { background: var(--accent-soft); align-self: flex-end; max-width: 85%; white-space: pre-wrap; }
.msg-assistant { background: var(--bg); border: 1px solid var(--border); }
.msg-error { background: var(--err-bg); }
.answer-text { white-space: pre-wrap; margin: 6px 0; }
.answer-sources { margin: 0; font-size: 0.8rem; color: var(--muted); }
.msg-warning { color: var(--warn); font-size: 0.82rem; margin: 0; }
.chat-form textarea { width: 100%; resize: vertical; }
.chat-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 6px; }
```

- [ ] **Step 7: Run all**: `npm test` → Expected: all files pass (api 10, citations 8, report 4, App 7). `npm run typecheck && npm run build` → exit 0.

- [ ] **Step 8: Commit**: `git add frontend && git commit -m "feat(fe): two-pane app, chat with shared citations, terminal-lite theme"`

---

### Task 6: End-to-end check against the stub API

**Files:** none unless a defect is found (fix inside `frontend/` only, add a test for it first).

- [ ] **Step 1: Stub API + dev server**: backend running (Task 1 Step 3), then `cd frontend && npm run dev` (background). Open http://localhost:5173.
- [ ] **Step 2: Walk the flow**:
  - (a) On load: disclaimer is in the header, `sessionStorage['fap.thread_id']` is set, the chat input is disabled and reads "Generate a report first". Reload → the Network tab shows no new `POST /api/sessions`.
  - (b) Enter `AAPL`:
    - "Fetching quote & filings…" appears with a stage line that advances and skeleton cards.
    - Then the quote card: change coloured with ▲/▼, and a source footer.
    - Then the filings list with sec.gov links, four sections with `[n]`, and one Sources list at the bottom of the report pane.
  - (c) Hover `[2]` → popover (form · section · filed date · "Open on sec.gov ↗"). Tab to it → the popover opens on focus and Escape closes it. Click → the report pane scrolls to Sources entry 2 and it flashes.
  - (d) Ask "What are the main risk factors?" in chat. You should see the route badge, and inline `[n]` numbers that **match the report's numbers** for the same chunks. Any new chunks get the next number and appear at the end of Sources. The report stays visible while you chat.
  - (e) `ZZZZ` → "Ticker not found".
  - (f) Restart the stub backend (it forgets reports), then chat → "Generate a report first".
  - (g) Stop the backend and submit → error with Retry. Restart the backend, press Retry → the report loads.
- [ ] **Step 3: Fixture mode**: `VITE_USE_FIXTURES=1 npm run dev` with the backend stopped → the AAPL report and chat render from fixtures. The fixture chat's inline `[chunk_id]` markers render as `[n]`, never as raw accession numbers.
- [ ] **Step 4: Responsive + theme**: at 375px wide, nothing scrolls horizontally, the panes stack (chat below the report), and long sec.gov URLs wrap. In DevTools, emulate `prefers-color-scheme: light` → the light palette applies and green/red still read clearly.
- [ ] **Step 5: Final gate**: `npm run typecheck && npm run build && npm test` → all green. Commit any fixes: `git commit -am "fix(fe): <what>"`.

---

## Done checklist

- [ ] `npm run typecheck`, `npm run build`, `npm test` green. Coverage includes: shared citation registry and inline parsing, the popover/jump, ApiError parsing (including non-JSON and FastAPI 422), and the 404/409/network/partial-report UI.
- [ ] Types mirror `models.py` field names; no files outside `frontend/` changed.
- [ ] Manual walk (Task 6 Steps 2–4) passed against the stub API and in fixture mode.
- [ ] Report status to the human (what passed, which cut-order items were cut if any, any contract questions). **Do not merge**; the integrator merges in Phase 2.
