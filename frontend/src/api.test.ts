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
