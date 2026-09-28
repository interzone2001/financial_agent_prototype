import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

it("chat log auto-scrolls to the newest message", async () => {
  const orig = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollHeight");
  Object.defineProperty(HTMLElement.prototype, "scrollHeight", { configurable: true, get: () => 500 });
  try {
    route({ "/api/sessions": [session], "/api/report": [() => json(200, sampleReport)],
      "/api/chat": [() => json(200, { text: "Risk [K:item-1a:0].",
        citations: [src10k], route: "filings", warnings: [], disclaimer: "x" })] });
    render(<App />);
    await submit("AAPL");
    await screen.findByText("Apple Inc. (AAPL)");
    await ask("what are the risks?");
    await waitFor(() => expect(document.querySelector(".msg-assistant")).toBeTruthy());
    const log = document.querySelector(".chat-log") as HTMLOListElement;
    expect(log.scrollTop).toBe(500);
  } finally {
    if (orig) Object.defineProperty(HTMLElement.prototype, "scrollHeight", orig);
    else delete (HTMLElement.prototype as unknown as { scrollHeight?: number }).scrollHeight;
  }
});

it("stale chat answer from a remounted ChatPanel doesn't leak into the new report's registry", async () => {
  let resolveChat!: (r: Response) => void;
  const second = { ...sampleReport, generated_at: "2026-09-28T19:00:00Z" };
  route({
    "/api/sessions": [session],
    "/api/report": [() => json(200, sampleReport), () => json(200, second)],
    "/api/chat": [() => new Promise<Response>((r) => { resolveChat = r; })],
  });
  render(<App />);
  await submit("AAPL");
  await screen.findByText("Apple Inc. (AAPL)");
  await ask("risks and growth?");
  await submit("AAPL"); // second report while chat still pending; ChatPanel remounts
  expect(await screen.findByText(/What did they say about supply-chain risk/)).toBeTruthy();

  const srcQ = { ...src10k, chunk_id: "Q:item-2:0", section: "Item 2. MD&A" };
  await act(async () => {
    resolveChat(json(200, { text: "Risk [K:item-1a:0]; growth [Q:item-2:0].",
      citations: [src10k, srcQ], route: "filings", warnings: [], disclaimer: "x" }));
  });

  expect(document.getElementById("src-4")).toBeNull();
  expect(document.querySelector(".msg-assistant")).toBeNull();
});
