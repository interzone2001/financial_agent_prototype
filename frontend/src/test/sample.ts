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
