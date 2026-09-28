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
