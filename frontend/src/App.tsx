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
