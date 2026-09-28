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
