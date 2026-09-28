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
