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
