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
