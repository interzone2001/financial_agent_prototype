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
