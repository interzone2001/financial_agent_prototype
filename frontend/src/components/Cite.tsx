import { useId, useState, type MouseEvent } from "react";
import { describeSource } from "../citations";
import type { SourceRef } from "../types";

export const sourceAnchor = (n: number) => `src-${n}`;

function jumpTo(e: MouseEvent, n: number) {
  e.preventDefault();
  const el = document.getElementById(sourceAnchor(n));
  if (!el) return;
  el.scrollIntoView?.({ behavior: "smooth", block: "center" }); // absent in jsdom
  el.classList.remove("flash");
  void el.offsetWidth; // force reflow so the animation restarts on repeat clicks
  el.classList.add("flash");
}

export function Cite({ n, source }: { n: number; source: SourceRef | null }) {
  const [open, setOpen] = useState(false);
  const tipId = useId();
  return (
    <sup
      className="cite"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOpen(false); }}
      onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}
    >
      <a href={`#${sourceAnchor(n)}`} onClick={(e) => jumpTo(e, n)} aria-describedby={open ? tipId : undefined}>
        {`[${n}]`}
      </a>
      {open && (
        <span role="tooltip" id={tipId} className="cite-pop">
          {source ? (
            <>
              <span>{describeSource(source)}</span>
              <a href={source.url} target="_blank" rel="noreferrer">
                Open on {source.provider === "sec_edgar" ? "sec.gov" : "alphavantage.co"} ↗
              </a>
            </>
          ) : (
            <span className="muted">Source unavailable</span>
          )}
        </span>
      )}
    </sup>
  );
}
