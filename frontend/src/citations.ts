import { formatDateTime } from "./format";
import type { ChatAnswer, FilingsSummary, SourceRef } from "./types";

export type ClaimSection = "key_developments" | "risk_factors" | "financial_highlights" | "material_events";
export const SECTIONS: ReadonlyArray<readonly [ClaimSection, string]> = [
  ["key_developments", "Key developments"],
  ["risk_factors", "Risk factors"],
  ["financial_highlights", "Financial highlights"],
  ["material_events", "Material events"],
];

export interface NumberedSource { n: number; key: string; source: SourceRef | null }

/** One citation numbering for the whole session: report claims first, chat answers append. */
export interface Registry { numberOf: ReadonlyMap<string, number>; sources: readonly NumberedSource[] }

export const EMPTY_REGISTRY: Registry = { numberOf: new Map(), sources: [] };
export const sourceKey = (s: SourceRef): string => s.chunk_id ?? s.url;

export function sourceFor(reg: Registry, key: string): SourceRef | null {
  const n = reg.numberOf.get(key);
  return n === undefined ? null : reg.sources[n - 1].source;
}

function add(reg: Registry, key: string, source: SourceRef | null): Registry {
  if (reg.numberOf.has(key)) return reg;
  const n = reg.sources.length + 1;
  return { numberOf: new Map(reg.numberOf).set(key, n), sources: [...reg.sources, { n, key, source }] };
}

export function registerReport(summary: FilingsSummary): Registry {
  let reg = EMPTY_REGISTRY;
  for (const [section] of SECTIONS) {
    for (const claim of summary[section]) {
      for (const id of claim.citations) reg = add(reg, id, summary.citation_index[id] ?? null);
    }
  }
  return reg;
}

// Chat answers cite inline as "[<chunk_id>]" (spec §6.3).
const MARKER = /\[([^[\]\s]+)\]/g;

export function registerAnswer(reg: Registry, answer: ChatAnswer): Registry {
  const byKey = new Map(answer.citations.map((s) => [sourceKey(s), s] as const));
  const inline = [...answer.text.matchAll(MARKER)].map((m) => m[1]).filter((id) => byKey.has(id));
  for (const key of [...inline, ...byKey.keys()]) reg = add(reg, key, byKey.get(key)!);
  return reg;
}

export type Segment = { kind: "text"; text: string } | { kind: "cite"; n: number; key: string };

export function parseInline(text: string, numberOf: ReadonlyMap<string, number>): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const n = numberOf.get(m[1]);
    if (n === undefined) continue; // unknown id: stays in the text verbatim
    if (m.index > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    out.push({ kind: "cite", n, key: m[1] });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

export function describeSource(s: SourceRef): string {
  if (s.provider === "alpha_vantage") return `Alpha Vantage · retrieved ${formatDateTime(s.retrieved_at)}`;
  return [s.form_type, s.section, s.filed_date && `filed ${s.filed_date}`].filter(Boolean).join(" · ");
}
