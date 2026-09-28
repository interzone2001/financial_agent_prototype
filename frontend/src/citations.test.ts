import { describe, expect, it } from "vitest";
import { describeSource, parseInline, registerAnswer, registerReport } from "./citations";
import { formatChange, formatCompactUsd, formatDateTime, formatInt, formatPrice } from "./format";
import { avSrc, sampleReport, src10k, src8k } from "./test/sample";
import type { ChatAnswer, SourceRef } from "./types";

const answer = (text: string, citations: SourceRef[]): ChatAnswer =>
  ({ text, citations, route: "filings", warnings: [], disclaimer: "" });
const srcQ: SourceRef = { ...src10k, chunk_id: "Q:item-2:0", section: "Item 2. MD&A" };

describe("registry", () => {
  const reg = registerReport(sampleReport.filings!);

  it("numbers report chunk ids by first appearance in section order; null source if unindexed", () => {
    expect([...reg.numberOf.entries()]).toEqual([["E:8k-items:0", 1], ["K:item-1a:0", 2], ["missing:id", 3]]);
    expect(reg.sources.map((s) => s.source?.url ?? null)).toEqual([src8k.url, src10k.url, null]);
  });

  it("chat answers reuse report numbers and append new sources, inline order first", () => {
    const next = registerAnswer(reg, answer("Growth [Q:item-2:0]; risk [K:item-1a:0].", [src10k, avSrc("OVERVIEW"), srcQ]));
    expect(next.numberOf.get("K:item-1a:0")).toBe(2);
    expect(next.sources.slice(3).map((s) => [s.n, s.key])).toEqual([[4, "Q:item-2:0"], [5, avSrc("OVERVIEW").url]]);
    expect(reg.sources).toHaveLength(3); // immutable: original untouched
  });

  it("registering an already-known answer changes nothing", () => {
    const once = registerAnswer(reg, answer("x [Q:item-2:0]", [srcQ]));
    expect(registerAnswer(once, answer("y [Q:item-2:0]", [srcQ])).sources).toHaveLength(4);
  });
});

describe("parseInline", () => {
  const m = new Map([["K:item-1a:0", 1], ["E:8k-items:0", 2]]);
  it("splits text and known markers into segments", () => {
    expect(parseInline("See [K:item-1a:0] and [E:8k-items:0].", m)).toEqual([
      { kind: "text", text: "See " }, { kind: "cite", n: 1, key: "K:item-1a:0" },
      { kind: "text", text: " and " }, { kind: "cite", n: 2, key: "E:8k-items:0" },
      { kind: "text", text: "." },
    ]);
  });
  it("keeps unknown markers as literal text (never silently dropped)", () => {
    expect(parseInline("x [nope] y", m)).toEqual([{ kind: "text", text: "x [nope] y" }]);
  });
  it("handles adjacent markers and plain text", () => {
    expect(parseInline("[K:item-1a:0][E:8k-items:0]", m).map((s) => s.kind)).toEqual(["cite", "cite"]);
    expect(parseInline("plain", m)).toEqual([{ kind: "text", text: "plain" }]);
  });
});

describe("describeSource / format", () => {
  it("describes filings as form · section · filed date (verbatim date)", () => {
    expect(describeSource(src10k)).toBe("10-K · Item 1A. Risk Factors · filed 2025-10-31");
    expect(describeSource(avSrc("OVERVIEW"))).toMatch(/^Alpha Vantage · retrieved /);
  });
  it("formats numbers", () => {
    expect(formatPrice(227.52)).toBe("$227.52");
    expect(formatPrice(null)).toBe("—");
    expect(formatCompactUsd(3410000000000)).toBe("$3.41T");
    expect(formatInt(1234567)).toBe("1,234,567");
    expect(formatChange(1.2, 0.53)).toBe("+1.20 (+0.53%)");
    expect(formatChange(-2.5, -1.1)).toBe("−2.50 (−1.10%)");
    expect(formatDateTime("garbage")).toBe("garbage");
  });
});
