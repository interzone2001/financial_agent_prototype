import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { registerReport } from "../citations";
import { sampleReport, src10k } from "../test/sample";
import { FilingsReport } from "./FilingsReport";
import { LoadingStages } from "./LoadingStages";
import { QuoteCard } from "./QuoteCard";
import { SourceList } from "./SourceList";

afterEach(() => { cleanup(); vi.useRealTimers(); });

describe("QuoteCard", () => {
  it("shows price, red change with ▼, dashes for missing fields, and the source", () => {
    render(<QuoteCard market={sampleReport.market!} />);
    expect(screen.getByText("$227.52")).toBeTruthy();
    expect(screen.getByText("−1.25 (−0.55%)").className).toContain("down");
    expect(screen.getByText("▼")).toBeTruthy(); // never colour alone
    expect(screen.getByText("$3.41T")).toBeTruthy();
    expect(screen.getAllByText("—")).toHaveLength(2); // pe_ratio + industry null
    expect(screen.getByRole("link", { name: "quote" }).getAttribute("href")).toContain("GLOBAL_QUOTE");
  });
});

describe("FilingsReport + SourceList", () => {
  const summary = sampleReport.filings!;
  const reg = registerReport(summary);
  const renderReport = () => render(
    <><FilingsReport summary={summary} numberOf={reg.numberOf} /><SourceList sources={reg.sources} /></>,
  );

  it("renders sections, numbered citations and one Sources list", () => {
    renderReport();
    for (const h of ["Key developments", "Risk factors", "Financial highlights", "Material events"]) {
      expect(screen.getByRole("heading", { name: h })).toBeTruthy();
    }
    expect(screen.getAllByText("[1]")).toHaveLength(3);
    expect(screen.getByText("[2]").getAttribute("href")).toBe("#src-2");
    expect(screen.getByText(/Source unavailable \(missing:id\)/)).toBeTruthy();
    expect(screen.getByText(/10-K · Item 1A\. Risk Factors · filed 2025-10-31/)).toBeTruthy();
    expect(document.getElementById("src-2")).toBeTruthy();
    expect(screen.getAllByRole("link", { name: "View on sec.gov" })).toHaveLength(2);
    expect(screen.getByText(/Nothing reported/)).toBeTruthy(); // empty financial_highlights
  });

  it("[n] shows a popover with source details and flashes its Sources entry on click", () => {
    renderReport();
    const two = screen.getByText("[2]");
    fireEvent.mouseEnter(two.parentElement!);
    const tip = screen.getByRole("tooltip");
    expect(tip.textContent).toContain("10-K · Item 1A. Risk Factors · filed 2025-10-31");
    expect(within(tip).getByRole("link").getAttribute("href")).toBe(src10k.url);
    fireEvent.keyDown(two, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).toBeNull();
    fireEvent.click(two);
    expect(document.getElementById("src-2")!.className).toContain("flash");
  });
});

describe("LoadingStages", () => {
  it("keeps the spec copy and advances the stage line on a timer", () => {
    vi.useFakeTimers();
    render(<LoadingStages />);
    expect(screen.getByText("Fetching quote & filings…")).toBeTruthy();
    expect(screen.getByText(/Resolving ticker…/)).toBeTruthy();
    act(() => { vi.advanceTimersByTime(6000); });
    expect(screen.getByText(/Reading SEC filings…/)).toBeTruthy();
  });
});
