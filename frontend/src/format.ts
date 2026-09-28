const DASH = "—";
type N = number | null | undefined;
const ok = (v: N): v is number => typeof v === "number" && Number.isFinite(v);
const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 });
const compactUsd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 2 });
const int = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const dec = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });

export const formatPrice = (v: N) => (ok(v) ? usd.format(v) : DASH);
export const formatCompactUsd = (v: N) => (ok(v) ? compactUsd.format(v) : DASH);
export const formatInt = (v: N) => (ok(v) ? int.format(v) : DASH);
export const formatNumber = (v: N) => (ok(v) ? dec.format(v) : DASH);

export function formatChange(change: number, pct: number): string {
  const sign = change > 0 ? "+" : change < 0 ? "−" : "";
  return `${sign}${Math.abs(change).toFixed(2)} (${sign}${Math.abs(pct).toFixed(2)}%)`;
}

/** For datetimes only. Date-only strings ("2025-10-31") are displayed verbatim elsewhere. */
export function formatDateTime(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("en-US", { dateStyle: "medium", timeStyle: "short" });
}
