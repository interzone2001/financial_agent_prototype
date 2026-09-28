import { Fragment, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { api, errorMessage } from "../api";
import { parseInline, sourceFor, sourceKey, type Registry } from "../citations";
import type { ChatAnswer, Route } from "../types";
import { Cite } from "./Cite";

const MAX = 1000;
const ROUTE_LABEL: Record<Route, string> = {
  market: "Market data", filings: "SEC filings", both: "Market + filings", off_topic: "Out of scope",
};
type Item = { role: "user"; text: string } | { role: "assistant"; answer: ChatAnswer } | { role: "error"; text: string };

function Answer({ answer, registry }: { answer: ChatAnswer; registry: Registry }) {
  const segments = parseInline(answer.text, registry.numberOf);
  const inline = new Set(segments.flatMap((s) => (s.kind === "cite" ? [s.key] : [])));
  // Citations the text didn't mention inline still get shown, as chips under the answer.
  const extra = [...new Set(answer.citations.map(sourceKey))].filter((k) => !inline.has(k) && registry.numberOf.has(k));
  return (
    <>
      <span className={`route route-${answer.route}`}>{ROUTE_LABEL[answer.route]}</span>
      <p className="answer-text">
        {segments.map((s, i) =>
          s.kind === "text" ? <Fragment key={i}>{s.text}</Fragment>
            : <Cite key={i} n={s.n} source={sourceFor(registry, s.key)} />)}
      </p>
      {extra.length > 0 && (
        <p className="answer-sources">
          Sources:{" "}
          {extra.map((k) => <Cite key={k} n={registry.numberOf.get(k)!} source={sourceFor(registry, k)} />)}
        </p>
      )}
      {answer.warnings.length > 0 && <p className="msg-warning">{answer.warnings.join(" ")}</p>}
    </>
  );
}

export function ChatPanel({ threadId, ticker, registry, onAnswer }: {
  threadId: string; ticker: string | null; registry: Registry; onAnswer: (a: ChatAnswer) => void;
}) {
  const [items, setItems] = useState<Item[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const ready = ticker !== null;
  const logRef = useRef<HTMLOListElement>(null);
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => { const el = logRef.current; if (el) el.scrollTop = el.scrollHeight; }, [items, busy]);

  async function send(e?: FormEvent) {
    e?.preventDefault();
    const msg = input.trim();
    if (!ready || !msg || busy || msg.length > MAX) return;
    setItems((xs) => [...xs, { role: "user", text: msg }]);
    setInput("");
    setBusy(true);
    try {
      const answer = await api.chat(threadId, msg);
      if (!alive.current) return;
      onAnswer(answer); // registry update + message append batch into one render
      setItems((xs) => [...xs, { role: "assistant", answer }]);
    } catch (err) {
      if (!alive.current) return;
      setItems((xs) => [...xs, { role: "error", text: errorMessage(err) }]);
      setInput(msg); // let them resend without retyping
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); }
  }

  return (
    <section className="chat" aria-label="Follow-up chat">
      <h2>Ask about {ticker ?? "a security"}</h2>
      <ol className="chat-log" aria-live="polite" ref={logRef}>
        {items.length === 0 && (
          <li className="muted">{ready ? "e.g. What did they say about supply-chain risk?" : "Generate a report, then ask follow-up questions here."}</li>
        )}
        {items.map((it, i) => (
          <li key={i} className={`msg msg-${it.role}`}>
            {it.role === "assistant" ? <Answer answer={it.answer} registry={registry} /> : it.text}
          </li>
        ))}
        {busy && <li className="msg muted">Thinking…</li>}
      </ol>
      <form className="chat-form" onSubmit={send}>
        <textarea aria-label="Ask a follow-up" value={input} maxLength={MAX} rows={2} disabled={!ready}
          onChange={(e) => setInput(e.target.value)} onKeyDown={onKey}
          placeholder={ready ? "Ask a follow-up…" : "Generate a report first"} />
        <div className="chat-actions">
          <span className="muted num">{input.length}/{MAX}</span>
          <button type="submit" disabled={!ready || busy || !input.trim()}>Send</button>
        </div>
      </form>
    </section>
  );
}
