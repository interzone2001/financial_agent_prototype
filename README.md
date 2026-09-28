# Financial Data Agent (prototype)

Multi-agent orchestrator (LangGraph) that gives wealth advisors a current quote (Alpha Vantage)
and a cited summary of recent SEC filings (EDGAR → Chroma), with follow-up chat.

Design spec: [`docs/superpowers/specs/2026-09-28-financial-agent-design.md`](docs/superpowers/specs/2026-09-28-financial-agent-design.md)

## LangGraph structure

![LangGraph structure: START → guard_in, which fans out in parallel to market_node and filings_node (joined at compose) for a report, or goes router → chat_answer for chat; refused injections skip straight to guard_out; everything exits through guard_out → END](docs/architecture/langgraph.svg)

One graph (`build_graph()` in [`backend/app/graph.py`](backend/app/graph.py)) serves both
requests. `guard_in` reads `mode` and picks a path. Every run enters through `guard_in` and
exits through `guard_out`, so the input and output guardrails cover both. The report path runs
`market_node` and `filings_node` in parallel, and `compose` waits for both. State is
checkpointed per `thread_id` by `SqliteSaver`, which is how chat remembers the report.

The full page, with a walk-through of each path and a node-by-node table (what each node calls
and what it writes to state), is at
[`docs/architecture/langgraph-map.html`](docs/architecture/langgraph-map.html). Open it in a
browser; GitHub shows HTML files as source.
