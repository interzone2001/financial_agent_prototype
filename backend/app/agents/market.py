"""WS1 Market Data Agent (spec §6.1). Public functions match app.contracts exactly.

Tests inject a client via `_snapshot_with(client, ticker)` or by monkeypatching
`_default_client`; production builds one lazily from config on first use.
"""

from __future__ import annotations

import threading

import httpx

from app import config
from app.cache import SqliteCache
from app.llm import LLM
from app.models import AgentAnswer, DataSourceError, MarketSnapshot
from app.sources.alpha_vantage import AlphaVantageClient
from app.tracing import traceable

SYSTEM_PROMPT = """You are a market-data assistant for a wealth-management advisor.
Answer ONLY from the JSON market snapshot in the user message (an Alpha Vantage quote
and company overview).
- If the snapshot does not contain the answer, say so plainly. Do not guess or use
  outside knowledge.
- Do not cite or mention any source other than the supplied snapshot.
- Never give investment advice: no buy/sell/hold views, price targets or recommendations.
- Be concise. State figures with units and mention the quote's latest trading day."""

_default_client: AlphaVantageClient | None = None
_lock = threading.Lock()


def _client() -> AlphaVantageClient:
    global _default_client
    with _lock:
        if _default_client is None:
            try:
                key = config.alpha_vantage_key()
            except KeyError:
                raise DataSourceError("alpha_vantage", "ALPHA_VANTAGE_API_KEY is not set") from None
            _default_client = AlphaVantageClient(
                api_key=key,
                http=httpx.Client(timeout=10.0),
                cache=SqliteCache(config.data_dir() / "cache.sqlite"),
            )
        return _default_client


def _snapshot_with(client: AlphaVantageClient, ticker: str) -> MarketSnapshot:
    symbol = ticker.strip().upper()
    # Quote first: an unknown ticker fails fast without spending the OVERVIEW call.
    return MarketSnapshot(quote=client.quote(symbol), overview=client.overview(symbol))


@traceable(run_type="tool", name="market.get_market_snapshot")
def get_market_snapshot(ticker: str) -> MarketSnapshot:
    return _snapshot_with(_client(), ticker)


@traceable(run_type="chain", name="market.answer_market_question")
def answer_market_question(question: str, snapshot: MarketSnapshot, llm: LLM) -> AgentAnswer:
    user = (
        f"Market snapshot (JSON):\n{snapshot.model_dump_json(indent=2)}\n\n"
        f"Question: {question}"
    )
    text = llm.text("agent", SYSTEM_PROMPT, [{"role": "user", "content": user}])
    return AgentAnswer(
        text=text.strip(), citations=[snapshot.quote.source, snapshot.overview.source]
    )
