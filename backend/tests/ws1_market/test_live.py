"""LIVE: 2 Alpha Vantage calls (shared 25/day quota). Run once: uv run pytest -m live tests/ws1_market

The AV fixtures were HAND-BUILT from AV docs. If this fails on key shape, REPORT the
diff to the human. Do NOT edit tests/fixtures/* (shared, read-only)."""

import os

import pytest

from app import config
from app.agents.market import _snapshot_with
from app.cache import SqliteCache
from app.sources.alpha_vantage import AlphaVantageClient
from tests.ws1_market.helpers import load_fixture

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("ALPHA_VANTAGE_API_KEY"), reason="no AV key"),
]
DRIFT = "Live AV shape differs from hand-built fixture - report to the human; do not edit fixtures"


def test_live_aapl_matches_fixture_shape(tmp_path):
    key = config.alpha_vantage_key()
    client = AlphaVantageClient(api_key=key, cache=SqliteCache(tmp_path / "cache.sqlite"))
    quote_body, _ = client.fetch("GLOBAL_QUOTE", "AAPL")
    overview_body, _ = client.fetch("OVERVIEW", "AAPL")

    fx_quote = load_fixture("av_global_quote_AAPL.json")["Global Quote"]
    assert set(quote_body["Global Quote"]) == set(fx_quote), DRIFT
    # Real OVERVIEW carries extra fields (margins, TTM figures), which is fine. Every
    # fixture key must exist live.
    missing = set(load_fixture("av_overview_AAPL.json")) - set(overview_body)
    assert not missing, f"{DRIFT}: missing {sorted(missing)}"

    snap = _snapshot_with(client, "AAPL")  # cache hits, no extra quota
    assert snap.quote.symbol == "AAPL" and snap.quote.price > 0
    assert snap.overview.name
    assert key not in snap.quote.source.url and key not in snap.overview.source.url
