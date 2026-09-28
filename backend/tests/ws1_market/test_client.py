import logging
import sqlite3

import httpx
import pytest

from app.models import DataSourceError, RateLimitedError, TickerNotFoundError
from tests.ws1_market.helpers import KEY, load_fixture

QUOTE = load_fixture("av_global_quote_AAPL.json")
OVERVIEW = load_fixture("av_overview_AAPL.json")


def test_quote_sends_expected_params(make_client):
    client, av = make_client(GLOBAL_QUOTE=QUOTE)
    assert client.quote("AAPL").price == 255.46
    assert dict(av.calls[0].url.params) == {
        "function": "GLOBAL_QUOTE", "symbol": "AAPL", "apikey": KEY}


@pytest.mark.parametrize("body", [
    {"Note": "Thank you for using Alpha Vantage! ...call frequency is 5 calls per minute"},
    load_fixture("av_rate_limited.json"),
])
def test_rate_limit_body_raises_and_is_not_cached(make_client, body):
    client, av = make_client(GLOBAL_QUOTE=body)
    for _ in range(2):
        with pytest.raises(RateLimitedError) as exc:
            client.quote("AAPL")
        assert exc.value.provider == "alpha_vantage"
    assert av.count("GLOBAL_QUOTE") == 2  # never served from cache


def test_rate_limit_message_redacts_echoed_key(make_client):
    client, _ = make_client(GLOBAL_QUOTE={"Information": f"We have detected your API key as {KEY}"})
    with pytest.raises(RateLimitedError) as exc:
        client.quote("AAPL")
    assert KEY not in str(exc.value)


def test_empty_quote_is_ticker_not_found(make_client):
    client, av = make_client(GLOBAL_QUOTE=load_fixture("av_empty_quote.json"))
    with pytest.raises(TickerNotFoundError) as exc:
        client.quote("ZZZZ")
    assert exc.value.ticker == "ZZZZ"
    with pytest.raises(TickerNotFoundError):
        client.quote("ZZZZ")
    assert av.count("GLOBAL_QUOTE") == 2  # empty body not cached


def test_empty_overview_is_ticker_not_found(make_client):
    client, _ = make_client(OVERVIEW={})
    with pytest.raises(TickerNotFoundError):
        client.overview("ZZZZ")


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="oops"),
    httpx.Response(200, text="<html>not json</html>"),
    {"Error Message": "Invalid API call."},
])
def test_upstream_failures_are_data_source_errors(make_client, response):
    client, _ = make_client(GLOBAL_QUOTE=response)
    with pytest.raises(DataSourceError) as exc:
        client.quote("AAPL")
    assert not isinstance(exc.value, RateLimitedError)


def test_transport_error_is_data_source_error_without_key(make_client):
    client, _ = make_client(GLOBAL_QUOTE=httpx.ConnectError(f"cannot reach ...apikey={KEY}"))
    with pytest.raises(DataSourceError) as exc:
        client.quote("AAPL")
    assert KEY not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


def test_cache_hit_avoids_second_call_and_keeps_retrieved_at(make_client, clock):
    client, av = make_client(GLOBAL_QUOTE=QUOTE)
    first = client.quote("AAPL")
    clock.advance(seconds=30)
    second = client.quote("AAPL")
    assert av.count("GLOBAL_QUOTE") == 1
    assert second.source.retrieved_at == first.source.retrieved_at


def test_quote_refetches_after_60s(make_client, clock):
    client, av = make_client(GLOBAL_QUOTE=QUOTE)
    t0 = client.quote("AAPL").source.retrieved_at
    clock.advance(seconds=61)
    assert client.quote("AAPL").source.retrieved_at > t0
    assert av.count("GLOBAL_QUOTE") == 2


def test_overview_cached_for_24h(make_client, clock):
    client, av = make_client(OVERVIEW=OVERVIEW)
    client.overview("AAPL")
    clock.advance(hours=23)
    client.overview("AAPL")
    assert av.count("OVERVIEW") == 1
    clock.advance(hours=2)
    client.overview("AAPL")
    assert av.count("OVERVIEW") == 2


def test_api_key_never_in_source_url_or_cache(make_client, tmp_path):
    client, _ = make_client(GLOBAL_QUOTE=QUOTE, OVERVIEW=OVERVIEW)
    assert KEY not in client.quote("AAPL").source.url
    assert KEY not in client.overview("AAPL").source.url
    rows = sqlite3.connect(tmp_path / "cache.sqlite").execute(
        "SELECT key, body FROM av_cache").fetchall()
    assert {r[0] for r in rows} == {"GLOBAL_QUOTE:AAPL", "OVERVIEW:AAPL"}
    assert all(KEY not in r[0] + r[1] for r in rows)


def test_unparseable_body_is_not_cached(make_client):
    client, av = make_client(OVERVIEW={"Symbol": "XYZ"})  # non-empty, passes checks, no Name
    for _ in range(2):
        with pytest.raises(DataSourceError):
            client.overview("XYZ")
    assert av.count("OVERVIEW") == 2


@pytest.mark.parametrize("quote", [
    {"01. symbol": "XYZ", "05. price": None},
    ["not", "a", "dict"],
    "junk",
])
def test_junk_quote_shapes_are_data_source_errors(make_client, quote):
    client, _ = make_client(GLOBAL_QUOTE={"Global Quote": quote})
    with pytest.raises(DataSourceError):
        client.quote("XYZ")


def test_httpx_request_log_never_contains_key(make_client, caplog):
    client, _ = make_client(GLOBAL_QUOTE=QUOTE)
    with caplog.at_level(logging.DEBUG):
        client.quote("AAPL")
    assert KEY not in caplog.text
