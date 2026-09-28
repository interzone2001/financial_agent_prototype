from datetime import date

import pytest

from app.models import DataSourceError
from app.sources.alpha_vantage import parse_overview, parse_quote, source_url
from tests.ws1_market.helpers import T0, load_fixture


def test_parse_quote_fixture():
    q = parse_quote(load_fixture("av_global_quote_AAPL.json"), T0)
    assert (q.symbol, q.price, q.change, q.volume) == ("AAPL", 255.46, 3.15, 46076258)
    assert q.change_pct == pytest.approx(1.2485)
    assert q.latest_trading_day == date(2026, 9, 25)
    assert q.source.provider == "alpha_vantage"
    assert q.source.url == source_url("GLOBAL_QUOTE", "AAPL")
    assert q.source.retrieved_at == T0


def test_parse_quote_negative_change_pct():
    body = load_fixture("av_global_quote_AAPL.json")
    body["Global Quote"]["10. change percent"] = "-0.8123%"
    assert parse_quote(body, T0).change_pct == pytest.approx(-0.8123)


def test_parse_quote_missing_price_is_data_source_error():
    body = load_fixture("av_global_quote_AAPL.json")
    del body["Global Quote"]["05. price"]
    with pytest.raises(DataSourceError) as exc:
        parse_quote(body, T0)
    assert exc.value.provider == "alpha_vantage"


def test_parse_overview_fixture():
    o = parse_overview(load_fixture("av_overview_AAPL.json"), T0)
    assert (o.symbol, o.name, o.sector, o.industry) == (
        "AAPL", "Apple Inc", "TECHNOLOGY", "ELECTRONIC COMPUTERS")
    assert o.market_cap == 3791234560000
    assert (o.pe_ratio, o.week52_high, o.week52_low) == (38.87, 260.10, 169.21)
    assert o.description.startswith("Apple Inc.")
    assert o.source.url == source_url("OVERVIEW", "AAPL")


def test_parse_overview_sparse_none_and_dash_become_none():
    o = parse_overview(load_fixture("av_overview_sparse.json"), T0)
    assert o.name == "Sparse Corp"
    assert [o.sector, o.industry, o.market_cap, o.pe_ratio, o.week52_high, o.week52_low,
            o.description] == [None] * 7


def test_parse_overview_non_numeric_optional_becomes_none():
    body = load_fixture("av_overview_AAPL.json") | {"PERatio": "N/A"}
    assert parse_overview(body, T0).pe_ratio is None


def test_parse_overview_without_name_is_data_source_error():
    with pytest.raises(DataSourceError):
        parse_overview({"Symbol": "XYZ", "Name": "None"}, T0)


def test_source_url_is_redacted():
    url = source_url("GLOBAL_QUOTE", "BRK.B")
    assert url.startswith("https://www.alphavantage.co/query?")
    assert "function=GLOBAL_QUOTE" in url and "symbol=BRK.B" in url
    assert url.endswith("apikey=REDACTED")


@pytest.mark.parametrize("junk", ["NaN", "Infinity", "-inf"])
def test_parse_overview_non_finite_numbers_become_none(junk):
    body = load_fixture("av_overview_AAPL.json") | {
        "MarketCapitalization": junk, "PERatio": junk, "52WeekHigh": junk}
    o = parse_overview(body, T0)
    assert (o.market_cap, o.pe_ratio, o.week52_high) == (None, None, None)
