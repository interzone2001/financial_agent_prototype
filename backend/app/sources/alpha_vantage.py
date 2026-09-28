"""Alpha Vantage source: parsers + cached client (spec §6.1).

The API key never leaves AlphaVantageClient: SourceRef URLs say apikey=REDACTED,
cache keys are "{function}:{symbol}", and error messages are scrubbed.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

from app.models import DataSourceError, Overview, Quote, SourceRef

BASE_URL = "https://www.alphavantage.co/query"
PROVIDER = "alpha_vantage"
_MISSING = {"", "none", "-", "null", "n/a"}


def source_url(function: str, symbol: str) -> str:
    query = urlencode({"function": function, "symbol": symbol, "apikey": "REDACTED"})
    return f"{BASE_URL}?{query}"


def _source(function: str, symbol: str, retrieved_at: datetime) -> SourceRef:
    return SourceRef(provider=PROVIDER, url=source_url(function, symbol), retrieved_at=retrieved_at)


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _MISSING else text


def _opt_float(value: object) -> float | None:
    text = _clean(value)
    try:
        return None if text is None else float(text)
    except ValueError:
        return None


def _opt_int(value: object) -> int | None:
    text = _clean(value)
    try:
        return None if text is None else int(Decimal(text))
    except InvalidOperation:
        return None


def parse_quote(body: dict, retrieved_at: datetime) -> Quote:
    q = body.get("Global Quote") or {}
    try:
        symbol = q["01. symbol"]
        return Quote(
            symbol=symbol,
            price=float(q["05. price"]),
            change=float(q["09. change"]),
            change_pct=float(q["10. change percent"].strip().rstrip("%")),
            volume=int(q["06. volume"]),
            latest_trading_day=date.fromisoformat(q["07. latest trading day"]),
            source=_source("GLOBAL_QUOTE", symbol, retrieved_at),
        )
    except (KeyError, ValueError, AttributeError) as e:  # pydantic ValidationError is a ValueError
        raise DataSourceError(PROVIDER, f"malformed GLOBAL_QUOTE response: {e!r}") from None


def parse_overview(body: dict, retrieved_at: datetime) -> Overview:
    symbol, name = _clean(body.get("Symbol")), _clean(body.get("Name"))
    if symbol is None or name is None:
        raise DataSourceError(PROVIDER, "malformed OVERVIEW response: missing Symbol/Name")
    return Overview(
        symbol=symbol,
        name=name,
        sector=_clean(body.get("Sector")),
        industry=_clean(body.get("Industry")),
        market_cap=_opt_int(body.get("MarketCapitalization")),
        pe_ratio=_opt_float(body.get("PERatio")),
        week52_high=_opt_float(body.get("52WeekHigh")),
        week52_low=_opt_float(body.get("52WeekLow")),
        description=_clean(body.get("Description")),
        source=_source("OVERVIEW", symbol, retrieved_at),
    )
